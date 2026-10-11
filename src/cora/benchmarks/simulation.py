"""Fixed-continuation rollouts with lossless native states for offline rendering."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def simulate(env, policy, state, key, horizon, gamma):
    def step(carry, unused):
        state, key, hidden, dones, alive = carry
        key, action_key, step_key = jax.random.split(key, 3)
        actions, hidden = policy.act(env.get_obs(state), hidden, dones, action_key)
        _, new_state, rewards, done, _ = getattr(env,'evaluation_step',env.step)(step_key, state, actions)
        new_state=jax.tree.map(lambda new,old:jnp.where(alive,new,old),new_state,state)
        reward = jnp.mean(jnp.stack([rewards[a] for a in env.agents])) * alive
        return (new_state, key, hidden, jnp.stack([done[a] for a in env.agents]), alive & ~done["__all__"]), (state.env_state, reward, jnp.stack([actions[a] for a in env.agents]))
    carry = (state, key, policy.initial_hidden(), jnp.zeros(env.num_agents, dtype=bool), jnp.array(True))
    _, (states, rewards, actions) = jax.lax.scan(step, carry, None, horizon)
    return {"states": states, "rewards": rewards, "actions": actions,
            "return": jnp.sum(rewards * gamma ** jnp.arange(horizon))}


def save_trace(path, trace, assignments):
    state = trace["states"]
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    arrays = {'state'+jax.tree_util.keystr(path): np.asarray(value) for path,value in leaves}
    # Preserve legacy flat trace keys; nested native states also retain lossless leaves.
    for field in state.__dataclass_fields__:
        value=getattr(state,field)
        if not jax.tree_util.tree_leaves(value) or hasattr(value,'__dataclass_fields__'):continue
        if hasattr(value,'shape'):arrays[field]=np.asarray(value)
    arrays.update(rewards=np.asarray(trace["rewards"]), actions=np.asarray(trace["actions"]), assignments=np.asarray(assignments))
    np.savez_compressed(path, **arrays)
