import jax
import jax.numpy as jnp
import numpy as np

from cora.interventions import Intervention
from cora.rollout import guided_actions, rollout


def test_observation_intervention_and_noop(env):
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)
    original, inner = env._env.reset(key)
    for a in env.agents:
        assert obs[a].shape == (21,)
        np.testing.assert_array_equal(obs[a][:18], original[a])
    changed_obs, changed_state = env.intervene(state, Intervention((1, 0, 0)))
    assert not np.array_equal(obs["agent_0"], changed_obs["agent_0"])
    for a in env.agents:
        np.testing.assert_array_equal(obs[a][:18], changed_obs[a][:18])
    unchanged, noop = env.intervene(state, Intervention())
    for a in env.agents:
        np.testing.assert_array_equal(unchanged[a], obs[a])
    for a, b in zip(jax.tree.leaves(state.env_state), jax.tree.leaves(changed_state.env_state)):
        np.testing.assert_array_equal(a, b)


def test_assignment_changes_guidance_without_privileged_state(env):
    # Known visible landmark vectors guarantee the target action changes.
    obs = {a: jnp.concatenate((jnp.zeros(4), jnp.array([1., 0., -1., 0., 0., 1.]), jnp.zeros(8), jnp.array([1., 0., 0.]))) for a in env.agents}
    swapped = {a: o.at[-3:].set(jnp.array([0., 1., 0.])) for a, o in obs.items()}
    key = next(jax.random.PRNGKey(i) for i in range(20) if int(guided_actions(obs, jax.random.PRNGKey(i), env.agents)["agent_0"]) == 2 and int(guided_actions(swapped, jax.random.PRNGKey(i), env.agents)["agent_0"]) == 1)
    assert int(guided_actions(obs, key, env.agents)["agent_0"]) != int(guided_actions(swapped, key, env.agents)["agent_0"])


def test_twenty_steps_reproducible_and_vectorizable(env):
    obs, state = env.reset(jax.random.PRNGKey(7))
    one = rollout(env, state, jax.random.PRNGKey(1), 20, 0.99)
    two = rollout(env, state, jax.random.PRNGKey(1), 20, 0.99)
    np.testing.assert_array_equal(one["rewards"], two["rewards"])
    assert one["actions"].shape == (20, 3)
    assert np.all((np.asarray(one["actions"]) >= 0) & (np.asarray(one["actions"]) < 5))
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(one))
    _, other = env.reset(jax.random.PRNGKey(8))
    assert not np.array_equal(state.env_state.p_pos, other.env_state.p_pos)
    batched_obs, batch = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    assert batched_obs["agent_0"].shape == (2, 21)


def test_wrapper_preserves_native_step(env):
    _, state = env.reset(jax.random.PRNGKey(1))
    actions = {a: jnp.array(0) for a in env.agents}
    key = jax.random.PRNGKey(2)
    wrapped = env.step(key, state, actions)
    native = env._env.step(key, state.env_state, actions)
    for a, b in zip(jax.tree.leaves(wrapped[2:]), jax.tree.leaves(native[2:])):
        np.testing.assert_array_equal(a, b)

