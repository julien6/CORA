"""Controlled simulator branches under a fixed observation-only policy."""
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cora.interventions import apply

POLICY_VERSION = "assignment-guidance-v1-epsilon-0.1"


def guided_actions(obs, key, agents):
    """Move toward assigned landmark using positions already in MPE observations.

    Actions are upstream discrete actions: 0 stay, 1/2 -/+x, 3/4 -/+y.
    The assignment one-hot is the only added observation. Ten percent uniform
    exploration exercises explicit policy PRNG handling; no state is consulted.
    """
    keys = jax.random.split(key, len(agents))
    actions = {}
    for a, k in zip(agents, keys):
        mission = jnp.argmax(obs[a][-3:])
        relative = obs[a][4:10].reshape(3, 2)[mission]
        axis = jnp.argmax(jnp.abs(relative))
        target = 1 + 2 * axis + (relative[axis] >= 0).astype(jnp.int32)
        target = jnp.where(jnp.linalg.norm(relative) < 0.05, 0, target)
        explore, sample = jax.random.split(k)
        actions[a] = jnp.where(jax.random.uniform(explore) < 0.1, jax.random.randint(sample, (), 0, 5), target)
    return actions


@partial(jax.jit, static_argnums=(0, 3, 4))
def rollout(env, state, key, horizon: int, gamma: float):
    """Reward is the mean of native per-agent rewards; stop value at terminal.

    Positions in the returned trace are evaluation-only visualization data.
    """
    def step(carry, unused):
        state, key, alive = carry
        key, policy_key, step_key = jax.random.split(key, 3)
        obs = env.get_obs(state)
        actions = guided_actions(obs, policy_key, env.agents)
        obs, new_state, rewards, done, info = env.step(step_key, state, actions)
        reward = jnp.mean(jnp.stack([rewards[a] for a in env.agents])) * alive
        record = (reward, state.env_state.p_pos, jnp.stack([actions[a] for a in env.agents]))
        return (new_state, key, alive & ~done["__all__"]), record

    (final, _, _), (rewards, positions, actions) = jax.lax.scan(step, (state, key, jnp.array(True)), None, horizon)
    value = jnp.sum(rewards * gamma ** jnp.arange(horizon))
    return {"return": value, "rewards": rewards, "positions": positions, "actions": actions, "final_state": final}


def collect(env, base_state, options, root_key, horizon, gamma, swap_cost, sample_count):
    """Randomized uniform permutation without replacement for initial design.

    One reset context and common rollout keys across branches are intentional.
    Resulting outcomes are correlated; no fixed-design independence bound is used.
    """
    if not 1 <= sample_count <= len(options):
        raise ValueError("Sample count must fit candidate set")
    order_key, branch_key = jax.random.split(root_key)
    order = np.asarray(jax.random.permutation(order_key, len(options)))[:sample_count]
    rows = []
    for t, idx in enumerate(order):
        z = options[int(idx)]
        org = apply(env.spec, base_state.organization, z)
        trace = rollout(env, base_state.replace(organization=org), branch_key, horizon, gamma)
        rows.append({
            "intervention": list(z.bits), "organization": np.asarray(org.assignments).tolist(),
            "roles": np.asarray(org.roles).tolist(), "organization_version": int(org.version),
            "policy_version": POLICY_VERSION, "reward": float(trace["return"]),
            "cost": z.cost(swap_cost), "assignment_probability": 1.0 / (len(options) - t),
            "context_id": "shared-reset-0", "rollout_key": np.asarray(branch_key).tolist(),
            "horizon": horizon, "gamma": gamma,
        })
    return rows


def validate_rows(rows):
    from cora.interventions import Intervention
    from cora.organization import Organization, OrganizationSpec
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty list")
    spec = OrganizationSpec()
    for row in rows:
        z = Intervention(tuple(row["intervention"]))
        org = Organization(jnp.asarray(row["organization"]), jnp.asarray(row["roles"]), jnp.asarray(row["organization_version"]))
        spec.validate(org)
        expected = apply(spec, spec.initial(), z)
        if not np.array_equal(expected.assignments, org.assignments) or int(expected.version) != int(org.version):
            raise ValueError("Logged organization does not match intervention")
        if row["policy_version"] != POLICY_VERSION or row["context_id"] != "shared-reset-0":
            raise ValueError("Mixed policy versions or contexts")
        if not np.isfinite([row["reward"], row["cost"], row["assignment_probability"], row["gamma"]]).all():
            raise ValueError("Non-finite dataset values")
        if row["cost"] < 0 or not 0 < row["assignment_probability"] <= 1 or not 0 <= row["gamma"] <= 1 or not 1 <= row["horizon"] <= 20:
            raise ValueError("Invalid rollout metadata")
        key = np.asarray(row["rollout_key"])
        if key.shape != (2,) or not np.issubdtype(key.dtype, np.integer) or np.any(key < 0) or np.any(key > 2**32 - 1):
            raise ValueError("Malformed rollout key")
    if len({r["policy_version"] for r in rows}) != 1 or len({(r["horizon"], r["gamma"], tuple(r["rollout_key"])) for r in rows}) != 1:
        raise ValueError("Inconsistent continuation protocol")


def save_dataset(path: Path, rows):
    validate_rows(rows)
    path.write_text(json.dumps({"schema_version": 1, "records": rows}, indent=2, allow_nan=False) + "\n")


def load_dataset(path: Path):
    data = json.loads(path.read_text())
    if data["schema_version"] != 1:
        raise ValueError("Unknown dataset schema")
    validate_rows(data["records"])
    return data["records"]

