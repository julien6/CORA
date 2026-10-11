"""One-command PoC. CPU default; fixed tiny runtime bounds."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

# Set before importing JAX in the CLI process. Library callers control their backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from omegaconf import OmegaConf

from cora.env_wrapper import OrganizationWrapper
from cora.estimator import RidgeEstimator
from cora.exploration import uniform, uncertainty
from cora.features import FeatureEncoder
from cora.interventions import Intervention, apply, candidates
from cora.ippo_adapter import ROOT, smoke_update
from cora.organization import OrganizationSpec
from cora.rollout import POLICY_VERSION, collect, load_dataset, rollout, save_dataset
from cora.selector import select
from cora.visualization import render


def run(config, *, run_ippo=True):
    cfg = dict(config)
    for name, low, high in (("horizon", 10, 20), ("samples", 4, 8), ("ippo_steps", 1, 20)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("Seed must be a uint32 integer")
    for name in ("gamma", "swap_cost", "ridge", "diagnostic_scale"):
        if not np.isfinite(cfg[name]):
            raise ValueError(f"{name} must be finite")
    if not 0 <= cfg["gamma"] <= 1 or cfg["swap_cost"] < 0 or cfg["ridge"] <= 0 or cfg["diagnostic_scale"] < 0:
        raise ValueError("Invalid numeric configuration")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    spec = OrganizationSpec()
    env = OrganizationWrapper(jaxmarl.make("MPE_simple_spread_v3"), spec)
    root = jax.random.PRNGKey(cfg["seed"])
    reset_key, collect_key, explore_key, continuation_key = jax.random.split(root, 4)
    obs, base = env.reset(reset_key)
    options = candidates(spec, base.organization)
    encoder = FeatureEncoder.from_hyperedges(3, ((0, 1, 2),))
    features = encoder.encode(jnp.array([z.bits for z in options]))
    rows = collect(env, base, options, collect_key, cfg["horizon"], cfg["gamma"], cfg["swap_cost"], cfg["samples"] - 1)
    initial_x = encoder.encode(jnp.array([r["intervention"] for r in rows]))
    initial_model = RidgeEstimator.fit(initial_x, jnp.array([r["reward"] for r in rows]), cfg["ridge"], cfg["diagnostic_scale"])
    uniform_idx, uniform_probability = uniform(explore_key, len(options))
    active_idx, active_probability = uncertainty(initial_model, features)
    active_z = options[active_idx]
    active_org = apply(spec, base.organization, active_z)
    branch_key = jnp.asarray(rows[0]["rollout_key"], dtype=jnp.uint32)
    active_trace = rollout(env, base.replace(organization=active_org), branch_key, cfg["horizon"], cfg["gamma"])
    # Same frozen policy/context as the uniform design; deterministic acquisition
    # has conditional probability one and provides no general positivity claim.
    active_row = dict(rows[0])
    active_row.update(intervention=list(active_z.bits), organization=np.asarray(active_org.assignments).tolist(),
                      organization_version=int(active_org.version), reward=float(active_trace["return"]),
                      cost=active_z.cost(cfg["swap_cost"]), assignment_probability=active_probability)
    rows.append(active_row)
    save_dataset(output / "dataset.json", rows)
    reloaded = load_dataset(output / "dataset.json")
    if rows != reloaded:
        raise RuntimeError("Dataset round trip failed")
    x = encoder.encode(jnp.array([r["intervention"] for r in rows]))
    model = RidgeEstimator.fit(x, jnp.array([r["reward"] for r in rows]), cfg["ridge"], cfg["diagnostic_scale"])
    chosen, decision = select(spec, base.organization, options, features, model, cfg["swap_cost"])
    org = apply(spec, base.organization, chosen)
    # Paired continuation uses a fresh common random stream and identical state.
    noop_trace = rollout(env, base, continuation_key, cfg["horizon"], cfg["gamma"])
    selected_trace = rollout(env, base.replace(organization=org), continuation_key, cfg["horizon"], cfg["gamma"])
    traces = {
        "noop_positions": np.asarray(noop_trace["positions"]),
        "selected_positions": np.asarray(selected_trace["positions"]),
        "noop_rewards": np.asarray(noop_trace["rewards"]),
        "selected_rewards": np.asarray(selected_trace["rewards"]),
    }
    if not all(np.isfinite(a).all() for a in traces.values()):
        raise RuntimeError("Non-finite continuation")
    np.savez(output / "traces.npz", **traces)
    summary = {
        "schema_version": 1, "purpose": "engineering_smoke_test", "certified": False,
        "config": {k: v for k, v in cfg.items() if k != "output"},
        "environment": "MPE_simple_spread_v3", "policy_version": POLICY_VERSION,
        "reward_convention": "discounted mean of native agent rewards",
        "initial_assignments": np.asarray(base.organization.assignments).tolist(),
        "selected_assignments": np.asarray(org.assignments).tolist(),
        "selected_intervention": list(chosen.bits), "selection": decision,
        "dataset_size": len(rows), "feature_family": [list(t) for t in encoder.family],
        "design_rank": model.rank, "parameter_count": len(encoder.family),
        "coefficients": np.asarray(model.coefficients).tolist(),
        "exploration": {"uniform_candidate": list(options[uniform_idx].bits),
                        "uniform_probability": uniform_probability,
                        "uncertainty_candidate": list(options[active_idx].bits),
                        "uncertainty_probability": active_probability,
                        "acquisition_executed": True},
        "continuation_returns": {"noop": float(noop_trace["return"]), "selected": float(selected_trace["return"])},
        "observation_shape": list(obs[env.agents[0]].shape),
        "backend": jax.default_backend(),
        "dependencies": {name: version(name) for name in ("jax", "jaxlib", "jaxmarl", "flax", "optax", "distrax", "numpy")},
        "ippo": smoke_update(env, cfg["seed"], cfg["ippo_steps"]) if run_ippo else {"skipped": True},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/poc.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-ippo", action="store_true", help="Pass 1 only; does not meet full PoC acceptance")
    parser.add_argument("--render-only", action="store_true", help="Regenerate figure from saved raw output")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config))
    if args.output is not None:
        config["output"] = str(args.output)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.render_only:
        render(Path(config["output"]))
        return
    summary = run(config, run_ippo=not args.skip_ippo)
    print(json.dumps({"output": config["output"], "samples": summary["dataset_size"], "selected": summary["selected_intervention"], "ippo": summary["ippo"]}, indent=2))


if __name__ == "__main__":
    main()

