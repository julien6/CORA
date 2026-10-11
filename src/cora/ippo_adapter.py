"""Reuse pinned upstream IPPO unchanged; inject environment and local logging."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import hashlib
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

UPSTREAM_REVISION = "976aeb152cb184a5095021968bba94da96eb6394"
ROOT = Path(__file__).resolve().parents[2]


def load_upstream():
    checkout = ROOT / "third_party/JaxMARL"
    if not checkout.is_dir():
        raise RuntimeError("Run scripts/setup.sh to fetch pinned JaxMARL source")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError("JaxMARL checkout does not match pinned revision")
    source = checkout / "baselines/IPPO/ippo_ff_mpe.py"
    # Verify the actual code as well as HEAD; source changes can invalidate reuse.
    original = subprocess.check_output(["git", "-C", str(checkout), "show", "HEAD:baselines/IPPO/ippo_ff_mpe.py"])
    if source.read_bytes() != original:
        raise RuntimeError("Upstream IPPO source has local modifications")
    spec = importlib.util.spec_from_file_location("cora_upstream_ippo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(original).hexdigest()


def smoke_update(env, seed: int, steps: int = 10):
    if not 1 <= steps <= 20:
        raise ValueError("IPPO smoke rollout must have 1–20 steps")
    module, digest = load_upstream()
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "third_party/JaxMARL/baselines/IPPO/config/ippo_ff_mpe.yaml"))
    config.update(NUM_ENVS=1, NUM_STEPS=steps, TOTAL_TIMESTEPS=steps,
                  UPDATE_EPOCHS=1, NUM_MINIBATCHES=1, ANNEAL_LR=False)
    logged = []
    # Replace references in this privately loaded module; no global monkeypatch,
    # upstream file edit, wandb initialization, or remote logging is needed.
    module.jaxmarl = SimpleNamespace(make=lambda *args, **kwargs: env)
    module.wandb = SimpleNamespace(log=lambda metric: logged.append(metric))
    key = jax.random.PRNGKey(seed)
    network = module.ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    _, init_key = jax.random.split(key)
    initial_params = network.init(init_key, jnp.zeros(env.observation_space(env.agents[0]).shape))
    result = jax.jit(module.make_train(config))(key)
    jax.block_until_ready(result)
    train_state = result["runner_state"][0]
    leaves = jax.tree.leaves((train_state.params, train_state.opt_state, result["metrics"]))
    if not all(np.isfinite(np.asarray(v)).all() for v in leaves):
        raise RuntimeError("IPPO generated non-finite parameters, optimizer state or metrics")
    delta = float(jnp.sqrt(sum(jnp.sum((a - b) ** 2) for a, b in zip(jax.tree.leaves(initial_params), jax.tree.leaves(train_state.params)))))
    if int(train_state.step) != 1 or delta <= 0 or len(logged) != 1:
        raise RuntimeError("Expected exactly one finite, nonzero optimizer update")
    # Adam's first moment after the first update is (1-beta1)*clipped_gradient.
    # This verifies finiteness of the actual gradient used by the upstream trainer.
    adam_state = train_state.opt_state[1][0]
    gradient_norm = float(jnp.sqrt(sum(jnp.sum((v / 0.1) ** 2) for v in jax.tree.leaves(adam_state.mu))))
    if not np.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("Expected finite nonzero applied gradients")
    return {
        "optimizer_updates": int(train_state.step), "rollout_steps": steps,
        "parameter_delta_l2": delta, "applied_gradient_l2": gradient_norm,
        "finite": True, "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_sha256": digest,
        "metrics": {k: np.asarray(v).tolist() for k, v in result["metrics"].items()},
    }

