"""Adapters for unchanged pinned upstream JaxMARL trainers and frozen actors."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cora.ippo_adapter import ROOT, UPSTREAM_REVISION
from cora.rollout import guided_actions

SOURCES = {
    "ippo": ("IPPO/ippo_ff_mpe.py", "IPPO/config/ippo_ff_mpe.yaml"),
    "mappo": ("MAPPO/mappo_rnn_mpe.py", "MAPPO/config/mappo_homogenous_rnn_mpe.yaml"),
    "iql": ("QLearning/iql_rnn.py", "QLearning/config/alg/ql_rnn_mpe.yaml"),
    "vdn": ("QLearning/vdn_rnn.py", "QLearning/config/alg/ql_rnn_mpe.yaml"),
    "qmix": ("QLearning/qmix_rnn.py", "QLearning/config/alg/ql_rnn_mpe.yaml"),
}


def upstream(name):
    if name not in SOURCES:
        raise ValueError(f"Unknown upstream learner: {name}")
    checkout = ROOT / "third_party/JaxMARL"
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError("Upstream checkout revision mismatch; run scripts/setup.sh")
    relative = "baselines/" + SOURCES[name][0]
    source = checkout / relative
    original = subprocess.check_output(["git", "-C", str(checkout), "show", f"HEAD:{relative}"])
    if original != source.read_bytes():
        raise RuntimeError(f"Upstream source modified: {relative}")
    spec = importlib.util.spec_from_file_location(f"cora_upstream_{name}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = OmegaConf.to_container(OmegaConf.load(checkout / "baselines" / SOURCES[name][1]))
    if name in ("iql", "vdn", "qmix"):
        common = OmegaConf.to_container(OmegaConf.load(checkout / "baselines/QLearning/config/config.yaml"))
        config = {**common, **config}
    return module, config, hashlib.sha256(original).hexdigest()


@dataclass(eq=False)
class FrozenPolicy:
    name: str
    agents: tuple[str, ...]
    params: object = None
    network: object = None
    hidden_size: int = 1
    version: str = "fixed"

    def initial_hidden(self):
        return jnp.zeros((len(self.agents), self.hidden_size))

    def act(self, observations, hidden, dones, key):
        if self.name == "guided":
            return guided_actions(observations, key, self.agents), hidden
        if self.name == "random":
            values = jax.random.randint(key, (len(self.agents),), 0, 5)
        else:
            obs = jnp.stack([observations[a] for a in self.agents])
            if self.name == "ippo":
                distribution, _ = self.network.apply(self.params, obs)
                values = distribution.sample(seed=key)
            elif self.name == "mappo":
                hidden, distribution = self.network.apply(self.params, hidden, (obs[None], dones[None]))
                values = distribution.sample(seed=key)[0]
            else:
                # CTRolloutManager appends the public agent identity in Q learning.
                obs = jnp.concatenate((obs, jnp.eye(len(self.agents))), axis=-1)
                hidden, q = self.network.apply(self.params, hidden, obs[None], dones[None])
                scores=q[0]
                if 'action_masks' in observations:
                    masks=jnp.stack([observations['action_masks'][a] for a in self.agents])
                    scores=jnp.where(masks,scores,-jnp.inf)
                values = jnp.argmax(scores, axis=-1)
        return {a: values[i] for i, a in enumerate(self.agents)}, hidden


def actor(name, env, params, config, version):
    module, _, _ = upstream(name)
    if name == "ippo":
        network = module.ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
        hidden = 1
    elif name == "mappo":
        network = module.ActorRNN(env.action_space(env.agents[0]).n, config=config)
        hidden = config["GRU_HIDDEN_DIM"]
    else:
        network = module.RNNQNetwork(action_dim=env.action_space(env.agents[0]).n, hidden_dim=config["HIDDEN_SIZE"])
        hidden = config["HIDDEN_SIZE"]
        if name == "qmix":
            params = params["agent"]
    return FrozenPolicy(name, tuple(env.agents), params, network, hidden, version)


def prepare_training(name, env, settings):
    """Construct a trainer without executing any training or evaluation."""
    module, config, digest = upstream(name)
    steps, environments, updates = (int(settings[k]) for k in ("steps", "environments", "updates"))
    if min(steps, environments, updates) < 1 or steps < 2:
        raise ValueError("Positive budgets and at least two training steps required")
    epochs = int(settings.get("epochs", 1))
    minibatches = int(settings.get("minibatches", 1))
    if epochs < 1 or minibatches < 1 or (env.num_agents * environments) % minibatches:
        raise ValueError("Epochs must be positive and minibatches must divide actor count")
    config.update(NUM_STEPS=steps, NUM_ENVS=environments, TOTAL_TIMESTEPS=steps * environments * updates,
                  UPDATE_EPOCHS=epochs, NUM_MINIBATCHES=minibatches, ANNEAL_LR=False,
                  TEST_DURING_TRAINING=False, WANDB_MODE="disabled")
    records = []
    module.wandb = SimpleNamespace(log=lambda metric, **kwargs: records.append(jax.tree.map(lambda x: np.asarray(x).tolist(), metric)))
    module.jaxmarl = SimpleNamespace(make=lambda *args, **kwargs: env)
    if name in ("ippo", "mappo"):
        if name == "mappo":
            config.update(FC_DIM_SIZE=int(settings.get("hidden_size", 32)), GRU_HIDDEN_DIM=int(settings.get("hidden_size", 32)))
        trainer = module.make_train(config)
    else:
        config.update(BUFFER_SIZE=max(4 * environments, updates * environments * 2), BUFFER_BATCH_SIZE=1,
                      HIDDEN_SIZE=int(settings.get("hidden_size", 32)), NUM_EPOCHS=epochs,
                      LEARNING_STARTS=0, LR_LINEAR_DECAY=False, TEST_INTERVAL=1.0,
                      TEST_NUM_ENVS=1, TEST_NUM_STEPS=steps,
                      LOG_AGENTS_SEPARATELY=False, WANDB_MODE="local")
        # "local" activates upstream callbacks but this module's WandB reference
        # only writes to the local records list. No remote client is initialized.
        trainer = module.make_train(config, env)
    return trainer, config, digest, records


def train(name, env, seed, settings, output: Path):
    """Execute only an explicitly requested training budget."""
    trainer, config, digest, records = prepare_training(name, env, settings)
    steps, environments, updates = (int(settings[k]) for k in ("steps", "environments", "updates"))
    start = time.perf_counter()
    result = jax.jit(trainer)(jax.random.PRNGKey(seed))
    jax.block_until_ready(result)
    jax.effects_barrier()
    elapsed = time.perf_counter() - start
    if name == "mappo":
        actor_state, critic_state = result["runner_state"][0][0]
        train_states = (actor_state, critic_state)
        params = actor_state.params
        actual_updates = int(actor_state.step)
    else:
        state = result["runner_state"][0]
        train_states = (state,)
        params = state.params
        actual_updates = int(state.step)
    if actual_updates < 1:
        raise RuntimeError(f"{name} produced no optimizer updates")
    leaves = jax.tree.leaves([(s.params, s.opt_state) for s in train_states])
    if not all(np.isfinite(np.asarray(x)).all() for x in leaves):
        raise RuntimeError(f"Non-finite {name} parameters or optimizer state")
    if not records or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(records)):
        raise RuntimeError(f"Missing or non-finite {name} training metrics")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.msgpack"
    checkpoint.write_bytes(serialization.msgpack_serialize(jax.device_get(params)))
    version = f"{name}-seed{seed}-updates{actual_updates}-{hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]}"
    summary = {"learner": name, "seed": seed, "policy_version": version,
               "optimizer_updates": actual_updates, "training_environment_steps": steps * environments * updates,
               "buffer_initialization_steps": steps * environments if name in ("iql", "vdn", "qmix") else 0,
               "wall_seconds": elapsed, "finite": True, "config": config,
               "upstream_revision": UPSTREAM_REVISION, "upstream_source_sha256": digest,
               "metrics": records, "checkpoint": "checkpoint.msgpack"}
    (output / "training.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    # Save all states, including centralized critic and Q mixer/target state.
    (output / "train_state.msgpack").write_bytes(serialization.to_bytes(tuple(train_states)))
    return actor(name, env, params, config, version), summary


def load_policy(output: Path, env):
    summary = json.loads((output / "training.json").read_text())
    params = jax.tree.map(jnp.asarray, serialization.msgpack_restore((output / "checkpoint.msgpack").read_bytes()))
    return actor(summary["learner"], env, params, summary["config"], summary["policy_version"])
