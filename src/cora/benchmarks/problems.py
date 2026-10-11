"""Public decision metadata and evaluation-only ground-truth/oracle backends."""
from dataclasses import dataclass
from itertools import combinations, product

import jax
import jax.numpy as jnp
import numpy as np

from cora.env_wrapper import OrganizationWrapper
from cora.features import FeatureEncoder
from cora.interventions import apply, candidates
from cora.organization import OrganizationSpec
from cora.benchmarks.learners import FrozenPolicy
from cora.benchmarks.simulation import simulate


def encode(bits, family):
    bits = np.asarray(bits, dtype=float)
    return np.stack([np.prod(bits[:, term], axis=1) if term else np.ones(len(bits)) for term in family], axis=1)


@dataclass(frozen=True)
class PublicProblem:
    name: str
    bits: np.ndarray
    costs: np.ndarray
    valid: np.ndarray
    family: tuple[tuple[int, ...], ...]
    noise_scale: float
    policy_version: str

    def __post_init__(self):
        if self.bits.ndim != 2 or not np.all((self.bits == 0) | (self.bits == 1)):
            raise ValueError("Interventions must be a binary matrix")
        if self.costs.shape != (len(self.bits),) or self.valid.shape != self.costs.shape:
            raise ValueError("Candidate metadata dimensions do not match")
        if not np.isfinite(self.costs).all() or np.any(self.costs < 0):
            raise ValueError("Finite nonnegative costs required")
        if np.count_nonzero(np.all(self.bits == 0, axis=1)) != 1 or not self.valid[self.noop]:
            raise ValueError("Exactly one valid no-op required")
        if not np.isfinite(self.noise_scale) or self.noise_scale < 0:
            raise ValueError("Finite nonnegative noise scale required")
        FeatureEncoder(self.bits.shape[1], self.family)

    @property
    def noop(self):
        return int(np.flatnonzero(np.all(self.bits == 0, axis=1))[0])


class SyntheticTrial:
    """Truth and noise available only to the experiment/evaluation service."""
    def __init__(self, config, seed, scenario="sparse"):
        m = int(config["edits"])
        if not 2 <= m <= 10:
            raise ValueError("Exact enumerated synthetic runner supports 2–10 edits")
        order = int(config["interaction_order"])
        density = float(config["density"])
        if not 1 <= order <= m or not 0 <= density <= 1:
            raise ValueError("Invalid interaction order or density")
        rng = np.random.default_rng(seed)
        bits = np.array(list(product((0, 1), repeat=m)), dtype=int)
        if scenario == "dense":
            hyperedges = (tuple(range(m)),)
        else:
            width = min(m, max(order, 3))
            hyperedges = tuple(tuple(range(start, min(start + width, m))) for start in range(0, m, max(1, width - 1)))
        full_family = FeatureEncoder.from_hyperedges(m, hyperedges).family
        terms = [term for term in full_family if len(term) <= order]
        coefficients = rng.normal(size=len(terms))
        for i, term in enumerate(terms):
            if len(term) > 1 and rng.random() > density:
                coefficients[i] = 0
        # Generating coefficients is oracle-side; methods receive only hyperedges.
        self.truth = encode(bits, tuple(terms)) @ coefficients
        self.reference_truth = self.truth.copy()
        if scenario == "policy_shift":
            shift = float(config.get("shift_magnitude", 0.5))
            self.truth += shift * (bits[:, 0] - bits[:, 1])
        forbidden = tuple(tuple(pair) for pair in config.get("forbidden_pairs", []))
        valid = np.ones(len(bits), dtype=bool)
        for i, j in forbidden:
            if i == j or not 0 <= i < m or not 0 <= j < m:
                raise ValueError("Invalid forbidden edit pair")
            valid &= ~(bits[:, i].astype(bool) & bits[:, j].astype(bool))
        public_family = full_family
        if scenario == "misspecified":
            public_family = FeatureEncoder.from_hyperedges(m, tuple((i,) for i in range(m))).family
        noise = float(config["noise"])
        self.public = PublicProblem(f"synthetic_{scenario}", bits, float(config["edit_cost"]) * bits.sum(axis=1), valid,
                                    public_family, noise, "synthetic-reference-v1")
        self.seed = seed
        self.oracle_standard_error = np.zeros(len(bits))
        self.oracle_samples = 0
        self.metadata = {"truth_kind": "exact", "scenario": scenario, "hyperedges": [list(e) for e in hyperedges],
                         "true_family": [list(t) for t in terms], "coefficients": coefficients.tolist(),
                         "deployment_policy_shift": scenario == "policy_shift"}

    def observe(self, index, step):
        noise = np.random.default_rng(np.random.SeedSequence([self.seed, 211, step])).normal(scale=self.public.noise_scale)
        return float(self.reference_truth[index] + noise)


class MPETrial:
    def __init__(self, env, policy, config, seed):
        self.env, self.policy, self.config, self.seed = env, policy, config, seed
        _, self.base = env.reset(jax.random.PRNGKey(seed))
        self.options = candidates(env.spec, self.base.organization)
        bits = np.array([z.bits for z in self.options], dtype=int)
        family = FeatureEncoder.from_hyperedges(3, ((0, 1, 2),)).family
        self.public = PublicProblem(f"simple_spread_{policy.name}", bits, np.array([z.cost(config["swap_cost"]) for z in self.options]),
                                    np.ones(len(bits), dtype=bool), family, float(config["diagnostic_scale"]), policy.version)
        repeats = int(config["oracle_repeats"])
        if repeats < 2:
            raise ValueError("Monte Carlo oracle needs at least two replicates")
        raw = np.array([[self.value(i, 100000 + rep) for i in range(len(bits))] for rep in range(repeats)])
        self.truth = raw.mean(axis=0)
        self.oracle_standard_error = raw.std(axis=0, ddof=1) / np.sqrt(repeats)
        self.oracle_samples = raw.size
        self.metadata = {"truth_kind": "monte_carlo", "oracle_repeats": repeats,
                         "oracle_values": raw.tolist(), "reset_key": [0, seed],
                         "policy_version": policy.version, "horizon": config["horizon"], "gamma": config["gamma"]}

    def trace(self, index, stream):
        org = apply(self.env.spec, self.base.organization, self.options[index])
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), stream)
        return simulate(self.env, self.policy, self.base.replace(organization=org), key, int(self.config["horizon"]), float(self.config["gamma"]))

    def value(self, index, stream):
        return float(self.trace(index, stream)["return"])

    def observe(self, index, step):
        return self.value(index, step + 1)

