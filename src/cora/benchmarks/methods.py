"""Intervention baselines sharing the same public information and query budget.

No method receives oracle values, simulator state, or synthetic coefficients.
"""
from dataclasses import dataclass
from itertools import combinations
import time
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from cora.benchmarks.problems import PublicProblem, encode
from cora.reliability import BoundAssumptions, adaptive_bounds, conservative_decision
from cora.benchmarks.planning import value_iteration


METHODS = {
    "cora": "Structured ridge; decision-difference active design; diagnostic conservative commit",
    "cora_conditional": "Structured active design with Eq. 6 bounds; requires explicitly established hypotheses",
    "random": "Uniform experiments; random admissible final intervention",
    "uniform_tabular": "Unstructured arm means via one-hot ridge; uniform design",
    "global_linear": "Global additive linear value model; uniform design",
    "global_neural": "Global MLP value model; uniform design",
    "compositional_modular": "Adapted additive modular architecture from Pruthi/Jensen (2025); uniform design; not a paper reproduction",
    "factorized_uniform": "Same structured ridge as CORA; uniform design and greedy commit",
    "structure_blind_active": "Unstructured arm ridge; decision-difference active design",
    "cora_no_confidence": "CORA acquisition with greedy net-value commit",
    "cora_incomplete_graph": "CORA with singleton-only dependencies",
    "cora_no_constraints": "Synthetic-only ablation admitting forbidden edit combinations",
    "greedy_swap": "Internal singleton-edit reorganization heuristic; not a published baseline",
    "organization_mdp": "Empirical one-decision cost-aware organizational MDP; generic planner, not a paper reproduction",
    "no_op": "Always retain the current organization; no experiments",
    "oracle": "Privileged evaluation-only best admissible net value; no learning budget",
}

UNRESOLVED = {
    "published_compositional": "Paper/implementation not identified in the supplied specification",
    "published_organizational": "Paper/implementation not identified in the supplied specification",
}


class ValueMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(32)(x))
        x = nn.tanh(nn.Dense(32)(x))
        return nn.Dense(1)(x).squeeze(-1)


class ModularValueMLP(nn.Module):
    edges: tuple

    @nn.compact
    def __call__(self, x):
        outputs = []
        for edge in self.edges:
            local = x[:, jnp.array(edge, dtype=jnp.int32)]
            h = nn.tanh(nn.Dense(16)(local))
            outputs.append(nn.Dense(1)(h).squeeze(-1))
        return sum(outputs)


@partial(jax.jit, static_argnames=("edges",))
def fit_modular(x, y, key, edges):
    """Unit-outcome MSE, modular additive architecture; no component oracle labels.

    CCOR adaptation: binary edit subsets replace shared treatment/covariates.
    This changes the source paper's estimand and is explicitly not reproduction.
    """
    model = ModularValueMLP(edges)
    mean, scale = y.mean(), jnp.maximum(y.std(), 1.)
    target = (y - mean) / scale
    params = model.init(key, x)
    optimizer = optax.adam(.01)
    state = optimizer.init(params)
    def update(carry, unused):
        params, state = carry
        loss, gradient = jax.value_and_grad(lambda p: jnp.mean((model.apply(p, x) - target) ** 2))(params)
        updates, state = optimizer.update(gradient, state, params)
        return (optax.apply_updates(params, updates), state), loss
    (params, _), losses = jax.lax.scan(update, (params, state), None, 40)
    return params, mean, scale, losses[-1]


@jax.jit
def fit_neural(x, y, key):
    model = ValueMLP()
    mean, scale = y.mean(), jnp.maximum(y.std(), 1.0)
    target = (y - mean) / scale
    params = model.init(key, x)
    optimizer = optax.adam(0.01)
    state = optimizer.init(params)
    def update(carry, unused):
        params, state = carry
        loss, gradient = jax.value_and_grad(lambda p: jnp.mean((model.apply(p, x) - target) ** 2))(params)
        updates, state = optimizer.update(gradient, state, params)
        return (optax.apply_updates(params, updates), state), loss
    (params, _), losses = jax.lax.scan(update, (params, state), None, 40)
    return params, mean, scale, losses[-1]


@dataclass
class Fit:
    prediction: np.ndarray
    radius: np.ndarray
    precision_inverse: np.ndarray
    rank: int
    identifiable: np.ndarray
    parameters: list


def ridge_fit(features, indices, outcomes, regularization, scale):
    x = features[indices]
    y = np.asarray(outcomes)
    precision = x.T @ x + regularization * np.eye(features.shape[1])
    inverse = np.linalg.solve(precision, np.eye(len(precision)))
    theta = inverse @ x.T @ y
    prediction = features @ theta
    leverage = np.maximum(np.einsum("ij,jk,ik->i", features, inverse, features), 0)
    projector = np.linalg.pinv(x) @ x
    identifiable = np.linalg.norm(features - features @ projector, axis=1) < 1e-6
    return Fit(prediction, scale * np.sqrt(leverage), inverse, int(np.linalg.matrix_rank(x)), identifiable, theta.tolist())


def acquisition(features, fit, admissible, costs, experiment_cost_weight=0.0):
    """Eq. 12: exact minimax decision-difference variance for small finite sets.

    Use all plausibly optimal candidates, then Sherman–Morrison for each query.
    Radius is diagnostic; no confidence certification is attached to this set.
    """
    net = fit.prediction - costs
    threshold = np.max((net - fit.radius)[admissible])
    plausible = admissible[(net + fit.radius)[admissible] >= threshold]
    if len(plausible) < 2:
        plausible = admissible
    f = features[plausible]
    diffs = (f[:, None, :] - f[None, :, :]).reshape(-1, features.shape[1])
    inv = fit.precision_inverse
    scores = []
    for idx in admissible:
        u = inv @ features[idx]
        updated = inv - np.outer(u, u) / (1 + features[idx] @ u)
        uncertainty = np.einsum("ij,jk,ik->i", diffs, updated, diffs).max(initial=0)
        scores.append(uncertainty + experiment_cost_weight * costs[idx])
    return np.asarray(scores)


def run_method(name: str, public: PublicProblem, observe, budgets, seed, config):
    if name in UNRESOLVED:
        raise ValueError(f"{name}: {UNRESOLVED[name]}")
    if name not in METHODS or name == "oracle":
        raise ValueError("Oracle is available only through the evaluation runner")
    if public.name.startswith("simple_spread") and name == "cora_no_constraints":
        raise ValueError("Constraint ablation is synthetic-only; MPE transactions remain validated")
    regularization = float(config["ridge"])
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Positive finite ridge required")
    scale = float(config["diagnostic_scale"])
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Finite nonnegative diagnostic scale required")
    assumptions = None
    if name == "cora_conditional":
        assumptions = BoundAssumptions(**config["confidence_assumptions"])
        assumptions.validate()
        structural = np.asarray(config["structural_error_bounds"], float)
        shift = np.asarray(config["policy_shift_error_bounds"], float)
        if structural.shape != public.costs.shape or shift.shape != public.costs.shape:
            raise ValueError("One justified structural and shift bound per candidate required")
    allowed = np.arange(len(public.bits)) if name == "cora_no_constraints" else np.flatnonzero(public.valid)
    m = public.bits.shape[1]
    if name in ("uniform_tabular", "structure_blind_active", "organization_mdp"):
        features = np.eye(len(public.bits))
    elif name in ("global_linear", "cora_incomplete_graph"):
        features = np.column_stack((np.ones(len(public.bits)), public.bits))
    else:
        features = encode(public.bits, public.family)
    rng = np.random.default_rng(seed)
    queried, outcomes, logs, decisions = [], [], [], []
    start = time.perf_counter()
    max_budget = max(budgets)
    for step in range(max_budget):
        if name == "no_op":
            break
        if step == 0:
            chosen, probability = public.noop, 1.0
        elif name in ("cora", "cora_conditional", "cora_no_confidence", "cora_incomplete_graph", "cora_no_constraints", "structure_blind_active"):
            fit = ridge_fit(features, queried, outcomes, regularization, scale)
            if assumptions is not None:
                fit.prediction, stat = adaptive_bounds(features[queried], outcomes, features, assumptions, regularization, float(config.get("delta", .05)))
                fit.radius = stat + structural + shift
            scores = acquisition(features, fit, allowed, public.costs, float(config.get("experiment_cost_weight", 0)))
            # Randomized ties, with logged actual conditional assignment propensity.
            tied = allowed[np.isclose(scores, scores.min(), rtol=1e-8, atol=1e-10)]
            chosen, probability = int(rng.choice(tied)), 1.0 / len(tied)
        elif name == "greedy_swap":
            singleton = allowed[public.bits[allowed].sum(axis=1) <= 1]
            chosen = int(singleton[step % len(singleton)])
            probability = 1.0
        else:
            chosen, probability = int(rng.choice(allowed)), 1.0 / len(allowed)
        outcome = float(observe(chosen, step))
        if not np.isfinite(outcome):
            raise RuntimeError("Non-finite experiment outcome")
        queried.append(chosen)
        outcomes.append(outcome)
        logs.append({"step": step + 1, "index": chosen, "intervention": public.bits[chosen].tolist(),
                     "outcome": outcome, "cost": float(public.costs[chosen]), "assignment_probability": probability,
                     "policy_version": public.policy_version, "admissible": bool(public.valid[chosen])})
        if step + 1 not in budgets:
            continue
        fit = ridge_fit(features, queried, outcomes, regularization, scale)
        if assumptions is not None:
            fit.prediction, stat = adaptive_bounds(features[queried], outcomes, features, assumptions, regularization, float(config.get("delta", .05)))
            fit.radius = stat + structural + shift
        if name in ("global_neural", "compositional_modular"):
            x = jnp.asarray(public.bits[queried], dtype=jnp.float32)
            y = jnp.array(outcomes)
            key = jax.random.PRNGKey(seed)
            if name == "compositional_modular":
                edges = tuple(t for t in public.family if t and not any(set(t) < set(other) for other in public.family))
                if not edges:
                    edges = (tuple(range(m)),)
                params, mean, sd, loss = fit_modular(x, y, key, edges)
                model = ModularValueMLP(edges)
            else:
                params, mean, sd, loss = fit_neural(x, y, key)
                model = ValueMLP()
            fit.prediction = np.asarray(model.apply(params, jnp.asarray(public.bits, dtype=jnp.float32)) * sd + mean)
            fit.parameters = []
            if not np.isfinite(fit.prediction).all() or not np.isfinite(float(loss)):
                raise RuntimeError("Non-finite neural estimator")
        net = fit.prediction - public.costs
        if name == "random":
            selected = int(rng.choice(allowed))
        elif name == "organization_mdp":
            # The CCOR estimand is one organizational decision followed by a
            # fixed continuation. State 1 is terminal; no invented transition
            # dynamics or privileged future organizational values are used.
            transition = np.zeros((2, len(public.bits), 2))
            transition[:, :, 1] = 1.
            rewards = np.vstack((fit.prediction, np.zeros(len(public.bits))))
            costs = np.vstack((public.costs, np.zeros(len(public.bits))))
            validity = np.vstack((public.valid, np.ones(len(public.bits), bool)))
            planning = value_iteration(transition, rewards, costs, validity, gamma=0.)
            selected = int(planning['policy'][0])
        elif name == "greedy_swap":
            means = {i: np.mean([y for idx, y in zip(queried, outcomes) if idx == i]) for i in set(queried)}
            selected = max(means, key=lambda i: means[i] - public.costs[i])
        elif name == "cora_conditional":
            selected = conservative_decision(fit.prediction, stat, structural, shift, public.costs, public.valid, public.noop)
        elif name in ("cora", "cora_incomplete_graph", "cora_no_constraints"):
            selected = int(allowed[np.argmax((net - fit.radius)[allowed])])
            if not (fit.identifiable[selected] and fit.identifiable[public.noop] and
                    net[selected] - fit.radius[selected] > net[public.noop] + fit.radius[public.noop]):
                selected = public.noop
        else:
            selected = int(allowed[np.argmax(net[allowed])])
        decisions.append({"budget": step + 1, "selected_index": selected,
                          "predictions": fit.prediction.tolist(), "diagnostic_radius": fit.radius.tolist(),
                          "rank": fit.rank, "parameters": fit.parameters,
                          "value_identifiable": fit.identifiable.tolist(), "certified": assumptions is not None,
                          "queries": len(queried), "experiment_cost": sum(public.costs[queried]),
                          "wall_seconds": time.perf_counter() - start})
        if assumptions is not None:
            decisions[-1].update({
                "certification_basis": "conditional on caller-established exact features, predictable martingale design, fixed reference context/policy and known bounds",
                "confidence_assumptions": config["confidence_assumptions"],
                "statistical_radius": stat.tolist(),
                "structural_error_bounds": structural.tolist(),
                "policy_shift_error_bounds": shift.tolist(),
                "delta": float(config.get("delta", .05)),
                "guarantee": "expected net-value non-degradation on simultaneous confidence event; not trajectory safety",
            })
    if name == "no_op":
        decisions = [{"budget": b, "selected_index": public.noop, "predictions": None, "diagnostic_radius": None,
                      "rank": 0, "certified": False, "queries": 0, "experiment_cost": 0,
                      "wall_seconds": time.perf_counter() - start} for b in budgets]
    return decisions, logs
