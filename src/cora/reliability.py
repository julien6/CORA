"""Draft Eqs. 5, 6, 10, 11. Bounds are conditional on caller-supplied hypotheses."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BoundAssumptions:
    exact_features: bool
    fixed_context: bool
    fixed_policy: bool
    martingale_subgaussian: bool
    coefficient_norm_bound: float
    noise_bound: float

    def validate(self):
        if not all((self.exact_features, self.fixed_context, self.fixed_policy,
                    self.martingale_subgaussian)):
            raise ValueError("Conditional certification hypotheses are not established")
        if not np.isfinite([self.coefficient_norm_bound, self.noise_bound]).all() or min(self.coefficient_norm_bound, self.noise_bound) < 0:
            raise ValueError("Known finite nonnegative bounds required")


def adaptive_bounds(x, y, targets, assumptions, ridge=1., delta=.05):
    """Time-uniform self-normalized ridge confidence ellipsoid, draft Eq. 6.

    Predictable design and known bounds must hold; this API cannot verify them
    from observations. No graph guess or empirical residual scale suffices.
    """
    assumptions.validate()
    x, y, targets = map(lambda a: np.asarray(a, float), (x, y, targets))
    if x.ndim != 2 or y.shape != (len(x),) or targets.ndim != 2 or targets.shape[1] != x.shape[1] or not 0 < delta < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Invalid dimensions or confidence parameters")
    if not all(np.isfinite(a).all() for a in (x, y, targets)):
        raise ValueError("Nonfinite data")
    p = x.shape[1]
    v = ridge * np.eye(p) + x.T @ x
    inverse = np.linalg.solve(v, np.eye(p))
    theta = inverse @ x.T @ y
    log_ratio = .5 * (np.linalg.slogdet(v)[1] - p * np.log(ridge))
    beta = assumptions.noise_bound * np.sqrt(2 * (log_ratio + np.log(1 / delta))) + np.sqrt(ridge) * assumptions.coefficient_norm_bound
    radius = beta * np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', targets, inverse, targets)))
    return targets @ theta, radius


def fixed_design_bounds(x, y, targets, sigma, delta=.05):
    """OLS simultaneous finite-target bound; fixed independent design required."""
    x, y, targets = map(lambda a: np.asarray(a, float), (x, y, targets))
    if x.ndim != 2 or np.linalg.matrix_rank(x) != x.shape[1] or y.shape != (len(x),) or targets.ndim != 2 or targets.shape[1] != x.shape[1] or not len(targets) or not 0 < delta < 1 or sigma < 0:
        raise ValueError("Full-rank fixed design and valid bounds required")
    if not all(np.isfinite(a).all() for a in (x, y, targets)) or not np.isfinite(sigma):
        raise ValueError("Nonfinite inputs")
    inv = np.linalg.inv(x.T @ x)
    return targets @ inv @ x.T @ y, sigma * np.sqrt(2 * np.log(2 * len(targets) / delta) * np.maximum(0, np.einsum('ij,jk,ik->i', targets, inv, targets)))


def policy_shift_bound(alpha, reward_bound, horizon, gamma=1.):
    """Joint-policy per-history TV bound, not individual-agent TV, Eq. 11."""
    if not 0 <= alpha <= 1 or not np.isfinite(reward_bound) or reward_bound < 0 or not isinstance(horizon, int) or horizon < 1 or not 0 <= gamma <= 1:
        raise ValueError("Invalid shift parameters")
    t = np.arange(horizon)
    return float(2 * reward_bound * np.sum(gamma ** t * (1 - (1 - alpha) ** (t + 1))))


def conservative_decision(predictions, statistical, structural, shift, costs, valid, noop=0):
    """Eq. 10; caller must establish all three simultaneous error bounds."""
    prediction, statistical, structural, shift, costs = map(lambda a: np.asarray(a, float), (predictions, statistical, structural, shift, costs))
    valid = np.asarray(valid, bool)
    if any(a.shape != prediction.shape for a in (statistical, structural, shift, costs, valid)) or prediction.ndim != 1 or not valid[noop]:
        raise ValueError("Invalid candidate set")
    if not all(np.isfinite(a).all() for a in (prediction, statistical, structural, shift, costs)) or any(np.any(a < 0) for a in (statistical, structural, shift, costs)):
        raise ValueError("Finite nonnegative costs and error bounds required")
    b = statistical + structural + shift
    lower = np.where(valid, prediction - costs - b, -np.inf)
    chosen = int(np.argmax(lower))
    return chosen if chosen != noop and lower[chosen] >= prediction[noop] - costs[noop] + b[noop] else noop
