"""Ridge value surrogate; leverage radii are uncalibrated diagnostics."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RidgeEstimator:
    coefficients: jax.Array
    precision: jax.Array
    row_projector: jax.Array
    rank: int
    scale: float

    @classmethod
    def fit(cls, x, y, ridge=0.01, scale=1.0):
        x_host, y_host = np.asarray(x), np.asarray(y)
        if x_host.ndim != 2 or x_host.shape[0] == 0 or y_host.shape != (x_host.shape[0],):
            raise ValueError("Expected nonempty N-by-p features and N outcomes")
        if not np.isfinite(x_host).all() or not np.isfinite(y_host).all() or not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(scale) or scale < 0:
            raise ValueError("Finite data, positive ridge, nonnegative scale required")
        x, y = jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
        precision = x.T @ x + ridge * jnp.eye(x.shape[1])
        coef = jnp.linalg.solve(precision, x.T @ y)
        projector = np.linalg.pinv(x_host.astype(float)) @ x_host
        return cls(coef, precision, jnp.asarray(projector), int(np.linalg.matrix_rank(x_host)), float(scale))

    def predict(self, features):
        return jnp.asarray(features) @ self.coefficients

    def radius(self, features):
        f = jnp.asarray(features)
        variance = jnp.sum(f * jnp.linalg.solve(self.precision, f.T).T, axis=-1)
        return self.scale * jnp.sqrt(jnp.maximum(variance, 0))

    def identifiable(self, features, tolerance=1e-5):
        f = jnp.asarray(features)
        return jnp.linalg.norm(f - f @ self.row_projector, axis=-1) <= tolerance

