"""Downward-closed monomial features from fixed dependency hyperedges."""
from dataclasses import dataclass
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FeatureEncoder:
    edits: int
    family: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if self.edits <= 0 or () not in self.family or len(set(self.family)) != len(self.family):
            raise ValueError("Feature family must include intercept and have unique terms")
        for term in self.family:
            if tuple(sorted(set(term))) != term or any(i < 0 or i >= self.edits for i in term):
                raise ValueError("Invalid interaction term")
            for k in range(len(term)):
                if any(sub not in self.family for sub in combinations(term, k)):
                    raise ValueError("Interaction family must be downward closed")

    @classmethod
    def from_hyperedges(cls, edits: int, hyperedges: tuple[tuple[int, ...], ...]):
        terms = {()}
        for edge in hyperedges:
            if tuple(sorted(set(edge))) != edge or any(i < 0 or i >= edits for i in edge):
                raise ValueError("Invalid dependency hyperedge")
            terms.update(sub for k in range(1, len(edge) + 1) for sub in combinations(edge, k))
        return cls(edits, tuple(sorted(terms, key=lambda x: (len(x), x))))

    def encode(self, z: jax.Array) -> jax.Array:
        z = jnp.asarray(z, dtype=jnp.float32)
        if z.shape[-1] != self.edits:
            raise ValueError("Incorrect intervention dimension")
        return jnp.stack([jnp.prod(z[..., jnp.asarray(t, dtype=jnp.int32)], axis=-1) for t in self.family], axis=-1)


def mobius(values: np.ndarray) -> np.ndarray:
    """Exact transform; array indices are integer bitmasks, not product order."""
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or len(result) < 1 or len(result) & (len(result) - 1):
        raise ValueError("Expected power-of-two full factorial vector")
    for j in range((len(result) - 1).bit_length()):
        for mask in range(len(result)):
            if mask & (1 << j):
                result[mask] -= result[mask ^ (1 << j)]
    return result

