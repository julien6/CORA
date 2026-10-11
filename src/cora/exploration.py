"""Tiny admissible experimental design utilities."""
import jax
import jax.numpy as jnp


def uniform(key, count: int) -> tuple[int, float]:
    if count <= 0:
        raise ValueError("No admissible experiments")
    return int(jax.random.randint(key, (), 0, count)), 1.0 / count


def uncertainty(model, features) -> tuple[int, float]:
    """Deterministic maximum-leverage acquisition, with first-index ties.

    This is a smoke-test heuristic, not the draft's best-decision design.
    """
    if len(features) == 0:
        raise ValueError("No admissible experiments")
    return int(jnp.argmax(model.radius(features))), 1.0

