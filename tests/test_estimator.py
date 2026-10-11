from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cora.estimator import RidgeEstimator
from cora.features import FeatureEncoder, mobius


def test_mobius_reconstruction_and_interaction_counterexample():
    values = np.array([0, 0, 0, 10.0])
    coef = mobius(values)
    np.testing.assert_array_equal(coef, [0, 0, 0, 10])
    for mask in range(4):
        reconstructed = sum(coef[sub] for sub in range(4) if sub & mask == sub)
        assert reconstructed == values[mask]
    full = FeatureEncoder.from_hyperedges(2, ((0, 1),))
    additive = FeatureEncoder.from_hyperedges(2, ((0,), (1,)))
    z = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    x = full.encode(z)
    np.testing.assert_array_equal(x @ coef, values)
    # Omitted pair cannot represent this ground-truth set function exactly.
    ax = np.asarray(additive.encode(z))
    assert np.linalg.norm(ax @ np.linalg.lstsq(ax, values, rcond=None)[0] - values) > 1


def test_deterministic_fit_shapes_and_identifiability():
    encoder = FeatureEncoder.from_hyperedges(3, ((0, 1, 2),))
    z = jnp.array(list(product((0, 1), repeat=3)))
    x = jax.jit(encoder.encode)(z)
    theta = jnp.arange(8, dtype=jnp.float32)
    model = RidgeEstimator.fit(x, x @ theta, ridge=1e-5)
    assert x.shape == (8, 8) and model.rank == 8
    assert np.all(model.identifiable(x))
    np.testing.assert_allclose(model.predict(x), x @ theta, atol=0.003)
    repeat = RidgeEstimator.fit(x, x @ theta, ridge=1e-5)
    np.testing.assert_array_equal(model.coefficients, repeat.coefficients)
    assert np.isfinite(model.radius(x)).all()


def test_rank_deficient_design_and_decision_identifiability():
    # Coefficients not identifiable, but values on experimental row space are.
    x = np.array([[1, 0, 0], [1, 1, 0]], dtype=float)
    model = RidgeEstimator.fit(x, [0, 1])
    assert model.rank == 2
    np.testing.assert_array_equal(model.identifiable([[1, 0, 0], [1, 1, 0], [1, 0, 1]]), [True, True, False])
    assert model.identifiable(np.array([0, 1, 0]))  # supported difference


def test_invalid_features_and_fit():
    with pytest.raises(ValueError):
        FeatureEncoder(2, ((), (0, 1)))
    with pytest.raises(ValueError):
        RidgeEstimator.fit([[1]], [np.nan])
    with pytest.raises(ValueError):
        RidgeEstimator.fit([[1]], [0], ridge=0)

