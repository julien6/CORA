import jax
import numpy as np
import pytest

from cora.estimator import RidgeEstimator
from cora.exploration import uncertainty, uniform
from cora.features import FeatureEncoder
from cora.interventions import Intervention, candidates
from cora.organization import OrganizationSpec
from cora.selector import select


def test_selection_acceptance_cost_and_noop_fallback():
    spec = OrganizationSpec()
    org = spec.initial()
    options = candidates(spec, org)
    encoder = FeatureEncoder.from_hyperedges(3, ((0, 1, 2),))
    x = encoder.encode(np.array([z.bits for z in options]))
    y = np.zeros(8)
    y[1] = 100
    model = RidgeEstimator.fit(x, y, ridge=1e-5, scale=0.01)
    chosen, info = select(spec, org, options, x, model, 0)
    assert chosen == options[1] and not info["certified"]
    chosen, _ = select(spec, org, options, x, model, 1000)
    assert chosen == Intervention()
    flat = RidgeEstimator.fit(x, np.zeros(8))
    assert select(spec, org, options, x, flat, 0)[0] == Intervention()
    unsupported = RidgeEstimator.fit(x[:1], [0])
    assert select(spec, org, options, x, unsupported, 0)[0] == Intervention()
    assert select(spec, org, options, x, model, 0, structural_radius=np.full(8, 1000))[0] == Intervention()
    i, p = uniform(jax.random.PRNGKey(1), 8)
    assert 0 <= i < 8 and p == 0.125
    assert uniform(jax.random.PRNGKey(1), 8) == (i, p)
    i, p = uncertainty(flat, x)
    assert 0 <= i < 8 and p == 1
    with pytest.raises(ValueError):
        select(spec, org, options[1:], x[1:], model, 0)

