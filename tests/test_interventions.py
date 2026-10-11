import numpy as np
import pytest

from cora.interventions import Intervention, apply, candidates
from cora.organization import OrganizationSpec


def test_atomic_order_and_noop():
    spec = OrganizationSpec()
    org = spec.initial()
    assert apply(spec, org, Intervention()) is org
    result = apply(spec, org, Intervention((1, 1, 0)))
    np.testing.assert_array_equal(result.assignments, [2, 0, 1])
    assert int(result.version) == 1
    np.testing.assert_array_equal(org.assignments, [0, 1, 2])
    assert len(candidates(spec, org)) == 8


def test_forbidden_transaction_does_not_commit():
    spec = OrganizationSpec(role_names=("a", "b", "c"), permissions=((True, False, False), (False, True, False), (False, False, True)))
    org = OrganizationSpec().initial().replace(roles=np.array([0, 1, 2], dtype=np.int32))
    with pytest.raises(ValueError):
        apply(spec, org, Intervention((1, 0, 0)))
    np.testing.assert_array_equal(org.assignments, [0, 1, 2])
    # Duplicate resulting assignments can have distinct edit costs; retain edits.
    assert all(np.array_equal(apply(spec, org, z).assignments, org.assignments) for z in candidates(spec, org))
    assert Intervention() in candidates(spec, org)


@pytest.mark.parametrize("bits", [(1, 0), (0, 2, 0), (0.0, 0, 0)])
def test_invalid_edit(bits):
    with pytest.raises(ValueError):
        Intervention(bits)

