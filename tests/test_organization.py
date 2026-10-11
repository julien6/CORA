import jax.numpy as jnp
import pytest

from cora.organization import OrganizationSpec


@pytest.mark.parametrize("assignments", [[0, 0, 2], [0, 1, 3], [-1, 1, 2], [0, 1], [0.0, 1.0, 2.0]])
def test_reject_invalid_assignments(assignments):
    spec = OrganizationSpec()
    with pytest.raises(ValueError):
        spec.validate(spec.initial().replace(assignments=jnp.array(assignments)))


def test_permissions_and_roles():
    spec = OrganizationSpec()
    with pytest.raises(ValueError):
        spec.validate(spec.initial().replace(roles=jnp.array([0, 1, 0])))
    restricted = OrganizationSpec(permissions=((True, False, True),))
    with pytest.raises(ValueError):
        restricted.validate(spec.initial())

