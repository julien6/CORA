"""Binary edit sets with explicit ascending-index atomic ordering."""
from dataclasses import dataclass
from itertools import product
import math

import jax.numpy as jnp

from cora.organization import Organization, OrganizationSpec

SWAPS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Intervention:
    bits: tuple[int, ...] = (0, 0, 0)

    def __post_init__(self):
        if len(self.bits) != len(SWAPS) or any(type(b) is not int or b not in (0, 1) for b in self.bits):
            raise ValueError("Expected three binary edit indicators")

    def cost(self, swap_cost: float) -> float:
        if not math.isfinite(swap_cost) or swap_cost < 0:
            raise ValueError("Swap cost must be finite and nonnegative")
        return sum(self.bits) * swap_cost


def apply(spec: OrganizationSpec, org: Organization, z: Intervention) -> Organization:
    """Validate before commit; overlapping swaps execute in SWAPS order.

    Only the final transaction is exposed; invalid results leave the input intact.
    The no-op preserves all fields including version.
    """
    spec.validate(org)
    if len(spec.agents) != 3:
        raise ValueError("This swap language is defined for three agents")
    if not any(z.bits):
        return org
    a = org.assignments
    for bit, (i, j) in zip(z.bits, SWAPS):
        if bit:
            a = a.at[i].set(a[j]).at[j].set(a[i])
    result = org.replace(assignments=a, version=org.version + jnp.asarray(any(z.bits), dtype=jnp.int32))
    spec.validate(result)
    return result


def candidates(spec: OrganizationSpec, org: Organization) -> tuple[Intervention, ...]:
    spec.validate(org)
    valid = []
    for bits in product((0, 1), repeat=3):
        z = Intervention(bits)
        try:
            apply(spec, org, z)
        except ValueError:
            continue
        valid.append(z)
    return tuple(valid)

