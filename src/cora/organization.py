"""Small symbolic model with JAX-compatible dynamic assignments."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class Organization:
    assignments: jax.Array  # agent index -> mission index
    roles: jax.Array  # agent index -> role index
    version: jax.Array


@dataclass(frozen=True)
class OrganizationSpec:
    agents: tuple[str, ...] = ("agent_0", "agent_1", "agent_2")
    missions: tuple[str, ...] = ("landmark_0", "landmark_1", "landmark_2")
    role_names: tuple[str, ...] = ("executor",)
    # Role-to-mission permission relation; one executor can serve any mission.
    permissions: tuple[tuple[bool, ...], ...] = ((True, True, True),)

    def __post_init__(self):
        if not self.agents or len(self.agents) != len(self.missions):
            raise ValueError("PoC requires equal nonzero agent and mission counts")
        if len(set(self.agents)) != len(self.agents) or len(set(self.missions)) != len(self.missions):
            raise ValueError("Agent and mission names must be unique")
        if np.asarray(self.permissions).shape != (len(self.role_names), len(self.missions)):
            raise ValueError("Permission relation has incorrect shape")

    def initial(self) -> Organization:
        org = Organization(jnp.arange(len(self.agents)), jnp.zeros(len(self.agents), dtype=jnp.int32), jnp.array(0))
        self.validate(org)
        return org

    def validate(self, org: Organization) -> None:
        """Host boundary: reject malformed values before entering JIT code."""
        a, r = np.asarray(org.assignments), np.asarray(org.roles)
        n = len(self.agents)
        if a.shape != (n,) or r.shape != (n,) or not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(r.dtype, np.integer):
            raise ValueError("Assignments and roles must be integer vectors of agent count")
        if sorted(a.tolist()) != list(range(n)):
            raise ValueError("Every mission must have exactly one assigned executor")
        if np.any(r < 0) or np.any(r >= len(self.role_names)):
            raise ValueError("Unknown role")
        if not np.all(np.asarray(self.permissions)[r, a]):
            raise ValueError("Role is not permitted to execute assigned mission")
        v = np.asarray(org.version)
        if v.shape != () or not np.issubdtype(v.dtype, np.integer) or v < 0:
            raise ValueError("Organization version must be a nonnegative integer scalar")

