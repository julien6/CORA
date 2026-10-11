"""Authorized assignment observations; original MPE actions/rewards unchanged."""
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.spaces import Box

from cora.organization import Organization, OrganizationSpec
from cora.interventions import Intervention, apply


@struct.dataclass
class OrganizedState:
    env_state: object
    organization: Organization


class OrganizationWrapper:
    def __init__(self, env, spec: OrganizationSpec, organization: Organization | None = None):
        self._env = env
        self.spec = spec
        self.initial = organization if organization is not None else spec.initial()
        spec.validate(self.initial)
        if tuple(env.agents) != spec.agents or env.num_landmarks != len(spec.missions):
            raise ValueError("Environment entities do not match organization")
        self.observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (env.observation_space(a).shape[0] + len(spec.missions),))
            for a in env.agents
        }

    def __getattr__(self, name):
        return getattr(self._env, name)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def augment(self, observations, org):
        # Each agent sees only its own assignment. No new simulator state is exposed.
        return {a: jnp.concatenate((observations[a], jax.nn.one_hot(org.assignments[i], len(self.spec.missions))))
                for i, a in enumerate(self.agents)}

    def get_obs(self, state):
        return self.augment(self._env.get_obs(state.env_state), state.organization)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, state = self._env.reset(key)
        return self.augment(obs, self.initial), OrganizedState(state, self.initial)

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, actions):
        obs, inner, rewards, dones, info = self._env.step(key, state.env_state, actions)
        # Organization persists over upstream automatic episode reset.
        return self.augment(obs, state.organization), OrganizedState(inner, state.organization), rewards, dones, info

    def intervene(self, state, z: Intervention):
        new_state = state.replace(organization=apply(self.spec, state.organization, z))
        return self.get_obs(new_state), new_state

