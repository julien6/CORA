"""Organizational adapters over pinned, unmodified JaxMARL physics and rewards.

Assignment hints extend local observations. JaxNav delegation also permutes native
navigation goals. World Comm controls only pursuers, with observation-based fixed
foragers. Its homogeneous discrete interface repeats follower movement actions
across communication codes; the leader retains all twenty native actions.
"""
from dataclasses import dataclass
from functools import partial
from itertools import combinations, product
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
import jaxmarl
from jaxmarl.environments.spaces import Box, Discrete
from cora.organization import OrganizationSpec

NATIVE_ENVIRONMENTS = ('world_comm', 'smax', 'overcooked_v2', 'jaxnav')


@dataclass(frozen=True)
class NativeIntervention:
    bits: tuple[int, ...]

    def __post_init__(self):
        if not self.bits or any(type(b) is not int or b not in (0, 1) for b in self.bits):
            raise ValueError('Nonempty binary intervention required')


@struct.dataclass
class NativeState:
    env_state: object
    organization: object
    mission_goals: jax.Array


class NativeOrganizationEnv:
    def __init__(self, domain, horizon=25, **options):
        if domain not in NATIVE_ENVIRONMENTS or horizon < 2:
            raise ValueError('Expected native JaxMARL environment and horizon >= 2')
        self.domain=domain;self.horizon=horizon
        kwargs=dict(options)
        if domain=='world_comm':
            from jaxmarl.environments.mpe.default_params import DISCRETE_ACT
            kwargs.setdefault('action_type',DISCRETE_ACT)
            if kwargs['action_type']!=DISCRETE_ACT:raise ValueError('World Comm learner interface requires discrete actions')
            self._env=jaxmarl.make('MPE_simple_world_comm_v3',max_steps=horizon,**kwargs)
            self.agents=tuple([self._env.leader,*self._env.adversaries])
            count=20
            missions=tuple(f'pursuit_sector_{i}' for i in range(len(self.agents)))
        elif domain=='smax':
            from jaxmarl.environments.smax import map_name_to_scenario
            scenario=kwargs.pop('scenario','3m')
            self._env=jaxmarl.make('HeuristicEnemySMAX',scenario=map_name_to_scenario(scenario),max_steps=horizon,see_enemy_actions=False,**kwargs)
            self.agents=tuple(self._env.agents);count=self._env.action_space(self.agents[0]).n
            missions=tuple(f'focus_target_{i}' for i in range(len(self.agents)))
        elif domain=='overcooked_v2':
            kwargs.setdefault('layout','cramped_room_v2')
            self._env=jaxmarl.make('overcooked_v2',max_steps=horizon,**kwargs)
            self.agents=tuple(self._env.agents);count=self._env.action_space(self.agents[0]).n
            missions=tuple(('ingredient_cook','plate_deliver') if len(self.agents)==2 else (f'kitchen_duty_{i}' for i in range(len(self.agents))))
        else:
            from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
            kwargs.setdefault('num_agents',3);kwargs.setdefault('map_params',{'map_size':(7,7),'fill':.2})
            kwargs.setdefault('lidar_num_beams',32)
            self._env=JaxNav(act_type='Discrete',normalise_obs=True,max_steps=horizon,**kwargs)
            self.agents=tuple(self._env.agents);count=self._env.action_space(self.agents[0]).n
            missions=tuple(f'navigation_goal_{i}' for i in range(len(self.agents)))
        self.agents=list(self.agents)  # Upstream Q trainers concatenate agent lists.
        self.num_agents=len(self.agents)
        if not 2<=self.num_agents<=4:raise ValueError('Finite reference suite supports two to four controlled agents')
        self.spec=OrganizationSpec(tuple(self.agents),missions,permissions=(tuple(True for _ in missions),))
        self.initial=self.spec.initial();self.swaps=tuple(combinations(range(self.num_agents),2))
        self.observation_length=max(int(np.prod(self._env.observation_space(a).shape)) for a in self.agents)
        self.action_spaces={a:Discrete(count) for a in self.agents}
        self.observation_spaces={a:Box(-jnp.inf,jnp.inf,(self.observation_length+self.num_agents,)) for a in self.agents}
        self.options=options

    @property
    def name(self):return f'CORA-native-{self.domain}'

    def action_space(self, agent):return self.action_spaces[agent]
    def observation_space(self, agent):return self.observation_spaces[agent]

    def augment(self,obs,org):
        result={a:jnp.concatenate((jnp.pad(obs[a].reshape(-1),(0,self.observation_length-obs[a].size)),jax.nn.one_hot(org.assignments[i],self.num_agents))) for i,a in enumerate(self.agents)}
        # Centralized critics may consume these local observations; actors do not.
        result['world_state']=jnp.concatenate([result[a] for a in self.agents])
        return result

    def get_obs(self,state):
        obs=self._env._get_obs(state.env_state) if self.domain=='jaxnav' else self._env.get_obs(state.env_state)
        if self.domain=='jaxnav':obs={a:obs[i] for i,a in enumerate(self.agents)}
        result=self.augment(obs,state.organization)
        if self.domain=='smax':result['action_masks']=self._env.get_avail_actions(state.env_state)
        return result

    def reset(self,key):
        return self._reset(key,self.initial)

    @partial(jax.jit,static_argnums=0)
    def _reset(self,key,organization):
        _,inner=self._env.reset(key)
        goals=inner.goal if self.domain=='jaxnav' else jnp.zeros((self.num_agents,2))
        if self.domain=='jaxnav':inner=inner.replace(goal=goals[organization.assignments])
        state=NativeState(inner,organization,goals)
        return self.get_obs(state),state

    def native_actions(self,state,actions):
        actions={a:jnp.asarray(actions[a]).reshape(()) for a in self.agents}
        if self.domain!='world_comm':return actions
        obs=self._env.get_obs(state.env_state)
        result={a:actions[a] if a==self._env.leader else actions[a]%5 for a in self.agents}
        for a in self._env.good_agents:
            food=obs[a][6:10].reshape(2,2);delta=food[jnp.argmin(jnp.linalg.norm(food,axis=1))]
            desired=1.2*delta-obs[a][:2];axis=jnp.argmax(abs(desired))
            result[a]=jnp.where(jnp.linalg.norm(desired)<.03,0,1+2*axis+(desired[axis]>=0).astype(jnp.int32))
        return result

    @partial(jax.jit,static_argnums=0)
    def step(self,key,state,actions):
        _,inner,rewards,dones,info=self._env.step(key,state.env_state,self.native_actions(state,actions))
        goals=state.mission_goals
        if self.domain=='jaxnav':
            goals=jnp.where(dones['__all__'],inner.goal,goals)
            inner=inner.replace(goal=goals[state.organization.assignments])
        result=NativeState(inner,state.organization,goals)
        # Upstream MPE-style loggers expect one telemetry value per actor.
        info=jax.tree.map(lambda x:jnp.broadcast_to(jnp.asarray(x),(self.num_agents,)) if jnp.asarray(x).size==1 else jnp.asarray(x),info)
        # The objective is the controlled team mean. Never mix competitive teams.
        return self.get_obs(result),result,{a:rewards[a] for a in self.agents},{**{a:dones[a] for a in self.agents},'__all__':dones['__all__']},info

    @partial(jax.jit,static_argnums=0)
    def evaluation_step(self,key,state,actions):
        _,inner,rewards,dones,info=self._env.step_env(key,state.env_state,self.native_actions(state,actions))
        result=state.replace(env_state=inner)
        return self.get_obs(result),result,{a:rewards[a] for a in self.agents},{**{a:dones[a] for a in self.agents},'__all__':dones['__all__']},info

    @partial(jax.jit,static_argnums=0)
    def get_avail_actions(self,state):
        if self.domain=='smax':
            inner=state
            while hasattr(inner,'env_state'):inner=inner.env_state
            return self._env.get_avail_actions(inner)
        return {a:jnp.ones(self.action_spaces[a].n,dtype=bool) for a in self.agents}

    def apply_organization(self,org,z):
        self.spec.validate(org)
        if len(z.bits)!=len(self.swaps):raise ValueError('Edit dimension does not match native organization')
        assignments=org.assignments
        for bit,(i,j) in zip(z.bits,self.swaps):
            if bit:assignments=assignments.at[i].set(assignments[j]).at[j].set(assignments[i])
        result=org.replace(assignments=assignments,version=org.version+jnp.asarray(any(z.bits),dtype=jnp.int32))
        self.spec.validate(result);return result

    def candidates(self,org):
        return tuple(NativeIntervention(tuple(bits)) for bits in product((0,1),repeat=len(self.swaps)) if self._valid(org,bits))

    def _valid(self,org,bits):
        try:self.apply_organization(org,NativeIntervention(tuple(bits)));return True
        except ValueError:return False

    def intervene(self,state,z):
        org=self.apply_organization(state.organization,z);inner=state.env_state
        if self.domain=='jaxnav':inner=inner.replace(goal=state.mission_goals[org.assignments])
        result=state.replace(organization=org,env_state=inner)
        return self.get_obs(result),result

    def protocol_metadata(self):
        return {'backend':'native_jaxmarl','environment':self.domain,'controlled_agents':list(self.agents),'missions':list(self.spec.missions),'ordered_swaps':[list(s) for s in self.swaps],'organizational_intervention':'assignment hints; goal delegation also changes native goals in JaxNav','objective':'mean return of controlled agents','world_comm_opponents':'fixed local-observation food-seeking foragers' if self.domain=='world_comm' else None,'world_comm_action_mapping':'leader: native 20 actions; follower: code modulo 5' if self.domain=='world_comm' else None,'dependency_support':'conservative complete interaction family; no locality reduction assumed','options':self.options,'horizon':self.horizon}


@dataclass(eq=False)
class NativeRandomPolicy:
    agents: tuple
    action_counts: tuple
    name: str='random'
    version: str='native-random-v1'

    def initial_hidden(self):return jnp.zeros((len(self.agents),1))
    def act(self,obs,hidden,dones,key):
        keys=jax.random.split(key,len(self.agents))
        return {a:jax.random.randint(keys[i],(),0,self.action_counts[i]) for i,a in enumerate(self.agents)},hidden


def make_native_env(domain, environment):
    config=dict(environment);options=config.pop('options',{})
    if set(config)-{'horizon'}:raise ValueError('Native configurations accept horizon and per-environment options; custom-domain physics removed')
    if set(options)-set(NATIVE_ENVIRONMENTS):raise ValueError('Unknown native environment options')
    return NativeOrganizationEnv(domain,**config,**options.get(domain,{}))
