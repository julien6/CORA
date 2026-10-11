"""Functional JAX application simulator with JaxMARL learner interfaces.

Three agents, five movement actions, local sensing, heterogeneous service rates,
mission precedence and finite shared service capacity. This is an abstract model.
"""
from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.spaces import Box, Discrete
from cora.env_wrapper import OrganizedState
from cora.organization import OrganizationSpec
from cora.interventions import apply


@struct.dataclass
class DomainState:
    positions: object
    queue: object
    completed: object
    failed: object
    resource: object
    step: object


class DomainEnv:
    agents = ['agent_0','agent_1','agent_2']
    num_agents = 3
    num_landmarks = 3
    name = 'CORA_domain'

    def __init__(self, domain='warehouse', horizon=25, disruption_step=10, capacity=2, radius=2):
        if domain not in ('warehouse','rescue','wildfire','hospital') or horizon < 2 or not 1 <= capacity <= 3 or radius < 0:
            raise ValueError('Invalid application configuration')
        self.domain,self.horizon,self.disruption_step,self.capacity,self.radius=domain,horizon,disruption_step,capacity,radius
        self.spec=OrganizationSpec(missions=('stage_0','stage_1','stage_2'))
        self.initial=self.spec.initial()
        self.sites=jnp.array([[1,1],[3,1],[3,3]])
        self.rates=jnp.array([[1.,.6,.4],[.5,1.,.7],[.7,.5,1.]])
        self.action_spaces={a:Discrete(5) for a in self.agents}
        self.observation_spaces={a:Box(-jnp.inf,jnp.inf,(19,)) for a in self.agents}

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def get_obs(self,state):
        s=state.env_state
        result={}
        for i,a in enumerate(self.agents):
            relative=self.sites-s.positions[i]
            visible=jnp.abs(relative).sum(axis=1)<=self.radius
            # Unseen queues are masked, with explicit visibility indicators.
            result[a]=jnp.concatenate((s.positions[i]/4,relative.reshape(-1)/4,
                jnp.where(visible,s.queue,0)/10,visible.astype(float),
                jnp.array([s.failed[i],s.step/self.horizon],dtype=float)))
            result[a]=jnp.concatenate((result[a],jax.nn.one_hot(state.organization.assignments[i],3)))
        return result

    @partial(jax.jit,static_argnums=0)
    def reset(self,key):
        queue=jnp.array([2.,2.,2.]) if self.domain=='wildfire' else jnp.array([3.,0.,0.])
        inner=DomainState(jnp.array([[0,0],[0,2],[2,0]]),queue,jnp.zeros(3),jnp.zeros(3,bool),jnp.array(12.),jnp.array(0))
        state=OrganizedState(inner,self.initial)
        return self.get_obs(state),state

    @partial(jax.jit,static_argnums=0)
    def step(self,key,state,actions):
        s=state.env_state
        movement=jnp.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])
        action=jnp.stack([actions[a] for a in self.agents]).reshape(3)
        proposed=jnp.clip(s.positions+movement[action],0,4)
        failed=s.failed.at[0].set(s.failed[0] | (s.step>=self.disruption_step))
        proposed=jnp.where(failed[:,None],s.positions,proposed)
        # Simultaneous collision resolution rejects duplicate target cells.
        collision=(jnp.all(proposed[:,None,:]==proposed[None,:,:],axis=-1)&~jnp.eye(3,dtype=bool)).any(axis=1)
        positions=jnp.where(collision[:,None],s.positions,proposed)
        mission=state.organization.assignments
        request=jnp.all(positions==self.sites[mission],axis=1)&~failed&(s.queue[mission]>0)
        if self.domain=='hospital':
            request=request&~((mission==2)&(s.queue[1]>0))
        if self.domain=='wildfire':
            request=request&(s.resource>0)
        service_key,arrival_key=jax.random.split(key)
        # Stable agent order defines contention allocation; no hidden scheduler randomness.
        eligible=request&(jnp.cumsum(request)<=self.capacity)
        if self.domain=='wildfire':
            eligible=eligible&(jnp.cumsum(eligible)<=s.resource)
        success=eligible&(jax.random.uniform(service_key,(3,))<self.rates[jnp.arange(3),mission])
        served=jnp.zeros(3).at[mission].add(success.astype(float))
        queue=jnp.maximum(0,s.queue-served)
        if self.domain!='wildfire':
            queue=queue.at[1:].add(served[:-1])
        if self.domain=='wildfire':
            queue=queue+jax.random.bernoulli(arrival_key,jnp.minimum(.7,.1+.1*queue)).astype(float)
        elif self.domain in ('warehouse','hospital'):
            probability=.15 if self.domain=='warehouse' else .4
            queue=queue.at[0].add(jax.random.bernoulli(arrival_key,probability).astype(float))
        queue=queue.at[0].add(jnp.where(s.step==self.disruption_step,2.,0.))
        resource=s.resource-served.sum() if self.domain=='wildfire' else s.resource
        inner=DomainState(positions,queue,s.completed+served,failed,resource,s.step+1)
        new=state.replace(env_state=inner)
        reward=served.sum()-.05*queue.sum()-.1*collision.sum()
        done=inner.step>=self.horizon
        rewards={a:reward for a in self.agents}
        dones={a:done for a in self.agents};dones['__all__']=done
        # Upstream MPE PPO expects each info leaf to have an agent axis.
        info={k:jnp.full((3,),v) for k,v in {'backlog':queue.sum(),'throughput':inner.completed[2],'congestion':collision.sum(),'resource':resource}.items()}
        _,reset=self.reset(key)
        reset=reset.replace(organization=state.organization)
        returned=jax.tree.map(lambda r,n:jnp.where(done,r,n),reset,new)
        return self.get_obs(returned),returned,rewards,dones,info

    def intervene(self,state,z):
        new=state.replace(organization=apply(self.spec,state.organization,z))
        return self.get_obs(new),new
