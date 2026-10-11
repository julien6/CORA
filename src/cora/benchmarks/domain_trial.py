"""Application trials using the same public experiment API as MPE."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from cora.benchmarks.problems import PublicProblem
from cora.benchmarks.simulation import simulate
from cora.features import FeatureEncoder
from cora.interventions import candidates


@dataclass(eq=False)
class DomainGuidedPolicy:
    agents: tuple
    name: str = 'domain_guided'
    version: str = 'domain-guided-v2'

    def initial_hidden(self):
        return jnp.zeros((3,1))

    def act(self,observations,hidden,dones,key):
        result={}
        for a in self.agents:
            obs=observations[a]
            mission=jnp.argmax(obs[-3:])
            delta=obs[2:8].reshape(3,2)[mission]
            axis=jnp.argmax(jnp.abs(delta))
            movement=jnp.where(axis==0,jnp.where(delta[0]<0,1,2),jnp.where(delta[1]<0,3,4))
            result[a]=jnp.where(jnp.all(delta==0),0,movement)
        return result,hidden


class DomainTrial:
    def __init__(self,env,policy,config,seed,context_round=0):
        self.env,self.policy,self.config,self.seed=env,policy,config,seed
        _,self.base=env.reset(jax.random.PRNGKey(seed))
        self.options=candidates(env.spec,self.base.organization)
        bits=np.asarray([z.bits for z in self.options],int)
        self.public=PublicProblem(f'{env.domain}_{policy.name}_round{context_round}',bits,bits.sum(axis=1)*config['swap_cost'],np.ones(len(bits),bool),FeatureEncoder.from_hyperedges(3,((0,1,2),)).family,config['diagnostic_scale'],policy.version)
        self.oracle_samples=int(config['oracle_repeats'])
        if self.oracle_samples<2:
            raise ValueError('At least two Monte Carlo oracle replicas required')
        values=np.array([[self.value(i,100000+r) for i in range(len(bits))] for r in range(self.oracle_samples)])
        self.truth=values.mean(axis=0)
        self.oracle_standard_error=values.std(axis=0,ddof=1)/np.sqrt(self.oracle_samples)
        self.metadata={'truth_kind':'monte_carlo','oracle_values':values.tolist(),'domain':env.domain,'continuation':'frozen','context_round':context_round,'policy_version':policy.version}

    def trace(self,index,stream):
        _,state=self.env.intervene(self.base,self.options[index])
        key=jax.random.fold_in(jax.random.PRNGKey(self.seed),stream)
        return simulate(self.env,self.policy,state,key,self.config['horizon'],self.config['gamma'])

    def value(self,index,stream):
        return float(self.trace(index,stream)['return'])

    def observe(self,index,step):
        return self.value(index,step+1)
