"""Same decision-focused CORA experiment interface on native JaxMARL contexts."""
import jax
import numpy as np
from cora.benchmarks.problems import PublicProblem
from cora.benchmarks.simulation import simulate
from cora.features import FeatureEncoder


class NativeTrial:
    def __init__(self,env,policy,config,seed,context_round=0):
        self.env,self.policy,self.config,self.seed=env,policy,config,seed
        _,self.base=env.reset(jax.random.PRNGKey(seed));self.options=env.candidates(self.base.organization)
        bits=np.asarray([z.bits for z in self.options],int);m=bits.shape[1]
        self.public=PublicProblem(f'{env.domain}_{policy.name}_round{context_round}',bits,bits.sum(1)*config['swap_cost'],np.ones(len(bits),bool),FeatureEncoder.from_hyperedges(m,(tuple(range(m)),)).family,config['diagnostic_scale'],policy.version)
        self.oracle_samples=int(config['oracle_repeats'])
        if self.oracle_samples<2:raise ValueError('At least two Monte Carlo reference replicas required')
        values=np.array([[self.value(i,100000+r) for i in range(len(bits))] for r in range(self.oracle_samples)])
        self.truth=values.mean(0);self.oracle_standard_error=values.std(0,ddof=1)/np.sqrt(self.oracle_samples)
        self.metadata={'truth_kind':'monte_carlo','oracle_values':values.tolist(),'continuation':'frozen','context_round':context_round,'policy_version':policy.version,**env.protocol_metadata()}

    def trace(self,index,stream):
        _,state=self.env.intervene(self.base,self.options[index]);key=jax.random.fold_in(jax.random.PRNGKey(self.seed),stream)
        return simulate(self.env,self.policy,state,key,self.config['horizon'],self.config['gamma'])

    def value(self,index,stream):return float(self.trace(index,stream)['return'])
    def observe(self,index,step):return self.value(index,step+1)
