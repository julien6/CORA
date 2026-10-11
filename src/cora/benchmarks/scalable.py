"""Sampled candidate backend for 4–64 edits; never enumerates 2**m arms.

Decision optimum is relative to the returned pool. Global regret/optimization
 gap is unknown unless a separate oracle supplies a rigorous bound.
"""
from itertools import combinations
import numpy as np
from cora.benchmarks.problems import PublicProblem
from cora.features import FeatureEncoder


def candidate_pool(edits, seed=0, size=128, forbidden_pairs=()):
    if not 4 <= edits <= 64 or size < edits + 1:
        raise ValueError('4–64 edits and room for all singleton candidates required')
    pairs=tuple(tuple(p) for p in forbidden_pairs)
    if any(len(p)!=2 or p[0]==p[1] or min(p)<0 or max(p)>=edits for p in pairs):
        raise ValueError('Invalid constraint pair')
    rng=np.random.default_rng(seed)
    pool={tuple(np.zeros(edits,dtype=int))}
    pool.update(tuple(np.eye(edits,dtype=int)[i]) for i in range(edits))
    attempts=0
    while len(pool)<size and attempts<size*30:
        attempts+=1
        z=rng.binomial(1,rng.uniform(.02,.5),edits)
        if any(z[i] and z[j] for i,j in pairs):
            continue
        pool.add(tuple(z))
    bits=np.array(sorted(pool),dtype=int)
    return bits,{'requested_candidates':size,'actual_candidates':len(bits),'proposals':attempts,'global_optimization_gap':None,'reference':'candidate_pool_only'}


class ScalableSynthetic:
    def __init__(self, edits=64, agents=64, order=2, density=.2, seed=0, candidates=128, noise=.1):
        if agents not in (4,8,16,32,64) or not 1<=order<=4 or not 0<=density<=1 or noise<0:
            raise ValueError('Invalid synthetic dimensions')
        self.agents=agents
        rng=np.random.default_rng(seed)
        # Local clusters bound feature count; density controls retained terms.
        edges=tuple(tuple(range(s,min(s+order,edits))) for s in range(0,edits,order))
        encoder=FeatureEncoder.from_hyperedges(edits,edges)
        self._theta=rng.normal(size=len(encoder.family))
        for i,term in enumerate(encoder.family):
            if len(term)>1 and rng.random()>density:
                self._theta[i]=0
        bits,self.diagnostics=candidate_pool(edits,seed,candidates,((0,1),))
        self.public=PublicProblem('scalable_synthetic',bits,bits.sum(axis=1)*.1,np.ones(len(bits),bool),encoder.family,noise,'synthetic-fixed-v1')
        self._features=np.asarray(encoder.encode(bits),float)
        self._seed=seed

    def observe(self,index,step):
        rng=np.random.default_rng(np.random.SeedSequence([self._seed,step,123]))
        return float(self._features[index]@self._theta+rng.normal(0,self.public.noise_scale))

    def evaluator_values(self):
        return self._features@self._theta
