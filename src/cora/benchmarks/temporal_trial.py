"""Small known-SCM sequential benchmark for conditional bound validation.

Two local reservoirs have distinct dynamics, with optional shared influence.
Only interventions and sampled scalar outcomes are exposed to the learner.
"""
from dataclasses import dataclass
import numpy as np
from cora.temporal import TemporalGraph
from cora.reliability import BoundAssumptions, adaptive_bounds, conservative_decision
from cora.features import mobius


@dataclass(frozen=True)
class ReservoirSCM:
    horizon: int = 4
    gamma: float = .9
    coupling: bool = False
    noise: float = .05

    def __post_init__(self):
        if not isinstance(self.horizon, int) or self.horizon < 1 or not 0 <= self.gamma <= 1 or not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError('Invalid known-SCM configuration')

    def graph(self):
        parents = {'u0': (), 'u1': ()}
        returns = []
        for t in range(self.horizon):
            for i in range(2):
                state = f's{i}_{t}'
                parents[state] = (f'u{i}',) if t == 0 else (f's{i}_{t-1}', f'u{i}')
                if self.coupling:
                    parents[state] += (f'u{1-i}',)
                reward = f'r{i}_{t}'
                parents[reward] = (state,)
                returns.append(reward)
        return TemporalGraph(parents, ('u0','u1'), tuple(returns))

    def branch(self, z):
        z = np.asarray(z,float)
        state = np.zeros(2)
        rewards = []
        for t in range(self.horizon):
            inflow = z * (1 - .7 * z[::-1]) if self.coupling else z
            state = .5 * state + inflow
            rewards.append(self.gamma ** t * float(state.sum()))
        return sum(rewards)

    def run(self, seed=0, budget=12):
        bits = np.array([[mask & 1, (mask >> 1) & 1] for mask in range(4)])
        encoder = self.graph().encoder()
        features = np.asarray(encoder.encode(bits),float)
        # Public coefficient bound derived from |F| <= 2 * sum_t gamma^t * 2.
        # Mobius triangle bound |phi(T)| <= 2^|T| max|F|, without oracle values.
        value_bound = 4 * sum(self.gamma ** t for t in range(self.horizon))
        norm_bound = value_bound * np.sqrt(sum(4 ** len(term) for term in encoder.family))
        assumptions = BoundAssumptions(True,True,True,True,norm_bound,self.noise)
        rng = np.random.default_rng(seed)
        x,y,logs = [],[],[]
        for n in range(budget):
            arm = int(rng.integers(4))
            outcome = self.branch(bits[arm]) + rng.normal(0,self.noise)
            x.append(features[arm]); y.append(outcome)
            logs.append({'step':n,'intervention':bits[arm].tolist(),'outcome':outcome,'propensity':.25,'policy_version':'reservoir-fixed-v1','context':'empty-reservoirs'})
        prediction,radius = adaptive_bounds(x,y,features,assumptions,ridge=.001)
        selected = conservative_decision(prediction,radius,np.zeros(4),np.zeros(4),bits.sum(axis=1)*.1,np.ones(4,bool))
        truth = np.array([self.branch(z) for z in bits])  # evaluator only
        phi = mobius(truth)
        omitted = [mask for mask in range(4) if tuple(i for i in range(2) if mask & (1<<i)) not in encoder.family]
        assert all(abs(phi[mask]) < 1e-10 for mask in omitted)
        return {'seed':seed,'coupling':self.coupling,'family':encoder.family,'queries':logs,'prediction':prediction.tolist(),'radius':radius.tolist(),'selected':selected,'evaluation_only_values':truth.tolist(),'simultaneous_coverage':bool(np.all(abs(prediction-truth)<=radius)), 'conditional_certification':True,'guarantee':'expected value on confidence event, assuming exact SCM and stated noise bound'}
