"""Resource-limited sequential application prototypes, not real-world models.

Warehouse has pickup/inspection/loading precedence. Rescue has detection then
 evacuation. Wildfire spreads stochastic hazards and consumes water. Hospital
 has triage/treatment precedence and priority queues. A shared public protocol
 uses the same CORA decision mechanism for all four.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from cora.benchmarks.problems import PublicProblem
from cora.temporal import TemporalGraph


@dataclass(frozen=True)
class DomainConfig:
    name: str
    horizon: int = 12
    gamma: float = .95
    disruption_step: int = 5
    resources: int = 2
    observation_radius: int = 2

    def __post_init__(self):
        if self.name not in ('warehouse','rescue','wildfire','hospital') or self.horizon < 2 or self.resources < 1 or not 0 <= self.gamma <= 1:
            raise ValueError('Invalid domain configuration')


class OrganizationalDomain:
    swaps = ((0,1),(1,2))

    def __init__(self, config):
        self.config = config
        self.capabilities = np.array([[1.,.6,.4],[.5,1.,.7],[.7,.5,1.]])
        self.bits = np.array(list(product((0,1),repeat=2)))
        self.public = PublicProblem(config.name,self.bits,self.bits.sum(axis=1)*.1,np.ones(4,bool),self.temporal_graph().encoder().family,1.,'local-script-v1')

    def temporal_graph(self):
        """Conservative time-unrolled support graph for the explicit simulator.

        State groups positions, task queues, failures, and remaining resource.
        Shared contention/collision mechanisms prevent assumed independence.
        Public graphs specify support, not noise realizations or oracle values.
        """
        parents={'u0':(), 'u1':(), 'organization':('u0','u1'), 'state_0':(), 'policy':()}
        rewards=[]
        for t in range(self.config.horizon):
            parents[f'noise_{t}']=()
            actions=[]
            for i in range(3):
                obs=f'observation_{i}_{t}'; action=f'action_{i}_{t}'
                parents[obs]=(f'state_{t}','organization')
                parents[action]=(obs,'policy')
                actions.append(action)
            parents[f'state_{t+1}']=(f'state_{t}',*actions,f'noise_{t}','organization')
            reward=f'reward_{t}'
            parents[reward]=(f'state_{t+1}',)
            rewards.append(reward)
        return TemporalGraph(parents,('u0','u1'),tuple(rewards))

    def assignments(self,z):
        if np.shape(z) != (2,) or not np.all((np.asarray(z)==0)|(np.asarray(z)==1)):
            raise ValueError('Binary transaction required')
        a = np.arange(3)
        for active,(i,j) in zip(z,self.swaps):
            if active:
                a[i],a[j] = a[j],a[i]
        assert sorted(a) == [0,1,2]
        return a

    def rollout(self,z,seed):
        c = self.config
        rng = np.random.default_rng(seed)
        assignment = self.assignments(z)
        positions = np.array([[0,0],[0,2],[2,0]])
        sites = np.array([[1,1],[3,1],[3,3]])
        queue = np.array([3.,0.,0.]) if c.name != 'wildfire' else np.array([2.,2.,2.])
        completed = np.zeros(3)
        water = 12.
        frames,rewards,metrics = [],[],[]
        failed = np.zeros(3,bool)
        for t in range(c.horizon):
            if t == c.disruption_step:
                failed[0] = True
                queue[0] += 2
            # Decentralized action sees own assignment, own position and nearby site only.
            requests = []
            for i in range(3):
                mission = assignment[i]
                distance = np.abs(sites[mission]-positions[i]).sum()
                if failed[i]:
                    continue
                if distance > 0:
                    delta = sites[mission]-positions[i]
                    axis = int(np.argmax(abs(delta)))
                    proposal = positions[i].copy(); proposal[axis] += int(np.sign(delta[axis]))
                    if not any(np.array_equal(proposal,p) for j,p in enumerate(positions) if j != i):
                        positions[i] = proposal
                if distance <= c.observation_radius and queue[mission] > 0:
                    requests.append((i,mission))
            rng.shuffle(requests)
            processed = 0.
            priority_violations = 0
            for i,mission in requests[:c.resources]:
                if c.name == 'hospital' and mission == 2 and queue[1] > 0:
                    priority_violations += 1
                    continue
                if c.name == 'wildfire' and water <= 0:
                    continue
                if rng.random() < self.capabilities[i,mission] and queue[mission] > 0:
                    queue[mission] -= 1; completed[mission] += 1; processed += 1
                    if c.name == 'wildfire':
                        water -= 1
                    elif mission < 2:
                        queue[mission+1] += 1
            if c.name == 'wildfire':
                queue += rng.binomial(1,np.minimum(.7,.1+.1*queue))
            elif c.name == 'hospital':
                queue[0] += rng.binomial(1,.4)
            elif c.name == 'warehouse':
                queue[0] += rng.binomial(1,.15)
            # Rescue has finite victims and must follow detection/evacuation chain.
            reward = processed - .05*queue.sum() - priority_violations
            rewards.append(reward)
            metrics.append({'backlog':float(queue.sum()),'throughput':float(completed[-1]),'resource_remaining':water,'priority_violations':priority_violations,'failed_agents':int(failed.sum())})
            frames.append({'positions':positions.copy().tolist(),'sites':sites.tolist(),'assignment':assignment.tolist(),'queue':queue.copy().tolist(),'metrics':metrics[-1]})
        return {'value':float(np.dot(c.gamma**np.arange(c.horizon),rewards)),'rewards':rewards,'frames':frames,'metrics':metrics,'seed':seed,'policy_version':self.public.policy_version,'config':c.__dict__}

    def observe(self,index,step,seed=0):
        return self.rollout(self.bits[index],int(np.random.SeedSequence([seed,step,123]).generate_state(1)[0]))['value']

    def oracle(self,seed=0,repeats=8):
        if repeats < 2:
            raise ValueError('At least two oracle replicas required for standard error')
        # Independent held-out evaluation streams, never passed to run_method.
        outcomes = np.array([[self.rollout(z,int(np.random.SeedSequence([seed,r,999]).generate_state(1)[0]))['value'] for z in self.bits] for r in range(repeats)])
        return outcomes.mean(axis=0), outcomes.std(axis=0,ddof=1)/np.sqrt(repeats)
