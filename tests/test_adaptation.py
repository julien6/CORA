import pytest
from cora.adaptation import AdaptationBudget,adapt
from cora.benchmarks.domains import DomainConfig,OrganizationalDomain
from dataclasses import replace


def test_adaptive_round_boundaries():
    def train(i,steps):
        return None,f'v{i}',steps
    def trial(policy,version,i):
        d=OrganizationalDomain(DomainConfig('warehouse',horizon=4))
        return replace(d.public,policy_version=version),d.observe,f'x{i}'
    events=adapt(train,trial,lambda z,v:v+1,AdaptationBudget(2,1,4))
    assert len(events)==2
    assert sum(e['training_steps'] for e in events)==2
    assert events[0]['policy_version']!=events[1]['policy_version']
    assert all(not e['carried_effect_estimates'] and len(e['experiments'])==4 for e in events)
    with pytest.raises(ValueError):
        adapt(lambda i,n:(None,'same',n),trial,lambda z,v:v+1,AdaptationBudget(2,1,4))
