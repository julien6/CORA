import numpy as np
import pytest
from cora.benchmarks.domains import DomainConfig,OrganizationalDomain
from cora.benchmarks.methods import run_method


@pytest.mark.parametrize('name',['warehouse','rescue','wildfire','hospital'])
def test_domain_branch_and_common_algorithm(name):
    domain = OrganizationalDomain(DomainConfig(name,horizon=6,disruption_step=3))
    a = domain.rollout([0,0],7)
    assert a == domain.rollout([0,0],7)
    assert np.isfinite(a['value']) and len(a['frames']) == 6
    assert a['metrics'][3]['failed_agents'] == 1
    assert all(f['metrics']['resource_remaining'] >= 0 for f in a['frames'])
    out,logs = run_method('cora',domain.public,domain.observe,[4],7,{'ridge':.01,'diagnostic_scale':1.})
    assert len(logs) == 4 and out[0]['selected_index'] in range(4)
    means,se = domain.oracle(repeats=2)
    assert np.isfinite(means).all() and np.isfinite(se).all()
    assert not hasattr(domain.public,'capabilities')
