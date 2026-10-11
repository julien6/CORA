import numpy as np
import pytest
from cora.reliability import BoundAssumptions, adaptive_bounds, fixed_design_bounds, conservative_decision, policy_shift_bound
from cora.temporal import TemporalGraph, PolicyVersionedData


def test_ancestry_includes_indirect_resource_coupling():
    graph = TemporalGraph({'u0': (), 'u1': (), 'resource': ('u0','u1'), 'action': ('resource',), 'return': ('action',)}, ('u0','u1'), ('return',))
    assert (0,1) in graph.encoder().family
    with pytest.raises(ValueError):
        TemporalGraph({'u':(), 'a':('b',), 'b':('a',)}, ('u',), ('a',)).encoder()


def test_policy_change_rejects_stale_samples():
    data = PolicyVersionedData('v1','x',[])
    data.append('v1','x',1)
    with pytest.raises(ValueError):
        data.append('v2','x',2)


def test_fixed_bounds_and_rank():
    x = np.array([[1,0],[1,1],[1,0],[1,1]],float)
    theta = np.array([2.,3.])
    p,b = fixed_design_bounds(x,x@theta,x,0)
    np.testing.assert_allclose(p,x@theta)
    assert np.all(b == 0)
    with pytest.raises(ValueError):
        fixed_design_bounds(x[:1],np.array([2]),x,1)


def test_adaptive_coverage_predictable_design():
    # Outcome-dependent designs; coverage event covers all steps and targets.
    targets = np.array([[1,0],[1,1]],float)
    theta = np.array([.2,.3])
    assumptions = BoundAssumptions(True,True,True,True,np.linalg.norm(theta),.1)
    rng = np.random.default_rng(9)
    covered = 0
    for _ in range(100):
        xs,ys,ok = [],[],True
        for t in range(12):
            index = int(t > 0 and ys[-1] > .2)
            xs.append(targets[index]); ys.append(targets[index]@theta+rng.normal(0,.1))
            p,b = adaptive_bounds(xs,ys,targets,assumptions,delta=.05)
            ok &= bool(np.all(np.abs(p-targets@theta) <= b+1e-12))
        covered += ok
    assert covered >= 95


def test_conditional_decision_and_shift():
    assert conservative_decision([0,3],[.1,.1],[0,0],[0,0],[0,.1],[True,True]) == 1
    assert conservative_decision([0,3],[.1,.1],[0,4],[0,0],[0,.1],[True,True]) == 0
    assert policy_shift_bound(0,1,10) == 0
    assert policy_shift_bound(1,1,3) == 6
    with pytest.raises(ValueError):
        adaptive_bounds([[1]],[0],[[1]],BoundAssumptions(False,True,True,True,1,1))


def test_executed_harm_denominator():
    from cora.benchmarks.statistics import aggregate
    rows=[{'benchmark':'x','method':'cora','budget':4,'seed':i,'simple_regret':0,'abstention':int(i<2),'harmful_reference':int(i==2)} for i in range(3)]
    result=aggregate(rows,{'bootstrap_samples':10,'confidence':.95})
    harm=result['summaries'][0]['metrics']['harm_rate_among_executed_nonnoop']
    assert harm['executed']==1 and harm['rate']==1


def test_conditional_runner_requires_hypotheses_and_known_bounds():
    from cora.benchmarks.methods import run_method
    from cora.benchmarks.problems import PublicProblem
    bits=np.array([[0,0],[0,1],[1,0],[1,1]])
    public=PublicProblem('known',bits,np.zeros(4),np.ones(4,bool),((),(0,),(1,),(0,1)),0,'v1')
    config={'ridge':.0001,'diagnostic_scale':1.,'confidence_assumptions':{'exact_features':True,'fixed_context':True,'fixed_policy':True,'martingale_subgaussian':True,'coefficient_norm_bound':2.,'noise_bound':0.},'structural_error_bounds':[0]*4,'policy_shift_error_bounds':[0]*4}
    decisions,logs=run_method('cora_conditional',public,lambda index,step:float(bits[index].sum()),[8],0,config)
    assert decisions[0]['certified']
    assert len(logs)==8
    config['confidence_assumptions']['fixed_policy']=False
    with pytest.raises(ValueError):
        run_method('cora_conditional',public,lambda i,s:0.,[8],0,config)
