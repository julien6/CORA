import json
import numpy as np
import pytest
from cora.benchmarks.fixed_campaign import FixedMissionTrial,run
from cora.benchmarks.methods import run_method
from moise_diagram import parse

def test_causal_locality_and_effective_mission_swaps():
    t=FixedMissionTrial(horizon=12,repeats=2)
    values=np.array([t.rollout(bits,20)[0] for bits in t.bits])
    for a in (0,1):
        for b in (2,3):
            z=np.zeros(4,int);base=t.rollout(z,20)[0]
            z[a]=1;v_a=t.rollout(z,20)[0];z[a]=0;z[b]=1;v_b=t.rollout(z,20)[0]
            z[a]=1;v_ab=t.rollout(z,20)[0]
            assert v_ab-v_a-v_b+base==0
    _,before=t.rollout(t.bits[0],20);_,after=t.rollout(t.bits[-1],20)
    assert before[0]['assignments']!=after[0]['assignments']
    assert np.ptp(values)>0
    assert len(parse(t.source(15)).commits)==8

def test_g_design_budget_constraints_and_independent_observations():
    t=FixedMissionTrial(horizon=6,repeats=2);calls=[]
    def observe(i,s):calls.append((i,s));return t.observe(i,s)
    decisions,logs=run_method('linear_g_optimal',t.public,observe,[2,4],0,{'ridge':.01,'diagnostic_scale':1.})
    assert len(calls)==len(logs)==4
    assert all(d['selected_index'] in range(16) and not d['certified'] for d in decisions)

def test_end_to_end(tmp_path):
    run(tmp_path,[0],[2],4,2)
    assert (tmp_path/'results.csv').exists()
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['training_steps']==0
    assert len(list(tmp_path.glob('*/seed_0/*/replay.json')))==21
    with pytest.raises(ValueError,match='empty'):run(tmp_path,[0],[2],4,2)
