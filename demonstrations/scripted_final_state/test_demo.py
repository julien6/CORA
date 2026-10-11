"""Tests stay in the isolated demonstration folder; no training is invoked."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('cora_scripted_demo',HERE/'run.py')
demo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=demo;spec.loader.exec_module(demo)


def test_choices_and_privileged_reference():
    bits=np.array([[0,0],[0,1],[1,0],[1,1]])
    values=np.array([0,1,1,10.])
    costs=np.zeros(4);valid=np.ones(4,bool)
    assert demo.select('status_quo',bits,values,costs,valid,7)==0
    assert demo.select('singleton_reference',bits,values,costs,valid,7) in (1,2)
    assert demo.select('compound_reference',bits,values,costs,valid,7)==3
    valid[3]=False
    assert demo.select('compound_reference',bits,values,costs,valid,7)!=3


def test_observation_only_policy():
    import jax
    import jax.numpy as jnp
    from cora.benchmarks.domain_env import DomainEnv
    env=DomainEnv('warehouse',horizon=4)
    observations,_=env.reset(jax.random.PRNGKey(0))
    policy=demo.DomainGuidedPolicy(tuple(env.agents))
    actions,_=policy.act(observations,policy.initial_hidden(),jnp.zeros(3,bool),jax.random.PRNGKey(1))
    assert all(0<=int(a)<5 for a in actions.values())


def test_mock_names_are_not_reported_as_executed_methods():
    import json
    catalog=json.loads((HERE/'mock_catalog.json').read_text())
    assert set(catalog['learners'])=={'IPPO','MAPPO','IQL','VDN','QMIX'}
    assert all(not x['learned_checkpoint'] for x in catalog['learners'].values())
    assert all(not x['method_executed'] and not x['faithful_reproduction'] for x in catalog['organizational_methods'].values())


def test_contextual_renderer_preserves_recorded_states():
    from types import SimpleNamespace
    from scene_renderer import application_frames
    from cora.benchmarks.domain_env import DomainEnv
    path=HERE/'results/warehouse/seed_7/status_quo_trace.npz'
    with np.load(path) as arrays:
        names=('positions','queue','completed','failed','resource','step')
        states=SimpleNamespace(**{k:arrays[k][:2].copy() for k in names})
        before={k:getattr(states,k).copy() for k in names}
        frames=application_frames({'states':states},DomainEnv('warehouse'),arrays['assignments'],'test')
        assert len(frames)==2 and frames[0].size==(1000,650)
        assert all(np.array_equal(getattr(states,k),before[k]) for k in names)
        assert frames[0].tobytes()!=frames[1].tobytes()


def test_native_world_communication_action_contract():
    import jax
    import jaxmarl
    from library_demo import world_actions
    env=jaxmarl.make('MPE_simple_world_comm_v3')
    obs,_=env.reset(jax.random.PRNGKey(7))
    silent=world_actions(env,obs,False);guided=world_actions(env,obs,True)
    for a in env.agents:
        assert guided[a].shape==env.action_space(a).shape
        assert np.all(np.asarray(guided[a])>=0) and np.all(np.asarray(guided[a])<=1)
        assert np.array_equal(silent[a][:5],guided[a][:5])
    assert np.array_equal(np.asarray(silent[env.leader][5:]),np.zeros(4))


def test_no_spread_in_current_gallery():
    import json
    assert all(r['benchmark']!='simple_spread' for r in json.loads((HERE/'results/results.json').read_text()))


def test_saved_native_rollouts_and_gifs():
    import json
    from library_demo import SCENARIOS,DEST,verify
    verify(required_all=True)
    for name in SCENARIOS:
        metadata=json.loads((DEST/name/'guided/metadata.json').read_text())
        with np.load(DEST/name/'guided/trace.npz') as arrays:
            assert len(metadata['state_leaf_paths'])==len([k for k in arrays.files if k.startswith('state_')])
            assert set(metadata['action_sizes'])
            # Native episodes freeze after termination rather than silently resetting.
            if np.any(arrays['dones']):
                first=int(np.flatnonzero(arrays['dones'])[0])
                assert np.all(arrays['rewards'][first+1:]==0)
                for k in arrays.files:
                    if k.startswith('state_') and first+2<len(arrays[k]):
                        assert np.all(arrays[k][first+2:]==arrays[k][first+1])
