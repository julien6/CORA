import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cora.benchmarks.domain_env import DomainEnv
from cora.benchmarks.domain_trial import DomainGuidedPolicy
from cora.symbolic import SymbolicOrganization,Edit,transaction


@pytest.mark.parametrize('domain',['warehouse','rescue','wildfire','hospital'])
def test_domain_interfaces_without_baseline_execution(domain):
    env=DomainEnv(domain,horizon=4)
    obs,state=env.reset(jax.random.PRNGKey(0))
    assert all(v.shape==env.observation_space(a).shape for a,v in obs.items())
    shape=jax.eval_shape(env.step,jax.random.PRNGKey(1),state,{a:jnp.array(0) for a in env.agents})
    assert shape[1].env_state.positions.shape==(3,2)
    assert shape[3]['__all__'].shape==()
    policy=DomainGuidedPolicy(tuple(env.agents))
    actions,_=policy.act(obs,policy.initial_hidden(),jnp.zeros(3,bool),jax.random.PRNGKey(1))
    assert all(0<=int(a)<5 for a in actions.values())
    batched=jax.eval_shape(jax.vmap(env.reset),jax.random.split(jax.random.PRNGKey(0),2))
    assert batched[1].env_state.queue.shape==(2,3)


def test_richer_atomic_norms():
    org=SymbolicOrganization(('a','b'),('r',),('m0','m1'),(0,1),(0,0),frozenset({(0,0),(0,1)}),obligations=frozenset({(0,0)}))
    org.validate()
    swapped=transaction(org,[Edit('assign',0,1),Edit('assign',1,0),Edit('link',0,1)],0)
    assert swapped.assignments==(1,0) and swapped.version==1
    with pytest.raises(ValueError):transaction(org,[Edit('prohibit',0,0)],0)
    with pytest.raises(ValueError):transaction(swapped,[],0)
    assert org.assignments==(0,1)


@pytest.mark.parametrize('domain',['warehouse','rescue','wildfire','hospital'])
@pytest.mark.parametrize('learner',['ippo','mappo','iql','vdn','qmix'])
def test_learner_abstract_compilation_only(learner,domain):
    from cora.benchmarks.learners import prepare_training
    env=DomainEnv(domain,horizon=4)
    trainer,config,_,records=prepare_training(learner,env,{'steps':4,'environments':1,'updates':1,'epochs':1,'minibatches':1,'hidden_size':8})
    shape=jax.eval_shape(trainer,jax.random.PRNGKey(0))
    assert 'runner_state' in shape
    assert records==[]  # No callback or numerical optimizer update executed.


def test_release_preflight_never_executes(tmp_path,monkeypatch):
    from omegaconf import OmegaConf
    from cora.benchmarks import release
    config=OmegaConf.to_container(OmegaConf.load('configs/applications.yaml'))
    monkeypatch.setattr(release,'train',lambda *a:pytest.fail('Training forbidden in preflight'))
    monkeypatch.setattr(release,'NativeTrial',lambda *a:pytest.fail('Evaluation forbidden in preflight'))
    config['output']=str(tmp_path/'uncreated')
    result=release.preflight(config)
    assert not result['experiments_executed']
    assert not (tmp_path/'uncreated').exists()


def test_org_mdp_known_solution():
    from cora.benchmarks.planning import value_iteration
    # Solver arithmetic unit check, no simulator or training.
    p=np.array([[[1,0],[0,1]],[[0,1],[0,1]]],float)
    r=np.array([[0,1],[2,0]],float)
    result=value_iteration(p,r,np.zeros((2,2)),np.ones((2,2),bool),gamma=.5)
    np.testing.assert_allclose(result['values'],[3,4],atol=1e-6)
    assert result['policy'].tolist()==[1,0]


def test_domain_trial_metadata_with_mocked_simulation(monkeypatch):
    import cora.benchmarks.domain_trial as module
    from cora.benchmarks.learners import FrozenPolicy
    monkeypatch.setattr(module,'simulate',lambda *args,**kwargs:{'return':jnp.array(1.)})
    env=DomainEnv('rescue',horizon=4)
    trial=module.DomainTrial(env,FrozenPolicy('random',tuple(env.agents)),{'swap_cost':.1,'diagnostic_scale':1.,'oracle_repeats':2,'horizon':4,'gamma':.9},0)
    assert trial.public.bits.shape==(8,3)
    assert not hasattr(trial.public,'truth')
    assert trial.metadata['continuation']=='frozen'


def test_saved_data_analysis_without_running_baselines(tmp_path):
    import json
    from omegaconf import OmegaConf
    from cora.benchmarks.release import postprocess
    config=OmegaConf.to_container(OmegaConf.load('configs/applications.yaml'))
    (tmp_path/'manifest.json').write_text(json.dumps({'config':config}))
    # Empty raw dataset checks the no-experiment analysis path.
    assert postprocess(tmp_path)['result_rows']==0
    assert (tmp_path/'index.html').exists()


def test_single_transition_precedence_and_resource_invariants():
    # Simulator transition unit checks, not policy/baseline rollouts.
    env=DomainEnv('rescue',horizon=4)
    _,state=env.reset(jax.random.PRNGKey(0))
    state=state.replace(env_state=state.env_state.replace(positions=env.sites))
    _,next_state,_,_,_=env.step(jax.random.PRNGKey(1),state,{a:jnp.array(0) for a in env.agents})
    np.testing.assert_array_equal(next_state.env_state.queue,[2,1,0])
    np.testing.assert_array_equal(next_state.env_state.completed,[1,0,0])
    wildfire=DomainEnv('wildfire',horizon=4,capacity=3)
    _,state=wildfire.reset(jax.random.PRNGKey(0))
    state=state.replace(env_state=state.env_state.replace(positions=wildfire.sites,resource=jnp.array(1.)))
    _,next_state,_,_,_=wildfire.step(jax.random.PRNGKey(1),state,{a:jnp.array(0) for a in wildfire.agents})
    assert float(next_state.env_state.resource)>=0
    assert float(next_state.env_state.completed.sum())<=1
