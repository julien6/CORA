"""Native simulator/adapter and abstract learner checks; no optimization runs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cora.benchmarks.native_env import NativeOrganizationEnv,NativeRandomPolicy,NativeIntervention,NATIVE_ENVIRONMENTS


@pytest.mark.parametrize('domain',NATIVE_ENVIRONMENTS)
def test_native_transition_and_atomic_delegation(domain,tmp_path):
    from cora.benchmarks.simulation import save_trace
    from cora.benchmarks.native_media import load_states,render_trace
    env=NativeOrganizationEnv(domain,horizon=3)
    obs,state=env.reset(jax.random.PRNGKey(3))
    assert all(obs[a].shape==env.observation_space(a).shape for a in env.agents)
    options=env.candidates(state.organization)
    assert len(options)==2**len(env.swaps)
    changed=env.intervene(state,NativeIntervention((1,)+(0,)*(len(env.swaps)-1)))[1]
    assert int(changed.organization.version)==1
    assert sorted(changed.organization.assignments.tolist())==list(range(env.num_agents))
    assert int(state.organization.version)==0
    env.initial=changed.organization
    _,reset_changed=env.reset(jax.random.PRNGKey(3))
    np.testing.assert_array_equal(reset_changed.organization.assignments,changed.organization.assignments)
    env.initial=state.organization
    if domain=='jaxnav':np.testing.assert_array_equal(changed.env_state.goal,state.mission_goals[changed.organization.assignments])
    policy=NativeRandomPolicy(env.agents,tuple(env.action_space(a).n for a in env.agents))
    actions,_=policy.act(obs,policy.initial_hidden(),jnp.zeros(env.num_agents,bool),jax.random.PRNGKey(4))
    key=jax.random.PRNGKey(5)
    _,native,rewards,dones,_=env._env.step_env(key,state.env_state,env.native_actions(state,actions))
    _,wrapped,wr,wd,_=env.evaluation_step(key,state,actions)
    for x,y in zip(jax.tree.leaves(native),jax.tree.leaves(wrapped.env_state)):np.testing.assert_allclose(x,y)
    assert all(float(wr[a])==float(rewards[a]) for a in env.agents)
    assert bool(wd['__all__'])==bool(dones['__all__'])
    # Lossless nested state storage and actual upstream snapshot rendering.
    states=jax.tree.map(lambda x:jnp.stack([x,x]),state.env_state)
    save_trace(tmp_path/'trace.npz',{'states':states,'rewards':jnp.zeros(2),'actions':jnp.zeros((2,env.num_agents))},state.organization.assignments)
    restored,_,_=load_states(tmp_path/'trace.npz',env)
    for x,y in zip(jax.tree.leaves(restored[0]),jax.tree.leaves(state.env_state)):np.testing.assert_array_equal(x,y)
    render_trace(tmp_path/'trace.npz',tmp_path/'media',env,'Interface test',8)
    assert (tmp_path/'media/replay.gif').exists()
    jax.clear_caches()


@pytest.mark.parametrize('domain',NATIVE_ENVIRONMENTS)
@pytest.mark.parametrize('learner',['ippo','mappo','iql','vdn','qmix'])
def test_native_learner_shape_without_training(domain,learner):
    from cora.benchmarks.learners import prepare_training
    env=NativeOrganizationEnv(domain,horizon=3)
    trainer,_,_,records=prepare_training(learner,env,{'steps':2,'environments':1,'updates':1,'epochs':1,'minibatches':1,'hidden_size':8})
    result=jax.eval_shape(trainer,jax.random.PRNGKey(0))
    assert 'runner_state' in result and records==[]
    jax.clear_caches()


def test_native_trial_public_boundary(monkeypatch):
    import cora.benchmarks.native_trial as module
    monkeypatch.setattr(module,'simulate',lambda *args:{'return':jnp.array(1.)})
    env=NativeOrganizationEnv('overcooked_v2',horizon=3)
    policy=NativeRandomPolicy(env.agents,(6,6))
    trial=module.NativeTrial(env,policy,{'swap_cost':.1,'diagnostic_scale':1.,'oracle_repeats':2,'horizon':2,'gamma':.99},0)
    assert trial.public.bits.shape==(2,1)
    assert not hasattr(trial.public,'truth')
    assert trial.metadata['backend']=='native_jaxmarl'


def test_native_default_preflight_forbids_custom_domains(monkeypatch):
    from cora.benchmarks import release
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load('configs/applications.yaml'))
    monkeypatch.setattr(release,'train',lambda *args:pytest.fail('Training forbidden'))
    monkeypatch.setattr(release,'NativeTrial',lambda *args:pytest.fail('Experiment forbidden'))
    result=release.preflight(cfg)
    assert not result['experiments_executed'] and result['backend']=='native_jaxmarl'
    cfg['domains']=['warehouse']
    with pytest.raises(ValueError):release.preflight(cfg)


def test_frozen_q_actor_respects_native_smax_mask():
    from cora.benchmarks.learners import FrozenPolicy
    env=NativeOrganizationEnv('smax',horizon=3)
    obs,_=env.reset(jax.random.PRNGKey(0));count=env.action_space(env.agents[0]).n
    class Scores:
        def apply(self,params,hidden,observations,dones):
            return hidden,jnp.broadcast_to(jnp.arange(count),(1,env.num_agents,count))
    policy=FrozenPolicy('iql',tuple(env.agents),network=Scores())
    actions,_=policy.act(obs,policy.initial_hidden(),jnp.zeros(env.num_agents,bool),jax.random.PRNGKey(1))
    for a in env.agents:
        assert bool(obs['action_masks'][a][actions[a]])
        assert int(actions[a])==int(jnp.argmax(jnp.where(obs['action_masks'][a],jnp.arange(count),-jnp.inf)))
