"""Scripted online adoption replay; policies remain handwritten and frozen."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import jax
import jax.numpy as jnp
from run import HERE,CONFIG,LABEL,DomainEnv,DomainGuidedPolicy,application_frames,save_frames,comparison,write
from cora.interventions import Intervention
from PIL import Image


def run():
    rows=[]
    for name in ('warehouse','rescue','wildfire','hospital'):
        env=DomainEnv(name,horizon=CONFIG['horizon'],disruption_step=CONFIG['disruption_step'])
        policy=DomainGuidedPolicy(tuple(env.agents))
        for seed in CONFIG['seeds']:
            for profile in ('status_quo','scripted_reconfigure'):
                _,state=env.reset(jax.random.PRNGKey(seed))
                states,rewards,actions,assignments,versions,commits=[],[],[],[],[],[]
                for t in range(CONFIG['horizon']):
                    # Predeclared response after the failure transition; no oracle.
                    if t==CONFIG['disruption_step']+1 and profile=='scripted_reconfigure':
                        _,state=env.intervene(state,Intervention((1,0,0)))
                        commits.append({'step':t,'bits':[1,0,0],'version':int(state.organization.version),'cost':CONFIG['swap_cost'],'selection':'hardcoded, not learned or oracle-selected'})
                    states.append(state.env_state);assignments.append(np.asarray(state.organization.assignments));versions.append(int(state.organization.version))
                    key=jax.random.fold_in(jax.random.PRNGKey(seed),900000+t)
                    action_key,transition_key=jax.random.split(key)
                    action,_=policy.act(env.get_obs(state),policy.initial_hidden(),jnp.zeros(3,bool),action_key)
                    _,state,reward,_,_=env.step(transition_key,state,action)
                    actions.append([int(action[a]) for a in env.agents]);rewards.append(float(reward[env.agents[0]]))
                stacked=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
                trace={'states':stacked,'rewards':np.array(rewards)}
                dest=HERE/'adaptive/results'/name/f'seed_{seed}'
                dest.mkdir(parents=True,exist_ok=True)
                arrays={field:np.asarray(getattr(stacked,field)) for field in stacked.__dataclass_fields__}
                np.savez_compressed(dest/f'{profile}_trace.npz',**arrays,rewards=rewards,actions=actions,assignments=assignments,organization_versions=versions)
                value=float(np.dot(CONFIG['gamma']**np.arange(CONFIG['horizon']),rewards))
                cost=sum(c['cost']*CONFIG['gamma']**c['step'] for c in commits)
                row={'domain':name,'seed':seed,'profile':profile,'discounted_native_return':value,'discounted_commit_cost':cost,'net_return':value-cost,'commits':commits,'policy_version':policy.version,'policy_updates':0,'training_steps':0,'label':LABEL,'continuation':'same frozen controller before and after live adoption'}
                rows.append(row);write(dest/f'{profile}.json',row)
                if seed==CONFIG['render_seed']:
                    save_frames(application_frames(trace,env,np.array(assignments),'LIVE '+profile),HERE/'adaptive/visualizations'/name/profile,CONFIG['fps'])
            if seed==CONFIG['render_seed']:
                comparison(HERE/'adaptive/visualizations'/name/'status_quo/replay.gif',HERE/'adaptive/visualizations'/name/'scripted_reconfigure/replay.gif',HERE/'adaptive/visualizations'/name/'comparison')
        jax.clear_caches()
    write(HERE/'adaptive/results.json',rows)
    gifs=list((HERE/'adaptive/visualizations').rglob('replay.gif'))
    assert len(gifs)==12 and len(rows)==24
    for path in gifs:
        with Image.open(path) as im:assert im.is_animated and im.n_frames==CONFIG['horizon']
    write(HERE/'adaptive/verification.json',{'rows':24,'gifs':12,'training_steps':0,'policy_updates':0,'adoption':'one predeclared swap after failure; no learned adaptive policy','label':LABEL})
    cards=''.join(f'<h2>{n}</h2><img src="visualizations/{n}/comparison/replay.gif" width="100%">' for n in ('warehouse','rescue','wildfire','hospital'))
    (HERE/'adaptive/index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Scripted live adaptation</title><style>body{max-width:1400px;margin:30px auto;font:16px system-ui;background:#eef2f6;color:#172b40}img{max-width:100%;background:white}h1{font-size:22px}p{padding:18px;background:#fff0c2}</style><h1>'+LABEL+'</h1><p>Left: status quo. Right: a predeclared swap after failure. The same handwritten policy is used throughout. No training, optimality, improvement or safety claim.</p>'+cards+'</html>')
    p=HERE/'index.html';s=p.read_text();link='<section><h2>Live organizational adoption</h2><p><a href="adaptive/index.html">Four matched failure/reconfiguration comparisons</a>. One predefined swap, same policy, no training.</p></section>'
    if 'adaptive/index.html' not in s:p.write_text(s.replace('<h2>All analytical scenarios</h2>',link+'<h2>All analytical scenarios</h2>'))
    print(json.dumps({'rows':24,'gifs':12,'training_steps':0}))


if __name__=='__main__':run()
