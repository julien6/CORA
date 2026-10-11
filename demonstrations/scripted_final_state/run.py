"""Isolated no-training demonstration. Writes exclusively beneath this folder."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import hashlib
import html
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from cora.benchmarks.domain_env import DomainEnv
from cora.benchmarks.domain_trial import DomainTrial,DomainGuidedPolicy
from cora.benchmarks.problems import SyntheticTrial
from cora.benchmarks.scalable import ScalableSynthetic
from cora.benchmarks.temporal_trial import ReservoirSCM
from cora.benchmarks.simulation import save_trace
from cora.interventions import apply

LABEL='SCRIPTED DEMO — NO TRAINING — NOT LEARNED BASELINE RESULTS'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')



def select(profile,bits,values,costs,valid,seed):
    allowed=np.flatnonzero(valid)
    noop=int(np.flatnonzero(np.all(bits==0,axis=1))[0])
    net=values-costs
    if profile=='status_quo':return noop
    if profile=='random_admissible':return int(np.random.default_rng(seed).choice(allowed))
    if profile=='fixed_swap':
        choices=allowed[bits[allowed].sum(axis=1)==1]
        return int(choices[0]) if len(choices) else noop
    if profile=='singleton_reference':
        allowed=allowed[bits[allowed].sum(axis=1)<=1]
    return int(allowed[np.argmax(net[allowed])])


from scene_renderer import application_frames


def save_frames(frames,dest,fps):
    dest.mkdir(parents=True,exist_ok=True)
    frames[0].save(dest/'replay.gif',save_all=True,append_images=frames[1:],duration=1000//fps,loop=0)
    for t in sorted({0,len(frames)//2,len(frames)-1}):frames[t].save(dest/f'frame_{t:03d}.png')


def comparison(left,right,dest):
    frames=[]
    with Image.open(left) as a,Image.open(right) as b:
        if a.n_frames!=b.n_frames:raise ValueError('Unmatched replay lengths')
        for t in range(a.n_frames):
            a.seek(t);b.seek(t)
            frame=Image.new('RGB',(a.width+b.width,max(a.height,b.height)),'white')
            frame.paste(a.convert('RGB'),(0,0));frame.paste(b.convert('RGB'),(a.width,0));frames.append(frame)
    save_frames(frames,dest,CONFIG['fps'])


def record_problem(name,seed,bits,values,costs,valid,truth_kind,se,rows):
    folder=HERE/'results'/name/f'seed_{seed}'
    write(folder/'evaluation_reference.json',{'bits':bits.tolist(),'values':values.tolist(),'costs':costs.tolist(),'valid':valid.tolist(),'standard_errors':se.tolist(),'reference_kind':truth_kind,'privileged':True,'label':LABEL})
    noop=int(np.flatnonzero(np.all(bits==0,axis=1))[0]);net=values-costs
    selected={}
    for profile in CONFIG['profiles']:
        index=select(profile,bits,values,costs,valid,seed)
        selected[profile]=index
        rows.append({'benchmark':name,'seed':seed,'profile':profile,'selected_index':index,'selected_bits':bits[index].tolist(),'reference_net_value':float(net[index]),'reference_regret':float(net[valid].max()-net[index]),'harmful_reference':bool(index!=noop and net[index]<net[noop]),'is_noop':index==noop,'selection_uses_reference':profile in ('singleton_reference','compound_reference'),'reference_kind':truth_kind,'training_steps':0,'certified':False,'result_kind':'scripted_demonstration'})
    write(folder/'scripted_decisions.json',{p:{'index':i,'bits':bits[i].tolist()} for p,i in selected.items()})
    return selected


def run():
    started=time.perf_counter();rows=[]
    # A trainer call is a hard failure, including accidental calls by adapters.
    import cora.benchmarks.learners as learners
    learners.train=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Training is forbidden in this demo'))
    for seed in CONFIG['seeds']:
        for scenario in ('sparse','dense','misspecified','policy_shift'):
            trial=SyntheticTrial({'edits':5,'interaction_order':2,'density':.6,'noise':.1,'edit_cost':.1,'forbidden_pairs':[[0,4]],'shift_magnitude':.5},seed,scenario)
            record_problem(trial.public.name,seed,trial.public.bits,trial.truth,trial.public.costs,trial.public.valid,'exact_synthetic',np.zeros(len(trial.truth)),rows)
        scalable=ScalableSynthetic(edits=64,agents=64,order=4,candidates=80,seed=seed)
        p=scalable.public;values=scalable.evaluator_values()
        record_problem('scalable_64_edit_pool',seed,p.bits,values,p.costs,p.valid,'exact_within_sampled_pool',np.zeros(len(values)),rows)
        for coupled in (False,True):
            scm=ReservoirSCM(coupling=coupled)
            bits=np.array([[m&1,(m>>1)&1] for m in range(4)])
            values=np.array([scm.branch(z) for z in bits])
            record_problem('temporal_coupled' if coupled else 'temporal_local',seed,bits,values,.1*bits.sum(axis=1),np.ones(4,bool),'exact_known_scm',np.zeros(4),rows)
    # No policies or estimators are trained. All spatial results are measured.
    for name in ('warehouse','rescue','wildfire','hospital'):
        print(f'Running scripted environment: {name}',flush=True)
        env=DomainEnv(name,horizon=CONFIG['horizon'],disruption_step=CONFIG['disruption_step'])
        policy=DomainGuidedPolicy(tuple(env.agents))
        for seed in CONFIG['seeds']:
            settings={'horizon':CONFIG['horizon'],'gamma':CONFIG['gamma'],'swap_cost':CONFIG['swap_cost'],'diagnostic_scale':1.,'oracle_repeats':CONFIG['oracle_repeats']}
            trial=DomainTrial(env,policy,settings,seed)
            p=trial.public
            selected=record_problem(name,seed,p.bits,trial.truth,p.costs,p.valid,'monte_carlo_scripted_policy',trial.oracle_standard_error,rows)
            for profile,index in selected.items():
                trace=trial.trace(index,900000)
                org=apply(env.spec,trial.base.organization,trial.options[index])
                folder=HERE/'results'/name/f'seed_{seed}'
                save_trace(folder/f'{profile}_trace.npz',trace,org.assignments)
                row=next(r for r in rows if r['benchmark']==name and r['seed']==seed and r['profile']==profile)
                row['heldout_rollout_return']=float(trace['return'])
                if seed==CONFIG['render_seed']:
                    dest=HERE/'visualizations'/name/profile
                    save_frames(application_frames(trace,env,np.asarray(org.assignments),profile),dest,CONFIG['fps'])
            if seed==CONFIG['render_seed']:
                comparison(HERE/'visualizations'/name/'status_quo/replay.gif',HERE/'visualizations'/name/'compound_reference/replay.gif',HERE/'visualizations'/name/'comparison')
        jax.clear_caches()
    write(HERE/'results/results.json',rows)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (HERE/'results/results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    manifest={'label':LABEL,'training_steps':0,'gradient_updates':0,'training_mock':'handwritten observation-conditioned controller; no learned checkpoint','learner_aliases':{name:'same scripted controller; algorithm not executed' for name in ('IPPO','MAPPO','IQL','VDN','QMIX')},'method_aliases':'Five explicit scripted decision profiles; not reproductions of CORA/estimators/published methods','config':CONFIG,'spatial_reference_repeats':CONFIG['oracle_repeats'],'heldout_stream':900000,'reference_streams':'100000 + replica','wall_seconds':time.perf_counter()-started,'source_sha256':{str(p.relative_to(HERE.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE.parent.parent/'src/cora').rglob('*.py')},'rows':len(rows),'benchmark_conditions':len({r['benchmark'] for r in rows})}
    write(HERE/'manifest.json',manifest)
    figures_and_gallery(rows)
    verify()
    print(json.dumps({'folder':str(HERE),'rows':len(rows),'training_steps':0}))


def figures_and_gallery(rows):
    figures=HERE/'visualizations/plots';figures.mkdir(parents=True,exist_ok=True)
    names=sorted({r['benchmark'] for r in rows});profiles=CONFIG['profiles']
    matrix=np.array([[np.mean([r['reference_regret'] for r in rows if r['benchmark']==n and r['profile']==p]) for p in profiles] for n in names])
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(matrix,aspect='auto',cmap='viridis');ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(profiles)),profiles,rotation=25,ha='right');fig.colorbar(im,ax=ax,label='Mean reference regret (different scales across domains)');ax.set_title('Scripted decision profiles; privileged reference selection, no training');fig.tight_layout();fig.savefig(figures/'regret.png');fig.savefig(figures/'regret.pdf');plt.close(fig)
    for name in names:
        fig,ax=plt.subplots(figsize=(7,4))
        data=[[r['reference_net_value'] for r in rows if r['benchmark']==name and r['profile']==p] for p in profiles]
        ax.bar(range(len(profiles)),[np.mean(x) for x in data],yerr=[np.std(x) for x in data],color='#4c78a8');ax.set_xticks(range(len(profiles)),profiles,rotation=25,ha='right');ax.set_ylabel('Scripted reference net value; bars ± seed SD');ax.set_title(name+' / NO TRAINING');fig.tight_layout();fig.savefig(figures/f'{name}.png');fig.savefig(figures/f'{name}.pdf');plt.close(fig)
        if (HERE/'results'/name/f"seed_{CONFIG['render_seed']}/status_quo_trace.npz").exists():
            fig,ax=plt.subplots(figsize=(7,4))
            for profile in profiles:
                with np.load(HERE/'results'/name/f"seed_{CONFIG['render_seed']}/{profile}_trace.npz") as trace:ax.plot(trace['rewards'],label=profile)
            ax.axvline(CONFIG['disruption_step'],ls='--',color='red',label='disruption')
            ax.set(xlabel='Step',ylabel='Native collective reward',title=name+' / scripted rollout');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figures/f'{name}_reward.png');plt.close(fig)
    cards=[]
    catalog=json.loads((HERE/'mock_catalog.json').read_text())
    cards.append('<section><h2>Training and baseline substitutes</h2><p>All learner names share one handwritten policy; algorithms are not executed.</p><p><a href="mock_catalog.json">Mock mappings and limitations</a></p><table><tr><th>Mock of</th><th>Replacement profile</th></tr>'+''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(entry["replacement_profile"])}</td></tr>' for name,entry in catalog['organizational_methods'].items())+'</table></section>')
    if (HERE/'adaptive/index.html').exists():
        cards.append('<section><h2>Live organizational adoption</h2><a href="adaptive/index.html">Four matched scripted failure/reconfiguration comparisons</a><p>Predefined swap; frozen handwritten policy; zero training.</p></section>')
    if (HERE/'library_environments/index.html').exists():
        cards.append('<section><h2>Native JaxMARL environments</h2><p>Simple World Comm replaces Simple Spread. SMAX, OvercookedV2 and JaxNav run their genuine upstream dynamics.</p><a href="library_environments/index.html">Open rendered environments, screenshots and scripted returns</a><img src="library_environments/world_comm/guided/replay.gif" width="100%"></section>')
    for name in ('warehouse','rescue','wildfire','hospital'):
        cards.append(f'<section><h2>{name}</h2><p>Left: status quo. Right: compound reference (privileged finite branch search).</p><img src="visualizations/{name}/comparison/replay.gif" width="100%"><p>'+ ' · '.join(f'<a href="visualizations/{name}/{p}/replay.gif">{p}</a>' for p in profiles)+'</p></section>')
    (HERE/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>CORA scripted demonstration</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui;background:#f8fafc;color:#172033}section{background:white;padding:20px;margin:24px 0}aside{background:#fff0c2;padding:20px}a{color:#174cad}</style><h1>CORA — no-training demonstration</h1><aside><b>'+LABEL+'</b><p>Real simulator outcomes under hand-coded policies. These are not expected trained IPPO/MAPPO/IQL/VDN/QMIX performance. Reference selection uses privileged branch values. No gradients, learner checkpoints, convergence claims or safety certification.</p></aside><p><a href="manifest.json">Provenance</a> · <a href="results/results.csv">CSV</a> · <a href="results/results.json">Raw results</a> · <a href="verification.json">Verification</a></p>'+''.join(cards)+'<h2>Analytical scripted profiles</h2><img src="visualizations/plots/regret.png" width="100%"><h2>All analytical scenarios</h2>'+''.join(f'<p>{html.escape(n)}<br><img src="visualizations/plots/{n}.png" width="650"></p>' for n in names)+'</html>')


def verify():
    rows=json.loads((HERE/'results/results.json').read_text())
    assert len(rows)==11*len(CONFIG['seeds'])*len(CONFIG['profiles'])
    assert all(r['training_steps']==0 and not r['certified'] and np.isfinite(r['reference_net_value']) for r in rows)
    for name in {r['benchmark'] for r in rows}:
        for seed in CONFIG['seeds']:
            reference=json.loads((HERE/'results'/name/f'seed_{seed}/evaluation_reference.json').read_text())
            for r in [r for r in rows if r['benchmark']==name and r['seed']==seed]:assert reference['valid'][r['selected_index']]
    gifs=list((HERE/'visualizations').rglob('replay.gif'))
    assert len(gifs)==24
    for path in gifs:
        with Image.open(path) as im:assert im.is_animated and im.n_frames==CONFIG['horizon']
    for name in ('warehouse','rescue','wildfire','hospital'):
        for seed in CONFIG['seeds']:
            for profile in CONFIG['profiles']:
                with np.load(HERE/'results'/name/f'seed_{seed}'/f'{profile}_trace.npz') as trace:
                    value=float(np.sum(trace['rewards']*CONFIG['gamma']**np.arange(CONFIG['horizon'])))
                    row=next(r for r in rows if r['benchmark']==name and r['seed']==seed and r['profile']==profile)
                    assert np.isclose(value,row['heldout_rollout_return'],atol=1e-5)
    write(HERE/'verification.json',{'training_steps':0,'gradient_updates':0,'result_rows':len(rows),'conditions':11,'validated_replay_gifs':len(gifs),'valid_decisions':True,'finite_results':True,'label':LABEL})


CONFIG=json.loads((HERE/'config.json').read_text())
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');parser.add_argument('--plots-only',action='store_true');args=parser.parse_args()
    if args.verify_only:verify()
    elif args.plots_only:figures_and_gallery(json.loads((HERE/'results/results.json').read_text()));verify()
    else:run()
