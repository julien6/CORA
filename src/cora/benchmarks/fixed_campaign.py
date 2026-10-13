"""Fixed-controller sequential mission benchmark; no MARL training.
Independent resource modules justify sparse intervention support by construction.
This controlled benchmark complements, and does not replace, native environments.
"""
from pathlib import Path
from itertools import product
import argparse
import csv
import hashlib
import json
import numpy as np
from cora.features import FeatureEncoder
from cora.benchmarks.problems import PublicProblem
from cora.benchmarks.methods import run_method

METHODS = ['cora','linear_g_optimal','factorized_uniform','structure_blind_active','global_linear','cora_no_confidence','no_op']

class FixedMissionTrial:
    def __init__(self, seed=0, horizon=20, repeats=16, scenario='sparse'):
        if horizon < 1 or repeats < 2 or scenario not in ('sparse','dense','misspecified'):
            raise ValueError('Invalid horizon, reference repetitions or scenario')
        self.seed,self.horizon,self.scenario=seed,horizon,scenario
        self.bits=np.array(list(product((0,1),repeat=4)),int)
        edges=((0,1),(2,3)) if scenario != 'dense' else ((0,1,2,3),)
        if scenario=='misspecified':edges=tuple((i,) for i in range(4))
        family=FeatureEncoder.from_hyperedges(4,edges).family
        self.public=PublicProblem('fixed_missions_'+scenario,self.bits,self.bits.sum(1)*.15,np.ones(16,bool),family,1.,'fixed-queue-controller-v1')
        reference=np.array([[self.rollout(z,10000+r)[0] for z in self.bits] for r in range(repeats)])
        self.truth=reference.mean(0)
        self.se=reference.std(0,ddof=1)/np.sqrt(repeats)
        self.reference=reference

    def rollout(self,bits,stream):
        # Every edit swaps two agents' mission commitments in its local module.
        assignments=np.arange(8)
        for edit,enabled in enumerate(bits):
            if enabled:
                a=2*edit;assignments[a:a+2]=assignments[a:a+2][::-1]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,stream]))
        # Fixed exogenous draws indexed by time and agent: matched across choices.
        draws=rng.random((self.horizon,8))
        remaining=np.zeros(8,int);deliveries=np.zeros(8,int);records=[]
        for time in range(self.horizon):
            requests={}
            for agent,mission in enumerate(assignments):
                module=agent//4
                # Mission-specific resource and service duration.
                resource=(0 if self.scenario=='dense' else module, int(mission%2))
                requests.setdefault(resource,[]).append(agent)
            served=[]
            for resource,agents in sorted(requests.items()):
                # Deterministic rotating priority; same controller for all methods.
                ordered=sorted(agents,key=lambda a:(a-time)%8)
                agent=ordered[0];mission=assignments[agent]
                # Heterogeneous, fixed capabilities, independent of method.
                probability=.9 if agent%2==mission%2 else .55
                if draws[time,agent]<probability:
                    remaining[agent]+=1
                    if remaining[agent]>=1+(mission//2)%2:
                        deliveries[agent]+=1;remaining[agent]=0
                served.append(agent)
            records.append({'time':time,'assignments':assignments.tolist(),'served':served,'deliveries':deliveries.tolist()})
        return float(deliveries.sum()),records

    def observe(self,index,step):return self.rollout(self.bits[index],step+1)[0]

    def source(self,index):
        _,trace=self.rollout(self.bits[index],50000)
        lines=['@moise "Engagements de mission"','group team "Équipe"','role executor "Exécutant"','contains team executor 8..8']
        for i in range(8):
            lines += [f'goal g{i} "Livraisons de mission {i}"',f'scheme s{i} g{i}',f'mission m{i}',f'goals m{i} g{i}',f'per executor m{i} Any',f'agent a{i} "Agent {i}"',f'plays a{i} executor team']
        lines += [f'commits a{i} m{m}' for i,m in enumerate(trace[0]['assignments'])]
        return '\n'.join(lines+['@end'])+'\n'


def run(output,seeds,budgets,horizon,repeats):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Choose an empty output directory to preserve previous results')
    output.mkdir(parents=True,exist_ok=True);rows=[]
    config={'ridge':.01,'diagnostic_scale':1.,'experiment_cost_weight':0.}
    for scenario in ('sparse','dense','misspecified'):
        for seed in seeds:
            trial=FixedMissionTrial(seed,horizon,repeats,scenario)
            for method in METHODS:
                decisions,logs=run_method(method,trial.public,trial.observe,budgets,seed,config)
                result={'decisions':decisions,'experiments':logs}
                folder=output/scenario/f'seed_{seed}'/method;folder.mkdir(parents=True)
                (folder/'decisions.json').write_text(json.dumps(result,indent=2)+'\n')
                for decision in result['decisions']:
                    index=decision['selected_index'];net=trial.truth-trial.public.costs
                    rows.append({'scenario':scenario,'seed':seed,'method':method,'budget':decision['budget'],'selected_index':index,'net_value':float(net[index]),'simple_regret':float(net.max()-net[index]),'harmful':bool(net[index]<net[trial.public.noop]),'noop':index==trial.public.noop,'reference_se':float(trial.se[index]),'certified':False})
                selected=result['decisions'][-1]['selected_index']
                before=trial.rollout(trial.bits[trial.public.noop],50000)[1]
                after=trial.rollout(trial.bits[selected],50000)[1]
                (folder/'replay.json').write_text(json.dumps({'decision':result['decisions'][-1],'before':before,'after':after},indent=2)+'\n')
                (folder/'organization.moise').write_text(trial.source(selected))
                if seed == seeds[0] and scenario == 'sparse' and method in ('cora','linear_g_optimal'):
                    from moise_diagram import compile_file
                    from PIL import Image, ImageDraw
                    (folder/'before.moise').write_text(trial.source(trial.public.noop))
                    compile_file(folder/'organization.moise',folder/'organization.svg',views=['ss','oe'])
                    compile_file(folder/'before.moise',folder/'before.svg',views=['ss','oe'])
                    frames=[]
                    for pre,post in zip(before,after):
                        image=Image.new('RGB',(1000,440),'white');draw=ImageDraw.Draw(image)
                        draw.text((20,12),f"{method}: recorded matched rollout, step {pre['time']} — selected {selected}",fill='black')
                        for column,(label,state) in enumerate((('Initial organization',pre),('Selected organization',post))):
                            x=20+column*500;draw.text((x,42),label,fill='black')
                            for agent,(mission,count) in enumerate(zip(state['assignments'],state['deliveries'])):
                                y=80+agent*38
                                draw.text((x,y),f'Agent {agent} -> mission {mission}: {count} deliveries',fill='black')
                                draw.rectangle((x+300,y,x+300+count*8,y+14),fill='#007a83' if agent in state['served'] else '#9baab3')
                        frames.append(image)
                    frames[0].save(folder/'replay.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
                    frames[-1].save(folder/'screenshot.png')
            (output/scenario/f'seed_{seed}'/'reference.json').write_text(json.dumps({'values':trial.truth.tolist(),'se':trial.se.tolist(),'samples':trial.reference.tolist()},indent=2)+'\n')
    with (output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    rng=np.random.default_rng(431)
    for scenario in ('sparse','dense','misspecified'):
        for method in METHODS:
            for budget in budgets:
                group=[r for r in rows if r['scenario']==scenario and r['method']==method and r['budget']==budget]
                values=np.array([r['simple_regret'] for r in group])
                ci=None if len(values)<2 else np.quantile(rng.choice(values,(1000,len(values))).mean(1),[.025,.975]).tolist()
                summaries.append({'scenario':scenario,'method':method,'budget':budget,'mean_regret':float(values.mean()),'seed_bootstrap_ci':ci,'harm_rate':float(np.mean([r['harmful'] for r in group]))})
    (output/'statistics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'seeds':seeds,'budgets':budgets,'horizon':horizon,'reference_repeats':repeats,'methods':METHODS,'training_steps':0,'controller':'fixed-queue-controller-v1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'confidence':'diagnostic, not certified','baseline':'linear_g_optimal is a generic myopic design, not RAGE reproduction'},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,scenario in zip(axes,('sparse','dense','misspecified')):
        for method in METHODS:
            means=[np.mean([r['simple_regret'] for r in rows if r['scenario']==scenario and r['method']==method and r['budget']==b]) for b in budgets]
            ax.plot(budgets,means,marker='o',label=method)
        ax.set(title=scenario,xlabel='Experiment budget',ylabel='Mean simple regret')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'regret.png');fig.savefig(output/'regret.pdf');plt.close(fig)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Fixed-controller campaign</h1><p>Measured sequential simulator outcomes; diagnostic decisions. Reference Monte Carlo uncertainty is recorded. No MARL training.</p><img width="100%" src="regret.png"><p><a href="results.csv">Results</a> · <a href="manifest.json">Protocol</a></p>')

    with (output/'index.html').open('a') as page:
        for method in ('cora','linear_g_optimal'):
            prefix=f'sparse/seed_{seeds[0]}/{method}'
            page.write(f'<p>{method}: <a href="{prefix}/replay.gif">Replay</a> · <a href="{prefix}/organization.svg">Organization</a> · <a href="{prefix}/decisions.json">Decisions</a></p>')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/fixed-missions');p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    plan={'output':a.output,'seeds':[0] if a.smoke else list(range(10)),'budgets':[2,4] if a.smoke else [4,8,16,32],'horizon':6 if a.smoke else 30,'repeats':2 if a.smoke else 64}
    if a.dry_run:print(json.dumps(plan,indent=2));return
    run(**plan)

if __name__=='__main__':main()
