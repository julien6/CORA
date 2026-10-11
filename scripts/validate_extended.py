"""Bounded CPU-only validation; no behavioral-policy training."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from cora.benchmarks.domains import DomainConfig, OrganizationalDomain
from cora.benchmarks.methods import run_method, METHODS
from cora.benchmarks.temporal_trial import ReservoirSCM


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    temporal = [ReservoirSCM(coupling=c).run(seed=s,budget=16) for c in (False,True) for s in (7,8)]
    (output/'temporal.json').write_text(json.dumps(temporal,indent=2))
    rows=[]
    for name in ('warehouse','rescue','wildfire','hospital'):
        domain = OrganizationalDomain(DomainConfig(name))
        dest = output/name; dest.mkdir(exist_ok=True)
        values,se = domain.oracle(seed=7,repeats=8)
        (dest/'evaluation_oracle.json').write_text(json.dumps({'values':values.tolist(),'standard_errors':se.tolist(),'privileged':True,'repeats':8},indent=2))
        for method in METHODS:
            if method in ('oracle','cora_no_constraints','cora_conditional'):
                continue
            decisions,logs = run_method(method,domain.public,lambda index,step: domain.observe(index,step,7),[4,8],7,{'ridge':.01,'diagnostic_scale':1.})
            (dest/f'{method}.json').write_text(json.dumps({'decisions':decisions,'queries':logs},indent=2))
            chosen = decisions[-1]['selected_index']
            net=values-domain.public.costs
            rows.append({'domain':name,'method':method,'regret_mc_reference':float(net.max()-net[chosen]),'selected':chosen,'certified':False})
            if method not in ('cora','no_op'):
                continue
            replay=domain.rollout(domain.bits[chosen],900007)
            (dest/f'{method}_replay.json').write_text(json.dumps(replay,indent=2))
            frames=[]
            for t,frame in enumerate(replay['frames']):
                fig,ax=plt.subplots(figsize=(4,4))
                pos=np.array(frame['positions']); sites=np.array(frame['sites'])
                ax.scatter(sites[:,0],sites[:,1],marker='s',s=180,c='lightgray')
                ax.scatter(pos[:,0],pos[:,1],s=60,c=['red','green','blue'])
                for i,(x,y) in enumerate(sites):
                    ax.text(x,y+.2,f"mission {i}: {frame['queue'][i]:.0f}",fontsize=8)
                ax.set(xlim=(-1,4),ylim=(-1,4),title=f'{name} / {method} / step {t}')
                fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
                plt.close(fig)
            frames[0].save(dest/f'{method}.gif',save_all=True,append_images=frames[1:],duration=150,loop=0)
            frames[-1].save(dest/f'{method}.png')
    (output/'results.json').write_text(json.dumps(rows,indent=2))
    fig,ax=plt.subplots(figsize=(9,4))
    for i,name in enumerate(('warehouse','rescue','wildfire','hospital')):
        data=[r for r in rows if r['domain']==name]
        ax.plot(range(len(data)),[r['regret_mc_reference'] for r in data],marker='o',label=name)
    ax.set_xticks(range(len(data)),[r['method'] for r in data],rotation=60,ha='right')
    ax.set_ylabel('Regret against 8-rollout MC reference'); ax.set_title('Technical validation, one seed; not a performance comparison')
    ax.legend();fig.tight_layout();fig.savefig(output/'technical_regret.png');plt.close(fig)
    links=''.join(f'<li>{n}: <a href="{n}/cora.gif">CORA replay</a> / <a href="{n}/no_op.gif">no-op replay</a></li>' for n in ('warehouse','rescue','wildfire','hospital'))
    (output/'index.html').write_text(f'<!doctype html><html><meta charset="utf-8"><title>Extended technical validation</title><h1>Small application prototypes</h1><p>One seed, frozen scripted policy, diagnostic confidence; no scientific performance claim.</p><ul>{links}</ul><img src="technical_regret.png" width="900"></html>')
    return {'rows':len(rows),'temporal_trials':len(temporal),'behavioral_training_steps':0}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/extended-smoke'))
    print(json.dumps(run(parser.parse_args().output)))
