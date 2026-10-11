"""Publication artifacts from saved application traces and result rows only."""
import html
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cora.benchmarks.evaluation import performance_profile


def render_all(output,rows,config):
    links=[]
    for path in sorted((output/'runs').glob('*/*/*_trace.npz')):
        with np.load(path) as trace:
            positions=trace['positions'];queues=trace['queue'];assignments=trace['assignments'];rewards=trace['rewards']
        dest=output/'media'/path.parent.parent.name/path.parent.name/path.stem.replace('_trace','')
        dest.mkdir(parents=True,exist_ok=True)
        frames=[]
        for t,pos in enumerate(positions):
            fig,ax=plt.subplots(figsize=(5,5))
            sites=np.array([[1,1],[3,1],[3,3]])
            ax.scatter(sites[:,0],sites[:,1],marker='s',s=200,color='lightgray')
            ax.scatter(pos[:,0],pos[:,1],c=['red','green','blue'],s=70)
            for i in range(3):
                ax.text(*pos[i],f'a{i} → m{assignments[i]}',fontsize=8)
                ax.text(sites[i,0],sites[i,1]+.3,f'm{i}: {queues[t,i]:.0f}',fontsize=8)
            ax.set(xlim=(-1,4),ylim=(-1,4),title=f'{path.parent.parent.name} / {path.stem} / t={t}')
            fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'));plt.close(fig)
        frames[0].save(dest/'replay.gif',save_all=True,append_images=frames[1:],duration=1000//config['media']['fps'],loop=0)
        for t in sorted({0,len(frames)//2,len(frames)-1}):frames[t].save(dest/f'frame_{t:03d}.png')
        fig,ax=plt.subplots(figsize=(5,3));ax.plot(rewards);ax.axvline(config['environment']['disruption_step'],color='red',linestyle='--');ax.set(xlabel='Step',ylabel='Collective native reward');fig.tight_layout();fig.savefig(dest/'reward.png');fig.savefig(dest/'reward.pdf');plt.close(fig)
        links.append(f'<li><a href="{html.escape(str((dest/"replay.gif").relative_to(output)))}">{html.escape(path.parent.parent.name+" / "+path.stem)}</a></li>')
    figures=output/'figures';figures.mkdir(exist_ok=True)
    # Matched seed/context comparisons; no simulator is invoked during analysis.
    for cora in sorted((output/'media').glob('*/*/cora/replay.gif')):
        noop=cora.parent.parent/'no_op/replay.gif'
        if not noop.exists():continue
        with Image.open(cora) as left, Image.open(noop) as right:
            if left.n_frames!=right.n_frames:raise ValueError('Comparison horizons differ')
            paired=[]
            for t in range(left.n_frames):
                left.seek(t);right.seek(t)
                frame=Image.new('RGB',(left.width+right.width,max(left.height,right.height)),color='white')
                frame.paste(left.convert('RGB'),(0,0));frame.paste(right.convert('RGB'),(left.width,0))
                paired.append(frame)
            destination=cora.parent.parent/'comparison.gif'
            paired[0].save(destination,save_all=True,append_images=paired[1:],duration=1000//config['media']['fps'],loop=0)
            links.append(f'<li><a href="{html.escape(str(destination.relative_to(output)))}">Matched CORA / no-op: {html.escape(cora.parent.parent.parent.name)}</a></li>')
    for benchmark in sorted({r['benchmark'] for r in rows}):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['benchmark']==benchmark and r['method']==method]
            if not selected:continue
            budgets=sorted({r['budget'] for r in selected})
            means=[np.mean([r['simple_regret'] for r in selected if r['budget']==b and r['simple_regret'] is not None]) for b in budgets]
            axes[0].plot(budgets,means,label=method)
            regret=[r['simple_regret'] for r in selected if r['budget']==max(budgets) and r['simple_regret'] is not None]
            if regret:
                thresholds=np.linspace(0,max(1,max(regret)),50)
                axes[1].plot(thresholds,performance_profile(regret,thresholds),label=method)
        axes[0].set(xlabel='Intervention budget',ylabel='Mean MC-reference regret');axes[1].set(xlabel='Regret threshold',ylabel='Fraction below threshold');axes[0].legend(fontsize=6)
        fig.suptitle(benchmark);fig.tight_layout();fig.savefig(figures/f'{benchmark}.png');fig.savefig(figures/f'{benchmark}.pdf');plt.close(fig)
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>CORA application results</title><h1>Saved application experiments</h1><p>Frozen policies per decision; retraining protocol recorded in manifest. Confidence is diagnostic; MC oracle precision is logged.</p><ul>'+''.join(links)+'</ul><p><a href="statistics.json">Seed statistics</a> / <a href="results.json">Raw rows</a></p></html>')
