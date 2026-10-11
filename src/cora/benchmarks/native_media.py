"""Native upstream visualizers, reconstructed from lossless saved state leaves."""
import html
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from cora.benchmarks.native_env import make_native_env


def load_states(path,env):
    prototype=jax.eval_shape(env.reset,jax.random.PRNGKey(0))[1].env_state
    paths,tree=jax.tree_util.tree_flatten_with_path(prototype)
    with np.load(path) as arrays:
        count=len(arrays['rewards'])
        leaves=[arrays['state'+jax.tree_util.keystr(p)].copy() for p,_ in paths]
        rewards=arrays['rewards'].copy();assignments=arrays['assignments'].copy()
    states=[jax.tree.unflatten(tree,[jnp.asarray(v[t]) for v in leaves]) for t in range(count)]
    return states,rewards,assignments


def render_trace(path,dest,env,title,fps):
    states,rewards,assignments=load_states(path,env);dest.mkdir(parents=True,exist_ok=True);frames=[]
    native=env._env
    if env.domain=='overcooked_v2':
        from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
        visualizer=OvercookedV2Visualizer(tile_size=48,subdivs=1)
    elif env.domain=='world_comm':
        from jaxmarl.environments.mpe import MPEVisualizer
        visualizer=MPEVisualizer(native,states)
    for t,state in enumerate(states):
        if env.domain=='overcooked_v2':
            frame=Image.fromarray(np.asarray(visualizer._render_state(state))).convert('RGB')
            from PIL import ImageDraw
            canvas=Image.new('RGB',(max(frame.width,700),frame.height+76),'white');canvas.paste(frame,(0,76))
            d=ImageDraw.Draw(canvas);d.text((8,8),f'{title} / step {t} / reward {rewards[t]:.3f}',fill='black');d.text((8,28),f'Assignments: {assignments.tolist()} / native OvercookedV2',fill='black');frame=canvas
        else:
            if env.domain=='world_comm':fig,ax=visualizer.fig,visualizer.ax;visualizer.update(t)
            else:
                fig,ax=plt.subplots(figsize=(6,6))
                if env.domain=='smax':native._env.init_render(ax,(None,state.state,{a:jnp.array(0) for a in native._env.agents}),0,t)
                else:
                    obs_arr=native._get_obs(state);obs={a:obs_arr[i] for i,a in enumerate(env.agents)}
                    native.init_render(ax,state,obs=obs,colour_agents_by_idx=True,num_to_plot=12)
            ax.set_title(f'{title} / step {t}\nNative controlled-team mean reward {rewards[t]:.3f}',fontsize=10)
            fig.text(.02,.01,'Assignments: '+str(assignments.tolist())+' / native decision-step snapshot',fontsize=8)
            fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
            if env.domain!='world_comm':plt.close(fig)
        frames.append(frame)
    if env.domain=='world_comm':plt.close(visualizer.fig)
    frames[0].save(dest/'replay.gif',save_all=True,append_images=frames[1:],duration=1000//fps,loop=0)
    for t in sorted({0,len(frames)//2,len(frames)-1}):frames[t].save(dest/f'frame_{t:04d}.png')
    fig,ax=plt.subplots(figsize=(6,3));ax.plot(rewards);ax.set(xlabel='Step',ylabel='Controlled-team mean native reward',title=title);fig.tight_layout();fig.savefig(dest/'reward.png');fig.savefig(dest/'reward.pdf');plt.close(fig)


def render_all(output,rows,config):
    output=Path(output);links=[]
    for path in sorted((output/'runs').glob('*/*/*_trace.npz')):
        context=path.parent.parent.name
        domain=next(d for d in config['domains'] if context.startswith(d+'_'))
        env=make_native_env(domain,config['environment']);dest=output/'media'/context/path.parent.name/path.stem.replace('_trace','')
        render_trace(path,dest,env,context+' / '+path.stem,config['media']['fps'])
        screenshots=' · '.join(f'<a href="{html.escape(str((dest/f"frame_{t:04d}.png").relative_to(output)))}">step {t} PNG</a>' for t in sorted({0,config['evaluation']['horizon']//2,config['evaluation']['horizon']-1}))
        links.append(f'<li><a href="{html.escape(str((dest/"replay.gif").relative_to(output)))}">{html.escape(context+" / "+path.stem)}</a> — {screenshots}</li>')
    figures=output/'figures';figures.mkdir(exist_ok=True)
    statistics=json.loads((output/'statistics.json').read_text())
    figure_links=[]
    for benchmark in sorted({r['benchmark'] for r in rows}):
        fig,ax=plt.subplots(figsize=(7,4))
        for method in sorted({r['method'] for r in rows if r['benchmark']==benchmark}):
            selected=[r for r in rows if r['benchmark']==benchmark and r['method']==method]
            budgets=sorted({r['budget'] for r in selected})
            ax.plot(budgets,[np.mean([r['simple_regret'] for r in selected if r['budget']==b and r['simple_regret'] is not None]) for b in budgets],label=method)
        ax.set(xlabel='Intervention budget',ylabel='Mean MC-reference simple regret',title=benchmark);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figures/f'{benchmark}.png');fig.savefig(figures/f'{benchmark}.pdf');plt.close(fig)
    for benchmark in sorted({r['benchmark'] for r in rows}):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        metrics=(('simple_regret','MC-reference simple regret'),('heldout_effect_mae','Held-out causal effect MAE'),('selected_net_value','Selected net value'),('wall_seconds','Decision computation seconds'))
        for ax,(metric,label) in zip(axes.flat,metrics):
            for method in sorted({r['method'] for r in rows if r['benchmark']==benchmark}):
                entries=sorted([s for s in statistics['summaries'] if s['benchmark']==benchmark and s['method']==method and metric in s['metrics']],key=lambda s:s['budget'])
                if not entries:continue
                x=[s['budget'] for s in entries];y=[s['metrics'][metric]['iqm'] for s in entries]
                line=ax.plot(x,y,label=method)[0]
                bounded=[s for s in entries if s['metrics'][metric]['iqm_ci'] is not None]
                if bounded:ax.fill_between([s['budget'] for s in bounded],[s['metrics'][metric]['iqm_ci'][0] for s in bounded],[s['metrics'][metric]['iqm_ci'][1] for s in bounded],color=line.get_color(),alpha=.12)
            ax.set(xlabel='Intervention budget',ylabel=label);ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
        fig.suptitle(benchmark+' / seed IQM with bootstrap confidence intervals');fig.tight_layout(rect=(0,.13,1,.95))
        path=figures/f'{benchmark}_metrics.png';fig.savefig(path,dpi=150);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
        figure_links.append(f'<li><a href="figures/{html.escape(path.name)}">{html.escape(benchmark)}: regret, effect error, value and computation</a> · <a href="figures/{html.escape(path.with_suffix(".pdf").name)}">PDF</a></li>')
    for cora in sorted((output/'media').glob('*/*/cora/replay.gif')):
        noop=cora.parent.parent/'no_op/replay.gif'
        if not noop.exists():continue
        paired=[]
        with Image.open(cora) as left,Image.open(noop) as right:
            if left.n_frames!=right.n_frames:raise ValueError('Unmatched native comparison horizons')
            for t in range(left.n_frames):
                left.seek(t);right.seek(t);frame=Image.new('RGB',(left.width+right.width,max(left.height,right.height)),'white');frame.paste(left.convert('RGB'),(0,0));frame.paste(right.convert('RGB'),(left.width,0));paired.append(frame)
        dest=cora.parent.parent/'comparison.gif';paired[0].save(dest,save_all=True,append_images=paired[1:],duration=1000//config['media']['fps'],loop=0)
        links.append(f'<li><a href="{html.escape(str(dest.relative_to(output)))}">CORA / no-op: {html.escape(cora.parent.parent.parent.name)}</a></li>')
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><h1>Native JaxMARL experiments</h1><p>Actual saved rollouts; assignment-conditioned continuation; World Comm scores only the controlled pursuer team. SMAX shows decision-step snapshots. MC reference precision and protocol are recorded in raw metadata.</p><ul>'+''.join(links)+'</ul><h2>Quantitative figures</h2><ul>'+''.join(figure_links)+'</ul><p><a href="results.json">Raw results</a> / <a href="statistics.json">Statistics</a></p></html>')
