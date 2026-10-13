"""Short genuine JaxMARL rollouts. No training; writes only into this demo folder."""
from run import HERE,write,save_frames,comparison
import sys
sys.path.insert(0,str(HERE/'.deps'))
from scene_renderer import text,tree,agent,COLORS,kitchen_image
import argparse, hashlib, json, time, subprocess
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image,ImageDraw
import jaxmarl

DEST=HERE/'library_environments'
FPS=8
SEED=7
SCENARIOS={
 'world_comm':{'steps':70,'description':'Native MPE Simple World Comm: four pursuers, two foragers, food, obstacle and forests. Leader broadcasts a four-channel target-position code.','source':'jaxmarl/environments/mpe/simple_world_comm.py','policy_access':'local observations, leader communication'},
 'smax':{'steps':45,'description':'Native SMAX 3m: three marines against three heuristic enemies. Upstream observation-based heuristic versus random valid ally actions. Frames are decision-step snapshots, not interpolated combat ticks.','source':'jaxmarl/environments/smax/smax_env.py','policy_access':'unit-list observations; native legal-action mask'},
 'overcooked_v2':{'steps':110,'description':'Native OvercookedV2 cramped_room_v2 (contextual kitchen rendering and original renderer screenshots): two chefs, multiple ingredients, recipe indicator, cooking and delivery. Full-state task planner versus random actions. This illustration does not test decentralized partial-observation learning.','source':'jaxmarl/environments/overcooked_v2/overcooked.py','policy_access':'privileged full simulator state for kitchen planner'},
 'jaxnav':{'steps':100,'description':'Native JaxNav BlankCross2: differential-drive robots crossing toward opposing goals. Local goal-bearing controller with LiDAR braking versus random actions.','source':'jaxmarl/environments/jaxnav/jaxnav_env.py','policy_access':'local goal bearing, range and LiDAR observations'},
}


def frame_base(name,profile,t,lines):
    im=Image.new('RGB',(1000,650),'#eef2f6');d=ImageDraw.Draw(im)
    d.rectangle((0,0,1000,76),fill='#172b40');text(d,(24,14),f'JaxMARL / {name.upper()} / {profile.upper()}','white',22)
    text(d,(24,47),f'Step {t:03d}  |  Native dynamics • recorded state','#b8d1e4',15)
    d.rounded_rectangle((638,96,978,580),radius=12,fill='white')
    text(d,(656,112),'LIVE SIMULATOR DATA',size=18)
    for index,line in enumerate(lines):
        text(d,(656,153+32*index),line,size=14)
    return im,d


def world_actions(env,obs,guided):
    actions={};leader=np.asarray(obs[env.leader]);targets=leader[14:24].reshape(5,2)[-2:]
    target=targets[np.argmin(np.linalg.norm(targets,axis=1))]+leader[2:4]
    msg=np.array([max(-target[0],0),max(target[0],0),max(-target[1],0),max(target[1],0)])
    for i,a in enumerate(env.agents):
        o=np.asarray(obs[a]);others=o[14:24].reshape(5,2)
        if i<4:
            candidates=others[-2:];visible=np.linalg.norm(candidates,axis=1)>.0001
            if i and guided and np.sum(o[-4:])>.001:
                c=o[-4:];delta=np.array([c[1]-c[0],c[3]-c[2]])-o[2:4]
            elif np.any(visible):delta=candidates[np.argmin(np.where(visible,np.linalg.norm(candidates,axis=1),np.inf))]
            else:delta=-o[2:4]
        else:
            food=o[6:10].reshape(2,2);delta=food[np.argmin(np.linalg.norm(food,axis=1))]
            threats=others[:4];dist=np.linalg.norm(threats,axis=1);mask=(dist>.001)&(dist<.45)
            delta=delta+np.sum(-threats/(dist[:,None]**2+.02)*mask[:,None],axis=0)*.16
            delta=delta-.35*o[2:4]*(np.abs(o[2:4])>.85)
        force=np.clip(1.7*delta-.65*o[:2],-1,1)
        move=np.array([0,max(-force[0],0),max(force[0],0),max(-force[1],0),max(force[1],0)],np.float32)
        actions[a]=jnp.asarray(np.r_[move,np.clip(msg,0,1) if guided else np.zeros(4)] if i==0 else move)
    return actions


def world_frame(env,state,t,profile,rewards):
    pos=np.asarray(state.p_pos);c=np.asarray(state.c[0]);im,d=frame_base('world comm',profile,t,['Red: leader + 3 pursuers','Teal: 2 foragers','Blue discs: food reward','Green discs: forest visibility','Grey rock: collision obstacle','',f'Pursuer mean return: {rewards[:4].mean():.2f}',f'Forager mean return: {rewards[4:].mean():.2f}','',f'Leader channels:',np.array2string(c,precision=2)])
    def p(v):return (int(322+v[0]*205),int(337-v[1]*205))
    d.rounded_rectangle((24,96,614,580),radius=12,fill='#e2edd9',outline='#9bac9b',width=2)
    for i in range(9,11):
        x,y=p(pos[i]);r=int(float(env.rad[i])*205);d.ellipse((x-r,y-r,x+r,y+r),fill='#bfd5ae',outline='#749768',width=2)
        for dx,dy in [(-r//2,0),(r//2,r//4),(0,-r//3)]:tree(d,x+dx,y+dy,12)
    x,y=p(pos[6]);r=int(float(env.rad[6])*205);d.ellipse((x-r,y-r,x+r,y+r),fill='#879491',outline='#556562',width=3)
    for i in (7,8):
        x,y=p(pos[i]);d.ellipse((x-9,y-9,x+9,y+9),fill='#458bde',outline='white',width=2)
    for i in range(6):
        x,y=p(pos[i]);r=max(8,int(float(env.rad[i])*205));color='#a73838' if i==0 else '#e47160' if i<4 else '#279b7f'
        d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2);text(d,(x-5,y-8),str(i),'white',11)
        if i==0:d.arc((x-26,y-26,x+26,y+26),210,330,fill='#aa4b4b',width=2)
    return im


def chef_actions(env,state):
    from jaxmarl.environments.overcooked_v2.common import DynamicObject as D
    g=np.asarray(state.grid);positions=list(zip(np.asarray(state.agents.pos.x),np.asarray(state.agents.pos.y)));dirs=np.asarray(state.agents.dir);inv=np.asarray(state.agents.inventory)
    pot_y,pot_x=np.argwhere(g[:,:,0]==5)[0];pot=g[pot_y,pot_x];counts=lambda n:np.array([(int(n)>>(2+2*i))&3 for i in range(2)])
    missing=counts(state.recipe)-counts(pot[1]);ready=bool(int(pot[1])&2);actions={};occupied=set(positions)
    moves=[(1,0,0,2),(0,1,1,1),(-1,0,2,3),(0,-1,3,0)]
    for i,a in enumerate(env.agents):
        item=int(inv[i]);target=None
        if item&2:target=tuple(np.argwhere(g[:,:,0]==4)[0][::-1])
        elif item&1:
            if ready:target=(int(pot_x),int(pot_y))
        elif item:target=(int(pot_x),int(pot_y))
        elif i==0 and np.any(missing>0) and not pot[2] and not ready:
            ingredient=int(np.flatnonzero(missing>0)[0]);target=tuple(np.argwhere(g[:,:,0]==10+ingredient)[0][::-1])
        elif i==1:target=tuple(np.argwhere(g[:,:,0]==9)[0][::-1])
        parking=target is None
        if parking:target=(1,1) if i==0 else (3,2)
        start=tuple(map(int,positions[i]));q=deque([(start,[])]);seen={start};chosen=4
        while q:
            p,path=q.popleft()
            if parking and p==target:
                chosen=path[0] if path else 4;break
            adj=[m for m in moves if (p[0]+m[0],p[1]+m[1])==target]
            if adj and not parking:
                m=adj[0];chosen=path[0] if path else 5 if dirs[i]==m[3] else m[2];break
            for dx,dy,act,_ in moves:
                nxt=(p[0]+dx,p[1]+dy)
                if nxt not in seen and 0<=nxt[0]<g.shape[1] and 0<=nxt[1]<g.shape[0] and g[nxt[1],nxt[0],0]==0 and nxt not in occupied-{start}:
                    seen.add(nxt);q.append((nxt,path+[act]))
        actions[a]=jnp.array(chosen)
    return actions


def render_smax(env,state,t,profile,returns):
    s=state.state;im,d=frame_base('SMAX 3m',profile,t,['Blue: allied marines','Red: heuristic enemies','Bars: actual remaining health','Cross: defeated unit','',f'Ally mean return: {returns.mean():.3f}',f'Allies alive: {int(np.sum(s.unit_alive[:3]))}/3',f'Enemies alive: {int(np.sum(s.unit_alive[3:]))}/3','','Decision-step snapshots','No fabricated attack beams'])
    d.rounded_rectangle((24,96,614,580),radius=12,fill='#d7cfb6')
    for x in range(50,614,36):d.line((x,100,x,576),fill='#c7bea5')
    for y in range(110,580,36):d.line((28,y,610,y),fill='#c7bea5')
    for i,(x,y) in enumerate(np.asarray(s.unit_positions)):
        x=int(40+x/env.map_width*554);y=int(565-y/env.map_height*450);c='#5689dc' if i<3 else '#e56b58'
        d.ellipse((x-20,y+10,x+22,y+22),fill='#a39880')
        d.line((x-7,y+8,x-10,y+21),fill='#344652',width=7);d.line((x+7,y+8,x+10,y+21),fill='#344652',width=7)
        d.rounded_rectangle((x-16,y-9,x+16,y+12),radius=5,fill=c,outline='#33414b',width=2)
        d.ellipse((x-11,y-24,x+11,y-3),fill='#50636e',outline=c,width=3);d.rectangle((x-7,y-17,x+7,y-12),fill='#9ed0d7')
        d.rectangle((x+8,y-2,x+31,y+5),fill='#3d4c56');text(d,(x-5,y-6),i,'white',11)
        health=float(s.unit_health[i]);maxhp=float(env.unit_type_health[int(s.unit_types[i])]);d.rectangle((x-22,y-30,x+22,y-24),fill='#873d3d');d.rectangle((x-22,y-30,x-22+44*health/maxhp,y-24),fill='#399e68')
        if health<=0:d.line((x-18,y-18,x+18,y+18),fill='#222',width=4)
    return im


def run_case(name):
    cfg=SCENARIOS[name];steps=cfg['steps'];rows=[]
    if name=='world_comm':env=jaxmarl.make('MPE_simple_world_comm_v3',max_steps=steps+1)
    elif name=='smax':
        from jaxmarl.environments.smax import map_name_to_scenario
        env=jaxmarl.make('HeuristicEnemySMAX',scenario=map_name_to_scenario('3m'),max_steps=steps+1,see_enemy_actions=False)
    elif name=='overcooked_v2':env=jaxmarl.make('overcooked_v2',layout='cramped_room_v2',max_steps=steps+1)
    else:
        from jaxmarl.environments.jaxnav.jaxnav_singletons import make_jaxnav_singleton
        env=make_jaxnav_singleton('BlankCross2',map_id='Grid-Rand-Poly',normalise_obs=True,lidar_num_beams=32,max_steps=steps+1)
    transition=jax.jit(env.step_env)
    if name=='smax':
        from jaxmarl.environments.smax.heuristic_enemy import create_heuristic_policy,get_heuristic_policy_initial_state
        policy=jax.jit(jax.vmap(create_heuristic_policy(env._env,0)))
    if name=='overcooked_v2':
        from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
        viz=OvercookedV2Visualizer(tile_size=48,subdivs=1)
    for profile in ('reference','guided'):
        key=jax.random.PRNGKey(SEED);key,rk=jax.random.split(key);obs,state=env.reset(rk);frames=[];states=[];keys=[];actions_seq=[];rews=[];dones_seq=[];total=np.zeros(env.num_agents);terminal=False
        if name=='smax':hidden=jax.tree.map(lambda *xs:jnp.stack(xs),*([get_heuristic_policy_initial_state()]*env.num_agents))
        for t in range(steps):
            key,ak,sk=jax.random.split(key,3)
            if name=='world_comm':actions=world_actions(env,obs,profile=='guided')
            elif profile=='reference':
                ks=jax.random.split(ak,env.num_agents);actions={a:env.action_space(a).sample(ks[i]) for i,a in enumerate(env.agents)}
                if name=='smax':
                    masks=env.get_avail_actions(state);actions={a:jax.random.categorical(ks[i],jnp.where(masks[a],0.,-jnp.inf)) for i,a in enumerate(env.agents)}
            elif name=='smax':
                arr,hidden=policy(jax.random.split(ak,env.num_agents),hidden,jnp.stack([obs[a] for a in env.agents]));actions={a:arr[i] for i,a in enumerate(env.agents)}
            elif name=='overcooked_v2':actions=chef_actions(env,state)
            else:
                actions={}
                for a in env.agents:
                    o=np.asarray(obs[a]);angle=float(o[-2])*np.pi;lidar=(o[:env.lidar_num_beams]+.5)*env.lidar_max_range;front=lidar[np.abs(np.asarray(env.lidar_angles))<.45]
                    v=.8*max(0,np.cos(angle))*(float(front.min())>.7)
                    actions[a]=jnp.array([v,np.clip(1.5*angle,-.6,.6)],dtype=jnp.float32)
            states.append(state);keys.append(np.asarray(sk));actions_seq.append([np.pad(np.asarray(actions[a]).reshape(-1),(0,9-np.asarray(actions[a]).size)) for a in env.agents])
            if name=='world_comm':frame=world_frame(env,state,t,profile,total)
            elif name=='smax':frame=render_smax(env,state,t,profile,total)
            elif name=='overcooked_v2':
                frame,d=frame_base('OvercookedV2',profile,t,['Cramped room V2 / 2 chefs','Red: chef 0 / Blue: chef 1','Ingredients → pot → plate','Cooked dish → delivery','',f'Team mean return: {total.mean():.1f}',f'Recipe encoding: {int(state.recipe)}','', 'Guided planner: full-state access','Reference: random actions','State-faithful kitchen illustration'])
                kitchen=kitchen_image(state);frame.paste(kitchen,(32,106))
                if t in (0,steps//2,steps-1):
                    native=Image.fromarray(np.asarray(viz._render_state(state))).convert('RGB')
                    native_dest=DEST/name/profile;native_dest.mkdir(parents=True,exist_ok=True);native.save(native_dest/f'native_frame_{t:03d}.png')
            else:
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(6,5),dpi=100);env.init_render(ax,state,obs=obs,colour_agents_by_idx=True,num_to_plot=12);ax.set_title('Goals, robot footprints and actual LiDAR');fig.tight_layout();fig.canvas.draw();image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB');plt.close(fig)
                frame,d=frame_base('JaxNav',profile,t,['BlankCross2 / two robots','Native differential-drive physics','Markers: assigned goals','Dots: LiDAR obstacle readings','',f'Mean return: {total.mean():.2f}',f'Goals reached: {int(np.sum(state.goal_reached))}/2',f'Collisions: {int(np.sum(state.move_term))}','', 'Local observation controller']);frame.paste(image.resize((600,485)),(20,91))
            if terminal:
                ImageDraw.Draw(frame).rounded_rectangle((32,542,347,573),radius=4,fill='#172b40')
                text(ImageDraw.Draw(frame),(42,547),'EPISODE ENDED / final state frozen','white',14)
            frames.append(frame)
            if terminal:reward=np.zeros(env.num_agents);done=True
            else:
                obs,state,r,done_dict,_=transition(sk,state,actions);reward=np.array([float(r[a]) for a in env.agents]);done=bool(done_dict['__all__']);terminal=done
            rews.append(reward);dones_seq.append(done);total+=reward
        dest=DEST/name/profile;save_frames(frames,dest,FPS)
        leaves,structure=jax.tree.flatten(jax.tree.map(lambda *xs:jnp.stack(xs),*states))
        paths,_=jax.tree_util.tree_flatten_with_path(states[0]);schema=[jax.tree_util.keystr(p) for p,_ in paths]
        np.savez_compressed(dest/'trace.npz',**{f'state_{i}':np.asarray(v) for i,v in enumerate(leaves)},step_keys=keys,actions=np.asarray(actions_seq),rewards=rews,dones=dones_seq)
        row={'environment':name,'profile':profile,'seed':SEED,'steps':steps,'per_agent_return':total.tolist(),'mean_return':float(total.mean()),'team_returns':{'pursuers_mean':float(total[:4].mean()),'foragers_mean':float(total[4:].mean())} if name=='world_comm' else None,'policy_access':cfg['policy_access'] if profile=='guided' or name=='world_comm' else 'random legal actions','training_steps':0,'terminal_reached':terminal,'terminal_handling':'freeze final state and zero rewards after episode ends; no autoreset','state_leaf_paths':schema,'action_storage':'Each agent action flattened and right-zero-padded to length 9; native sizes in action_sizes','action_sizes':{a:int(np.asarray(actions[a]).size) for a in env.agents}}
        write(dest/'metadata.json',row);rows.append(row)
    comparison(DEST/name/'reference/replay.gif',DEST/name/'guided/replay.gif',DEST/name/'comparison')
    write(DEST/name/'results.json',rows);return rows


def gallery():
    rows=[];cards=[]
    for name,cfg in SCENARIOS.items():
        if not (DEST/name/'results.json').exists():continue
        data=json.loads((DEST/name/'results.json').read_text());rows+=data
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for row in data:
            with np.load(DEST/name/row['profile']/'trace.npz') as trace:
                cumulative=trace['rewards'].cumsum(0)
                if name=='world_comm':
                    axes[0].plot(cumulative[:,:4].mean(1),label=row['profile']+' pursuers');axes[1].plot(cumulative[:,4:].mean(1),label=row['profile']+' foragers')
                else:
                    axes[0].plot(cumulative.mean(1),label=row['profile']);axes[1].plot(trace['rewards'].mean(1),label=row['profile'])
        axes[0].set_title('Cumulative native return'+(' / pursuers' if name=='world_comm' else ''))
        axes[1].set_title('Cumulative native return / foragers' if name=='world_comm' else 'Native reward per step')
        for ax in axes:ax.set_xlabel('Step');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(name+' / one episode / no training');fig.tight_layout();fig.savefig(DEST/name/'returns.png',dpi=150);fig.savefig(DEST/name/'returns.pdf');plt.close(fig)
        scores=' | '.join(f"{r['profile']}: {r['mean_return']:.3f}" for r in data)
        if name=='world_comm':scores=' | '.join(f"{r['profile']}: pursuers {r['team_returns']['pursuers_mean']:.3f}, foragers {r['team_returns']['foragers_mean']:.3f}" for r in data)
        cards.append(f'<section><h2>{name}</h2><p>{cfg["description"]}</p><p>Left: reference ({"communication disabled" if name=="world_comm" else "random legal actions"}). Right: guided script. Matched seed and reset.</p><img src="{name}/comparison/replay.gif" width="100%"><p>Measured undiscounted episode returns — {scores}</p><img src="{name}/returns.png" width="850"><p><a href="{name}/guided/frame_000.png">Initial screenshot</a> · <a href="{name}/guided/frame_{cfg["steps"]//2:03d}.png">Middle screenshot</a> · <a href="{name}/guided/frame_{cfg["steps"]-1:03d}.png">Final screenshot</a> · <a href="{name}/results.json">Raw metrics and state schema</a></p></section>')
        if name=='overcooked_v2':cards[-1]=cards[-1].replace('</section>','<p><a href="overcooked_v2/guided/native_frame_055.png">Original JaxMARL renderer screenshot</a></p></section>')
    write(DEST/'results.json',rows)
    (DEST/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Native JaxMARL demonstration gallery</title><style>body{max-width:1400px;margin:30px auto;font:16px system-ui;background:#eef2f6;color:#172b40}section,aside{background:white;padding:22px;margin:24px 0}aside{background:#fff0c2}img{max-width:100%}</style><h1>Native JaxMARL environments — no training</h1><aside>Real simulator trajectories under policies. Overcooked guided controller uses privileged state. World Comm team rewards must be interpreted separately.</aside><p><a href="../index.html">Main gallery</a> · <a href="manifest.json">Provenance</a> · <a href="verification.json">Verification</a></p>'+''.join(cards)+'</html>')
    root=HERE.parent.parent;revision=subprocess.check_output(['git','-C',str(root/'third_party/JaxMARL'),'rev-parse','HEAD'],text=True).strip()
    write(DEST/'manifest.json',{'training_steps':0,'gradient_updates':0,'jaxmarl_revision':revision,'upstream':'https://github.com/bold-lab-ai/JaxMARL','scenarios':SCENARIOS,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256((HERE/'scene_renderer.py').read_bytes()).hexdigest(),'seed':SEED,'fps':FPS,'scope':'visual demonstrations, not CORA integration or learned baselines'})
    verify()
    update_inventory()


def update_inventory():
    files=[p for p in HERE.rglob('*') if p.is_file() and not any(v in p.relative_to(HERE).parts for v in ('legacy_simple_spread','.deps','.cache','__pycache__'))]
    static=json.loads((HERE/'results/results.json').read_text())
    adaptive=json.loads((HERE/'adaptive/results.json').read_text())
    native=json.loads((DEST/'results.json').read_text())
    write(HERE/'summary.json',{'label':'TRAIN','training_steps':0,'gradient_updates':0,'core_conditions':len({r['benchmark'] for r in static}),'native_environments':len({r['environment'] for r in native}),'static_rows':len(static),'live_adoption_rows':len(adaptive),'native_rollouts':len(native),'replay_gifs':sum(p.name=='replay.gif' for p in files),'traces':sum(p.suffix=='.npz' for p in files),'png_images':sum(p.suffix=='.png' for p in files),'pdf_plots':sum(p.suffix=='.pdf' for p in files),'legacy_excluded':'legacy_simple_spread/','all_outputs_within':'demonstrations/scripted_final_state'})
    path=HERE/'manifest.json';manifest=json.loads(path.read_text());manifest['demo_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')};manifest['current_gallery']='Simple World Comm, SMAX, OvercookedV2, JaxNav and four illustrated abstract domains';write(path,manifest)


def verify(required_all=False):
    checked=0
    if required_all:assert all((DEST/name/'results.json').exists() for name in SCENARIOS)
    for name,cfg in SCENARIOS.items():
        if not (DEST/name/'results.json').exists():continue
        for profile in ('reference','guided'):
            folder=DEST/name/profile;row=json.loads((folder/'metadata.json').read_text())
            with np.load(folder/'trace.npz') as trace:
                assert np.isfinite(trace['rewards']).all();assert np.allclose(trace['rewards'].sum(0),row['per_agent_return']);assert len(trace['actions'])==cfg['steps']
            for p in (folder/'replay.gif',DEST/name/'comparison/replay.gif'):
                with Image.open(p) as im:assert im.n_frames==cfg['steps']
            assert row['training_steps']==0;checked+=1
    write(DEST/'verification.json',{'validated_rollouts':checked,'training_steps':0,'finite_rewards':True,'trace_returns_match':True,'gif_frames_match':True})


from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=list(SCENARIOS));p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if args.verify_only:verify(required_all=True)
    else:
        for name in ([args.environment] if args.environment else SCENARIOS):
            print('Rendering native '+name,flush=True);run_case(name);jax.clear_caches()
        gallery()
