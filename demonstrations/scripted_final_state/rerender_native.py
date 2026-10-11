"""Refresh World Comm and SMAX drawings directly from their recorded native states."""
from library_demo import DEST,SCENARIOS,SEED,FPS,world_frame,render_smax,gallery
from run import save_frames,comparison
from scene_renderer import text
from types import SimpleNamespace
import json
import numpy as np
import jaxmarl
from PIL import ImageDraw


def run():
    for name in ('world_comm','smax'):
        if name=='world_comm':env=jaxmarl.make('MPE_simple_world_comm_v3')
        else:
            from jaxmarl.environments.smax import map_name_to_scenario
            env=jaxmarl.make('HeuristicEnemySMAX',scenario=map_name_to_scenario('3m'))
        for profile in ('reference','guided'):
            folder=DEST/name/profile;meta=json.loads((folder/'metadata.json').read_text());frames=[]
            with np.load(folder/'trace.npz') as arrays:
                mapping={p:f'state_{i}' for i,p in enumerate(meta['state_leaf_paths'])}
                for t in range(SCENARIOS[name]['steps']):
                    prefix='.state.' if name=='smax' else '.'
                    fields={p[len(prefix):]:arrays[k][t] for p,k in mapping.items() if p.startswith(prefix)}
                    state=SimpleNamespace(**fields);returns=arrays['rewards'][:t].sum(0)
                    im=render_smax(env,SimpleNamespace(state=state),t,profile,returns) if name=='smax' else world_frame(env,state,t,profile,returns)
                    if t and arrays['dones'][t-1]:
                        d=ImageDraw.Draw(im);d.rounded_rectangle((32,542,347,573),radius=4,fill='#172b40');text(d,(42,547),'EPISODE ENDED / final state frozen','white',14)
                    frames.append(im)
            save_frames(frames,folder,FPS)
        comparison(DEST/name/'reference/replay.gif',DEST/name/'guided/replay.gif',DEST/name/'comparison')
    gallery()


if __name__=='__main__':run()
