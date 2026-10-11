"""Refresh existing domain images from saved traces, without rerunning simulation."""
from run import HERE,CONFIG,DomainEnv,application_frames,save_frames,comparison,write,figures_and_gallery,verify
from types import SimpleNamespace
import json, shutil, hashlib
import numpy as np


def run():
    trace_paths=sorted(list((HERE/'results').rglob('*_trace.npz'))+list((HERE/'adaptive/results').rglob('*_trace.npz')))
    original={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in trace_paths}
    # Preserve superseded Simple Spread data, but remove it from current gallery/results.
    for section in ('results','visualizations'):
        old=HERE/section/'simple_spread'
        if old.exists():
            dest=HERE/'legacy_simple_spread'/section;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(old),str(dest))
    rows=json.loads((HERE/'results/results.json').read_text())
    oldrows=[r for r in rows if r['benchmark']=='simple_spread']
    if oldrows:write(HERE/'legacy_simple_spread/results.json',oldrows)
    rows=[r for r in rows if r['benchmark']!='simple_spread'];write(HERE/'results/results.json',rows)
    import csv
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (HERE/'results/results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for adaptive in (False,True):
        base=HERE/'adaptive' if adaptive else HERE
        for name in ('warehouse','rescue','wildfire','hospital'):
            env=DomainEnv(name,horizon=CONFIG['horizon'],disruption_step=CONFIG['disruption_step'])
            profiles=('status_quo','scripted_reconfigure') if adaptive else CONFIG['profiles']
            for profile in profiles:
                path=base/'results'/name/f"seed_{CONFIG['render_seed']}"/f'{profile}_trace.npz'
                with np.load(path) as arrays:
                    states=SimpleNamespace(**{k:arrays[k] for k in ('positions','queue','completed','failed','resource','step')})
                    trace={'states':states,'rewards':arrays['rewards']}
                    save_frames(application_frames(trace,env,arrays['assignments'],('LIVE ' if adaptive else '')+profile),base/'visualizations'/name/profile,CONFIG['fps'])
            comparison(base/'visualizations'/name/'status_quo/replay.gif',base/'visualizations'/name/('scripted_reconfigure' if adaptive else 'compound_reference')/'replay.gif',base/'visualizations'/name/'comparison')
    figures_and_gallery(rows);verify()
    manifest=json.loads((HERE/'manifest.json').read_text());manifest.update(rows=len(rows),benchmark_conditions=11,visual_refresh='Trace-preserving contextual render; Simple Spread superseded by native World Comm gallery');write(HERE/'manifest.json',manifest)
    preserved=all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h for p,h in original.items() if '/simple_spread/' not in p)
    assert preserved
    write(HERE/'visual_refresh_verification.json',{'saved_domain_traces_unchanged':preserved,'new_training_steps':0,'renderer':'scene_renderer.py','native_environments':'library_environments/','legacy':'legacy_simple_spread/','current_static_rows':len(rows)})
    print('Refreshed all domain and adaptation images; preserved trajectory data.')


if __name__=='__main__':run()
