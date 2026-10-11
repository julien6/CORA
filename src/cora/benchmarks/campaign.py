"""One-command, sequential, resumable research campaign. --dry-run executes nothing."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from omegaconf import OmegaConf
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()):
    from cora.benchmarks.runner import ROOT,validate_config,write_json
    from cora.benchmarks.release import preflight

SUITES=(('synthetic','bench_research.yaml','cora.benchmarks.runner'),('native-frozen','applications.yaml','cora.benchmarks.release'),('native-adaptive','applications_adaptive.yaml','cora.benchmarks.release'))


def plan(output):
    jobs=[]
    for name,filename,module in SUITES:
        config=OmegaConf.to_container(OmegaConf.load(ROOT/'configs'/filename))
        config['output']=str(output/name)
        if name=='synthetic':
            validate_config(config)
            check={'experiments_executed':False,'result_rows':len(config['seeds'])*len(config['synthetic']['scenarios'])*len(config['methods'])*len(config['budgets'])}
        else:check=preflight(config)
        jobs.append({'name':name,'config':config,'module':module,'check':check})
    return {'jobs':jobs,'parallel_jobs':1,'training_is_executed_only_when_launched':True,'scope':'All configured runnable methods and MARL learners; not unspecified/unavailable published reproductions.'}


def signature(execution_plan):
    return hashlib.sha256(json.dumps({'plan':execution_plan,'source':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/cora').rglob('*.py'))}},sort_keys=True).encode()).hexdigest()


def verify_suite(job):
    directory=Path(job['config']['output']);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('status')!='complete':raise RuntimeError('Suite did not complete: '+job['name'])
    if job['name']=='synthetic':rows=[json.loads(s) for s in (directory/'results.jsonl').read_text().splitlines() if s.strip()]
    else:rows=json.loads((directory/'results.json').read_text())
    expected=job['check']['result_rows']
    if len(rows)!=expected:raise RuntimeError(f"{job['name']}: expected {expected} result rows, found {len(rows)}")
    keys=[(r['benchmark'],r['method'],r['seed'],r['budget']) for r in rows]
    if len(set(keys))!=len(keys):raise RuntimeError('Duplicate result conditions')
    for file in ('statistics.json','index.html'):
        if not (directory/file).exists():raise RuntimeError('Missing suite artifact: '+file)
    expected_conditions=set()
    if job['name']=='synthetic':
        c=job['config']
        expected_conditions={(f'synthetic_{scenario}',m,s,b) for scenario in c['synthetic']['scenarios'] for m in c['methods'] for s in c['seeds'] for b in c['budgets']}
        if set(keys)!=expected_conditions:raise RuntimeError('Missing or unexpected synthetic result conditions')
    if job['name']!='synthetic':
        c=job['config']
        expected_conditions={(f'{d}_{p}_round{r}',m,s,b) for d in c['domains'] for p in c['policies'] for r in range(c['rounds']) for m in c['methods'] for s in c['seeds'] for b in c['budgets']}
        if set(keys)!=expected_conditions:raise RuntimeError('Missing or unexpected native result conditions')
        if c['media']['enabled']:
            from PIL import Image
            for context,method,seed in sorted({(c,m,s) for c,m,s,b in expected_conditions}):
                if seed!=c['media']['seed']:continue
                folder=directory/'media'/context/f'seed_{seed}'/method
                if not (directory/'runs'/context/f'seed_{seed}'/f'{method}_trace.npz').exists():raise RuntimeError('Missing saved native trace')
                with Image.open(folder/'replay.gif') as im:
                    if im.n_frames!=c['evaluation']['horizon']:raise RuntimeError('Incomplete native animation')
                for t in {0,c['evaluation']['horizon']//2,c['evaluation']['horizon']-1}:
                    if not (folder/f'frame_{t:04d}.png').exists():raise RuntimeError('Missing native screenshot')
    return rows


def collect(output,jobs):
    rows=[];links=[]
    for job in jobs:
        values=verify_suite(job)
        rows.extend({'suite':job['name'],**r} for r in values)
        links.append(f'<li><a href="{job["name"]}/index.html">{html.escape(job["name"])}: {len(values)} result rows, figures and replays</a></li>')
    write_json(output/'results.json',rows)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    # Keep each estimand/protocol separate rather than pooling unlike results.
    write_json(output/'statistics.json',{j['name']:json.loads((Path(j['config']['output'])/'statistics.json').read_text()) for j in jobs})
    write_json(output/'verification.json',{'complete':True,'result_rows':len(rows),'suites':[j['name'] for j in jobs],'coverage_checked':True,'native_trace_and_media_coverage_checked':True})
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>CORA baseline campaign</title><h1>CORA baseline results</h1><p>Actual computational experiments. Synthetic, frozen and adaptive estimands are reported separately. These are configured training budgets, not certificates of convergence or safety. Published comparator adaptations remain labeled in the suite manifests.</p><ul>'+''.join(links)+'</ul><p><a href="results.csv">Combined CSV</a> · <a href="results.json">Raw results</a> · <a href="statistics.json">Suite statistics</a> · <a href="verification.json">Coverage verification</a> · <a href="campaign.json">Execution manifest</a></p></html>')
    return len(rows)


def execute(output,execution_plan,analyze_only=False):
    digest=signature(execution_plan);manifest_path=output/'campaign.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if not analyze_only and manifest['signature']!=digest:raise ValueError('Source/configuration changed; choose a new --output directory')
    elif analyze_only:raise ValueError('Analysis requires an existing campaign')
    else:manifest={'signature':digest,'plan':execution_plan,'status':'prepared','completed_suites':[]}
    # Single writer: prevent two commands from training into the same outputs.
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'.campaign.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This campaign is already running')
        write_json(manifest_path,manifest)
        for job in manifest['plan']['jobs']:
            directory=Path(job['config']['output']);config_path=output/'configs'/f'{job["name"]}.yaml';config_path.parent.mkdir(exist_ok=True)
            OmegaConf.save(OmegaConf.create(job['config']),config_path)
            argv=[sys.executable,'-u','-m',job['module'],'--config',str(config_path)]
            if analyze_only:argv.append('--analyze-only')
            else:
                if job['module'].endswith('release'):argv.append('--run')
                argv.append('--resume')
            if not analyze_only and job['name'] in manifest['completed_suites']:
                verify_suite(job);continue
            manifest.update(status='running',active_suite=job['name']);write_json(manifest_path,manifest)
            print(f"\nRunning {job['name']}; log: {output/'logs'/(job['name']+'.log')}",flush=True)
            logs=output/'logs';logs.mkdir(exist_ok=True)
            child=None
            try:
                with (logs/(job['name']+'.log')).open('a') as log:
                    child=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
                    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    if child.wait()!=0:raise RuntimeError(f"Suite {job['name']} failed; inspect its log and rerun the same command to resume")
                verify_suite(job)
                if job['name'] not in manifest['completed_suites']:manifest['completed_suites'].append(job['name'])
                write_json(manifest_path,manifest)
            except BaseException as error:
                if child is not None and child.poll() is None:child.terminate();child.wait()
                manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));write_json(manifest_path,manifest);raise
        count=collect(output,manifest['plan']['jobs']);manifest.update(status='complete',active_suite=None,result_rows=count);write_json(manifest_path,manifest)
        print(f'Complete: {output/"index.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/all-baselines')
    parser.add_argument('--dry-run',action='store_true',help='Validate and print the full plan without creating outputs or running experiments')
    parser.add_argument('--analyze-only',action='store_true',help='Regenerate artifacts from an existing campaign without training')
    args=parser.parse_args();output=args.output.resolve()
    with contextlib.redirect_stdout(io.StringIO()):execution_plan=plan(output)
    if args.dry_run:print(json.dumps(execution_plan,indent=2));return
    execute(output,execution_plan,args.analyze_only)


if __name__=='__main__':main()
