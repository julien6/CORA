"""Explicit execution and saved-data analysis for native JaxMARL experiments.

Default command is a read-only preflight. Only --run executes experiments.
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import jax
import numpy as np
from omegaconf import OmegaConf
from cora.benchmarks.native_env import make_native_env,NativeRandomPolicy,NativeIntervention
from cora.benchmarks.native_trial import NativeTrial
from cora.benchmarks.learners import SOURCES,FrozenPolicy,train,load_policy
from cora.benchmarks.methods import METHODS,UNRESOLVED,run_method
from cora.benchmarks.runner import ROOT,learner_settings,evaluate,write_json
from cora.benchmarks.simulation import save_trace
from cora.benchmarks.statistics import aggregate


def preflight(config):
    if config['protocol'] not in ('frozen','retrain_between_decisions'):
        raise ValueError('Unknown continuation protocol')
    if not config['seeds'] or len(set(config['seeds']))!=len(config['seeds']) or any(type(s)!=int or not 0<=s<2**32 for s in config['seeds']):
        raise ValueError('Unique uint32 seeds required')
    if config['budgets']!=sorted(set(config['budgets'])) or not config['budgets'] or min(config['budgets'])<1:
        raise ValueError('Increasing positive intervention budgets required')
    if config['rounds']<1 or (config['protocol']=='frozen' and config['rounds']!=1):
        raise ValueError('Frozen protocol requires one round')
    if not config['domains'] or not config['policies'] or not config['methods']:
        raise ValueError('Nonempty domain/policy/method lists required')
    if any(len(set(config[k]))!=len(config[k]) for k in ('domains','policies','methods')):
        raise ValueError('Duplicate conditions would overwrite experiment records')
    if config['media']['fps']<1 or (config['media']['enabled'] and config['media']['seed'] not in config['seeds']):
        raise ValueError('Positive frame rate and an included media seed required')
    if config['statistics']['bootstrap_samples']<1 or not 0<config['statistics']['confidence']<1:
        raise ValueError('Invalid statistical settings')
    for name in config['domains']:
        make_native_env(name,config['environment'])
    for name in config['policies']:
        if name not in ('random',*SOURCES):raise ValueError('Unknown policy')
        if name in SOURCES:learner_settings(name,config)
    for method in config['methods']:
        if method not in METHODS or method in UNRESOLVED:raise ValueError('Unknown/unavailable method')
        if method in ('cora_no_constraints','cora_conditional'):
            raise ValueError('Application constraints stay enforced; certified bounds unavailable in these domains')
    if not 1<=config['evaluation']['horizon']<=config['environment']['horizon'] or config['evaluation']['oracle_repeats']<2:
        raise ValueError('Invalid oracle/evaluation horizon')
    numerical=[config['evaluation']['gamma'],config['evaluation']['swap_cost'],config['evaluation']['diagnostic_scale']]
    if not np.isfinite(numerical).all() or not 0<=numerical[0]<=1 or min(numerical[1:])<0:
        raise ValueError('Invalid evaluation parameters')
    learned=sum(p in SOURCES for p in config['policies'])
    contexts=len(config['seeds'])*len(config['domains'])*len(config['policies'])*config['rounds']*len(config['methods'])
    return {'status':'ready_for_explicit_execution','result_rows':contexts*len(config['budgets']),
            'upper_bound_training_environment_steps':len(config['seeds'])*len(config['domains'])*learned*config['rounds']*len(config['methods'])*config['training']['total_environment_steps'],
            'protocol':config['protocol'],'policy_transfer':'mechanism reuse only; no zero-shot learned-weight claim',
            'external_comparators':'adapted/internal methods; faithful published reproductions remain unavailable',
            'backend':'native_jaxmarl','environments':{name:make_native_env(name,config['environment']).protocol_metadata() for name in config['domains']},'experiments_executed':False}


def run(config,resume=False):
    preflight(config)
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/cora').rglob('*.py')}
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text())
        if not resume or prior['config_sha256']!=signature or prior['source_sha256']!=source:
            raise ValueError('Output exists or configuration/source differs; use a fresh directory')
    manifest={'status':'running','config':config,'config_sha256':signature,'source_sha256':source,'hardware':platform.platform(),'devices':[str(x) for x in jax.devices()],'dependencies':{name:version(name) for name in ('jax','jaxlib','jaxmarl','flax','numpy','optax','pillow','imageio')},'external_reproductions_complete':False,'backend':'native_jaxmarl','environment_protocols':{d:make_native_env(d,config['environment']).protocol_metadata() for d in config['domains']}}
    write_json(manifest_path,manifest)
    try:
        for seed in config['seeds']:
            for domain in config['domains']:
                for name in config['policies']:
                    for method in config['methods']:
                        env=make_native_env(domain,config['environment'])
                        for round_index in range(config['rounds']):
                            location=out/'learners'/domain/name/f'seed_{seed}'/('shared_round0' if round_index==0 else f'{method}_round{round_index}')
                            if name=='random':policy=NativeRandomPolicy(tuple(env.agents),tuple(env.action_space(a).n for a in env.agents))
                            elif (location/'training.json').exists():policy=load_policy(location,env)
                            else:policy,_=train(name,env,(seed+round_index)%2**32,learner_settings(name,config),location)
                            directory=out/'runs'/f'{domain}_{name}_round{round_index}'/f'seed_{seed}'
                            path=directory/f'{method}.json'
                            if resume and path.exists():
                                saved=json.loads(path.read_text())
                                env.initial=env.apply_organization(env.initial,NativeIntervention(tuple(saved['selected_bits'])))
                                continue
                            trial=NativeTrial(env,policy,config['evaluation'],seed,round_index)
                            if method=='oracle':
                                selected=int(np.argmax(trial.truth-trial.public.costs))
                                decisions=[{'budget':b,'selected_index':selected,'predictions':trial.truth.tolist(),'diagnostic_radius':None,'rank':0,'certified':False,'queries':0,'experiment_cost':0,'wall_seconds':0} for b in config['budgets']]
                                logs=[]
                            else:decisions,logs=run_method(method,trial.public,trial.observe,config['budgets'],seed+round_index,config['estimation'])
                            rows=evaluate(trial,method,decisions,logs,seed,config)
                            selected=decisions[-1]['selected_index']
                            bits=list(trial.options[selected].bits)
                            # Each method has its own adoption/policy sequence after round zero.
                            record={'rows':rows,'decisions':decisions,'experiments':logs,'selected_bits':bits,'policy_version':policy.version,'organization_before':np.asarray(env.initial.assignments).tolist(),'protocol':config['protocol']}
                            write_json(directory/f'{method}_evaluation_oracle.json',{'values':trial.truth.tolist(),'standard_errors':trial.oracle_standard_error.tolist(),'metadata':trial.metadata,'privileged':True})
                            if config['media']['enabled'] and seed==config['media']['seed']:
                                trace=trial.trace(selected,900000)
                                org=env.apply_organization(env.initial,trial.options[selected])
                                save_trace(directory/f'{method}_trace.npz',trace,org.assignments)
                            # Completion marker follows all required raw artifacts.
                            write_json(path,record)
                            env.initial=env.apply_organization(env.initial,trial.options[selected])
                            del trial,policy
                            jax.clear_caches()
        result=postprocess(out)
        manifest.update(status='complete',result_rows=result['result_rows']);write_json(manifest_path,manifest)
    except Exception as error:
        manifest.update(status='failed',failure=repr(error));write_json(manifest_path,manifest);raise
    return result


def postprocess(out):
    config=json.loads((out/'manifest.json').read_text())['config']
    rows=[]
    for path in sorted((out/'runs').glob('*/*/*.json')):
        data=json.loads(path.read_text())
        rows.extend(data.get('rows',[]))
    write_json(out/'results.json',rows)
    write_json(out/'statistics.json',aggregate(rows,config['statistics']))
    if set(config['domains']) <= {'warehouse','rescue','wildfire','hospital'}:
        from cora.benchmarks.domain_media import render_all  # Historical data only.
    else:
        from cora.benchmarks.native_media import render_all
    render_all(out,rows,config)
    return {'result_rows':len(rows),'output':str(out)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/applications.yaml')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    config=OmegaConf.to_container(OmegaConf.load(args.config))
    result=postprocess(Path(config['output'])) if args.analyze_only else run(config,args.resume) if args.run else preflight(config)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
