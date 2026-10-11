"""Validate and print the execution plan. Never invokes experiments."""
import json
import shlex
import contextlib
import io
import argparse
from pathlib import Path
from omegaconf import OmegaConf
with contextlib.redirect_stdout(io.StringIO()):
    from cora.benchmarks.runner import ROOT,validate_config,learner_settings
    from cora.benchmarks.release import preflight

jobs=[]
for name in ('bench_research.yaml','applications.yaml','applications_adaptive.yaml'):
    path=ROOT/'configs'/name
    config=OmegaConf.to_container(OmegaConf.load(path))
    if name=='bench_research.yaml':
        validate_config(config)
        for learner in ('ippo','mappo','iql','vdn','qmix'):learner_settings(learner,config)
        module='cora.benchmarks.runner';flags=[]
        check={'status':'configuration_valid','experiments_executed':False}
    else:
        check=preflight(config)
        module='cora.benchmarks.release';flags=['--run']
    argv=[str(ROOT/'.venv/bin/python'),'-m',module,'--config',str(path),*flags]
    jobs.append({'name':name,'check':check,'argv':argv,'command':shlex.join(argv)})
plan={'experiments_executed':False,'jobs':jobs,'publication_scope_complete':False,'limitation':'Published comparator reproductions remain unspecified/incompatible; runnable adaptations are explicitly labeled.'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,help='Write an exact JSON artifact; no experiments are executed')
args=parser.parse_args()
serialized=json.dumps(plan,indent=2)
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(serialized+'\n')
print(serialized)
