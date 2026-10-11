"""Orchestration checks with fake child completion; no experiments or training."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from cora.benchmarks import campaign


def test_full_plan_native_and_read_only(tmp_path):
    output=tmp_path/'absent';plan=campaign.plan(output)
    assert not output.exists()
    assert [j['name'] for j in plan['jobs']]==['synthetic','native-frozen','native-adaptive']
    assert plan['jobs'][0]['config']['benchmarks']==['synthetic']
    for job in plan['jobs'][1:]:
        assert job['config']['domains']==['world_comm','smax','overcooked_v2','jaxnav']
        assert set(job['config']['policies'])=={'random','ippo','mappo','iql','vdn','qmix'}
        assert job['check']['experiments_executed'] is False


def test_launcher_completion_resume_and_failure(tmp_path,monkeypatch):
    output=tmp_path/'campaign';job={'name':'test-suite','module':'cora.benchmarks.release','config':{'output':str(output/'test-suite')},'check':{'result_rows':1}}
    plan={'jobs':[job]};launched=[]
    class Child:
        def __init__(self,argv,**kwargs):
            launched.append(argv);self.stdout=['fake completed job\n']
            d=Path(job['config']['output']);d.mkdir(parents=True,exist_ok=True)
            (d/'manifest.json').write_text(json.dumps({'status':'complete'}))
            (d/'statistics.json').write_text('{}');(d/'index.html').write_text('fake test data')
        def wait(self):return 0
        def poll(self):return 0
    monkeypatch.setattr(campaign.subprocess,'Popen',Child)
    monkeypatch.setattr(campaign,'verify_suite',lambda job:[{'benchmark':'test_fixture','method':'no_op','seed':0,'budget':1}])
    campaign.execute(output,plan)
    assert '--run' in launched[0] and '--resume' in launched[0]
    assert (output/'results.csv').exists() and (output/'index.html').exists()
    assert json.loads((output/'verification.json').read_text())['result_rows']==1
    campaign.execute(output,plan)
    assert len(launched)==1
    monkeypatch.setattr(campaign,'signature',lambda p:'changed-source')
    with pytest.raises(ValueError,match='changed'):campaign.execute(output,plan)


def test_failed_child_does_not_publish_results(tmp_path,monkeypatch):
    job={'name':'failed','module':'cora.benchmarks.release','config':{'output':str(tmp_path/'failed')},'check':{}}
    class Child:
        stdout=['failure\n']
        def __init__(self,*args,**kwargs):pass
        def wait(self):return 1
        def poll(self):return 1
    monkeypatch.setattr(campaign.subprocess,'Popen',Child)
    campaign_dir=tmp_path/'campaign'
    with pytest.raises(RuntimeError,match='failed'):campaign.execute(campaign_dir,{'jobs':[job]})
    assert json.loads((campaign_dir/'campaign.json').read_text())['status']=='failed'
    assert not (campaign_dir/'results.csv').exists()


def test_suite_rejects_incomplete_results(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'complete'}))
    (tmp_path/'results.json').write_text('[]')
    job={'name':'native-frozen','config':{'output':str(tmp_path)},'check':{'result_rows':1}}
    with pytest.raises(RuntimeError,match='expected 1'):campaign.verify_suite(job)


def test_quantitative_native_plots_from_saved_rows(tmp_path):
    from cora.benchmarks.native_media import render_all
    from cora.benchmarks.statistics import aggregate
    rows=[{'benchmark':'test_fixture','method':'cora','seed':s,'budget':b,'simple_regret':1/(b+s+1),'heldout_effect_mae':.2,'selected_net_value':2.,'wall_seconds':.01,'harmful_reference':0,'abstention':1} for s in range(3) for b in (1,2)]
    config={'statistics':{'bootstrap_samples':10,'confidence':.95},'domains':['smax'],'media':{'fps':8}}
    (tmp_path/'statistics.json').write_text(json.dumps(aggregate(rows,config['statistics'])))
    render_all(tmp_path,rows,config)
    assert (tmp_path/'figures/test_fixture_metrics.png').exists()
    assert (tmp_path/'figures/test_fixture_metrics.pdf').exists()
    assert 'test_fixture_metrics.png' in (tmp_path/'index.html').read_text()
