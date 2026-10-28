"""Generated full-shaped diagnostics, immutable-source and refusal checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


def module():
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_monte_carlo.py'
    spec=importlib.util.spec_from_file_location('mc_analysis',script)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(run):
    rows=[]
    for i in range(3):
        cloud=dict(applied=i>0,reason='bootstrap' if i==0 else 'common_cloud_robust_likelihood')
        if i:cloud.update(ess=40.,max_weight=.1,valid_particles=60,selected_coverage=.9)
        rows.append(dict(idx=i,timestamp_ns=1727239790226594548+i*36_000_000,monte_carlo_ms=4.,prediction_ms=1.,
            monte_carlo=dict(kalman_used=False,posterior_recentered_by_refinement=False,cloud=cloud,
                             particles=64,resampled=i==2,measurement_accepted=True)))
    (run/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (run/'parameters.json').write_text(json.dumps(dict(monte_carlo=dict(ess_fraction=.5))))
    (run/'result_3.csv').write_text('idx,x,y,yaw\n0,0,0,0\n')
    (run/'poses.txt').write_text('generated evidence only\n')
    hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ('observations.jsonl','result_3.csv','poses.txt')}
    (run/'test_info.json').write_text(json.dumps(dict(status='success',mode='full',frame_count=3,
        prediction='gpu_monte_carlo',output_sha256=hashes)))


def test_particle_charts_are_read_only_with_source_manifest(tmp_path,monkeypatch):
    fixture(tmp_path);m=module();monkeypatch.setattr(m,'FRAME_COUNT',3)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    metrics=m.analyze(tmp_path)
    assert metrics['frames']==3 and metrics['resampling_frames']==1
    for name,data in before.items():assert (tmp_path/name).read_bytes()==data
    out=tmp_path/'analysis/monte_carlo'
    assert len(list(out.glob('*.png')))==3 and len(list(out.glob('*.pdf')))==3
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['source_sha256']['poses.txt']==hashlib.sha256(before['poses.txt']).hexdigest()
    assert not manifest['accuracy_claim']


@pytest.mark.parametrize('failure',['partial','sha','float_timestamp'])
def test_analysis_refuses_incomplete_or_unverifiable_inputs(tmp_path,monkeypatch,failure):
    fixture(tmp_path);m=module();monkeypatch.setattr(m,'FRAME_COUNT',3)
    p=tmp_path/'test_info.json';info=json.loads(p.read_text())
    if failure=='partial':info['mode']='test'
    elif failure=='sha':(tmp_path/'poses.txt').write_text('changed')
    else:
        obs=tmp_path/'observations.jsonl';rows=[json.loads(l) for l in obs.read_text().splitlines()]
        rows[0]['timestamp_ns']=float(rows[0]['timestamp_ns'])
        obs.write_text(''.join(json.dumps(r)+'\n' for r in rows));info['output_sha256']['observations.jsonl']=m.digest(obs)
    p.write_text(json.dumps(info))
    with pytest.raises(ValueError):m.analyze(tmp_path)
    assert not (tmp_path/'analysis').exists()


def test_v8_1_local_search_analysis_does_not_require_resampling_parameters(tmp_path,monkeypatch):
    fixture(tmp_path);m=module();monkeypatch.setattr(m,'FRAME_COUNT',3)
    p=tmp_path/'test_info.json';info=json.loads(p.read_text());info['monte_carlo_mode']='causal_local_search_not_calibrated_posterior';p.write_text(json.dumps(info))
    (tmp_path/'parameters.json').write_text(json.dumps(dict(monte_carlo=dict(particles=64))))
    metrics=m.analyze(tmp_path)
    assert metrics['mode']=='causal_local_search' and not metrics['scores_are_probabilities']
    assert (tmp_path/'analysis/monte_carlo/02_support_updates.png').exists()
    readme=(tmp_path/'analysis/monte_carlo/README.md').read_text()
    assert '沒有重採樣步驟' in readme and 'posterior' in readme


def test_bias_kalman_plots_are_separate_from_geometric_map_update(tmp_path,monkeypatch):
    fixture(tmp_path);m=module();monkeypatch.setattr(m,'FRAME_COUNT',3)
    p=tmp_path/'test_info.json';info=json.loads(p.read_text())
    info.update(monte_carlo_mode='causal_local_search_not_calibrated_posterior',kalman_used=True,
                kalman_scope='stationary_gyro_bias_only_no_pose_fusion',pose_kalman_used=False)
    obs=tmp_path/'observations.jsonl';rows=[json.loads(l) for l in obs.read_text().splitlines()]
    for i,row in enumerate(rows):
        row['inertial']=dict(prediction=dict(gyro_bias_filter=dict(updated=i>0,
            bias_radps=[0.,0.,.001*i],bias_std_radps=[.05/(i+1)]*3)))
    obs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    info['output_sha256']['observations.jsonl']=m.digest(obs);p.write_text(json.dumps(info))
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    metrics=m.analyze(tmp_path)
    assert metrics['kalman_used'] and not metrics['pose_kalman_used']
    assert metrics['gyro_bias_update_frames']==2
    assert (tmp_path/'analysis/monte_carlo/04_gyro_bias.png').exists()
    for name,data in before.items():assert (tmp_path/name).read_bytes()==data
