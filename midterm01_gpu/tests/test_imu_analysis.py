"""A saved generated fusion trace can be plotted without changing evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from track3.inertial import InertialState
from track3.imu import ImuSample
from track3.imu_parameters import resolve_imu_config


def test_short_imu_analysis_preserves_sources_and_excludes_bootstrap(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('imu_analysis',root/'scripts/analyze_imu.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cfg=json.loads((root/'config/track3.json').read_text())
    cfg['imu']=resolve_imu_config(cfg['imu'])
    state=InertialState(np.eye(4),3.,cfg['imu']);stamp=1727239790258245070
    rows=[]
    for idx in range(3):
        t=stamp+idx*10_000_000
        sample=ImuSample(t,t-100,np.zeros(3),np.array([0,0,9.80665]),'imu_link')
        prior=state.predict(t,[sample]);candidate=prior.copy();candidate[0,3]+=.01
        decision=state.commit(dict(T=candidate,scale=3.,accepted=idx!=2,reason='accepted' if idx!=2 else 'low_coverage'))
        rows.append(dict(idx=idx,timestamp_ns=t,inertial=decision['inertial'],
                         registration_accepted=idx!=2,registration_reason=decision['reason'],
                         prediction_ms=.1,measurement_update_ms=.1))
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'parameters.json').write_text(json.dumps(cfg))
    (tmp_path/'imu_sources.json').write_text(json.dumps(dict(extrinsic_status='SYNTHETIC_ONLY')))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'test_info.json').write_text(json.dumps(dict(status='success',mode='test',imu_used=True,frame_count=3,
        output_sha256={'observations.jsonl':digest(tmp_path/'observations.jsonl')})))
    before={p:digest(p) for p in tmp_path.iterdir()}
    assert mod.analyze(tmp_path)['status']=='success'
    metrics=json.loads((tmp_path/'analysis/imu/metrics.json').read_text())
    assert metrics['frames']==3 and metrics['map_accepted']==2
    assert all(digest(p)==h for p,h in before.items())
    for name in ('01_fusion','02_input_timing'):
        for ext in ('png','pdf'):assert (tmp_path/f'analysis/imu/{name}.{ext}').stat().st_size>1000
    with pytest.raises(FileExistsError):mod.analyze(tmp_path)
    # A changed observation cannot be silently replotted as original evidence.
    with (tmp_path/'observations.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='SHA'):mod.analyze(tmp_path)
