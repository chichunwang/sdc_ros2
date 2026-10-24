"""Preflight must work with PLY/map/init only, without any ROS bag."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def test_inspection_without_bag_preserves_integer_clock(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/inspect_data.py'
    spec=importlib.util.spec_from_file_location('inspect_data_fixture',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=tmp_path/'data';data.mkdir();(data/'raw_ply').mkdir()
    stamp=1727239790258245070
    for idx in range(3):
        (data/'raw_ply'/f'{idx:06d}_ts_{stamp+idx*100_000_000}.ply').write_bytes(b'preflight only; contents not parsed')
    np.savetxt(data/'init_guess.txt',np.eye(4),delimiter=',')
    (data/'map.pcd').write_text('FIELDS x y z timestamp\nDATA ascii\n1 2 3 999\n')
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(version='SYNTHETIC_PREFLIGHT_ONLY',data_dir=str(data),
                                     ply_subdir='raw_ply',expected_frames=3)))
    run=tmp_path/'run';run.mkdir();(run/'source_manifest.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',[str(script),'--config',str(config),'--record-dir',str(run)])
    module.main()
    report=json.loads((run/'data_inspection.json').read_text())
    info=json.loads((run/'test_info.json').read_text())
    assert info['status']=='success' and info['frame_count']==0
    assert report['first_timestamp_ns']==stamp and report['last_timestamp_ns']==stamp+200_000_000
    assert report['localization_map_fields']==['x','y','z'] and report['imu_used'] is False
    assert not (data/'Track3.db3').exists()
    before=(run/'data_inspection.json').read_bytes()
    import pytest
    with pytest.raises(SystemExit):module.main()
    assert (run/'data_inspection.json').read_bytes()==before
