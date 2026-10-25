"""CPU-only checks of explicit trial config creation and snapshot integrity."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track3 import imu_parameters


SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


def test_python_extrinsic_resolves_changed_parameters_and_preserves_input(monkeypatch):
    monkeypatch.setattr(imu_parameters,'IMU_RPY_DEG',(10.,20.,30.))
    monkeypatch.setattr(imu_parameters,'IMU_XYZ_M',(1.,-2.,.5))
    original={'extrinsic_source':'python','other_setting':42}
    resolved=imu_parameters.resolve_imu_config(original)
    T=np.asarray(resolved['T_ply_imu'])
    np.testing.assert_allclose(T[:3,:3],Rotation.from_euler('xyz',[10,20,30],degrees=True).as_matrix())
    np.testing.assert_array_equal(T[:3,3],[1,-2,.5])
    assert resolved['extrinsic_status']=='assumed_python_unverified_not_calibrated'
    assert resolved['other_setting']==42
    assert original=={'extrinsic_source':'python','other_setting':42}
    # Reproducible legacy explicit matrices do not pick up current Python edits.
    explicit={'T_ply_imu':np.eye(4).tolist(),'extrinsic_status':'explicit fixture'}
    assert imu_parameters.resolve_imu_config(explicit)==explicit


def test_python_extrinsic_rejects_nonfinite_parameters(monkeypatch):
    monkeypatch.setattr(imu_parameters,'IMU_RPY_DEG',(float('nan'),0.,0.))
    with pytest.raises(ValueError,match='有限數值'):
        imu_parameters.resolve_imu_config({'extrinsic_source':'python'})


def script(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_config_is_explicit_and_does_not_overwrite(tmp_path,monkeypatch):
    module=script('configure_imu')
    base=tmp_path/'base.json';output=tmp_path/'trial.json'
    cfg={'version':'synthetic','registration':{'unchanged':True},
         'imu':{'noise_source':'unchanged','extrinsic_status':'old'}}
    base.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys,'argv',['configure_imu','--base',str(base),'--rpy-deg','10','20','30',
                                   '--xyz-m','1','-2','0.5','--output',str(output)])
    module.main()
    result=json.loads(output.read_text())
    T=np.asarray(result['imu']['T_ply_imu'])
    np.testing.assert_allclose(T[:3,:3],Rotation.from_euler('xyz',[10,20,30],degrees=True).as_matrix())
    np.testing.assert_array_equal(T[:3,3],[1,-2,.5])
    assert result['imu']['extrinsic_status']=='assumed_trial_unverified_not_calibrated'
    assert result['registration']==cfg['registration']
    assert result['imu']['noise_source']=='unchanged'
    original=output.read_bytes()
    with pytest.raises(FileExistsError):module.main()
    assert output.read_bytes()==original and json.loads(base.read_text())==cfg


@pytest.mark.parametrize('bad',['nan','inf'])
def test_trial_rejects_nonfinite_before_writing(tmp_path,monkeypatch,bad):
    module=script('configure_imu');output=tmp_path/'trial.json'
    monkeypatch.setattr(sys,'argv',['configure_imu','--rpy-deg',bad,'0','0',
                                   '--xyz-m','0','0','0','--output',str(output)])
    with pytest.raises(SystemExit) as exc:module.main()
    assert exc.value.code==2 and not output.exists()


@pytest.mark.parametrize('custom',[False,True])
def test_create_run_freezes_selected_config_with_source_hash(tmp_path,monkeypatch,capsys,custom):
    module=script('create_run')
    root=tmp_path/'project';(root/'scripts').mkdir(parents=True)
    (root/'config').mkdir()
    default=root/'config/track3.json';default.write_text('{"version":"default"}\n')
    external=tmp_path/'external config.json';external.write_text('{"version":"trial"}\n')
    monkeypatch.setattr(module,'__file__',str(root/'scripts/create_run.py'))
    monkeypatch.setattr(module.subprocess,'run',lambda args,**kwargs:SimpleNamespace(
        stdout='synthetic_commit\n' if 'rev-parse' in args else ''))
    argv=['create_run','--kind','test','--track','00']
    if custom:argv+=['--config',str(external)]
    monkeypatch.setattr(sys,'argv',argv)
    module.main()
    run=Path(capsys.readouterr().out.strip())
    manifest=json.loads((run/'source_manifest.json').read_text())
    assert (run/'source/config/track3.json').read_bytes()==default.read_bytes()
    if custom:
        frozen=run/'source/config/selected_config.json'
        sha=hashlib.sha256(external.read_bytes()).hexdigest()
        assert frozen.read_bytes()==external.read_bytes()
        assert manifest['sources']['config/selected_config.json']==sha
        assert manifest['selected_config']==dict(source_path=str(external),source_sha256=sha,
            snapshot_path='config/selected_config.json',snapshot_sha256=sha)
        external.write_text('{"version":"changed_after_snapshot"}\n')
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==sha
    else:
        assert 'selected_config' not in manifest
        assert not (run/'source/config/selected_config.json').exists()
