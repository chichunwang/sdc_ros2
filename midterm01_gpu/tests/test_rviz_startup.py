import importlib.util
import json
from pathlib import Path
import sys


def test_gui_failure_is_recorded_before_any_localization(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/start_rviz.py'
    spec=importlib.util.spec_from_file_location('start_rviz',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=tmp_path/'generated_full_fixture';run.mkdir()
    monkeypatch.delenv('DISPLAY',raising=False)
    monkeypatch.setattr(sys,'argv',[str(script),'--run-dir',str(run)])
    assert module.main()==1
    status=json.loads((run/'rviz_status.json').read_text())
    info=json.loads((run/'test_info.json').read_text())
    assert status['status']==info['status']=='failed'
    assert info['frame_count']==0 and info['stage']=='rviz_startup'
    assert not (run/'poses.txt').exists()
