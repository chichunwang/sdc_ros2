import csv
import importlib.util
import json
from pathlib import Path
import numpy as np


def test_full_plot_contract_and_source_protection(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_run.py'
    spec=importlib.util.spec_from_file_location('analyze_run',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    run=tmp_path/'SYNTHETIC_FULL_FORMAT_ONLY';run.mkdir()
    (run/'test_info.json').write_text(json.dumps(dict(status='success',mode='full',frame_count=3142,test_type='SYNTHETIC format only; no localization or competition score')))
    poses=[]
    with (run/'result_3.csv').open('w',newline='') as out,(run/'timings.csv').open('w',newline='') as times:
        w=csv.writer(out);w.writerow(['idx','x','y','yaw']);tw=csv.writer(times);tw.writerow(mod.TIMING_FIELDS)
        for i in range(3142):
            angle=i*.001;T=np.eye(4);T[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
            T[:3,3]=[np.cos(angle)*10,np.sin(angle)*10,0];poses.append(T[:3].ravel())
            w.writerow([i,T[0,3],T[1,3],angle]);tw.writerow([i,i,1727239790258245070+i*36000000,'SYNTHETIC',1,2,4,1,3])
    np.savetxt(run/'poses.txt',poses)
    sources={p:mod.sha256(p) for p in run.iterdir()}
    result=mod.plot_run(run);assert result['status']=='success'
    for name in mod.PLOT_STEMS:
        for ext in ['png','pdf']:assert (run/'analysis'/f'{name}.{ext}').stat().st_size>1000
    assert all(mod.sha256(p)==h for p,h in sources.items())
    try:mod.plot_run(run)
    except FileExistsError:pass
    else:raise AssertionError('Existing complete analysis was overwritten')
    partial=tmp_path/'partial';partial.mkdir();(partial/'test_info.json').write_text(json.dumps(dict(status='failed',mode='test',frame_count=3)))
    try:mod.validate_inputs(partial,np)
    except ValueError:pass
    else:raise AssertionError('Partial run accepted')


def test_full_memory_chart_is_optional_and_read_only(tmp_path):
    # Generate the same full-format fixture through the existing contract test.
    test_full_plot_contract_and_source_protection(tmp_path)
    run=tmp_path/'SYNTHETIC_FULL_FORMAT_ONLY'
    import shutil
    shutil.rmtree(run/'analysis')  # Only this test's generated plot artifacts.
    with (run/'gpu_memory.jsonl').open('w') as f:
        for i in range(3142):
            f.write(json.dumps(dict(stage='frame_end',idx=i,pool_used_bytes=100*1024**2,
                pool_reserved_bytes=300*1024**2,pool_limit_bytes=4096*1024**2,
                device_free_bytes=6000*1024**2,device_total_bytes=8151*1024**2))+'\n')
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_run.py'
    spec=importlib.util.spec_from_file_location('analyze_run_memory',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    before={p.name:mod.sha256(p) for p in run.iterdir() if p.is_file()}
    assert mod.plot_run(run)['status']=='success'
    for ext in ['png','pdf']:assert (run/'analysis'/f'04_gpu_memory.{ext}').stat().st_size>1000
    assert all(mod.sha256(run/name)==sha for name,sha in before.items())
