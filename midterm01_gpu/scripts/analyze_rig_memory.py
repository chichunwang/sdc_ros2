#!/usr/bin/env python3
"""彙整交錯基線的顯存／耗時／數值一致性；不把資料一致性當精度。"""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory):
    rows=json.loads((directory/'model_results.json').read_text())
    summary=[]
    for name in dict.fromkeys(r['variant'] for r in rows):
        chosen=[r for r in rows if r['variant']==name and r['status']=='ok']
        failures=[r for r in rows if r['variant']==name and r['status']!='ok']
        if not chosen:
            summary.append({'variant':name,'successful_samples':0,'failures':failures});continue
        paired=[]
        if name!='baseline':
            for row in chosen:
                neighbor=[r['frame_seconds'] for r in rows if r['status']=='ok' and r['variant']=='baseline'
                    and r['repeat']==row['repeat'] and r['frame']==row['frame'] and abs(r['phase']-row['phase'])==1]
                if len(neighbor)==2:paired.append(statistics.mean(neighbor))
        comparisons=[r['comparison'] for r in chosen]
        errors=[c['prediction'][cam]['depth']['max_absolute_difference'] for c in comparisons for cam in range(3)]
        mask_changes=[c['prediction'][cam]['mask']['changed_fraction'] for c in comparisons for cam in range(3)]
        result={'variant':name,'successful_samples':len(chosen),'failures':failures,
                'mean_frame_seconds':statistics.mean(r['frame_seconds'] for r in chosen),
                'median_frame_seconds':statistics.median(r['frame_seconds'] for r in chosen),
                'mean_inference_seconds':statistics.mean(r['inference_seconds'] for r in chosen),
                'mean_sampling_seconds':statistics.mean(r['sampling_seconds'] for r in chosen),
                'max_allocated_gib':max(r['peak_allocated_bytes'] for r in chosen)/2**30,
                'max_reserved_gib':max(r['peak_reserved_bytes'] for r in chosen)/2**30,
                'all_prediction_arrays_identical':all(x['identical'] for c in comparisons for camera in c['prediction'] for x in camera.values()),
                'all_selected_indices_identical':all(c['selected_indices_identical'] for c in comparisons),
                'all_ply_identical':all(c['ply_identical'] for c in comparisons),
                'all_labels_identical':all(c['labels_identical'] for c in comparisons),
                'max_depth_absolute_difference_raw':max(errors),'max_mask_changed_fraction':max(mask_changes)}
        if paired:
            result['paired_reference_mean_seconds']=statistics.mean(paired)
            result['paired_time_reduction_pct']=100*(1-result['mean_frame_seconds']/statistics.mean(paired))
        summary.append(result)
    chunks=json.loads((directory/'chunk_results.json').read_text())
    chunk_summary=[]
    for chunk in dict.fromkeys(r['chunk'] for r in chunks):
        chosen=[r for r in chunks if r['chunk']==chunk]
        chunk_summary.append({'chunk':chunk,'mean_warm_sampling_seconds':statistics.mean(r['warm_median_seconds'] for r in chosen),
            'max_allocated_gib':max(r['peak_allocated_bytes'] for r in chosen)/2**30,
            'max_reserved_gib':max(r['peak_reserved_bytes'] for r in chosen)/2**30,
            'all_indices_and_reports_identical':all(r['indices_and_report_identical'] for r in chosen)})
    value={'model':summary,'pca_chunk':chunk_summary,
           'scope':'Small real-data interleaved performance test. Array/cloud equality is not a ground-truth localization accuracy measure.'}
    (directory/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    summarize(p.parse_args().directory)
