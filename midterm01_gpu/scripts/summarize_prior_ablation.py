#!/usr/bin/env python3
"""Summarize a frozen position-prior experiment, never call localization."""
from output_layout import loads_saved_json, saved_path

import argparse
import json
from pathlib import Path

import numpy as np

from prior_ablation import dump, rows
from three_camera_source import sha256


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=saved_path)
    parser.add_argument('--probe-directory',default='horizontal_probes_verified')
    args=parser.parse_args();root=args.experiment
    block_reports=loads_saved_json((root/'fixed_time_block_validation.json').read_text())
    summaries=[]
    for r in block_reports:
        for split in ('design','holdout'):
            selected=[x for x in r['blocks'] if x['split']==split]
            count=sum(x['validation_rows'] for x in selected)
            summaries.append({'track':r['track'],'split':split,'blocks':len(selected),'heldout_measurements':count,
                'heading_rts_prediction_rmse_m':float(np.sqrt(sum(x['heading_rts_prediction_rmse_m']**2*x['validation_rows']for x in selected)/count)),
                'motion_rts_prediction_rmse_m':float(np.sqrt(sum(x['motion_rts_prediction_rmse_m']**2*x['validation_rows']for x in selected)/count)),
                'motion_rts_better_blocks':sum(x['motion_rts_prediction_rmse_m']<x['heading_rts_prediction_rmse_m']for x in selected)})
    pairs=loads_saved_json((root/'ablation_runs.json').read_text());paired=[]
    for pair in pairs:
        a,b=[Path(pair['runs'][key])for key in ('heading_rts','motion_rts')]
        pa,pb=[np.loadtxt(p/'result.txt').reshape(-1,3,4) for p in (a,b)]
        checks=[loads_saved_json((p/'validation.json').read_text()) for p in (a,b)]
        if any(not c['output_complete_and_finite'] or c['validation_errors'] or c['returncode'] for c in checks):
            raise ValueError('Failed output in paired experiment')
        qa,qb=rows(a/'sensor_prior.csv'),rows(b/'sensor_prior.csv')
        if len(qa)!=len(qb) or any(x[k]!=y[k]for x,y in zip(qa,qb)for k in ('timestamp_ns','z','qx','qy','qz','qw')):
            raise ValueError('Expected frozen original attitude and Z')
        xy=np.linalg.norm(pa[:,:2,3]-pb[:,:2,3],axis=1)
        yaw_a,yaw_b=np.arctan2(pa[:,1,0],pa[:,0,0]),np.arctan2(pb[:,1,0],pb[:,0,0])
        dyaw=np.degrees(np.arctan2(np.sin(yaw_b-yaw_a),np.cos(yaw_b-yaw_a)))
        paired.append({'track':pair['track'],'class':pair['class'],'split':pair['split'],'start':pair['start'],
            'frames_per_method':len(pa),'xy_version_difference_rms_m':rmse(xy),'yaw_version_difference_rms_deg':rmse(dyaw),
            'z_version_difference_rms_m':rmse(pb[:,2,3]-pa[:,2,3]),
            'heading_map_accepted':checks[0]['guidance_not_accuracy']['map_accepted_frames'],
            'motion_map_accepted':checks[1]['guidance_not_accuracy']['map_accepted_frames'],
            'identical_input_attitude_z_timestamp_strings':True,
            'same_binary':(a/'binary_sha256.txt').read_bytes()==(b/'binary_sha256.txt').read_bytes(),
            'same_source':(a/'source_input.json').read_bytes()==(b/'source_input.json').read_bytes(),
            'output_sha256':{key:{f:sha256(Path(path)/f)for f in ('result.txt','diagnostic.csv','validation.json')}
                              for key,path in pair['runs'].items()}})
    probes=loads_saved_json((root/args.probe_directory/'report.json').read_text())
    generic=loads_saved_json((root/'generic_smoke.json').read_text()) if (root/'generic_smoke.json').is_file() else []
    holdout=[r for r in summaries if r['split']=='holdout']
    eligible=(len(holdout)==2 and all(r['motion_rts_prediction_rmse_m']<r['heading_rts_prediction_rmse_m']for r in holdout)
              and all(p['same_binary']and p['same_source']for p in paired))
    result={'position_validation':summaries,'short_pairs':paired,'total_poses':sum(2*p['frames_per_method']for p in paired),
        'generic_smoke_poses':sum(p['poses']for p in generic),
        'total_short_poses_including_generic_smoke':sum(2*p['frames_per_method']for p in paired)+sum(p['poses']for p in generic),
        'new_full_localization_started':False,
        'image_pairs':sum(len(p['motions'])for p in probes),
        'image_pair_rotation_gate_pass':sum(m['relative_rotation_candidate']for p in probes for m in p['motions']),
        'three_view_rotation_gate_pass':sum(p['rotation_observation_candidate']for p in probes),
        'horizontal_map_structure_gate_pass':sum(p['horizontal_map']['structurally_supported']for p in probes),
        'joint_fusion_allowed':False,
        'decision':{'position_only_motion_rts':('Ready for user-run full candidate evaluation; not promoted to best' if eligible
                    else 'Evidence mixed or incomplete; inspect reports before full evaluation'),
                    'visual_map_joint':'Do not fuse the current probes; observation/frame requirements failed'},
        'limits':['Blocked GNSS prediction is not ground-truth pose accuracy; shared GNSS bias remains unobservable.',
                  'Short carry windows reset Z history and do not prove full-run score improvements.',
                  'No noise/process parameters were tuned; no new full localization or submission was started.',
                  'Geometric gates are conservative diagnostics, not calibrated confidence or proof of no useful visual information.']}
    dump(root/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
