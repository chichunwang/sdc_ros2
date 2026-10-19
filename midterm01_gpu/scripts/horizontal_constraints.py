#!/usr/bin/env python3
"""Independent image-motion and horizontal-map probes, with explicit fusion gates."""
from output_layout import loads_saved_json, saved_path

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from calibrated_rig_geometry import OPTICAL_TO_FLU, rectify
from prior_ablation import dump, rows
from three_camera_source import sha256, camera_headers
from three_camera_calibration import load_ply, load_binary_map, SurfaceMap


def sampson_pixels(essential, a, b, k):
    f = np.linalg.inv(k).T @ essential @ np.linalg.inv(k)
    aa, bb = np.column_stack((a,np.ones(len(a)))), np.column_stack((b,np.ones(len(b))))
    l2, l1 = aa@f.T, bb@f
    return np.abs(np.sum(bb*l2,axis=1))/np.sqrt(np.maximum(np.sum(l1[:,:2]**2+l2[:,:2]**2,axis=1),1e-20))


def refined_pose(essential, a, b, k, mask):
    """Refine the five-point proposal on fitted inliers only, never holdout."""
    _,rotation,translation,_=cv2.recoverPose(essential,a,b,k,mask=mask.copy())
    use=mask.ravel()>0
    aa=np.column_stack((a[use],np.ones(use.sum())))
    bb=np.column_stack((b[use],np.ones(use.sum())))
    inverse=np.linalg.inv(k)
    def essential_from(x):
        t=x[3:]/max(np.linalg.norm(x[3:]),1e-12)
        skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
        return skew@Rotation.from_rotvec(x[:3]).as_matrix()
    def residual(x):
        f=inverse.T@essential_from(x)@inverse
        l2,l1=aa@f.T,bb@f
        epi=np.sum(bb*l2,axis=1)/np.sqrt(np.maximum(np.sum(l1[:,:2]**2+l2[:,:2]**2,axis=1),1e-20))
        return np.r_[epi,10*(np.linalg.norm(x[3:])-1)]
    initial=np.r_[Rotation.from_matrix(rotation).as_rotvec(),translation.ravel()]
    solution=least_squares(residual,initial,loss='soft_l1',f_scale=1.,max_nfev=100)
    essential=essential_from(solution.x)
    fitted_mask=(sampson_pixels(essential,a,b,k)<=1.).astype('uint8')[:,None]
    count,rotation,translation,positive=cv2.recoverPose(essential,a,b,k,mask=fitted_mask.copy())
    return essential,count,rotation,translation,positive,fitted_mask


def relative_motion(a, b, k, image_size=(448,294)):
    """Fit on spatially separated feature cells; evaluate unused feature cells.

    x_camera2 = R21*x_camera1 + t21. Monocular t21 has NO metric scale.
    No GNSS, IMU, map, depth model or trajectory is used in estimation.
    """
    a,b=np.asarray(a,float),np.asarray(b,float)
    report={'matches':len(a),'relative_rotation_candidate':False,'metric_translation_observed':False,
            'translation_scale_source':None}
    if len(a)<60:
        return {**report,'rejections':['insufficient_matches']}
    cell=np.floor(a/np.array(image_size)*[4,3]).astype(int)
    test=(cell[:,0]+2*cell[:,1])%3==0
    fit=~test
    if min(fit.sum(),test.sum())<20:
        return {**report,'rejections':['insufficient_spatial_holdout']}
    cv2.setRNGSeed(20261019)
    essential,mask=cv2.findEssentialMat(a[fit],b[fit],k,method=cv2.RANSAC,prob=.999,threshold=1.)
    if essential is None:
        return {**report,'rejections':['essential_estimation_failed']}
    essential=essential[:3,:3]
    try:
        essential,count,rotation,translation,positive,mask=refined_pose(essential,a[fit],b[fit],k,mask)
    except cv2.error:
        return {**report,'rejections':['pose_recovery_failed']}
    good=positive.ravel()>0
    aa=np.column_stack((a[fit],np.ones(fit.sum())))@np.linalg.inv(k).T
    bb=np.column_stack((b[fit],np.ones(fit.sum())))@np.linalg.inv(k).T
    rays1=aa@rotation.T; rays1/=np.linalg.norm(rays1,axis=1)[:,None]
    bb/=np.linalg.norm(bb,axis=1)[:,None]
    angles=np.degrees(np.arccos(np.clip(np.sum(rays1*bb,axis=1),-1,1)))
    parallax=float(np.median(angles[good])) if good.any() else 0.
    errors=sampson_pixels(essential,a[test],b[test],k)
    cells=len(np.unique(cell[fit][good],axis=0))
    # Independent held-out model is a consistency diagnostic, never used to
    # refit the fitted model or turn validation observations into training.
    cv2.setRNGSeed(20261020)
    other,omask=cv2.findEssentialMat(a[test],b[test],k,method=cv2.RANSAC,prob=.999,threshold=1.)
    disagreement=None
    if other is not None:
        try:
            _,_,other_r,_,_,_=refined_pose(other[:3,:3],a[test],b[test],k,omask)
            disagreement=float(np.degrees(Rotation.from_matrix(other_r@rotation.T).magnitude()))
        except cv2.error:
            pass
    checks={'cheirality_support':count>=40,'cheirality_fraction':count/max(1,int((mask>0).sum()))>=.6,
            'spatial_support':cells>=6,'parallax':parallax>=1.,
            'heldout_epipolar':float(np.median(errors))<=1. and float(np.quantile(errors,.9))<=3.,
            'independent_rotation_agreement':disagreement is not None and disagreement<=1.}
    report.update(fit_matches=int(fit.sum()),heldout_matches=int(test.sum()),cheirality_inliers=int(count),
        occupied_fit_cells=cells,median_rotation_compensated_parallax_deg=parallax,
        heldout_sampson_median_px=float(np.median(errors)),heldout_sampson_p90_px=float(np.quantile(errors,.9)),
        fit_vs_heldout_rotation_difference_deg=disagreement,
        rotation_camera2_from_camera1=rotation.tolist(),translation_direction_camera2_from_camera1=translation.ravel().tolist(),
        rotation_flu2_from_flu1=(OPTICAL_TO_FLU@rotation@OPTICAL_TO_FLU.T).tolist(),
        gates=checks,rejections=[k for k,v in checks.items() if not v],relative_rotation_candidate=all(checks.values()))
    return report


def matched_points(features_a,features_b):
    ka,da=features_a;kb,db=features_b
    if da is None or db is None:return np.empty((0,2)),np.empty((0,2))
    matcher=cv2.BFMatcher()
    def pairs(a,b):
        return {p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(a,b,k=2)
                if len(p)==2 and p[0].distance<.8*p[1].distance}
    ab,ba=pairs(da,db),pairs(db,da)
    selected=[];seen=set()
    for i,j in sorted(ab.items()):
        if ba.get(j)!=i:continue
        # One match per 4px cell in both images limits duplicate SIFT scales.
        cell=(tuple(np.floor(np.array(ka[i].pt)/4).astype(int)),tuple(np.floor(np.array(kb[j].pt)/4).astype(int)))
        if cell in seen:continue
        seen.add(cell);selected.append((i,j))
    return (np.array([ka[i].pt for i,_ in selected]).reshape(-1,2),
            np.array([kb[j].pt for _,j in selected]).reshape(-1,2))


def horizontal_map_probe(scene,raw,pose):
    rotated=raw@pose[:3,:3].T;points=rotated+pose[:3,3]
    distances,ids=scene.tree.query(points)
    normals,quality=scene.surfaces(ids)
    reliable=(distances<=2)&(quality>0)
    vertical=reliable&(np.abs(normals[:,2])<=.5)
    selected=np.flatnonzero(vertical)
    # Count each map voxel once instead of treating many image pixels as
    # independent measurements of the same surface voxel.
    order=selected[np.argsort(distances[selected],kind='stable')]
    _,first=np.unique(ids[order],return_index=True)
    selected=order[np.sort(first)]
    jac=np.column_stack((normals,np.sum(normals*np.column_stack((-rotated[:,1],rotated[:,0],np.zeros(len(raw)))),axis=1)))
    jac*=np.array([1.,1.,1.,np.deg2rad(3)])
    residual=np.sum(normals*(points-scene.points[ids]),axis=1)
    weight=2/(len(raw)*.3**2)*quality*np.minimum(1.,.5/np.maximum(np.abs(residual),1e-12))
    information=jac[selected].T@(jac[selected]*weight[selected,None])
    k=[0,1,3];cross=information[k,2]
    schur=information[np.ix_(k,k)]-np.outer(cross,cross)/max(information[2,2],1e-12)
    values=np.maximum(np.linalg.eigvalsh(schur),0.)
    relative=float(values[0]/max(values[-1],1e-12))
    structurally_supported=len(selected)>=20 and relative>=.01
    return ({'raw_points':len(raw),'reliable_points':int(reliable.sum()),'vertical_source_matches':int(vertical.sum()),
        'unique_vertical_targets':len(selected),'horizontal_information_free_z_eigenvalues':values.tolist(),
        'weakest_to_strongest_information_ratio':relative,'structurally_supported':structurally_supported,
        'accepted_for_fusion':False,'reason':'Structure alone does not validate source depth/static correspondence or calibrated uncertainty',
        'normal_geometry_scope':'Data-only normal information, deduplicated targets, no prior; not calibrated covariance'},
        dict(raw_points=raw[selected],target_points=scene.points[ids[selected]],normals=normals[selected],quality=quality[selected]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=saved_path,required=True);p.add_argument('--data-root',type=saved_path,required=True)
    p.add_argument('--output-name',default='horizontal_probes')
    p.add_argument('--width',type=int,default=896);p.add_argument('--height',type=int,default=588)
    for track in (1,2):p.add_argument(f'--map-cache-track{track}',type=saved_path,required=True)
    args=p.parse_args()
    if Path(args.output_name).name!=args.output_name or args.width<448 or args.height<294:
        p.error('Use a single output-directory name and at least 448x294 rectification')
    size=(args.width,args.height)
    out=args.experiment/args.output_name;out.mkdir(exist_ok=False)
    snapshots=out/'sources';snapshots.mkdir()
    source_names=('horizontal_constraints.py','prior_ablation.py','three_camera_calibration.py',
                  'calibrated_rig_geometry.py','three_camera_source.py')
    source_hashes={}
    for name in source_names:
        path=Path(__file__).with_name(name);shutil.copy2(path,snapshots/name);source_hashes[name]=sha256(path)
    dump(out/'runtime.json',{'command':sys.argv,'numpy':np.__version__,'scipy':scipy.__version__,
        'opencv':cv2.__version__,'python':sys.version,'code_sha256':source_hashes,
        'thresholds':{'ratio':.8,'min_matches':60,'fit_cheirality_inliers':40,'min_holdout':20,
            'occupied_fit_cells':6,'grid_wh':[4,3],'min_parallax_deg':1.,'holdout_median_px':1.,
            'holdout_p90_px':3.,'independent_rotation_agreement_deg':1.,'cycle_deg':1.},
        'threshold_scope':'Conservative diagnostic checks, not calibrated measurement covariance; no score tuning'})
    reports=loads_saved_json((args.experiment/'prior_ablation.json').read_text());summary=[]
    sift=cv2.SIFT_create(nfeatures=5000)
    for report in reports:
        track=report['track'];prior=args.experiment/f'track{track}_motion_prior'
        baseline=Path(loads_saved_json((prior/'sensor_prior.json').read_text())['position_only_ablation']['original_prior_run'])
        frames=rows(baseline/'frames.csv');poses=np.loadtxt(baseline/'result.txt').reshape(-1,3,4)
        data=args.data_root/f'Track{track}';image_root=data/f'Track{track}_images'
        cache=getattr(args,f'map_cache_track{track}')
        identity=loads_saved_json(cache.with_suffix(cache.suffix+'.json').read_text())
        if identity['track']!=track or identity['map_voxel_leaf_size']!=1 or identity['map_sha256']!=sha256(data/'map.pcd'):
            raise ValueError('Map source identity mismatch')
        scene=SurfaceMap(load_binary_map(cache));headers=camera_headers(data/f'Track{track}.db3')['f']
        calibration=image_root/'camera_info/gige_100_f_hdr_camera_info.yaml'
        for window in report['windows']:
            indices=[window['start']+6,window['start']+20,window['start']+34]
            provenance=[];features=[];ks=[]
            for idx in indices:
                stamp=int(frames[idx]['timestamp_ns'])
                path=image_root/f'Track{track}_f_images/{idx:06d}_ts_{stamp}.jpg'
                image,k,valid=rectify(path,calibration,size,image_root/'camera_info/gige_100_f_hdr_mask.png')
                features.append(sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),valid.astype('uint8')*255));ks.append(k.astype(float))
                header_ns,frame_id=headers[stamp]
                provenance.append({'idx':idx,'bag_ns':stamp,'header_ns':header_ns,'frame_id':frame_id,'path':str(path.resolve()),'sha256':sha256(path)})
            motions=[]
            for i,j in [(0,1),(1,2),(0,2)]:
                a,b=matched_points(features[i],features[j]);motion=relative_motion(a,b,ks[i],size)
                motion.update(indices=[indices[i],indices[j]],header_dt_s=(provenance[j]['header_ns']-provenance[i]['header_ns'])/1e9)
                motions.append(motion)
            cycle=None
            if all('rotation_camera2_from_camera1'in m for m in motions):
                ab,bc,ac=[np.array(m['rotation_camera2_from_camera1'])for m in motions]
                cycle=float(np.degrees(Rotation.from_matrix(ac.T@bc@ab).magnitude()))
            center=window['center'];raw_path=data/'raw_ply'/frames[center]['filename']
            raw=3*load_ply(raw_path,max_points=6000,leaf=.1)
            geometry,pool=horizontal_map_probe(scene,raw,poses[center])
            prefix=f'track{track}_{window["class"]}_{window["split"]}'
            np.savez_compressed(out/f'{prefix}_map_pool.npz',**pool)
            record={'track':track,**window,'images':provenance,'rectified_size_wh':size,
                'essential_refinement':'5-point RANSAC, then robust fitted-inlier epipolar least-squares; holdout fitted separately for agreement only',
                'calibration_sha256':sha256(calibration),
                'motions':motions,'three_view_rotation_cycle_deg':cycle,
                'rotation_observation_candidate':all(m['relative_rotation_candidate']for m in motions) and cycle is not None and cycle<=1.,
                'horizontal_map':geometry,'raw_ply_sha256':sha256(raw_path),'map_cache_sha256':sha256(cache),
                'joint_fusion_gate':{'metric_translation_independently_observed':False,
                    'camera_imu_body_extrinsic_verified':False,'depth_uncertainty_and_static_map_correspondence_verified':False},
                'joint_fusion_allowed':False,
                'limits':['Monocular essential translation is only a direction. GNSS scale is not independent metric odometry.',
                    'Image-only consensus/cycle consistency cannot certify that all matches are static.',
                    'Fixed front optical-to-FLU conversion is known; camera-to-IMU/body lever-arm/rotation remains unverified.',
                    'Map-pool selection uses a saved pose; pool is diagnostic, not an independent pose truth or automatically accepted factor.']}
            dump(out/f'{prefix}.json',record);summary.append(record)
            print(json.dumps({'track':track,'class':window['class'],'split':window['split'],
                'rotation_candidate':record['rotation_observation_candidate'],'cycle_deg':cycle,
                'vertical_targets':geometry['unique_vertical_targets'],'map_structure_supported':geometry['structurally_supported']}),flush=True)
    dump(out/'report.json',summary)


if __name__=='__main__':main()
