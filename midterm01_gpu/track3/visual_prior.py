"""Causal current/previous-image PnP prior using previous accepted PLY scale.

No invented forward speed: motion must be supported by tracked image features
and previous-frame 3D depths. PnP is a small pose solve on CPU; matching those
features to the cloud projection is performed on CUDA.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import cv2
import yaml
from scipy.spatial.transform import Rotation
from .geometry import valid_pose
from .io import frames

RDF_TO_FLU=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])


def relative_pose_from_pnp(rvec,tvec):
    current_from_previous=np.eye(4)
    current_from_previous[:3,:3]=RDF_TO_FLU@cv2.Rodrigues(rvec)[0]
    current_from_previous[:3,3]=RDF_TO_FLU@np.asarray(tvec).ravel()
    return np.linalg.inv(current_from_previous)


def filter_by_camera_mask(points,K,valid_mask):
    p=cp.asarray(points,dtype=cp.float64)
    finite=cp.all(cp.isfinite(p),axis=1)&(p[:,0]>0)
    safe=cp.where(finite,p[:,0],1.)
    u=cp.rint(K[0,2]-K[0,0]*p[:,1]/safe).astype(cp.int64)
    v=cp.rint(K[1,2]-K[1,1]*p[:,2]/safe).astype(cp.int64)
    h,w=valid_mask.shape
    inside=finite&(u>=0)&(u<w)&(v>=0)&(v<h)
    allowed=cp.asarray(valid_mask)[cp.clip(v,0,h-1),cp.clip(u,0,w-1)]>0
    keep=inside&allowed
    return cp.asnumpy(p[keep]),dict(input_points=len(p),kept_points=int(keep.sum()),
                                    removed_points=int((~keep).sum()),method='provided_front_ego_mask_and_rectified_FOV')


class VisualPrior:
    def __init__(self,data,cfg,expected_frames):
        self.cfg=cfg;self.previous=None;self.current=None
        rows=frames(Path(data)/'Track3_images/Track3_f_images',suffix='.jpg',expected=expected_frames)
        self.images={r['timestamp_ns']:r['path'] for r in rows}
        camera=Path(data)/'Track3_images/camera_info'
        calibration=yaml.safe_load((camera/'gige_100_f_hdr_camera_info.yaml').read_text())
        K=np.array(calibration['camera_matrix']['data']).reshape(3,3)
        D=np.array(calibration['distortion_coefficients']['data'])
        self.K=np.array(calibration['projection_matrix']['data']).reshape(3,4)[:,:3].copy()
        size=(calibration['image_width'],calibration['image_height'])
        self.maps=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),self.K,size,cv2.CV_32FC1)
        mask=cv2.imread(str(camera/'gige_100_f_hdr_mask.png'),0)
        if mask is None:raise ValueError('Missing front-camera ego mask')
        mask=cv2.remap(mask,*self.maps,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        self.mask=cv2.erode(np.where(mask==0,255,0).astype('uint8'),np.ones((7,7),dtype='uint8'))

    def predict(self,stamp,enabled):
        path=self.images[stamp]
        raw=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if raw is None:raise ValueError(f'Unreadable image: {path}')
        gray=cv2.remap(raw,*self.maps,cv2.INTER_LINEAR)
        self.current=(int(stamp),gray,path)
        diag=dict(accepted=False,reason='awaiting_map_initialization',image=str(path),timestamp_ns=int(stamp))
        if self.previous is None or not enabled:return None,diag
        previous_stamp,previous_gray,points,scale=self.previous
        if stamp<=previous_stamp:raise ValueError('Visual prior requires strictly increasing timestamps')
        p0=cv2.goodFeaturesToTrack(previous_gray,maxCorners=self.cfg['max_features'],
                                   qualityLevel=.01,minDistance=12,mask=self.mask)
        if p0 is None or len(p0)<self.cfg['min_inliers']:
            diag['reason']='insufficient_image_features';return None,diag
        lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        p1,s1,_=cv2.calcOpticalFlowPyrLK(previous_gray,gray,p0,None,**lk)
        if p1 is None:diag['reason']='optical_flow_failed';return None,diag
        back,s0,_=cv2.calcOpticalFlowPyrLK(gray,previous_gray,p1,None,**lk)
        good=(s1.ravel()!=0)&(s0.ravel()!=0)&(np.linalg.norm(back-p0,axis=2).ravel()<1.)
        before=p0.reshape(-1,2)[good];after=p1.reshape(-1,2)[good]
        p=cp.asarray(points,dtype=cp.float64)
        p=p[cp.all(cp.isfinite(p),axis=1)&(p[:,0]>0)]
        uv=cp.stack([self.K[0,2]-self.K[0,0]*p[:,1]/p[:,0],
                     self.K[1,2]-self.K[1,1]*p[:,2]/p[:,0]],axis=1)
        ids=[];distance=[]
        for start in range(0,len(before),128):
            delta=cp.asarray(before[start:start+128])[:,None,:]-uv[None,:,:]
            d2=cp.sum(delta*delta,axis=2);idx=cp.argmin(d2,axis=1)
            ids.append(cp.asnumpy(idx));distance.append(cp.asnumpy(cp.sqrt(d2[cp.arange(len(idx)),idx])))
        if not ids:diag['reason']='no_bidirectional_tracks';return None,diag
        ids=np.concatenate(ids);distance=np.concatenate(distance)
        keep=distance<=self.cfg['max_depth_projection_px']
        objects=cp.asnumpy(p)[ids[keep]]*scale;pixels=after[keep]
        diag.update(tracked_features=len(before),depth_correspondences=len(objects))
        if len(objects)<self.cfg['min_inliers']:
            diag['reason']='insufficient_tracked_depths';return None,diag
        cv2.setRNGSeed(246)
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(objects,pixels,self.K,None,iterationsCount=200,
                                              reprojectionError=self.cfg['reprojection_px'],confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        count=0 if inliers is None else len(inliers)
        diag.update(inliers=count,inlier_fraction=count/len(objects))
        if not ok or count<self.cfg['min_inliers'] or count/len(objects)<self.cfg['min_inlier_fraction']:
            diag['reason']='pnp_outliers';return None,diag
        inliers=inliers.ravel()
        ok,rvec,tvec=cv2.solvePnP(objects[inliers],pixels[inliers],self.K,None,rvec,tvec,True,flags=cv2.SOLVEPNP_ITERATIVE)
        projected,_=cv2.projectPoints(objects[inliers],rvec,tvec,self.K,None)
        reprojection=np.linalg.norm(projected.reshape(-1,2)-pixels[inliers],axis=1)
        relative=relative_pose_from_pnp(rvec,tvec)
        dt=(int(stamp)-previous_stamp)*1e-9
        speed=np.linalg.norm(relative[:3,3])/dt
        turn=Rotation.from_matrix(relative[:3,:3]).magnitude()/dt
        diag.update(reprojection_median_px=float(np.median(reprojection)),translation_m=float(np.linalg.norm(relative[:3,3])),
                    speed_mps=float(speed),rotation_rate_radps=float(turn),relative_previous_current=relative.tolist())
        accepted=bool(ok and valid_pose(relative) and np.median(reprojection)<=self.cfg['reprojection_px'] and
                      speed<=self.cfg['max_speed_mps'] and turn<=self.cfg['max_rotation_rate_radps'])
        diag.update(accepted=accepted,reason='visual_pnp' if accepted else 'visual_motion_gate')
        return relative if accepted else None,diag

    def store(self,points,scale):
        stamp,gray,_=self.current
        self.previous=(stamp,gray,np.asarray(points,dtype=float).copy(),float(scale))
