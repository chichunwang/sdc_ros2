"""Sparse metric 3D--2D fits; CPU small solves, never map registration.

Objects are previous-camera FLU metres; pixels are current rectified optical
coordinates. Output T_previous_current maps current FLU into previous FLU.
Covariance coordinates are [translation_FLU_m, rotation_vector_FLU_rad].
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from . import parameters as p

OPTICAL_TO_FLU=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])


def relative_from_cv(rvec,tvec):
    T=np.eye(4);T[:3,:3]=OPTICAL_TO_FLU@cv2.Rodrigues(rvec)[0]
    T[:3,3]=OPTICAL_TO_FLU@np.asarray(tvec).ravel()
    return np.linalg.inv(T)


def coordinates(T):
    return np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_rotvec()]


def project(objects,R,t,K):
    q=np.asarray(objects)@R.T+np.asarray(t).ravel()
    depth=q[:,2];safe=np.where(depth>1e-9,depth,1.)
    uv=q[:,:2]/safe[:,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    return uv,depth


def grid_cells(pixels,shape):
    h,w=shape;cells=np.floor(np.asarray(pixels)/[w/8,h/6]).astype(int)
    return len(np.unique(cells,axis=0))


def depth_on_feature_rays(points,pixels,K):
    """Borrow only forward depth, not the neighbouring PLY point's ray.

    Nearest-pixel association is approximate. 3D--2D correspondences must
    project to the ACTUAL keypoint at the anchor time, even at zero motion.
    """
    points=np.asarray(points,dtype=float);pixels=np.asarray(pixels,dtype=float)
    return points[:,0,None]*np.c_[np.ones(len(pixels)),
        -(pixels[:,0]-K[0,2])/K[0,0],-(pixels[:,1]-K[1,2])/K[1,1]]


def refine_pixel_translation(objects,pixels,K,R,t):
    """Fixed-rotation Huber reprojection solve, in pixels (not depth*pixel).

    RANSAC supplies the fixed consensus; backtracking never increases its
    robust pixel cost. Final consensus is recomputed by the caller.
    """
    t=np.asarray(t,dtype=float).copy();trace=[];threshold=p.IMAGE_SPEED_REPROJECTION_PX
    def evaluate(value):
        uv,depth=project(objects,R,value,K)
        error=uv-pixels;norm=np.linalg.norm(error,axis=1)
        cost=float(np.sum(np.where(norm<=threshold,.5*norm**2,threshold*(norm-.5*threshold))))
        if not np.all(depth>1e-9):cost=np.inf
        return cost,error,depth,norm
    for _ in range(20):
        cost,error,depth,norm=evaluate(t)
        if not np.isfinite(cost):break
        q=objects@R.T+t
        J=np.zeros((len(q),2,3));J[:,0,0]=K[0,0]/depth;J[:,1,1]=K[1,1]/depth
        J[:,0,2]=-K[0,0]*q[:,0]/depth**2;J[:,1,2]=-K[1,1]*q[:,1]/depth**2
        weight=np.minimum(1.,threshold/np.maximum(norm,1e-12))
        H=np.einsum('nij,nik,n->jk',J,J,weight)
        gradient=np.einsum('nij,ni,n->j',J,error,weight)
        if np.linalg.matrix_rank(H)<3:break
        delta=-np.linalg.solve(H,gradient);accepted=False
        for power in range(12):
            step=delta*2.**-power;candidate=t+step;after=evaluate(candidate)[0]
            if after<=cost:
                t=candidate;trace.append(dict(pixel_cost_before=cost,pixel_cost_after=after,
                    step_m=step.tolist()));accepted=True;break
        if not accepted or np.linalg.norm(step)<1e-8:break
    return t,trace


def finish(objects,pixels,K,R,t,keep,cov,dt,shape,solver,gyro_R,gyro_variance,scale_variance):
    uv,depth=project(objects,R,t,K);errors=np.linalg.norm(uv-pixels,axis=1)
    T=relative_from_cv(cv2.Rodrigues(R)[0],t);y=coordinates(T)
    count=len(keep);threshold=p.IMAGE_SPEED_REPROJECTION_PX
    cells=grid_cells(pixels[keep],shape) if count else 0
    median=float(np.median(errors[keep])) if count else None
    relative_angle=float(Rotation.from_matrix(T[:3,:3]).magnitude())
    consistency=None;allowance=None
    if gyro_R is not None:
        consistency=float(Rotation.from_matrix(gyro_R.T@T[:3,:3]).magnitude())
        allowance=3*np.sqrt(max(float(np.linalg.eigvalsh(cov[3:,3:]).max())+gyro_variance,0.))
    valid=bool(count>=p.IMAGE_SPEED_MIN_TRACKS and count>=.6*len(objects)
        and cells>=p.IMAGE_SPEED_MIN_GRID_CELLS and np.all(depth[keep]>0)
        and median is not None and median<=threshold
        and np.linalg.norm(T[:3,3])/dt<=p.IMAGE_SPEED_MAX_MPS
        and relative_angle/dt<=p.IMAGE_SPEED_MAX_ROTATION_RATE_RADPS
        and (consistency is None or consistency<=allowance+32*np.finfo(float).eps))
    cov=cov.copy();cov[:3,:3]+=np.eye(3)*p.FEATURE_STD_M**2
    cov[:3,:3]+=np.outer(T[:3,3],T[:3,3])*scale_variance
    cov[3:,3:]+=np.eye(3)*np.deg2rad(p.CORRECTION_MEAS_YAW_FLOOR_DEG)**2
    return dict(accepted=valid,reason='relative_geometry' if valid else 'relative_geometry_quality_gate',
        solver=solver,relative_previous_current=T.tolist(),pose_covariance=cov.tolist(),
        covariance_coordinates='previous FLU [tx,ty,tz,rotvec_x,rotvec_y,rotvec_z]',
        inliers=count,inlier_fraction=count/max(1,len(objects)),grid_cells=cells,
        reprojection_median_px=median,common_capped_pixel_cost=float(np.mean(np.minimum(errors,3*threshold)**2)),
        speed_mps=float(np.linalg.norm(T[:3,3])/dt),rotation_rate_radps=relative_angle/dt,
        gyro_rotation_disagreement_rad=consistency,gyro_rotation_allowance_rad=allowance,
        static_scope='spatially distributed dominant rigid-motion consensus; no semantic guarantee')


def pnp_fit(objects,pixels,K,dt,shape,*,gyro_R=None,gyro_variance=0.,scale_variance=0.):
    failure=dict(accepted=False,reason='pnp_outliers',solver='pnp',inliers=0)
    if len(objects)<p.IMAGE_SPEED_MIN_TRACKS:return failure
    cv2.setRNGSeed(7513)
    ok,rv,tv,ids=cv2.solvePnPRansac(objects,pixels,K,None,iterationsCount=200,
        reprojectionError=p.IMAGE_SPEED_REPROJECTION_PX,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    keep=np.array([],dtype=int) if ids is None else ids.ravel()
    failure['inliers']=len(keep)
    if not ok or len(keep)<p.IMAGE_SPEED_MIN_TRACKS or len(keep)<.6*len(objects):return failure
    ok,rv,tv=cv2.solvePnP(objects[keep],pixels[keep],K,None,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:return failure
    projected,J=cv2.projectPoints(objects[keep],rv,tv,K,None)
    sigma=max(.5,float(np.median(np.linalg.norm(projected.reshape(-1,2)-pixels[keep],axis=1))))
    units=np.diag([.05]*3+[1.]*3);H=units@(J[:,:6].T@J[:,:6]/sigma**2)@units
    eig,U=np.linalg.eigh((H+H.T)*.5)
    if eig.min()<=max(1e-10,eig.max()*1e-10):return dict(failure,reason='pnp_geometry_rank_deficient')
    C=units@((U/eig)@U.T)@units
    theta=np.r_[rv.ravel(),tv.ravel()]
    G=np.column_stack([(coordinates(relative_from_cv((theta+np.eye(6)[i]*1e-6)[:3],(theta+np.eye(6)[i]*1e-6)[3:]))
                       -coordinates(relative_from_cv((theta-np.eye(6)[i]*1e-6)[:3],(theta-np.eye(6)[i]*1e-6)[3:])))/2e-6 for i in range(6)])
    return finish(objects,pixels,K,cv2.Rodrigues(rv)[0],tv,keep,G@C@G.T,dt,shape,'pnp',gyro_R,gyro_variance,scale_variance)


def gyro_translation_fit(objects,pixels,K,dt,shape,gyro_R,*,gyro_variance=0.,scale_variance=0.):
    failure=dict(accepted=False,reason='gyro_translation_outliers',solver='gyro_translation',inliers=0)
    if gyro_R is None:return dict(failure,reason='missing_gyro_rotation_matrix')
    if len(objects)<p.IMAGE_SPEED_MIN_TRACKS:return failure
    normalized=(pixels-[K[0,2],K[1,2]])/[K[0,0],K[1,1]]
    A=np.zeros((len(objects),2,3));A[:,0,0]=1.;A[:,1,1]=1.;A[:,:,2]=-normalized
    A*=np.array([K[0,0],K[1,1]])[None,:,None]
    def solve(rotation,ids):
        R=OPTICAL_TO_FLU.T@rotation.T;q=objects@R.T
        b=(normalized*q[:,2,None]-q[:,:2])*[K[0,0],K[1,1]]
        return np.linalg.lstsq(A[ids].reshape(-1,3),b[ids].ravel(),rcond=None)[0]
    R=OPTICAL_TO_FLU.T@gyro_R.T;rng=np.random.default_rng(7513);best=None
    for _ in range(200):
        ids=rng.choice(len(objects),3,replace=False);t=solve(gyro_R,ids)
        uv,depth=project(objects,R,t,K);error=np.linalg.norm(uv-pixels,axis=1)
        keep=np.flatnonzero((depth>0)&(error<=p.IMAGE_SPEED_REPROJECTION_PX))
        score=(len(keep),-float(np.median(error[keep])) if len(keep) else -np.inf)
        if best is None or score>best[0]:best=(score,keep,t)
    keep=best[1];failure['inliers']=len(keep)
    if len(keep)<p.IMAGE_SPEED_MIN_TRACKS or len(keep)<.6*len(objects):return failure
    t=solve(gyro_R,keep);trace=[]
    if p.v514_enabled():
        t,trace=refine_pixel_translation(objects[keep],pixels[keep],K,R,t)
        uv,depth=project(objects,R,t,K)
        keep=np.flatnonzero((depth>0)&(np.linalg.norm(uv-pixels,axis=1)<=p.IMAGE_SPEED_REPROJECTION_PX))
        if len(keep)<p.IMAGE_SPEED_MIN_TRACKS or len(keep)<.6*len(objects):
            return dict(failure,inliers=len(keep),reason='gyro_translation_final_consensus',pixel_refinement=trace)
    q=objects[keep]@R.T+t
    J=np.zeros((len(keep),2,3));J[:,0,0]=K[0,0]/q[:,2];J[:,1,1]=K[1,1]/q[:,2]
    J[:,0,2]=-K[0,0]*q[:,0]/q[:,2]**2;J[:,1,2]=-K[1,1]*q[:,1]/q[:,2]**2
    uv,_=project(objects[keep],R,t,K);sigma=max(.5,float(np.median(np.linalg.norm(uv-pixels[keep],axis=1))))
    H=J.reshape(-1,3).T@J.reshape(-1,3)/sigma**2;eig=np.linalg.eigvalsh(H)
    if eig.min()<=max(1e-10,eig.max()*1e-10):return dict(failure,reason='gyro_translation_rank_deficient')
    T=relative_from_cv(cv2.Rodrigues(R)[0],t);C=np.zeros((6,6));B=-gyro_R@OPTICAL_TO_FLU
    C[:3,:3]=B@np.linalg.inv(H)@B.T
    vector=Rotation.from_matrix(gyro_R).as_rotvec()
    def changed(v):
        rotation=Rotation.from_rotvec(v).as_matrix();rr=OPTICAL_TO_FLU.T@rotation.T
        tt=solve(rotation,keep)
        if p.v514_enabled():tt,_=refine_pixel_translation(objects[keep],pixels[keep],K,rr,tt)
        return coordinates(relative_from_cv(cv2.Rodrigues(rr)[0],tt))
    G=np.column_stack([(changed(vector+np.eye(3)[i]*1e-6)-changed(vector-np.eye(3)[i]*1e-6))/2e-6 for i in range(3)])
    C+=G@G.T*gyro_variance
    result=finish(objects,pixels,K,R,t,keep,C,dt,shape,'gyro_translation',gyro_R,gyro_variance,scale_variance)
    if p.v514_enabled():result.update(pixel_refinement=trace,translation_objective='Huber pixel reprojection; final inliers recomputed')
    return result


def compare_relative(objects,pixels,K,dt,shape,gyro_R,gyro_variance,scale_variance):
    objects=np.asarray(objects,dtype=float);pixels=np.asarray(pixels,dtype=float)
    if (dt<=0 or objects.ndim!=2 or objects.shape[1]!=3 or pixels.shape!=(len(objects),2)
            or not np.isfinite(objects).all() or not np.isfinite(pixels).all()):
        raise ValueError('Invalid relative-motion observations')
    return dict(pnp=pnp_fit(objects,pixels,K,dt,shape,gyro_R=gyro_R,
                    gyro_variance=gyro_variance,scale_variance=scale_variance),
        gyro_translation=gyro_translation_fit(objects,pixels,K,dt,shape,gyro_R,
                    gyro_variance=gyro_variance,scale_variance=scale_variance))
