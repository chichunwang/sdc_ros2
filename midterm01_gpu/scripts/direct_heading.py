"""Yaw-source replacement ablation; no inferred sensor independence or covariance."""
import numpy as np
from scipy.spatial.transform import Rotation


def interpolate_pose_yaw(times, quaternions, queries, max_edge_s=.3):
    """Unwrap GNSS Pose yaw and interpolate, with explicit short edge extrapolation.

    Roll/pitch are deliberately not returned. The caller retains the original
    IMU tilt and replaces yaw only. This tests the supplied orientation itself,
    without adding a second correlated orientation measurement to a filter.
    """
    times=np.asarray(times,float);queries=np.asarray(queries,float)
    q=np.asarray(quaternions,float)
    if (times.ndim!=1 or queries.ndim!=1 or len(times)<2 or not len(queries)
            or q.shape!=(len(times),4) or not np.isfinite(times).all()
            or not np.isfinite(queries).all() or not np.isfinite(q).all()
            or np.any(np.diff(times)<=0) or np.max(np.diff(times))>.6
            or np.any(abs(np.linalg.norm(q,axis=1)-1)>.02)):
        raise ValueError('Direct pose yaw requires finite increasing covered GNSS times and valid quaternions')
    if (not np.isfinite(max_edge_s) or not 0<=max_edge_s<=.3
            or queries.min()<times[0]-max_edge_s or queries.max()>times[-1]+max_edge_s):
        raise ValueError('Direct pose yaw outside the explicit <=0.3s edge-extrapolation limit')
    yaw=np.unwrap(Rotation.from_quat(q).as_euler('xyz')[:,2])
    result=np.interp(queries,times,yaw)
    for mask,a,b in ((queries<times[0],0,1),(queries>times[-1],-2,-1)):
        result[mask]=yaw[a]+(queries[mask]-times[a])*(yaw[b]-yaw[a])/(times[b]-times[a])
    return result,dict(heading_source='raw GNSS Pose yaw, directly replaces IMU yaw',
        heading_source_verified=False,heading_independence_assumed=False,
        heading_method='circularly unwrapped linear yaw interpolation; endpoint constant angular-rate extrapolation <=0.3s',
        heading_covariance_available=False,heading_bias_smoothing_applied=False,
        heading_edge_extrapolated_frames=int(((queries<times[0])|(queries>times[-1])).sum()),
        experimental=True)
