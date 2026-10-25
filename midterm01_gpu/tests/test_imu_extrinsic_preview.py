"""獨立外參預覽的方向、原點、時間精度及因果檢查；不執行定位。"""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SPEC=importlib.util.spec_from_file_location('preview_imu_extrinsic',Path(__file__).resolve().parents[1]/'scripts/preview_imu_extrinsic.py')
preview=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)
EPOCH=1727239790221121554


def test_rpy_convention_matches_independent_rotation_library():
    expected=Rotation.from_euler('xyz',[13,-27,91],degrees=True).as_matrix()
    np.testing.assert_allclose(preview.rotation_from_rpy((13,-27,91)),expected,atol=1e-14)


def test_yaw_90_maps_positive_x_to_positive_y_and_keeps_gravity():
    result=preview.transform_imu([EPOCH,EPOCH+10_000_000],[[1,0,9.81]]*2,[[2,0,0]]*2,
                               (0,0,90),(0,0,0))
    np.testing.assert_allclose(result['accel_target'],[[0,1,9.81]]*2,atol=1e-14)
    np.testing.assert_allclose(result['gyro_target'],[[0,2,0]]*2,atol=1e-14)
    assert result['accel_valid'].all() and not result['alpha_valid'][0]


def test_constant_rotation_lever_arm_sign_at_ply_origin():
    # IMU 在 PLY +X 一公尺；繞 PLY 原點 Z 軸 2 rad/s，IMU 向心 -4X。
    result=preview.transform_imu([EPOCH,EPOCH+10_000_000],[[-4,0,9.81]]*2,[[0,0,2]]*2,
                               (0,0,0),(1,0,0))
    assert not result['accel_valid'][0] and np.isnan(result['accel_target'][0]).all()
    np.testing.assert_allclose(result['accel_target'][1],[0,0,9.81],atol=1e-14)
    np.testing.assert_allclose(result['centripetal'][1],[4,0,0])


def test_tangential_acceleration_sign_with_nonidentity_rotation():
    # 在 PLY +Y 的 IMU；PLY alpha +Z=3，IMU 切向加速度 -3X。
    ts=np.array([EPOCH,EPOCH+10_000_000,EPOCH+20_000_000],dtype=np.int64)
    omega_ply=np.array([[0,0,0],[0,0,.03],[0,0,.06]])
    r=Rotation.from_euler('xyz',[20,-15,80],degrees=True).as_matrix()
    t=np.array([0.,1.,0.])
    expected=np.tile([.5,-.2,9.81],(3,1))
    imu_force=expected+np.cross([0,0,3],t)+np.cross(omega_ply,np.cross(omega_ply,t))
    result=preview.transform_imu(ts,imu_force@r,omega_ply@r,(20,-15,80),t)
    np.testing.assert_allclose(result['accel_target'][1:],expected[1:],atol=1e-13)
    np.testing.assert_allclose(result['alpha'][1:],[[0,0,3]]*2,atol=1e-13)


def test_causal_regression_uses_exact_ns_and_is_independent_of_future():
    # 不規則取樣及 epoch 上 101 ns 差，避免先轉float造成時序誤差。
    ts=np.array([EPOCH,EPOCH+101,EPOCH+1200,EPOCH+8000,EPOCH+20_000],dtype=np.int64)
    relative=(ts-ts[0])*1e-9
    slope=np.array([3.,-2.,.7])
    omega=relative[:,None]*slope+[.3,.2,-.1]
    alpha,valid=preview.causal_alpha(ts,omega,.1)
    np.testing.assert_allclose(alpha[valid],np.tile(slope,(4,1)),rtol=1e-9,atol=1e-9)
    altered=omega.copy();altered[3:]=1000
    changed,_=preview.causal_alpha(ts,altered,.1)
    np.testing.assert_array_equal(alpha[:3],changed[:3])


def test_large_gap_is_invalid_instead_of_inventing_derivative():
    ts=[EPOCH,EPOCH+10_000_000,EPOCH+1_000_000_000]
    result=preview.transform_imu(ts,[[0,0,9.81]]*3,[[0,0,0]]*3,(0,0,0),(1,0,0))
    assert result['accel_valid'].tolist()==[False,True,False]


def test_disabled_lever_arm_retains_imu_origin_and_translation_does_not_change_gyro():
    args=([EPOCH,EPOCH+10_000_000],[[1,2,3]]*2,[[0,0,2]]*2,(10,20,30))
    off=preview.transform_imu(*args,(2,-1,.5),compensate=False)
    on=preview.transform_imu(*args,(2,-1,.5),compensate=True)
    np.testing.assert_allclose(off['accel_target'],off['accel_rotated'])
    np.testing.assert_array_equal(off['gyro_target'],on['gyro_target'])
    assert off['acceleration_origin']=='imu_origin_rotated_only' and off['accel_valid'].all()


@pytest.mark.parametrize('stamps',[[float(EPOCH),float(EPOCH+10_000_000)],[EPOCH,EPOCH]])
def test_rejects_float_epoch_or_repeated_header(stamps):
    with pytest.raises(ValueError):preview.causal_alpha(stamps,np.zeros((2,3)),.1)


@pytest.mark.parametrize('rpy,xyz,window',[(('nan',0,0),(0,0,0),.1),((0,0,0),(0,float('inf'),0),.1),
                                        ((0,0,0),(0,0,0),0),((0,0,0),(0,0,0),float('nan'))])
def test_rejects_nonfinite_parameters(rpy,xyz,window):
    with pytest.raises(ValueError):preview.transform_imu([EPOCH],[[0,0,9.81]],[[0,0,0]],rpy,xyz,window_s=window)


def test_csv_preserves_int_epoch_and_blanks_unknown_lever_sample(tmp_path):
    records=[(1,EPOCH,EPOCH-500,[1.,2.,3.],[0.,0.,2.],True,True),
             (2,EPOCH+10_000_000,EPOCH+9_999_500,[1.,2.,3.],[0.,0.,2.],True,True)]
    result=preview.transform_imu([r[2] for r in records],[r[3] for r in records],[r[4] for r in records],
                               (0,0,0),(1,0,0))
    path=tmp_path/'output.csv'
    preview.save_csv(path,records,[0,.01],result)
    with path.open() as handle:rows=list(csv.DictReader(handle))
    assert rows[0]['timestamp_ns']==str(EPOCH) and rows[1]['header_timestamp_ns']==str(EPOCH+9_999_500)
    assert rows[0]['ax_target_mps2']=='' and rows[0]['accel_valid']=='0'
    assert rows[1]['ax_target_mps2']=='5.0' and rows[1]['accel_valid']=='1'
    previous=path.read_bytes()
    with pytest.raises(FileExistsError):preview.save_csv(path,records,[0,.01],result)
    assert path.read_bytes()==previous
