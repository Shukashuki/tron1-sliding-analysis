import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physx_reference_math import smooth_basis, reduce_contacts, trajectory_metrics


def test_spline_preserves_independent_channels_and_endpoints():
    b=smooth_basis(349,12)
    assert np.max(abs(b[[0,-1]])) < 1e-25
    assert np.all(b>=0)
    z=np.zeros((12,8));z[:,0]=.1;z[:,1]=-.2
    r=b@z
    np.testing.assert_allclose(r[:,0]*-2,r[:,1])
    np.testing.assert_array_equal(r[:,2:],0)


def test_actual_points_vs_separate_anchors_and_power():
    torch=pytest.importorskip('torch')
    f=lambda x:torch.tensor(x,dtype=torch.float32)
    normal=(f([10.,20.]),f([[0,0,0],[0,1,0]]),f([[0,0,1],[0,0,1]]),f([0,0]),
            torch.tensor([[1],[1]]),torch.tensor([[1],[0]]))
    friction=(f([[0,-7,0],[0,-14,0]]),f([[0,0,0],[0,1,0]]),
              torch.tensor([[1],[1]]),torch.tensor([[1],[0]]))
    out=reduce_contacts(normal,friction,f([[0,0,0],[0,0,0]]),f([[0,2,0],[0,2,0]]),
                        f([[0,0,0],[0,0,0]]),f([[1,0,0,0],[1,0,0,0]]),.7,cap=4)
    np.testing.assert_allclose(out['normal_fz'],[20,10])
    np.testing.assert_allclose(out['lateral_slip_rms'],[2,2])
    np.testing.assert_allclose(out['friction_anchor_power'],[-28,-14])
    np.testing.assert_allclose(out['utilization_proxy'],[1,1])
    assert not out['fault'].any()


def test_unloaded_zero_is_not_saturation_and_overflow_is_fault():
    torch=pytest.importorskip('torch')
    z=torch.zeros
    normal=(z(1),z(1,3),z(1,3),z(1),torch.tensor([[0]]),torch.tensor([[0]]))
    friction=(z(1,3),z(1,3),torch.tensor([[0]]),torch.tensor([[0]]))
    out=reduce_contacts(normal,friction,z(1,3),z(1,3),z(1,3),torch.tensor([[1.,0,0,0]]),.7,cap=2)
    assert out['lateral_slip_rms'].item()==out['utilization_proxy'].item()==0
    assert not out['fault'].any()
    bad=list(normal);bad[4]=torch.tensor([[3]])
    assert reduce_contacts(bad,friction,z(1,3),z(1,3),z(1,3),torch.tensor([[1.,0,0,0]]),.7,cap=2)['fault'].any()


def test_tracking_uses_native_time_interpolation_and_preserves_contact_gap():
    n=2793;t=np.arange(n)*.0025;rt=np.arange(349)*.02
    rp=np.zeros((349,1,3));rp[:,0,0]=rt
    rq=np.zeros((349,1,4));rq[:,0,0]=1
    rv=np.zeros((349,1,3));rv[:,0,0]=1
    ref=dict(body_pos_w=rp,body_quat_w=rq,body_lin_vel_w=rv)
    p=np.zeros((n,3));p[:,0]=t
    q=np.zeros((n,4));q[:,0]=1
    v=np.zeros((n,3));v[:,0]=1
    force=np.zeros((n,2));force[:,0]=100.;force[100:104]=0
    trace=dict(root_pos=p,root_quat=q,root_com_vel=v,normal_fz=force,utilization_proxy=np.zeros((n,2)))
    m=trajectory_metrics(trace,ref)
    assert m['core']['xy_rmse_m']<1e-12 and m['whole']['xy_rmse_m']<1e-12
    assert m['core']['speed_retention']==1. and m['core']['beta_rmse_deg']==0.
    assert m['longest_support_gap_s']==.01
    assert m['task_success_claimed'] is False
