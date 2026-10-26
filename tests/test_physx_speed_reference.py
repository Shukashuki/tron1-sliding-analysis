import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_physx_speed_references import retime,interpolate_quaternions
from physx_reference_math import trajectory_metrics


def source():
    t=np.arange(349)*.02
    p=np.zeros((349,1,3));p[:,0,0]=2*np.sin(t);p[:,0,1]=2*(1-np.cos(t))
    v=np.zeros_like(p);v[:,0,0]=2*np.cos(t);v[:,0,1]=2*np.sin(t)
    q=np.zeros((349,1,4));q[:,0,0]=np.cos((t-.3)/2);q[:,0,3]=np.sin((t-.3)/2)
    return dict(fps=np.array([50.]),joint_pos=np.zeros((349,8)),joint_vel=np.ones((349,8)),
        body_pos_w=p,body_quat_w=q,body_lin_vel_w=v,body_ang_vel_w=np.tile([[[0,0,1.]]],(349,1,1)),
        joint_names=np.array([str(k) for k in range(8)]),body_names=np.array(['base']))


@pytest.mark.parametrize('scale,frames',[(1.5,233),(2.,175),(2.5,140),(3.,117)])
def test_retime_preserves_complete_path_and_scales_velocities(scale,frames):
    old=source();new,k=retime(old,scale)
    assert len(new['joint_pos'])==frames
    np.testing.assert_allclose(new['body_pos_w'][[0,-1]],old['body_pos_w'][[0,-1]])
    np.testing.assert_allclose(new['body_lin_vel_w'][[0,-1]],old['body_lin_vel_w'][[0,-1]]*k)
    np.testing.assert_allclose(new['joint_vel'],k)
    np.testing.assert_allclose(new['body_ang_vel_w'][...,2],k)
    np.testing.assert_allclose(np.linalg.norm(new['body_quat_w'],axis=-1),1.)
    q=new['body_quat_w'][:,0];yaw=np.unwrap(2*np.arctan2(q[:,3],q[:,0]))
    v=new['body_lin_vel_w'][:,0];beta=np.arctan2(np.sin(np.arctan2(v[:,1],v[:,0])-yaw),np.cos(np.arctan2(v[:,1],v[:,0])-yaw))
    np.testing.assert_allclose(beta,.3,atol=2e-6)
    assert new['source_time_s'][-1]==6.96


def test_quaternion_sign_flips_do_not_create_rotations():
    q=np.array([[[1.,0,0,0]],[[-1.,0,0,0]]])
    out=interpolate_quaternions(q,np.linspace(0,1,11))
    np.testing.assert_allclose(abs(out[:,:,0]),1.)
    np.testing.assert_allclose(out[:,:,1:],0.)


def test_rejects_invalid_scale():
    with pytest.raises(ValueError):retime(source(),0.)


def test_three_times_speed_metrics_use_phase_matched_core_window():
    ref,k=retime(source(),3.)
    t=np.arange(len(ref['joint_pos'])*8+1)*.0025;rt=ref['time_s']
    interp=lambda a:np.column_stack([np.interp(t,rt,a[:,j]) for j in range(a.shape[-1])])
    pos=interp(ref['body_pos_w'][:,0]);vel=interp(ref['body_lin_vel_w'][:,0])
    q=interpolate_quaternions(ref['body_quat_w'],np.minimum(t/.02,len(ref['joint_pos'])-1))[:,0]
    trace=dict(root_pos=pos,root_quat=q,root_com_vel=vel,normal_fz=np.ones((len(t),2))*100,
        utilization_proxy=np.zeros((len(t),2)))
    m=trajectory_metrics(trace,ref,core_interval_s=(1.44/k,4/k))
    assert m['core']['xy_rmse_m']<1e-12 and m['core']['speed_retention']==1.
    assert m['core']['beta_rmse_deg']<1e-10
    assert m['longest_support_gap_s']==0.
