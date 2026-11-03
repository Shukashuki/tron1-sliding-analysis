import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_hockey_equal_distance import direct_reference,forward_budget,persistent_stop


def test_distance_excludes_approach_and_is_not_path_length():
    t=np.array([0.,.4,1.8,2.76]);p=np.array([[0.,0.,0.],[.52,0.,0.],[1.44,.3,0.],[1.44,.3,0.]])
    b=forward_budget(t,p,.4,1.8)
    assert b['approach_distance_m']==pytest.approx(.52)
    assert b['braking_distance_m']==pytest.approx(.92)
    assert b['total_distance_m']==pytest.approx(1.44)


def test_direct_reference_keeps_xy_and_rigid_pose_but_not_hockey_velocity():
    t=np.arange(8)*.02;root=np.stack((t*1.3,t*0,t*.1+.75),1)
    offsets=np.array([[0,0,0],[0,.2,-.4]])
    p=root[:,None]+offsets[None]
    motion=dict(fps=np.array(50.),body_names=np.array(['base_Link','wheel_L_Link']),
        body_pos_w=p,body_quat_w=np.tile([1.,0,0,0],(8,2,1)),joint_pos=np.zeros((8,8)),
        joint_vel=np.ones((8,8)),body_lin_vel_w=np.zeros((8,2,3)),body_ang_vel_w=np.ones((8,2,3)))
    d=direct_reference(motion)
    np.testing.assert_array_equal(d['body_pos_w'][:,0,:2],p[:,0,:2])
    np.testing.assert_array_equal(d['body_pos_w'][0],p[0])
    np.testing.assert_allclose(d['body_lin_vel_w'][:,:,0],1.3)
    np.testing.assert_array_equal(d['joint_vel'],np.zeros((8,8)))
    assert not np.array_equal(d['body_lin_vel_w'][0],motion['body_lin_vel_w'][0])


def test_stop_requires_continuous_supported_hold_and_does_not_join_rebound():
    t=np.arange(0,2.01,.1);speed=np.zeros_like(t);eligible=np.ones_like(t,dtype=bool)
    speed[t<.4]=1.;speed[(t>=.7)&(t<.9)]=.2
    eligible[(t>=1.)&(t<1.3)]=False
    event=persistent_stop(t,speed,eligible,hold=.5)
    assert event['time_s']==pytest.approx(1.3)
    assert persistent_stop(t[:17],speed[:17],eligible[:17],hold=.5) is None
