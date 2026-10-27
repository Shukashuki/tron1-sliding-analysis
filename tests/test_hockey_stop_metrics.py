import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hockey_stop_metrics import stopping_metrics


def test_real_deceleration_can_stop_without_speed_retention_requirement():
    t=np.arange(-.2,2.01,.01);speed=np.where(t<0,2.,np.maximum(2.-2*t,0))
    v=np.zeros((len(t),3));v[:,0]=speed;p=np.zeros_like(v)
    p[:,0]=np.cumsum(speed)*.01
    m=stopping_metrics(t,p,v,0.,metric_world_trajectory_verified=True)
    assert m['kinematic_stop_detected'] and m['entry_has_nontrivial_motion']
    assert m['braking_duration_s']==pytest.approx(.95,abs=.011)
    assert .95<m['stopping_displacement_xy_m']<1.05
    assert m['task_success_claimed'] is False
    assert not m['beta_valid_mask'][-1]


def test_short_low_speed_interval_is_not_a_completed_stop():
    t=np.arange(0,2,.01);v=np.zeros((len(t),3));v[:,0]=1.;v[50:70,0]=0
    assert not stopping_metrics(t,np.zeros_like(v),v,.1,metric_world_trajectory_verified=True)['kinematic_stop_detected']


def test_already_stationary_is_not_a_successful_hockey_stop():
    t=np.arange(0,1,.01);z=np.zeros((len(t),3));m=stopping_metrics(t,z,z,.1,metric_world_trajectory_verified=True)
    assert m['kinematic_stop_detected'] and not m['entry_has_nontrivial_motion'] and not m['task_success_claimed']


def test_hip_relative_pose_does_not_yield_fake_global_stopping_distance():
    m=stopping_metrics(None,None,None,0.)
    assert not m['available'] and 'stopping_displacement_xy_m' not in m


def test_missing_frames_cannot_fake_a_stop_hold():
    t=np.array([0.,.01,.02,1.,1.01]);z=np.zeros((len(t),3))
    with pytest.raises(ValueError):stopping_metrics(t,z,z,.01,metric_world_trajectory_verified=True)
