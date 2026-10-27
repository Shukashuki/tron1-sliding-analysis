import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hockey_physx_probe import hockey_metrics
from prepare_hockey_physx_reference import stopping_speed


def sample():
    n=801
    trace=dict(root_pos=np.zeros((n,3)),root_quat=np.tile([1.,0,0,0],(n,1)),
        root_com_vel=np.zeros((n,3)),normal_fz=np.tile([100.,0],(n,1)),
        lateral_slip_rms=np.tile([.2,0],(n,1)),utilization_proxy=np.tile([1.,0],(n,1)),
        friction_anchor_power=np.tile([-10.,0],(n,1)))
    ref=dict(body_pos_w=np.zeros((101,1,3)))
    return trace,ref


def test_stationary_beta_unavailable_and_fallen_stop_not_success():
    trace,ref=sample();report=hockey_metrics(trace,ref)
    assert report['body_beta_absolute_peak_deg'] is None
    assert report['independent_hockey_evaluation_required']
    assert not report['task_success_claimed']
    trace['root_quat'][:]=[0.,1.,0.,0.]
    report=hockey_metrics(trace,ref)
    assert not report['task_success_claimed']
    assert report['upright_supported_stop_hold_s']==0


def test_slip_needs_load_saturation_and_dissipation_on_same_wheel():
    trace,ref=sample()
    for field,value in [('normal_fz',[0,100]),('utilization_proxy',[.5,0]),('friction_anchor_power',[10,0])]:
        modified={k:v.copy() for k,v in trace.items()};modified[field][:]=value
        assert hockey_metrics(modified,ref)['per_wheel_loaded_slip_saturation_dissipation_longest_s']==[0.,0.]


def test_stop_command_is_monotonic_and_holds_zero():
    t=np.arange(139)*.02
    v=stopping_speed(t,1.3,.4,1.8)
    np.testing.assert_allclose(v[t<=.4],1.3)
    np.testing.assert_allclose(v[t>=1.8],0.)
    assert np.all(np.diff(v)<=1e-12)
