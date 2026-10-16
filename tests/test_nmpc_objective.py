import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from nmpc_tracking_objective import (
    BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective, MujocoObjectiveState,
    append_contact_history,
)


def objective():
    profile=json.loads((Path(__file__).parents[1]/"config/car_drift_height_smooth_task.json").read_text())
    return HeightSmoothTrackingObjective(profile)


def state(beta=.4, yaw=.3, vz=.7):
    n=len(BODY_NAMES)
    p=np.zeros((n,3));p[:,2]=np.linspace(.75,.127,n)
    q=np.tile([np.cos(yaw/2),0.,0.,np.sin(yaw/2)],(n,1))
    v=np.tile([np.cos(yaw+beta),np.sin(yaw+beta),vz],(n,1))
    forces=np.zeros((n,3));forces[-2:,2]=100.
    return {"body_names":BODY_NAMES,"body_pos_w":p,"body_quat_w":q,
            "body_lin_vel_w":v,"body_ang_vel_w":np.zeros((n,3)),
            "body_net_forces_w":forces,"joint_pos":np.zeros(8),
            "soft_joint_pos_limits":np.tile([-2.,2.],(8,1)),
            "soft_joint_limits_source":"isaac_runtime",
            "contact_force_history_w":np.tile(forces[None],(3,1,1))}


def evaluate(o,s,ref=None,pre=None,previous_vz=.7,reset=False):
    ref=s if ref is None else ref
    pre=s if pre is None else pre
    return o.evaluate(s,o.hold_reference(ref,pre),np.zeros(8),np.zeros(8),previous_vz,reset_sample=reset)


def test_objective_has_no_absolute_height_or_target_vz_cost():
    o=objective();s=state();r=copy.deepcopy(s)
    expected=evaluate(o,s,r)
    shifted=copy.deepcopy(s);shifted["body_pos_w"][:,2]+=10.
    shifted["body_lin_vel_w"][:,2]=3.2
    result=evaluate(o,shifted,r,previous_vz=3.2)
    assert result["raw_terms"]==pytest.approx(expected["raw_terms"])
    assert result["stage_cost"]==pytest.approx(expected["stage_cost"])
    assert o.contract()["absolute_height_target"] is False
    assert o.contract()["reference_vertical_velocity_target"] is False


def test_large_reference_sideslip_is_rewarded_not_pushed_toward_zero():
    o=objective();ref=state(beta=.65)
    matched=evaluate(o,ref)
    zero_slip=state(beta=0.)
    unmatched=evaluate(o,zero_slip,ref)
    assert matched["raw_terms"]["drift_sideslip"]==pytest.approx(1.)
    assert matched["diagnostics"]["target_beta_rad"]==pytest.approx(.65)
    assert unmatched["raw_terms"]["drift_sideslip"]<1e-10
    assert matched["stage_cost"]<unmatched["stage_cost"]


def test_orientation_target_is_current_reference_heading_not_zero():
    o=objective();ref=state(yaw=1.2)
    assert evaluate(o,ref)["raw_terms"]["motion_global_anchor_ori"]==pytest.approx(1.)
    wrong=state(yaw=0.)
    assert evaluate(o,wrong,ref)["raw_terms"]["motion_global_anchor_ori"]<.001


def test_smoothness_uses_previous_com_vz_and_is_reset_safe():
    o=objective();s=state(vz=3.)
    assert evaluate(o,s,previous_vz=3.)["raw_terms"]["height_vertical_acceleration"]==0.
    value=evaluate(o,s,previous_vz=2.9)["raw_terms"]["height_vertical_acceleration"]
    assert value==pytest.approx(1-np.exp(-1.))
    assert evaluate(o,s,previous_vz=None,reset=True)["raw_terms"]["height_vertical_acceleration"]==0.
    with pytest.raises(ValueError,match="previous_base_com_vz"):
        evaluate(o,s,previous_vz=None)


def test_body_target_reanchoring_is_frozen_before_action():
    o=objective();pre=state(yaw=0.)
    pre["body_pos_w"][:,0]=np.linspace(0.,.4,len(BODY_NAMES))
    ref=copy.deepcopy(pre)
    held=o.hold_reference(ref,pre)
    post=copy.deepcopy(pre);post["body_pos_w"][:,0]+=.5
    actual=o.evaluate(post,held,np.zeros(8),np.zeros(8),.7)
    incorrectly_reanchored=evaluate(o,post,ref,pre=post)
    assert actual["raw_terms"]["motion_body_pos"]==pytest.approx(np.exp(-.5**2/.3**2))
    assert incorrectly_reanchored["raw_terms"]["motion_body_pos"]==pytest.approx(1.)
    # Mutating caller-owned reference does not alter held targets.
    ref["body_pos_w"][:]=999.
    assert np.max(held.body_pos_w)<1.


def test_wheel_spin_orientation_is_masked_but_body_orientation_is_not():
    o=objective();ref=state(yaw=0.)
    s=copy.deepcopy(ref)
    s["body_quat_w"][-2:]=[0.,0.,1.,0.]
    s["body_ang_vel_w"][-2:]=[0.,100.,0.]
    result=evaluate(o,s,ref)
    assert result["raw_terms"]["motion_body_ori"]==pytest.approx(1.)
    assert result["raw_terms"]["motion_body_ang_vel"]==pytest.approx(1.)
    s["body_quat_w"][1]=[0.,1.,0.,0.]
    assert evaluate(o,s,ref)["raw_terms"]["motion_body_ori"]<1.


def test_sensor_history_joint_limit_and_action_rate_terms_are_explicit():
    o=objective();s=state()
    s["contact_force_history_w"][0,1,0]=2.
    s["joint_pos"][0]=2.3
    action=np.ones(8)*.25
    r=o.evaluate(s,o.hold_reference(s,s),action,np.zeros(8),.7)
    assert r["raw_terms"]["undesired_contacts"]==1.
    assert r["raw_terms"]["joint_limit"]==pytest.approx(.3)
    assert r["raw_terms"]["action_rate_l2"]==pytest.approx(.5)
    assert r["raw_terms"]["drift_grounded"]==1.
    assert r["surrogate_flags"]==[]
    del s["contact_force_history_w"]
    s["soft_joint_limits_source"]="mujoco_ranges_times_0.95_surrogate"
    r=evaluate(o,s)
    assert r["raw_terms"]["undesired_contacts"]==0.
    assert len(r["surrogate_flags"])==2
    assert r["stage_cost"]==pytest.approx(o.dt*(r["reward_upper_bound"]-r["total_reward"]))


def test_force_history_tracks_physics_substeps_and_reset_zeros():
    force=np.ones((len(BODY_NAMES),3))
    history=append_contact_history(None,force)
    np.testing.assert_array_equal(history[:2],0.)
    for multiplier in (2.,3.,4.):
        history=append_contact_history(history,multiplier*force)
    np.testing.assert_array_equal(history[:,0,0],[2.,3.,4.])


def test_joint_limit_penalty_masks_unbounded_wheel_phase_like_tron_training():
    o=objective();s=state()
    s["joint_pos"][-2:]=[100.,-100.]
    assert evaluate(o,s)["raw_terms"]["joint_limit"]==0.
    s["joint_pos"][5]=-2.25
    assert evaluate(o,s)["raw_terms"]["joint_limit"]==pytest.approx(.25)


def test_mujoco_extraction_uses_each_link_com_not_link_origin_velocity():
    mujoco=pytest.importorskip("mujoco")
    # Synthetic model avoids private robot assets; body names match the task.
    bodies="".join(
        f'<body name="{name}" pos="0 0 -.01">'
        '<inertial pos=".1 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        f'<joint name="{joint}" type="hinge" axis="0 1 0" range="-2 2"/>'
        '</body>' for name,joint in zip(BODY_NAMES[1:],JOINT_NAMES)
    )
    xml=(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="base_Link" pos="0 0 1"><freejoint/>'
        '<inertial pos=".1 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        f'{bodies}</body></worldbody></mujoco>'
    )
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model);data.qvel[3:6]=[0.,0.,2.]
    mujoco.mj_forward(model,data)
    extractor=MujocoObjectiveState(model)
    actual=extractor.extract(data)
    # Base origin is stationary, but its COM at x=.1 has vy=.2 m/s.
    np.testing.assert_allclose(actual["body_pos_w"][0],[0.,0.,1.])
    np.testing.assert_allclose(actual["body_lin_vel_w"][0],[0.,.2,0.],atol=1e-12)
    np.testing.assert_allclose(actual["body_ang_vel_w"][0],[0.,0.,2.],atol=1e-12)
    assert actual["soft_joint_limits_source"]=="mujoco_ranges_times_0.95_surrogate"
    np.testing.assert_allclose(actual["soft_joint_pos_limits"][0],[-1.9,1.9])


def test_objective_requires_held_reference_and_valid_body_order():
    o=objective();s=state()
    with pytest.raises(TypeError,match="PRE-action"):
        o.evaluate(s,s,np.zeros(8),np.zeros(8),.7)
    wrong=copy.deepcopy(s);wrong["body_names"]=tuple(reversed(BODY_NAMES))
    with pytest.raises(ValueError,match="body order"):
        o.hold_reference(wrong,s)
