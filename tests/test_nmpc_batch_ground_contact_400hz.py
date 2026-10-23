"""V2 replaces one term only, honors eight floor samples and discounted cost."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import nmpc_batch_ground_contact_400hz as mod
from ground_contact_reward_v2 import reward_numpy
from nmpc_batch_objective_400hz import BatchHeightSmoothTrackingObjective400Hz
from nmpc_tracking_objective import BODY_NAMES


def fixture():
    shape=(2,3);n=9;native={}
    for name,width in (("body_pos_w",3),("body_quat_w",4),("body_lin_vel_w",3),("body_ang_vel_w",3)):
        pre=np.zeros((*shape,n,width));post=np.zeros((*shape,8,n,width))
        if width==4:pre[...,0]=1;post[...,0]=1
        if name=="body_lin_vel_w":pre[...,0]=1;post[...,0]=1
        native["pre_"+name]=pre;native[name]=post
    native.update(qpos=np.zeros((*shape,8,15)),body_net_forces_w=np.zeros((*shape,8,n,3)),
        contact_force_history_w=np.zeros((*shape,3,n,3)),
        interval_time_s=np.broadcast_to(np.arange(1,9)*.0025,(*shape,8)).copy(),
        ground_contact_count=np.ones((*shape,8,2),dtype=np.int32),ground_normal_force_w=np.zeros((*shape,8,2,3)))
    native["ground_normal_force_w"][...,0,2]=100. # Single wheel permitted, other wheel unloaded.
    reference={"body_names":BODY_NAMES,**{name:native["pre_"+name].copy() for name in
        ("body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w")}}
    profile=json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text());profile["action_rate_weight"]=-.5
    return native,reference,profile


def evaluate(objective,native,reference):
    return objective.evaluate_native(native,reference,np.zeros((2,3,8)),np.zeros((2,8)),np.zeros(2),
        joint_qpos_indices=np.arange(7,15),soft_joint_pos_limits=np.tile([-2.,2.],(8,1)),
        soft_joint_limits_source="isaac_runtime",initial_reset_samples=True)


def test_only_grounded_changes_and_all_accounting_is_consistent():
    native,reference,profile=fixture();before=copy.deepcopy(native)
    legacy=evaluate(BatchHeightSmoothTrackingObjective400Hz(profile,discount_factor=.99),native,reference)
    objective=mod.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.)
    actual=evaluate(objective,native,reference)
    for key in native:np.testing.assert_array_equal(native[key],before[key])
    assert len(actual["raw_terms"])==15 and len(actual["weighted_terms"])==15
    for name,value in legacy["raw_terms"].items():
        if name!="drift_grounded":np.testing.assert_array_equal(actual["raw_terms"][name],value)
    expected=reward_numpy(native["ground_normal_force_w"][...,2],100.,
        legacy["raw_terms"]["motion_global_anchor_pos"],legacy["raw_terms"]["drift_speed"])
    np.testing.assert_array_equal(actual["raw_terms"]["drift_grounded"],expected["raw_reward"])
    np.testing.assert_allclose(actual["total_reward"],sum(actual["weighted_terms"].values()),rtol=0,atol=0)
    np.testing.assert_allclose(actual["stage_cost"],sum(actual["term_costs"].values()),rtol=0,atol=1e-15)
    np.testing.assert_allclose(actual["candidate_costs"],(actual["stage_cost"]*.99**np.arange(3)).sum(-1))
    assert objective.contract()["stage_reward_formulas_changed"] is True
    assert objective.contract()["ground_contact_reward_v2"]["bilateral_support_required"] is False


def test_sustained_single_or_alternating_wheel_support_is_allowed():
    native,reference,profile=fixture();objective=mod.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.)
    first=evaluate(objective,native,reference)["raw_terms"]["drift_grounded"]
    native["ground_normal_force_w"][...,::2,0,2]=0
    native["ground_normal_force_w"][...,::2,1,2]=100
    np.testing.assert_array_equal(evaluate(objective,native,reference)["raw_terms"]["drift_grounded"],first)
    native["ground_normal_force_w"][...,4,:,:]=0
    assert np.all(evaluate(objective,native,reference)["raw_terms"]["drift_grounded"]<first)


@pytest.mark.parametrize("bad",["missing","wrongshape","tangent","negative_count","fractional_count","clock","negative_force"])
def test_missing_or_malformed_ground_channels_refused(bad):
    native,reference,profile=fixture()
    if bad=="missing":del native["ground_normal_force_w"]
    if bad=="wrongshape":native["ground_normal_force_w"]=native["ground_normal_force_w"][...,:4,:,:]
    if bad=="tangent":native["ground_normal_force_w"][0,0,0,0,0]=1.
    if bad=="negative_count":native["ground_contact_count"][0,0,0,0]=-1
    if bad=="fractional_count":native["ground_contact_count"]=native["ground_contact_count"].astype(float);native["ground_contact_count"][0,0,0,0]=.5
    if bad=="clock":native["interval_time_s"]*=2
    if bad=="negative_force":native["ground_normal_force_w"][0,0,0,0,2]=-1
    with pytest.raises((ValueError,KeyError)):
        evaluate(mod.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.),native,reference)
