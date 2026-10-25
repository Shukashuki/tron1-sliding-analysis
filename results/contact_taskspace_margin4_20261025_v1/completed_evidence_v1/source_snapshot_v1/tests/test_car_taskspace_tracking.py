from pathlib import Path
from types import SimpleNamespace as NS
import copy
import math
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import car_taskspace_tracking as targets
import train_contact_taskspace_refinement as training
import eval_contact_taskspace_refinement as evaluation
import height_smoothness as height


def quaternion(axis, angle):
    v = torch.tensor(axis, dtype=torch.float64)
    return torch.cat((torch.tensor([math.cos(angle/2)], dtype=torch.float64), v*math.sin(angle/2)))


def multiply(a, b):
    w = a[0]*b[0]-(a[1:]*b[1:]).sum()
    xyz = a[0]*b[1:]+b[0]*a[1:]+torch.linalg.cross(a[1:], b[1:])
    return torch.cat((w[None], xyz))


def test_heading_ignores_arbitrary_wheel_spin_but_detects_yaw_and_is_axial():
    ref = quaternion([0, 0, 1], .2)
    yaw = quaternion([0, 0, 1], .5)
    for spin in [-100., -math.pi, 0., 2.1, 400.]:
        actual = multiply(yaw, quaternion([0, 1, 0], spin))
        value = targets.heading_quality(ref.expand(1, 2, 4), actual.expand(1, 2, 4))
        torch.testing.assert_close(value, torch.tensor([math.exp(-(.3/.4)**2)], dtype=torch.float64))
    flipped = multiply(ref, quaternion([1, 0, 0], math.pi))
    torch.testing.assert_close(targets.heading_quality(ref.expand(1, 2, 4), flipped.expand(1, 2, 4)), torch.ones(1, dtype=torch.float64))


def test_projection_singularity_is_zero_and_nonfinite_inputs_are_not_hidden():
    ref = quaternion([0, 0, 1], 0.).expand(1, 2, 4)
    vertical = quaternion([1, 0, 0], math.pi/2).expand(1, 2, 4)
    assert targets.heading_quality(ref, vertical).item() == 0.
    bad = ref.clone();bad[0, 0, 0] = float("nan")
    assert not torch.isfinite(targets.heading_quality(ref, bad)).all()
    assert targets.heading_quality(ref, torch.zeros_like(ref)).item() == 0.


def test_adapter_changes_only_the_four_declared_targets_and_preserves_weights():
    # The actual pinned command lists abad_R before hip_L. Selection is by
    # membership; the adapter must not invent an ordering requirement.
    old_names = ["base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link", "knee_L_Link", "knee_R_Link"]
    reward = NS(motion_body_pos=NS(func=height.body_xy_tracking, weight=1., params={"command_name":"motion", "std":.3}),
                motion_body_lin_vel=NS(func=height.body_xy_velocity_tracking, weight=1., params={"command_name":"motion", "std":.5}),
                motion_body_ori=NS(func=object(), weight=1., params={"command_name":"motion", "std":.4, "body_names":old_names[:]}),
                motion_body_ang_vel=NS(func=object(), weight=1., params={"command_name":"motion", "std":.8, "body_names":old_names[:]}),
                untouched=object())
    untouched = reward.untouched
    proof = targets.configure(NS(rewards=reward))
    assert reward.untouched is untouched
    assert reward.motion_body_ang_vel.params["body_names"] == ["base_Link"]
    assert reward.motion_body_pos.params["body_names"] == targets.SETTINGS["position_bodies"]
    assert reward.motion_body_ori.func is targets.wheel_plane_heading_tracking
    assert proof["four_tracking_terms_changed"] == list(targets.CHANGED)
    bad = targets.settings();bad["contact_reward_formula_changed"] = True
    with pytest.raises(ValueError): targets.settings(bad)


def test_reward_uses_world_reference_and_named_wheels_not_leg_or_spin_targets():
    ref = torch.stack([quaternion([0, 0, 1], a) for a in [1., .2, -.4]]).unsqueeze(0)
    robot = ref.clone();robot[:, 0] = float("nan")
    command = NS(cfg=NS(body_names=["ignored_leg", "wheel_R_Link", "wheel_L_Link"]),
                 body_quat_w=ref, robot_body_quat_w=robot)
    env = NS(command_manager=NS(get_term=lambda name: command))
    value = targets.wheel_plane_heading_tracking(env, "motion", .4, targets.SETTINGS["wheel_bodies"])
    torch.testing.assert_close(value, torch.ones(1, dtype=torch.float64))


def test_pinned_training_hooks_keep_warm_initialization_and_live_target_proofs():
    source = Path(training.frozen.__file__).read_text()
    generated = training.materialize_parent(source)
    assert generated.count("_taskspace_configure(cfg, protocol['task_space_tracking'])") == 1
    assert "task_space_tracking=_taskspace_audit" in generated
    assert "task['task_space_tracking'] = _taskspace_validate(protocol)" in generated
    assert "initialize_runner(runner,own.initial_checkpoint)" in generated
    with pytest.raises(AssertionError): training.materialize_parent(source+"\n")


def test_native_evaluation_requires_new_strict_training_admission():
    source = (ROOT/"scripts/eval_contact_warm_refinement.py").read_text()
    generated = evaluation.materialize_parent(source)
    assert "import verify_contact_taskspace_refinement as verification" in generated
    assert "taskspace_contact_refinement_native_evaluation" in generated
    with pytest.raises(AssertionError): evaluation.materialize_parent(source+"\n")
