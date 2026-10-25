from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts"),str(ROOT/"tests")]
import ground_contact_reward_v2r as arithmetic
import ppo_ground_contact_v2r as contact
import test_ppo_ground_contact_v2m as baseline


@pytest.fixture
def native_fixture():
    base = baseline.GroundContactV2MTest("test_balanced_single_wheel_and_alternating_support_are_equivalent")
    base.setUp()
    env = base.env
    # Deliberately interleave wheels and reorder legs: selection must be named.
    names = ["wheel_R_Joint","knee_L_Joint","abad_R_Joint","hip_L_Joint",
             "wheel_L_Joint","knee_R_Joint","abad_L_Joint","hip_R_Joint"]
    robot = env.scene.robot
    robot.joint_names = names
    robot.data = NS(joint_vel=torch.zeros((2,8)))
    env.velocity_sequence = [torch.zeros((2,8)) for _ in range(8)]
    old = env.scene.update
    def update(scene,dt,**kwargs):
        result = old(dt,**kwargs)
        robot.data.joint_vel.copy_(env.velocity_sequence[(env._sim_step_counter-1)%8])
        return result
    env.scene.update = MethodType(update,env.scene)
    owner = contact.GroundContactV2R(env,allow_cpu_fixture=True)
    base.owners.append(owner)
    try:yield env,owner
    finally:base.doCleanups()


def test_all_actual_substeps_not_endpoint_and_exclude_reordered_wheel_spin(native_fixture):
    env,owner = native_fixture
    for v in env.velocity_sequence:
        v[:,env.scene.robot.joint_names.index("wheel_L_Joint")] = 500.
        v[:,env.scene.robot.joint_names.index("wheel_R_Joint")] = -500.
    torch.testing.assert_close(env.step(),torch.ones(2))
    env.velocity_sequence[3][:,env.scene.robot.joint_names.index("hip_L_Joint")] = torch.tensor([2.,4.])
    expected = torch.tensor([1.-.1*4/48,1.-.1*16/48])
    env.read_twice = env.extra_refresh = True
    torch.testing.assert_close(env.step(),expected)
    report = owner.audit(health=True)
    assert report["leg_velocity_physics_samples"]==report["physics_samples"]==16
    assert report["leg_velocity_unique_reads"]==report["unique_reward_windows"]==2
    assert report["device_finite_leg_velocity_health"] and not report["wheel_spin_penalized"]
    assert report["reward_definition_version"]=="ground_contact_v2r_any_wheel"
    assert report["leg_joint_names"]==arithmetic.LEGS
    assert abs(report["mean_raw_reward"]-float((expected.mean()+1)/2))<1e-6


def test_invalid_middle_leg_sample_poison_is_latched_and_reset_cannot_hide_it(native_fixture):
    env,owner = native_fixture
    env.velocity_sequence[2][0,env.scene.robot.joint_names.index("abad_L_Joint")] = float("nan")
    result = env.step()
    assert torch.isnan(result[0]) and torch.isfinite(result[1])
    owner.reset([0])
    env.velocity_sequence[2].zero_()
    assert torch.isfinite(env.step()).all()
    report = owner.audit(health=True)
    assert not report["device_finite_leg_velocity_health"] and not report["device_finite_reward_health"]
    assert report["device_finite_normal_force_health"]


def test_new_window_has_no_previous_velocity_cost_and_support_rule_is_retained(native_fixture):
    env,owner = native_fixture
    for v in env.velocity_sequence:v[:,env.scene.robot.joint_names.index("knee_R_Joint")] = 3.
    torch.testing.assert_close(env.step(),torch.full((2,),.85))
    for v in env.velocity_sequence:v.zero_()
    env.force_sequence[3].zero_()
    torch.testing.assert_close(env.step(),torch.full((2,),-3.34375))
    assert owner.total_samples==owner.total_velocity_samples==16


def test_numpy_torch_and_runtime_formula_agree_without_force_gating(native_fixture):
    env,owner = native_fixture
    for i,v in enumerate(env.velocity_sequence):
        for j,n in enumerate(arithmetic.LEGS):v[:,env.scene.robot.joint_names.index(n)] = (i-j)*.2
    actual = env.step()
    args = (owner.forces,owner.mg,torch.ones(2),torch.ones(2),torch.ones(2),torch.ones(2),owner.leg_velocities)
    expected = arithmetic.reward_torch(*args)["raw_reward"]
    torch.testing.assert_close(actual,expected)
    numpy = arithmetic.reward_numpy(*(v.numpy() for v in args))["raw_reward"]
    np.testing.assert_allclose(numpy,actual.numpy(),rtol=1e-6)
    force = torch.zeros((1,8,2));vel = torch.ones((1,8,6),requires_grad=True)
    result = arithmetic.reward_torch(force,torch.tensor([100.]),torch.ones(1),torch.ones(1),torch.ones(1),torch.ones(1),vel)
    result["raw_reward"].sum().backward()
    assert torch.isfinite(vel.grad).all() and (vel.grad<0).all()
    assert abs(float(result["leg_velocity_cost"])-.1)<1e-6


def test_incomplete_physical_window_and_joint_interface_changes_are_fatal(native_fixture):
    env,owner = native_fixture
    env.force_sequence = env.force_sequence[:4]
    with pytest.raises(ValueError):env.step()
    assert owner.completed_steps==0
    owner.close()
    env.scene.robot.joint_names[0] = "abad_L_Joint"
    with pytest.raises(ValueError):contact.GroundContactV2R(env,allow_cpu_fixture=True)
    assert not hasattr(env,contact.MARKER)
