"""3D axial orientation:spin/sign invariance,coupled yaw/camber and invalid data."""
from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import car_wheel_axis_tracking as targets
import car_taskspace_tracking as heading
import train_contact_wheel_axis as training
import verify_contact_wheel_axis as verification
import height_smoothness as height
import run_contact_wheel_axis as driver
import eval_contact_wheel_axis as evaluation
import continue_contact_wheel_axis as continuation


def q(axis, angle):
    v=torch.tensor(axis,dtype=torch.float64)
    return torch.cat((torch.tensor([math.cos(angle/2)],dtype=torch.float64),v*math.sin(angle/2)))


def mul(a,b):
    return torch.cat(((a[0]*b[0]-(a[1:]*b[1:]).sum())[None],a[0]*b[1:]+b[0]*a[1:]+torch.linalg.cross(a[1:],b[1:])))


@pytest.mark.parametrize("spin",[-400.,-math.pi,0.,2.1,100.])
def test_spin_invariance_and_camber_detection(spin):
    ref=q([0,0,1],.2)
    actual=mul(mul(ref,q([1,0,0],.5)),q([0,1,0],spin))
    value=targets.axis_quality(ref.expand(1,2,4),actual.expand(1,2,4))
    torch.testing.assert_close(value,torch.tensor([math.exp(-(.5/.4)**2)],dtype=torch.float64))
    # Heading-only control sees zero error for this camber rotation.
    torch.testing.assert_close(heading.heading_quality(ref.expand(1,2,4),actual.expand(1,2,4)),torch.ones(1,dtype=torch.float64))


def test_coupled_yaw_camber_matches_vector_geometry_and_world_rotation():
    ref=q([0,0,1],0.);actual=mul(q([0,0,1],.3),q([1,0,0],.5))
    angle=math.acos(abs(math.cos(.3)*math.cos(.5)))
    expected=torch.tensor([math.exp(-(angle/.4)**2)],dtype=torch.float64)
    torch.testing.assert_close(targets.axis_quality(ref.expand(1,2,4),actual.expand(1,2,4)),expected)
    global_q=mul(q([1,0,0],.8),q([0,0,1],-.4))
    torch.testing.assert_close(targets.axis_quality(mul(global_q,ref).expand(1,2,4),mul(global_q,actual).expand(1,2,4)),expected)


def test_axle_sign_and_quaternion_sign_are_equivalent_but_flat_wheel_is_not():
    ref=q([0,0,1],0.).expand(1,2,4)
    torch.testing.assert_close(targets.axis_quality(ref,-ref),torch.ones(1,dtype=torch.float64))
    reverse=q([1,0,0],math.pi).expand(1,2,4)
    torch.testing.assert_close(targets.axis_quality(ref,reverse),torch.ones(1,dtype=torch.float64))
    flat=q([1,0,0],math.pi/2).expand(1,2,4)
    torch.testing.assert_close(targets.axis_quality(ref,flat),torch.tensor([math.exp(-((math.pi/2)/.4)**2)],dtype=torch.float64))


def test_two_wheel_mean_and_finite_zero_vs_nonfinite_poison():
    ref=q([0,0,1],0.).expand(1,2,4)
    actual=torch.stack((q([0,0,1],0.),q([1,0,0],math.pi/3)))[None]
    expected=math.exp(-((math.pi/3)**2/2)/(.4**2))
    torch.testing.assert_close(targets.axis_quality(ref,actual),torch.tensor([expected],dtype=torch.float64))
    assert targets.axis_quality(ref,torch.zeros_like(ref)).item()==0.
    for value in [float('nan'),float('inf')]:
        bad=ref.clone();bad[0,0,0]=value
        assert not torch.isfinite(targets.axis_quality(ref,bad)).all()
    with pytest.raises(ValueError):targets.axis_quality(ref,actual,.3)


def test_named_wheels_and_only_original_four_bootstrap_targets_change():
    names=["base_Link","abad_L_Link","abad_R_Link","hip_L_Link","hip_R_Link","knee_L_Link","knee_R_Link"]
    rewards=NS(motion_body_pos=NS(func=height.body_xy_tracking,weight=1.,params={"command_name":"motion","std":.3}),
        motion_body_lin_vel=NS(func=height.body_xy_velocity_tracking,weight=1.,params={"command_name":"motion","std":.5}),
        motion_body_ori=NS(func=object(),weight=1.,params={"command_name":"motion","std":.4,"body_names":names[:]}),
        motion_body_ang_vel=NS(func=object(),weight=1.,params={"command_name":"motion","std":.8,"body_names":names[:]}),untouched=object())
    before=rewards.untouched;proof=targets.configure(NS(rewards=rewards))
    assert rewards.untouched is before and rewards.motion_body_ori.func is targets.wheel_plane_axis_tracking
    assert proof["settings"]["orientation_term"]=="world_axial_3D_wheel_axis"
    assert not proof["settings"]["contact_reward_formula_changed"]
    ref=torch.stack((q([0,0,1],0.),q([0,0,1],.2),q([0,0,1],.3)))[None]
    robot=ref.clone();robot[:,0]=float('nan')
    c=NS(cfg=NS(body_names=["ignored_leg","wheel_R_Link","wheel_L_Link"]),body_quat_w=ref,robot_body_quat_w=robot)
    env=NS(command_manager=NS(get_term=lambda name:c))
    torch.testing.assert_close(targets.wheel_plane_axis_tracking(env,"motion",.4,targets.SETTINGS["wheel_bodies"]),torch.ones(1,dtype=torch.float64))


def test_frozen_training_and_verification_hooks_preserve_warm_accounting():
    source=training.materialize(Path(training.parent.__file__).read_text())
    assert "import car_wheel_axis_tracking as targets" in source and training.COHORT_ID in source
    assert "verify_contact_wheel_axis.py" in source
    assert "**_axis_actual_sources()" in source
    assert "initialize_runner" not in source  # Preserved in immutable warm parent,not rewritten here.
    v=verification.materialize((ROOT/"scripts/verify_contact_taskspace_refinement.py").read_text())
    assert "live_axis_reward_shape" in v and "three_dimensional_axial_orientation_verified=True" in v
    with pytest.raises(AssertionError):training.materialize(Path(training.parent.__file__).read_text()+'\n')


def test_native_and_driver_wrappers_preserve_lifecycle_and_exact_training_wait():
    source=driver.materialize((ROOT/"scripts/run_contact_taskspace_refinement.py").read_text())
    assert "three_dimensional_axial_orientation_verified" in source and "prior_pids" in source
    assert 'phases.append(("formal", 512, 200))' in source and "helper.query_gpus()" in source
    native=evaluation.materialize((ROOT/"scripts/eval_contact_warm_refinement.py").read_text())
    assert "import verify_contact_wheel_axis as verification" in native and "parent._finish(report,arrays,current)" in native
    wait=continuation.materialize((ROOT/"scripts/continue_contact_warm_refinement.py").read_text())
    assert "['K','M']" not in wait and "deadline" not in wait and "run_contact_wheel_axis.py" in wait
    assert "p['start_ticks']==launch['start_ticks'] and p['argv']==launch['argv']" in wait


def test_protocol_matches_control_except_reference_axial_orientation():
    p=training.read(ROOT/"config/contact_wheel_axis_400hz_v1.json")
    control=training.read(ROOT/"config/contact_taskspace_refinement_400hz_v1.json")
    for n in ("algorithm","warm_start","refinement_optimizer","variant_contact_settings","training_randomization", "validation","nominal_acceptance_before_training"):
        assert p[n]==control[n]
    assert p["task_space_tracking"]==targets.settings() and p["variants"]==["K"]
    assert p["algorithm"]["transitions"]==2457600 and not p["task_space_tracking"]["contact_reward_formula_changed"]
