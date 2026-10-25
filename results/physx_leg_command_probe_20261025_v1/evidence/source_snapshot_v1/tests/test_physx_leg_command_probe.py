"""Actuator boundary invariants, including unchanged raw targets and wheels."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from physx_leg_command_probe import LegCommandProbe
import eval_physx_leg_command_probe as evaluator


def fixture(mode):
    data = SimpleNamespace(joint_pos_target=torch.zeros((1, 8)))
    robot = SimpleNamespace(data=data, joint_names=["leg"+str(i) for i in range(6)]+["wheelL", "wheelR"])
    def set_target(target, joint_ids):
        data.joint_pos_target[:, joint_ids] = target
    robot.set_joint_position_target = set_target
    term = SimpleNamespace(_asset=robot, _joint_ids=list(range(6)), processed_actions=torch.ones((1, 6)))
    term.apply_actions = lambda: set_target(term.processed_actions, term._joint_ids)
    wheels = SimpleNamespace(_joint_ids=[6, 7], processed_actions=torch.tensor([[3., -4.]]))
    env = SimpleNamespace(num_envs=1, physics_dt=.0025, step_dt=.02, device="cpu", _sim_step_counter=0,
        scene={"robot":robot}, action_manager=SimpleNamespace(get_term=lambda name: term if name=="joint_pos" else wheels))
    return env, term, wheels, LegCommandProbe(env, mode, allow_cpu_fixture=True)


@pytest.mark.parametrize("mode", ("passthrough", "linear_position_8substeps"))
def test_exact_endpoint_raw_target_and_wheel_commands_unchanged(mode):
    env, term, wheels, owner = fixture(mode)
    for step in range(1, 9):
        env._sim_step_counter = step
        term.apply_actions()
        wanted = 1. if mode=="passthrough" else step/8.
        torch.testing.assert_close(env.scene["robot"].data.joint_pos_target[:, :6], torch.full((1, 6), wanted), rtol=0, atol=0)
        torch.testing.assert_close(term.processed_actions, torch.ones((1, 6)), rtol=0, atol=0)
        torch.testing.assert_close(wheels.processed_actions, torch.tensor([[3., -4.]]), rtol=0, atol=0)
    term.processed_actions.fill_(-1.)
    env._sim_step_counter = 9
    term.apply_actions()
    wanted = -1. if mode=="passthrough" else .75
    torch.testing.assert_close(env.scene["robot"].data.joint_pos_target[:, :6], torch.full((1, 6), wanted), rtol=0, atol=0)


def test_duplicate_native_application_is_rejected():
    env, term, wheels, owner = fixture("linear_position_8substeps")
    env._sim_step_counter = 1
    term.apply_actions()
    with pytest.raises(AssertionError, match="consecutive physical"):
        term.apply_actions()


def test_full_episode_command_records_have_verified_exact_endpoints(tmp_path):
    env, term, wheels, owner = fixture("linear_position_8substeps")
    for step in range(1, 2793):
        if (step-1)%8 == 0:
            term.processed_actions.fill_(.625 if (step-1)//8%2 else -.375)
        env._sim_step_counter = step
        term.apply_actions()
    proof = owner.save(tmp_path/"commands.npz")
    assert proof["physical_steps"] == 2792 and proof["policy_decisions"] == 349
    assert proof["command_endpoints_exact"] and not proof["wheel_effort_interpolation"]
    assert proof["validation_only_intervention"] and not proof["trained_for_interpolated_execution"]


def test_parent_hooks_preserve_the_immutable_model_evaluator():
    parent = Path(__file__).resolve().parents[1]/"scripts/eval_physx_wheel_contact_models.py"
    original = parent.read_text()
    generated = evaluator.materialize(original)
    assert generated.count("_probe_install(env)") == 1 and generated.count("_probe_finish(") == 1
    assert parent.read_text() == original
    with pytest.raises(AssertionError):
        evaluator.materialize(original+"\n")
