"""O cost semantics and actual Torch collector lifecycle; no task-success claim."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2o as definition
import ppo_ground_contact_v2o as contact


def test_exact_stance_side_clipping_and_wheel_actions_do_not_enter_cost():
    action = np.zeros(8); old = np.zeros(8)
    action[list(definition.RIGHT)] = .8
    assert definition.command_numpy([100., 0.], action, old)["stance_command_penalty"] == 0.
    action[list(definition.LEFT)] = .1
    value = definition.command_numpy([100., 0.], action, old)
    assert value["stance_command_penalty"] == pytest.approx(1-np.exp(-1.))
    np.testing.assert_allclose(value["stance_command_side_delta_mean_square_rad2"], [.01, .64])
    action[6:] = 50.
    assert definition.command_numpy([100., 0.], action, old)["stance_command_penalty"] == value["stance_command_penalty"]
    for force in ([0., 0.], [5., 5.]):
        assert definition.command_numpy(force, action, old)["stance_command_penalty"] == 0.
    assert definition.command_numpy([5.00001, 0.], action, old)["stance_command_penalty"] > 0.
    action[:] = 2.; old[:] = 1.
    assert definition.command_numpy([100., 100.], action, old)["stance_command_penalty"] == 0.


def test_only_new_bounded_command_cost_changes_K_even_when_beta_bonus_is_zero():
    force = np.full((8, 2), 100.)
    action = np.full(8, .1)
    for beta in (0., .4, 1.):
        parent = definition.previous.reward_numpy(force, 200., .8, .9, beta, 1.)
        value = definition.reward_numpy(force, 200., .8, .9, beta, 1.,
            preceding_Fz=[100., 100.], action=action, previous_action=np.zeros(8))
        for name in parent.keys()-{"raw_reward"}:
            np.testing.assert_array_equal(value[name], parent[name])
        assert value["raw_reward"] == pytest.approx(parent["raw_reward"]-(1-np.exp(-1.)))
    meta = definition.formula_metadata()
    assert meta["raw_reward_bounds"] == [-2., 1.]
    assert not meta["wheel_torque_actions_in_new_cost"]
    assert not meta["continuous_contact_or_task_success_certified"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_real_torch_numpy_broadcast_parity_and_no_hotpath_host_reads(dtype):
    import torch
    rng = np.random.default_rng(73)
    force = rng.uniform(0., 200., (3, 4, 8, 2)).astype(dtype)
    before = rng.uniform(0., 200., (3, 1, 2)).astype(dtype); before[0] = 0.
    action = rng.uniform(-2., 2., (3, 4, 8)).astype(dtype)
    old = rng.uniform(-2., 2., (4, 8)).astype(dtype)
    expected = definition.reward_numpy(force, 200., .8, .9, .7, 1., preceding_Fz=before,
                                     action=action, previous_action=old)
    with patch.object(Path, "read_bytes", side_effect=AssertionError("hot path file read")), \
         patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")), \
         patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host bool")):
        actual = definition.reward_torch(torch.tensor(force), 200., .8, .9, .7, 1.,
            preceding_Fz=torch.tensor(before), action=torch.tensor(action), previous_action=torch.tensor(old), validate=False)
    for name in expected:
        np.testing.assert_allclose(actual[name].numpy(), expected[name], rtol=4e-6, atol=4e-6)


def test_invalid_masked_force_or_command_still_poison_and_all_settings_frozen():
    import torch
    for force, current, old in [([0., 0.], [float("nan")]*8, [0.]*8),
                               ([float("nan"), 0.], [0.]*8, [0.]*8),
                               ([-1., 0.], [0.]*8, [0.]*8)]:
        with pytest.raises(ValueError): definition.command_numpy(force, current, old)
        result = definition.command_torch(torch.tensor(force), torch.tensor(current), torch.tensor(old), validate=False)
        assert torch.isnan(result["stance_command_penalty"])
    for name, expected in definition.DEFAULT_SETTINGS.items():
        bad = copy.deepcopy(definition.DEFAULT_SETTINGS)
        bad[name] = "changed" if isinstance(expected, str) else expected+.1
        with pytest.raises(ValueError): definition.validate_settings(bad)
    with pytest.raises(ValueError): definition.command_numpy(np.zeros((0, 2)), np.zeros((0, 8)), np.zeros((0, 8)))


@pytest.fixture
def environment():
    import torch
    spec = importlib.util.spec_from_file_location("_o_collector_fixture", ROOT/"tests/test_ppo_ground_contact_v2m.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.Env
    class ActionEnv(original):
        def step(self, action):
            self.input_action = action
            return super().step()
    with patch.object(module, "contact", contact), patch.object(module, "Env", ActionEnv):
        fixture = module.GroundContactV2MTest("test_balanced_single_wheel_and_alternating_support_are_equivalent")
        fixture.setUp()
        fixture.env.scene.robot.joint_names = list(contact.JOINTS)
        try:
            owner = contact.GroundContactV2O(fixture.env, allow_cpu_fixture=True)
            fixture.owners.append(owner)
            yield fixture, owner, torch.zeros((2, 8))
        finally: fixture.doCleanups()


def test_preceding_native_force_not_post_command_contact_and_recovery(environment):
    import torch
    fixture, owner, action = environment
    torch.testing.assert_close(fixture.env.step(action), torch.ones(2))
    fixture.env.force_sequence = [torch.zeros((2, 2)) for _ in range(8)]
    action[:] = .1
    expected = -1.-(1.-np.exp(-1.))
    torch.testing.assert_close(fixture.env.step(action), torch.full((2,), expected))
    fixture.env.force_sequence = [fixture.env.normal.clone() for _ in range(8)]
    action[:] = .2
    torch.testing.assert_close(fixture.env.step(action), torch.ones(2))
    assert (owner.total_samples, owner.command_reads, owner.unique_reads) == (24, 3, 3)
    assert fixture.env.mass_reads == 1 and fixture.env.scene.calls == 24


def test_other_leg_free_repeat_read_unique_and_autoreset_cache_clear(environment):
    import torch
    fixture, owner, action = environment
    fixture.env.read_twice = True
    fixture.env.step(action)
    action[0, list(definition.LEFT)] = .5
    action[1, :6] = .1
    expected = torch.tensor([1., np.exp(-1.)], dtype=torch.float32)
    torch.testing.assert_close(fixture.env.step(action), expected)
    assert owner.command_reads == 2
    fixture.env.reset_after_reward = True  # Fixture resets environment0 only.
    fixture.env.step(action)
    torch.testing.assert_close(owner.last_command[0], torch.zeros(8))
    torch.testing.assert_close(owner.last_normal[0], torch.zeros(2))
    torch.testing.assert_close(owner.last_command[1], action[1])
    fixture.env.reset_after_reward = False
    action[:] = .3
    value = fixture.env.step(action)
    assert value[0] == 1.
    assert value[1] < 1.
    audit = owner.audit(health=True)
    assert audit["reward_definition_version"] == "ground_contact_v2o_any_wheel"
    assert audit["command_unique_reward_windows"] == audit["unique_reward_windows"] == 4
    assert audit["device_finite_stance_command_health"]
    assert audit["immutable_v2k_collector_sha256"] == contact.PARENT_COLLECTOR_SHA256


def test_action_is_not_mutated_and_kinematic_nan_cannot_be_masked(environment):
    import torch
    fixture, owner, action = environment
    action[:] = 2.
    original = action.clone()
    fixture.env.step(action)
    assert fixture.env.input_action is action
    torch.testing.assert_close(action, original)
    torch.testing.assert_close(owner.last_command, torch.ones_like(action))
    fixture.env.step(action)
    assert owner.command_penalty_sum == 0.
    action[0, 0] = float("nan")
    assert torch.isnan(fixture.env.step(action)[0])
    assert not owner.audit(health=True)["device_finite_stance_command_health"]


@pytest.mark.parametrize("fault", ["skip_reward", "reset_before_reward"])
def test_parent_eight_sample_reset_and_reward_order_still_enforced(environment, fault):
    fixture, _, action = environment
    setattr(fixture.env, fault, True)
    with pytest.raises(ValueError): fixture.env.step(action)


def test_protocol_preserves_K_dynamics_DR_budget_and_acceptance():
    before = json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    after = json.loads((ROOT/"config/ppo_ground_contact_v2o_400hz_v1.json").read_text())
    assert after["ground_contact_reward_v2"] == definition.validate_settings()
    for name in before["ground_contact_reward_v2"]:
        if name != "candidate":
            assert before["ground_contact_reward_v2"][name] == after["ground_contact_reward_v2"][name]
    compared = copy.deepcopy(after)
    for name in ("cohort_id", "reward_definition_version", "experiment_design", "ground_contact_reward_v2"):
        compared[name] = before[name]
    assert compared == before


def test_actual_physical_capture_and_source_pins_reuse_frozen_K():
    for name in ("_matrix", "update", "_complete", "_phase", "_attached", "close"):
        assert getattr(contact.GroundContactV2O, name) is getattr(contact.parent.GroundContactV2K, name)
    assert contact.configure_sensors is contact.parent.configure_sensors
    import continue_ground_contact_v2o_validation as continuation
    assert continuation.PROTOCOL_SHA256 == hashlib.sha256((ROOT/"config/ppo_ground_contact_v2o_400hz_v1.json").read_bytes()).hexdigest()
    for name, digest in continuation.DEPENDENCIES.items():
        assert hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() == digest, name
