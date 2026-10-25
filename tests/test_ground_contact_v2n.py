"""N arithmetic, real Torch collection and frozen experiment invariants.

CPU fixtures do not establish actual PhysX contact quality or task success.
"""
import ast
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
sys.path.insert(0, str(ROOT / "scripts"))
import ground_contact_reward_v2n as definition
import ground_contact_reward_v2m as old_m
import ppo_ground_contact_v2n as collector
import continue_ground_contact_v2n_validation as continuation


@pytest.mark.parametrize("load", [0., 5., 5.0001, 25., 100., 200.])
def test_exact_threshold_and_conditional_K_M_equivalence(load):
    forces = np.tile([load, 0.], (8, 1))
    args = (forces, 200., .8, .9, .7, 1.)
    old_k = definition.previous.reward_numpy(*args)
    new = definition.reward_numpy(*args)
    expected = old_m.reward_numpy(*args) if load <= 5. else old_k
    np.testing.assert_array_equal(new["raw_reward"], expected["raw_reward"])
    assert new["binary_support_gap"] == float(load <= 5.)
    assert new["support_gap_penalty"] == pytest.approx(
        np.clip((100.-load)/95., 0., 1.) + 3.*float(load <= 5.))
    for name in old_k.keys() - {"raw_reward", "support_gap_penalty"}:
        np.testing.assert_array_equal(new[name], old_k[name])


@pytest.mark.parametrize("missing", [0, 1, 3, 8])
@pytest.mark.parametrize("beta", [0., .7, 1.])
def test_random_windows_change_only_gap_cost_and_preserve_reference_beta(missing, beta):
    rng = np.random.default_rng(42)
    forces = rng.uniform(5.01, 180., (13, 8, 2))
    forces[:, :missing] = 0.
    args = (forces, 200., .8, .9, beta, 1.)
    k = definition.previous.reward_numpy(*args)
    value = definition.reward_numpy(*args)
    expected = old_m.reward_numpy(*args) if missing else k
    np.testing.assert_array_equal(value["raw_reward"], expected["raw_reward"])
    np.testing.assert_array_equal(value["G_gated_support_bonus"], k["G_gated_support_bonus"])
    np.testing.assert_array_equal(value["binary_gap_penalty"], np.full(13, 3.*bool(missing)))
    if missing:
        np.testing.assert_array_equal(value["support_gap_penalty"], np.full(13, 4.))


def test_single_alternating_support_and_low_speed_beta_gate():
    single = np.tile([200., 0.], (8, 1))
    alternating = single.copy(); alternating[1::2] = [0., 200.]
    for force in (single, alternating, np.full((8, 2), 100.)):
        assert definition.reward_numpy(force, 200., 1., 1., 1., 1.)["raw_reward"] == 1.
        assert definition.reward_numpy(force, 200., 1., 1., 0., .3)["raw_reward"] == 1.
    assert definition.reward_numpy(np.zeros((8, 2)), 200., 0., 0., 0., 1.)["raw_reward"] == -4.


def test_torch_parity_and_no_host_reads_during_hot_path():
    import torch
    rng = np.random.default_rng(8)
    for dtype in (torch.float32, torch.float64):
        force = torch.tensor(rng.uniform(0., 150., (17, 8, 2)), dtype=dtype)
        force[::2, 3] = 0.
        expected = definition.reward_numpy(force.numpy(), 200., .8, .9, .7, 1.)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host bool")):
            actual = definition.reward_torch(force, 200., .8, .9, .7, 1., validate=False)
        for name in expected:
            np.testing.assert_allclose(actual[name].numpy(), expected[name], rtol=3e-6, atol=3e-6)
    force[0, 1, 0] = float("nan")
    with pytest.raises(ValueError):
        definition.reward_numpy(force.numpy(), 200., 1., 1., 1., 1.)
    assert torch.isnan(definition.reward_torch(force, 200., 1., 1., 1., 1., validate=False)["raw_reward"][0])
    for name in definition.DEFAULT_SETTINGS:
        bad = copy.deepcopy(definition.DEFAULT_SETTINGS)
        bad[name] = "changed" if isinstance(bad[name], str) else bad[name]+.1
        with pytest.raises(ValueError): definition.validate_settings(bad)


@pytest.fixture
def native_fixture():
    spec = importlib.util.spec_from_file_location("_contact_n_fixture", ROOT/"tests/test_ppo_ground_contact_v2m.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with patch.object(module, "contact", collector):
        fixture = module.GroundContactV2MTest("test_balanced_single_wheel_and_alternating_support_are_equivalent")
        fixture.setUp()
        try:
            owner = collector.GroundContactV2N(fixture.env, allow_cpu_fixture=True)
            fixture.owners.append(owner)
            yield fixture, owner
        finally: fixture.doCleanups()


def test_eight_physical_samples_light_load_and_actual_gap(native_fixture):
    import torch
    fixture, owner = native_fixture
    torch.testing.assert_close(fixture.env.step(), torch.ones(2))
    fixture.env.force_sequence = [torch.tensor([[6., 0.], [0., 6.]])] * 8
    force = torch.stack(fixture.env.force_sequence, dim=1)
    expected = definition.reward_torch(force, owner.mg, fixture.anchor, fixture.speed, fixture.beta, 1.)["raw_reward"]
    torch.testing.assert_close(fixture.env.step(), expected)
    fixture.env.force_sequence[3] = torch.zeros((2, 2))
    # Seven light-load samples and a zero minimum: .75*(7/8)*(6/(.5mg)).
    torch.testing.assert_close(fixture.env.step(), torch.tensor([-3.92125, -3.960625]))
    assert (owner.completed_steps, owner.unique_reads, owner.total_samples) == (3, 3, 24)
    assert fixture.env.mass_reads == 1
    audit = owner.audit(health=True)
    assert audit["device_finite_normal_force_health"]
    assert audit["hybrid_margin_plus_binary_gap"]
    assert audit["support_gap_coefficient_current"] == 1.
    assert audit["binary_any_gap_coefficient"] == 3.


@pytest.mark.parametrize("defect", ["skip_reward", "reset_before_reward"])
def test_lifecycle_defects_fail_closed(native_fixture, defect):
    fixture, _ = native_fixture
    setattr(fixture.env, defect, True)
    with pytest.raises(ValueError): fixture.env.step()


def test_same_protocol_dynamics_DR_budget_and_acceptance_as_K():
    k = json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    n = json.loads((ROOT/"config/ppo_ground_contact_v2n_400hz_v1.json").read_text())
    assert n["ground_contact_reward_v2"] == definition.validate_settings()
    candidate = copy.deepcopy(n)
    for key in ("cohort_id", "reward_definition_version", "experiment_design"):
        candidate[key] = k[key]
    candidate["ground_contact_reward_v2"] = k["ground_contact_reward_v2"]
    assert candidate == k
    assert n["cohort_id"] == continuation.COHORT_ID
    assert n["experiment_design"]["existing_beta_reward_weight"] == 2.


def test_physical_capture_and_lifecycle_AST_identical_to_K():
    trees = [ast.parse((ROOT/"scripts"/name).read_text()) for name in
             ("ppo_ground_contact_v2k.py", "ppo_ground_contact_v2n.py")]
    methods = [{n.name: ast.dump(n, include_attributes=False) for cls in t.body
                if isinstance(cls, ast.ClassDef) for n in cls.body if isinstance(n, ast.FunctionDef)} for t in trees]
    for name in ("_matrix", "update", "_complete", "reset", "close", "_attached", "step", "reward", "_phase"):
        assert methods[0][name] == methods[1][name], name


def test_all_continuation_pins_match_actual_frozen_sources():
    assert continuation.PROTOCOL_SHA256 == hashlib.sha256((ROOT/"config/ppo_ground_contact_v2n_400hz_v1.json").read_bytes()).hexdigest()
    for name, digest in continuation.DEPENDENCIES.items():
        assert hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() == digest, name
    metadata = definition.formula_metadata()
    assert metadata["same_as_K_when_all_samples_supported"]
    assert metadata["same_total_unsupported_window_cost_as_M"]
    assert not metadata["continuous_contact_or_task_success_certified"]
