"""K arithmetic/provenance only: no native physics or learned outcomes."""
from pathlib import Path
import copy
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2k as k


def forces(left=200., right=0., dtype=np.float64):
    return np.tile(np.array([left, right], dtype=dtype), (8, 1))


def evaluate(value, mg=200.):
    return k.reward_numpy(value, mg, 1., 1., 1., 1.)


@pytest.mark.parametrize("load,expected", [(0., 1.), (5., 1.), (6., 94/95),
    (50., 50/95), (100., 0.), (200., 0.)])
def test_continuous_margin_and_original_strict_threshold(load, expected):
    result = evaluate(forces(load))
    assert float(result["support_gap"]) == pytest.approx(expected)
    assert float(result["I_binary_support_gap"]) == float(load <= 5.)
    assert float(result["load_margin_target_n"]) == 100.
    assert -1. <= result["raw_reward"] <= 1.


def test_single_or_alternating_wheel_full_support_requires_no_bilateral_target():
    for value in (forces(), forces(0., 200.), np.array([[200., 0.], [0., 200.]]*4)):
        result = evaluate(value)
        assert result["support_gap"] == 0. and result["raw_reward"] == 1.


def test_weakest_physical_substep_not_mean_or_force_peak_determines_cost():
    value = forces()
    value[3] = [50., 0.]
    assert evaluate(value)["support_gap"] == pytest.approx(50/95)
    value[4] = [500., 0.]
    assert evaluate(value)["support_gap"] == pytest.approx(50/95)
    value[1] = [5., 0.]
    assert evaluate(value)["support_gap"] == 1.


def test_preserves_entire_I_positive_bonus_reference_beta_and_cost_bound():
    value = forces(60., 0.)
    before = value.copy()
    old = k.previous.reward_numpy(value, 200., .8, .7, .3, 1.)
    result = k.reward_numpy(value, 200., .8, .7, .3, 1.)
    assert np.array_equal(before, value)
    assert result["G_gated_support_bonus"] == old["G_gated_support_bonus"]
    assert result["raw_reward"] == pytest.approx(old["G_gated_support_bonus"]-40/95)
    low = k.reward_numpy(value, 200., .8, .7, 0., .3)
    assert low["beta_contact_gate"] == 1.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_torch_matches_numpy_for_batched_mass_and_wheel_exchange(dtype):
    rng = np.random.default_rng(17)
    value = rng.uniform(0., 170., (3, 8, 2)).astype(dtype)
    weight = np.array([120., 200., 300.], dtype=dtype)
    expected = k.reward_numpy(value, weight, .9, .8, .7, 1.)
    result = k.reward_torch(torch.from_numpy(value), torch.from_numpy(weight), .9, .8, .7, 1.)
    swapped = k.reward_numpy(value[..., ::-1], weight, .9, .8, .7, 1.)
    assert expected["raw_reward"].dtype == dtype
    for name in expected:
        np.testing.assert_allclose(result[name].numpy(), expected[name], rtol=2e-6, atol=2e-7)
        if name != "per_step_support":
            np.testing.assert_allclose(expected[name], swapped[name], rtol=0., atol=0.)


@pytest.mark.parametrize("weight", [0., 5., 10., float("nan"), float("inf")])
def test_invalid_or_degenerate_load_scale_is_rejected(weight):
    with pytest.raises(ValueError):
        evaluate(forces(), weight)
    with pytest.raises(ValueError):
        k.reward_torch(torch.from_numpy(forces()), weight, 1., 1., 1., 1.)


def test_validate_false_poisons_invalid_load_or_force_instead_of_hiding_it():
    value = torch.from_numpy(np.stack((forces(), forces(), forces())))
    value[2, 2, 0] = float("nan")
    result = k.reward_torch(value, torch.tensor([200., 5., 200.]), 1., 1., 1., 1., validate=False)
    assert torch.isfinite(result["raw_reward"][0])
    for field in result.values():
        assert torch.isnan(field[1:]).all()


def test_runtime_math_has_no_file_reads_or_host_tensor_scalar_reads():
    value = torch.from_numpy(forces())
    with patch.object(Path, "read_bytes", side_effect=AssertionError("No reward file reads")), \
            patch.object(torch.Tensor, "item", side_effect=AssertionError("No scalar transfer")), \
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("No list transfer")):
        result = k.reward_torch(value, 200., 1., 1., 1., 1., validate=False)
    assert result["raw_reward"] == 1.


def test_source_and_settings_are_frozen_without_claiming_task_success():
    k.verify_base()
    cfg = copy.deepcopy(k.DEFAULT_SETTINGS)
    cfg["support_load_fraction"] = .25
    with pytest.raises(ValueError):
        k.validate_settings(cfg)
    metadata = k.formula_metadata()
    assert metadata["original_support_acceptance_threshold_n"] == 5.
    assert metadata["supported_light_load_penalized"] is True
    assert metadata["support_gap_cost_ungated"] is True
    assert metadata["other14_reward_terms_changed"] is False
    assert metadata["new_angular_velocity_input"] is False
    assert metadata["continuous_contact_or_task_success_certified"] is False
