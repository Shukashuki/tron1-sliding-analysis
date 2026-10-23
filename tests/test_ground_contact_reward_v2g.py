"""G arithmetic candidate only; no simulator/collector/policy is exercised."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_reward_v2g as mod


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_only_positive_bonus_changes_and_negative_gap_is_ungated(dtype):
    rng = np.random.default_rng(42)
    force = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype); force[:, :, 3] = 0.
    before = force.copy(); quality = rng.random((2, 3)).astype(dtype)
    f = mod.base.reward_numpy(force, 100., .8, .9)
    g = mod.reward_numpy(force, 100., .8, .9, quality, 1.)
    for key in f.keys()-{"raw_reward"}: np.testing.assert_array_equal(f[key], g[key])
    np.testing.assert_array_equal(g["raw_reward"], f["A_gated_support_bonus"]*quality-f["support_gap_penalty"])
    np.testing.assert_array_equal(force, before)
    assert g["raw_reward"].dtype == dtype


@pytest.mark.parametrize("speed", (0., .2, .3))
def test_low_reference_speed_gate_neutral_not_zero_beta_target(speed):
    force = np.tile([100., 0.], (8, 1)); force[0] = 0.
    f = mod.base.reward_numpy(force, 100., .8, .9)
    g = mod.reward_numpy(force, 100., .8, .9, 0., speed)
    assert g["beta_contact_gate"] == 1.
    assert g["raw_reward"] == f["raw_reward"]


def test_nonzero_reference_beta_quality_not_absolute_zero_beta():
    force = np.tile([100., 0.], (8, 1))
    desired_beta = .6
    matching_quality = np.exp(-((desired_beta-desired_beta)/.12)**2)
    zero_beta_quality = np.exp(-((0.-desired_beta)/.12)**2)
    assert mod.reward_numpy(force, 100., 1., 1., matching_quality, 1.)["raw_reward"] == 1.
    assert mod.reward_numpy(force, 100., 1., 1., zero_beta_quality, 1.)["raw_reward"] < 1e-8
    # The arithmetic requires caller quality provenance; this is NOT a runtime wiring test.


def test_one_wheel_and_alternating_support_allowed_and_zero_gate_keeps_gap_cost():
    single = np.tile([100., 0.], (8, 1)); alternate = single.copy(); alternate[1::2] = [0., 100.]
    for force in (single, alternate):
        assert mod.reward_numpy(force, 100., 1., 1., 1., 1.)["raw_reward"] == 1.
    single[3] = 0.
    assert mod.reward_numpy(single, 100., 1., 1., 0., 1.)["raw_reward"] == -.125


@pytest.mark.parametrize("quality,speed", ((np.nan, 0.), (np.inf, 1.), (-.1, 1.), (1.1, 1.), (True, 1.), (.5, -1.), (.5, np.nan)))
def test_invalid_quality_not_concealed_by_stop_gate(quality, speed):
    with pytest.raises(ValueError): mod.reward_numpy(np.zeros((8, 2)), 100., 1., 1., quality, speed)


def test_settings_and_source_guards_no_hotpath_IO():
    for key, bad in (("beta_gate_low_reference_speed_m_s", .25), ("support_gap_weight", 0.), ("mean_weight", 1.)):
        settings = copy.deepcopy(mod.DEFAULT_SETTINGS); settings[key] = bad
        with pytest.raises(ValueError): mod.validate_settings(settings)
    with patch.object(Path, "read_bytes", side_effect=AssertionError("No reward filesystem I/O")):
        assert mod.reward_numpy(np.zeros((8, 2)), 100., 1., 1., 1., 1.)["raw_reward"] == -1.
    with patch.object(mod, "BASE_F_DEFINITION_SHA256", "0"*64):
        with pytest.raises(ValueError, match="V2f arithmetic"): mod.verify_base()
    meta = mod.formula_metadata()
    assert meta["support_gap_cost_unchanged_and_ungated"]
    assert not meta["existing_beta_reward_weight_or_std_changed"]
    assert meta["arithmetic_alone_does_not_certify_runtime_or_task_success"]


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_real_torch_numpy_parity_and_invalid_window_poison(dtype):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    force = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype); force[:, :, 2] = 0.
    quality = rng.random((2, 3)).astype(dtype); speed = np.array([[0., 1., .3], [.2, .5, 1.]], dtype=dtype)
    expected = mod.reward_numpy(force, 100., .8, .9, quality, speed)
    actual = mod.reward_torch(torch.from_numpy(force), 100., .8, .9, torch.from_numpy(quality), torch.from_numpy(speed))
    for key in expected: np.testing.assert_allclose(actual[key].detach().numpy(), expected[key], rtol=2e-6, atol=2e-7)
    bad = torch.from_numpy(quality).clone(); bad[0, 0] = float("nan")
    poisoned = mod.reward_torch(torch.from_numpy(force), 100., .8, .9, bad, torch.from_numpy(speed), validate=False)
    for value in poisoned.values(): assert torch.isnan(value[0, 0]).all()
    assert torch.isfinite(poisoned["raw_reward"][1]).all()


def test_torch_quality_gradient_only_affects_positive_bonus():
    torch = pytest.importorskip("torch")
    force = torch.full((2, 8, 2), 50., dtype=torch.float64)
    quality = torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True)
    result = mod.reward_torch(force, 100., 1., 1., quality, torch.tensor([1., 0.], dtype=torch.float64))
    result["raw_reward"].sum().backward()
    torch.testing.assert_close(quality.grad, torch.tensor([1., 0.], dtype=torch.float64))
