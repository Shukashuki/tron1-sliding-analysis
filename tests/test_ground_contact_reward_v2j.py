"""J arithmetic only: no native data, training, runtime or task success claimed."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_reward_v2j as m


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_exact_single_component_difference_and_no_input_mutation(dtype):
    rng = np.random.default_rng(42)
    force = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype); force[:, :, 3] = 0.
    angular = rng.normal(size=(2, 3, 9, 2)).astype(dtype)
    original = force.copy(), angular.copy()
    i = m.previous.reward_numpy(force, 100., .8, .9, .7, 1.)
    j = m.reward_numpy(force, 100., .8, .9, .7, 1., angular)
    for key in i.keys()-{"raw_reward"}: np.testing.assert_array_equal(i[key], j[key])
    np.testing.assert_array_equal(j["raw_reward"], i["raw_reward"]-j["loaded_angular_increment_penalty"])
    np.testing.assert_array_equal(force, original[0]); np.testing.assert_array_equal(angular, original[1])
    assert j["raw_reward"].dtype == dtype


def test_constant_rate_and_small_smooth_adjustments_not_absolute_rate_target():
    force = np.tile([100., 0.], (8, 1))
    for angular in (np.full((9, 2), 100.), np.arange(9)[:, None]*np.array([[.02, -.02]])):
        assert m.reward_numpy(force, 100., 1., 1., 1., 1., angular)["raw_reward"] == 1.


def test_start_boundary_and_all8_actual_intervals_matter():
    force = np.tile([100., 0.], (8, 1)); angular = np.zeros((9, 2)); angular[0] = 100.
    result = m.reward_numpy(force, 100., 1., 1., 1., 1., angular)
    assert result["loaded_angular_increment_penalty"] == .5/8
    assert result["loaded_angular_increment_sample_cost"][0] == 1.
    for position in range(1, 9):
        angular = np.zeros((9, 2)); angular[position:] = 100.
        assert m.reward_numpy(force, 100., 1., 1., 1., 1., angular)["loaded_angular_increment_penalty"] == .5/8


def test_one_wheel_support_allowed_unloaded_angular_cost_zero_gap_unchanged():
    angular = np.arange(9)[:, None]*np.array([[100., 100.]])
    for force in (np.tile([100., 0.], (8, 1)), np.tile([0., 100.], (8, 1))):
        assert m.reward_numpy(force, 100., 1., 1., 1., 1., angular)["raw_reward"] == .5
    result = m.reward_numpy(np.zeros((8, 2)), 100., 1., 1., 1., 1., angular)
    assert result["loaded_angular_increment_penalty"] == 0. and result["raw_reward"] == -1.


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_unloaded_invalid_velocity_never_hidden(bad):
    angular = np.zeros((9, 2)); angular[0, 0] = bad
    with pytest.raises(ValueError): m.reward_numpy(np.zeros((8, 2)), 100., 1., 1., 1., 1., angular)


def test_settings_source_and_no_hotpath_io():
    for key in ("loaded_angular_increment_weight", "loaded_angular_increment_scale_rad_s"):
        cfg = copy.deepcopy(m.DEFAULT_SETTINGS); cfg[key] = 0.
        with pytest.raises(ValueError): m.validate_settings(cfg)
    with patch.object(m, "BASE_I_DEFINITION_SHA256", "0"*64):
        with pytest.raises(ValueError, match="Frozen I"): m.verify_base()
    with patch.object(Path, "read_bytes", side_effect=AssertionError("No hotpath IO")):
        m.reward_numpy(np.zeros((8, 2)), 100., 1., 1., 1., 1., np.zeros((9, 2)))


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_real_torch_parity_and_device_poison(dtype):
    torch = pytest.importorskip("torch"); rng = np.random.default_rng(7)
    force = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype)
    angular = rng.normal(size=(2, 3, 9, 2)).astype(dtype)
    expected = m.reward_numpy(force, 100., .8, .9, .7, 1., angular)
    actual = m.reward_torch(torch.from_numpy(force), 100., .8, .9, .7, 1., torch.from_numpy(angular))
    for key in expected: np.testing.assert_allclose(actual[key].numpy(), expected[key], rtol=2e-6, atol=2e-7)
    bad = torch.from_numpy(angular.copy()); bad[0, 0, 0, 0] = float("nan")
    poisoned = m.reward_torch(torch.from_numpy(force), 100., .8, .9, .7, 1., bad, validate=False)
    for value in poisoned.values(): assert torch.isnan(value[0, 0]).all()
    assert torch.isfinite(poisoned["raw_reward"][1]).all()
