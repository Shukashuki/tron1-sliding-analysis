"""F arithmetic tests. Synthetic force fixtures are not learned-policy results."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_reward_v2f as mod


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_only_difference_from_A_is_exact_gap_occupancy(dtype):
    rng = np.random.default_rng(42)
    forces = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype)
    forces[0, :, 2] = 0.
    forces[1, 0, 7] = [5., 5.]
    original = forces.copy()
    anchor, speed = rng.random((2, 3)).astype(dtype), rng.random((2, 3)).astype(dtype)
    a = mod.base.reward_numpy(forces, 200., anchor, speed)
    f = mod.reward_numpy(forces, 200., anchor, speed)
    for name in a.keys()-{"raw_reward"}:
        np.testing.assert_array_equal(f[name], a[name])
    expected = (np.max(forces, axis=-1) <= 5.).astype(dtype).mean(axis=-1)
    np.testing.assert_array_equal(f["support_gap"], expected)
    np.testing.assert_array_equal(f["raw_reward"], a["raw_reward"]-expected)
    np.testing.assert_array_equal(f["A_gated_support_bonus"], a["raw_reward"])
    np.testing.assert_array_equal(forces, original)
    assert f["raw_reward"].dtype == dtype


def test_one_wheel_alternation_and_supported_light_load_have_no_negative_term():
    for forces in (np.tile([100., 0.], (8, 1)),
                   np.tile([[100., 0.], [0., 100.]], (4, 1)),
                   np.full((8, 2), 50.), np.tile([5.000001, 0.], (8, 1))):
        result = mod.reward_numpy(forces, 218.5, 1., 1.)
        assert result["support_gap"] == 0.
        assert result["raw_reward"] == mod.base.reward_numpy(forces, 218.5, 1., 1.)["raw_reward"]
    # Summing two individually insufficient wheels does NOT count as one supported wheel.
    assert mod.reward_numpy(np.full((8, 2), 4.), 218.5, 1., 1.)["support_gap"] == 1.


@pytest.mark.parametrize("gate", (0., 1.))
def test_zero_load_cost_is_ungated_and_uses_all_substeps(gate):
    assert mod.reward_numpy(np.zeros((8, 2)), 100., gate, gate)["raw_reward"] == -1.
    forces = np.tile([100., 0.], (8, 1))
    forces[3] = 0.
    result = mod.reward_numpy(forces, 100., gate, gate)
    assert result["support_gap"] == .125
    assert result["raw_reward"] == mod.base.reward_numpy(forces, 100., gate, gate)["raw_reward"]-.125


def test_gap_cost_has_no_within_or_cross_window_packing_discount():
    packed = np.full((2, 8, 2), [100., 0.]); packed[0, 3:5] = 0.
    split = np.full((2, 8, 2), [100., 0.]); split[:, 3] = 0.
    boundary = np.full((2, 8, 2), [100., 0.]); boundary[0, 7] = 0.; boundary[1, 0] = 0.
    for force in (packed, split, boundary):
        assert mod.reward_numpy(force, 100., 1., 1.)["support_gap_penalty"].sum() == .25


@pytest.mark.parametrize("value", (np.nan, np.inf, -.001))
def test_invalid_forces_rejected(value):
    force = np.ones((8, 2)); force[0, 0] = value
    with pytest.raises(ValueError):
        mod.reward_numpy(force, 100., 1., 1.)


@pytest.mark.parametrize("shape", ((7, 2), (9, 2), (8, 3), (0, 8, 2)))
def test_wrong_shape_rejected(shape):
    with pytest.raises(ValueError):
        mod.reward_numpy(np.zeros(shape), 100., 1., 1.)


def test_mass_cannot_change_the_newtons_gap_threshold():
    force = np.tile([6., 0.], (8, 1))
    for mg in (10., 100., 1000.):
        assert mod.reward_numpy(force, mg, 1., 1.)["support_gap"] == 0.


def test_settings_pins_bounds_and_no_hotpath_IO():
    with patch.object(Path, "read_bytes", side_effect=AssertionError("No reward file I/O")):
        assert mod.reward_numpy(np.zeros((8, 2)), 100., 1., 1.)["raw_reward"] == -1.
    for name, bad in (("support_gap_weight", .5), ("mean_weight", 1.),
                      ("support_force_threshold_n", 0.), ("support_gap_weight", True)):
        settings = copy.deepcopy(mod.DEFAULT_SETTINGS); settings[name] = bad
        with pytest.raises(ValueError): mod.validate_settings(settings)
    with patch.object(mod, "BASE_DEFINITION_SHA256", "0"*64):
        with pytest.raises(ValueError): mod.verify_base()
    meta = mod.formula_metadata()
    assert meta["raw_reward_bounds"] == [-1., 1.]
    assert not meta["supported_light_load_penalized"]
    assert not meta["bilateral_support_required"]
    assert not meta["height_or_zero_sideslip_target"]


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_real_torch_numpy_agreement_and_invalid_poison(dtype):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    force = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype); force[:, :, 3] = 0.
    expected = mod.reward_numpy(force, 100., .8, .3)
    actual = mod.reward_torch(torch.from_numpy(force), 100., .8, .3)
    for key in expected:
        np.testing.assert_allclose(actual[key].detach().numpy(), expected[key], rtol=2e-6, atol=2e-7)
    tensor = torch.from_numpy(force).clone(); tensor[0, 0, 0, 0] = float("nan")
    invalid = mod.reward_torch(tensor, 100., .8, .3, validate=False)
    assert torch.isnan(invalid["raw_reward"][0, 0])
    assert torch.isnan(invalid["support_gap"][0, 0])
    assert torch.isfinite(invalid["raw_reward"][1]).all()
    with pytest.raises(ValueError): mod.reward_torch(tensor, 100., .8, .3)
