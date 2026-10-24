"""I arithmetic only. Native sensor wiring and task performance are separate."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_reward_v2i as mod


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("gaps", range(9))
def test_only_gap_aggregation_changes_for_all_gap_counts(dtype, gaps):
    force = np.tile(np.array([100., 0.], dtype=dtype), (8, 1)); force[:gaps] = 0.
    before = force.copy()
    g = mod.previous.reward_numpy(force, 100., .8, .9, .7, 1.)
    i = mod.reward_numpy(force, 100., .8, .9, .7, 1.)
    for key in g.keys()-{"raw_reward", "support_gap", "support_gap_penalty"}:
        np.testing.assert_array_equal(g[key], i[key])
    assert i["support_gap"] == float(gaps > 0)
    assert i["mean_substep_support_gap"] == gaps/8
    np.testing.assert_allclose(i["raw_reward"]-g["raw_reward"], -(float(gaps>0)-gaps/8), atol=1e-7)
    np.testing.assert_array_equal(force, before)
    assert i["raw_reward"].dtype == dtype


@pytest.mark.parametrize("position", range(8))
def test_one_gap_is_not_hidden_by_window_position_or_beta_gate(position):
    force = np.tile([100., 0.], (8, 1)); force[position] = 0.
    for anchor, speed, quality in ((0., 0., 0.), (1., 1., 0.)):
        assert mod.reward_numpy(force, 100., anchor, speed, quality, 1.)["raw_reward"] == -1.


def test_single_or_alternating_support_and_low_speed_beta_neutrality():
    single = np.tile([100., 0.], (8, 1)); alternate = single.copy(); alternate[1::2] = [0., 100.]
    for force in (single, alternate):
        assert mod.reward_numpy(force, 100., 1., 1., 1., 1.)["raw_reward"] == 1.
        assert mod.reward_numpy(force, 100., 1., 1., 0., .3)["raw_reward"] == 1.
    # Strict >5 N per wheel, not summed >5 N; no bilateral requirement.
    for pair, expected in (([5., 5.], 1.), ([5.0001, 0.], 0.), ([0., 5.0001], 0.)):
        assert mod.reward_numpy(np.tile(pair, (8, 1)), 100., 1., 1., 1., 1.)["support_gap"] == expected


def test_settings_ancestry_metadata_and_no_hotpath_file_io():
    for key, value in (("support_gap_aggregation", "mean8"), ("support_gap_weight", .5),
                       ("support_force_threshold_n", 0.), ("external_reward_weight", .8)):
        settings = copy.deepcopy(mod.DEFAULT_SETTINGS); settings[key] = value
        with pytest.raises(ValueError): mod.validate_settings(settings)
    with patch.object(Path, "read_bytes", side_effect=AssertionError("No reward file I/O")):
        assert mod.reward_numpy(np.zeros((8, 2)), 100., 1., 1., 1., 1.)["raw_reward"] == -1.
    with patch.object(mod, "BASE_G_DEFINITION_SHA256", "0"*64):
        with pytest.raises(ValueError, match="V2g arithmetic"): mod.verify_base()
    meta = mod.formula_metadata()
    assert "support_gap=max_substeps(" in meta["formula"]
    assert meta["support_gap_is_window_event_not_duration_fraction"]
    assert meta["no_velocity_proxy_or_absolute_height_input"]
    assert not meta["continuous_contact_or_task_success_certified"]


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_real_torch_parity_and_all_invalid_components_poison(dtype):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    force = rng.uniform(0., 100., (2, 3, 8, 2)).astype(dtype); force[:, :, 2] = 0.
    quality = rng.random((2, 3)).astype(dtype)
    reference = np.array([[0., .3, 1.], [1., .2, .9]], dtype=dtype)
    expected = mod.reward_numpy(force, 100., .8, .9, quality, reference)
    actual = mod.reward_torch(torch.from_numpy(force), 100., .8, .9,
        torch.from_numpy(quality), torch.from_numpy(reference))
    for key in expected:
        np.testing.assert_allclose(actual[key].numpy(), expected[key], rtol=2e-6, atol=2e-7)
    for bad_value in (float("nan"), float("inf"), -1.):
        bad = torch.from_numpy(force.copy()); bad[0, 0, 1, 0] = bad_value
        with pytest.raises(ValueError): mod.reward_torch(bad, 100., .8, .9, .7, 1.)
        poisoned = mod.reward_torch(bad, 100., .8, .9, .7, 1., validate=False)
        for value in poisoned.values(): assert torch.isnan(value[0, 0]).all()
        assert torch.isfinite(poisoned["raw_reward"][1]).all()
