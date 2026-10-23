"""F planner and actual-stage reward accounting; no policy result is inferred."""
import copy
import contextlib
import io
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"tests"))
import nmpc_batch_ground_contact_400hz as original
import nmpc_batch_ground_contact_v2f_400hz as candidate
import run_cem_ground_contact_v2 as original_runner
import run_cem_ground_contact_v2f as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture, evaluate
from test_run_cem_ground_contact_v2e import objective


@pytest.mark.parametrize("gamma", (.8, .99, 1.))
def test_only_grounded_changes_from_A_with_exact_horizon_cost_delta(gamma):
    native, reference, profile, actions = fixture()
    frozen = copy.deepcopy(native)
    a = evaluate(original.BatchGroundContactTrackingObjective400Hz(
        profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
    f = evaluate(candidate.BatchGroundContactTrackingObjective400Hz(
        profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        assert len(a[channel]) == len(f[channel]) == 15
        for term in a[channel]:
            if term != "drift_grounded":
                np.testing.assert_array_equal(a[channel][term], f[channel][term])
    for key in a["ground_contact_v2"].keys()-{"raw_reward"}:
        np.testing.assert_array_equal(a["ground_contact_v2"][key], f["ground_contact_v2"][key])
    gap = (np.max(native["ground_normal_force_w"][..., 2], axis=-1) <= 5.).mean(-1)
    expected = .02*.4*gap
    np.testing.assert_array_equal(f["ground_contact_v2"]["support_gap"], gap)
    np.testing.assert_allclose(f["stage_cost"]-a["stage_cost"], expected, rtol=1e-12, atol=2e-16)
    np.testing.assert_allclose(f["candidate_costs"]-a["candidate_costs"],
        (expected*gamma**np.arange(5)).sum(-1), rtol=1e-12, atol=4e-16)
    for key in native: np.testing.assert_array_equal(native[key], frozen[key])


@pytest.mark.parametrize("missing", (0, 1, 3, 8))
def test_actual_stage_same_math_and_all_other14_unchanged(missing):
    legacy = objective(); frozen = copy.deepcopy(legacy)
    force = np.zeros((8, 2, 3)); force[:, 0, 2] = 100.; force[:missing] = 0.
    a = original_runner.replace_actual_grounded(legacy, force, 100.)
    f = runner.replace_actual_grounded(legacy, force, 100.)
    assert legacy == frozen
    assert f["ground_contact_v2"]["support_gap"] == missing/8.
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        for name in a[channel]:
            if name != "drift_grounded": assert a[channel][name] == f[channel][name]
    assert f["raw_terms"]["drift_grounded"] == a["raw_terms"]["drift_grounded"]-missing/8.
    assert f["stage_cost"]-a["stage_cost"] == pytest.approx(.02*.4*missing/8., abs=2e-16)


def test_no_negative_cost_for_supported_light_load_even_with_zero_tracking_gate():
    legacy = objective(); legacy["raw_terms"]["motion_global_anchor_pos"] = 0.
    for loads, expected in (([6., 0.], 0.), ([0., 6.], 0.), ([5., 5.], -1.), ([0., 0.], -1.)):
        force = np.zeros((8, 2, 3)); force[..., 2] = loads
        assert runner.replace_actual_grounded(legacy, force, 218.5)["raw_terms"]["drift_grounded"] == expected


def test_cli_cannot_change_friction_optimizer_weights_or_acceptance():
    parser = runner.parser(); fixed = ["--native-library", "frozen.so", "--output-dir", "fresh"]
    for extra in (["--mu", ".15"], ["--grounded-weight", "1"], ["--horizon-steps", "10"], ["--force-threshold", "0"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            parser.parse_args(fixed+extra)


def test_F_identity_and_source_protocol_binding():
    assert runner.sha256(ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json") == runner.PROTOCOL_SHA256
    assert runner.validate_settings()["support_gap_weight"] == 1.
    source = Path(runner.__file__).read_text()
    for forbidden in ("torch.load(", "torch.jit.load(", "build_native_library("):
        assert forbidden not in source
    assert '"PPO_actions_or_future_states_used_for_initial_plan": False' in source
    assert '"termination_or_reset_after_initialization": False' in source
