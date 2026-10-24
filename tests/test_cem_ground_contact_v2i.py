"""I CEM: only G gap aggregation mean8->max8; no policy success inferred."""
import copy
import contextlib
import io
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"tests"))
import nmpc_batch_ground_contact_v2g_400hz as parent
import nmpc_batch_ground_contact_v2i_400hz as candidate
import run_cem_ground_contact_v2g as parent_runner
import run_cem_ground_contact_v2i as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture, evaluate
from test_run_cem_ground_contact_v2e import objective as old_fixture

def objective():
    result = old_fixture()
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        result[channel]["drift_sideslip"] = result[channel].pop("unchanged0")
    return result

@pytest.mark.parametrize("gamma", (.8, .99, 1.))
def test_only_grounded_changes_from_f_with_exact_stage_and_horizon_delta(gamma):
    native, reference, profile, actions = fixture()
    frozen = copy.deepcopy(native); before_ref = copy.deepcopy(reference)
    f = evaluate(parent.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
    g = evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        assert len(f[channel]) == len(g[channel]) == 15
        for term in f[channel]:
            if term != "drift_grounded": np.testing.assert_array_equal(f[channel][term], g[channel][term])
    for key in f["ground_contact_v2"].keys()-{"raw_reward", "support_gap", "support_gap_penalty"}:
        np.testing.assert_array_equal(f["ground_contact_v2"][key], g["ground_contact_v2"][key])
    speed = np.linalg.norm(reference["body_lin_vel_w"][..., 0, :2], axis=-1)
    gate = np.where(speed>.3, f["raw_terms"]["drift_sideslip"], 1.)
    np.testing.assert_array_equal(g["ground_contact_v2"]["beta_contact_gate"], gate)
    expected = .02*.4*(g["ground_contact_v2"]["support_gap_penalty"]-f["ground_contact_v2"]["support_gap_penalty"])
    np.testing.assert_allclose(g["stage_cost"]-f["stage_cost"], expected, rtol=1e-12, atol=4e-16)
    np.testing.assert_allclose(g["candidate_costs"]-f["candidate_costs"],
                              (expected*gamma**np.arange(5)).sum(-1), rtol=1e-12, atol=6e-16)
    for key in native: np.testing.assert_array_equal(native[key], frozen[key])
    for key in reference: np.testing.assert_array_equal(reference[key], before_ref[key])

@pytest.mark.parametrize("broadcast", ("batch", "horizon", "body"))
def test_beta_gate_neutral_at_stopped_reference_in_every_supported_reference_shape(broadcast):
    native, reference, profile, actions = fixture()
    if broadcast == "horizon": reference = {key: (value if key=="body_names" else value[0]) for key,value in reference.items()}
    if broadcast == "body": reference = {key: (value if key=="body_names" else value[0, 0]) for key,value in reference.items()}
    reference["body_lin_vel_w"][..., 0, :2] = 0.
    reference["body_lin_vel_w"][..., 0, 2] = 100.
    f = evaluate(parent.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.), native, reference, actions)
    g = evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.), native, reference, actions)
    np.testing.assert_array_equal(g["ground_contact_v2"]["beta_contact_gate"], np.ones((3,5)))
    np.testing.assert_allclose(g["raw_terms"]["drift_grounded"]-f["raw_terms"]["drift_grounded"],
        -(g["ground_contact_v2"]["support_gap_penalty"]-f["ground_contact_v2"]["support_gap_penalty"]))

@pytest.mark.parametrize("missing", (0, 1, 3, 8))
@pytest.mark.parametrize("beta,refspeed", ((0., 1.), (.37, 1.), (1., 1.), (0., .3), (0., 0.)))
def test_actual_stage_uses_same_unweighted_quality_and_ungated_gap(missing, beta, refspeed):
    legacy = objective(); legacy["raw_terms"]["drift_sideslip"] = beta
    frozen = copy.deepcopy(legacy)
    force = np.zeros((8,2,3)); force[:,0,2] = 100.; force[:missing] = 0.
    f = parent_runner.replace_actual_grounded(legacy, force, 100., reference_speed=refspeed)
    g = runner.replace_actual_grounded(legacy, force, 100., reference_speed=refspeed)
    assert legacy == frozen
    gate = beta if refspeed>.3 else 1.
    cf, cg = f["ground_contact_v2"], g["ground_contact_v2"]
    assert cg["beta_contact_gate"] == gate
    assert cg["support_gap_penalty"] == float(missing>0)
    assert cf["support_gap_penalty"] == missing/8.
    assert g["raw_terms"]["drift_grounded"] == cf["A_gated_support_bonus"]*gate-float(missing>0)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        for name in f[channel]:
            if name!="drift_grounded": assert f[channel][name] == g[channel][name]
    assert g["stage_cost"]-f["stage_cost"] == pytest.approx(.02*.4*(float(missing>0)-missing/8.), abs=3e-16)
    assert g["ground_contact_beta_gate_provenance"]["held_reference_xy_speed_m_s"] == refspeed

@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -.1, True, [1.]))
def test_actual_stage_requires_valid_explicit_held_reference_speed(bad):
    force = np.zeros((8,2,3)); force[:,0,2] = 100.
    with pytest.raises(ValueError): runner.replace_actual_grounded(objective(), force, 100., reference_speed=bad)

def test_missing_beta_quality_or_explicit_reference_speed_is_never_imputed():
    legacy = objective(); legacy["raw_terms"].pop("drift_sideslip")
    force = np.zeros((8,2,3)); force[:,0,2] = 100.
    with pytest.raises((ValueError, KeyError)): runner.replace_actual_grounded(legacy, force, 100., reference_speed=1.)
    with pytest.raises(TypeError): runner.replace_actual_grounded(objective(), force, 100.)

def test_cli_cannot_change_friction_optimizer_weights_or_acceptance():
    parser = runner.parser(); fixed = ["--native-library", "frozen.so", "--output-dir", "fresh"]
    for extra in (["--mu",".15"], ["--grounded-weight","1"], ["--horizon-steps","10"], ["--beta-std",".4"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit): parser.parse_args(fixed+extra)

def test_g_source_identity_and_actual_reward_use_same_pre_action_held_reference():
    assert runner.sha256(ROOT/"config/ppo_ground_contact_v2i_400hz_v1.json") == runner.PROTOCOL_SHA256
    source = Path(runner.__file__).read_text()
    for forbidden in ("torch.load(", "torch.jit.load(", "build_native_library("): assert forbidden not in source
    assert source.count("reference_speed=float(np.linalg.norm(held.body_lin_vel_w[0, :2]))") == 2
    assert '"ground_contact_reward_v2f.py"' in source
    assert '"PPO_actions_or_future_states_used_for_initial_plan": False' in source
    assert '"termination_or_reset_after_initialization": False' in source
