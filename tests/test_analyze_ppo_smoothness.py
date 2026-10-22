"""Deterministic offline diagnostic tests; no simulator, policy, or GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_ppo_smoothness as analyzer


NAMES = np.array(["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
                  "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"])


def make_recording(duration=8., action_amplitude=.4, angular_amplitude=2.):
    count = int(round(duration/.005))+1
    time = np.arange(count)*.005
    frames = np.maximum((np.arange(count)-1)//4, 0)
    action = action_amplitude*np.sin(2*np.pi*20*frames*.02)
    actions = np.repeat(action[:, None], 8, axis=1)
    actions[0] = 0.
    angular = np.zeros((count, 3))
    angular[:, 0] = angular_amplitude*np.sin(2*np.pi*30*time)
    position = np.column_stack((time, np.zeros(count), np.full(count, .75)))
    quaternion = np.tile([1., 0., 0., 0.], (count, 1))
    velocity = np.tile([1., 0., 0.], (count, 1))
    joint_velocity = np.repeat((2.*np.sin(2*np.pi*16*time))[:, None], 8, axis=1)
    joint_position = np.repeat((.02*np.sin(2*np.pi*16*time))[:, None], 8, axis=1)
    torque = np.repeat((30.*np.sin(2*np.pi*30*time))[:, None], 8, axis=1)
    return {"time_s": time, "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
            "joint_names": NAMES, "root_position_m": position[:, None],
            "root_quaternion_wxyz": quaternion[:, None],
            "root_linear_velocity_w_m_s": velocity[:, None],
            "root_angular_velocity_w_rad_s": angular[:, None],
            "joint_position_rad": joint_position[:, None], "joint_velocity_rad_s": joint_velocity[:, None],
            "applied_joint_torque_nm": torque[:, None], "actions_clipped": actions[:, None],
            "actions_raw": actions[:, None].copy(), "action_was_applied": (np.arange(count)>0)[:, None],
            "reference_frame_for_action": frames[:, None], "simulation_step_counter": np.arange(count),
            "valid_mask": np.ones((count, 1), dtype=bool)}


@pytest.fixture
def inputs(tmp_path):
    raw = make_recording()
    trajectory = tmp_path/"physics_trajectory.npz"
    np.savez_compressed(trajectory, **raw)
    time = np.arange(401)/50.
    reference = tmp_path/"reference.npz"
    np.savez_compressed(reference, reference_time_s=time,
        reference_position=np.column_stack((time, np.zeros(len(time)), np.full(len(time), .75))),
        reference_com_velocity=np.tile([1., 0., 0.], (len(time), 1)),
        reference_quaternion=np.tile([1., 0., 0., 0.], (len(time), 1)))
    report = tmp_path/"report.json"
    report.write_text(json.dumps({"checkpoint_sha256": "1"*64,
        "evaluation_contact_friction_override": {"evaluation_contact_friction": .7}}))
    return raw, trajectory, reference, report


def test_duplicate_applied_flags_are_not_counted_as_policy_decisions(inputs):
    raw, *_ = inputs
    result = analyzer.deduplicate_actions(raw["time_s"], raw["actions_clipped"][:, 0],
        raw["action_was_applied"][:, 0], .02, frames=raw["reference_frame_for_action"][:, 0])
    assert result["decisions"] == 400 and result["held_physical_records"] == 1600
    assert result["decimation"] == 4
    np.testing.assert_array_equal(result["record_indices"], np.arange(1, 1601, 4))
    np.testing.assert_allclose(result["time_s"], np.arange(400)*.02, atol=1e-12, rtol=0)
    actual = np.sqrt(np.mean(np.diff(result["actions"][:, :6], axis=0)**2))
    naive = np.sqrt(np.mean(np.diff(raw["actions_clipped"][:, 0, :6], axis=0)**2))
    assert actual > 1.9*naive


def test_counter_fallback_still_checks_held_blocks(inputs):
    raw, *_ = inputs
    result = analyzer.deduplicate_actions(raw["time_s"], raw["actions_clipped"][:, 0],
        raw["action_was_applied"][:, 0], .02, counters=raw["simulation_step_counter"])
    assert result["decisions"] == 400
    changed = raw["actions_clipped"][:, 0].copy()
    changed[7, 0] += .01
    with pytest.raises(ValueError, match="inside a held decision block"):
        analyzer.deduplicate_actions(raw["time_s"], changed, raw["action_was_applied"][:, 0],
                                    .02, counters=raw["simulation_step_counter"])


def test_missing_decision_identity_cannot_use_flags_alone(inputs):
    raw, *_ = inputs
    with pytest.raises(ValueError, match="Need decision frame IDs or physical-step counters"):
        analyzer.deduplicate_actions(raw["time_s"], raw["actions_clipped"][:, 0], raw["action_was_applied"][:, 0], .02)


def test_counter_and_decision_identity_cannot_disagree(inputs):
    raw, *_ = inputs
    counters = raw["simulation_step_counter"].copy()
    counters[10] = counters[9]
    with pytest.raises(ValueError, match="Physical-step counters"):
        analyzer.deduplicate_actions(raw["time_s"], raw["actions_clipped"][:, 0],
            raw["action_was_applied"][:, 0], .02,
            frames=raw["reference_frame_for_action"][:, 0], counters=counters)


def test_report_cannot_attach_a_different_trajectory(inputs):
    _, path, reference, report = inputs
    original = json.loads(report.read_text())
    report.write_text(json.dumps({**original, "physics_trajectory_sha256": "0"*64}))
    with pytest.raises(ValueError, match="Report-to-trajectory SHA binding mismatch"):
        analyzer.analyze_trajectory(path, reference=reference, report=report)
    report.write_text(json.dumps({**original, "physics_trajectory_sha256": analyzer.sha256(path)}))
    result = analyzer.analyze_trajectory(path, reference=reference, report=report)
    assert result["report_to_trajectory_sha_verified"] is True


def test_known_sine_absolute_power_and_peak_frequency():
    time = np.arange(2000)*.005
    high = 2.*np.sin(2*np.pi*30*time)
    low = 3.*np.sin(2*np.pi*3*time)
    metrics = analyzer.spectral_metrics(high+low, .005, 10.)
    assert metrics["dominant_frequency_hz"] == pytest.approx(3., abs=.1)
    assert metrics["high_frequency_mean_square"] == pytest.approx(2., rel=5e-4)
    assert metrics["high_frequency_rms"] == pytest.approx(np.sqrt(2.), rel=5e-4)
    assert metrics["total_mean_square"] == pytest.approx(6.5, rel=5e-4)
    assert metrics["high_frequency_power_fraction"] == pytest.approx(2./6.5, rel=1e-3)


def test_zero_signal_and_low_frequency_only_are_not_nan():
    zero = analyzer.spectral_metrics(np.zeros((1000, 2)), .005)
    assert zero["dominant_frequency_hz"] is None
    assert zero["high_frequency_rms"] == zero["high_frequency_power_fraction"] == 0.
    signal = np.sin(2*np.pi*2*np.arange(1000)*.005)
    low = analyzer.spectral_metrics(signal, .005)
    assert low["high_frequency_power_fraction"] < 1e-8


@pytest.mark.parametrize("change", ["nonfinite", "irregular", "backwards"])
def test_bad_physical_clock_is_rejected(change):
    clock = np.arange(50)*.005
    if change == "nonfinite":
        clock[4] = np.nan
    elif change == "irregular":
        clock[4] += .0001
    else:
        clock[4] = clock[3]
    with pytest.raises(ValueError, match="clock"):
        analyzer.validate_clock(clock)


@pytest.mark.parametrize("values,dt,cutoff", [([1, 2, np.nan, 3], .005, 10),
    ([0, 0, 0, 0], 0, 10), ([0, 0, 0, 0], .02, 25), ([0, 0], .005, 10)])
def test_invalid_spectral_input_is_rejected(values, dt, cutoff):
    with pytest.raises(ValueError):
        analyzer.spectral_metrics(values, dt, cutoff)


def test_native_metrics_reference_tracking_and_body_frame_are_reproducible(inputs):
    raw, trajectory, reference, report = inputs
    before = analyzer.sha256(trajectory)
    result = analyzer.analyze_trajectory(trajectory, reference=reference, report=report)
    assert result["physical_samples"] == 1601 and result["control_decisions"] == 400
    assert result["evaluation_mu"] == .7 and result["checkpoint_sha256"] == "1"*64
    core = result["windows"]["core"]
    assert core["physical_samples"] == 513 and core["control_decisions"] == 129
    assert core["coverage_complete"] is True
    assert core["tracking"]["xy_rmse_m"] < 1e-12
    assert core["tracking"]["speed_retention"] == pytest.approx(1.)
    assert core["tracking"]["beta_rmse_deg"] == pytest.approx(0.)
    assert core["action"]["leg_raw_outside_clip_fraction"] == 0.
    assert core["torque"]["leg_near_80nm_fraction"] == 0.
    assert "NOT inferred motor saturation" in core["torque"]["near_limit_definition"]
    assert core["body"]["angular_xy_spectrum"]["high_frequency_rms"] > .9
    assert core["body"]["angular_body_xy_spectrum"] == core["body"]["angular_xy_spectrum"]
    assert result["windows"]["full"]["tracking"]["available"] is True
    assert not result["new_simulation"] and not result["new_inference"]
    assert analyzer.sha256(trajectory) == before


def test_body_frame_rotation_uses_recorded_orientation_not_quaternion_derivatives():
    quaternion = np.tile([np.sqrt(.5), 0., 0., np.sqrt(.5)], (4, 1))
    angular = np.tile([1., 0., 0.], (4, 1))
    body = analyzer._world_to_body(quaternion, angular)
    np.testing.assert_allclose(body, np.tile([0., -1., 0.], (4, 1)), atol=1e-14, rtol=0)


def test_missing_fields_and_nonfinite_states_are_rejected(inputs, tmp_path):
    raw, *_ = inputs
    missing = dict(raw)
    missing.pop("root_angular_velocity_w_rad_s")
    path = tmp_path/"missing.npz"
    np.savez_compressed(path, **missing)
    with pytest.raises(ValueError, match="Missing required field: root_angular_velocity"):
        analyzer.analyze_trajectory(path)
    changed = copy.deepcopy(raw)
    changed["joint_velocity_rad_s"][30, 0, 0] = np.nan
    bad = tmp_path/"nonfinite.npz"
    np.savez_compressed(bad, **changed)
    with pytest.raises(ValueError, match="nonfinite"):
        analyzer.analyze_trajectory(bad)


def test_invalid_tail_is_excluded_but_reset_splicing_is_rejected(inputs, tmp_path):
    raw, *_ = inputs
    changed = copy.deepcopy(raw)
    changed["valid_mask"][1001:] = False
    changed["root_position_m"][1001:] = np.nan
    path = tmp_path/"tail.npz"
    np.savez_compressed(path, **changed)
    result = analyzer.analyze_trajectory(path)
    assert result["physical_samples"] == 1001 and result["excluded_invalid_tail_samples"] == 600
    changed["valid_mask"][1100] = True
    np.savez_compressed(path, **changed)
    with pytest.raises(ValueError, match="splices episodes"):
        analyzer.analyze_trajectory(path)


def test_no_reference_means_no_invented_tracking_or_success(inputs):
    _, path, _, report = inputs
    result = analyzer.analyze_trajectory(path, report=report)
    assert result["windows"]["core"]["tracking"]["available"] is False
    screen = analyzer.compare_candidate(result, result, beta_extra_rmse_deg=5.)
    assert screen["status"] == "insufficient_evidence" and not screen["passes"]


def test_candidate_screen_requires_absolute_reductions_tracking_speed_and_beta(inputs):
    _, path, reference, report = inputs
    baseline = analyzer.analyze_trajectory(path, reference=reference, report=report)
    candidate = copy.deepcopy(baseline)
    core = candidate["windows"]["core"]
    core["body"]["angular_xy_spectrum"]["high_frequency_rms"] *= .4
    core["action"]["leg_delta_rms"] *= .4
    result = analyzer.compare_candidate(candidate, baseline, beta_extra_rmse_deg=5.)
    assert result["status"] == "passes_unvalidated_screen" and result["passes"]
    assert not result["statistical_success_claimed"] and not result["safety_qualification_claimed"]
    assert result["criteria"]["xy_rmse_m"]["maximum"] == pytest.approx(.03)
    core["tracking"]["speed_retention"] = .89
    assert not analyzer.compare_candidate(candidate, baseline, beta_extra_rmse_deg=5.)["passes"]
    core["tracking"]["speed_retention"] = 1.
    core["tracking"]["beta_rmse_deg"] = 5.01
    assert not analyzer.compare_candidate(candidate, baseline, beta_extra_rmse_deg=5.)["passes"]
    core["tracking"]["beta_rmse_deg"] = 0.
    core["tracking"]["xy_rmse_m"] = .031
    assert not analyzer.compare_candidate(candidate, baseline, beta_extra_rmse_deg=5.)["passes"]


def test_fractional_power_reduction_alone_does_not_pass(inputs):
    _, path, reference, report = inputs
    baseline = analyzer.analyze_trajectory(path, reference=reference, report=report)
    candidate = copy.deepcopy(baseline)
    candidate["windows"]["core"]["body"]["angular_xy_spectrum"]["high_frequency_power_fraction"] = 0.
    candidate["windows"]["core"]["action"]["leg_delta_rms"] *= .1
    result = analyzer.compare_candidate(candidate, baseline)
    assert result["status"] == "fails_unvalidated_screen"
    assert not result["criteria"]["absolute_hf_body_angular_xy_rms_ratio"]["passes"]


def test_zero_baseline_does_not_fabricate_a_reduction(inputs):
    _, path, reference, report = inputs
    baseline = analyzer.analyze_trajectory(path, reference=reference, report=report)
    baseline["windows"]["core"]["body"]["angular_xy_spectrum"]["high_frequency_rms"] = 0.
    result = analyzer.compare_candidate(baseline, baseline)
    assert not result["passes"] and result["status"] == "insufficient_evidence"


def test_requested_beta_guard_requires_reference_beta_coverage(inputs):
    _, path, reference, report = inputs
    baseline = analyzer.analyze_trajectory(path, reference=reference, report=report)
    candidate = copy.deepcopy(baseline)
    core = candidate["windows"]["core"]
    core["body"]["angular_xy_spectrum"]["high_frequency_rms"] *= .1
    core["action"]["leg_delta_rms"] *= .1
    core["tracking"]["beta_rmse_deg"] = None
    result = analyzer.compare_candidate(candidate, baseline, beta_extra_rmse_deg=5.)
    assert not result["passes"] and result["status"] == "insufficient_evidence"


@pytest.mark.parametrize("key,value,match", [("evaluation_mu", .15, "friction"),
    ("reference_sha256", "0"*64, "reference identities"), ("high_frequency_hz", 5., "cutoffs")])
def test_mismatched_comparisons_are_rejected(inputs, key, value, match):
    _, path, reference, report = inputs
    baseline = analyzer.analyze_trajectory(path, reference=reference, report=report)
    candidate = {**baseline, key: value}
    with pytest.raises(ValueError, match=match):
        analyzer.compare_candidate(candidate, baseline)


def test_cli_paired_reports_hashes_and_no_overwrite(inputs, tmp_path):
    _, path, reference, report = inputs
    output = tmp_path/"metrics.json"
    args = analyzer.parser().parse_args(["--trajectory", str(path), "--report", str(report),
        "--reference", str(reference), "--output", str(output)])
    result = analyzer.run(args)
    assert result["runs"][0]["reference_sha256"] == analyzer.sha256(reference)
    assert result["analysis_source_sha256"] == analyzer.sha256(analyzer.__file__)
    assert json.loads(output.read_text()) == result
    old = output.read_bytes()
    with pytest.raises(ValueError, match="Preserve existing analysis"):
        analyzer.run(args)
    assert output.read_bytes() == old


def test_cli_paired_baselines_can_match_candidate_by_reported_mu(inputs, tmp_path):
    _, path, reference, report = inputs
    second_report = tmp_path/"low_mu_report.json"
    second_report.write_text(json.dumps({"checkpoint_sha256": "1"*64,
        "evaluation_contact_friction_override": {"evaluation_contact_friction": .15}}))
    baseline_file = tmp_path/"baseline_pair.json"
    baseline_args = analyzer.parser().parse_args(["--trajectory", str(path), "--trajectory", str(path),
        "--report", str(report), "--report", str(second_report), "--reference", str(reference),
        "--output", str(baseline_file)])
    baseline = analyzer.run(baseline_args)
    assert [item["evaluation_mu"] for item in baseline["runs"]] == [.7, .15]
    args = analyzer.parser().parse_args(["--trajectory", str(path), "--report", str(second_report),
        "--reference", str(reference), "--baseline", str(baseline_file), "--beta-extra-rmse-deg", "5",
        "--output", str(tmp_path/"candidate.json")])
    result = analyzer.run(args)
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["status"] == "fails_unvalidated_screen"


def test_incomplete_reference_is_censored_not_held(inputs, tmp_path):
    _, path, _, report = inputs
    time = np.arange(101)/50.
    reference = tmp_path/"short_reference.npz"
    np.savez_compressed(reference, reference_time_s=time,
        reference_position=np.column_stack((time, np.zeros(len(time)), np.full(len(time), .75))),
        reference_com_velocity=np.tile([1., 0., 0.], (len(time), 1)),
        reference_quaternion=np.tile([1., 0., 0., 0.], (len(time), 1)))
    result = analyzer.analyze_trajectory(path, reference=reference, report=report)
    assert result["windows"]["core"]["tracking"]["available"] is False
    assert "no extrapolation" in result["windows"]["core"]["tracking"]["reason"].lower()


def test_optional_historical_baseline_replays_prior_core_action_metrics():
    root = ROOT/"outputs/policy_primary_grid_raw_20261016_v1/PPO"
    if not (root/"mu0p150/isaac/physics_trajectory.npz").is_file():
        pytest.skip("Optional private original PhysX recording is unavailable")
    expected = {"mu0p150": (.4046927883103126, 62.423866364253456, 2.9252813400373148),
                "mu0p700": (.2970092484419763, 53.398661329305575, 2.079266478617777)}
    for name, (action, torque, angular) in expected.items():
        result = analyzer.analyze_trajectory(root/name/"isaac/physics_trajectory.npz", report=root/name/"isaac/report.json")
        core = result["windows"]["core"]
        assert result["control_decisions"] == 349
        assert core["action"]["leg_delta_rms"] == pytest.approx(action, abs=1e-14)
        assert core["torque"]["leg_rms_nm"] == pytest.approx(torque, abs=1e-12)
        assert core["body"]["angular_xy_rms_rad_s"] == pytest.approx(angular, abs=1e-14)
