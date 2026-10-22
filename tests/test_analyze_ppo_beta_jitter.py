"""Offline archived-pair tests only; no training, inference or simulation."""
from pathlib import Path
import copy
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_beta_jitter as audit


@pytest.fixture
def reports():
    return [json.loads((folder/"report.json").read_text()) for folder in (audit.DEFAULT_ON, audit.DEFAULT_OFF)]


@pytest.fixture(scope="module")
def comparison():
    return audit.analyze_pair()


def test_exact_archived_pair_coverage_and_hashes(comparison):
    arrays, summary = comparison
    assert summary["evaluation_mu"] == .7
    assert summary["beta_reward_weights"] == {"beta_on": 2., "beta_off": 0.}
    for label in ("beta_on", "beta_off"):
        case = summary["cases"][label]
        assert case["physical_samples"] == 1397 and case["control_decisions"] == 349
        assert case["windows"]["core"]["physical_samples"] == 513
        assert case["windows"]["core"]["control_decisions"] == 129
        assert case["checkpoint_sha256"] == audit.IDENTITIES[label]["checkpoint"]
        assert case["trajectory_sha256"] == audit.IDENTITIES[label]["trajectory"]
        assert np.array_equal(arrays[label+"_time_s"], np.arange(1397)*.005)
        assert arrays[label+"_decision_actions_clipped"].shape == (349, 8)
        assert arrays[label+"_native_actions_clipped"].shape == (1397, 8)
        assert arrays[label+"_native_valid_mask"].all()
    assert not summary["new_training"] and not summary["new_simulation"] and not summary["new_policy_inference"]
    assert not summary["causal_beta_effect_claimed"] and not summary["jitter_resolved_claimed"]


def test_nominal_observation_contains_jitter_without_beta_reward(comparison):
    _, summary = comparison
    changes = {(row["window"], row["signal"]): row for row in summary["relative_changes"]}
    row = changes[("core", "body_world_xy_angular_hf_rms")]
    assert row["beta_on"] == pytest.approx(2.172409620288345)
    assert row["beta_off"] == pytest.approx(1.86958031328147)
    assert row["off_vs_on_change_percent"] == pytest.approx(-13.939788526929863)
    assert changes[("full", "body_world_xy_angular_hf_rms")]["off_vs_on_change_percent"] > 0
    confound = summary["training_run_confounds_from_separate_manifest_audit"]
    assert confound["setup_smoke_control_steps"] == {"beta_on": 0, "beta_off": 8}
    assert confound["reported_training_cuda_devices"] == {"beta_on": "cuda:2", "beta_off": "cuda:7"}
    assert confound["independently_reverified_by_this_script"] is False


def test_profile_comparison_ignores_prose_only(reports):
    on, off = reports
    off["task_profile"]["description"] = "Different prose does not change reward math"
    result = audit.audit_reports(on, off)
    assert result["wheel_xy_profile_changed"] is False
    assert result["training_recipe_parity_independently_verified_by_this_script"] is False


@pytest.mark.parametrize("field,value", [("seed", 99), ("physics_dt_s", .01),
    ("observation_noise_enabled", True), ("domain_randomization_enabled", True),
    ("asset_root_file_sha256", "0"*64), ("motion_file_sha256", "0"*64),
    ("checkpoint_sha256", "0"*64), ("reference_frames", 100)])
def test_changed_nominal_report_contract_rejected(reports, field, value):
    on, off = reports; off[field] = value
    with pytest.raises(ValueError):
        audit.audit_reports(on, off)


@pytest.mark.parametrize("field", ["contact_material_audit", "evaluator_source_sha256", "drift_module_sha256", "runner_config_sha256"])
def test_changed_material_or_source_rejected(reports, field):
    on, off = reports
    off[field] = "changed"
    with pytest.raises(ValueError, match="Shared physical/source"):
        audit.audit_reports(on, off)


def test_other_reward_or_wheel_xy_profile_changes_rejected(reports):
    on, off = reports
    off["task_profile"]["reward_weights"]["yaw_rate"] = 0.
    with pytest.raises(ValueError, match="Profiles differ"):
        audit.audit_reports(on, off)


def test_reward_removed_not_zero_beta_reference(reports):
    on, off = reports
    off["task_profile"]["reward_weights"]["sideslip"] = -.1
    with pytest.raises(ValueError, match="weights 2 -> 0"):
        audit.audit_reports(on, off)


def test_changed_actuator_or_action_scale_rejected(reports):
    on, off = reports
    off["policy_export"]["leg_position_scale"] = .5
    with pytest.raises(ValueError, match="export contracts differ"):
        audit.audit_reports(on, off)


def test_exact_raw_hash_guard_rejects_corruption(tmp_path):
    folder = tmp_path/"bad-on"; folder.mkdir()
    (folder/"report.json").write_bytes((audit.DEFAULT_ON/"report.json").read_bytes())
    (folder/"physics_trajectory.npz").write_bytes(b"corrupted archive")
    with pytest.raises(ValueError, match="Original artifact SHA"):
        audit.analyze_pair(folder, audit.DEFAULT_OFF)


def test_fresh_output_rule_runs_before_reading_inputs(tmp_path):
    args = audit.parser().parse_args(["--on-dir", "missing", "--off-dir", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="Preserve existing"):
        audit.run(args)


def test_zero_baseline_ratio_is_not_invented():
    row = audit._relative(0., 1.)
    assert row["off_over_on_ratio"] is None and row["off_vs_on_change_percent"] is None


def test_packaged_arrays_and_artifact_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "render_diagnostics", lambda arrays, summary, path: Path(path).write_bytes(b"unit-test image stub"))
    output = tmp_path/"fresh"
    result = audit.run(audit.parser().parse_args(["--output-dir", str(output)]))
    assert set(result["artifacts"]) == {"native_comparison.npz", "diagnostics.png"}
    for name, artifact in result["artifacts"].items():
        assert artifact["sha256"] == audit.sha256(output/name)
    with np.load(output/"native_comparison.npz", allow_pickle=False) as archive:
        assert archive["beta_off_native_actions_clipped"].shape == (1397, 8)
    assert json.loads((output/"summary.json").read_text())["causal_beta_effect_claimed"] is False
