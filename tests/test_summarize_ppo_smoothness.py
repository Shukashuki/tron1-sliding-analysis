"""Synthetic copied metadata/recordings only: no real policy, training, or GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import summarize_ppo_smoothness as publication
from test_analyze_ppo_smoothness import make_recording


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


@pytest.fixture
def evidence(tmp_path):
    candidate, baseline = tmp_path/"candidate", tmp_path/"baseline"
    time = np.arange(349)/50.
    reference = tmp_path/"motion.npz"
    np.savez_compressed(reference, time_s=time, body_names=np.array(["base_Link"]),
        body_pos_w=np.column_stack((time, np.zeros(len(time)), np.full(len(time), .75)))[:, None],
        body_lin_vel_w=np.tile([1., 0., 0.], (len(time), 1, 1)),
        body_quat_w=np.tile([1., 0., 0., 0.], (len(time), 1, 1)))
    motion_hash = publication.sha256(reference)
    historical = {"reference_id": "drift_drl_traj4", "checkpoint_sha256": "1"*64,
        "motion_sha256": motion_hash, "training_asset_sha256": "3"*64,
        "trainer_sha256": "4"*64, "height_adapter_sha256": "5"*64,
        "task_profile_sha256": "6"*64,
        "source_training_modules_sha256": {"training/task.py": "7"*64},
        "training_domain_randomization": {"ground_friction": 1., "friction_static": [.3, 1.6], "friction_dynamic": [.3, 1.2]}}
    historical_path = tmp_path/"historical_training.json"
    write_json(historical_path, historical)
    manifest = {"algorithm": "PPO", "smoke_only": False, "num_envs": 512, "iterations": 1000,
        "seed": 42, "setup_smoke_control_steps": 0, "num_steps_per_env": 24,
        "environment_transitions_requested": 12288000, "motion_sha256": motion_hash,
        "asset_sha256": historical["training_asset_sha256"], "trainer_sha256": historical["trainer_sha256"],
        "height_adapter_sha256": historical["height_adapter_sha256"],
        "task_profile_sha256": historical["task_profile_sha256"],
        "source_training_modules_sha256": historical["source_training_modules_sha256"],
        "recipe": historical["training_domain_randomization"], "training_reward_overrides": {"action_rate_l2": -.3}}
    training = candidate/"training"
    write_json(training/"manifest.json", manifest)
    write_json(training/"run_status.json", {"status": "training_completed", "trained_policy_claimed": True,
        "environment_transitions": 12288000})
    write_json(training/"runner_config.json", {"num_steps_per_env": 24})
    ablation = {"algorithm": "PPO", "training_engine": "PhysX", "status": "training_completed",
        "trained_policy_claimed": True, "initialization": "from_scratch_no_checkpoint_resume", "smoke_only": False,
        "num_envs": 512, "iterations": 1000, "seed": 42, "expected_environment_transitions": 12288000,
        "actual_environment_transitions": 12288000, "checkpoint_sha256": "2"*64,
        "motion_sha256": motion_hash, "asset_sha256": historical["training_asset_sha256"],
        "historical_trainer_sha256": historical["trainer_sha256"], "height_adapter_sha256": historical["height_adapter_sha256"],
        "task_profile_sha256": historical["task_profile_sha256"],
        "baseline_source_sha256": historical["source_training_modules_sha256"],
        "reward_override": {"term": "action_rate_l2", "baseline_weight": -.1, "weight": -.3,
            "applied_before_environment_creation": True, "function": "isaaclab.envs.mdp.rewards.action_rate_l2",
            "divided_by_control_dt": False},
        "DR_recipe_modified": False, "base_task_profile_file_modified": False, "baseline_checkpoint_modified": False,
        "final_training_manifest_sha256": publication.sha256(training/"manifest.json")}
    write_json(training/"ablation_manifest.json", ablation)
    public = {"reference_id": "drift_drl_traj4", "input_sha256": {"motion": motion_hash}, "runs": {},
        "policies": {"PPO": {"source_xy_grid": {"input_sha256": {"training_metadata.json": publication.sha256(historical_path)}}}}}
    for mu in publication.COEFFICIENTS:
        name = publication.case_name(mu)
        for cohort, root, checkpoint in (("baseline", baseline, "1"*64), ("candidate", candidate/"validation", "2"*64)):
            folder = root/name/"isaac"
            folder.mkdir(parents=True)
            raw = make_recording(6.98, action_amplitude=.4 if cohort == "baseline" else .12,
                                 angular_amplitude=2. if cohort == "baseline" else .7)
            raw["wheel_ground_contact_count"] = np.ones((1397, 1, 2), dtype=int)
            force = np.zeros((1397, 1, 2, 3)); force[:, :, :, 2] = 100.
            raw["wheel_ground_contact_force_w_n"] = force
            np.savez_compressed(folder/"physics_trajectory.npz", **raw)
            report = {"simulator": "IsaacLab/PhysX", "status": "evaluated", "checkpoint_sha256": checkpoint,
                "motion_file_sha256": motion_hash, "physics_trajectory_sha256": publication.sha256(folder/"physics_trajectory.npz"),
                "evaluation_contact_friction_override": {"evaluation_contact_friction": mu},
                "domain_randomization_enabled": False, "observation_noise_enabled": False, "full_horizon_diagnostic": True,
                "summary": {"episodes": [{"completed_full_reference": True, "start_reference_frame": 0}]},
                "evaluator_source_sha256": "8"*64, "drift_module_sha256": "9"*64, "contact_slip_helper_sha256": "a"*64}
            write_json(folder/"report.json", report)
            if cohort == "baseline":
                public["runs"]["PPO_"+name] = {"input_sha256": {
                    "isaac_trace": publication.sha256(folder/"physics_trajectory.npz"),
                    "isaac_report": publication.sha256(folder/"report.json")}}
    public_path = tmp_path/"public.json"
    write_json(public_path, public)
    return {"candidate": candidate, "baseline": baseline, "reference": reference,
            "public": public_path, "historical": historical_path, "output": tmp_path/"fresh_publication"}


def load(evidence):
    return publication.load_publication(evidence["candidate"], evidence["reference"], evidence["baseline"],
        public_summary=evidence["public"], baseline_metadata=evidence["historical"])


def summarize(evidence):
    return publication.summarize(evidence["candidate"], evidence["reference"], evidence["baseline"], evidence["output"],
        public_summary=evidence["public"], baseline_metadata=evidence["historical"])


def test_publication_binds_completed_metadata_and_both_friction_recordings_without_weights(evidence):
    arrays, summary = load(evidence)
    assert len(summary["table"]) == 4 and summary["coefficients"] == [.7, .15]
    assert summary["training_engine"] == summary["evaluation_engine"] == "PhysX"
    assert summary["candidate_training"]["action_rate_weight"] == -.3
    assert summary["candidate_training"]["checkpoint_bytes_verified_locally"] is False
    assert summary["engineering_screen_passes_both_conditions"] is True
    assert summary["status"] == "screen_passed_requires_fullbody_visual_review"
    assert summary["requires_fullbody_visual_review"] is True and not summary["jitter_resolved_claimed"]
    assert not summary["new_training"] and not summary["new_simulation"] and not summary["new_policy_inference"]
    assert not summary["statistical_success_claimed"] and not summary["causal_improvement_claimed"]
    for name, case in summary["cases"].items():
        assert case["comparison"]["passes"] is True
        for cohort in ("baseline", "candidate"):
            assert case[cohort]["windows"]["core"]["physical_samples"] == 513
            assert case[cohort]["windows"]["core"]["control_decisions"] == 129
            assert case[cohort]["windows"]["full"]["tracking"]["available"] is False
            assert not case[cohort]["posture_diagnostics"]["full"]["height_or_flight_failure_gate"]
            assert arrays[name+"_"+cohort+"_decision_actions_clipped"].shape == (349, 8)
    measured = [row for row in summary["table"] if row["cohort"] == "candidate"]
    assert all(row["hf_rms_reduction_percent"] == pytest.approx(65., abs=1e-10) for row in measured)
    assert all(row["leg_action_delta_reduction_percent"] == pytest.approx(70., abs=1e-10) for row in measured)
    assert not (evidence["candidate"]/"training/model_final.pt").exists()


def test_preexisting_local_weight_bytes_are_deliberately_not_verified(evidence):
    (evidence["candidate"]/"training/model_final.pt").write_bytes(b"not a policy, must never be loaded")
    _, summary = load(evidence)
    assert summary["candidate_training"]["checkpoint_bytes_verified_locally"] is False
    assert "not opened" in summary["candidate_training"]["checkpoint_evidence_scope"]


@pytest.mark.parametrize("field,value,match", [
    ("status", "training", "completed formal"), ("actual_environment_transitions", 100, "12,288,000"),
    ("initialization", "resume", "from-scratch"), ("smoke_only", True, "formal candidate"),
    ("DR_recipe_modified", True, "not isolated"), ("checkpoint_sha256", "bad", "Invalid SHA"),
    ("final_training_manifest_sha256", "0"*64, "manifest SHA")])
def test_incomplete_or_nonisolated_candidate_metadata_is_rejected(evidence, field, value, match):
    path = evidence["candidate"]/"training/ablation_manifest.json"
    payload = publication.read_json(path); payload[field] = value; write_json(path, payload)
    with pytest.raises(ValueError, match=match):
        load(evidence)
    assert not evidence["output"].exists()


def test_changed_historical_training_recipe_is_rejected_even_with_refreshed_manifest_hash(evidence):
    training = evidence["candidate"]/"training"
    manifest = publication.read_json(training/"manifest.json")
    manifest["recipe"]["friction_static"] = [.1, 2.]
    write_json(training/"manifest.json", manifest)
    ablation = publication.read_json(training/"ablation_manifest.json")
    ablation["final_training_manifest_sha256"] = publication.sha256(training/"manifest.json")
    write_json(training/"ablation_manifest.json", ablation)
    with pytest.raises(ValueError, match="historical DR recipe differs"):
        load(evidence)


@pytest.mark.parametrize("field,value,match", [("checkpoint_sha256", "f"*64, "checkpoint/motion"),
    ("simulator", "MuJoCo", "PhysX recordings"), ("observation_noise_enabled", True, "deterministic"),
    ("physics_trajectory_sha256", "0"*64, "trajectory SHA")])
def test_mixed_or_unbound_candidate_recordings_are_rejected(evidence, field, value, match):
    path = evidence["candidate"]/"validation/mu0p150/isaac/report.json"
    payload = publication.read_json(path); payload[field] = value; write_json(path, payload)
    with pytest.raises(ValueError, match=match):
        load(evidence)


def test_changed_evaluator_source_is_not_blindly_pooled(evidence):
    path = evidence["candidate"]/"validation/mu0p150/isaac/report.json"
    payload = publication.read_json(path); payload["drift_module_sha256"] = "b"*64; write_json(path, payload)
    with pytest.raises(ValueError, match="frozen evaluator source differs"):
        load(evidence)


def test_failure_is_published_without_claiming_jitter_resolved(evidence):
    folder = evidence["candidate"]/"validation/mu0p150/isaac"
    with np.load(folder/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    raw["root_angular_velocity_w_rad_s"] *= 10.
    np.savez_compressed(folder/"physics_trajectory.npz", **raw)
    report = publication.read_json(folder/"report.json")
    report["physics_trajectory_sha256"] = publication.sha256(folder/"physics_trajectory.npz")
    write_json(folder/"report.json", report)
    _, summary = load(evidence)
    assert summary["engineering_screen_passes_both_conditions"] is False
    assert summary["status"] == "screen_not_passed_requires_review"
    assert summary["cases"]["mu0p150"]["comparison"]["status"] == "fails_unvalidated_screen"
    assert summary["requires_fullbody_visual_review"] is True and not summary["jitter_resolved_claimed"]


def test_post_core_pose_loss_is_visible_as_diagnostic_not_silently_a_new_gate(evidence):
    folder = evidence["candidate"]/"validation/mu0p150/isaac"
    with np.load(folder/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    late = raw["time_s"] > 5.
    raw["root_quaternion_wxyz"][late, 0] = [0., 1., 0., 0.]
    np.savez_compressed(folder/"physics_trajectory.npz", **raw)
    report = publication.read_json(folder/"report.json")
    report["physics_trajectory_sha256"] = publication.sha256(folder/"physics_trajectory.npz")
    write_json(folder/"report.json", report)
    _, summary = load(evidence)
    assert summary["engineering_screen_passes_both_conditions"] is True
    posture = summary["cases"]["mu0p150"]["candidate"]["posture_diagnostics"]
    assert posture["full"]["max_tilt_deg"] == pytest.approx(180.)
    assert posture["core"]["max_tilt_deg"] == pytest.approx(0.)
    assert not posture["full"]["posture_failure_gate"]
    assert summary["requires_fullbody_visual_review"] is True and not summary["jitter_resolved_claimed"]


def test_plot_psd_has_same_absolute_normalization_as_screen(evidence):
    arrays, summary = load(evidence)
    for case, data in summary["cases"].items():
        for cohort in ("baseline", "candidate"):
            prefix = case+"_"+cohort
            frequency = arrays[prefix+"_core_psd_frequency_hz"]
            psd = arrays[prefix+"_core_angular_xy_psd_rad2_s2_hz"]
            absolute_hf = float(np.sqrt(np.sum(psd[frequency > 10.])*(frequency[1]-frequency[0])))
            expected = data[cohort]["windows"]["core"]["body"]["angular_xy_spectrum"]["high_frequency_rms"]
            assert absolute_hf == pytest.approx(expected, rel=1e-12)


def test_fresh_publication_preserves_arrays_sources_hashes_and_refuses_overwrite(evidence, monkeypatch):
    before = {str(path): publication.sha256(path) for root in (evidence["candidate"], evidence["baseline"])
              for path in root.rglob("*") if path.is_file()}
    def render(_arrays, _summary, destination):
        Path(destination).write_bytes(b"synthetic diagnostic plot fixture")
    monkeypatch.setattr(publication, "render_diagnostic_figure", render)
    summary = summarize(evidence)
    output = evidence["output"]
    assert {path.name for path in output.iterdir()} == {"summary.json", "metrics.csv", "diagnostics.png", "derived_traces.npz"}
    assert json.loads((output/"summary.json").read_text()) == summary
    for name, item in summary["artifacts"].items():
        assert publication.sha256(output/name) == item["sha256"]
    assert summary["analysis_source_sha256"]["summarize_ppo_smoothness.py"] == publication.sha256(publication.__file__)
    arrays, _ = load(evidence)
    with np.load(output/"derived_traces.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        for name in arrays:
            np.testing.assert_array_equal(archive[name], arrays[name])
    assert {str(path): publication.sha256(path) for root in (evidence["candidate"], evidence["baseline"])
            for path in root.rglob("*") if path.is_file()} == before
    produced = {path.name: publication.sha256(path) for path in output.iterdir()}
    with pytest.raises(ValueError, match="Preserve existing publication"):
        summarize(evidence)
    assert {path.name: publication.sha256(path) for path in output.iterdir()} == produced
    for path in (output/"summary.json", output/"metrics.csv"):
        assert "/home/" not in path.read_text() and "/data/" not in path.read_text()


def test_actual_diagnostic_figure_uses_native_signals_and_shared_row_axes(evidence, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    arrays, summary = load(evidence)
    real_subplots = plt.subplots
    captured = []
    def subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured.append((fig, axes))
        return fig, axes
    monkeypatch.setattr(plt, "subplots", subplots)
    destination = tmp_path/"diagnostics.png"
    publication.render_diagnostic_figure(arrays, summary, destination)
    assert destination.stat().st_size > 1000
    _fig, axes = captured[0]
    assert axes.shape == (3, 2)
    for row in range(3):
        assert axes[row, 0].get_shared_y_axes().joined(*axes[row])
    assert len(axes[0, 0].lines[0].get_xdata()) == 1397
    assert len(axes[1, 0].lines[0].get_xdata()) == 349
    assert axes[0, 0].get_ylabel() == "Signed world omega X [rad/s]"
    assert axes[2, 0].get_yscale() == "log"
    np.testing.assert_array_equal(axes[2, 0].get_xlim(), [0., 100.])
    assert "safety/statistical" in _fig._suptitle.get_text()
