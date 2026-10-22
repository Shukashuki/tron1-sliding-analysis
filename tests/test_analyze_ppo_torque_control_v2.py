"""Synthetic, provenance-bound torque-v2 analysis; no inference/simulation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_torque_control_v2 as module
from test_analyze_ppo_ground_support_comparison import cohort, write_json, change_json
from test_train_ppo_torque_control_v2 import managers, configuration, runtime


def rebind(folder):
    metadata = module.shared.read_json(folder/"ablation_manifest.json")
    for key, name in (("final_training_manifest_sha256", "manifest.json"), ("runtime_audit_sha256", module.trainer.RUNTIME_FILE),
        ("run_status_sha256", "run_status.json")):
        metadata[key] = module.smooth.sha256(folder/name)
    write_json(folder/"ablation_manifest.json", metadata)


@pytest.fixture
def data(cohort, managers, monkeypatch):
    monkeypatch.setattr(module.trainer, "BASELINE_MOTION_SHA256", module.shared.BASELINE_MOTION_SHA256)
    monkeypatch.setattr(module.trainer, "BASELINE_METADATA", module.shared.BASELINE_METADATA)
    folder = cohort.training
    (folder/"ablation_manifest.json").unlink()
    cfg = configuration(managers)
    plan = {"output_dir": folder, "inherited_argv": [], "smoke_only": False, "num_envs": 512, "iterations": 1000,
        "expected_policy_steps": 24000, "expected_environment_transitions": 12288000}
    module.trainer.make_pre_env_hook(lambda *a, **kw: {}, plan)(cfg)
    metadata = module.shared.read_json(folder/"ablation_manifest.json")
    metadata.update(status="training_completed", child_returncode=0, trained_policy_claimed=True,
        prepared_method_runtime_verified=True, actual_environment_transitions=12288000, checkpoint_sha256="c"*64)
    write_json(folder/"ablation_manifest.json", metadata)
    manifest = module.shared.read_json(folder/"manifest.json")
    manifest.pop("contact_reward_replacement", None)
    manifest.update(training_reward_overrides={"action_rate_l2": -.5}, original_grounded_reward_preserved=True,
        baseline_action_rate_weight=-.5, additional_training_reward_terms={module.trainer.TERM_NAME: metadata["additional_reward"]})
    write_json(folder/"manifest.json", manifest)
    audit = runtime(smoke=False)
    audit["gain_readback_source"] = "robot.actuators['legs'].stiffness/damping; explicit DC motor software PD, not native PhysX drive gains"
    write_json(folder/module.trainer.RUNTIME_FILE, audit)
    write_json(folder/"runner_config.json", {"num_steps_per_env": 24})
    rebind(folder)
    for label, directory, amplitude in (("baseline", cohort.baseline, 40.), ("candidate", cohort.candidate, 10.)):
        path = directory/"physics_trajectory.npz"
        with np.load(path) as archive: raw = {name: archive[name] for name in archive.files}
        raw["applied_joint_torque_nm"][:, 0, :6] = amplitude*np.sin(2*np.pi*35*raw["time_s"])[:, None]
        np.savez(path, **raw)
        change_json(directory, "report.json", lambda report: report.update(physics_trajectory_sha256=module.smooth.sha256(path),
            runner_config_sha256=module.smooth.sha256(folder/"runner_config.json")))
    monkeypatch.setattr(module.shared, "BASELINE_TRACE_SHA256", module.smooth.sha256(cohort.baseline/"physics_trajectory.npz"))
    return cohort


def load(data, **kwargs):
    return module.load_comparison(data.candidate, data.training, data.reference, data.baseline, **kwargs)


def test_correct_torque_v2_policy_audit_and_original_grounded(data):
    raw, summary = load(data)
    assert summary["kind"] == "recorded_nominal_physx_ppo_torque_control_v2_comparison"
    assert summary["case_order"] == ["baseline", "candidate"]
    assert summary["original_grounded_reward_preserved"] and summary["original_grounded_weight"] == .4
    assert summary["candidate_additional_reward"]["weight"] == -.05
    assert summary["status"] == "requires_visual_and_tracking_review"
    assert not summary["stable_contact_claimed"] and not summary["contact_success_threshold_added"]
    provenance = summary["cases"]["candidate"]["training_provenance"]
    assert not provenance["checkpoint_bytes_verified_locally"]
    for name in ("core", "full"):
        metrics = summary["relative_changes"][name]
        assert metrics["torque_control.native_adjacent_leg_torque_change_rms_nm"]["ratio"] == pytest.approx(.25)
        assert metrics["torque_control.offline_training_formula_cost_mean"]["ratio"] == pytest.approx(.25**2)
        row = summary["cases"]["candidate"]["windows"][name]["torque_control"]
        assert row["fully_contained_control_windows"] == (128 if name == "core" else 349)
        assert row["training_equivalent_weighted_dt_contribution_mean"] == pytest.approx(-.001*row["offline_training_formula_cost_mean"])
        assert "NOT logged evaluation reward" in row["weighted_contribution_scope"]
    core = summary["cases"]["candidate"]["windows"]["core"]
    assert core["tracking"]["xy_rmse_m"] == pytest.approx(.01)
    assert not summary["cases"]["candidate"]["windows"]["full"]["tracking"]["available"]
    assert raw["candidate"]["root_position_m"].shape == (1397, 1, 3)


def test_offline_cost_retains_intercontrol_history_but_ignores_initial_reset_delta(data):
    raw, _ = load(data)
    candidate = {name: value.copy() for name, value in raw["candidate"].items()}
    candidate["applied_joint_torque_nm"][:, 0, :6] = np.r_[0., np.tile([20., 40., 60., 80.], 349)][:, None]
    candidate["applied_joint_torque_nm"][:, 0, 6:] = 1.e6  # Wheels must never enter six-leg cost.
    result = module.torque_diagnostics(candidate)
    assert result["full"]["offline_training_formula_cost_mean"] == pytest.approx((.75+348*3.)/349)
    assert result["full"]["offline_training_formula_cost_peak"] == pytest.approx(3.)
    assert result["core"]["offline_training_formula_cost_mean"] == pytest.approx(3.)


@pytest.mark.parametrize("field,value", [("kind", "historical_physx_ppo_substep_ground_support_replacement"),
    ("status", "smoke_passed"), ("smoke_only", True), ("actual_environment_transitions", 48),
    ("original_grounded_reward_preserved", False), ("baseline_checkpoint_used_for_initialization", True),
    ("wrapper_sha256", "0"*64)])
def test_reject_wrong_method_partial_or_warmstart(data, field, value):
    change_json(data.training, "ablation_manifest.json", lambda doc: doc.update({field: value}))
    with pytest.raises(ValueError): load(data)


def test_reject_wrong_added_torque_weight(data):
    change_json(data.training, "ablation_manifest.json", lambda doc: doc["additional_reward"].update(weight=-.01))
    with pytest.raises(ValueError, match="16th"): load(data)


def test_reject_fabricated_reward_count_lifecycle(data):
    change_json(data.training, "ablation_manifest.json", lambda doc: doc["pre_height_active_reward_terms_before"].append("extra"))
    with pytest.raises(ValueError, match="lifecycle"): load(data)


def test_reject_changed_runtime_audit_bytes(data):
    change_json(data.training, module.trainer.RUNTIME_FILE, lambda doc: doc.update(physics_samples_total=95999))
    with pytest.raises(ValueError, match="SHA"): load(data)


@pytest.mark.parametrize("field,value", [("physics_samples_total", 95999), ("device_health_finite", False),
    ("wheel_cost_included", True), ("original_grounded_reward_preserved", False), ("additional_reward_weight", -.01),
    ("gain_readback_source", "native PhysX drives are500/10")])
def test_reject_rebound_invalid_runtime_audit(data, field, value):
    change_json(data.training, module.trainer.RUNTIME_FILE, lambda doc: doc.update({field: value})); rebind(data.training)
    with pytest.raises(ValueError): load(data)


def test_reject_failed_ground_support_term_in_manifest(data):
    change_json(data.training, "manifest.json", lambda doc: doc.update(contact_reward_replacement={"weight": .4}))
    rebind(data.training)
    with pytest.raises(ValueError, match="original-grounded"): load(data)


def test_reject_changed_recipe_even_if_sources_same(data):
    change_json(data.training, "manifest.json", lambda doc: doc.update(setup_smoke_control_steps=2)); rebind(data.training)
    with pytest.raises(ValueError, match="Training changed"): load(data)


def test_optional_owned_status_proof_is_strict(data):
    path = data.tmp/"torque_control_v2_status.json"
    write_json(path, {"training": {"checkpoint_sha256": "c"*64,
        "ablation_manifest_sha256": module.smooth.sha256(data.training/"ablation_manifest.json"),
        "physical_sampling_audit_sha256": module.smooth.sha256(data.training/module.trainer.RUNTIME_FILE)}})
    _, summary = load(data, candidate_status=path)
    assert summary["optional_dispatcher_status"]["exact_owned_parent_artifacts_verified"]
    change_json(data.tmp, path.name, lambda doc: doc["training"].pop("physical_sampling_audit_sha256"))
    with pytest.raises(ValueError, match="dispatcher proof"): load(data, candidate_status=path)


def test_local_weights_checked_only_if_available(data):
    (data.training/"model_final.pt").write_bytes(b"wrong local weights")
    with pytest.raises(ValueError, match="Local checkpoint"): load(data)


def test_nooverwrite_and_raw_states_not_smoothed(data, monkeypatch):
    output = data.tmp/"new_analysis"
    original_sha = module.smooth.sha256(data.candidate/"physics_trajectory.npz")
    monkeypatch.setattr(module, "render_contact_plot", lambda raw, summary, path: path.write_bytes(b"test plot only"))
    args = SimpleNamespace(candidate_dir=data.candidate, candidate_training_dir=data.training, baseline_dir=data.baseline,
        reference=data.reference, candidate_status=None, output_dir=output)
    result = module.run(args)
    assert set(result["artifacts"]) == {"measured_states.npz", "contact_comparison.png"}
    assert module.smooth.sha256(data.candidate/"physics_trajectory.npz") == original_sha
    with np.load(output/"measured_states.npz") as archive:
        assert archive["candidate_applied_joint_torque_nm"].shape == (1397, 1, 8)
        assert not any("actions" in name for name in archive.files)
    with pytest.raises(ValueError, match="fresh"): module.run(args)
