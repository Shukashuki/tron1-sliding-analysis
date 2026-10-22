"""Synthetic native-recording/provenance tests; no Isaac, Torch or GPU required."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_ground_support_comparison as module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def rebind_training(folder):
    ablation = module.read_json(folder/"ablation_manifest.json")
    for key, name in (("final_training_manifest_sha256", "manifest.json"),
                      ("runtime_audit_sha256", module.RUNTIME_FILE), ("run_status_sha256", "run_status.json")):
        ablation[key] = module.smooth.sha256(folder/name)
    write_json(folder/"ablation_manifest.json", ablation)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    baseline = tmp_path/"baseline"/"validation"/"mu0p700"/"isaac"
    candidate = tmp_path/"candidate"/"validation"/"mu0p700"/"isaac"
    training = tmp_path/"candidate"/"training"
    baseline_training = tmp_path/"baseline"/"training"
    for directory in (baseline, candidate, training, baseline_training):
        directory.mkdir(parents=True)
    reference = tmp_path/"motion.npz"
    ref_time = np.arange(349)*.02
    ref_position = np.column_stack((ref_time, np.zeros((349, 2))))
    ref_position[:, 2] = .75
    ref_quat = np.tile([1., 0., 0., 0.], (349, 1))
    np.savez(reference, reference_time_s=ref_time, reference_position=ref_position,
             reference_com_velocity=np.tile([1., 0., 0.], (349, 1)), reference_quaternion=ref_quat)
    motion_sha = module.smooth.sha256(reference)
    monkeypatch.setattr(module, "BASELINE_MOTION_SHA256", motion_sha)
    checkpoint = "c"*64
    replacement = {"term": "drift_grounded", "weight": .4, "scale_fraction": .20,
        "additional_reward_term": False, "raw_reward_bounds": [0., 1.], "reward_manager_dt_factor_applied_once": .02,
        "new_formula": "1-mean_4_substeps(mean_2_wheels(exp(-clamp(ground_Fz_N,min=0)/(.20*mg_N))))"}
    shared_manifest = {key: {"same": key} for key in module.TRAINING_SHARED_KEYS}
    shared_manifest.update(algorithm="PPO", reference_id="drift_drl_traj4", seed=42, num_envs=512,
        iterations=1000, num_steps_per_env=24, policy_steps_requested=24000,
        environment_transitions_requested=12288000, setup_smoke_control_steps=0,
        trainer_sha256=module.BASELINE_TRAINER_SHA256, motion_sha256=motion_sha,
        asset_sha256=module.BASELINE_ASSET_SHA256, task_profile_sha256=module.BASELINE_PROFILE_SHA256,
        source_training_modules_sha256=module.BASELINE_SOURCES)
    write_json(baseline_training/"manifest.json", shared_manifest)
    write_json(baseline_training/"ablation_manifest.json", {"status": "training_completed", "synthetic": True})
    write_json(baseline_training/"run_status.json", {"status": "training_completed", "synthetic": True})
    baseline_hashes = {name: module.smooth.sha256(baseline_training/name) for name in module.BASELINE_METADATA}
    monkeypatch.setattr(module, "BASELINE_METADATA", baseline_hashes)
    manifest = copy.deepcopy(shared_manifest)
    manifest.update(training_reward_overrides={"action_rate_l2": -.5, "drift_grounded": replacement},
        contact_reward_replacement=replacement, comparison_baseline_action_rate_weight=-.5,
        historical_profile_hash_does_not_include_explicit_reward_override=True)
    write_json(training/"manifest.json", manifest)
    write_json(training/"runner_config.json", {"algorithm": "PPO", "synthetic": True})
    write_json(training/"run_status.json", {"status": "training_completed", "trained_policy_claimed": True,
        "environment_transitions": 12288000, "policy_steps_completed": 24000})
    audit = {"module_sha256": "d"*64, "wrapper_sha256": "e"*64,
        "clock_helper_sha256": module.smooth.sha256(module.ROOT/"scripts/ppo_leg_torque_smoothness.py"),
        "steps_started": 24000, "steps_completed": 24000, "unique_cost_windows": 24000,
        "physics_samples_total": 96000, "steps_failed": 0, "samples_last_window": 4, "num_envs": 512,
        "closed": True, "device_health_checked": True, "device_health_finite": True, "cpu_fixture_only": False,
        "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4, "scale_fraction": .20,
        "external_reward_weight": .4, "raw_reward_bounds": [0., 1.], "mass_gravity_reads": 1,
        "sensor_names": ["wheel_L_ground", "wheel_R_ground"], "wheel_body_names": ["wheel_L_Link", "wheel_R_Link"],
        "force_matrix_shape_each_sensor": [512, 1, 1, 3],
        "ground_filter": {"path": "/World/ground/terrain/GroundPlane/CollisionPlane", "collision_api_verified": True},
        "gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]},
        "mass_kg_range": [22., 22.], "mg_N_range": [215.82, 215.82], "gravity_m_s2": [0., 0., -9.81],
        "positive_normal_force_samples_left_right": [100, 100], "negative_normal_force_samples": 0}
    write_json(training/module.RUNTIME_FILE, audit)
    ablation = {"kind": module.KIND, "status": "training_completed", "smoke_only": False, "child_returncode": 0,
        "trained_policy_claimed": True, "prepared_method_runtime_verified": True,
        "initialization": "from_scratch_no_checkpoint_resume", "num_envs": 512, "iterations": 1000, "seed": 42,
        "expected_policy_steps": 24000, "expected_environment_transitions": 12288000,
        "actual_environment_transitions": 12288000, "baseline_checkpoint_sha256": module.BASELINE_CHECKPOINT_SHA256,
        "baseline_metadata_sha256": baseline_hashes, "baseline_action_rate_weight": -.5, "source_default_action_rate_weight": -.1,
        "historical_trainer_sha256": module.BASELINE_TRAINER_SHA256, "height_adapter_sha256": module.BASELINE_HEIGHT_SHA256,
        "task_profile_sha256": module.BASELINE_PROFILE_SHA256, "motion_sha256": motion_sha,
        "asset_sha256": module.BASELINE_ASSET_SHA256, "baseline_source_sha256": module.BASELINE_SOURCES,
        "reward_replacement": replacement, "checkpoint_sha256": checkpoint,
        "wrapper_sha256": audit["wrapper_sha256"], "sampler_module_sha256": audit["module_sha256"]}
    write_json(training/"ablation_manifest.json", ablation)
    rebind_training(training)
    n = 1397; time = np.arange(n)*.005
    joint_names = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"]
    bodies = ["base_Link", "abad_L_Link", "abad_R_Link", "limx_imu", "hip_L_Link", "hip_R_Link", "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link"]
    shared_report = {key: {"shared": key} for key in module.PHYSICAL_KEYS}
    shared_report.update(status="evaluated", simulator="IsaacLab/PhysX", full_horizon_diagnostic=True,
        terminal_states_captured_before_auto_reset=True, domain_randomization_enabled=False, observation_noise_enabled=False,
        evaluation_contact_friction_override={"evaluation_contact_friction": .7}, motion_file_sha256=motion_sha,
        contact_slip_measurement={"active_point_normal_force_threshold_n": 5., "sample_dt_s": .02},
        contact_material_audit={"leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]}},
        runner_config_sha256=module.smooth.sha256(training/"runner_config.json"))
    for label, directory, amplitude in (("baseline", baseline, 2.), ("candidate", candidate, .5)):
        counts = np.ones((n, 1, 2), dtype=np.int64)
        counts[::3, 0, 0] = 0
        force = np.zeros((n, 1, 2, 3)); force[..., 2] = 100.
        force[::5, 0, 1, 2] = 3.  # count is present but fails the independent upward-load gate.
        angular = np.column_stack((amplitude*np.sin(2*np.pi*35*time), np.zeros((n, 2))))
        raw = {"time_s": time, "valid_mask": np.ones((n, 1), dtype=bool),
            "simulation_step_counter": np.arange(n), "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
            "root_position_m": np.column_stack((time+.01, np.zeros(n), np.full(n, .75)))[:, None],
            "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (n, 1))[:, None],
            "root_linear_velocity_w_m_s": np.tile([1., 0., 0.], (n, 1))[:, None],
            "root_angular_velocity_w_rad_s": angular[:, None], "wheel_ground_contact_count": counts,
            "wheel_ground_contact_force_w_n": force, "nonwheel_ground_contact_count": np.zeros((n, 1)),
            "joint_position_rad": np.zeros((n, 1, 8)), "joint_velocity_rad_s": np.zeros((n, 1, 8)),
            "applied_joint_torque_nm": np.zeros((n, 1, 8)), "actions_clipped": np.zeros((n, 1, 8)),
            "action_was_applied": np.r_[False, np.ones(n-1, dtype=bool)][:, None],
            "reference_frame_for_action": np.r_[0, np.repeat(np.arange(349), 4)][:, None],
            "wheel_body_names": np.array(["wheel_L_Link", "wheel_R_Link"]),
            "body_names": np.array(bodies), "joint_names": np.array(joint_names)}
        np.savez(directory/"physics_trajectory.npz", **raw)
        report = copy.deepcopy(shared_report)
        report["checkpoint_sha256"] = module.BASELINE_CHECKPOINT_SHA256 if label == "baseline" else checkpoint
        report["physics_trajectory_sha256"] = module.smooth.sha256(directory/"physics_trajectory.npz")
        report["policy_export"] = {"actor_dim": 51, "critic_dim": 120, "leg_kp": 500., "leg_kd": 10.,
            "physics_dt": .005, "decimation": 4, "checkpoint_sha256": report["checkpoint_sha256"], "actual_body_order": bodies}
        write_json(directory/"report.json", report)
    monkeypatch.setattr(module, "BASELINE_TRACE_SHA256", module.smooth.sha256(baseline/"physics_trajectory.npz"))
    return SimpleNamespace(baseline=baseline, candidate=candidate, training=training, reference=reference,
        baseline_training=baseline_training, tmp=tmp_path)


def load(data):
    return module.load_comparison(data.candidate, data.training, data.reference, data.baseline)


def change_json(folder, filename, update):
    value = module.read_json(folder/filename)
    update(value)
    write_json(folder/filename, value)


def test_full_audited_native_comparison_no_threshold_or_smoothing(cohort):
    raw, result = load(cohort)
    assert result["case_order"] == ["baseline", "candidate"]
    assert result["status"] == "requires_visual_and_tracking_review"
    assert not result["stable_contact_claimed"] and not result["contact_success_threshold_added"]
    assert not result["pose_smoothing_or_state_correction"]
    assert not result["cases"]["candidate"]["training_provenance"]["checkpoint_bytes_verified_locally"]
    assert raw["candidate"]["root_position_m"].shape == (1397, 1, 3)
    assert np.array_equal(raw["candidate"]["time_s"], np.arange(1397)*.005)
    for label in module.CASE_ORDER:
        core = result["cases"][label]["windows"]["core"]
        assert core["tracking"]["xy_rmse_m"] == pytest.approx(.01)
        assert core["tracking"]["speed_retention"] == pytest.approx(1.)
        assert core["tracking"]["beta_rmse_deg"] == pytest.approx(0.)
        assert core["tilt"]["peak_deg"] == 0.
        assert core["contact"]["measured_force_bearing_contact"]["right"]["sample_duty_fraction"] == 1.
        assert core["contact"]["load_eligibility"]["right"]["sample_duty_fraction"] < 1.
        assert core["contact"]["measured_force_bearing_contact"]["left"]["state_switch_count"] > 0
        assert core["force_z"]["right"]["step_abs_peak_n"] == 97.
        assert not result["cases"][label]["windows"]["full"]["tracking"]["available"]
        assert result["cases"][label]["reference_overlap_tracking"]["interval_s"] == [0., 6.96]
        assert result["cases"][label]["reference_overlap_tracking"]["available"]
    hf = result["relative_changes"]["core"]["body_hf_rms.angular_world_xy"]
    assert hf["ratio"] == pytest.approx(.25)


@pytest.mark.parametrize("field,value", [("status", "configured_before_environment_creation"),
    ("smoke_only", True), ("trained_policy_claimed", False), ("prepared_method_runtime_verified", False),
    ("actual_environment_transitions", 48), ("checkpoint_sha256", "broken")])
def test_reject_nonfinal_or_partial_training(cohort, field, value):
    change_json(cohort.training, "ablation_manifest.json", lambda doc: doc.update({field: value}))
    with pytest.raises(ValueError):
        load(cohort)


def test_reject_changed_training_audit_bytes_even_valid_json(cohort):
    change_json(cohort.training, module.RUNTIME_FILE, lambda doc: doc.update(physics_samples_total=95999))
    with pytest.raises(ValueError, match="hash binding"):
        load(cohort)


@pytest.mark.parametrize("field,value", [("physics_samples_total", 95999), ("device_health_finite", False),
    ("cpu_fixture_only", True), ("negative_normal_force_samples", 1), ("positive_normal_force_samples_left_right", [0, 42]),
    ("mass_kg_range", [0., 22.]), ("gravity_m_s2", [0., 0., 9.81]), ("physics_dt_s", .02)])
def test_reject_invalid_rebound_runtime_audit(cohort, field, value):
    change_json(cohort.training, module.RUNTIME_FILE, lambda doc: doc.update({field: value}))
    rebind_training(cohort.training)
    with pytest.raises(ValueError):
        load(cohort)


@pytest.mark.parametrize("field,value", [("domain_randomization_enabled", True), ("observation_noise_enabled", True),
    ("full_horizon_diagnostic", False), ("simulator", "MuJoCo"),
    ("evaluation_contact_friction_override", {"evaluation_contact_friction": .15}),
    ("physics_trajectory_sha256", "0"*64), ("checkpoint_sha256", "0"*64), ("runner_config_sha256", "0"*64)])
def test_reject_wrong_evaluation_provenance(cohort, field, value):
    change_json(cohort.candidate, "report.json", lambda doc: doc.update({field: value}))
    with pytest.raises(ValueError):
        load(cohort)


def test_reject_mismatched_actual_pd_gain(cohort):
    change_json(cohort.candidate, "report.json", lambda doc: doc["contact_material_audit"]["leg_pd_gain_readback"].update(kd_range=[5., 5.]))
    with pytest.raises(ValueError, match="PD"):
        load(cohort)


@pytest.mark.parametrize("key", ["evaluator_source_sha256", "contact_slip_helper_sha256", "contact_material_audit"])
def test_reject_changed_physical_contract(cohort, key):
    change_json(cohort.candidate, "report.json", lambda doc: doc[key].update(changed=True))
    with pytest.raises(ValueError, match="Shared physical"):
        load(cohort)


def test_reject_observation_contract_change(cohort):
    change_json(cohort.candidate, "report.json", lambda doc: doc["policy_export"].update(extra_sensor_observation=True))
    with pytest.raises(ValueError, match="observation"):
        load(cohort)


def test_reject_training_recipe_change_despite_rebound_hash(cohort):
    change_json(cohort.training, "manifest.json", lambda doc: doc.update(setup_smoke_control_steps=2))
    rebind_training(cohort.training)
    with pytest.raises(ValueError, match="Training recipe"):
        load(cohort)


def test_reject_truncated_or_spliced_native_recording(cohort):
    path = cohort.candidate/"physics_trajectory.npz"
    with np.load(path) as archive:
        raw = {key: archive[key] for key in archive.files}
    raw["valid_mask"][700, 0] = False
    np.savez(path, **raw)
    change_json(cohort.candidate, "report.json", lambda doc: doc.update(physics_trajectory_sha256=module.smooth.sha256(path)))
    with pytest.raises(ValueError, match="native 200Hz"):
        load(cohort)


def test_local_checkpoint_bytes_are_verified_if_present(cohort):
    (cohort.training/"model_final.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="Local candidate checkpoint"):
        load(cohort)


def test_optional_dispatcher_status_binding(cohort):
    status_path = cohort.tmp/"ground_support_status.json"
    write_json(status_path, {"training": {"checkpoint_sha256": "c"*64,
        "ablation_manifest_sha256": module.smooth.sha256(cohort.training/"ablation_manifest.json"),
        "physical_sampling_audit_sha256": module.smooth.sha256(cohort.training/module.RUNTIME_FILE)}})
    _, result = module.load_comparison(cohort.candidate, cohort.training, cohort.reference, cohort.baseline,
                                     candidate_status=status_path)
    assert result["optional_dispatcher_status"]["checkpoint_binding_verified"]
    change_json(cohort.tmp, status_path.name, lambda doc: doc["training"].update(checkpoint_sha256="0"*64))
    with pytest.raises(ValueError, match="dispatcher checkpoint"):
        module.load_comparison(cohort.candidate, cohort.training, cohort.reference, cohort.baseline, candidate_status=status_path)


def test_fresh_outputs_measured_states_only_and_no_source_modification(cohort, monkeypatch):
    output = cohort.tmp/"publication"
    before = module.smooth.sha256(cohort.candidate/"physics_trajectory.npz")
    def plot(raw, summary, path):
        Path(path).write_bytes(b"synthetic plot placeholder")
    monkeypatch.setattr(module, "render_contact_plot", plot)
    args = SimpleNamespace(candidate_dir=cohort.candidate, candidate_training_dir=cohort.training,
        baseline_dir=cohort.baseline, reference=cohort.reference, output_dir=output, candidate_status=None)
    result = module.run(args)
    assert set(result["artifacts"]) == {"measured_states.npz", "contact_comparison.png"}
    assert module.smooth.sha256(cohort.candidate/"physics_trajectory.npz") == before
    with np.load(output/"measured_states.npz") as archive:
        assert "candidate_root_position_m" in archive
        assert not any("actions" in key or "reference_frame" in key for key in archive.files)
    with pytest.raises(ValueError, match="fresh"):
        module.run(args)


def test_parse_cli_is_cpu_only(tmp_path):
    args = module.parser().parse_args(["--candidate-dir", "a", "--candidate-training-dir", "b", "--output-dir", str(tmp_path/"out")])
    assert args.candidate_dir == Path("a")
