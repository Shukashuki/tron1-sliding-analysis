"""Asset-free audit fixtures; no AppLauncher, learned inference or mj_step."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(Path(__file__).parent)]
import analyze_controller_pair as pair
from controller_metrics import heading_signals
from controller_references import get_reference
from controller_trace import native_trace
from test_controller_trace import (canonical_data, encoded_trace, measured_kinematic_trace,
                                   synthetic_tron1_kinematic_model)


CHECKPOINT = "c"*64
IA_HASH, MJ_HASH, MODEL_HASH = "1"*64, "2"*64, "3"*64


def height_contract():
    return {"mode": "vertical_acceleration_only", "weight": -.1,
            "acceleration_scale_m_s2": 5., "absolute_height_reward": False,
            "height_termination": False, "reference_vertical_velocity_reward": False,
            "position_and_linear_velocity_tracking_axes": ["x", "y"]}


def policy_reports(algorithm="PPO"):
    contract = {"root_prescribed_during_steps": False, "motion_end_hidden_teleport": False,
                "policy_algorithm": algorithm, "checkpoint_sha256": CHECKPOINT,
                "height_regularization": height_contract()}
    ia = {"status": "evaluated", "simulator": "IsaacLab/PhysX", "physics_trajectory_sha256": IA_HASH,
          "motion_file_sha256": get_reference()["motion_npz_sha256"],
          "checkpoint_sha256": CHECKPOINT, "policy_export": copy.deepcopy(contract),
          "reference_state_initialization_during_evaluation": False,
          "domain_randomization_enabled": False, "observation_noise_enabled": False,
          "terminal_states_captured_before_auto_reset": True, "episodes_per_environment": 1,
          "deterministic_start_frame": 0, "full_horizon_diagnostic": True,
          "physics_recording": {"terminal_states_before_reset": True,
                                "same_counter_reset_and_forward_updates_excluded": True,
                                "first_episode_only_valid_mask": True},
          "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
          "contact_material_audit": {"expected_friction": .7, "robot_runtime_friction_range": [.7, .7],
                                     "ground_colliders": [{"static_friction": .7, "dynamic_friction": .7}]}}
    mj = {"status": "completed", "engine": "MuJoCo", "rollout_sha256": MJ_HASH,
          "motion_sha256": get_reference()["motion_npz_sha256"], "model_sha256": MODEL_HASH,
          "contract": copy.deepcopy(contract), "physics_stepped": True, "root_state_writes": 1,
          "hidden_resets": 0, "domain_parameters_resampled_during_episode": False,
          "full_horizon_diagnostic": True,
          "contact_material_audit": {"expected_friction": .7, "compiled_friction_range": [.7, .7]}}
    for report in (ia, mj):
        report.update(wheel_lateral_contact_slip_evidence_kind="actual_ground_contact_points",
                      contact_slip_measurement={"ground_is_stationary": True})
    return ia, mj


def nmpc_reports():
    objective = {"absolute_height_target": False, "reference_vertical_velocity_target": False,
                 "height_regularization": {"mode": "vertical_acceleration_only"},
                 "weights": {"speed": 1.}, "stds": {"speed": .35}, "dt_s": .02,
                 "reference_order": "held_pre_action"}
    shared = {"status": "completed", "motion_sha256": get_reference()["motion_npz_sha256"],
              "reference_id": get_reference()["reference_id"], "learned_policy_used": False,
              "tvlqr_used": False, "root_state_prescribed_after_reset": False,
              "external_root_forces_used": False, "prediction_receives_plant_mu": False,
              "truncated_smoke": False, "prediction_mu_scenarios": [.7, 1.2],
              "plant_mu": .18, "pose_height_failure_gate": False, "objective_contract": objective,
              "effective_cem_config": {"horizon_steps": 3, "control_blocks": 1, "candidates": 4,
                                       "iterations": 2, "elite_count": 2, "seed": 42},
              "contact_slip_definition": "Measured contact-point force-weighted lateral-speed RMS"}
    ia, mj = copy.deepcopy(shared), copy.deepcopy(shared)
    ia.update(engine="Isaac/PhysX", rollout_sha256=IA_HASH, mujoco_model_sha256=MODEL_HASH,
              fixed_evaluation={"all_termination_terms_removed": True},
              material_audit={"expected_mu": .18, "robot_static_dynamic_range": [.18, .18],
                              "ground_bound_materials": [{"coefficients": [.18, .18, 0.]}]})
    mj.update(engine="MuJoCo", rollout_sha256=MJ_HASH, model_sha256=MODEL_HASH,
              plant_material_readback={"compiled_sliding_friction_range": [.18, .18]})
    return ia, mj


def audited(algorithm, reports, **kwargs):
    return pair.audit_reports(algorithm, get_reference(), *reports,
                              isaac_trace_sha256=IA_HASH, mujoco_trace_sha256=MJ_HASH,
                              model_sha256=MODEL_HASH, **kwargs)


@pytest.mark.parametrize("algorithm", ["PPO", "SAC", "NMPC"])
def test_explicit_algorithms_matched_fixed_mu_and_report_bound_trace_identity(algorithm):
    result = audited(algorithm, nmpc_reports() if algorithm == "NMPC" else policy_reports(algorithm))
    assert result["fixed_mu"] == (.18 if algorithm == "NMPC" else .7)
    assert result["trace_bindings"]["mujoco"]["hash_source"] == "report"
    assert result["height_free_objective_and_evaluation"] is True


def test_legacy_mujoco_hash_requires_explicit_capture_hash_never_isaac_observation_hash():
    ia, mj = policy_reports(); del mj["rollout_sha256"]
    mj["isaac_observation_validation"] = {"trajectory_sha256": MJ_HASH}
    with pytest.raises(ValueError, match="independently captured"):
        audited("PPO", (ia, mj))
    with pytest.raises(ValueError, match="SHA-256"):
        audited("PPO", (ia, mj), mujoco_capture_sha256=IA_HASH)
    result = audited("PPO", (ia, mj), mujoco_capture_sha256=MJ_HASH)
    assert result["trace_bindings"]["mujoco"]["hash_source"] == "user_supplied_capture_sha256"


@pytest.mark.parametrize("engine,path,value", [
    (0, ("physics_trajectory_sha256",), "0"*64),
    (0, ("simulator",), "kinematic_playback"),
    (1, ("engine",), "Isaac/PhysX"),
    (1, ("motion_sha256",), "0"*64),
    (1, ("model_sha256",), "0"*64),
    (0, ("contact_material_audit", "robot_runtime_friction_range"), [.6, .7]),
    (0, ("contact_material_audit", "ground_colliders"), []),
    (1, ("contact_material_audit", "compiled_friction_range"), [.18, .18]),
    (0, ("reference_state_initialization_during_evaluation",), True),
    (0, ("domain_randomization_enabled",), True),
    (0, ("terminal_states_captured_before_auto_reset",), False),
    (0, ("wheel_lateral_contact_slip_evidence_kind",), "wheel_center_surrogate"),
    (1, ("contact_slip_measurement", "ground_is_stationary"), False),
    (0, ("physics_recording", "first_episode_only_valid_mask"), False),
    (1, ("physics_stepped",), False),
    (1, ("root_state_writes",), 2),
    (1, ("hidden_resets",), 1),
    (1, ("contract", "root_prescribed_during_steps"), True),
    (1, ("contract", "motion_end_hidden_teleport"), True),
    (1, ("contract", "policy_algorithm"), "SAC"),
    (1, ("contract", "checkpoint_sha256"), "0"*64),
    (1, ("contract", "wheel_torque_scale_nm"), 8.),
    (0, ("policy_export", "height_regularization", "height_termination"), True),
    (1, ("full_horizon_diagnostic",), False),
])
def test_reject_policy_provenance_material_resets_prescription_and_height_gates(engine, path, value):
    reports = policy_reports(); target = reports[engine]
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        audited("PPO", reports)


@pytest.mark.parametrize("engine,path,value", [
    (0, ("prediction_receives_plant_mu",), True),
    (0, ("fixed_evaluation", "all_termination_terms_removed"), False),
    (1, ("contact_slip_definition",), None),
    (1, ("prediction_mu_scenarios",), [.18]),
    (0, ("learned_policy_used",), True),
    (0, ("action_replay",), True),
    (1, ("root_state_prescribed_after_reset",), True),
    (1, ("external_root_forces_used",), True),
    (1, ("plant_material_readback", "compiled_sliding_friction_range"), []),
    (1, ("plant_mu",), .7),
    (1, ("objective_contract", "weights"), {"old_qr": 1.}),
    (0, ("pose_height_failure_gate",), True),
])
def test_nmpc_rejects_oracle_scenarios_learned_replay_missing_material_and_objective_mismatch(engine, path, value):
    reports = nmpc_reports(); target = reports[engine]
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        audited("NMPC", reports, integration_only=True)


@pytest.mark.parametrize("integration_only", [False, True])
@pytest.mark.parametrize("field,value", [("horizon_steps", 30), ("control_blocks", 2),
                                         ("candidates", 64), ("iterations", 3),
                                         ("elite_count", 3), ("seed", 314),
                                         ("parameterization", "warm_start_residual_blocks")])
def test_nmpc_rejects_mismatched_effective_search_config_even_in_integration_mode(field, value, integration_only):
    reports = nmpc_reports()
    reports[1]["effective_cem_config"][field] = value
    with pytest.raises(ValueError, match="effective_cem_config"):
        audited("NMPC", reports, integration_only=integration_only)


@pytest.mark.parametrize("problem", ["one_missing", "both_missing", "empty", "not_dict"])
def test_nmpc_effective_search_config_is_required_nonempty_object(problem):
    reports = nmpc_reports()
    if problem == "one_missing": del reports[0]["effective_cem_config"]
    elif problem == "both_missing":
        for report in reports: del report["effective_cem_config"]
    else:
        for report in reports: report["effective_cem_config"] = {} if problem == "empty" else []
    with pytest.raises(ValueError, match="effective_cem_config"):
        audited("NMPC", reports)


def test_legacy_absolute_nmpc_without_new_contracts_still_passes_and_records_its_identity():
    reports = nmpc_reports()
    result = audited("NMPC", reports)["nmpc_controller_identity"]
    assert result["exact_configuration_match"] is True
    assert result["parameterization_mode"] == "absolute_blocks"
    assert result["legacy_absolute_without_parameterization_contract"] is True
    assert result["action_parameterization_contract"] is None
    result["effective_cem_config"]["seed"] = 314
    assert reports[0]["effective_cem_config"]["seed"] == 42


@pytest.mark.parametrize("mode", ["absolute_blocks", "warm_start_residual_blocks"])
def test_matching_explicit_nmpc_parameterization_contracts_agree_with_config(mode):
    reports = nmpc_reports()
    contract = {"mode": mode, "baseline_copied_and_fixed_per_solve": mode == "warm_start_residual_blocks"}
    for report in reports:
        report["effective_cem_config"]["parameterization"] = mode
        report["action_parameterization"] = copy.deepcopy(contract)
    result = audited("NMPC", reports)["nmpc_controller_identity"]
    assert result["parameterization_mode"] == mode
    assert result["action_parameterization_contract"] == contract
    assert result["legacy_absolute_without_parameterization_contract"] is False
    result["action_parameterization_contract"]["mode"] = "changed"
    assert reports[0]["action_parameterization"]["mode"] == mode


@pytest.mark.parametrize("integration_only", [False, True])
@pytest.mark.parametrize("problem", ["isaac_only", "mujoco_only", "different_contract", "wrong_mode", "null", "empty", "residual_missing"])
def test_nmpc_parameterization_contract_missing_mismatch_or_inconsistent_mode_rejected(problem, integration_only):
    reports = nmpc_reports()
    for report in reports:
        report["effective_cem_config"]["parameterization"] = "warm_start_residual_blocks"
        report["action_parameterization"] = {"mode": "warm_start_residual_blocks", "baseline_copied_and_fixed_per_solve": True}
    if problem == "isaac_only": del reports[1]["action_parameterization"]
    elif problem == "mujoco_only": del reports[0]["action_parameterization"]
    elif problem == "different_contract": reports[1]["action_parameterization"]["baseline_copied_and_fixed_per_solve"] = False
    elif problem == "wrong_mode":
        for report in reports: report["action_parameterization"]["mode"] = "absolute_blocks"
    elif problem in ("null", "empty"):
        for report in reports: report["action_parameterization"] = None if problem == "null" else {}
    else:
        for report in reports: del report["action_parameterization"]
    with pytest.raises(ValueError, match="action_parameterization"):
        audited("NMPC", reports, integration_only=integration_only)


def test_smoke_is_explicit_integration_only_and_never_ranking():
    reports = nmpc_reports()
    for report in reports: report.update(status="smoke_completed", truncated_smoke=True)
    with pytest.raises(ValueError, match="integration-only"):
        audited("NMPC", reports)
    result = audited("NMPC", reports, integration_only=True)
    assert result["eligible_for_controller_ranking"] is False
    # An archived height-tracking PPO smoke is still only an integration input.
    ia, mj = policy_reports()
    del ia["policy_export"]["height_regularization"]
    del mj["contract"]["height_regularization"]
    result = audited("PPO", (ia, mj), integration_only=True)
    assert result["height_free_objective_and_evaluation"] is False
    assert result["eligible_for_controller_ranking"] is False


def training_fixture(tmp_path, algorithm="PPO"):
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "model_final.pt"; checkpoint.write_bytes(b"opaque checkpoint; never deserialize")
    checkpoint_hash = pair.sha256(checkpoint)
    ia, mj = policy_reports(algorithm)
    for report in (ia, mj):
        contract = report["policy_export"] if report is ia else report["contract"]
        contract["checkpoint_sha256"] = checkpoint_hash
    ia["checkpoint_sha256"] = checkpoint_hash
    recipe = {"friction_static": [.3, 1.6], "friction_dynamic": [.3, 1.2], "material_buckets": 64}
    profile = pair.read_json(ROOT / "config/car_drift_height_smooth_task.json")
    training = {"algorithm": algorithm, "smoke_only": False,
                "motion_sha256": get_reference()["motion_npz_sha256"],
                "reference_id": get_reference()["reference_id"], "reference_catalog_sha256": pair.sha256(pair.DEFAULT_CATALOG),
                "pooled_success_statistics": False, "task_profile": profile,
                "task_profile_sha256": pair.sha256(ROOT / "config/car_drift_height_smooth_task.json"),
                "height_adapter_sha256": pair.sha256(ROOT / "scripts/height_smoothness.py"),
                "height_regularization": height_contract(), "recipe": recipe}
    status = {"status": "training_completed", "trained_policy_claimed": True}
    for contract in (ia["policy_export"], mj["contract"]):
        contract.update(trained_task_profile=profile, height_regularization=height_contract(), training_domain_randomization=recipe)
    manifest.write_text(json.dumps(training)); manifest.with_name("run_status.json").write_text(json.dumps(status))
    return manifest, training, status, ia, mj


@pytest.mark.parametrize("algorithm", ["PPO", "SAC"])
def test_formal_training_requires_completed_run_and_hashes_opaque_checkpoint_bytes(tmp_path, algorithm):
    path, _, _, ia, mj = training_fixture(tmp_path, algorithm)
    result = pair.audit_training(path, algorithm, get_reference(), ia, mj)
    assert result["checkpoint_bytes_verified"] is True
    assert result["verified_formal_training"] is True
    path.with_name("model_final.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Actual checkpoint"):
        pair.audit_training(path, algorithm, get_reference(), ia, mj)


@pytest.mark.parametrize("change", ["running", "smoke", "wrong_algorithm", "wrong_motion", "wrong_profile", "old_height_adapter", "wrong_reference", "different_dr", "missing_checkpoint"])
def test_formal_training_negative_provenance_cases(tmp_path, change):
    path, training, status, ia, mj = training_fixture(tmp_path)
    if change == "running": status["status"] = "training"
    elif change == "smoke": training["smoke_only"] = True
    elif change == "wrong_algorithm": training["algorithm"] = "SAC"
    elif change == "wrong_motion": training["motion_sha256"] = "0"*64
    elif change == "wrong_profile": training["task_profile_sha256"] = "0"*64
    elif change == "old_height_adapter": training["height_adapter_sha256"] = "0"*64
    elif change == "wrong_reference": training["reference_id"] = "doa_sa29_scale2"
    elif change == "different_dr": ia["policy_export"]["training_domain_randomization"] = {}
    else: path.with_name("model_final.pt").unlink()
    path.write_text(json.dumps(training)); path.with_name("run_status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError):
        pair.audit_training(path, "PPO", get_reference(), ia, mj)


def test_training_manifest_mandatory_formal_and_not_accepted_for_nmpc():
    ia, mj = policy_reports()
    with pytest.raises(ValueError, match="training-manifest"):
        pair.audit_training(None, "PPO", get_reference(), ia, mj)
    assert not pair.audit_training(None, "PPO", get_reference(), ia, mj, integration_only=True)["verified_formal_training"]
    with pytest.raises(ValueError, match="NMPC"):
        pair.audit_training(Path("unused"), "NMPC", get_reference(), ia, mj)


def geometry(trace):
    wheels = np.tile([1., 0., 0., 0.], (len(trace["time_s"]), 2, 1))
    velocity = np.repeat(trace["com_velocity"][:, None], 2, axis=1)
    return {"signals": heading_signals(trace["quaternion"], trace["com_velocity"], wheels, velocity),
            "wheel_quaternion_wxyz": wheels, "wheel_center_velocity_w": velocity, "audit": {}}


def test_partial_core_censored_offsets_are_features_and_contact_patch_slip_is_separate():
    reference = canonical_data(1001); reference["time_s"] = np.arange(1001)*.005
    reference["position"][:, :2] = np.column_stack((reference["time_s"], np.zeros(1001)))
    reference["com_velocity"][:] = [1., 0., 0.]
    isaac, mujoco = copy.deepcopy(reference), copy.deepcopy(reference)
    isaac["position"][:, 0] += .1; mujoco["position"][:, 0] += .2
    isaac["contact_slip"][:] = [.4, 999.]
    traces = {"reference": reference, "isaac": isaac, "mujoco": mujoco}
    result = pair.summarize_pair(get_reference(), traces, {name: geometry(trace) for name, trace in traces.items()})
    core = result["core"]
    assert core["reference"]["ground_contact_heading_geometry"] is None
    assert core["reference"]["ground_contact_lateral_slip_m_s"] is None
    assert core["isaac"]["reference_xy_rmse_m"] == pytest.approx(.1)
    assert core["mujoco"]["reference_xy_rmse_m"] == pytest.approx(.2)
    assert core["sim2sim_xy_gap"]["xy_rmse_m"] == pytest.approx(.1)
    assert core["isaac"]["ground_contact_lateral_slip_m_s"] == {"left": pytest.approx(.4), "right": None}
    assert core["isaac"]["heading_offset_is_mismatch"] is False
    for key in list(isaac):
        if isinstance(isaac[key], np.ndarray) and isaac[key].shape[0] == 1001: isaac[key] = isaac[key][:30]
    reconstructions = {name: geometry(trace) for name, trace in traces.items()}
    censored = pair.summarize_pair(get_reference(), traces, reconstructions)["core"]
    assert censored["isaac"]["coverage_complete"] is False
    assert censored["isaac"]["reference_xy_rmse_m"] is None
    assert censored["sim2sim_xy_gap"]["xy_rmse_m"] is None


def test_supplement_has_separate_core_and_descriptive_events_never_primary_clock():
    trace = canonical_data(1001); trace["time_s"] = np.arange(1001)*.005
    traces = {key: copy.deepcopy(trace) for key in ("reference", "isaac", "mujoco")}
    result = pair.summarize_pair(get_reference("doa_sa29_scale2"), traces, {name: geometry(trace) for name, trace in traces.items()})
    assert result["core"]["isaac"]["window_s"] == [0., 3.14]
    assert result["supplementary_event_1"]["isaac"]["window_s"] == [.78, 1.54]
    assert result["supplementary_event_2"]["window_role"] == "supplementary_descriptive_event"


def test_named_motion_uses_original_base_com_velocity_and_eight_joint_velocities():
    n = 3
    motion = {"time_s": np.arange(n)/50, "fps": np.array([50.]),
              "body_names": np.array(["wheel_R_Link", "base_Link", "wheel_L_Link"]),
              "joint_names": np.array(pair.JOINT_NAMES)[::-1],
              "body_pos_w": np.arange(n*9).reshape(n, 3, 3),
              "body_quat_w": np.tile([1., 0., 0., 0.], (n, 3, 1)),
              "body_lin_vel_w": np.arange(n*9).reshape(n, 3, 3)+100,
              "body_ang_vel_w": np.arange(n*9).reshape(n, 3, 3)+200,
              "joint_pos": np.arange(n*8).reshape(n, 8), "joint_vel": np.arange(n*8).reshape(n, 8)+300}
    result = pair.reference_trace(motion)
    np.testing.assert_array_equal(result["com_velocity"], motion["body_lin_vel_w"][:, 1])
    np.testing.assert_array_equal(result["wheel_pos"], motion["body_pos_w"][:, [2, 0]])
    np.testing.assert_array_equal(result["joint_vel"], motion["joint_vel"][:, ::-1])
    assert "contact_count" not in result


def test_old_scalar_nmpc_trace_missing_joint_velocity_and_contacts_is_rejected():
    data = encoded_trace("nmpc"); del data["joint_vel"]
    with pytest.raises(KeyError, match="joint_vel"):
        native_trace(data, "nmpc")


def test_no_overwrite_is_checked_before_loading_inputs(tmp_path):
    target = tmp_path / "existing"; target.mkdir(); marker = target / "keep"; marker.write_bytes(b"user data")
    with pytest.raises(ValueError, match="Preserving"):
        pair.run(SimpleNamespace(output_dir=target))
    assert marker.read_bytes() == b"user data"


def test_fixture_end_to_end_no_step_png_compact_replay_and_sanitized_summary(tmp_path, monkeypatch):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("matplotlib")
    model = synthetic_tron1_kinematic_model()
    trace, _, _, _ = measured_kinematic_trace(model)
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
        monkeypatch.setattr(mj, name, lambda *args, **kwargs: pytest.fail("offline analyzer stepped dynamics"))
    motion = {"time_s": trace["time_s"], "fps": np.array([200.]), "joint_names": np.array(pair.JOINT_NAMES),
              "body_names": np.array(["base_Link", "wheel_L_Link", "wheel_R_Link"]),
              "body_pos_w": np.concatenate((trace["position"][:, None], trace["wheel_pos"]), axis=1),
              "body_quat_w": np.repeat(trace["quaternion"][:, None], 3, axis=1),
              "body_lin_vel_w": np.repeat(trace["com_velocity"][:, None], 3, axis=1),
              "body_ang_vel_w": np.repeat(trace["angular_velocity"][:, None], 3, axis=1),
              "joint_pos": trace["joint_pos"], "joint_vel": trace["joint_vel"]}
    files = {key: tmp_path / (key + (".npz" if "trace" in key or key in ("motion_file", "reference_file") else ".json"))
             for key in ("motion_file", "reference_file", "isaac_trace", "mujoco_trace", "isaac_report", "mujoco_report", "model")}
    np.savez(files["motion_file"], **motion)
    np.savez(files["reference_file"], time_s=trace["time_s"], root_target_position_m=trace["position"])
    np.savez(files["isaac_trace"], **encoded_trace("policy_isaac", trace))
    np.savez(files["mujoco_trace"], **encoded_trace("policy_mujoco", trace))
    files["model"].write_bytes(b"fake model provenance; injected synthetic model")
    ia, mr = policy_reports()
    spec = get_reference(); spec.update(motion_npz_sha256=pair.sha256(files["motion_file"]),
                                       evaluation_core_interval_s=[0., .015], source_demo_interval_s=[0., .015])
    ia.update(physics_trajectory_sha256=pair.sha256(files["isaac_trace"]), motion_file_sha256=spec["motion_npz_sha256"])
    mr.update(rollout_sha256=pair.sha256(files["mujoco_trace"]), motion_sha256=spec["motion_npz_sha256"], model_sha256=pair.sha256(files["model"]))
    files["isaac_report"].write_text(json.dumps(ia)); files["mujoco_report"].write_text(json.dumps(mr))
    monkeypatch.setattr(pair, "verify_reference_pair", lambda *args: copy.deepcopy(spec))
    @contextmanager
    def fake_source(*args):
        yield model, {"source_helpers_sha256": {"fake.py": "4"*64}, "dynamics_stepping": False}
    monkeypatch.setattr(pair, "source_model", fake_source)
    args = SimpleNamespace(**files, source_root=tmp_path, algorithm="PPO", reference_id=spec["reference_id"],
                           output_dir=tmp_path / "analysis", integration_only=True,
                           training_manifest=None, checkpoint=None, mujoco_trace_sha256=None)
    result = pair.run(args)
    assert result["core_coverage_complete"] is True
    assert result["eligible_for_controller_ranking"] is False
    assert result["controlled_execution_success_claimed"] is False
    assert result["windows"]["core"]["sim2sim_xy_gap"]["xy_peak_m"] == pytest.approx(0.)
    assert result["fk_audit"]["isaac"]["dynamics_stepping"] is False
    assert result["reference_course_point_audit"]["endpoint_frames"] == [0, 3]
    assert str(tmp_path) not in (args.output_dir / "summary.json").read_text()
    compact = pair.load_arrays(args.output_dir / "pair_trace.npz")
    assert not any(value.dtype.hasobject for value in compact.values())
    assert "reference_contact_count" not in compact
    assert "isaac_contact_slip" in compact and "mujoco_joint_vel" in compact
    assert pair.sha256(args.output_dir / "pair_trace.npz") == result["artifacts"]["compact_trace_sha256"]
    pair.render_topdown(compact, result, tmp_path / "replay.png")
    assert (tmp_path / "replay.png").stat().st_size > 1000


def test_reference_course_point_transport_identity_is_explicit_and_leaves_catalog_unchanged():
    n = 21; time = np.arange(n)*.02
    # Linear root-origin trajectory and a rotating original COM offset.
    origin = np.column_stack((time, .2*time, np.ones(n)))
    angle = 2*time
    quaternion = np.column_stack((np.cos(angle/2), np.zeros((n, 2)), np.sin(angle/2)))
    omega = np.tile([0., 0., 2.], (n, 1))
    world_offset = pair.quaternion_rotate(quaternion, pair.WheelKinematics.ROOT_COM)
    com_velocity = np.tile([1., .2, 0.], (n, 1)) + np.cross(omega, world_offset)
    motion = {"time_s": time, "fps": np.array([50.]), "joint_names": np.array(pair.JOINT_NAMES),
              "body_names": np.array(["base_Link", "wheel_L_Link", "wheel_R_Link"]),
              "body_pos_w": np.repeat(origin[:, None], 3, axis=1),
              "body_quat_w": np.repeat(quaternion[:, None], 3, axis=1),
              "body_lin_vel_w": np.repeat(com_velocity[:, None], 3, axis=1),
              "body_ang_vel_w": np.repeat(omega[:, None], 3, axis=1),
              "joint_pos": np.zeros((n, 8)), "joint_vel": np.zeros((n, 8))}
    spec = {"evaluation_core_interval_s": [0., .4], "core_course_turn_deg": 123.}
    before = copy.deepcopy(spec)
    result = pair.reference_course_point_audit(motion, {"time_s": time, "root_target_position_m": origin}, spec)
    assert result["transported_origin_vs_position_gradient_max_abs_component_m_s"] < 1e-13
    assert result["root_origin_gradient_course"]["net_course_change_deg"] == pytest.approx(0., abs=1e-12)
    assert abs(result["metric_com_course"]["net_course_change_deg"]) > .1
    assert result["catalog_core_course_turn_deg_unchanged"] == 123.
    assert result["endpoint_frames"] == [0, 20]
    assert spec == before


def test_topdown_shared_equal_aspect_bounds_include_wheels_outside_root_extents(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    trace = canonical_data()
    trace["position"][:, :2] = 0.
    trace["wheel_pos"][:] = [[-5., -4., .1], [8., 7., .1]]
    traces = {name: copy.deepcopy(trace) for name in ("reference", "isaac", "mujoco")}
    compact = pair.compact_arrays(traces, {name: geometry(value) for name, value in traces.items()})
    summary = {"algorithm": "PPO", "reference_id": "fixture", "fixed_mu": .7,
               "integration_only": True,
               "reference_spec": {"evaluation_core_interval_s": [0., .02], "source_demo_interval_s": [0., .02]},
               "windows": {"core": {name: {"coverage_complete": True} for name in traces}}}
    bounds = []
    original_close = plt.close
    def record_then_close(figure=None):
        if hasattr(figure, "axes"):
            bounds.extend((axis.get_xlim(), axis.get_ylim(), axis.get_aspect()) for axis in figure.axes)
        original_close(figure)
    monkeypatch.setattr(plt, "close", record_then_close)
    pair.render_topdown(compact, summary, tmp_path / "wheel_extents.png")
    assert len(bounds) == 3
    for xlim, ylim, aspect in bounds:
        assert xlim[0] < -5. and xlim[1] > 8.
        assert ylim[0] < -4. and ylim[1] > 7.
        assert aspect == 1.
    assert bounds[0] == bounds[1] == bounds[2]
