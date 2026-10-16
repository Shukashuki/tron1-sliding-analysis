"""Asset-free local-file audit fixtures; no simulator imports or physics."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nmpc_validation_contract as contract
from controller_references import get_reference
from nmpc_tracking_objective import HeightSmoothTrackingObjective, BODY_NAMES, JOINT_NAMES


def settings(tmp_path):
    source = ROOT / "config/nmpc_height_smooth_residual.json"
    path = tmp_path / "settings.json"
    path.write_bytes(source.read_bytes())
    return path, contract.settings_contract(path)


def fixture(tmp_path, reference_id="drift_drl_traj4", *, native=False):
    spec = get_reference(reference_id)
    path, config = settings(tmp_path)
    binding = {name: str(k) * 64 for k, name in enumerate(
        ("model", "asset", "contract", "task_profile", "reference_catalog"), start=1)}
    binding.update(motion=spec["motion_npz_sha256"], reference=spec["robot_reference_npz_sha256"],
                   settings=contract.file_sha(path))
    objective = HeightSmoothTrackingObjective(json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())).contract()
    count = spec["frames"]; samples = 4 * count + 1
    raw = {"time_s": np.arange(samples) * .005, "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
           "root_pos": np.zeros((samples, 3)), "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (samples, 1)),
           "base_com_velocity_w": np.zeros((samples, 3)), "base_angular_velocity_w": np.zeros((samples, 3)),
           "joint_pos": np.zeros((samples, 8)), "joint_vel": np.zeros((samples, 8)), "wheel_pos": np.zeros((samples, 2, 3)),
           "action_joint_names": np.array(JOINT_NAMES), "body_names": np.array(BODY_NAMES),
           "actions": np.zeros((count, 8)), "held_action": np.zeros((samples, 8)),
           "reference_frame_for_action": np.r_[0, np.repeat(np.arange(count), 4)],
           "action_was_applied": np.r_[False, np.ones(samples - 1, dtype=bool)],
           "compute_s": np.full(count, .4), "wheel_ground_contact_count": np.zeros((samples, 2)),
           "wheel_ground_contact_force_w_n": np.zeros((samples, 2, 3)), "wheel_lateral_contact_slip_m_s": np.zeros((samples, 2)),
           "nonwheel_ground_contact_count": np.zeros(samples), "nonwheel_ground_contact_force_norm_n": np.zeros(samples)}
    log = [{"frame": k, "reference_frame_for_reward": k, "action": [0.] * 8, "compute_s": .4,
            "solver": {"optimization_failed": False, "fallback_used": False}} for k in range(count)]
    nominal = {}
    for engine in ("isaac", "mujoco"):
        directory = tmp_path / engine; directory.mkdir()
        report = {"engine": contract.ENGINES[engine], "status": "completed", "truncated_smoke": False,
                  "kind": "receding_horizon_nmpc" if engine == "isaac" else "full_body_height_smooth_cem_nmpc",
                  "reference_id": spec["reference_id"], "role": spec["role"], "stratum": spec["stratum"], "reference_spec": spec,
                  "motion_sha256": spec["motion_npz_sha256"], "effective_cem_config": config["effective_cem_config"],
                  "prediction_mu_scenarios": config["prediction_mu_scenarios"], "plant_mu": .7,
                  "source_sha256": {"/private/nominal/" + role: binding[role] for role in ("contract", "settings", "task_profile")},
                  "physics_dt_s": .005, "control_dt_s": .02, "objective_contract": objective,
                  "action_parameterization": {"mode": "warm_start_residual_blocks", "baseline_copied_and_fixed_per_solve": True},
                  "packed_state_contract": {"dimension": 111, "qpos_dimension": 15, "qvel_dimension": 14,
                                            "contact_force_history_shape": [3, 9, 3]},
                  "completed_policy_steps": count, "evaluated_policy_steps": count, "total_reference_policy_steps": count,
                  "duration_s": count * .02, "solver_failures": 0, "prediction_backend": "native" if native else "python",
                  "contact_slip_definition": "Measured ground contact-point lateral RMS, not kinematic heading",
                  **{name: False for name in ("learned_policy_used", "tvlqr_used", "root_state_prescribed_after_reset",
                       "external_root_forces_used", "prediction_receives_plant_mu", "pose_height_failure_gate")}}
        if engine == "isaac":
            report.update(mujoco_model_sha256=binding["model"], asset_sha256=binding["asset"],
                          fixed_evaluation={"all_termination_terms_removed": True},
                          material_audit={"expected_mu": .7, "robot_static_dynamic_range": [.7, .7],
                                          "ground_bound_materials": [{"coefficients": [.7, .7, 0.]}]})
        else:
            report.update(model_sha256=binding["model"], plant_material_readback={"compiled_sliding_friction_range": [.7, .7]})
        # Different per-engine Linux builds are valid; backend semantics stay fixed.
        if native:
            digest = "a" * 64 if engine == "isaac" else "b" * 64
            metadata = {"workers": {"0.7": {"native_library_sha256": digest}}}
            report["prediction_backend_metadata" if engine == "isaac" else "native_prediction_backend"] = metadata
            binding[engine + "_native_library"] = digest
            binding[engine + "_native_build_manifest"] = "c" * 64 if engine == "isaac" else "d" * 64
            if engine == "isaac":
                report.update(native_library_sha256=digest, native_build_manifest_sha256=binding["isaac_native_build_manifest"])
        write_episode(directory, report, raw, log)
        nominal[engine] = contract.episode_evidence(directory, engine, spec, config, .7)
    summary = {"kind": "offline_single_controller_fixed_mu_engine_pair", "algorithm": "NMPC", "reference_spec": spec,
               "reference_id": spec["reference_id"], "fixed_mu": .7, "integration_only": False,
               "core_coverage_complete": True, "eligible_for_controller_ranking": True, "pooled_success_statistics": False,
               "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
               "input_sha256": {"motion_file": binding["motion"], "reference_file": binding["reference"], "model": binding["model"]},
               "windows": {"core": {engine: {"window_s": spec["evaluation_core_interval_s"], "coverage_complete": True,
                    "large_xy_exploratory_screen": False, "reference_xy_rmse_m": .1, "reference_xy_peak_m": .2,
                    "mean_speed_retention_ratio": 0., "net_course_change_deg": None,
                    "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False} for engine in nominal}},
               "posture_diagnostics": {engine: {"max_base_tilt_deg": 179.} for engine in nominal}}
    for engine, item in nominal.items():
        summary["input_sha256"].update({engine + "_trace": item["hashes"]["trace"], engine + "_report": item["hashes"]["report"]})
    summary["audit"] = contract.audit_reports("NMPC", spec, nominal["isaac"]["report"], nominal["mujoco"]["report"],
        isaac_trace_sha256=nominal["isaac"]["hashes"]["trace"], mujoco_trace_sha256=nominal["mujoco"]["hashes"]["trace"],
        model_sha256=binding["model"])
    summary["analysis_source_sha256"] = {"analyze_controller_pair.py": contract.file_sha(ROOT / "scripts/analyze_controller_pair.py")}
    return dict(reference_spec=spec, settings_contract=config, bindings=binding, nominal=nominal, pair_summary=summary)


def write_episode(directory, report, raw, log):
    np.savez_compressed(directory / "rollout.npz", **raw)
    (directory / "solver_log.json").write_text(json.dumps(log))
    report = copy.deepcopy(report)
    report.update(rollout_sha256=contract.file_sha(directory / "rollout.npz"),
                  solver_log_sha256=contract.file_sha(directory / "solver_log.json"))
    (directory / "report.json").write_text(json.dumps(report))


def alter_episode(args, engine="mujoco", change=None):
    evidence = args["nominal"][engine]; directory = Path(evidence["_directory"])
    report = json.loads((directory / "report.json").read_text())
    with np.load(directory / "rollout.npz", allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    log = json.loads((directory / "solver_log.json").read_text())
    change(report, raw, log)
    write_episode(directory, report, raw, log)
    return directory


def refresh(args):
    for engine, old in args["nominal"].items():
        item = contract.episode_evidence(old["_directory"], engine, args["reference_spec"], args["settings_contract"], .7)
        args["nominal"][engine] = item
        args["pair_summary"]["input_sha256"].update({engine + "_trace": item["hashes"]["trace"], engine + "_report": item["hashes"]["report"]})


def test_input_bindings_hash_files_not_paths_and_preserve_missing_inputs(tmp_path):
    path = tmp_path / "token"; path.write_text("local fixture")
    result = contract.input_bindings({"model": path, "source:scripts/runner.py": path})
    assert set(result) == {"model", "source:scripts/runner.py"} and len(set(result.values())) == 1
    with pytest.raises(FileNotFoundError): contract.file_sha(tmp_path / "missing")
    with pytest.raises(ValueError): contract.input_bindings({"/private/role": path})
    with pytest.raises(ValueError): contract.input_bindings({})


@pytest.mark.parametrize("field,value", [("physics_dt_s", .01), ("prediction_mu_scenarios", [.7, .7]),
    ("prediction_mu_scenarios", [True]), ("prediction_mu_scenarios", [float("nan")]),
    ("nmpc.seed", True), ("nmpc.initial_std", [0.] * 8), ("nmpc.parameterization", "unknown")])
def test_invalid_settings_rejected_without_simulation(tmp_path, field, value):
    path, _ = settings(tmp_path); data = json.loads(path.read_text())
    target = data
    if "." in field: parent, field = field.split("."); target = target[parent]
    target[field] = value; path.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)): contract.settings_contract(path)


@pytest.mark.parametrize("reference_id,samples", [("drift_drl_traj4", 1397), ("doa_sa29_scale2", 1033)])
def test_complete_native_evidence_uses_selected_reference_frames(tmp_path, reference_id, samples):
    args = fixture(tmp_path, reference_id)
    assert args["nominal"]["mujoco"]["physics_samples"] == samples
    frozen = contract.make_freeze(**args)
    contract.validate_freeze(frozen, args["reference_spec"], args["settings_contract"], args["bindings"])
    assert not frozen["height_flight_tilt_beta_or_numeric_speed_gate"]
    assert not frozen["pooled_success_statistics"]
    # Low speed, no measured contact and a large posture diagnostic are not gates.
    assert frozen["nominal_core_diagnostics"]["isaac"]["mean_speed_retention_ratio"] == 0.
    assert str(tmp_path) not in json.dumps(frozen)


@pytest.mark.parametrize("problem", ["status", "short", "clock", "phase", "mask", "held", "bound", "precision",
    "nan_force", "negative_contact", "fractional_contact", "negative_slip", "nonwheel", "log_action", "log_phase",
    "log_time", "failure_count", "oracle", "settings", "height", "material", "role", "kind", "packed", "extra_nan", "json_nan"])
def test_incomplete_corrupt_nonnominal_or_changed_episode_rejected(tmp_path, problem):
    args = fixture(tmp_path)
    def change(report, raw, log):
        if problem == "status": report["status"] = "smoke_completed"
        elif problem == "short": report["completed_policy_steps"] -= 1
        elif problem == "clock": raw["time_s"][100:] += .005
        elif problem == "phase": raw["reference_frame_for_action"][100] += 1
        elif problem == "mask": raw["action_was_applied"] = raw["action_was_applied"].astype(int)
        elif problem == "held": raw["held_action"][1, 0] = .5
        elif problem == "bound": raw["actions"][0, 0] = 1.1
        elif problem == "precision": raw["actions"][0, 0] = .1
        elif problem == "nan_force": raw["wheel_ground_contact_force_w_n"][0, 0, 0] = np.nan
        elif problem == "negative_contact": raw["wheel_ground_contact_count"][0, 0] = -1
        elif problem == "fractional_contact": raw["wheel_ground_contact_count"][0, 0] = .5
        elif problem == "negative_slip": raw["wheel_lateral_contact_slip_m_s"][0, 0] = -.5
        elif problem == "nonwheel": raw["nonwheel_ground_contact_force_norm_n"][0] = -1
        elif problem == "log_action": log[0]["action"][0] = .5
        elif problem == "log_phase": log[0]["reference_frame_for_reward"] = 1
        elif problem == "log_time": log[0]["compute_s"] = 12.
        elif problem == "failure_count": report["solver_failures"] = 1
        elif problem == "oracle": report["prediction_receives_plant_mu"] = True
        elif problem == "settings": report["effective_cem_config"]["parameterization"] = "absolute_blocks"
        elif problem == "height": report["objective_contract"]["absolute_height_target"] = True
        elif problem == "material": report["plant_material_readback"]["compiled_sliding_friction_range"] = [.3, .3]
        elif problem == "kind": report["kind"] = "policy_replay"
        elif problem == "packed": report["packed_state_contract"]["dimension"] = 29
        elif problem == "extra_nan": raw["qacc_unchecked"] = np.full(len(raw["time_s"]), np.nan)
        elif problem == "json_nan": log[0]["solver"]["selected_model_cost"] = float("nan")
        else: report["role"] = "primary_pooled_with_supplement"
    directory = alter_episode(args, change=change)
    with pytest.raises(ValueError):
        contract.episode_evidence(directory, "mujoco", args["reference_spec"], args["settings_contract"], .7)


def test_hash_mismatch_not_repaired_and_finite_solver_fallback_does_not_reject(tmp_path):
    args = fixture(tmp_path)
    def fail(report, raw, log):
        log[0]["solver"].update(optimization_failed=True, fallback_used=True)
        report["solver_failures"] = 1
        raw["root_pos"][:, 2] = 50.  # No absolute-height restriction.
    directory = alter_episode(args, change=fail)
    result = contract.episode_evidence(directory, "mujoco", args["reference_spec"], args["settings_contract"], .7)
    assert result["solver_diagnostics"] == {"optimization_failed": 1, "fallback_used": 1}
    report = directory / "report.json"; data = json.loads(report.read_text()); data["rollout_sha256"] = "f" * 64
    report.write_text(json.dumps(data)); before = report.read_bytes()
    with pytest.raises(ValueError, match="SHA-256"):
        contract.episode_evidence(directory, "mujoco", args["reference_spec"], args["settings_contract"], .7)
    assert report.read_bytes() == before


def test_distinct_per_engine_native_builds_are_allowed_and_bound(tmp_path):
    args = fixture(tmp_path, native=True)
    frozen = contract.make_freeze(**args)
    assert frozen["nominal"]["isaac"]["native_identity"]["native_library_sha256"] != frozen["nominal"]["mujoco"]["native_identity"]["native_library_sha256"]
    args["bindings"]["mujoco_native_library"] = "e" * 64
    with pytest.raises(ValueError, match="native binding"):
        contract.make_freeze(**args)


@pytest.mark.parametrize("problem", ["one_engine", "xy_screen", "contradictory_xy", "censored", "summary_hash",
    "wrong_reference", "height_gate", "stale_capture", "stale_model", "stale_settings", "integration", "profile", "audit", "analyzer"])
def test_freeze_requires_actual_matching_nominal_two_engine_xy_qualification(tmp_path, problem):
    args = fixture(tmp_path)
    if problem == "one_engine": del args["nominal"]["isaac"]
    elif problem == "xy_screen": args["pair_summary"]["windows"]["core"]["isaac"]["large_xy_exploratory_screen"] = True
    elif problem == "contradictory_xy": args["pair_summary"]["windows"]["core"]["isaac"]["reference_xy_rmse_m"] = .16
    elif problem == "censored": args["pair_summary"]["windows"]["core"]["mujoco"]["coverage_complete"] = False
    elif problem == "summary_hash": args["pair_summary"]["input_sha256"]["isaac_report"] = "f" * 64
    elif problem == "wrong_reference": args["pair_summary"]["reference_id"] = "doa_sa29_scale2"
    elif problem == "height_gate": args["pair_summary"]["height_or_flight_failure_gate"] = True
    elif problem == "stale_capture": args["nominal"]["mujoco"]["hashes"]["report"] = "f" * 64
    elif problem == "stale_model": args["bindings"]["model"] = "e" * 64
    elif problem == "stale_settings": args["settings_contract"]["effective_cem_config"]["parameterization"] = "absolute_blocks"
    elif problem == "profile": args["bindings"]["task_profile"] = "f" * 64
    elif problem == "audit": args["pair_summary"]["audit"]["fixed_mu"] = .8
    elif problem == "analyzer": args["pair_summary"]["analysis_source_sha256"] = {}
    else: args["pair_summary"]["integration_only"] = True
    with pytest.raises(ValueError): contract.make_freeze(**args)


@pytest.mark.parametrize("change", ["status", "settings", "bindings", "nominal", "screen", "missing_hash", "audit", "pair_hash", "contradictory_xy"])
def test_validate_freeze_rejects_stale_or_incomplete_document(tmp_path, change):
    args = fixture(tmp_path); frozen = contract.make_freeze(**args)
    if change == "status": frozen["status"] = "development_analysis_only"
    elif change == "settings": frozen["settings_contract"]["seed"] = 5
    elif change == "bindings": frozen["bindings"]["contract"] = "f" * 64
    elif change == "nominal": del frozen["nominal"]["isaac"]
    elif change == "screen": frozen["nominal_core_diagnostics"]["isaac"]["large_xy_exploratory_screen"] = True
    elif change == "audit": frozen["nominal_pair_audit"]["fixed_mu"] = .3
    elif change == "pair_hash": frozen["nominal_pair_input_sha256"]["isaac_report"] = "f" * 64
    elif change == "contradictory_xy": frozen["nominal_core_diagnostics"]["isaac"]["reference_xy_peak_m"] = .31
    else: del frozen["nominal"]["isaac"]["hashes"]["trace"]
    with pytest.raises(ValueError): contract.validate_freeze(frozen, args["reference_spec"], args["settings_contract"], args["bindings"])


def test_pure_contract_import_does_not_import_simulators_torch_or_start_processes():
    code = "import sys;sys.path.insert(0,'scripts');import nmpc_validation_contract;assert not {'mujoco','torch','isaaclab','omni'} & set(sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_per_case_freeze_guard_allows_outcomes_but_rejects_model_and_objective_drift(tmp_path):
    args = fixture(tmp_path, native=True)
    frozen = contract.make_freeze(**args)
    evidence = copy.deepcopy(args["nominal"]["mujoco"])
    evidence["plant_mu"] = .01
    evidence["solver_diagnostics"]["fallback_used"] = 349
    contract.validate_case_against_nominal(evidence, frozen["nominal"]["mujoco"])
    for key in ("objective_contract", "packed_state_contract"):
        changed = copy.deepcopy(evidence)
        changed["controller_contract"][key] = {"changed": True}
        with pytest.raises(ValueError, match="controller_contract"):
            contract.validate_case_against_nominal(changed, frozen["nominal"]["mujoco"])
    evidence["controller_contract"]["backend"]["workers"]["0.7"]["model_snapshot_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="controller_contract"):
        contract.validate_case_against_nominal(evidence, frozen["nominal"]["mujoco"])


def test_reported_source_multiset_is_path_independent_but_detects_changed_installed_helpers(tmp_path):
    args = fixture(tmp_path)
    original = args["nominal"]["isaac"]
    sources = original["report"]["source_sha256"]
    assert original["controller_contract"]["reported_source_sha256"] == sorted(sources.values())
    def rehost(report, raw, log):
        report["source_sha256"] = {"/other-host/source-" + str(i): value
                                   for i, value in enumerate(report["source_sha256"].values())}
    directory = alter_episode(args, "isaac", rehost)
    moved = contract.episode_evidence(directory, "isaac", args["reference_spec"], args["settings_contract"], .7)
    contract.validate_case_against_nominal(moved, original)
    def installed_change(report, raw, log):
        report["source_sha256"]["/installed/whole_body_tracking/rewards.py"] = "e" * 64
    alter_episode(args, "isaac", installed_change)
    changed = contract.episode_evidence(directory, "isaac", args["reference_spec"], args["settings_contract"], .7)
    with pytest.raises(ValueError, match="controller_contract"):
        contract.validate_case_against_nominal(changed, original)
    # Repeated hashes remain repeated, rather than disappearing in a set.
    report = copy.deepcopy(original["report"])
    digest = next(iter(sources.values()))
    report["source_sha256"]["/duplicate/identical.py"] = digest
    assert contract._controller_contract(report, "isaac")["reported_source_sha256"].count(digest) == 2
    report["source_sha256"]["/bad/hash.py"] = "not-a-sha"
    with pytest.raises(ValueError, match="SHA-256"):
        contract._controller_contract(report, "isaac")


@pytest.mark.parametrize("role,engine", [("source:scripts/nmpc_shooting.py", "mujoco"),
    ("source:scripts/run_nmpc_tracking.py", "mujoco"), ("source:scripts/run_nmpc_tracking_isaac.py", "isaac"),
    ("source_root:training/tron1_tracking.py", "isaac"), ("mujoco_source:scripts/eval_tracking_mujoco.py", "mujoco"),
    ("source:native/nmpc_batch.cpp", "mujoco")])
def test_tracked_current_code_must_bind_to_corresponding_actual_nominal_sources(tmp_path, role, engine):
    args = fixture(tmp_path, native=True); digest = "e" * 64
    args["bindings"][role] = digest
    with pytest.raises(ValueError, match="source binding"):
        contract.make_freeze(**args)
    relative = role.partition(":")[2]
    # Common source files must match BOTH reports; root-specific helpers only their engine.
    engines = ("isaac", "mujoco") if role.startswith("source:") and "run_nmpc_tracking" not in relative else (engine,)
    for target in engines:
        alter_episode(args, target, lambda report, raw, log: report["source_sha256"].update({"/private/frozen/" + relative: digest}))
    refresh(args)
    contract.make_freeze(**args)
    alter_episode(args, engine, lambda report, raw, log: report["source_sha256"].update({"/private/frozen/" + relative: "f" * 64}))
    refresh(args)
    with pytest.raises(ValueError, match="source binding"):
        contract.make_freeze(**args)
