"""Audit and replay the nominal CEM predictor-by-plant physics comparison.

The four measured episodes are MuJoCo CPU/PhysX prediction crossed with
MuJoCo/PhysX validation plants. No dynamics, policy, optimizer or retargeting
is executed here. A missing episode is an error, never a synthetic result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import analyze_controller_pair as pair_analysis
from controller_metrics import heading_signals, summarize_turn
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
OLD_PAIR = ROOT / "results/controller_primary_nmpc_residual_h30_b10_nominal_pair_v1"
RAW_ROOT = ROOT / "outputs/physx_mpc_transfer_20261021_v1"
PUBLIC_INPUTS = ROOT / "results/mpc_prediction_transfer_mu0p7_20261021_v1_inputs"
DEFAULT_PROBE = PUBLIC_INPUTS / "probe_h30"
OLD_RAW = {
    "isaac": ROOT / "outputs/nmpc_isaac_primary_nominal_residual_h30_b10_20261016_v1/report.json",
    "mujoco": ROOT / "outputs/nmpc_primary_nominal_residual_h30_b10_20261016_v1/report.json",
}
PREDICTORS = {"mujoco": "MuJoCo CPU predictor", "physx": "PhysX predictor"}
PLANTS = {"isaac": "PhysX", "mujoco": "MuJoCo"}
TRACE_FIELDS = ("time_s", "physics_dt_s", "position", "quaternion", "com_velocity",
                "angular_velocity", "wheel_pos", "contact_count", "contact_force", "contact_slip")
REQUIRED_RUNTIME_SOURCES = (
    "physx_batch_predictor_worker.py", "physx_batch_predictor.py", "nmpc_batch_objective.py",
    "nmpc_tracking_objective.py", "run_nmpc_tracking_isaac.py", "car_drift_height_smooth_task.json",
    "tron1_tracking.py", "tron1_drift.py", "tron1_runtime_device.py", "tron1_domain_randomization.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(values, expected, label):
    array = np.asarray(values, dtype=float)
    require(array.size > 0 and np.isfinite(array).all()
            and np.allclose(array, expected, atol=1e-6, rtol=0), label + ": runtime readback differs")


def named_hashes(values, include=None):
    require(isinstance(values, dict) and bool(values), "Missing source identity evidence")
    result = {}
    for path, digest in values.items():
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "Malformed source SHA-256")
        name = Path(path).name
        if include is not None and name not in include:
            continue
        require(name not in result or result[name] == digest, "Ambiguous source basename: " + name)
        result[name] = digest
    return result


def audit_device(audit):
    require(isinstance(audit, dict) and audit.get("status") == "verified"
            and audit.get("native_and_tensor_device_consistency_verified") is True,
            "PhysX predictor lacks verified native/tensor CUDA device readbacks")
    requested = audit.get("requested_device", "")
    require(re.fullmatch(r"cuda:(0|[1-9][0-9]*)", requested), "Predictor device must be explicit cuda:N")
    ordinal = int(requested.split(":")[1])
    for key in ("environment_device", "physics_sim_view_frontend_device", "robot_native_transform_tensor_device"):
        require(audit.get(key) == requested, "Predictor CUDA device differs: " + key)
    for key in ("native_physics_backend_device_ordinal", "torch_current_device", "physics_cudaDevice_setting"):
        require(type(audit.get(key)) is int and audit[key] == ordinal, "Predictor native CUDA ordinal differs: " + key)
    require(audit.get("physics_sceneMultiGPUMode_setting") == 0
            and "cuda_visible_devices" in audit and audit["cuda_visible_devices"] is None,
            "Predictor requires the audited unmasked single-device runtime")
    return audit


def audit_worker(report, expected):
    require(report.get("prediction_engine") == "Isaac/PhysX" and report.get("prediction_backend") == "physx",
            "Both new episodes must use actual PhysX prediction")
    metadata = report.get("physx_prediction_backend") or report.get("prediction_backend_metadata")
    require(isinstance(metadata, dict) and metadata.get("kind") == "isolated_isaac_physx_batch_tracking_predictor"
            and metadata.get("prediction_engine") == "Isaac/PhysX"
            and metadata.get("population_batching") is True
            and metadata.get("production_cem_wired") is True
            and metadata.get("learned_initialization") is False,
            "Missing production-wired, nonlearned PhysX candidate worker evidence")
    require(metadata.get("runtime_cem_population_path") == "predictor.rollout_batch"
            and type(metadata.get("completed_rollout_batches")) is int
            and metadata["completed_rollout_batches"] > 0,
            "Missing actual completed PhysX CEM candidate-batch evidence")
    reset_audit = metadata.get("last_reset_audit", {})
    require(isinstance(reset_audit, dict)
            and type(reset_audit.get("physics_steps_during_reset")) is int
            and reset_audit["physics_steps_during_reset"] == 0,
            "PhysX candidate reconstruction must advance zero physical steps")
    require(metadata.get("objective") == expected["objective_contract"]
            and metadata.get("prediction_mu_scenarios") == [.7]
            and metadata.get("motion_sha256") == expected["motion_sha256"],
            "PhysX adapter reference/objective/scenario differs")
    semantics = metadata.get("integration_state_semantics", "")
    require("continuously" in semantics and "horizon" in semantics
            and "bias-force" in metadata.get("nominal_seed_source", ""),
            "Missing continuous-horizon/unchanged nominal seed disclosure")
    runtime = metadata.get("worker_runtime", {})
    require(runtime.get("engine") == "Isaac/PhysX"
            and runtime.get("num_candidate_environments") == 64
            and runtime.get("coordinate_identical_candidates") is True
            and runtime.get("cross_environment_collision_filtering") is True
            and runtime.get("environment_spacing_m") == 0.,
            "Worker is not the audited isolated, coordinate-identical 64-candidate PhysX runtime")
    require(runtime.get("objective_contract") == expected["objective_contract"]
            and runtime.get("motion_sha256") == expected["motion_sha256"]
            and runtime.get("body_order") == list(BODY_NAMES)
            and runtime.get("action_order") == list(JOINT_NAMES), "Worker objective/reference/state order differs")
    for source in (metadata, runtime):
        equal_number(source.get("physics_dt_s", []), .005, "Worker physics dt")
        equal_number(source.get("control_dt_s", []), .02, "Worker control dt")
    device = audit_device(runtime.get("device_audit"))
    for flag in ("plant_access", "trained_policy_used", "root_forces_used"):
        require(runtime.get(flag) is False, "Worker lacks isolation evidence: " + flag)
    require(runtime.get("fixed_evaluation", {}).get("all_termination_terms_removed") is True
            and runtime.get("legacy_joint_friction_audit", {}).get("all_environments_verified_zero") is True,
            "Worker reset gates or legacy joint friction are not audited")
    material = runtime.get("material_audit", {})
    equal_number(material.get("expected_mu", []), .7, "Worker nominal material")
    equal_number(material.get("robot_static_dynamic_range", []), .7, "Worker robot friction")
    equal_number(material.get("robot_restitution_range", []), 0., "Worker robot restitution")
    require(bool(material.get("ground_bound_materials")), "Worker lacks bound ground materials")
    for ground in material["ground_bound_materials"]:
        equal_number(ground.get("coefficients", [])[:2], .7, "Worker bound ground friction")
        equal_number(ground.get("coefficients", [])[2:], 0., "Worker bound ground restitution")
    actuators = runtime.get("actuator_audit", {})
    for name, kp, kd, torque, velocity in (("legs", 500., 10., 80., 15.), ("wheels", 0., 0., 12., 100.)):
        for key, value in {"stiffness": kp, "damping": kd, "effort_limit_nm": torque,
                           "velocity_limit_rad_s": velocity, "saturation_effort_nm": torque,
                           "velocity_at_effort_limit_rad_s": 2*velocity}.items():
            equal_number(actuators.get(name, {}).get(key, []), value, "Worker actuator " + name + "/" + key)
    sources = named_hashes(runtime.get("source_sha256"))
    require(all(name in sources for name in REQUIRED_RUNTIME_SOURCES), "Incomplete worker runtime source identities")
    require(sources["physx_batch_predictor_worker.py"] == metadata.get("worker_source_sha256")
            and sources["physx_batch_predictor.py"] == metadata.get("adapter_source_sha256"),
            "Worker/adapter source hash binding differs")
    shared_names = ("physx_batch_predictor_worker.py", "physx_batch_predictor.py", "nmpc_batch_objective.py", "nmpc_tracking_objective.py")
    parent_sources = named_hashes(report.get("source_sha256"), include=shared_names)
    for name in shared_names:
        require(parent_sources.get(name) == sources[name], "Parent and worker source identities differ: " + name)
    require(isinstance(runtime.get("asset_sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", runtime["asset_sha256"]), "Worker lacks asset identity")
    return {"engine": runtime["engine"], "candidate_environments": runtime["num_candidate_environments"],
            "source_sha256": sources, "asset_sha256": runtime["asset_sha256"],
            "device_audit": device, "cuda_device_name": runtime.get("cuda_device_name"),
            "material_audit": material, "actuator_audit": actuators,
            "integration_state_semantics": semantics, "nominal_seed_source": metadata["nominal_seed_source"],
            "runtime_cem_population_path": metadata["runtime_cem_population_path"],
            "completed_rollout_batches": metadata["completed_rollout_batches"],
            "last_reset_audit": reset_audit,
            "hidden_solver_state": metadata.get("hidden_solver_state"),
            "scope": "Report/runtime source identity and readback checks; not independent verification of engine internals"}


def archived_prediction_identity(report):
    """Recover the legacy missing label from native MuJoCo worker evidence."""
    declared = report.get("prediction_engine")
    if declared == "MuJoCo":
        return {"engine": "MuJoCo", "identity_source": "declared_prediction_engine"}
    require(declared is None and report.get("prediction_backend") == "native",
            "Archived predictor is not MuJoCo")
    metadata = report.get("native_prediction_backend") or report.get("prediction_backend_metadata") or {}
    require(metadata.get("kind") == "optional_native_batch_exact_objective_adapter"
            and metadata.get("production_cem_wired") is True
            and metadata.get("learned_initialization") is False
            and metadata.get("runtime_cem_population_path") == "callable predictor.rollout_batch",
            "Legacy prediction label requires production native-worker evidence")
    worker = metadata.get("workers", {}).get("0.7", {})
    sources = named_hashes(report.get("source_sha256"))
    require(worker.get("mujoco_version") == "3.8.1"
            and worker.get("source_sha256") == sources.get("nmpc_batch.cpp")
            and "nmpc_native_batch.py" in sources
            and re.fullmatch("[0-9a-f]{64}", worker.get("native_library_sha256", ""))
            and "EACH action interval" in worker.get("reset_semantics", ""),
            "Missing source-bound MuJoCo 3.8.1 native worker identity")
    return {"engine": "MuJoCo", "identity_source": "legacy_native_worker_source_and_build_metadata",
            "worker_source_sha256": worker["source_sha256"],
            "native_library_sha256": worker["native_library_sha256"],
            "original_report_modified": False}


def audit_raw_reports(old_summary, new_summary, old_reports, new_reports):
    reports = [old_reports[plant] for plant in PLANTS] + [new_reports[plant] for plant in PLANTS]
    expected = old_reports["isaac"]
    config = expected.get("effective_cem_config", {})
    require(all(config.get(key) == value for key, value in {
        "horizon_steps": 30, "control_blocks": 10, "candidates": 64,
        "iterations": 3, "elite_count": 8, "seed": 42,
        "parameterization": "warm_start_residual_blocks"}.items()), "Wrong nominal CEM experiment")
    for report in reports:
        for key in ("effective_cem_config", "action_parameterization", "objective_contract", "settings"):
            require(report.get(key) == expected.get(key) and report.get(key) is not None,
                    "Predictor/plant conditions differ: " + key)
        require(report.get("motion_sha256") == expected["motion_sha256"]
                and report.get("reference_id") == "drift_drl_traj4"
                and report.get("plant_mu") == .7
                and report.get("prediction_mu_scenarios") == [.7]
                and report.get("prediction_receives_plant_mu") is False
                and report.get("status") == "completed"
                and report.get("truncated_smoke") is False
                and report.get("completed_policy_steps") == 349
                and report.get("pose_height_failure_gate") is False,
                "Require four complete nominal, nonoracle, height-free traj_4 episodes")
    pair_contract_audits = {}
    for predictor, summary, raw in (("mujoco", old_summary, old_reports), ("physx", new_summary, new_reports)):
        for plant, report in raw.items():
            require(summary["input_sha256"][plant + "_report"] == report["_capture_sha256"],
                    "Raw report does not match the compact pair provenance")
            require(summary["input_sha256"][plant + "_trace"] == report.get("rollout_sha256"),
                    "Raw trajectory binding differs from compact pair")
            if "_rollout_capture_sha256" in report:
                require(report["_rollout_capture_sha256"] == report["rollout_sha256"],
                        "Published raw rollout differs from its report binding")
        pair_contract_audits[predictor] = pair_analysis.audit_reports(
            "NMPC", summary["reference_spec"], raw["isaac"], raw["mujoco"],
            isaac_trace_sha256=summary["input_sha256"]["isaac_trace"],
            mujoco_trace_sha256=summary["input_sha256"]["mujoco_trace"],
            model_sha256=summary["input_sha256"]["model"], integration_only=False)
    old_identities = {plant: archived_prediction_identity(report) for plant, report in old_reports.items()}
    worker_audits = {plant: audit_worker(report, expected) for plant, report in new_reports.items()}
    for key in ("source_sha256", "asset_sha256", "actuator_audit"):
        require(worker_audits["isaac"][key] == worker_audits["mujoco"][key],
                "PhysX predictor implementation/asset differs across plants: " + key)
    return {"effective_cem_config": config, "objective_contract": expected["objective_contract"],
            "action_parameterization": expected["action_parameterization"],
            "four_raw_reports_bound_to_compact_pairs": True,
            "raw_rollout_files_rehashed": sum("_rollout_capture_sha256" in report for report in reports),
            "archived_predictor_identity": old_identities,
            "four_report_objective_and_search_contracts_match_exactly": True,
            "pair_report_contract_audits": pair_contract_audits,
            "physx_predictor_worker_audits": worker_audits}


def read_report(path):
    report = pair_analysis.read_json(path)
    report["_capture_sha256"] = pair_analysis.sha256(path)
    rollout = path.with_name("rollout.npz")
    if rollout.is_file():
        report["_rollout_capture_sha256"] = pair_analysis.sha256(rollout)
    return report


def audit_probe_arrays(arrays, case):
    """Recheck the one saved rollout per case, without replaying any physics."""
    shapes = {"controls": (64, 30, 8), "costs": (64,), "stage_cost": (64, 30),
              "applied_joint_torque_nm": (64, 30, 4, 8),
              "trajectory_joint_pos": (64, 30, 8),
              "trajectory_contact_force_history_w": (64, 30, 3, 9, 3),
              "trajectory_body_net_forces_w": (64, 30, 9, 3)}
    for role in ("trajectory", "pre_trajectory"):
        for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            shapes[role + "_" + field] = (64, 30, 9, 4 if field == "body_quat_w" else 3)
    require(set(arrays) == set(shapes), "Unsupported or incomplete saved probe array schema")
    for name, shape in shapes.items():
        value = arrays[name]
        require(value.shape == shape and value.dtype.kind in "fiu" and np.isfinite(value).all(),
                "Invalid saved probe array: " + name)
    require(np.max(np.abs(arrays["controls"])) <= 1. + 1e-9, "Saved probe controls exceed normalized bounds")
    require(np.array_equal(arrays["costs"], np.asarray(case.get("candidate_costs"), dtype=float)),
            "Saved probe costs differ from JSON report")
    sum_error = float(np.max(np.abs(arrays["costs"] - arrays["stage_cost"].sum(axis=1))))
    require(sum_error <= 1e-12, "Saved probe costs differ from summed stage costs")
    require(np.array_equal(arrays["controls"][0], arrays["controls"][-1])
            and arrays["costs"][0] == arrays["costs"][-1],
            "Saved duplicate probe plans/costs are not identical")
    duplicate_error = float(abs(arrays["costs"][0] - arrays["costs"][-1]))
    require(case.get("duplicate_candidate_cost_abs_error") == duplicate_error,
            "Reported duplicate-candidate cost error differs from stored arrays")
    return {"finite_array_shapes": {name: list(shape) for name, shape in shapes.items()},
            "normalized_controls_bounded": True, "json_costs_match_exactly": True,
            "stage_cost_sum_max_abs_error": sum_error,
            "duplicate_controls_and_costs_match_exactly": True}


def audit_physx_probe(directory, worker_audits, expected):
    """Bind stored calibration evidence to the production worker implementation.

    Only saved single-rollout arrays are independently rechecked. The probe did
    not archive repeated/permuted trajectories or analytic torque/parity outputs,
    so those checks remain explicitly reported evidence rather than a new run.
    """
    report_path = directory / "report.json"
    report = pair_analysis.read_json(report_path)
    require(report.get("kind") == "actual_physx_prediction_probe" and report.get("status") == "passed"
            and report.get("physics_engine") == "Isaac/PhysX"
            and report.get("validation_plant_created") is False
            and report.get("learned_policy_used") is False
            and report.get("nominal_experiment_eligible") is True,
            "Require a passed actual PhysX calibration probe")
    worker = report.get("worker", {})
    runtime = worker.get("worker_runtime", {})
    require(worker.get("kind") == "isolated_isaac_physx_batch_tracking_predictor"
            and worker.get("prediction_engine") == "Isaac/PhysX"
            and runtime.get("engine") == "Isaac/PhysX"
            and worker.get("objective") == expected["objective_contract"]
            and runtime.get("objective_contract") == expected["objective_contract"]
            and worker.get("motion_sha256") == expected["motion_sha256"]
            and runtime.get("motion_sha256") == expected["motion_sha256"]
            and worker.get("prediction_mu_scenarios") == [.7]
            and runtime.get("num_candidate_environments") == 64,
            "Probe engine/reference/objective/scenario differs from production")
    for source in (worker, runtime):
        equal_number(source.get("physics_dt_s", []), .005, "Probe physics dt")
        equal_number(source.get("control_dt_s", []), .02, "Probe control dt")
    audit_device(runtime.get("device_audit"))
    sources = named_hashes(runtime.get("source_sha256"))
    require(sources.get("physx_batch_predictor_worker.py") == worker.get("worker_source_sha256")
            and sources.get("physx_batch_predictor.py") == worker.get("adapter_source_sha256"),
            "Probe worker/adapter hash binding differs")
    require(bool(worker_audits), "Missing production worker identities for probe binding")
    for production in worker_audits.values():
        require(production["source_sha256"] == sources
                and production["asset_sha256"] == runtime.get("asset_sha256"),
                "Calibration probe source/asset identity differs from a production worker")
    cases = report.get("cases", [])
    require([case.get("reference_frame") for case in cases] == [0, 100, 150],
            "Require exactly the three declared calibration frames")
    rechecked, reported = [], []
    hashes = {"report.json": pair_analysis.sha256(report_path)}
    for case in cases:
        frame = case["reference_frame"]
        require(case.get("passed") is True and case.get("horizon_steps") == 30
                and case.get("candidate_count") == 64 and case.get("physical_clock_verified") is True,
                "Incomplete H30/C64 physical calibration case")
        reset = case.get("reset_audit", {})
        require(type(reset.get("physics_steps_during_reset")) is int and reset["physics_steps_during_reset"] == 0,
                "Calibration reset must report zero physical steps")
        for key in ("root_link_position_max_abs_m", "root_quaternion_sign_invariant_max_abs",
                    "root_com_velocity_max_abs_m_s", "joint_position_max_abs_rad", "joint_velocity_max_abs_rad_s"):
            value = reset.get(key)
            require(isinstance(value, (int, float)) and np.isfinite(value) and 0. <= value <= 5e-4,
                    "Calibration reset readback exceeds tolerance: " + key)
        for key, upper in (("scalar_batch_stage_cost_max_abs_error", 1e-10),
                           ("first_substep_torque_max_abs_error_nm", .02)):
            value = case.get(key)
            require(isinstance(value, (int, float)) and np.isfinite(value) and 0. <= value < upper,
                    "Reported calibration check exceeds tolerance: " + key)
        repeat = case.get("repeatability", {})
        require(repeat.get("passed") is True
                and repeat.get("kind") == "empirical_physx_reset_repeatability_probe"
                and repeat.get("candidate_count") == 64 and repeat.get("horizon_steps") == 30
                and repeat.get("cost_tolerance") == 1e-4 and repeat.get("state_tolerance") == .005,
                "Missing matching reported repeat/permutation/isolation evidence")
        for key in ("cost_repeat_max_abs", "cost_permuted_max_abs", "cross_candidate_cost_max_abs"):
            value = repeat.get(key)
            require(isinstance(value, (int, float)) and np.isfinite(value) and 0. <= value <= 1e-4,
                    "Reported repeatability cost check exceeds tolerance")
        errors = repeat.get("state_errors", {})
        require(set(errors) == {"body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos"},
                "Incomplete reported repeatability state channels")
        for values in errors.values():
            require(set(values) == {"repeat_max_abs", "permuted_max_abs", "cross_candidate_max_abs"}
                    and all(isinstance(value, (int, float)) and np.isfinite(value) and 0. <= value <= .005
                            for value in values.values()), "Reported repeatability state check exceeds tolerance")
        name = f"frame_{frame:03d}.npz"
        arrays = pair_analysis.load_arrays(directory / name)
        checks = audit_probe_arrays(arrays, case)
        rechecked.append({"reference_frame": frame, **checks})
        hashes[name] = pair_analysis.sha256(directory / name)
        reported.append({"reference_frame": frame, "reset_audit": reset,
                         "physical_clock_verified": case["physical_clock_verified"],
                         "scalar_batch_stage_cost_max_abs_error": case["scalar_batch_stage_cost_max_abs_error"],
                         "first_substep_torque_max_abs_error_nm": case["first_substep_torque_max_abs_error_nm"],
                         "repeatability": repeat})
    return {"probe_sha256": hashes, "worker_source_sha256": worker["worker_source_sha256"],
            "adapter_source_sha256": worker["adapter_source_sha256"],
            "runtime_source_sha256": sources, "production_source_and_asset_binding_verified": True,
            "objective_and_motion_match_production": True, "reference_frames": [0, 100, 150],
            "horizon_steps": 30, "candidate_count": 64,
            "stored_array_checks_recomputed_here": rechecked,
            "probe_reported_checks_not_reexecuted_here": reported,
            "repeat_trajectories_recomputed_independently": False,
            "torque_and_scalar_batch_parity_recomputed_independently": False,
            "scope": "Stored single-rollout arrays and reported calibration evidence only; repeated/permuted/isolation trajectories and analytic comparison outputs were not archived; no physics stepped by analysis"}


def prefer_capture_report(original, public_case):
    """Prefer the working capture, with a public asset-free replay fallback."""
    return original if original.is_file() else PUBLIC_INPUTS / public_case / "report.json"


def prefer_capture_directory(original, public_case):
    return original if (original / "report.json").is_file() else PUBLIC_INPUTS / public_case


def load_pair(directory):
    summary = pair_analysis.read_json(directory / "summary.json")
    require(summary.get("kind") == "offline_single_controller_fixed_mu_engine_pair"
            and summary.get("algorithm") == "NMPC" and summary.get("fixed_mu") == .7
            and summary.get("reference_id") == "drift_drl_traj4"
            and summary.get("core_coverage_complete") is True
            and summary.get("integration_only") is False
            and summary.get("height_or_flight_failure_gate") is False,
            "Wrong/incomplete nominal measured pair: " + str(directory))
    trace_path = directory / "pair_trace.npz"
    require(summary["artifacts"]["compact_trace_sha256"] == pair_analysis.sha256(trace_path),
            "Compact pair trace identity changed")
    return summary, pair_analysis.load_arrays(trace_path)


def replay_pair(summary, arrays):
    traces, signals = {}, {}
    for label in ("reference", *PLANTS):
        prefix = label + "_"
        trace = {name: arrays[prefix + name] for name in TRACE_FIELDS if prefix + name in arrays}
        n, dt = (349, .02) if label == "reference" else (1397, .005)
        require(trace["time_s"].shape == (n,) and float(trace["physics_dt_s"]) == dt
                and np.allclose(trace["time_s"], np.arange(n)*dt, atol=1e-9, rtol=0),
                "Incomplete or resampled native clock: " + label)
        require(all(not value.dtype.hasobject and np.isfinite(value).all() for value in trace.values()),
                "Nonfinite compact physical trace: " + label)
        result = heading_signals(trace["quaternion"], trace["com_velocity"],
                                 arrays[prefix + "wheel_quaternion_wxyz"], arrays[prefix + "wheel_center_velocity_w"])
        require(all(np.array_equal(value, arrays[prefix + name]) for name, value in result.items()),
                "Exact heading/speed replay failed: " + label)
        traces[label], signals[label] = trace, result
    for window in summary["windows"].values():
        for label, trace in traces.items():
            contacts = {} if label == "reference" else {
                "wheel_ground_contact_count": trace["contact_count"],
                "wheel_ground_force_w": trace["contact_force"],
                "wheel_contact_lateral_slip_m_s": trace["contact_slip"],
            }
            measured = summarize_turn(trace["time_s"], trace["position"], signals[label],
                                      traces["reference"]["time_s"], traces["reference"]["position"][:, :2],
                                      signals["reference"]["speed_m_s"], window[label]["window_s"], **contacts)
            require(measured == window[label], "Exact metric replay failed: " + label)
        require(pair_analysis.pair_xy_gap(traces["isaac"], traces["mujoco"], window["reference"]["window_s"])
                == window["sim2sim_xy_gap"], "Exact paired XY-gap replay failed")
    return traces, signals


def draw_snapshot(ax, trace, signals, index):
    xy = trace["position"][index, :2]
    if signals["root_beta_valid"][index]:
        for key, color, length in (("body_heading_rad", "black", .15), ("course_rad", "#27945f", .13)):
            angle = signals[key][index]
            ax.arrow(*xy, length*np.cos(angle), length*np.sin(angle), color=color,
                     width=.002, head_width=.025, length_includes_head=True, zorder=5)
    for side, color in enumerate(("#c43c39", "#8653a2")):
        if not signals["root_course_to_wheel_valid"][index, side]:
            continue
        angle = signals["wheel_rolling_plane_heading_rad"][index, side]
        delta = .065*np.array([np.cos(angle), np.sin(angle)])
        segment = trace["wheel_pos"][index, side, :2] + np.array([-delta, delta])
        contact = trace["contact_count"][index, side] > 0 and trace["contact_force"][index, side, 2] > 5.
        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=1.8, alpha=1. if contact else .2, zorder=6)


def render_topdown(summary, bundle, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
    reference = bundle["mujoco"][0]["reference"]
    start, end = summary["reference_spec"]["source_demo_interval_s"]
    core_start, core_end = summary["reference_spec"]["evaluation_core_interval_s"]
    selected_reference = (reference["time_s"] >= start-1e-9) & (reference["time_s"] <= end+1e-9)
    for row, predictor in enumerate(PREDICTORS):
        traces, signals = bundle[predictor]
        for col, plant in enumerate(PLANTS):
            ax, trace, signal = axes[row, col], traces[plant], signals[plant]
            t, xy = trace["time_s"], trace["position"][:, :2]
            selected = (t >= start-1e-9) & (t <= end+1e-9)
            core = (t >= core_start-1e-9) & (t <= core_end+1e-9)
            ax.plot(reference["position"][selected_reference, 0], reference["position"][selected_reference, 1],
                    "--", color=".55", lw=1.3)
            ax.plot(xy[selected, 0], xy[selected, 1], color="#2672ae", alpha=.4, lw=1)
            ax.plot(xy[core, 0], xy[core, 1], color="#2672ae", lw=2.2)
            for side in range(2):
                wheels = trace["wheel_pos"][selected, side]
                ax.plot(wheels[:, 0], wheels[:, 1], color="#2672ae", alpha=.2, lw=.6)
            for time in np.arange(start+.2, end, .4):
                draw_snapshot(ax, trace, signal, int(np.argmin(np.abs(t-time))))
            metrics = summary["conditions"][predictor][plant]
            ax.set_title(f"{PREDICTORS[predictor]} → {PLANTS[plant]} plant\n"
                         f"Core XY RMSE {metrics['reference_xy_rmse_m']:.3f} m | speed {100*metrics['mean_speed_retention_ratio']:.1f}%")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.set_xlabel("World X [m]")
    axes[0, 0].set_ylabel("World Y [m]"); axes[1, 0].set_ylabel("World Y [m]")
    fig.suptitle("traj_4 · μ=0.70 · fixed CEM H30/B10/C64/I3/E8 · seed 42\n"
                 "Predictor implementation/backend comparison; one measured episode per condition")
    fig.legend(handles=[Line2D([], [], color=".55", ls="--", label="Kinematic reference"),
                        Line2D([], [], color="black", label="Body heading"),
                        Line2D([], [], color="#27945f", label="COM velocity direction"),
                        Line2D([], [], color="#c43c39", label="Left wheel plane"),
                        Line2D([], [], color="#8653a2", label="Right wheel plane (faint: contact gate unmet)")],
               loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(path, dpi=160); plt.close(fig)


def render_headings(summary, bundle, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    core = summary["reference_spec"]["evaluation_core_interval_s"]
    end = summary["reference_spec"]["source_demo_interval_s"][1]
    for row, predictor in enumerate(PREDICTORS):
        traces, signals = bundle[predictor]
        for col, plant in enumerate(PLANTS):
            ax, trace, signal = axes[row, col], traces[plant], signals[plant]
            t = trace["time_s"]
            selected = t <= end+1e-9
            beta = np.degrees(signal["root_directed_beta_rad"])
            ax.plot(t[selected], np.where(signal["root_beta_valid"], beta, np.nan)[selected],
                    color="black", lw=1, label="COM course − body (directed)")
            for side, color in enumerate(("#c43c39", "#8653a2")):
                gate = (trace["contact_count"][:, side] > 0) & (trace["contact_force"][:, side, 2] > 5.)
                eligible = gate & signal["wheel_center_course_to_wheel_valid"][:, side]
                gamma = np.degrees(signal["wheel_center_course_to_wheel_gamma_rad"][:, side])
                ax.plot(t[selected], np.where(eligible, gamma, np.nan)[selected], color=color, lw=.8,
                        label=("Left" if side == 0 else "Right") + " wheel-center course − wheel (axial, grounded)")
            ax.axvspan(*core, color=".7", alpha=.15); ax.axhline(0., color=".7", lw=.5)
            ax.set_title(f"{PREDICTORS[predictor]} → {PLANTS[plant]}")
            ax.grid(alpha=.2); ax.set_xlabel("Physical time [s]")
    axes[0, 0].set_ylabel("Heading separation [deg]"); axes[1, 0].set_ylabel("Heading separation [deg]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, fontsize=8)
    fig.suptitle("Geometric heading offsets; wheel samples require measured contact count >0 and upward force >5 N\n"
                 "Offsets describe sliding geometry; they are not mismatch or failure gates")
    fig.savefig(path, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserving existing outputs; choose a new output directory")
    new_pair_dir = args.physx_pair_dir
    if new_pair_dir is None:
        require(all(getattr(args, name) is not None for name in ("source_root", "model", "motion_file", "reference_file")),
                "Supply --physx-pair-dir or all --source-root/--model/--motion-file/--reference-file")
        new_pair_dir = output.with_name(output.name + "_physx_pair")
        pair_analysis.run(argparse.Namespace(
            output_dir=new_pair_dir, source_root=args.source_root, model=args.model,
            motion_file=args.motion_file, reference_file=args.reference_file,
            reference_id="drift_drl_traj4", algorithm="NMPC", integration_only=False,
            isaac_trace=args.physx_isaac_dir/"rollout.npz", isaac_report=args.physx_isaac_dir/"report.json",
            mujoco_trace=args.physx_mujoco_dir/"rollout.npz", mujoco_report=args.physx_mujoco_dir/"report.json",
            mujoco_trace_sha256=None, training_manifest=None, checkpoint=None))
    summaries, compact = {}, {}
    for predictor, directory in (("mujoco", args.mujoco_pair_dir), ("physx", new_pair_dir)):
        summaries[predictor], compact[predictor] = load_pair(directory)
    old, new = summaries["mujoco"], summaries["physx"]
    require(old["reference_spec"] == new["reference_spec"]
            and all(old["input_sha256"][key] == new["input_sha256"][key]
                    for key in ("motion_file", "reference_file", "model")),
            "Pairs must share exactly the locked motion, reference, and unmerged FK model")
    for key in compact["mujoco"]:
        if key.startswith("reference_"):
            require(np.array_equal(compact["mujoco"][key], compact["physx"][key]), "Reference replay differs: " + key)
    old_reports = {"isaac": read_report(args.mujoco_predictor_isaac_report),
                   "mujoco": read_report(args.mujoco_predictor_mujoco_report)}
    new_reports = {"isaac": read_report(args.physx_isaac_dir/"report.json"),
                   "mujoco": read_report(args.physx_mujoco_dir/"report.json")}
    identity = audit_raw_reports(old, new, old_reports, new_reports)
    probe_audit = audit_physx_probe(args.physx_probe_dir, identity["physx_predictor_worker_audits"], old_reports["isaac"])
    bundle = {predictor: replay_pair(summaries[predictor], compact[predictor]) for predictor in PREDICTORS}
    core = old["reference_spec"]["evaluation_core_interval_s"]
    rows, conditions, latency, gaps = [], {}, {}, {"between_plants_by_predictor": {}, "between_predictors_by_plant": {}}
    for predictor in PREDICTORS:
        conditions[predictor], latency[predictor] = {}, {}
        reports = old_reports if predictor == "mujoco" else new_reports
        for plant in PLANTS:
            metrics = summaries[predictor]["windows"]["core"][plant]
            conditions[predictor][plant] = metrics
            report = reports[plant]
            latency[predictor][plant] = {key: report.get(key) for key in
                ("compute_ms_median_p95_max", "compute_over_control_dt_fraction", "solver_failures", "real_time_verified")}
            grounded = metrics["ground_contact_heading_geometry"]
            row = {"predictor": PREDICTORS[predictor], "plant": PLANTS[plant], "mu": .7,
                   "core_xy_rmse_m": metrics["reference_xy_rmse_m"], "core_xy_peak_m": metrics["reference_xy_peak_m"],
                   "core_speed_retention_ratio": metrics["mean_speed_retention_ratio"],
                   "core_mean_speed_m_s": metrics["mean_speed_m_s"], "core_net_course_change_deg": metrics["net_course_change_deg"],
                   "core_body_beta_mean_abs_deg": metrics["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"],
                   "core_xy_screen_exceeded": metrics["large_xy_exploratory_screen"]}
            for j, side in enumerate(("left", "right")):
                row[side + "_contact_fraction"] = grounded["contact_fraction_left_right"][j]
                row[side + "_grounded_wheel_center_heading_offset_mean_abs_deg"] = grounded["per_wheel"][side]["wheel_center_course_to_wheel_gamma"]["mean_absolute_deg"]
                row[side + "_grounded_contact_lateral_slip_rms_m_s"] = metrics["ground_contact_lateral_slip_m_s"][side]
            rows.append(row)
        gaps["between_plants_by_predictor"][predictor] = pair_analysis.pair_xy_gap(*[bundle[predictor][0][p] for p in PLANTS], core)
    for plant in PLANTS:
        gaps["between_predictors_by_plant"][plant] = pair_analysis.pair_xy_gap(*[bundle[p][0][plant] for p in PREDICTORS], core)
    deltas = {"physx_minus_mujoco_predictor_by_plant": {}, "cross_minus_native_plant_by_predictor": {}}
    delta_fields = ("reference_xy_rmse_m", "reference_xy_peak_m", "mean_speed_retention_ratio")
    for plant in PLANTS:
        deltas["physx_minus_mujoco_predictor_by_plant"][plant] = {
            key: conditions["physx"][plant][key] - conditions["mujoco"][plant][key] for key in delta_fields}
    for predictor, native, cross in (("mujoco", "mujoco", "isaac"), ("physx", "isaac", "mujoco")):
        deltas["cross_minus_native_plant_by_predictor"][predictor] = {
            key: conditions[predictor][cross][key] - conditions[predictor][native][key] for key in delta_fields}
    summary = {"schema_version": 1, "kind": "measured_cem_predictor_by_plant_nominal_2x2",
               "reference_id": old["reference_id"], "reference_spec": old["reference_spec"], "plant_mu": .7,
               "prediction_mu_scenarios": [.7], "prediction_receives_plant_mu": False,
               "conditions": conditions, "paired_xy_gaps": gaps, "performance_deltas": deltas,
               "compute_diagnostics": latency,
               "posture_diagnostics": {predictor: summaries[predictor]["posture_diagnostics"]
                                       for predictor in PREDICTORS},
               "controller_contract_audit": identity,
               "physx_calibration_probe_audit": probe_audit,
               "compact_pair_sha256": {predictor: {name: pair_analysis.sha256(directory/name)
                    for name in ("summary.json", "pair_trace.npz")} for predictor, directory in
                    (("mujoco", args.mujoco_pair_dir), ("physx", new_pair_dir))},
               "metric_audit": {"physical_episode_metrics_exactly_replayed": 4,
                   "pair_xy_gaps_exactly_replayed": 2, "heading_and_speed_exactly_replayed": True,
                   "physics_dt_s": .005, "samples_per_episode": 1397,
                   "ground_contact_gate": "measured ground-contact count >0 and upward wheel force >5 N",
                   "position_alignment": "same physical time and world coordinates; no registration or time warp"},
               "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
               "new_simulation_performed_by_analysis": False, "pooled_success_statistics": False,
               "algorithm_ranking_claimed": False,
               "causal_physics_backend_only_comparison": False,
               "limitations": [
                   "One seed-42 physical episode per condition; results are descriptive, not success probabilities.",
                   "Historical MuJoCo prediction cold-resets each 20 ms control interval; PhysX prediction reconstructs once per candidate and integrates continuously over the full horizon. This compares predictor implementations/backends, not a causal engine-only change.",
                   "Both predictors retain the same MuJoCo-derived physics-only bias-force reference seed; PhysX prediction therefore is not independent of all MuJoCo computations.",
                   "Predictor source instrumentation and archived runner revisions differ; reported objective, action parameterization, search configuration, state/reference identities and runtime hashes are retained.",
                   "Equal numeric friction does not calibrate equivalent wheel-ground contact laws, asset imports or solver states across engines.",
                   "Native MuJoCo prediction uses float64 candidate controls/state; PhysX quantizes candidate controls to float32 before stepping/scoring. Both validation plants consume float32 actions, so predictor precision is an additional implementation difference.",
                   "Contact reward measurements differ: MuJoCo post-step mj_forward re-evaluates constraint forces at advanced q/dq, while PhysX ContactSensor reports preceding completed-step impulses/dt. Shared reward math/thresholds do not imply a shared measurement operator; XY and speed are the strongest comparisons.",
                   "Each condition replans from its own plant state; the four episodes do not share an identical action sequence.",
                   "PhysX internal contact caches cannot be serialized by the scene-state API; reset-repeatability probes cover tested states and plans only.",
                   "Measured body/wheel heading offsets are descriptive geometry, distinct from recorded contact-point lateral slip. Neither offset nor height/flight is an automatic failure label.",
                   "The current PhysX worker steps all 64 environments even for small rescoring batches; padded slots are excluded from returned costs. Timing reflects the present batching, IPC and hardware, not intrinsic engine speed.",
                   "Offline computation can exceed 20 ms; no wall-clock action delay is injected, and these results do not establish real-time control."]}
    output.mkdir(parents=True)
    with (output/"metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    arrays = {key: value for key, value in compact["mujoco"].items() if key.startswith("reference_") or key in ("schema_version", "wheel_order")}
    for predictor, values in compact.items():
        arrays.update({predictor + "_predictor_" + key: value for key, value in values.items()
                       if key.startswith("isaac_") or key.startswith("mujoco_")})
    np.savez_compressed(output/"prediction_transfer_trace.npz", **arrays)
    render_topdown(summary, bundle, output/"topdown.png")
    render_headings(summary, bundle, output/"heading_offsets.png")
    summary["analysis_source_sha256"] = {name: pair_analysis.sha256(ROOT/"scripts"/name) for name in
        ("analyze_mpc_prediction_transfer.py", "analyze_controller_pair.py", "controller_metrics.py", "nmpc_tracking_objective.py")}
    summary["artifacts"] = {path.name: {"file": path.name, "sha256": pair_analysis.sha256(path)} for path in sorted(output.iterdir())}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output), "conditions": rows, "paired_xy_gaps": gaps}, indent=2))
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mujoco-pair-dir", type=Path, default=OLD_PAIR)
    result.add_argument("--physx-pair-dir", type=Path, help="Existing audited pair; otherwise create it from the raw episodes using the supplied assets")
    result.add_argument("--physx-isaac-dir", type=Path,
                        default=prefer_capture_directory(RAW_ROOT/"isaac", "physx_isaac"))
    result.add_argument("--physx-mujoco-dir", type=Path,
                        default=prefer_capture_directory(RAW_ROOT/"mujoco", "physx_mujoco"))
    result.add_argument("--mujoco-predictor-isaac-report", type=Path,
                        default=prefer_capture_report(OLD_RAW["isaac"], "mujoco_isaac"))
    result.add_argument("--mujoco-predictor-mujoco-report", type=Path,
                        default=prefer_capture_report(OLD_RAW["mujoco"], "mujoco_mujoco"))
    result.add_argument("--physx-probe-dir", type=Path, default=DEFAULT_PROBE,
                        help="Passed H30/C64 calibration report and saved arrays; source-bound to both production workers")
    for name in ("source-root", "model", "motion-file", "reference-file"):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
