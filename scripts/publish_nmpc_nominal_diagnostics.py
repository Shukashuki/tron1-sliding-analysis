"""Publish one allowlisted, offline-only nominal diagnostic JSON; no plots.

Inputs are the two exact completed private diagnostic artifacts. This script
never invokes either analyzer, FK, simulation, inference, a native worker or
network access. It intentionally does not copy arbitrary input metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


PROBE_SHA = "62da1211cf468a9d3a0f0dc4c02b1a6fe5a04dd52a0dad3b6c75848bc1ecbcde"
XY_SHA = "c1677c1f408272b2174d71f6eb7ce06c7bc41f29bbc9dd0e51d6a4600836c24e"
PROBE_SOURCE = "0f4b0caac0633b0db6817a2c32b4b636ac5fde0fdaeb2895676cee92f1d52ca6"
XY_SOURCE = "ed8ee67b7051a8e92c528cf4dd6923a71cda7e921b9b3cd909977ed7fd5ce9bd"
OBJECTIVE_SOURCE = "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860"
FRAMES = [0, 14, 19, 43, 72, 73, 90, 100, 117, 125, 145, 150, 161, 175, 195, 225]
PROBE_SOURCES = tuple("scripts/" + name for name in ("run_nmpc_tracking.py", "run_nmpc_tracking_isaac.py",
    "nmpc_shooting.py", "nmpc_tracking_objective.py", "nmpc_batch_objective.py", "nmpc_native_batch.py",
    "classical_tracking.py", "controller_references.py", "height_smoothness.py")) + ("native/nmpc_batch.cpp",) + tuple(
    "source-root/" + name for name in ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
        "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py", "training/tron1_tracking.py",
        "training/tron1_drift.py", "training/tron1_runtime_device.py", "training/tron1_physics_clock.py"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA-256")
    return value


def hash_map(value):
    require(isinstance(value, dict) and bool(value), "Missing source/input identities")
    return {name: digest(item) for name, item in value.items()}


def finite(value, shape):
    value = np.asarray(value, dtype=float)
    require(value.shape == shape and np.isfinite(value).all(), "Invalid diagnostic array shape/finiteness")
    return value


def component_summary(value, samples):
    require(value["projection_valid_samples"] == samples
            and value["total_error_m_projection_valid"]["samples"] == samples
            and value["total_error_m_all_samples"]["samples"] == samples, "Censored or inconsistent XY sample coverage")
    result = {"samples": samples,
        "total_xy_rmse_m": value["total_error_m_projection_valid"]["rmse"],
        "total_xy_peak_m": value["total_error_m_projection_valid"]["absolute_peak"],
        "along_signed_mean_m": value["along_m"]["signed_mean"], "along_rmse_m": value["along_m"]["rmse"],
        "normal_signed_mean_m": value["normal_m"]["signed_mean"], "normal_rmse_m": value["normal_m"]["rmse"],
        "along_energy_fraction": value["along_energy_fraction"], "normal_energy_fraction": value["normal_energy_fraction"],
        "behind_fraction": value["behind_fraction"],
        "maximum_squared_decomposition_residual_m2": value["maximum_squared_decomposition_residual_m2"]}
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in result.values()),
            "Nonfinite XY diagnostic")
    require(all(result[key] >= 0 for key in ("total_xy_rmse_m", "total_xy_peak_m", "along_rmse_m", "normal_rmse_m"))
            and all(0 <= result[key] <= 1 for key in ("along_energy_fraction", "normal_energy_fraction", "behind_fraction")),
            "Invalid XY diagnostic ranges")
    require(math.isclose(result["total_xy_rmse_m"]**2, result["along_rmse_m"]**2 + result["normal_rmse_m"]**2,
                         abs_tol=1e-12, rel_tol=1e-10), "XY energy decomposition is inconsistent")
    return result


def xy_summary(xy):
    require(xy["kind"] == "same_time_root_origin_xy_components" and xy["reference_id"] == "drift_drl_traj4"
            and xy["source_demo_interval_s"] == [0., 4.96] and xy["core_interval_s"] == [1.44, 4.], "Wrong XY task/clock")
    require(all(xy.get(key) is False for key in ("registration_or_time_warp", "physics_or_fk_used",
        "height_or_flight_failure_gate", "heading_offset_zero_target", "causal_contact_engine_diagnosis")), "Unsupported XY interpretation")
    sources = hash_map({key: xy["source_sha256"][key] for key in ("analyze_xy_error_components.py", "nmpc_tracking_objective.py")})
    require(sources.get("analyze_xy_error_components.py") == XY_SOURCE
            and sources.get("nmpc_tracking_objective.py") == OBJECTIVE_SOURCE, "XY source identity mismatch")
    definitions = {name: xy["definitions"][name] for name in ("error", "along", "normal", "derivative", "endpoints",
        "measured_velocity", "original_com_offset_m", "speed_validity_threshold_m_s", "subwindows")}
    cases = {}
    for label in ("baseline", "residual"):
        cases[label] = {}
        for engine in ("isaac", "mujoco"):
            value = xy["cases"][label][engine]
            require(value["coverage_complete"] is True and value["interval_s"] == [1.44, 4.]
                    and value["native_samples"] == 513, "Incomplete XY core clock")
            bins = value["core_subwindows"]
            require([row["interval_s"] for row in bins] == [[1.44, 2.], [2., 3.], [3., 4.]], "Changed descriptive XY bins")
            cases[label][engine] = {"core": component_summary(value["decomposition"], 513),
                "descriptive_subwindows": [{"interval_s": row["interval_s"], **component_summary(row["decomposition"], count)}
                    for row, count in zip(bins, (113, 201, 201))]}
    return {"definitions": definitions, "input_pair_sha256": {label: hash_map({key: xy["input_sha256"][label][key]
                for key in ("summary_sha256", "compact_trace_sha256")}) for label in cases},
            "source_sha256": sources, "cases": cases}


def probe_summary(probe):
    require(probe["kind"] == "bounded_existing_isaac_action_one_interval_prediction_audit" and probe["status"] == "audited"
            and probe["predicted_control_intervals"] == 16 and probe["predicted_physics_substeps"] == 64
            and probe["selected_policy_frames"] == FRAMES and probe["probe_source_sha256"] == PROBE_SOURCE,
            "Wrong bounded probe scope/source")
    require(all(probe[key] is False for key in ("new_closed_loop_episode", "new_cem_search", "new_native_build",
                                               "new_isaac_app", "learned_policy_data_used")), "Probe scope is not observation-only audit")
    sources = hash_map({key: probe["matched_frozen_source_sha256"][key] for key in PROBE_SOURCES})
    require(sources["scripts/nmpc_tracking_objective.py"] == OBJECTIVE_SOURCE, "Probe objective source mismatch")
    rows = probe["rows"]
    require([row["policy_frame"] for row in rows] == FRAMES and len(probe["selection"]) == 16, "Probe selection mismatch")
    metrics = {
        "maximum_root_xy_prediction_error_m": ("root_xy_prediction_error_each_substep_m", (4,)),
        "maximum_root_rotation_prediction_error_rad": ("root_rotation_tangent_prediction_error_each_substep_rad", (4,)),
        "maximum_root_COM_velocity_component_error_m_s": ("root_COM_velocity_error_each_substep_m_s", (4, 3)),
        "maximum_root_angular_velocity_component_error_rad_s": ("root_angular_velocity_error_each_substep_rad_s", (4, 3)),
        "maximum_torque_prediction_error_nm": ("predicted_minus_measured_torque_each_substep_nm", (4, 8)),
        "maximum_net_body_contact_force_prediction_error_n": ("net_body_contact_force_error_each_substep_max_n", (4,))}
    all_substeps, endpoints = {}, {}
    for output_key, (field, shape) in metrics.items():
        arrays = [finite(row["direct_measured_comparisons"][field], shape) for row in rows]
        all_substeps[output_key] = max(float(np.max(np.abs(value))) for value in arrays)
        endpoints[output_key] = max(float(np.max(np.abs(value[-1]))) for value in arrays)
        require(math.isclose(all_substeps[output_key], probe["aggregate"][output_key], abs_tol=1e-12, rel_tol=1e-12),
                "Probe row/aggregate mismatch")
    for row in rows:
        k = row["policy_frame"]
        require(np.allclose(finite(row["interval_s"], (2,)), [.02*k, .02*(k+1)], atol=1e-12, rtol=0), "Probe interval clock mismatch")
    first_tau = max(float(np.max(np.abs(finite(row["direct_measured_comparisons"]["predicted_minus_measured_torque_each_substep_nm"], (4, 8))[0]))) for row in rows)
    actual_prestates_tau = max(float(np.max(np.abs(finite(row["direct_measured_comparisons"]["pure_DC_actuator_model_on_actual_prestates_minus_measured_torque_nm"], (4, 8))))) for row in rows)
    for name, value in (("maximum_first_substep_torque_prediction_error_nm", first_tau),
                        ("maximum_pure_actuator_model_error_on_actual_prestates_nm", actual_prestates_tau)):
        require(math.isclose(value, probe["aggregate"][name], abs_tol=1e-12, rel_tol=1e-12), "Probe torque aggregate mismatch")
    native, original = probe["native_backend"], probe["original_remote_native_backend"]["workers"]["0.7"]
    shared = {}
    for key in ("source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "model_snapshot_sha256"):
        require(digest(native[key]) == digest(original[key]), "Native source/runtime/model identity mismatch")
        shared[key] = native[key]
    require(shared["source_sha256"] == sources["native/nmpc_batch.cpp"], "Native C++ source identity mismatch")
    for key, expected in (("mujoco_version", "3.8.1"), ("header_version", 3008001), ("abi_version", 1)):
        require(native[key] == original[key] == expected, "Native version/ABI mismatch")
        shared[key] = expected
    local_so = digest(probe["local_native_library_sha256"]); original_so = digest(probe["original_remote_native_library_sha256"])
    require(local_so == native["native_library_sha256"] and original_so == original["native_library_sha256"]
            and probe["native_binary_matches_original_remote"] is (local_so == original_so), "Native binary identity contradiction")
    selection = []
    for expected, item in zip(FRAMES, probe["selection"]):
        require(item["policy_frame"] == expected and item["kind"] in ("fixed_clock_stratum", "first_unused_measured_wheel_contact_transition"), "Invalid probe selection")
        selection.append({key: item[key] for key in ("kind", "policy_frame", "stratum_s", "event_sample", "event_time_s", "active_before_L_R", "active_after_L_R") if key in item})
    return {"selected_policy_frames": FRAMES, "selected_intervals": selection, "control_intervals": 16,
        "substeps_per_interval": 4, "physics_substeps": 64, "control_dt_s": .02, "physics_dt_s": .005,
        "selection_scope": "12 fixed clock frames plus earliest unused measured wheel-ground contact transition in each of four predeclared time strata; no prediction-error selection.",
        "input_episode_sha256": hash_map({key: probe["archive_sha256"][key] for key in ("report.json", "rollout.npz", "solver_log.json")}),
        "input_file_sha256": {key: digest(probe[key]) for key in ("model_sha256", "motion_sha256", "settings_sha256", "task_profile_sha256", "actuator_contract_sha256")},
        "source_sha256": {"probe_nmpc_isaac_prediction.py": PROBE_SOURCE, **sources},
        "runtime_identity": {"shared": shared, "local_native_library_sha256": local_so,
            "original_remote_native_library_sha256": original_so,
            "local_native_build_manifest_sha256": digest(probe["local_native_build_manifest_sha256"]),
            "native_binary_matches_original_remote": local_so == original_so},
        "all_64_substeps_maxima": all_substeps, "final_20ms_endpoints_maxima": endpoints,
        "maximum_definitions": "XY is Euclidean XY-position error; rotation is SO(3) tangent norm. Velocity, torque and net-body-force maxima are absolute COMPONENT maxima, not vector magnitudes or tire-normal forces.",
        "actuator_checks": {"first_substep_torque_max_abs_error_nm": first_tau,
            "PD_DC_on_actual_prestates_max_abs_error_nm": actual_prestates_tau,
            "interpretation": "Later predicted torque differs after predicted state diverges; this is not identification of an actuator-map error."},
        "objective_and_reconstruction": {key: probe["aggregate"][key] for key in (
            "maximum_absolute_stage_cost_prediction_error", "maximum_FK_reconstruction_vs_logged_stage_cost_error", "maximum_FK_reconstruction_vs_logged_weighted_term_error")},
        "independent_converter_vs_recorded_packed_max_abs_error": probe["independent_converter_vs_recorded_packed_max_abs_error"],
        "FK_wheel_origin_max_abs_component_error_m": max(item["wheel_origins_max_abs_error_m"] for item in probe["independent_FK_vs_direct_measured_channel_checks"]),
        "measured_channels": "Base-link pose, base original-COM twist, eight joint q/dq, two wheel-origin positions, applied torques and nine-body net contact forces are recorded.",
        "FK_scope": "Six intermediate-link positions and eight non-root orientations/COM velocities are reconstructed, not extra measured Isaac channels; wheel-origin measurements separately check FK.",
        "contact_history": "Exact raw newest-first PhysX history, including nonzero reset history, is supplied. Native chronological output is reversed only for array comparison; reward history-max norm is order-invariant. No unobserved solver warm-start is invented.",
        "limits": "One observed action per selected state, not a horizon-rollout or counterfactual-policy certificate. Engine-specific net-force differences do not identify a calibrated tire-force or causal contact parameter."}


def build_summary(probe, xy):
    result = {"schema_version": 1, "kind": "sanitized_existing_nominal_nmpc_diagnostics",
        "reference_id": "drift_drl_traj4", "plant_mu": .7, "prediction_mu_scenarios": [.7],
        "controller_development_only": True, "frozen_controller": False, "friction_grid": False,
        "controller_ranking_claimed": False, "causal_contact_engine_diagnosis": False,
        "height_flight_or_heading_failure_gate": False,
        "input_diagnostic_json_sha256": {"isaac_one_interval_probe": PROBE_SHA, "xy_components": XY_SHA},
        "xy_components": xy_summary(xy), "isaac_one_interval_probe": probe_summary(probe),
        "limitations": ["Same-time along/normal projection is not nearest-path distance, contact slip or measured controller latency.",
            "Small short-interval position error does not exclude angular/force/model errors or establish long-horizon fidelity.",
            "Single nominal coefficient and seed; no other-friction tuning, frozen-controller qualification or causal optimization diagnosis is established."]}
    text = json.dumps(result, allow_nan=False)
    require(not any(token in text for token in ("/home/", "/data/", "/private/", "zengxiong", "14.103.", "117.50.", "compiler_command", "worker_pid")),
            "Private or unallowlisted runtime metadata in public output")
    return result


def publish(probe_path, xy_path, output_dir):
    output = Path(output_dir)
    require(not output.exists(), "Preserving existing diagnostics publication")
    require(file_sha(probe_path) == PROBE_SHA and file_sha(xy_path) == XY_SHA, "Raw diagnostic JSON SHA-256 mismatch")
    result = build_summary(json.loads(Path(probe_path).read_text()), json.loads(Path(xy_path).read_text()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe-json", "xy-json", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    publish(args.probe_json, args.xy_json, args.output_dir)
