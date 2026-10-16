"""Allowlist-publication tests on synthetic JSON, without robot assets/physics."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_nmpc_nominal_diagnostics as public


def component(samples):
    return {"projection_valid_samples": samples,
        "total_error_m_projection_valid": {"samples": samples, "rmse": .1, "absolute_peak": .2},
        "total_error_m_all_samples": {"samples": samples},
        "along_m": {"signed_mean": -.05, "rmse": .06}, "normal_m": {"signed_mean": .04, "rmse": .08},
        "along_energy_fraction": .36, "normal_energy_fraction": .64, "behind_fraction": 1.,
        "maximum_squared_decomposition_residual_m2": 1e-16}


def fixtures():
    case = {"coverage_complete": True, "interval_s": [1.44, 4.], "native_samples": 513,
        "decomposition": component(513), "core_subwindows": [{"interval_s": interval, "decomposition": component(count)}
            for interval, count in zip([[1.44, 2.], [2., 3.], [3., 4.]], [113, 201, 201])]}
    xy = {"kind": "same_time_root_origin_xy_components", "reference_id": "drift_drl_traj4",
        "source_demo_interval_s": [0., 4.96], "core_interval_s": [1.44, 4.],
        **{key: False for key in ("registration_or_time_warp", "physics_or_fk_used", "height_or_flight_failure_gate",
                                 "heading_offset_zero_target", "causal_contact_engine_diagnosis")},
        "source_sha256": {"analyze_xy_error_components.py": public.XY_SOURCE, "nmpc_tracking_objective.py": public.OBJECTIVE_SOURCE},
        "definitions": {name: "synthetic portable scientific definition" for name in ("error", "along", "normal", "derivative", "endpoints",
            "measured_velocity", "original_com_offset_m", "speed_validity_threshold_m_s", "subwindows")},
        "cases": {label: {engine: copy.deepcopy(case) for engine in ("isaac", "mujoco")} for label in ("baseline", "residual")},
        "input_sha256": {label: {key: "a"*64 for key in ("summary_sha256", "compact_trace_sha256")} for label in ("baseline", "residual")}}
    common = {key: "d"*64 for key in ("source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "model_snapshot_sha256")}
    common.update(mujoco_version="3.8.1", header_version=3008001, abi_version=1)
    native = {**common, "native_library_sha256": "b"*64, "compiler_command": ["/private/user/build"], "worker_pid": 991}
    original = {**common, "native_library_sha256": "c"*64, "compiler_command": ["/data/user/build"], "worker_pid": 992}
    probe = {"kind": "bounded_existing_isaac_action_one_interval_prediction_audit", "status": "audited",
        "predicted_control_intervals": 16, "predicted_physics_substeps": 64, "selected_policy_frames": list(public.FRAMES),
        "probe_source_sha256": public.PROBE_SOURCE,
        **{key: False for key in ("new_closed_loop_episode", "new_cem_search", "new_native_build", "new_isaac_app", "learned_policy_data_used")},
        "matched_frozen_source_sha256": {key: "d"*64 for key in public.PROBE_SOURCES},
        "native_backend": native, "original_remote_native_backend": {"workers": {"0.7": original}},
        "local_native_library_sha256": "b"*64, "original_remote_native_library_sha256": "c"*64,
        "local_native_build_manifest_sha256": "e"*64, "native_binary_matches_original_remote": False,
        "archive_sha256": {key: "a"*64 for key in ("report.json", "rollout.npz", "solver_log.json")},
        **{key: "a"*64 for key in ("model_sha256", "motion_sha256", "settings_sha256", "task_profile_sha256", "actuator_contract_sha256")},
        "independent_converter_vs_recorded_packed_max_abs_error": 1e-15,
        "independent_FK_vs_direct_measured_channel_checks": [{"wheel_origins_max_abs_error_m": 6e-7}],
        "selection": [{"kind": "fixed_clock_stratum", "policy_frame": frame} for frame in public.FRAMES]}
    probe["matched_frozen_source_sha256"]["scripts/nmpc_tracking_objective.py"] = public.OBJECTIVE_SOURCE
    direct = {"root_xy_prediction_error_each_substep_m": [.001, .002, .003, .001],
        "root_rotation_tangent_prediction_error_each_substep_rad": [.001]*4,
        "root_COM_velocity_error_each_substep_m_s": [[.02, 0., 0.]]*4,
        "root_angular_velocity_error_each_substep_rad_s": [[.7, 0., 0.]]*3+[[.5, 0., 0.]],
        "predicted_minus_measured_torque_each_substep_nm": [[.00001]*8]+[[21.]*8]*2+[[15.]*8],
        "pure_DC_actuator_model_on_actual_prestates_minus_measured_torque_nm": [[.00002]*8]*4,
        "net_body_contact_force_error_each_substep_max_n": [100., 456., 200., 219.]}
    probe["rows"] = [{"policy_frame": frame, "interval_s": [frame*.02, (frame+1)*.02], "direct_measured_comparisons": copy.deepcopy(direct)} for frame in public.FRAMES]
    probe["aggregate"] = dict(zip(("maximum_root_xy_prediction_error_m", "maximum_root_rotation_prediction_error_rad",
        "maximum_root_COM_velocity_component_error_m_s", "maximum_root_angular_velocity_component_error_rad_s",
        "maximum_torque_prediction_error_nm", "maximum_net_body_contact_force_prediction_error_n"), [.003, .001, .02, .7, 21., 456.]))
    probe["aggregate"].update(maximum_first_substep_torque_prediction_error_nm=.00001,
        maximum_pure_actuator_model_error_on_actual_prestates_nm=.00002,
        maximum_absolute_stage_cost_prediction_error=.003,
        maximum_FK_reconstruction_vs_logged_stage_cost_error=2e-8,
        maximum_FK_reconstruction_vs_logged_weighted_term_error=7e-7)
    return probe, xy


def test_allowlist_drops_private_metadata_and_distinguishes_substep_from_endpoint():
    probe, xy = fixtures()
    probe["arbitrary_private_metadata"] = {"address": "14.103.233.29", "model": "/home/private/model.xml"}
    xy["unneeded_full_metadata"] = {"weight": "/data/secret.pt"}
    result = public.build_summary(probe, xy)
    text = json.dumps(result)
    assert all(token not in text for token in ("/home/", "/data/", "/private/", "14.103.", "worker_pid", "compiler_command", "secret.pt"))
    assert result["friction_grid"] is False and result["frozen_controller"] is False
    p = result["isaac_one_interval_probe"]
    assert p["all_64_substeps_maxima"]["maximum_torque_prediction_error_nm"] == 21.
    assert p["final_20ms_endpoints_maxima"]["maximum_torque_prediction_error_nm"] == 15.
    assert p["actuator_checks"]["PD_DC_on_actual_prestates_max_abs_error_nm"] == .00002
    assert p["runtime_identity"]["native_binary_matches_original_remote"] is False
    assert len(result["xy_components"]["cases"]["residual"]["isaac"]["descriptive_subwindows"]) == 3


@pytest.mark.parametrize("problem", ["probe_source", "xy_source", "clock", "scope", "aggregate", "selection", "header", "model_snapshot", "binary", "cpp", "xy_samples", "xy_bins", "energy", "nan", "private_definition"])
def test_reject_wrong_source_scope_clock_runtime_or_metric_consistency(problem):
    probe, xy = fixtures()
    if problem == "probe_source": probe["probe_source_sha256"] = "f"*64
    elif problem == "xy_source": xy["source_sha256"]["analyze_xy_error_components.py"] = "f"*64
    elif problem == "clock": probe["rows"][0]["interval_s"] = [0., .04]
    elif problem == "scope": probe["new_cem_search"] = True
    elif problem == "aggregate": probe["aggregate"]["maximum_torque_prediction_error_nm"] = 22.
    elif problem == "selection": probe["selected_policy_frames"] = [0]*16
    elif problem == "header": probe["native_backend"]["header_version"] += 1
    elif problem == "model_snapshot": probe["native_backend"]["model_snapshot_sha256"] = "f"*64
    elif problem == "binary": probe["native_binary_matches_original_remote"] = True
    elif problem == "cpp": probe["native_backend"]["source_sha256"] = probe["original_remote_native_backend"]["workers"]["0.7"]["source_sha256"] = "f"*64
    elif problem == "xy_samples": xy["cases"]["residual"]["isaac"]["native_samples"] = 512
    elif problem == "xy_bins": xy["cases"]["baseline"]["mujoco"]["core_subwindows"][0]["interval_s"] = [1.44, 2.1]
    elif problem == "energy": xy["cases"]["residual"]["isaac"]["decomposition"]["along_m"]["rmse"] = .2
    elif problem == "nan": probe["rows"][0]["direct_measured_comparisons"]["root_COM_velocity_error_each_substep_m_s"][0][0] = np.nan
    else: xy["definitions"]["error"] = "/home/user/private-source"
    with pytest.raises(ValueError): public.build_summary(probe, xy)


def test_publish_refuses_unbound_json_and_existing_outputs(tmp_path):
    probe = tmp_path/"probe.json"; xy = tmp_path/"xy.json"; output = tmp_path/"out"
    probe.write_text("{}"); xy.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        public.publish(probe, xy, output)
    assert not output.exists()
    output.mkdir(); token = output/"user-owned"; token.write_text("keep")
    with pytest.raises(ValueError, match="Preserving"):
        public.publish(probe, xy, output)
    assert token.read_text() == "keep"


def test_published_bundle_has_only_one_path_free_summary_and_replays_when_inputs_exist():
    directory = ROOT/"results/nmpc_nominal_diagnostics_20261016_v1"
    assert sorted(path.name for path in directory.iterdir()) == ["summary.json"]
    text = (directory/"summary.json").read_text()
    assert all(token not in text for token in ("/home/", "/data/", "/private/", "compiler_command", "worker_pid", "zengxiong"))
    result = json.loads(text)
    assert result["input_diagnostic_json_sha256"] == {"isaac_one_interval_probe": public.PROBE_SHA, "xy_components": public.XY_SHA}
    probe = ROOT/"outputs/nmpc_isaac_nominal_prediction_probe_20261016_v1.json"
    xy = ROOT/"outputs/nmpc_nominal_xy_components_20261016_v1/summary.json"
    if probe.is_file() and xy.is_file():
        assert public.file_sha(probe) == public.PROBE_SHA and public.file_sha(xy) == public.XY_SHA
        assert public.build_summary(json.loads(probe.read_text()), json.loads(xy.read_text())) == result
    assert result["isaac_one_interval_probe"]["physics_substeps"] == 64
    assert result["isaac_one_interval_probe"]["final_20ms_endpoints_maxima"]["maximum_root_angular_velocity_component_error_rad_s"] == pytest.approx(.5190866325506109)
    assert result["xy_components"]["cases"]["residual"]["isaac"]["core"]["along_energy_fraction"] == pytest.approx(.7265, abs=.0001)
