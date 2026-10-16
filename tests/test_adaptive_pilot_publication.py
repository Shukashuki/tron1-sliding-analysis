"""Asset-free replay of the published four-episode adaptive-MPC pilot.

Only derived public arrays and JSON are read: no source episodes, licensed
model, policy weights, native library, simulator stepping or remote access.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from analyze_controller_pair import posture_diagnostic
from analyze_nmpc_nominal_development import configuration_sha256, contact_diagnostic
from controller_metrics import heading_signals, summarize_turn
from controller_references import get_reference
from nmpc_tracking_objective import JOINT_NAMES


FOLDER = ROOT/"results/adaptive_mpc_pilot_20261016_v1"
NOTE = ROOT/"results/2026-10-16-adaptive-mpc.md"
CASES = (
    ("fixed_mu0p7", .7, False,
     "3e10ea13e84d4d2da47694cae20b5147582bee837a8d4d02c1238e1635700e26",
     "9b89a75b8a09318d215d43d16e89b131731a07fc9d26eb2d86e8b061cb575d2b",
     "108ccee2107b87f243b9741bb0f65ca958a05dac18bfc6c47422c3fdc4e297cc"),
    ("adaptive_mu0p7", .7, True,
     "cc56aa2adb94dee5e317fce355fc6baca2d04755cc7c27b98e1a66b164476382",
     "b2de37e92c082e6627db6d5cd2e97a7029b3e6d775076112710715aafc4333ce",
     "8a77aaf9239c4363818db5e35d20bf907c5c83361bdcd3f46d8a2cd52f533ece"),
    ("fixed_mu0p3", .3, False,
     "24471754e2fcd8fbfcb090b3ece8659ba51062bf3652974f7cbc027fd67e9d74",
     "14b3a6f249279eb55afcc2ed6382d8b5fa78d135f8cb95ca3e1ce0095bb23c5f",
     "e75bc14d5905edc1c2bd5a9c0e7ce782982e0df250d62d93a99e2ecbad1b5355"),
    ("adaptive_mu0p3", .3, True,
     "2c141ff6b20733e6b07c208b73bfc18718f91ee361108487cf0fefc9b48f5ea8",
     "b2bdb69f62e2aefeb26aced87933b87dc3329cd4342e84cabcd44732cad8973e",
     "e657b8f4c6f65e238f5cf8d81f7a82879fbad6d12bfa8d01ae392b566000b8ea"),
)
LABELS = ("reference",) + tuple(case[0] for case in CASES)
STATE = ("time_s", "physics_dt_s", "joint_names", "position", "quaternion", "com_velocity",
         "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")
GEOMETRY = ("wheel_quaternion_wxyz", "wheel_center_velocity_w")
SIGNALS = ("speed_m_s", "wheel_center_speed_m_s", "course_rad", "body_heading_rad",
           "wheel_rolling_plane_heading_rad", "root_directed_beta_rad", "root_course_to_wheel_gamma_rad",
           "wheel_center_course_to_wheel_gamma_rad", "root_beta_valid", "root_course_to_wheel_valid",
           "wheel_center_course_to_wheel_valid", "speed_threshold_m_s")
MEASURED = ("contact_count", "contact_force", "contact_slip", "nonwheel_ground_contact_count",
            "nonwheel_ground_contact_force_norm_n", "held_action")
PRIVATE = ("/home/", "/data/", "/private/", "/tmp/", "file://", "zengxiong",
           "14.103.233.29", "117.50.226.213", "BEGIN OPENSSH PRIVATE KEY")
RUNTIME = {
    "scripts/run_nmpc_tracking.py": "5a5f714db4d8165f5d39d98372d96f97dd36d8e87f82c53bef740a620c5c1980",
    "scripts/nmpc_shooting.py": "7c667ef0cbd32c58ce53882302e75f5d33487bcfea51c2f6ec3435d454a480ce",
    "scripts/nmpc_native_batch.py": "bb19aeea51cf03eedd87c0fb9e4561c11bc8ab5569cabf31c5465287a815fb76",
    "scripts/nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
    "scripts/nmpc_batch_objective.py": "1c9da90ae66778a71ceb2e6bb9d0dcf7f0a2259fafee39b37cbe958b79178fae",
    "scripts/classical_tracking.py": "3fcafdb6a16058280aa1e5a0ade6113f6963ddf3085c25ebe2ed6fbb30661c57",
    "native/nmpc_batch.cpp": "deb3afe3367e11c28295e00e4a081e84968818070dcfb7044f06466761875ed2",
    "config/nmpc_height_smooth_residual.json": "643262e037d766e924c5ffa165385f2f9a85d5af0257192f222d7106e2685266",
    "config/car_drift_height_smooth_task.json": "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6",
}
ADAPTIVE_RUNTIME = {
    "scripts/adaptive_contact_model.py": "7edcbe37d2c30b57d9f204a09703f6292f783b614fa5a8f33d40fa5c3ae5259c",
    "scripts/adaptive_mpc_runtime.py": "02924542acefde8e0f06392984290c57e057ec2715d9e1da709dc04e1d4803e2",
    "scripts/adaptive_nmpc.py": "a958046f258d9b91201e5e91232f924058e7840d45e2ac2ddd37d2675c4400c1",
    "config/adaptive_contact_mpc.json": "b353a2916c45fca10681a91a05bdbd625f6f4017b4cfe8c1bcb8e8fff4f03fd3",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def publication():
    summary = json.loads((FOLDER/"summary.json").read_text())
    with np.load(FOLDER/"development_trace.npz", allow_pickle=False) as archive:
        compact = {name: archive[name].copy() for name in archive.files}
    return summary, compact


def trace(compact, label):
    return {name: compact[f"{label}_{name}"] for name in STATE + (() if label == "reference" else MEASURED)}


def signals(compact, label):
    return heading_signals(compact[f"{label}_quaternion"], compact[f"{label}_com_velocity"],
        compact[f"{label}_wheel_quaternion_wxyz"], compact[f"{label}_wheel_center_velocity_w"],
        speed_threshold_m_s=float(compact[f"{label}_speed_threshold_m_s"]))


def test_exact_three_file_bundle_hashes_allowlist_and_privacy(publication):
    summary, compact = publication
    assert {path.name for path in FOLDER.iterdir()} == {"summary.json", "development_trace.npz", "topdown.png"}
    for name, key, expected in (
        ("summary.json", None, "7d5c3bca9007207c57f34a1fb445f5508405a834c161b7819be1a0e65bdc8f44"),
        ("development_trace.npz", "compact_trace_sha256", "611ea7502676ff5e8c43eba24347108ff136e5a92d91f8f1a1a16c2a7d368e39"),
        ("topdown.png", "topdown_sha256", "51815fde250c7c163b38c6e38437d01b8554f4dd041beb022504990e2eab286b"),
    ):
        assert digest(FOLDER/name) == expected
        if key: assert summary["artifacts"][key] == expected
    expected = {"schema_version", "wheel_order"}
    for label in LABELS:
        expected.update(f"{label}_{name}" for name in STATE+GEOMETRY+SIGNALS
                        + (() if label == "reference" else MEASURED))
    assert set(compact) == expected
    assert compact["schema_version"] == 1 and compact["wheel_order"].tolist() == ["left", "right"]
    for value in compact.values():
        assert not value.dtype.hasobject
        if value.dtype.kind in "biufc": assert np.isfinite(value).all()
        else: assert not any(token in str(value.tolist()) for token in PRIVATE)
    assert not any(token in (FOLDER/"summary.json").read_text() for token in PRIVATE)
    assert not any(name.startswith(("isaac_", "reference_contact_")) for name in compact)
    assert summary["run_order"] == list(LABELS[1:])
    assert summary["reference_spec"] == get_reference()
    assert summary["input_sha256"]["motion_file"] == get_reference()["motion_npz_sha256"]
    assert summary["input_sha256"]["reference_file"] == get_reference()["robot_reference_npz_sha256"]
    assert summary["input_sha256"]["model"] == "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"


@pytest.mark.parametrize("label,mu,adaptive,report_sha,trace_sha,solver_sha", CASES)
def test_native_clock_held_phase_material_bindings_and_search_budget(publication, label, mu, adaptive,
                                                                  report_sha, trace_sha, solver_sha):
    summary, compact = publication
    measured = trace(compact, label); run = summary["runs"][label]
    np.testing.assert_allclose(measured["time_s"], np.arange(1397)*.005, atol=1e-9, rtol=0)
    assert float(measured["physics_dt_s"]) == .005
    assert measured["position"].shape == (1397, 3) and measured["joint_vel"].shape == (1397, 8)
    np.testing.assert_array_equal(measured["joint_names"], JOINT_NAMES)
    held = measured["held_action"]
    assert held.shape == (1397, 8) and np.max(np.abs(held)) <= 1
    np.testing.assert_array_equal(held[0], np.zeros(8))
    holds = held[1:].reshape(349, 4, 8)
    np.testing.assert_array_equal(holds, np.repeat(holds[:, :1], 4, axis=1))
    np.testing.assert_array_equal(held, held.astype(np.float32).astype(float))
    assert run["input_sha256"] == {"report.json": report_sha, "rollout.npz": trace_sha, "solver_log.json": solver_sha}
    assert run["actual_plant_mu"] == mu and run["plant_friction_readback"] == [mu, mu]
    assert run["prediction_mu_scenarios"] == (None if adaptive else [.7])
    assert run["prediction_mu_model_bank"] == ([.3, .7, 1.6] if adaptive else None)
    config = run["effective_cem_config"]
    assert (config["horizon_steps"], config["control_blocks"], config["iterations"]) == (30, 10, 3)
    assert (config["candidates"], config["elite_count"], config["seed"]) == (64, 8, 42)
    assert config["parameterization"] == "warm_start_residual_blocks"
    assert run["effective_cem_config_sha256"] == configuration_sha256(config)
    assert run["reported_settings_sha256"] == configuration_sha256(run["reported_settings"])
    assert run["model_generated_nominal_plan"] == {"enabled": False}
    assert run["control_block_durations_s"] == [.06]*10
    assert run["actuator_identity"]["physics_dt_s"] == .005 and run["actuator_identity"]["control_dt_s"] == .02
    diagnostics = run["solver_diagnostics"]; counts = diagnostics["counts"]
    assert diagnostics["decision_samples"] == diagnostics["timing"]["sample_count"] == 349
    assert counts["candidate_evaluations"] == counts["finite_candidates"] == counts["scenario_evaluations"] == 68054
    assert counts["batch_prediction_calls"] == 2093 and counts["warm_start_used"] == 348
    for name in ("failed_candidates", "failed_scenarios", "failed_batch_prediction_calls", "optimization_failed",
                 "fallback_used", "initial_candidate_generation_failed"):
        assert counts[name] == 0
    timing = diagnostics["timing"]
    assert timing["all_calls_above_20ms"] and not timing["real_time_verified"]
    assert not timing["deadline_enforced"] and not timing["actuation_delay_injected"]
    budget = run["simulation_budget_diagnostic"]
    assert budget["planning_predicted_physics_substeps"] == 68054*30*4 == 8166480
    assert budget["actual_plant_physics_substeps"] == 1396
    assert budget["total_predicted_physics_substeps"] == 8166480+budget["identification_predicted_physics_substeps"]


@pytest.mark.parametrize("label", LABELS)
def test_heading_metrics_posture_and_ground_contact_replay(publication, label):
    summary, compact = publication
    measured, geometry = trace(compact, label), signals(compact, label)
    assert set(geometry) == set(SIGNALS)
    for name, value in geometry.items(): np.testing.assert_array_equal(value, compact[f"{label}_{name}"])
    reference = trace(compact, "reference")
    np.testing.assert_allclose(reference["time_s"], np.arange(349)*.02, atol=1e-12, rtol=0)
    contact = {} if label == "reference" else dict(wheel_ground_contact_count=measured["contact_count"],
        wheel_ground_force_w=measured["contact_force"], wheel_contact_lateral_slip_m_s=measured["contact_slip"])
    replay = summarize_turn(measured["time_s"], measured["position"], geometry, reference["time_s"],
        reference["position"][:, :2], signals(compact, "reference")["speed_m_s"], [1.44, 4.], **contact)
    assert replay == summary["windows"]["core"][label]
    assert replay["coverage_complete"] and replay["samples_in_window"] == (129 if label == "reference" else 513)
    assert not replay["heading_offset_is_mismatch"] and not replay["height_or_flight_failure_gate"]
    if label == "reference":
        assert replay["ground_contact_heading_geometry"] is None and replay["ground_contact_lateral_slip_m_s"] is None
    else:
        assert posture_diagnostic(measured) == summary["posture_diagnostics"][label]
        assert contact_diagnostic(measured, [1.44, 4.]) == summary["contact_diagnostics"][label]
        assert replay["ground_contact_heading_geometry"]["measured_contact_gate_applied"]
        assert replay["ground_contact_heading_geometry"]["threshold_upward_force_n"] == 5.
        assert summary["posture_diagnostics"][label]["max_base_tilt_deg"] < 20.
        assert summary["contact_diagnostics"][label]["full_clip_nonwheel_ground_fraction"] == 0


@pytest.mark.parametrize("suffix,low_spread,no_contact,identification_calls,id_steps,total_s", [
    ("0p7", 288, 59, 867, 3468, 2.4141160579165444),
    ("0p3", 278, 69, 837, 3348, 2.280821303022094),
])
def test_actual_no_switch_selection_causal_clock_contact_skips_and_id_cost(publication, suffix, low_spread,
                                                                       no_contact, identification_calls, id_steps, total_s):
    summary, compact = publication
    run = summary["runs"]["adaptive_mu"+suffix]; diagnostic = run["adaptation_diagnostics"]
    contract = run["adaptation_contract"]; selector = contract["selector"]
    assert contract["model_bank"] == selector["mu_hypotheses"] == [.3, .7, 1.6]
    assert contract["prediction_models_per_solve"] == 1
    assert contract["identification_model_substeps"] == 4 and contract["identification_interval_s"] == .02
    for key in ("plant_mu_used_for_prediction", "future_plant_state_used", "reference_used_for_identification",
                "robustness_certificate", "selected_mu_is_physical_friction_estimate"):
        assert contract[key] is False
    assert selector["window_size"] == 10 and selector["minimum_evidence"] == 5
    assert selector["minimum_spread"] == 1 and selector["maximum_fit_rms"] == 3
    assert selector["switch_improvement_ratio"] == .8 and selector["initial_mu"] == .7
    assert diagnostic["decisions"] == 349 and diagnostic["enabled"]
    assert diagnostic["selected_model_per_decision"] == [.7]*349
    np.testing.assert_array_equal(diagnostic["selection_time_s"], np.arange(349)*.02)
    assert diagnostic["model_switches"] == 0 and diagnostic["selected_model_counts"] == {"0.7": 349}
    statuses = diagnostic["status_per_decision"]
    assert len(statuses) == 349 and statuses[0] == "initial_prior"
    assert diagnostic["status_counts"] == dict(Counter(statuses)) == {
        "initial_prior": 1, "low_spread": low_spread, "no_contact": no_contact, "insufficient_evidence": 1}
    measured = trace(compact, "adaptive_mu"+suffix)
    for k in range(1, 349):
        interval = slice(4*(k-1)+1, 4*k+1)
        contact = bool(np.any(measured["contact_count"][interval] > 0))
        assert (statuses[k] == "no_contact") == (not contact)
        assert not np.any(measured["nonwheel_ground_contact_count"][interval])
    assert identification_calls == (348-no_contact)*3
    assert diagnostic["identification_prediction_calls"] == identification_calls
    assert diagnostic["identification_predicted_physics_substeps"] == id_steps == identification_calls*4
    assert diagnostic["identification_compute_total_s"] == total_s
    assert diagnostic["included_in_controller_compute_s"] and diagnostic["identification_compute_ms_median_p95_max"][0] > 0
    assert not diagnostic["physical_friction_estimate_claimed"] and not diagnostic["robustness_certificate"]
    assert run["simulation_budget_diagnostic"]["identification_predicted_physics_substeps"] == id_steps


@pytest.mark.parametrize("suffix", ("0p7", "0p3"))
def test_paired_applied_action_state_and_contact_arrays_are_bitwise_identical(publication, suffix):
    summary, compact = publication
    for name in STATE+GEOMETRY+SIGNALS+MEASURED:
        np.testing.assert_array_equal(compact[f"fixed_mu{suffix}_{name}"], compact[f"adaptive_mu{suffix}_{name}"])
    assert summary["windows"]["core"]["fixed_mu"+suffix] == summary["windows"]["core"]["adaptive_mu"+suffix]
    fixed, adaptive = [summary["runs"][name+"_mu"+suffix] for name in ("fixed", "adaptive")]
    assert fixed["effective_cem_config"] == adaptive["effective_cem_config"]
    assert fixed["actuator_identity"] == adaptive["actuator_identity"]
    assert fixed["adaptation_diagnostics"] == {"enabled": False, "prediction_mu_scenarios": [.7]}


def test_actual_runtime_sources_pinned_and_match_frozen_current_implementation(publication):
    summary, _ = publication
    for path, expected in {**RUNTIME, **ADAPTIVE_RUNTIME}.items(): assert digest(ROOT/path) == expected
    for label, _, adaptive, *_ in CASES:
        provenance = summary["runs"][label]["reported_plant_source_provenance"]
        assert provenance["mujoco_version"] == "3.8.1"
        rows = provenance["reported_source_hashes"]
        assert len({row["source_name"] for row in rows}) == len(rows)
        sources = {row["source_name"]: row["sha256"] for row in rows}
        for path, expected in RUNTIME.items(): assert sources[Path(path).name] == expected
        for path, expected in ADAPTIVE_RUNTIME.items():
            if adaptive: assert sources[Path(path).name] == expected
            else: assert Path(path).name not in sources
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values())
    for path, expected in summary["analysis_source_sha256"].items(): assert digest(ROOT/"scripts"/path) == expected


def test_pilot_scope_xy_screens_documented_results_and_local_links(publication):
    summary, _ = publication
    assert summary["engine"] == "MuJoCo" and summary["status"] == "development_analysis_only"
    for key in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking",
                "continuous_uncertainty_robustness_certified", "controlled_execution_success_claimed",
                "pooled_success_statistics", "height_or_flight_failure_gate", "heading_offset_is_mismatch",
                "dynamics_stepping", "learned_inference", "identification_bank_is_worst_case_scenario_set"):
        assert summary[key] is False
    for label, mu, *_ in CASES:
        core = summary["windows"]["core"][label]
        assert core["large_xy_exploratory_screen"] is (mu == .7)
        expected = (.170030215967865, .30637733265429923) if mu == .7 else (.07203751633927757, .10922908916997574)
        assert (core["reference_xy_rmse_m"], core["reference_xy_peak_m"]) == expected
    note = NOTE.read_text()
    assert not any(token in note for token in PRIVATE)
    for phrase in ("MuJoCo-only", "zero model switches", "no-switch outcome", "not real-time", "bitwise identical",
                   "0.1700 / 0.3064", "0.0720 / 0.1092", "not demonstrated closed-loop adaptation",
                   "not sim2sim", "not a frozen controller ranking", "not an Isaac qualification", "TVLQR",
                   "v3 diagnostic snapshot was uploaded and hash-verified", "not** part of that snapshot"):
        assert phrase in note
    for target in re.findall(r"\]\(([^)]+)\)", note):
        if not target.startswith(("http:", "https:")): assert (NOTE.parent/target).is_file()
