"""Asset-free geometry replay of the published fixed MPPI pilot.

MPPI importance/ESS summaries and rollout budgets remain source-bound logged
aggregates: private candidate costs, weights and actions are not published.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from analyze_controller_pair import posture_diagnostic
from analyze_nmpc_nominal_development import configuration_sha256, contact_diagnostic
from controller_metrics import heading_signals, summarize_turn
from controller_references import DEFAULT_CATALOG, get_reference

FOLDER=ROOT/"results/mppi_primary_mu0p7_pilot_20261016_v1"
LABELS=("reference","CEM","MPPI")
STATE=("time_s","physics_dt_s","joint_names","position","quaternion","com_velocity","angular_velocity","wheel_pos")
MEASURED=("contact_count","contact_force","contact_slip","nonwheel_ground_contact_count","nonwheel_ground_contact_force_norm_n")
GEOMETRY=("wheel_quaternion_wxyz","wheel_center_velocity_w")
SIGNALS=("speed_m_s","wheel_center_speed_m_s","course_rad","body_heading_rad","wheel_rolling_plane_heading_rad",
    "root_directed_beta_rad","root_course_to_wheel_gamma_rad","wheel_center_course_to_wheel_gamma_rad",
    "root_beta_valid","root_course_to_wheel_valid","wheel_center_course_to_wheel_valid","speed_threshold_m_s")
PRIVATE=("/home/","/data/","/private/","/tmp/","file://","zengxiong","14.103.233.29","117.50.226.213")
ARTIFACT_HASHES={
    "summary.json":"2f7f1b7779bc6f452bae7f842846024c2785c75764c07fd1b5cc8fc73b23afc8",
    "development_trace.npz":"cb547fc3a8fe0fd15da7c12e35988ad664aa75bdad988ce33c7016ac22c9a27f",
    "topdown.png":"92d9a64505290c6d2a495805f07b6b05f3bd4ede9224579d225b006cd77f4128"}
ARCHIVE_HASHES={
    "CEM":{"report.json":"3e10ea13e84d4d2da47694cae20b5147582bee837a8d4d02c1238e1635700e26",
        "rollout.npz":"9b89a75b8a09318d215d43d16e89b131731a07fc9d26eb2d86e8b061cb575d2b",
        "solver_log.json":"108ccee2107b87f243b9741bb0f65ca958a05dac18bfc6c47422c3fdc4e297cc"},
    "MPPI":{"report.json":"1a134aa5c43414876190e75f9418a5d1409dc62dea36a41081d8a6d553a64ef9",
        "rollout.npz":"f898e7d79e54edfc5a856047f5e6ddfd456fb3230332a44620786c199762144a",
        "solver_log.json":"2fb44077dd45ed9589d29546a6901861b2e1887e5b0ee8dbde9dd206fa41f88c"}}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def publication():
    summary=json.loads((FOLDER/"summary.json").read_text())
    with np.load(FOLDER/"development_trace.npz",allow_pickle=False) as archive:
        data={key:archive[key].copy() for key in archive.files}
    return summary,data


def trace(data,label):
    return {key:data[f"{label}_{key}"] for key in STATE+(() if label=="reference" else MEASURED)}


def signals(data,label):
    return heading_signals(data[f"{label}_quaternion"],data[f"{label}_com_velocity"],
        data[f"{label}_wheel_quaternion_wxyz"],data[f"{label}_wheel_center_velocity_w"],
        speed_threshold_m_s=float(data[f"{label}_speed_threshold_m_s"]))


def assert_tree(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:assert_tree(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):assert_tree(a,b)
    elif type(expected) in (float,int):assert actual==pytest.approx(expected,rel=1e-11,abs=1e-10)
    else:assert actual==expected


def test_exact_three_artifacts_strict_geometry_allowlist_and_privacy(publication):
    summary,data=publication
    assert set(ARTIFACT_HASHES)=={"summary.json","development_trace.npz","topdown.png"}
    assert {path.name for path in FOLDER.iterdir()}==set(ARTIFACT_HASHES)
    for name,digest in ARTIFACT_HASHES.items():assert sha(FOLDER/name)==digest
    for artifact in summary["artifacts"].values():assert ARTIFACT_HASHES[artifact["file"]]==artifact["sha256"]
    assert (FOLDER/"topdown.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    expected={"schema_version","wheel_order"}
    for label in LABELS:
        expected.update(f"{label}_{key}" for key in STATE+GEOMETRY+SIGNALS+(() if label=="reference" else MEASURED))
    assert set(data)==expected and data["wheel_order"].tolist()==["left","right"]
    assert not any(name.endswith(("qpos","qvel","actions","held_action","weights","plans","candidate_costs")) for name in data)
    assert not any(token in (FOLDER/"summary.json").read_text() for token in PRIVATE)
    for array in data.values():
        assert not array.dtype.hasobject
        if array.dtype.kind in "biufc":assert np.isfinite(array).all()
        else:assert not any(token in str(array.tolist()) for token in PRIVATE)


def test_model_reference_sources_sampler_and_actual_native_identity(publication):
    summary,_=publication;spec=get_reference()
    assert summary["reference_spec"]==spec and summary["reference_catalog_sha256"]==sha(DEFAULT_CATALOG)
    assert summary["input_sha256"]=={"motion_file":spec["motion_npz_sha256"],"reference_file":spec["robot_reference_npz_sha256"],
        "model":"bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"}
    for name,digest in summary["analysis_source_sha256"].items():assert sha(ROOT/"scripts"/name)==digest
    cem,mppi=summary["runs"]["CEM"],summary["runs"]["MPPI"]
    assert {label:summary["runs"][label]["input_sha256"] for label in LABELS[1:]}==ARCHIVE_HASHES
    assert cem["controller_kind"]=="full_body_height_smooth_cem_nmpc"
    assert mppi["controller_kind"]=="full_body_height_smooth_mppi"
    assert mppi["actuator_identity"]==cem["actuator_identity"]
    assert mppi["shared_prediction_plant_reward_source_sha256"]==cem["shared_prediction_plant_reward_source_sha256"]
    config=mppi["effective_mppi_config"]
    assert config=={"horizon_steps":30,"control_blocks":10,"samples":63,"iterations":3,"temperature":1.,
        "noise_std":[.04]*6+[.35]*2,"seed":42,"reference_convention":"held_pre_action","fallback_action":"previous"}
    assert mppi["effective_mppi_config_sha256"]==configuration_sha256(config)
    assert "effective_cem_config" not in mppi
    native=mppi["native_prediction_backend"]
    assert native["production_cem_wired"] is False and native["objective_production_cem_wired"] is False
    assert native["production_mppi_wired"] is True and native["objective_production_mppi_wired"] is True
    assert native["runtime_mppi_population_path"]=="callable predictor.rollout_batch"
    assert set(native["worker_identities"])=={"0.7"}
    assert all(re.fullmatch(r"[0-9a-f]{64}",value) for value in native["worker_identities"]["0.7"].values())
    sources={item["source_name"]:item["sha256"] for item in mppi["reported_source_provenance"]["reported_source_hashes"]}
    for name in ("run_mppi_tracking.py","mppi_shooting.py"):
        assert sources[name]==sha(ROOT/"scripts"/name)


def test_complete_native_clocks_and_heading_geometry_replay_without_assets(publication):
    _,data=publication
    for label in LABELS:
        n,dt=(349,.02) if label=="reference" else (1397,.005)
        np.testing.assert_allclose(data[f"{label}_time_s"],np.arange(n)*dt,atol=1e-10,rtol=0)
        assert float(data[f"{label}_physics_dt_s"])==dt
        assert data[f"{label}_position"].shape==(n,3) and data[f"{label}_wheel_pos"].shape==(n,2,3)
        replay=signals(data,label);assert set(replay)==set(SIGNALS)
        for key,value in replay.items():
            actual=data[f"{label}_{key}"]
            if actual.dtype.kind=="b":np.testing.assert_array_equal(actual,value)
            else:np.testing.assert_allclose(actual,value,atol=1e-12,rtol=1e-12)
    assert "reference_contact_count" not in data and "reference_contact_force" not in data


def test_core_xy_speed_and_grounded_sideslip_replay_from_public_geometry(publication):
    summary,data=publication;reference=trace(data,"reference")
    for label in LABELS:
        measured=trace(data,label)
        contact={} if label=="reference" else dict(wheel_ground_contact_count=measured["contact_count"],
            wheel_ground_force_w=measured["contact_force"],wheel_contact_lateral_slip_m_s=measured["contact_slip"])
        replay=summarize_turn(measured["time_s"],measured["position"],signals(data,label),reference["time_s"],
            reference["position"][:,:2],signals(data,"reference")["speed_m_s"],summary["reference_spec"]["evaluation_core_interval_s"],**contact)
        assert_tree(replay,summary["windows"]["core"][label])
    for label in LABELS[1:]:
        core=summary["windows"]["core"][label]
        assert core["large_xy_exploratory_screen"]==(core["reference_xy_rmse_m"]>.15 or core["reference_xy_peak_m"]>.30)
    core=summary["windows"]["core"]
    assert core["MPPI"]["reference_xy_rmse_m"]==pytest.approx(.666229907206021)
    assert core["MPPI"]["reference_xy_peak_m"]==pytest.approx(1.4216346109507263)
    assert core["MPPI"]["mean_speed_retention_ratio"]==pytest.approx(.6047921248248701)
    assert core["CEM"]["reference_xy_rmse_m"]==pytest.approx(.170030215967865)
    assert core["CEM"]["reference_xy_peak_m"]==pytest.approx(.30637733265429923)


def test_source_bound_rollout_budgets_and_importance_diagnostics_not_per_sample_replay(publication):
    summary,_=publication
    for label in LABELS[1:]:
        run=summary["runs"][label];solver=run["solver_diagnostics"];counts=solver["counts"]
        assert solver["decision_samples"]==349
        assert counts["scenario_evaluations"]==counts["candidate_evaluations"]==68054
        assert counts["finite_candidates"]==68054 and counts["failed_candidates"]==0
        assert counts["optimization_failed"]==counts["fallback_used"]==0
        assert counts["batch_prediction_calls"]==({"CEM":2093,"MPPI":3140}[label])
        assert run["simulation_budget_diagnostic"]=={"planning_predicted_physics_substeps":68054*30*4,
            "actual_plant_physics_substeps":1396,
            "scope":"Requested budget from saved counters/horizon; failed requests are not proof of completed native steps"}
        for digest in run["input_sha256"].values():assert re.fullmatch(r"[0-9a-f]{64}",digest)
    diagnostic=summary["runs"]["MPPI"]["solver_diagnostics"]
    weighted=diagnostic["weighted_update_diagnostics"]
    assert diagnostic["all_population_failed_iterations"]==0
    for name,metric in weighted.items():
        assert metric["samples"]==349*3 and metric["null_or_unavailable_samples"]==0
        assert len(metric["median_p95_max"])==3
    assert 1<=weighted["ess"]["minimum"]<=weighted["ess"]["median_p95_max"][-1]<=63+1e-10
    assert weighted["ess"]["median_p95_max"][0]==pytest.approx(21.63172995688841)
    assert 1/63<=weighted["max_weight"]["minimum"]<=weighted["max_weight"]["median_p95_max"][-1]<=1
    assert "not independently replayed" in diagnostic["weighted_diagnostics_scope"]
    assert diagnostic["selected_plan_rule"]=="Final re-evaluated MPPI weighted mean, not best raw-cost population member"


def test_posture_contacts_and_unmatched_objective_do_not_claim_sim2sim_or_ranking(publication):
    summary,data=publication
    assert summary["run_order"]==["CEM","MPPI"] and summary["engine"]=="MuJoCo" and summary["fixed_mu"]==.7
    assert summary["kind"]=="mujoco_only_mppi_fixed_pilot"
    assert summary["common_15_term_evaluation_objective"] is True
    for name in ("sim2sim_comparison","frozen_benchmark","eligible_for_controller_ranking","matched_optimization_objective",
        "robustness_certificate","controlled_execution_success_claimed","height_or_flight_failure_gate",
        "heading_offset_is_mismatch","dynamics_stepping","learned_inference"):
        assert summary[name] is False
    for label in LABELS[1:]:
        measured=trace(data,label);run=summary["runs"][label]
        assert_tree(posture_diagnostic(measured),summary["posture_diagnostics"][label])
        assert_tree(contact_diagnostic(measured,summary["reference_spec"]["evaluation_core_interval_s"]),summary["contact_diagnostics"][label])
        timing=run["solver_diagnostics"]["timing"]
        assert timing["real_time_verified"] is False and timing["deadline_enforced"] is False
        assert timing["actuation_delay_injected"] is False
    assert summary["posture_diagnostics"]["MPPI"]["max_base_tilt_deg"]==pytest.approx(111.58857195684287)
    assert summary["contact_diagnostics"]["MPPI"]["first_nonwheel_ground_time_s"]==pytest.approx(2.395)
    contract=summary["runs"]["MPPI"]["mppi_algorithm_contract"]
    assert contract["optimizer"]=="fixed_covariance_latent_residual_mppi"
    assert contract["task_rollout_score_modified"] is False
    assert contract["implicit_regularization"]=="Gaussian-base KL-to-initial-baseline prior; not identical overall optimization objective to CEM"
    for name in ("covariance_fitting","temperature_adaptation","elite_selection","plant_friction_oracle",
        "controller_added_terminal_value","predictor_terminal_cost_hook_used_on_scalar_path"):
        assert contract[name] is False
