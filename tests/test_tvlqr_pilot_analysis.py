"""Portable TVLQR analysis fixtures only; simulator stepping is prohibited."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_tvlqr_pilot as analysis
import analyze_nmpc_nominal_development as nominal
from controller_references import get_reference
from design_tvlqr_nominal import design_objective_contract
from test_nmpc_nominal_development import fixture_arguments, raw_fixture, report_fixture, geometry, MODEL, TRACE, SOLVER


def tvlqr_report():
    report = report_fixture()
    report.update(kind=analysis.TVLQR_KIND, physics_samples=1397, feedback_log_sha256=SOLVER,
        nominal_prediction_mu=.7, design_sha256="a"*64, design_report_sha256="b"*64,
        actuator_contract_sha256="5"*64, design_cost_equals_15_term_objective=False,
        design_objective_contract=design_objective_contract(), clipped_action_component_fraction=0.,
        offline_design_budget={"attempted_transition_calls": 349*73, "completed_transition_calls": 349*73,
                               "predicted_physics_substeps": 349*73*4, "wall_s": 3.},
        donor_identity={"archive_sha256": {"report.json": "c"*64, "rollout.npz": "d"*64, "solver_log.json": "e"*64},
                        "learned_policy_used": False, "source_sha256": report["source_sha256"].copy(),
                        "generation_budget": {"scenario_evaluations": 67706, "predicted_physics_substeps": 4062360,
                                              "plant_physics_substeps": 1396, "wall_s": 12.}})
    report.update({name: False for name in ("height_or_flight_failure_gate", "heading_offset_is_mismatch",
        "robustness_certificate", "sim2sim_comparison", "action_replay")})
    for name in ("run_tvlqr_tracking.py", "tvlqr_core.py", "design_tvlqr_nominal.py"):
        report["source_sha256"]["/private/source/scripts/"+name] = "f"*64
    return report


def audited(report):
    return analysis.audit_tvlqr_report(report, get_reference(), model_sha256=MODEL,
                                      trace_sha256=TRACE, log_sha256=SOLVER)


def test_report_keeps_unmatched_design_and_both_offline_budgets_without_private_paths():
    report = tvlqr_report(); before = copy.deepcopy(report)
    result = audited(report)
    assert result["design_cost_equals_15_term_objective"] is False
    assert result["offline_design_budget"]["predicted_physics_substeps"] == 101908
    assert result["donor_identity"]["generation_budget"]["plant_physics_substeps"] == 1396
    assert "/private/" not in json.dumps(result) and report == before


@pytest.mark.parametrize("field,value", [("kind", "full_body_height_smooth_cem_nmpc"),
    ("status", "running"), ("engine", "Isaac/PhysX"), ("plant_mu", .3), ("completed_policy_steps", 348),
    ("physics_samples", 1396), ("rollout_sha256", "0"*64), ("feedback_log_sha256", "0"*64),
    ("design_cost_equals_15_term_objective", True), ("action_replay", True), ("learned_policy_used", True),
    ("root_state_prescribed_after_reset", True), ("height_or_flight_failure_gate", True),
    ("real_time_verified", True), ("design_sha256", None), ("offline_design_budget", None)])
def test_invalid_controller_identity_incomplete_hash_or_false_claim_fails(field, value):
    report = tvlqr_report(); report[field] = value
    with pytest.raises(ValueError): audited(report)


def prepared_raw():
    raw, log = raw_fixture()
    raw.update(qpos=np.column_stack((raw["root_pos"], raw["root_quaternion_wxyz"], raw["joint_pos"])),
        qvel=np.column_stack((np.ones(1397), np.zeros((1397, 5)), raw["joint_vel"])),
        qpos_joint_names=np.array(analysis.JOINT_NAMES), stage_cost=np.full(349, .5))
    for row in log: row["objective"] = {"stage_cost": .5}
    return raw, log


def feedback_fixture():
    raw, log = prepared_raw()
    for row in log:
        del row["solver"]
        row["feedback"] = {"raw_action": [0.]*8, "clipped_action_components": 0,
                           "online_optimization": False, "action_replay": False}
    return raw, log


@pytest.mark.parametrize("problem", ["raw_action", "clip_counter", "replay", "timing", "qpos", "qvel", "joint_names", "phase", "clock"])
def test_trace_and_real_feedback_log_reject_modified_commands_states_or_clocks(problem):
    raw, log = feedback_fixture(); report = tvlqr_report()
    if problem == "raw_action": log[4]["feedback"]["raw_action"][0] = .2
    if problem == "clip_counter": log[4]["feedback"]["clipped_action_components"] = 1
    if problem == "replay": log[4]["feedback"]["action_replay"] = True
    if problem == "timing": log[4]["compute_s"] += .01
    if problem == "qpos": raw["qpos"][2, 0] += .2
    if problem == "qvel": raw["qvel"][3, -1] = np.nan
    if problem == "joint_names": raw["qpos_joint_names"][0] = "wrong_joint"
    if problem == "phase": log[4]["reference_frame_for_reward"] += 1
    if problem == "clock": raw["time_s"][20:] += .005
    with pytest.raises(ValueError):
        trace = analysis.audit_trace(raw, log, get_reference())
        analysis.audit_physical_state(raw, trace)
        analysis.feedback_diagnostic(report, raw, log)


def arguments(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path, monkeypatch)
    for name in ("verify_reference_pair", "source_model", "WheelKinematics"):
        monkeypatch.setattr(analysis, name, getattr(nominal, name))
    args.run = args.run[:2]
    donor = Path(args.run[0].partition("=")[2]); feedback = Path(args.run[1].partition("=")[2])
    raw, log = prepared_raw(); np.savez(donor/"rollout.npz", **raw)
    (donor/"solver_log.json").write_text(json.dumps(log))
    report = analysis.read_json(donor/"report.json")
    report.update(rollout_sha256=analysis.sha256(donor/"rollout.npz"), solver_log_sha256=analysis.sha256(donor/"solver_log.json"), total_stage_cost=174.5)
    (donor/"report.json").write_text(json.dumps(report))
    report = tvlqr_report()
    raw, log = feedback_fixture(); np.savez(feedback/"rollout.npz", **raw)
    (feedback/"feedback_log.json").write_text(json.dumps(log))
    report.update(model_sha256=analysis.sha256(args.model), motion_sha256=analysis.sha256(args.motion_file),
        rollout_sha256=analysis.sha256(feedback/"rollout.npz"), feedback_log_sha256=analysis.sha256(feedback/"feedback_log.json"), total_stage_cost=174.5)
    report["donor_identity"]["archive_sha256"] = {name: analysis.sha256(donor/name) for name in ("report.json", "rollout.npz", "solver_log.json")}
    (feedback/"report.json").write_text(json.dumps(report))
    return args


def test_end_to_end_native_topdown_has_shared_geometry_without_raw_model_design_or_actions(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    args = arguments(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = analysis.run(args)
    assert result["engine"] == "MuJoCo" and result["status"] == "development_analysis_only"
    assert result["input_sha256"] == {name: analysis.sha256(getattr(args, name))
                                      for name in ("motion_file", "reference_file", "model")}
    assert result["reference_catalog_sha256"] == analysis.sha256(analysis.DEFAULT_CATALOG)
    for name in ("sim2sim_comparison", "frozen_benchmark", "eligible_for_controller_ranking", "matched_design_cost",
                 "independent_trajectory_design", "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping"):
        assert result[name] is False
    payload = (args.output_dir/"summary.json").read_text()
    assert "/private/" not in payload and str(tmp_path) not in payload
    data = analysis.load_arrays(args.output_dir/"development_trace.npz")
    assert data["H30_time_s"].shape == (1397,)
    assert "H30_contact_count" in data and "reference_contact_count" not in data
    assert not any(name.endswith(("qpos", "qvel", "actions", "held_action", "K", "feed")) for name in data)
    for name, artifact in result["artifacts"].items():
        assert analysis.sha256(args.output_dir/artifact["file"]) == artifact["sha256"]
    assert (args.output_dir/"topdown.png").stat().st_size > 1000
    for path, contents in before.items(): assert path.read_bytes() == contents


def test_public_projection_explicitly_rejects_new_unreviewed_raw_or_derived_fields():
    raw, log = prepared_raw()
    trace = analysis.audit_trace(raw, log, get_reference())
    trace.update(qpos=raw["qpos"], private_path=np.array("/private/model"), K=np.ones((2, 2)))
    reconstruction = geometry(trace)
    reconstruction["signals"].update(private_path=np.array("/private/plan"), raw_actions=np.ones(8))
    arrays = analysis.public_compact({"test": trace}, {"test": reconstruction})
    assert not any(key.endswith(("qpos", "private_path", "K", "raw_actions", "held_action")) for key in arrays)
    assert "test_wheel_quaternion_wxyz" in arrays and "test_root_course_to_wheel_gamma_rad" in arrays


@pytest.mark.parametrize("problem", ["donor_binding", "donor_budget", "actuator", "evaluation_cost", "stage_cost"])
def test_run_rejects_unmatched_nominal_budget_actuator_or_common_evaluation(tmp_path, monkeypatch, problem):
    args = arguments(tmp_path, monkeypatch)
    path = tmp_path/"H30/report.json"; report = analysis.read_json(path)
    if problem == "donor_binding": report["donor_identity"]["archive_sha256"]["rollout.npz"] = "0"*64
    if problem == "donor_budget": report["donor_identity"]["generation_budget"]["predicted_physics_substeps"] += 4
    if problem == "actuator": report["actuator_contract_sha256"] = "0"*64
    if problem == "evaluation_cost": report["objective_contract"]["weights"]["drift_speed"] += .1
    if problem == "stage_cost": report["total_stage_cost"] += 1.
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): analysis.run(args)
    assert not args.output_dir.exists()


def test_parser_and_existing_output_preservation(tmp_path):
    parsed = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x", "--source-root", "s",
        "--output-dir", "o", "--run", "donor=d", "--run", "tvlqr=t"])
    assert parsed.motion_file == Path("m")
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.run(SimpleNamespace(output_dir=tmp_path))
