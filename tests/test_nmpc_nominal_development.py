"""Asset-free development-audit fixtures: no original episode/model or physics."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_nmpc_nominal_development as dev
from controller_metrics import heading_signals
from controller_references import get_reference
from nmpc_tracking_objective import HeightSmoothTrackingObjective


MODEL, TRACE, SOLVER = "1"*64, "2"*64, "3"*64


def report_fixture():
    settings = json.loads((ROOT / "config/nmpc_height_smooth.json").read_text())
    profile = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    result = {"kind": "full_body_height_smooth_cem_nmpc", "engine": "MuJoCo", "status": "completed",
              "reference_id": dev.PRIMARY_REFERENCE_ID, "motion_sha256": get_reference()["motion_npz_sha256"],
              "model_sha256": MODEL, "reference_catalog_sha256": dev.sha256(dev.DEFAULT_CATALOG),
              "rollout_sha256": TRACE, "solver_log_sha256": SOLVER, "plant_mu": .7, "prediction_mu_scenarios": [.7],
              "plant_material_readback": {"compiled_sliding_friction_range": [.7, .7]},
              "completed_policy_steps": 349, "evaluated_policy_steps": 349, "total_reference_policy_steps": 349,
              "duration_s": 6.98, "physics_dt_s": .005, "control_dt_s": .02,
              "compute_ms_median_p95_max": [30., 30., 30.], "compute_over_control_dt_fraction": 1.,
              "real_time_verified": False, "solver_failures": 0, "wall_s": 12.,
              "objective_contract": HeightSmoothTrackingObjective(profile).contract(),
              "packed_state_contract": {"dimension": 111, "qpos_dimension": 15, "qvel_dimension": 14,
                                        "contact_force_history_shape": [3, 9, 3]},
              "contact_slip_definition": "Measured ground-contact-point lateral-speed RMS",
              "effective_cem_config": copy.deepcopy(settings["nmpc"]), "settings": settings,
              "source_sha256": {"/private/archived/scripts/nmpc_shooting.py": dev.LEGACY_CEM_SHA256,
                                "/private/archived/scripts/classical_tracking.py": "4"*64,
                                "/private/archived/scripts/eval_tracking_mujoco.py": "6"*64,
                                "/private/archived/policy_contract.json": "5"*64}}
    result.update({key: False for key in ("learned_policy_used", "tvlqr_used", "old_qr_used", "root_state_prescribed_after_reset",
                  "external_root_forces_used", "prediction_receives_plant_mu", "truncated_smoke", "pose_height_failure_gate")})
    return result


def audited(report):
    return dev.audit_report(report, get_reference(), model_sha256=MODEL, trace_sha256=TRACE, solver_sha256=SOLVER)


def test_legacy_mode_requires_known_reported_cem_identity_and_keeps_horizon_block_knobs():
    report = report_fixture(); result = audited(report)
    assert result["parameterization"]["mode"] == "absolute_blocks"
    assert result["parameterization"]["source"] == "known_legacy_reported_cem_source_sha256"
    assert result["prediction_horizon_s"] == pytest.approx(.3)
    assert result["control_block_durations_s"] == pytest.approx([.06]*5)
    report["effective_cem_config"]["horizon_steps"] = 30
    result = audited(report)
    assert result["prediction_horizon_s"] == pytest.approx(.6)
    assert result["control_block_durations_s"] == pytest.approx([.12]*5)
    assert result["effective_cem_config"] == report["effective_cem_config"]
    report["source_sha256"]["/private/archived/scripts/nmpc_shooting.py"] = "0"*64
    with pytest.raises(ValueError, match="known legacy"):
        audited(report)


def test_explicit_future_residual_and_absolute_modes_preserve_full_decode_declaration():
    report = report_fixture()
    report["source_sha256"]["/private/archived/scripts/nmpc_shooting.py"] = "a"*64
    report["effective_cem_config"]["parameterization"] = "warm_start_residual_blocks"
    declared = {"mode": "warm_start_residual_blocks", "fixed_baseline_per_solve": True,
                "decode": "clip(fixed_baseline + residual_blocks)", "clipping": [-1., 1.]}
    report["action_parameterization"] = declared
    audit = audited(report)
    assert audit["parameterization"]["mode"] == "warm_start_residual_blocks"
    assert audit["parameterization"]["declarations"]["report.action_parameterization"] == declared
    assert audit["parameterization"]["source"] == "explicit_reported_configuration"
    report["action_parameterization"]["mode"] = "absolute_blocks"
    with pytest.raises(ValueError, match="conflicting"):
        audited(report)
    report["effective_cem_config"]["parameterization"] = "absolute_blocks"
    assert audited(report)["parameterization"]["mode"] == "absolute_blocks"


def test_optional_model_plan_identity_and_offline_budget_are_not_lost_in_development_summary():
    report = report_fixture()
    assert audited(report)["model_generated_nominal_plan"] == {"enabled": False}
    metadata = {"enabled": True, "artifact_sha256": "a" * 64,
                "contract": {"kind": "model_generated_nominal_action_plan", "actions_content_sha256": "b" * 64},
                "offline_generation_budget": {"source_predicted_physics_substeps": 8166480}}
    report["model_generated_nominal_plan"] = metadata
    result = audited(report)
    assert result["model_generated_nominal_plan"] == metadata
    result["model_generated_nominal_plan"]["contract"]["kind"] = "modified"
    assert report["model_generated_nominal_plan"]["contract"]["kind"] == "model_generated_nominal_action_plan"
    report["model_generated_nominal_plan"]["contract"]["path"] = "/private/plan.json"
    with pytest.raises(ValueError, match="private/absolute"):
        audited(report)
    del report["model_generated_nominal_plan"]["contract"]["path"]
    report["model_generated_nominal_plan"]["artifact_sha256"] = "invalid"
    with pytest.raises(ValueError, match="artifact_sha256"):
        audited(report)


@pytest.mark.parametrize("path,value", [
    (("status",), "running"), (("engine",), "Isaac/PhysX"), (("plant_mu",), .18),
    (("prediction_mu_scenarios",), [.7, 1.2]), (("model_sha256",), "0"*64),
    (("motion_sha256",), "0"*64), (("rollout_sha256",), "0"*64), (("solver_log_sha256",), "0"*64),
    (("reference_catalog_sha256",), "0"*64), (("completed_policy_steps",), 348),
    (("physics_dt_s",), .01), (("duration_s",), 6.96), (("learned_policy_used",), True),
    (("root_state_prescribed_after_reset",), True), (("external_root_forces_used",), True),
    (("pose_height_failure_gate",), True), (("truncated_smoke",), True),
    (("objective_contract", "absolute_height_target"), True),
    (("objective_contract", "height_regularization", "mode"), "absolute_z"),
    (("objective_contract", "body_names"), ["base_Link"]),
    (("packed_state_contract", "contact_force_history_shape"), [1, 9, 3]),
    (("plant_material_readback", "compiled_sliding_friction_range"), []),
    (("effective_cem_config", "parameterization"), "unknown_decoder"),
    (("settings", "private_checkpoint"), "/private/asset.pt"),
])
def test_fail_closed_incomplete_nonnominal_prescription_hash_and_height_contracts(path, value):
    report = report_fixture(); target = report
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        audited(report)


def canonical(samples, dt):
    t = np.arange(samples)*dt
    position = np.column_stack((t, np.zeros(samples), np.ones(samples)))
    return {"time_s": t, "physics_dt_s": dt, "joint_names": dev.JOINT_NAMES,
            "position": position, "quaternion": np.tile([1., 0., 0., 0.], (samples, 1)),
            "com_velocity": np.tile([1., 0., 0.], (samples, 1)), "angular_velocity": np.zeros((samples, 3)),
            "joint_pos": np.zeros((samples, 8)), "joint_vel": np.zeros((samples, 8)),
            "wheel_pos": position[:, None] + [[0., .14, -.5], [0., -.14, -.5]],
            "contact_count": np.tile([1., 0.], (samples, 1)),
            "contact_force": np.tile([[0., 0., 20.], [0., 0., 0.]], (samples, 1, 1)),
            "contact_slip": np.tile([.2, 0.], (samples, 1))}


def raw_fixture():
    trace = canonical(1397, .005)
    fields = dict(position="root_pos", quaternion="root_quaternion_wxyz", com_velocity="base_com_velocity_w",
                  angular_velocity="base_angular_velocity_w", joint_pos="joint_pos", joint_vel="joint_vel", wheel_pos="wheel_pos",
                  contact_count="wheel_ground_contact_count", contact_force="wheel_ground_contact_force_w_n",
                  contact_slip="wheel_lateral_contact_slip_m_s")
    raw = {source: trace[key].copy() for key, source in fields.items()}
    raw.update(time_s=trace["time_s"], physics_dt_s=np.array(.005), policy_dt_s=np.array(.02),
               action_joint_names=np.array(dev.JOINT_NAMES), body_names=np.array(dev.BODY_NAMES),
               actions=np.zeros((349, 8)), held_action=np.zeros((1397, 8)),
               compute_s=np.full(349, .03),
               reference_frame_for_action=np.r_[0, np.repeat(np.arange(349), 4)],
               action_was_applied=np.r_[False, np.ones(1396, dtype=bool)],
               nonwheel_ground_contact_count=np.zeros(1397), nonwheel_ground_contact_force_norm_n=np.zeros(1397))
    counters = dict(candidate_evaluations=194, finite_candidates=194, failed_candidates=0, scenario_evaluations=194,
                    failed_scenarios=0, batch_prediction_calls=5, failed_batch_prediction_calls=0,
                    optimization_failed=False, fallback_used=False, warm_start_used=False,
                    initial_candidate_generation_failed=False, control_deadline_enforced=False,
                    real_time_verified=False, timing_scope="controller_including_predictor_rollouts")
    solver = [{"frame": k, "reference_frame_for_reward": k, "time_s": (k+1)*.02,
               "action": [0.]*8, "compute_s": .03, "solver": copy.deepcopy(counters)} for k in range(349)]
    return raw, solver


@pytest.mark.parametrize("problem", ["clock_gap", "samples", "missing_contacts", "phase", "held", "int_applied", "nonwheel"])
def test_native_349_decision_1397_sample_contact_and_action_clock_audit(problem):
    raw, solver = raw_fixture()
    if problem == "clock_gap": raw["time_s"][100:] += .005
    elif problem == "samples": raw["time_s"] = raw["time_s"][:-1]
    elif problem == "missing_contacts": del raw["wheel_ground_contact_count"]
    elif problem == "phase": solver[100]["reference_frame_for_reward"] = 101
    elif problem == "held": raw["held_action"][100, 0] = .1
    elif problem == "int_applied": raw["action_was_applied"] = raw["action_was_applied"].astype(int)
    else: raw["nonwheel_ground_contact_count"][3] = -.1
    with pytest.raises((ValueError, KeyError)):
        dev.audit_trace(raw, solver, get_reference())


@pytest.mark.parametrize("problem", ["quantiles", "timing_log", "deadline_fraction", "real_time", "counter", "failure_total", "bad_optional"])
def test_saved_solver_timing_counts_and_optional_diagnostics_fail_closed(problem):
    raw, solver = raw_fixture(); report = report_fixture()
    mode = dev.parameterization_audit(report)
    if problem == "quantiles": report["compute_ms_median_p95_max"][1] = 31.
    elif problem == "timing_log": solver[10]["compute_s"] = .04
    elif problem == "deadline_fraction": report["compute_over_control_dt_fraction"] = 0.
    elif problem == "real_time": solver[10]["solver"]["real_time_verified"] = True
    elif problem == "counter": solver[10]["solver"]["finite_candidates"] = 1
    elif problem == "failure_total": report["solver_failures"] = 1
    else: solver[10]["solver"]["residual_action_decode_clip_fraction"] = 1.01
    with pytest.raises(ValueError):
        dev.solver_diagnostic(report, raw, solver, mode)


def test_missing_null_and_finite_optional_residual_steps_are_not_replaced_by_zero():
    raw, solver = raw_fixture(); report = report_fixture()
    solver[0]["solver"]["initial_baseline_block_projection_max_abs_error"] = None
    solver[1]["solver"]["initial_baseline_block_projection_max_abs_error"] = .4
    result = dev.solver_diagnostic(report, raw, solver, dev.parameterization_audit(report))
    item = result["optional_residual_diagnostics"]["initial_baseline_block_projection_max_abs_error"]
    assert item["present_steps"] == 2 and item["null_steps"] == 1 and item["finite_value_steps"] == 1
    assert item["missing_steps"] == 347 and item["mean"] == pytest.approx(.4)
    assert item["median_p95_max"] == pytest.approx([.4]*3)


def geometry(trace):
    n = len(trace["time_s"])
    q = np.tile([1., 0., 0., 0.], (n, 2, 1)); velocity = np.repeat(trace["com_velocity"][:, None], 2, axis=1)
    return {"signals": heading_signals(trace["quaternion"], trace["com_velocity"], q, velocity),
            "wheel_quaternion_wxyz": q, "wheel_center_velocity_w": velocity,
            "audit": {"dynamics_stepping": False, "measured_contact_replaced": False}}


def fixture_arguments(tmp_path, monkeypatch):
    reference = canonical(349, .02)
    motion = {"time_s": reference["time_s"], "fps": np.array([50.]), "joint_names": np.array(dev.JOINT_NAMES),
              "body_names": np.array(["base_Link", "wheel_L_Link", "wheel_R_Link"]),
              "body_pos_w": np.concatenate((reference["position"][:, None], reference["wheel_pos"]), axis=1),
              "body_quat_w": np.repeat(reference["quaternion"][:, None], 3, axis=1),
              "body_lin_vel_w": np.repeat(reference["com_velocity"][:, None], 3, axis=1),
              "body_ang_vel_w": np.repeat(reference["angular_velocity"][:, None], 3, axis=1),
              "joint_pos": reference["joint_pos"], "joint_vel": reference["joint_vel"]}
    motion_file, reference_file, model = [tmp_path / name for name in ("motion.npz", "reference.npz", "model.xml")]
    np.savez(motion_file, **motion); np.savez(reference_file, time_s=reference["time_s"], root_target_position_m=reference["position"])
    model.write_bytes(b"synthetic provenance token; actual model loader is mocked")
    spec = get_reference(); spec.update(motion_npz_sha256=dev.sha256(motion_file), robot_reference_npz_sha256=dev.sha256(reference_file))
    monkeypatch.setattr(dev, "verify_reference_pair", lambda *args: copy.deepcopy(spec))
    @contextmanager
    def loader(*args):
        yield None, {"dynamics_stepping": False, "source_helpers_sha256": {"fixture.py": "f"*64}}
    monkeypatch.setattr(dev, "source_model", loader)
    class FK:
        def __init__(self, model): pass
        def reconstruct(self, trace): return geometry(trace)
    monkeypatch.setattr(dev, "WheelKinematics", FK)
    def prohibited(*args, **kwargs): pytest.fail("Development analyzer must never step physics/infer actions")
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_step=prohibited, mj_step1=prohibited, mj_step2=prohibited, mj_forward=prohibited))
    runs = []
    for label in ("baseline", "H30", "residual"):
        directory = tmp_path / label; directory.mkdir(); raw, solver = raw_fixture()
        np.savez(directory / "rollout.npz", **raw)
        (directory / "solver_log.json").write_text(json.dumps(solver))
        report = report_fixture()
        report.update(model_sha256=dev.sha256(model), motion_sha256=spec["motion_npz_sha256"],
                      rollout_sha256=dev.sha256(directory / "rollout.npz"), solver_log_sha256=dev.sha256(directory / "solver_log.json"))
        if label == "H30": report["effective_cem_config"]["horizon_steps"] = 30
        if label == "residual":
            report["effective_cem_config"]["parameterization"] = "warm_start_residual_blocks"
            report["source_sha256"]["/private/archived/scripts/nmpc_shooting.py"] = "a"*64
            for decision in solver:
                decision["solver"].update(action_parameterization="warm_start_residual_blocks",
                    residual_baseline_fixed_per_solve=True, initial_baseline_block_projection_max_abs_error=.12,
                    residual_parameter_clip_fraction=.1, residual_action_decode_clip_fraction=.2,
                    residual_selected_parameter_linf=.03)
            (directory / "solver_log.json").write_text(json.dumps(solver))
            report["solver_log_sha256"] = dev.sha256(directory / "solver_log.json")
        (directory / "report.json").write_text(json.dumps(report)); runs.append(f"{label}={directory}")
    return SimpleNamespace(motion_file=motion_file, reference_file=reference_file, model=model, source_root=tmp_path,
                           run=runs, output_dir=tmp_path / "analysis")


def test_end_to_end_asset_free_four_panels_named_compact_provenance_and_no_step(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    args = fixture_arguments(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = dev.run(args)
    assert result["run_order"] == ["baseline", "H30", "residual"]
    assert result["status"] == "development_analysis_only" and result["engine"] == "MuJoCo"
    assert not result["sim2sim_comparison"] and not result["frozen_benchmark"] and not result["eligible_for_controller_ranking"]
    assert not result["dynamics_stepping"] and not result["learned_inference"]
    assert not result["height_or_flight_failure_gate"] and not result["heading_offset_is_mismatch"]
    assert result["runs"]["baseline"]["parameterization"]["mode"] == "absolute_blocks"
    assert result["runs"]["residual"]["parameterization"]["mode"] == "warm_start_residual_blocks"
    assert result["runs"]["H30"]["control_block_durations_s"] == pytest.approx([.12]*5)
    diagnostics = result["runs"]["residual"]["solver_diagnostics"]
    assert diagnostics["timing"]["compute_ms_median_p95_max"] == pytest.approx([30.]*3)
    assert diagnostics["timing"]["all_calls_above_20ms"] and not diagnostics["timing"]["actuation_delay_injected"]
    assert diagnostics["counts"]["candidate_evaluations"] == 194*349
    assert diagnostics["optional_residual_diagnostics"]["residual_parameter_clip_fraction"]["mean"] == pytest.approx(.1)
    absent = result["runs"]["baseline"]["solver_diagnostics"]["optional_residual_diagnostics"]["residual_parameter_clip_fraction"]
    assert absent["missing_steps"] == 349 and absent["mean"] is None
    assert result["windows"]["core"]["baseline"]["ground_contact_lateral_slip_m_s"] == {"left": pytest.approx(.2), "right": None}
    assert result["posture_diagnostics"]["baseline"]["max_base_tilt_deg"] == pytest.approx(0.)
    text = (args.output_dir / "summary.json").read_text()
    assert str(tmp_path) not in text and "/private/" not in text
    compact = dev.load_arrays(args.output_dir / "development_trace.npz")
    assert "baseline_joint_vel" in compact and "residual_contact_slip" in compact and "H30_held_action" in compact
    assert "reference_contact_count" not in compact
    assert not any(value.dtype.hasobject for value in compact.values())
    assert dev.sha256(args.output_dir / "development_trace.npz") == result["artifacts"]["compact_trace_sha256"]
    assert (args.output_dir / "topdown.png").stat().st_size > 1000
    for path, contents in before.items(): assert path.read_bytes() == contents
    with pytest.raises(ValueError, match="Preserving"):
        dev.run(args)


@pytest.mark.parametrize("change", ["objective", "actuator"])
def test_runs_cannot_silently_change_objective_or_actuator_contract(tmp_path, monkeypatch, change):
    args = fixture_arguments(tmp_path, monkeypatch)
    path = tmp_path / "H30/report.json"; report = dev.read_json(path)
    if change == "objective": report["objective_contract"]["weights"]["drift_speed"] += .1
    else: report["source_sha256"]["/private/archived/policy_contract.json"] = "e"*64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="objective|actuator"):
        dev.run(args)
    assert not args.output_dir.exists()


def test_run_labels_and_directories_cannot_be_duplicated_or_encode_private_paths(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    assert list(dev.parse_runs([f"base={a}", f"H30={b}"])) == ["base", "H30"]
    for values in ([f"base={a}"], [f"base={a}", f"base={b}"], [f"base={a}", f"H30={a}"],
                   [f"reference={a}", f"H30={b}"], [f"/private/label={a}", f"H30={b}"]):
        with pytest.raises(ValueError): dev.parse_runs(values)


def test_shared_world_four_panel_2x2_bounds_include_wheels_and_bottom_legend(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    traces = {name: canonical(349, .02) for name in ("reference", "base", "H30", "residual")}
    for trace in traces.values(): trace["wheel_pos"][:] = [[-5., -4., .1], [8., 7., .1]]
    compact = dev.compact_arrays(traces, {name: geometry(trace) for name, trace in traces.items()})
    summary = {"reference_spec": get_reference(), "run_order": ["base", "H30", "residual"]}
    captured = []; close = plt.close
    def inspect_then_close(figure=None):
        if hasattr(figure, "axes"):
            figure.canvas.draw()
            captured.extend((axis.get_xlim(), axis.get_ylim(), axis.get_aspect(), axis.get_subplotspec().get_gridspec().nrows,
                             axis.get_subplotspec().get_gridspec().ncols) for axis in figure.axes)
            legend = figure.legends[0].get_window_extent()
            xlabel_bottom = min(axis.xaxis.label.get_window_extent().y0 for axis in figure.axes)
            assert legend.y1 < xlabel_bottom
            assert "DEVELOPMENT" in figure._suptitle.get_text() and "not sim2sim / not frozen" in figure._suptitle.get_text()
        close(figure)
    monkeypatch.setattr(plt, "close", inspect_then_close)
    dev.render_topdown(compact, summary, tmp_path / "four.png")
    assert len(captured) == 4 and all(item == captured[0] for item in captured)
    xlim, ylim, aspect, rows, columns = captured[0]
    assert xlim[0] < -5 and xlim[1] > 8 and ylim[0] < -4 and ylim[1] > 7
    assert aspect == 1. and (rows, columns) == (2, 2)
