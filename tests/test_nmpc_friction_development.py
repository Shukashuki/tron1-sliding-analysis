"""Bounded offline development-grid tests: no physics, inference or network."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_nmpc_friction_development as analysis
import analyze_nmpc_nominal_development as nominal
from controller_references import get_reference
from test_nmpc_nominal_development import fixture_arguments, raw_fixture, report_fixture


def write_case(directory, mu, *, spec=None, model_sha256=None):
    spec = copy.deepcopy(spec or get_reference()); directory.mkdir(exist_ok=True)
    report = report_fixture()
    report.update(role=spec["role"], stratum=spec["stratum"], reference_spec=spec,
        reference_id=spec["reference_id"], motion_sha256=spec["motion_npz_sha256"],
        plant_mu=mu, effective_cem_config=copy.deepcopy(analysis.PILOT_CEM),
        model_generated_nominal_plan={"enabled": False}, prediction_backend="native", robustness_certificate=False,
        settings=json.loads((ROOT/"config/nmpc_height_smooth_residual.json").read_text()), total_stage_cost=174.5)
    report["plant_material_readback"]["compiled_sliding_friction_range"] = [mu, mu]
    if model_sha256: report["model_sha256"] = model_sha256
    for name, value in analysis.PILOT_SOURCE_SHA256.items():
        keys = [key for key in report["source_sha256"] if key.endswith("/"+name)]
        report["source_sha256"][keys[0] if keys else "/private/archived/"+name] = value
    report["source_sha256"].update({
        "/private/archived/policy_contract.json": "6e105cbd6efe7d35117a895fa097e8ec64b6ac822a9de07c00acfaef4d09642f",
        "/private/archived/config/nmpc_height_smooth_residual.json": "643262e037d766e924c5ffa165385f2f9a85d5af0257192f222d7106e2685266",
        "/private/archived/config/car_drift_height_smooth_task.json": "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6"})
    for name in analysis.FK_HELPERS:
        if not any(key.endswith("/"+name) for key in report["source_sha256"]):
            report["source_sha256"]["/private/archived/"+name] = "f"*64
    report["native_prediction_backend"] = {"production_cem_wired": True, "plant_mu_used_for_prediction": False,
        "prediction_mu_scenarios": [.7], "learned_initialization": False, "terminal_cost": None,
        "workers": {"0.7": {"native_library_sha256": analysis.PILOT_NATIVE_LIBRARY_SHA256,
            "compiler_command": ["c++", "/private/source.cpp"], "worker_pid": 987}}}
    raw, log = raw_fixture()
    raw.update(qpos=np.column_stack((raw["root_pos"], raw["root_quaternion_wxyz"], raw["joint_pos"])),
        qvel=np.column_stack((np.ones(1397), np.zeros((1397, 5)), raw["joint_vel"])),
        qpos_joint_names=np.array(nominal.JOINT_NAMES),
        stage_cost=np.full(349, .5))
    for row in log: row["objective"] = {"stage_cost": .5}
    np.savez_compressed(directory/"rollout.npz", **raw)
    (directory/"solver_log.json").write_text(json.dumps(log))
    report.update(rollout_sha256=analysis.sha256(directory/"rollout.npz"), solver_log_sha256=analysis.sha256(directory/"solver_log.json"))
    (directory/"report.json").write_text(json.dumps(report))
    return report


def test_actual_non_nominal_material_is_audited_without_relabeling(tmp_path):
    report = write_case(tmp_path, .1); before = copy.deepcopy(report)
    evidence, trace, public = analysis.audit_episode(tmp_path, .1, get_reference(), report["model_sha256"])
    assert evidence["plant_mu"] == public["actual_plant_mu"] == .1
    assert report == before and public["prediction_mu_scenarios"] == [.7]
    assert trace["time_s"].shape == (1397,) and public["complete_physical_evidence"] is True
    payload = json.dumps(public)
    assert "/private/" not in payload and "compiler_command" not in payload and "worker_pid" not in payload


@pytest.mark.parametrize("problem", ["mu", "material", "prediction", "config", "source", "settings_source", "library",
    "learned", "model_plan", "model", "clock", "report_hash", "phase", "physical_state", "stage_cost"])
def test_changed_material_controller_or_physical_evidence_fails_closed(tmp_path, problem):
    report = write_case(tmp_path, .1)
    model_sha256 = report["model_sha256"]
    if problem == "mu": report["plant_mu"] = .7
    elif problem == "material": report["plant_material_readback"]["compiled_sliding_friction_range"] = [.7, .7]
    elif problem == "prediction": report["prediction_mu_scenarios"] = [.1]
    elif problem == "config": report["effective_cem_config"]["iterations"] = 6
    elif problem == "source": report["source_sha256"]["/private/archived/scripts/nmpc_shooting.py"] = "0"*64
    elif problem == "settings_source": report["source_sha256"]["/private/archived/config/nmpc_height_smooth_residual.json"] = "0"*64
    elif problem == "library": report["native_prediction_backend"]["workers"]["0.7"]["native_library_sha256"] = "0"*64
    elif problem == "learned": report["learned_policy_used"] = True
    elif problem == "model_plan": report["model_generated_nominal_plan"]["enabled"] = True
    elif problem == "model": report["model_sha256"] = "0"*64
    elif problem == "report_hash": report["rollout_sha256"] = "0"*64
    elif problem == "stage_cost": report["total_stage_cost"] += 1.
    else:
        raw = analysis.load_arrays(tmp_path/"rollout.npz")
        if problem == "clock": raw["time_s"][5:] += .005
        elif problem == "phase": raw["reference_frame_for_action"][5] += 1
        else: raw["qpos"][5, 0] += .2
        np.savez_compressed(tmp_path/"rollout.npz", **raw)
        report["rollout_sha256"] = analysis.sha256(tmp_path/"rollout.npz")
    (tmp_path/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): analysis.audit_episode(tmp_path, .1, get_reference(), model_sha256)


@pytest.mark.parametrize("status", ["failed", "running"])
def test_incomplete_runtime_is_censored_not_a_fake_physical_failure_or_pass(tmp_path, status):
    report = write_case(tmp_path, .1); report["status"] = status; report["completed_policy_steps"] = 12
    (tmp_path/"report.json").write_text(json.dumps(report))
    evidence, trace, public = analysis.audit_episode(tmp_path, .1, get_reference(), report["model_sha256"])
    assert evidence is None and trace is None and public["censored"] is True
    assert public["physical_xy_failure_claimed"] is False and public["complete_physical_evidence"] is False
    assert set(public["input_sha256"]) == {"report"}
    assert "error" not in public and public["reported_completed_policy_steps"] == 12


@pytest.mark.parametrize("algorithm", ["PPO", "SAC"])
def test_authoritative_public_policy_grid_replays_exactly_without_invented_speed(algorithm):
    directory = ROOT/("results/controller_primary_"+algorithm.lower()+"_height_smooth_seed42_v1")
    public, rows, trace = analysis.load_policy_grid(directory, algorithm, get_reference())
    assert len(rows) == 17 and rows[.7]["mujoco"]["core_complete"] is True
    assert public["measured_com_speed_available"] is False and public["measured_contact_slip_available"] is False
    assert "joint_pos" not in trace and "wheel_pos" not in trace


@pytest.mark.parametrize("filename", ["summary.json", "xy_trace.npz", "xy_trace.provenance.json"])
def test_policy_grid_bytes_are_bound_to_existing_training_metadata(tmp_path, filename):
    source = ROOT/"results/controller_primary_ppo_height_smooth_seed42_v1"
    for name in ("summary.json", "xy_trace.npz", "xy_trace.provenance.json", "training_metadata.json"):
        shutil.copyfile(source/name, tmp_path/name)
    path = tmp_path/filename
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="published-artifact bindings"):
        analysis.load_policy_grid(tmp_path, "PPO", get_reference())


def arguments(tmp_path, monkeypatch, censored=False):
    args = fixture_arguments(tmp_path, monkeypatch)
    for name in ("verify_reference_pair", "source_model", "WheelKinematics"):
        monkeypatch.setattr(analysis, name, getattr(nominal, name))
    spec = nominal.verify_reference_pair(None, None, None)
    args.run = []
    for mu, directory in zip((.1, .3, .7), (tmp_path/"baseline", tmp_path/"H30", tmp_path/"residual")):
        report = write_case(directory, mu, spec=spec, model_sha256=analysis.sha256(args.model))
        if censored and mu == .3:
            report.update(status="failed", completed_policy_steps=7)
            (directory/"report.json").write_text(json.dumps(report))
        args.run.append(f"{mu}={directory}")
    args.ppo_grid, args.sac_grid = tmp_path/"ppo", tmp_path/"sac"
    helpers = {name: analysis.source_digest(report, name) for name in analysis.FK_HELPERS}
    @contextmanager
    def loader(*unused):
        yield None, {"dynamics_stepping": False, "source_helpers_sha256": helpers}
    monkeypatch.setattr(analysis, "source_model", loader)
    def policies(directory, algorithm, reference_spec):
        raw, _ = raw_fixture(); rows, traces = {}, {}
        for mu in analysis.expected_grid():
            rows[mu] = {"mujoco": {"full_core_reference_xy_rmse_m": .1,
                "full_core_reference_xy_peak_m": .2, "core_complete": True, "large_reference_xy_screen": False}}
            case = analysis.case_name(mu)
            traces[case+"_mujoco_t"] = raw["time_s"].copy()
            traces[case+"_mujoco_xy"] = raw["root_pos"][:, :2].copy()
        return {"algorithm": algorithm, "measured_com_speed_available": False}, rows, traces
    monkeypatch.setattr(analysis, "load_policy_grid", policies)
    return args


@pytest.mark.parametrize("censored", [False, True])
def test_end_to_end_partial_development_replay_is_portable_and_never_qualified(tmp_path, monkeypatch, censored):
    pytest.importorskip("matplotlib")
    args = arguments(tmp_path, monkeypatch, censored=censored)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    value = analysis.run(args)
    assert value["provided_mu"] == [.1, .3, .7] and value["complete_predeclared_grid"] is False
    assert value["evaluated_mu"] == ([.1, .7] if censored else [.1, .3, .7])
    assert value["censored_mu"] == ([.3] if censored else [])
    assert len(value["rows"]) == 17 and value["rows"][0]["NMPC"]["reference_xy_rmse_m"] is None
    for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
        "sim2sim_comparison", "eligible_for_controller_ranking", "pooled_success_statistics",
        "height_or_flight_failure_gate", "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference"):
        assert value[name] is False
    assert all(row["PPO"]["speed_retention"] is None and row["SAC"]["contact_slip"] is None for row in value["rows"])
    payload = (args.output_dir/"summary.json").read_text()
    assert "/private/" not in payload and str(tmp_path) not in payload
    arrays = analysis.load_arrays(args.output_dir/"development_trace.npz")
    assert np.array_equal(arrays["ppo_mu0p100_time_s"], np.arange(1397)*.005)
    assert "reference_contact_count" not in arrays and "mu0p100_contact_count" in arrays
    assert not any(name.endswith(("qpos", "qvel", "actions", "held_action")) for name in arrays)
    assert ("mu0p300_time_s" in arrays) is not censored
    for artifact in value["artifacts"].values(): assert analysis.sha256(args.output_dir/artifact["file"]) == artifact["sha256"]
    for path, content in before.items(): assert path.read_bytes() == content


def test_pipeline_rejects_changed_unpinned_helper_identity_across_mu(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    path = tmp_path/"H30/report.json"; report = analysis.read_json(path)
    report["source_sha256"]["/private/archived/scripts/eval_tracking_mujoco.py"] = "0"*64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="changed across coefficients"): analysis.run(args)
    assert not args.output_dir.exists()


def test_fk_loader_must_match_recorded_physical_helper_identity(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    @contextmanager
    def loader(*unused): yield None, {"source_helpers_sha256": {"fake.py": "f"*64}}
    monkeypatch.setattr(analysis, "source_model", loader)
    with pytest.raises(ValueError, match="FK loader helpers differ"): analysis.run(args)
    assert not args.output_dir.exists()


def test_cli_and_existing_output_and_duplicate_coefficient_guards(tmp_path):
    args = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x",
        "--source-root", "s", "--ppo-grid", "p", "--sac-grid", "a", "--output-dir", "o", "--run", ".1=d"])
    assert args.motion_file == Path("m")
    with pytest.raises(ValueError, match="Preserve existing"): analysis.run(SimpleNamespace(output_dir=tmp_path))
    with pytest.raises(ValueError, match="at least three"): analysis.parse_runs([".1="+str(tmp_path)])
    with pytest.raises(ValueError, match="Duplicate coefficient"):
        analysis.parse_runs([".1="+str(tmp_path), ".10="+str(tmp_path), ".3="+str(tmp_path)])
