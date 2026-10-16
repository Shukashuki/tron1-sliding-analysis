"""Synthetic clock/score rejection tests; golden MJ recordings are never rerun."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_sa29_controller_comparison as a
from nmpc_tracking_objective import HeightSmoothTrackingObjective, JOINT_NAMES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path, base):
    return {"path": str(path.relative_to(base)), "sha256": digest(path)}


def terminal_fixture(tmp_path, mu=.7):
    """Synthetic opaque files and receipts, never a simulator recording."""
    folder = tmp_path/a.case_name(mu); folder.mkdir()
    artifacts = {}
    for role in ("report", "trace", "solver"):
        path = folder/(role+".bin"); path.write_bytes((role+" synthetic capture token").encode())
        artifacts[role] = digest(path)
    command = ["/private/runtime/python", "-u", "/private/snapshot/scripts/run_nmpc_tracking_isaac.py",
        "--source-root", "/private/snapshot/upstream/tron1-learning-from-human", "--motion", "/private/motion.npz",
        "--reference-id", a.REFERENCE_ID, "--asset-path", "/private/robot.usda", "--mujoco-model", "/private/model.xml",
        "--contract", "/private/actuator.json", "--output-dir", "/private/new-output", "--task-profile", "/private/task.json",
        "--settings", "/private/settings.json", "--plant-mu", str(mu), "--prediction-mu-scenarios", "0.7",
        "--horizon-steps", "30", "--control-blocks", "10", "--candidates", "64", "--iterations", "3",
        "--elite-count", "8", "--num-envs", "1", "--seed", "42", "--contact-threshold-n", "5.0",
        "--prediction-backend", "native", "--native-library", "/private/existing/libtron1_nmpc_batch.so",
        "--device", "cuda:2", "--headless", "--kit_args",
        "--/renderer/activeGpu=2 --/renderer/multiGpu/enabled=false --/physics/cudaDevice=2 --/physics/sceneMultiGPUMode=0"]
    log = folder/"evaluation.log"; log.write_text("Synthetic terminal success, no simulator was run.\n")
    receipt = {"schema_version": 1, "kind": "sa29_nmpc_engine_terminal_receipt", "algorithm": "NMPC", "engine": "isaac",
        "reference_id": a.REFERENCE_ID, "mu": mu, "terminal_status": "exited", "exit_code": 0,
        "log_sha256": digest(log), "artifact_sha256": artifacts.copy(), "command": command, "requested_device": "cuda:2",
        "owned_processes_and_contexts_absent": True, "source_bindings_unchanged": True,
        "supervisor_pid": 12345, "private_path": "/private/supervisor/receipt.json"}
    path = folder/"terminal_receipt.json"; path.write_text(json.dumps(receipt))
    device = {"status": "verified", "requested_device": "cuda:2", "required_unmasked": True,
        "cuda_visible_devices": None, "native_and_tensor_device_consistency_verified": True}
    return {"receipt": path, "log": log}, receipt, artifacts, device


def execution_manifest_fixture(tmp_path):
    cases = []
    for mu in a.GRID:
        capture, *_ = terminal_fixture(tmp_path, mu)
        cases.append({"mu": mu, **{role: binding(path, tmp_path) for role, path in capture.items()}})
    document = {"schema_version": 1, "kind": "sa29_nmpc_isaac_terminal_capture", "reference_id": a.REFERENCE_ID, "cases": cases}
    path = tmp_path/"execution_manifest.json"; path.write_text(json.dumps(document))
    return path, document


def test_complete_execution_manifest_is_portable_hash_bound_and_does_not_promote_physical_outcomes(tmp_path):
    path, document = execution_manifest_fixture(tmp_path); before = copy.deepcopy(document)
    result = a.load_execution_manifest(path)
    assert document == before and set(result["cases"]) == set(a.GRID)
    assert result["manifest_sha256"] == digest(path)
    assert all(set(value) == {"receipt", "log"} for value in result["cases"].values())


@pytest.mark.parametrize("problem", ["partial", "duplicate", "reference", "kind", "extra_field", "missing_role", "bool_mu",
    "wrong_mu", "nan_mu", "missing_file", "changed_sha", "absolute", "parent", "symlink_escape", "path_alias", "inode_alias"])
def test_execution_manifest_rejects_incomplete_aliased_or_nonportable_evidence(tmp_path, problem):
    path, document = execution_manifest_fixture(tmp_path)
    if problem == "partial": document["cases"].pop()
    elif problem == "duplicate": document["cases"][-1] = copy.deepcopy(document["cases"][0])
    elif problem == "reference": document["reference_id"] = "drift_drl_traj4"
    elif problem == "kind": document["kind"] = "driver_declared_success"
    elif problem == "extra_field": document["qualified"] = True
    elif problem == "missing_role": del document["cases"][0]["log"]
    elif problem == "bool_mu": document["cases"][0]["mu"] = True
    elif problem == "wrong_mu": document["cases"][0]["mu"] = .04
    elif problem == "nan_mu": document["cases"][0]["mu"] = float("nan")
    elif problem == "missing_file": (tmp_path/document["cases"][0]["receipt"]["path"]).unlink()
    elif problem == "changed_sha": document["cases"][0]["receipt"]["sha256"] = "0"*64
    elif problem == "absolute": document["cases"][0]["receipt"]["path"] = str(tmp_path/document["cases"][0]["receipt"]["path"])
    elif problem == "parent": document["cases"][0]["receipt"]["path"] = "../outside.json"
    elif problem == "symlink_escape":
        target = tmp_path/document["cases"][0]["receipt"]["path"]
        outside = tmp_path.parent/(tmp_path.name+"-external.json"); outside.write_bytes(target.read_bytes())
        target.unlink(); target.symlink_to(outside)
    elif problem == "path_alias": document["cases"][1]["receipt"] = document["cases"][0]["receipt"]
    else:
        original = tmp_path/document["cases"][0]["receipt"]["path"]
        target = tmp_path/document["cases"][1]["receipt"]["path"]
        target.unlink(); os.link(original, target); document["cases"][1]["receipt"]["sha256"] = digest(target)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError): a.load_execution_manifest(path)


def test_actual_terminal_projection_contains_only_clean_exit_closure_and_hashes(tmp_path):
    capture, receipt, artifacts, device = terminal_fixture(tmp_path)
    result = a.audit_execution_case(capture, .7, artifacts, device)
    assert result["exit_code"] == 0 and result["traceback_error_scan_passed"] is True
    assert result["owned_processes_and_contexts_absent"] is result["source_bindings_unchanged"] is True
    assert result["receipt_sha256"] == digest(capture["receipt"]) and result["log_sha256"] == digest(capture["log"])
    serialized = json.dumps(result)
    assert "/private/" not in serialized and "command" not in result and "supervisor_pid" not in result
    assert receipt["private_path"].startswith("/private/")


def test_runtime_cli_requires_original_execution_manifest():
    argv = ["--motion", "motion.npz", "--reference-file", "robot_reference.npz", "--model", "model.xml",
        "--source-root", "source", "--mujoco-bundle", "bundle", "--output-dir", "fresh", "--run", ".7=case"]
    with pytest.raises(SystemExit): a.parser().parse_args(argv)
    parsed = a.parser().parse_args(argv+["--execution-manifest", "execution_manifest.json"])
    assert parsed.execution_manifest == Path("execution_manifest.json")


def test_complete_physical_episode_still_calls_required_terminal_audit(tmp_path, monkeypatch):
    directory = tmp_path/"episode"; directory.mkdir(); (directory/"solver_log.json").write_text("[]")
    capture, _, hashes, device = terminal_fixture(tmp_path)
    monkeypatch.setattr(a, "read_json", lambda _: {"status": "completed"})
    monkeypatch.setattr(a, "audit_isaac_header", lambda *unused: {"asset_sha256": a.ISAAC_ASSET_SHA256, "device_audit": device})
    monkeypatch.setattr(a, "episode_evidence", lambda *unused: {"hashes": hashes})
    monkeypatch.setattr(a, "load_arrays", lambda _: {})
    monkeypatch.setattr(a, "audit_isaac_trace", lambda *unused: ({}, {"counts": {"scenario_evaluations": 1}}))
    called = []
    def reject_terminal(actual_capture, mu, actual_hashes, audit):
        assert actual_capture is capture and mu == .7 and actual_hashes == hashes and audit is device
        called.append(True)
        raise ValueError("Synthetic terminal error remains a capture failure, never XY failure")
    monkeypatch.setattr(a, "audit_execution_case", reject_terminal)
    with pytest.raises(ValueError, match="Synthetic terminal error"):
        a.audit_isaac_episode(directory, .7, {}, "0"*64, {}, capture=capture)
    assert called == [True]


@pytest.mark.parametrize("problem", ["nonzero", "bool_exit", "running", "kind", "reference", "algorithm", "engine", "mu",
    "log_sha", "artifact_sha", "extra_artifact", "processes_remain", "changed_sources", "traceback_exit0", "cuda_exit0",
    "fatal_exit0", "invalid_utf8", "wrong_operand", "entrypoint_decoy", "wrong_device", "masked_device", "unverified_device",
    "horizon", "blocks", "candidates", "iterations", "elites", "envs", "seed", "prediction_mu", "extra_scenario",
    "plant_mu", "threshold", "python_backend", "no_native", "no_headless", "no_kit", "wrong_kit_gpu", "extra_kit",
    "max_steps", "adaptive", "model_plan"])
def test_exit_zero_or_complete_physics_cannot_hide_bad_nmpc_process_evidence(tmp_path, problem):
    capture, receipt, artifacts, device = terminal_fixture(tmp_path)
    command = receipt["command"]
    def argument(flag, value): command[command.index(flag)+1] = value
    fields = {"nonzero": ("exit_code", 1), "bool_exit": ("exit_code", False), "running": ("terminal_status", "running"),
        "kind": ("kind", "driver_success"), "reference": ("reference_id", "drift_drl_traj4"), "algorithm": ("algorithm", "PPO"),
        "engine": ("engine", "mujoco"), "mu": ("mu", .1), "log_sha": ("log_sha256", "0"*64),
        "processes_remain": ("owned_processes_and_contexts_absent", False), "changed_sources": ("source_bindings_unchanged", False)}
    if problem in fields: receipt[fields[problem][0]] = fields[problem][1]
    elif problem == "artifact_sha": receipt["artifact_sha256"]["trace"] = "0"*64
    elif problem == "extra_artifact": receipt["artifact_sha256"]["other"] = "0"*64
    elif problem in {"traceback_exit0", "cuda_exit0", "fatal_exit0", "invalid_utf8"}:
        content = {"traceback_exit0": b"Traceback (most recent call last):\n FileNotFoundError\n",
            "cuda_exit0": b"CUDA error: invalid device\n", "fatal_exit0": b"[FATAL] native error\n", "invalid_utf8": b"\xff"}[problem]
        capture["log"].write_bytes(content); receipt["log_sha256"] = digest(capture["log"])
    elif problem == "wrong_operand": command[2] = "/private/other.py"
    elif problem == "entrypoint_decoy": command.insert(2, "/private/other.py")
    elif problem == "wrong_device": receipt["requested_device"] = "cuda:1"
    elif problem == "masked_device": device["cuda_visible_devices"] = "2"
    elif problem == "unverified_device": device["native_and_tensor_device_consistency_verified"] = False
    elif problem == "extra_scenario": command.insert(command.index("--prediction-mu-scenarios")+2, "0.8")
    elif problem in {"max_steps", "adaptive", "model_plan"}:
        flag = {"max_steps": "--max-policy-steps", "adaptive": "--adaptive-contact-config", "model_plan": "--model-nominal-plan"}[problem]
        command.extend([flag, "1" if problem == "max_steps" else "/private/override.json"])
    elif problem == "no_native": del command[command.index("--native-library"):command.index("--native-library")+2]
    elif problem == "no_headless": command.remove("--headless")
    elif problem == "no_kit": del command[command.index("--kit_args"):command.index("--kit_args")+2]
    elif problem == "wrong_kit_gpu": argument("--kit_args", command[-1].replace("cudaDevice=2", "cudaDevice=1"))
    elif problem == "extra_kit": argument("--kit_args", command[-1]+" --/other=true")
    else:
        flag, value = {"horizon": ("--horizon-steps", "15"), "blocks": ("--control-blocks", "5"),
            "candidates": ("--candidates", "32"), "iterations": ("--iterations", "2"), "elites": ("--elite-count", "4"),
            "envs": ("--num-envs", "2"), "seed": ("--seed", "43"), "prediction_mu": ("--prediction-mu-scenarios", "0.8"),
            "plant_mu": ("--plant-mu", "0.1"), "threshold": ("--contact-threshold-n", "1"),
            "python_backend": ("--prediction-backend", "python")}[problem]
        argument(flag, value)
    capture["receipt"].write_text(json.dumps(receipt))
    with pytest.raises((ValueError, UnicodeDecodeError)):
        a.audit_execution_case(capture, .7, artifacts, device)


@pytest.fixture(scope="module")
def bundle():
    directory = ROOT/"results/nmpc_sa29_friction_development_20261016_v1"
    return directory, a.read_json(directory/"summary.json"), a.load_arrays(directory/"development_trace.npz")


def test_existing_mujoco_bundle_identity_and_metrics_replay(bundle):
    directory, summary, data = bundle
    reference, _ = a.compact_case(data, "reference")
    actual = a.load_mujoco_bundle(directory, summary["reference_spec"], summary["input_sha256"], reference)
    assert len(actual[-2]) == len(actual[-1]) == 17
    assert actual[1] == a.MUJOCO_BUNDLE_SHA256


@pytest.mark.parametrize("prefix", ["reference", "mu0p700"])
def test_compact_clocks_are_sa29_not_primary(bundle, prefix):
    _, _, data = bundle
    trace, geometry = a.compact_case(data, prefix)
    assert len(trace["time_s"]) == (258 if prefix == "reference" else 1033)
    assert geometry["signals"]["speed_threshold_m_s"] == .25


@pytest.mark.parametrize("field", ["time_s", "position", "quaternion", "com_velocity", "angular_velocity", "wheel_pos",
    "contact_count", "contact_force", "contact_slip", "nonwheel_ground_contact_count"])
def test_compact_rejects_partial_fields(bundle, field):
    _, _, original = bundle
    data = dict(original); key = "mu0p700_"+field
    data[key] = data[key][:-1]
    with pytest.raises((ValueError, IndexError)):
        a.compact_case(data, "mu0p700")


@pytest.mark.parametrize("field", ["speed_m_s", "course_rad", "root_directed_beta_rad", "wheel_rolling_plane_heading_rad"])
def test_compact_signals_must_replay_from_measured_velocity_and_planes(bundle, field):
    _, _, original = bundle
    data = dict(original); key = "mu0p700_"+field
    data[key] = data[key].copy()+.001
    with pytest.raises(ValueError, match="Compact speed/heading"):
        a.compact_case(data, "mu0p700")


def test_published_bundle_is_not_replaced_with_recomputed_self_hashes(bundle, monkeypatch):
    directory, summary, data = bundle
    reference, _ = a.compact_case(data, "reference")
    monkeypatch.setattr(a, "sha256", lambda _: "0"*64)
    with pytest.raises(ValueError, match="byte-for-byte"):
        a.load_mujoco_bundle(directory, summary["reference_spec"], summary["input_sha256"], reference)


def test_policy_join_retains_float64_cast_proof_without_overwriting_pinned_reference():
    reference = np.zeros((258,3),dtype=np.float32)
    compact = {"reference_position": reference}
    original = np.ones((258,3),dtype=np.float64)
    time = np.arange(258)*.02
    controller = np.ones((1033,3))
    data = {"reference_position": np.full((258,3),9.),
        "reference_robot_position_float64_m": original,
        "reference_robot_time_s": time, "PPO_mu0p700_isaac_position": controller}
    a.append_policy_compact(compact,data)
    assert compact["reference_position"] is reference
    assert compact["reference_robot_position_float64_m"].dtype == np.float64
    assert np.array_equal(compact["reference_robot_position_float64_m"],original)
    assert np.array_equal(compact["reference_robot_time_s"],time)
    assert np.array_equal(compact["PPO_mu0p700_isaac_position"],controller)
    original[:]=0.;time[:]=0.;controller[:]=0.
    assert compact["reference_robot_position_float64_m"].sum() == 774.
    assert compact["reference_robot_time_s"][-1] == 5.14
    assert compact["PPO_mu0p700_isaac_position"].sum() == 3099.


@pytest.fixture
def isaac_synthetic():
    n, f = a.SAMPLES, a.FRAMES
    q = np.tile([1.,0.,0.,0.], (n,1)); pos = np.zeros((n,3)); pos[:,0] = np.arange(n)*.005
    raw = {"time_s": np.arange(n)*.005, "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
        "root_pos": pos, "root_quaternion_wxyz": q,
        "base_com_velocity_w": np.zeros((n,3)), "base_angular_velocity_w": np.zeros((n,3)),
        "joint_pos": np.zeros((n,8)), "joint_vel": np.zeros((n,8)), "wheel_pos": np.zeros((n,2,3)),
        "action_joint_names": np.array(JOINT_NAMES), "wheel_ground_contact_count": np.zeros((n,2)),
        "wheel_ground_contact_force_w_n": np.zeros((n,2,3)), "wheel_lateral_contact_slip_m_s": np.zeros((n,2)),
        "nonwheel_ground_contact_count": np.zeros(n), "nonwheel_ground_contact_force_norm_n": np.zeros(n),
        "simulation_step_counter": np.arange(n)+123, "actions": np.zeros((f,8)), "compute_s": np.full(f,.03)}
    packed = np.zeros((f,111)); packed[:,:3] = pos[::4][:-1]; packed[:,3] = 1.
    raw["packed_controller_states"] = packed
    task = a.read_json(ROOT/"config/car_drift_height_smooth_task.json")
    contract = HeightSmoothTrackingObjective(task).contract(); weights = contract["weights"]
    upper = sum(v for v in weights.values() if v > 0)
    info = {"candidate_evaluations": 32, "finite_candidates": 32, "failed_candidates": 0,
        "scenario_evaluations": 32, "failed_scenarios": 0, "batch_prediction_calls": 1,
        "failed_batch_prediction_calls": 0, "optimization_failed": False, "fallback_used": False,
        "warm_start_used": False, "initial_candidate_generation_failed": False,
        "control_deadline_enforced": False, "real_time_verified": False, "timing_scope": "synthetic unit test"}
    score = {"raw_terms": dict.fromkeys(weights,0.), "weighted_terms": dict.fromkeys(weights,0.),
        "total_reward": 0., "reward_upper_bound": upper, "stage_cost": .02*upper, "base_com_vz": 0.}
    log = [{"frame": k, "reference_frame_for_reward": k, "action": [0.]*8,
        "raw_cem_action": [0.]*8, "compute_s": .03, "solver": copy.deepcopy(info),
        "actual_isaac_objective": copy.deepcopy(score), "same_state_dt_reward_abs_error": 0.,
        "coordinate_conversion_max_velocity_error": 0., "actual_dt_scaled_reward": 0.} for k in range(f)]
    report = {"objective_contract": contract, "compute_ms_median_p95_max": [30.]*3,
        "compute_over_control_dt_fraction": 1., "solver_failures": 0,
        "maximum_same_state_dt_reward_abs_error": 0., "maximum_coordinate_conversion_error": 0.,
        "real_time_verified": False, "elapsed_s": 100.}
    return report,raw,log


def test_synthetic_complete_isaac_diagnostics(isaac_synthetic):
    trace, result = a.audit_isaac_trace(*isaac_synthetic)
    assert len(trace["time_s"]) == 1033
    assert result["decision_samples"] == 258
    assert result["counts"]["optimization_failed"] == 0


@pytest.mark.parametrize("mutation", ["counter_gap", "counter_reset", "packed_pose", "packed_nan", "raw_action",
    "action_phase", "timing", "solver_count", "fallback", "weighted_score", "stage_cost", "action_memory",
    "absolute_height_replaces_acceleration", "conversion", "reward_error", "real_time", "missing_decision"])
def test_synthetic_isaac_rejects_invalid_evidence(isaac_synthetic, mutation):
    report,raw,log = isaac_synthetic
    if mutation == "counter_gap": raw["simulation_step_counter"][10:] += 1
    elif mutation == "counter_reset": raw["simulation_step_counter"][10:] -= 10
    elif mutation == "packed_pose": raw["packed_controller_states"][0,0] += 1
    elif mutation == "packed_nan": raw["packed_controller_states"][0,10] = np.nan
    elif mutation == "raw_action": log[0]["raw_cem_action"][0] = .1
    elif mutation == "action_phase": log[0]["reference_frame_for_reward"] = 1
    elif mutation == "timing": raw["compute_s"][0] = .01
    elif mutation == "solver_count": log[0]["solver"]["finite_candidates"] -= 1
    elif mutation == "fallback": log[0]["solver"]["fallback_used"] = True
    elif mutation == "weighted_score": log[0]["actual_isaac_objective"]["weighted_terms"]["action_rate_l2"] = 1.
    elif mutation == "stage_cost": log[0]["actual_isaac_objective"]["stage_cost"] += 1.
    elif mutation in ("action_memory", "absolute_height_replaces_acceleration"):
        name = "action_rate_l2" if mutation == "action_memory" else "height_vertical_acceleration"
        score = log[0]["actual_isaac_objective"]; score["raw_terms"][name] = .1
        score["weighted_terms"][name] = .1*report["objective_contract"]["weights"][name]
        score["total_reward"] = sum(score["weighted_terms"].values())
        score["stage_cost"] = .02*(score["reward_upper_bound"]-score["total_reward"])
    elif mutation == "conversion": log[0]["coordinate_conversion_max_velocity_error"] = .001
    elif mutation == "reward_error": log[0]["same_state_dt_reward_abs_error"] = .001
    elif mutation == "real_time": report["real_time_verified"] = True
    elif mutation == "missing_decision": log.pop()
    with pytest.raises(ValueError): a.audit_isaac_trace(report,raw,log)


@pytest.mark.parametrize("selection", [(), (.2,.2), (.25,), (True,), (float("nan"),)])
def test_topdown_rejects_undeclared_or_duplicate_display(tmp_path, selection):
    with pytest.raises(ValueError, match="Display requires"):
        a.render_topdown({}, {}, tmp_path/"plot.png", display_mu=selection)


def test_topdown_preserves_existing_plot(tmp_path):
    target = tmp_path/"plot.png"
    target.write_bytes(b"existing original")
    with pytest.raises(ValueError, match="Preserve"):
        a.render_topdown({}, {}, target, display_mu=(.2,))
    assert target.read_bytes() == b"existing original"


def test_topdown_preserves_dangling_plot_symlink(tmp_path):
    target = tmp_path/"plot.png"
    missing = tmp_path/"unrelated-missing.png"
    target.symlink_to(missing)
    with pytest.raises(ValueError, match="Preserve"):
        a.render_topdown({}, {}, target, display_mu=(.2,))
    assert target.is_symlink() and not missing.exists()


@pytest.mark.parametrize("selection,posthoc,rows", [(None,False,5), ((.2,),True,1)])
def test_topdown_posthoc_is_presentation_only(tmp_path, monkeypatch, selection, posthoc, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {}
    selected = a.SELECTED_MU if selection is None else selection
    for prefix in ["reference"] + [engine+"_"+a.case_name(mu) for mu in selected for engine in ("isaac","mujoco")]:
        data.update({prefix+"_time_s":np.array([0.,.02]), prefix+"_physics_dt_s":np.array(.02),
            prefix+"_position":np.array([[0.,0.,0.],[.02,0.,0.]]), prefix+"_wheel_pos":np.zeros((2,2,3)),
            prefix+"_root_course_to_wheel_valid":np.ones((2,2),bool),
            prefix+"_wheel_rolling_plane_heading_rad":np.zeros((2,2)),
            prefix+"_contact_count":np.ones((2,2)), prefix+"_contact_force":np.tile([0.,0.,10.],(2,2,1)),
            prefix+"_root_beta_valid":np.ones(2,bool), prefix+"_com_velocity":np.tile([1.,0.,0.],(2,1))})
    original = {key:value.copy() for key,value in data.items()}
    summary = {"complete_three_controller_two_engine_grid":False}
    calls = {}; real_subplots = plt.subplots
    def subplots(*args, **kwargs):
        calls.update(rows=args[0],figsize=kwargs["figsize"])
        fig,axes = real_subplots(*args, **kwargs)
        monkeypatch.setattr(fig,"savefig",lambda output,**kw: calls.update(title=fig._suptitle.get_text()))
        return fig,axes
    monkeypatch.setattr(plt,"subplots",subplots)
    kwargs = {} if selection is None else {"display_mu":selection}
    result = a.render_topdown(data,summary,tmp_path/"plot.png",**kwargs)
    assert calls["rows"] == rows and calls["figsize"] == (8.4,3.4*rows)
    assert ("POST-HOC" in calls["title"]) is posthoc
    assert result == {"display_mu":list(selected),"posthoc_display_selection":posthoc,
        "new_physical_episodes":False,"metrics_or_trajectories_changed":False}
    assert summary == {"complete_three_controller_two_engine_grid":False}
    assert all(np.array_equal(data[key],value) for key,value in original.items())
