"""Frozen400Hz MPC-CEM validation friction sweep; no policy training.

The native plant and predictor receive the same explicit known sliding mu.
Only that material coefficient changes. The completed nominal mu=.7 episodes
are reused by literal SHA bindings, not rerun or represented as new samples.
Two isolated CPU episodes maximum; never start GPU or remote work here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import render_ppo_beta_physx_to_mujoco_400hz as paired
import run_ppo_cem_matched_reward_400hz as nominal
from controller_references import get_reference
from eval_ppo_backend_400hz_mujoco import audit_materials
from nmpc_shooting import CEMConfig, ShootingNMPC
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState
from ppo_backend_400hz_contract import require, read_json, sha256
from run_nmpc_tracking import MujocoTrackingPredictor, build_provider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ROOT/"config/ppo_cem_friction_failure_400hz_v1.json"
DEFAULT_NOMINAL = ROOT/"outputs/ppo_cem_matched_reward_400hz_20261022_v1"
GRID = (.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.)
NOMINAL_SOURCE_SHA = "f083a62813e8e5c7bf7099d90e603eaa3eea7909e148cfb636e613b200177bea"
NOMINAL_HASHES = {
    "beta_on": {
        "report.json": "343f6270e8be135e2afd4f06df69d8d4f1273cca2cc651188a1198c81ab901da",
        "rollout.npz": "eff2b735b07b481e6ead32b76a716066f71815401f91231345dde84d451804be",
        "solver_log.json": "8550fc816a48c51ec4c856eb2baf8daa80f492855e9109d2161a200f1f09f34c"},
    "beta_off": {
        "report.json": "0687ee6edd2db42e4ebc54986dc350331a73da12aad08009fe9fbbdad2ec00aa",
        "rollout.npz": "bbf118117d3f9dc4dc0ada4e6494908211bd415c430ff00c7a9b70e178d1bd8f",
        "solver_log.json": "7552f852de511c3bc78d78caaf949e0af080a396feabf3cab0c07701c5cd7e8a"}}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def inputs(settings_file=DEFAULT_SETTINGS):
    settings = read_json(settings_file)
    require(settings["schema_version"] == 1
        and settings["kind"] == "frozen400hz_matched_reward_CEM_friction_failure_validation"
        and settings["mu_grid"] == list(GRID)
        and settings["representative_first"] == [.15, .3, 1.2]
        and settings["beta_conditions"] == ["beta_on", "beta_off"]
        and settings["prediction_friction_known"] is True
        and settings["new_training"] is False
        and settings["concurrent_local_CPU_episodes"] == 2,
        "Require frozen400Hz seventeen-point known-mu validation")
    require(sha256(nominal.__file__) == NOMINAL_SOURCE_SHA,
        "Frozen nominal runner changed; never silently import a different baseline")
    base, contracts, traces, reference, evidence = nominal.inputs()
    for key in ("reference_id", "validation_backend", "physics_dt_s", "decimation", "policy_dt_s",
                "action_rate_weight", "grounded_reward_weight", "discount_factor", "nmpc"):
        require(settings[key] == base[key], "Frozen nominal recipe changed: "+key)
    require(settings["nmpc"]["seed"] == 42, "Per-episode CEM seed must remain42")
    verify_nominal(DEFAULT_NOMINAL)
    return settings, contracts, traces, reference, evidence


def verify_nominal(root):
    """Reuse original on/off records only; do not copy/relabel as fresh episodes."""
    root = Path(root)
    for label, files in NOMINAL_HASHES.items():
        for name, digest in files.items():
            require(sha256(root/label/name) == digest, "Bound nominal evidence changed: "+label+"/"+name)
        report = read_json(root/label/"report.json")
        require(report["status"] == "completed" and report["plant_mu"] == .7
            and report["prediction_mu_scenarios"] == [.7]
            and report["completed_policy_steps"] == 349 and report["native_samples"] == 2793,
            "Nominal evidence is not a complete frozen episode")
    return {label: {"status": "reused_completed_nominal", "directory": str((root/label).resolve()),
                   "mu": .7, "file_sha256": dict(files), "new_episode": False}
            for label, files in NOMINAL_HASHES.items()}


def configure_native_model(model, expected_materials, mu):
    """Match nominal nuisances exactly; only colliding sliding mu may change."""
    audit = audit_materials(model, {}, {"mu": mu})
    expected = copy.deepcopy(expected_materials)
    for row in expected["geom_friction"]:
        row[0] = mu
    for name in ("colliding_geom_ids", "colliding_geom_names", "wheel_geom_ids", "geom_friction",
                 "geom_priority", "geom_condim", "geom_solref", "geom_solimp"):
        require(audit[name] == expected[name], "Native friction-only sweep mismatch: "+name)
    readback = expected_materials["native_plant_readback"]
    for name, value in readback["readbacks"].items():
        observed = np.asarray(getattr(model, name))
        require(observed.shape == tuple(value["shape"])
            and np.array_equal(observed, np.asarray(value["values"])), "Native nuisance mismatch: "+name)
    for name, expected_value in readback["solver_configuration"].items():
        require(float(getattr(model.opt, name)) == expected_value, "Native solver mismatch: "+name)
    return audit


def run_episode(output, label, settings, contract, raw_policy, native_library, mu, max_steps=None):
    from nmpc_batch_objective_400hz import NativeBatchTrackingPredictor400Hz
    require(not output.exists(), "Preserve existing episode; fresh output required")
    # Full simulator contract is copied, including the actual eight-substep
    # actuator mapping; provider ignores old Q/R and does not execute PPO.
    require(float(mu) in GRID, "Friction must belong to frozen seventeen-point grid")
    contract = copy.deepcopy(contract)
    profile = copy.deepcopy(contract["task_profile"])
    profile["contact_friction"] = float(mu)
    # The frozen legacy validator constrains TRAINING-profile mu<=1. Retain
    # that original contract, then explicitly override native plant/predictor
    # material arrays for this validation, including requested mu>1. No legacy
    # validator is relaxed or monkeypatched, and no reward term consumes mu.
    report_source = (paired.DEFAULT_ON/"cases/physx/mujoco/report.json" if label == "beta_on"
                     else paired.DEFAULT_OFF/"off/report.json")
    policy_report = read_json(report_source)
    expected = policy_report["contact_material_audit"]
    provider = build_provider(paired.DEFAULT_MODEL, paired.DEFAULT_REFERENCE, contract, settings, (mu,))
    scalar = MujocoTrackingPredictor(provider, paired.DEFAULT_MODEL, profile, (mu,))
    # Exact native wheel priority/contact law is important even for equal mu:
    # wheel-priority also selects solref/solimp when contacts are unequal.
    prediction_materials = configure_native_model(scalar.models[mu], expected, mu)
    scalar.measure_model = scalar.models[mu]
    model, _ = provider.api.load_model(paired.DEFAULT_MODEL, no_visual_mesh=True)
    provider.api.configure_model(model, provider.contract)
    model.geom_friction[:, 0] = mu
    plant_materials = configure_native_model(model, expected, mu)
    data = mujoco.MjData(model)
    physical = provider.api.MujocoState(model, data, provider.reference)
    # Only the reset state is read. No demonstrated action or future policy
    # state is used for CEM initialization or candidate generation.
    data.qpos[:] = raw_policy["initial_qpos"]
    data.qvel[:] = raw_policy["initial_qvel"]
    mujoco.mj_forward(model, data)
    initial = {"qpos": data.qpos.copy().tolist(), "qvel": data.qvel.copy().tolist()}
    extractor = MujocoObjectiveState(model)
    history = np.zeros((3, len(BODY_NAMES), 3))
    reference_spec = get_reference(settings["reference_id"])
    require(provider.reference.frames == 349 and provider.dt == .02 and model.opt.timestep == .0025,
            "Native400Hz reference/clock mismatch")
    steps = 349 if max_steps is None else max_steps
    require(type(steps) is int and 1 <= steps <= 349, "Invalid explicit episode length")
    config = copy.deepcopy(settings["nmpc"])
    for name in ("initial_std", "min_std"):
        config[name] = tuple(config[name])
    cem = CEMConfig(**config)
    report = {"schema_version": 1, "kind": "400hz_friction_sweep_CEM_with_completed_PPO_stage_reward",
        "status": "running", "condition": label, "engine": "native CPU MuJoCo", "mujoco_version": mujoco.__version__,
        "reference_id": settings["reference_id"], "reference_spec": reference_spec,
        "model_sha256": sha256(paired.DEFAULT_MODEL), "motion_sha256": sha256(paired.DEFAULT_REFERENCE),
        "matched_PPO_report_sha256": sha256(report_source), "task_profile": profile,
        "objective_contract": scalar.objective.contract(), "physics_dt_s": .0025, "policy_dt_s": .02, "decimation": 8,
        "plant_mu": float(mu), "prediction_mu_scenarios": [float(mu)], "prediction_friction_known": True,
        "trained_profile_contact_friction": contract["task_profile"]["contact_friction"],
        "evaluation_friction_override": float(mu), "legacy_contract_training_friction_retained": True,
        "CEM_config": asdict(cem), "discount_factor": settings["discount_factor"],
        "discounted_objective": "sum_h gamma**h * dt * (reward_upper_bound - total_stage_reward); fixed horizon, no terminal value",
        "plant_contact_material_audit": plant_materials, "prediction_contact_material_audit": prediction_materials,
        "prediction_state_contract": scalar.layout.contract(),
        "prediction_reset_caveat": "Predictor cold-resets solver warmstart/time each20ms stage; validation plant integrates continuously. Full hidden-contact-state equivalence is NOT claimed even within MuJoCo.",
        "initial_state": initial, "PPO_actions_or_future_states_used_for_initial_plan": False,
        "new_training_or_policy_inference": False, "new_GPU_job": False, "native_contact_substeps": 8,
        "termination_or_reset_after_initialization": False, "height_contact_tilt_failure_gate": False,
        "native_nuisance_arrays_match_nominal_PPO_validation_exactly": True, "sliding_mu_only_change_from_nominal": True, "state_access": "Full qpos/qvel, objective memory and future reference; PPO actor has a51dim observation",
        "PPO_total_reward_recomputed": False, "PPO_total_reward_unavailable_reason": "Saved PPO traces lack all-body contact force/history; do not manufacture an exact episode-return comparison.",
        "reward_parity_scope": "Matched mathematical15term stage function and recorded weights/stds; native contact observations and runtime softlimits need not be bitwise identical to PhysX training.",
        "soft_joint_limits_source": extractor.soft_limits_source,
        "soft_joint_pos_limits": extractor.soft_limits.tolist(),
        "N_per_condition": 1, "max_policy_steps": steps, "truncated_smoke": steps != 349,
        "concurrent_local_CPU_episodes": settings["concurrent_local_CPU_episodes"],
        "timing_caveat": "Solve timing includes IPC and at most two local CPU episodes; not isolated hardware-matched PPO inference latency.",
        "real_time_verified": False, "controller_tuned_on_this_episode": False, "started_utc": utc_now(),
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
        "native_library_sha256": sha256(native_library), "nominal_reward_profile_source_sha256": sha256(report_source)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"report.json", report)
    records, decisions = [], []
    held_action = np.zeros(8)
    held_frame = 0

    def record(current):
        state = extractor.extract(data, contact_force_history=history, body_net_forces_w=current)
        slip = physical.ground_contact_slip(force_threshold_n=5.)
        count, force = physical.nonwheel_ground_contacts()
        records.append({"time_s": float(data.time), "root_pos": state["body_pos_w"][0].copy(),
            "root_quaternion_wxyz": state["body_quat_w"][0].copy(), "base_com_velocity_w": state["body_lin_vel_w"][0].copy(),
            "base_angular_velocity_w": state["body_ang_vel_w"][0].copy(), "wheel_pos": state["body_pos_w"][-2:].copy(),
            "joint_pos": state["joint_pos"].copy(), "joint_vel": data.qvel[physical.vids].copy(),
            "applied_joint_torque_nm": data.ctrl[physical.aids].copy(), "held_action": held_action.copy(),
            "reference_frame_for_action": held_frame, "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "body_pos_w": state["body_pos_w"].copy(), "body_quat_w": state["body_quat_w"].copy(),
            "body_lin_vel_w": state["body_lin_vel_w"].copy(), "body_ang_vel_w": state["body_ang_vel_w"].copy(),
            "body_net_forces_w": state["body_net_forces_w"].copy(), "contact_force_history_w": history.copy(),
            "nonwheel_ground_contact_count": count, "nonwheel_ground_contact_force_norm_n": force, **slip})
        return state

    previous = np.zeros(8)
    pre = record(np.zeros((len(BODY_NAMES), 3)))
    previous_vz = float(pre["body_lin_vel_w"][0, 2])
    started = time.perf_counter()
    try:
        with NativeBatchTrackingPredictor400Hz(scalar, native_library, discount_factor=settings["discount_factor"]) as predictor:
            controller = ShootingNMPC(predictor, cem, action_low=-np.ones(8), action_high=np.ones(8), mu_scenarios=(mu,))
            predictor.record_cem_wiring(controller)
            report["native_backend"] = predictor.metadata
            report["action_parameterization"] = controller.parameterization_contract()
            write_json(output/"report.json", report)
            for frame in range(steps):
                packed = scalar.layout.pack(data.qpos, data.qvel, history, frame == 0)
                held = scalar.objective.hold_reference(scalar.reference_frame(frame), pre)
                solve_start = time.perf_counter()
                action, solver = controller.action(packed, frame, previous_action=previous, previous_vz=previous_vz)
                compute_s = time.perf_counter()-solve_start
                require(action.shape == (8,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.+1e-9,
                        "CEM returned invalid action")
                raw_action = action.copy()
                action = action.astype(np.float32).astype(float)
                held_action = action.copy(); held_frame = frame
                for _ in range(8):
                    data.ctrl[physical.aids] = provider.torque(action, data.qpos, data.qvel)
                    mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
                    require(not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied), "Unexpected external/root force")
                    require(not any(item.number for item in data.warning), "MuJoCo validation plant warning; no silent reset/retry")
                    history, current = extractor.update_contact_history(data, history)
                    pre = record(current)
                    require(np.isclose(data.time, (len(records)-1)*.0025, atol=1e-10, rtol=0)
                            and np.isfinite(np.r_[data.qpos, data.qvel]).all(), "Hidden reset or nonfinite native plant state")
                objective = scalar.objective.evaluate(pre, held, action, previous, previous_vz, reset_sample=frame == 0)
                decisions.append({"frame": frame, "time_s": float(data.time), "action": action.tolist(),
                    "raw_CEM_action": raw_action.tolist(), "compute_s": compute_s, "solver": solver, "objective": objective})
                previous = action.copy(); previous_vz = objective["base_com_vz"]
                if frame % 25 == 0:
                    # Runner-owned progress checkpoint is not a completed rollout.
                    np.savez_compressed(output/"checkpoint.npz", **{
                        name: np.asarray([row[name] for row in records]) for name in records[0]})
                    report.update(completed_policy_steps=len(decisions), native_samples=len(records),
                        duration_s=float(data.time), last_progress_utc=utc_now(),
                        checkpoint_sha256=sha256(output/"checkpoint.npz"))
                    write_json(output/"report.json", report)
                    print(f"{label} mu={mu:g} CEM frame={frame}/{steps} compute_ms={1000*compute_s:.2f}", flush=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), completed_policy_steps=len(decisions), finished_utc=utc_now())
        report["plant_warning_counts"] = [int(item.number) for item in data.warning]
        if records:
            np.savez_compressed(output/"failed_prefix.npz", **{name: np.asarray([row[name] for row in records]) for name in records[0]})
            report["failed_prefix_sha256"] = sha256(output/"failed_prefix.npz")
        write_json(output/"failed_solver_log.json", decisions)
        write_json(output/"report.json", report)
        raise
    trace = {name: np.asarray([row[name] for row in records]) for name in records[0]}
    trace.update(actions=np.asarray([row["action"] for row in decisions]),
        compute_s=np.asarray([row["compute_s"] for row in decisions]),
        stage_cost=np.asarray([row["objective"]["stage_cost"] for row in decisions]),
        stage_reward=np.asarray([row["objective"]["total_reward"]*.02 for row in decisions]),
        physics_dt_s=np.array(.0025), policy_dt_s=np.array(.02),
        action_joint_names=np.array(JOINT_NAMES), body_names=np.array(BODY_NAMES))
    np.savez_compressed(output/"rollout.npz", **trace)
    write_json(output/"solver_log.json", decisions)
    report.update(status="completed" if steps == 349 else "smoke_completed", completed_policy_steps=steps,
        native_samples=len(records), duration_s=float(data.time), wall_s=time.perf_counter()-started,
        compute_ms_median_p95_max=(1000*np.quantile(trace["compute_s"], [.5, .95, 1.])).tolist(),
        compute_over_control_dt_fraction=float(np.mean(trace["compute_s"] > .02)),
        solver_failures=sum(bool(row["solver"]["optimization_failed"]) for row in decisions),
        plant_warning_counts=[int(item.number) for item in data.warning],
        actual_CEM_reward_sum=float(trace["stage_reward"].sum()),
        rollout_sha256=sha256(output/"rollout.npz"), solver_log_sha256=sha256(output/"solver_log.json"))
    report["finished_utc"] = utc_now()
    write_json(output/"report.json", report)
    print(json.dumps({name: report[name] for name in ("condition", "status", "duration_s", "compute_ms_median_p95_max", "solver_failures")}), flush=True)
    return report


def job_id(label, mu):
    return "cem_"+label+"_mu"+format(float(mu), "g").replace(".", "p")


def validate_completed(directory, label, mu, settings, initial, max_steps=None, native_library_sha256=None):
    """Strict native prefix/full completion check; never promotes running/failed data."""
    directory = Path(directory)
    report = read_json(directory/"report.json")
    steps = 349 if max_steps is None else max_steps
    expected_status = "completed" if steps == 349 else "smoke_completed"
    require(report["kind"] == "400hz_friction_sweep_CEM_with_completed_PPO_stage_reward"
        and report["status"] == expected_status and report["condition"] == label
        and report["engine"] == "native CPU MuJoCo" and report["mujoco_version"] == "3.8.1"
        and report["plant_mu"] == mu and report["prediction_mu_scenarios"] == [mu]
        and report["prediction_friction_known"] is True, "Require genuine completed known-mu native CEM episode")
    require(report["max_policy_steps"] == report["completed_policy_steps"] == steps
        and report["native_samples"] == 8*steps+1
        and np.isclose(report["duration_s"], .02*steps, rtol=0, atol=1e-10)
        and report["truncated_smoke"] == (steps != 349)
        and report["physics_dt_s"] == .0025 and report["policy_dt_s"] == .02 and report["decimation"] == 8,
        "Incomplete or altered native clock")
    require(report["CEM_config"] == settings["nmpc"] and report["discount_factor"] == .99
        and report["task_profile"]["contact_friction"] == mu
        and report["task_profile"]["action_rate_weight"] == -.5
        and report["task_profile"]["reward_weights"]["grounded"] == .4
        and report["task_profile"]["reward_weights"]["sideslip"] == (2. if label == "beta_on" else 0.),
        "Frozen controller/reward differs")
    policy_report_path = (paired.DEFAULT_ON/"cases/physx/mujoco/report.json" if label == "beta_on"
                          else paired.DEFAULT_OFF/"off/report.json")
    policy_report = read_json(policy_report_path)
    expected_profile = copy.deepcopy(policy_report["contract"]["task_profile"])
    expected_profile["contact_friction"] = mu
    require(report["task_profile"] == expected_profile
        and report["matched_PPO_report_sha256"] == sha256(policy_report_path), "Full PPO reward/profile binding differs")
    material = copy.deepcopy(policy_report["contact_material_audit"])
    for row in material["geom_friction"]:
        row[0] = mu
    for key in ("plant_contact_material_audit", "prediction_contact_material_audit"):
        for field in ("colliding_geom_ids", "colliding_geom_names", "wheel_geom_ids", "geom_friction",
                      "geom_priority", "geom_condim", "geom_solref", "geom_solimp", "native_plant_readback"):
            require(report[key][field] == material[field], "Recorded friction-only material/nuisance differs: "+field)
        require(report[key]["pair_coefficient"] == mu, "Recorded effective pair friction differs")
    for flag in ("PPO_actions_or_future_states_used_for_initial_plan", "new_training_or_policy_inference",
                 "new_GPU_job", "termination_or_reset_after_initialization", "height_contact_tilt_failure_gate",
                 "PPO_total_reward_recomputed", "controller_tuned_on_this_episode", "real_time_verified"):
        require(report[flag] is False, "Unexpected task scope/gate: "+flag)
    require(report["native_nuisance_arrays_match_nominal_PPO_validation_exactly"] is True
        and report["sliding_mu_only_change_from_nominal"] is True
        and report["model_sha256"] == sha256(paired.DEFAULT_MODEL)
        and report["motion_sha256"] == sha256(paired.DEFAULT_REFERENCE), "Wrong model/reference or nuisance declaration")
    require(report["source_sha256"] == {str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
        "Completed episode runner source differs")
    if native_library_sha256 is not None:
        require(report["native_library_sha256"] == native_library_sha256, "Completed native library differs")
    paths = {name: directory/name for name in ("report.json", "rollout.npz", "solver_log.json")}
    hashes = {name: sha256(path) for name, path in paths.items()}
    require(report["rollout_sha256"] == hashes["rollout.npz"]
        and report["solver_log_sha256"] == hashes["solver_log.json"], "Completed raw/solver hash mismatch")
    with np.load(paths["rollout.npz"], allow_pickle=False) as stored:
        raw = {name: stored[name] for name in stored.files}
    t = raw["time_s"]
    require(t.shape == (steps*8+1,)
        and np.allclose(t, np.arange(len(t))*.0025, rtol=0, atol=1e-10), "Native timestamps differ/reset")
    for name, array in raw.items():
        if np.issubdtype(array.dtype, np.number):
            require(np.isfinite(array).all(), "Nonfinite completed data: "+name)
    require(raw["qpos"].shape == (len(t), 15) and raw["qvel"].shape == (len(t), 14)
        and raw["actions"].shape == (steps, 8)
        and np.max(np.abs(raw["actions"])) <= 1.+1e-9
        and raw["stage_cost"].shape == raw["stage_reward"].shape == raw["compute_s"].shape == (steps,),
        "Invalid native state/action/stage arrays")
    require(np.array_equal(raw["qpos"][0], initial["initial_qpos"])
        and np.array_equal(raw["qvel"][0], initial["initial_qvel"])
        and np.array_equal(raw["qpos"][0], report["initial_state"]["qpos"])
        and np.array_equal(raw["qvel"][0], report["initial_state"]["qvel"]), "Initial PPO reset state differs")
    decisions = json.loads(paths["solver_log.json"].read_text())
    require(isinstance(decisions, list) and len(decisions) == steps
        and all(row["frame"] == k for k, row in enumerate(decisions)),
        "Incomplete decision log")
    require(report["solver_failures"] == sum(bool(row["solver"]["optimization_failed"]) for row in decisions)
        and not any(report["plant_warning_counts"]), "Plant warned or solver failure counts differ")
    return {"status": expected_status, "report_sha256": hashes["report.json"],
            "recording_sha256": hashes["rollout.npz"], "solver_log_sha256": hashes["solver_log.json"],
            "native_samples": len(t), "completed_policy_steps": steps, "duration_s": float(t[-1])}


def selected_grid(values):
    if values is None:
        return list(GRID)
    require(len(values) > 0 and len(set(values)) == len(values)
        and all(type(value) in (float, int) and value in GRID for value in values),
        "Explicit subset must contain unique frozen-grid mu values")
    return [value for value in GRID if value in values]


def ordered_grid(selected, settings):
    return [value for value in settings["representative_first"] if value in selected]+[
        value for value in selected if value not in settings["representative_first"]]


def source_ledger(settings_file):
    sources = [Path(__file__), Path(settings_file), Path(nominal.__file__), nominal.DEFAULT_SETTINGS]
    sources += [ROOT/"scripts"/name for name in (
        "nmpc_shooting.py", "nmpc_tracking_objective.py", "nmpc_batch_objective.py",
        "nmpc_batch_objective_400hz.py", "nmpc_native_batch_400hz.py", "run_nmpc_tracking.py",
        "classical_tracking.py", "eval_ppo_backend_400hz_mujoco.py", "render_ppo_beta_physx_to_mujoco_400hz.py")]
    sources += [ROOT/"native/nmpc_batch_400hz.cpp"]
    return {str(path.resolve().relative_to(ROOT)): sha256(path) for path in sources}


def initial_states():
    result = {}
    for label in ("beta_on", "beta_off"):
        raw_path = (paired.DEFAULT_ON/"cases/physx/mujoco/rollout.npz" if label == "beta_on"
                    else paired.DEFAULT_OFF/"off/rollout.npz")
        with np.load(raw_path, allow_pickle=False) as raw:
            result[label] = {name: raw[name].copy() for name in ("initial_qpos", "initial_qvel")}
    return result


def run(args):
    output = Path(args.output_dir).resolve()
    resume = bool(getattr(args, "resume", False))
    require(output.exists() if resume else not output.exists(),
        "Explicit --resume requires existing evidence; otherwise fresh output directory required")
    settings, contracts, _, _, evidence = inputs(args.settings)
    selected = selected_grid(getattr(args, "mu", None))
    max_steps = args.max_policy_steps
    require(max_steps is None or type(max_steps) is int and 1 <= max_steps <= 349, "Invalid explicit episode length")
    library = Path(args.native_library).resolve()
    require(library.is_file(), "An explicitly built400Hz native library is required")
    library_sha = sha256(library)
    ledger = source_ledger(args.settings)
    initials = initial_states()
    nominal_bindings = verify_nominal(DEFAULT_NOMINAL)
    scope = {"mu_values_requested": selected, "full_grid_requested": selected == list(GRID),
        "max_policy_steps": max_steps, "native_library_sha256": library_sha,
        "source_sha256": ledger, "settings": settings, "PPO_input_evidence": evidence}
    if resume:
        metadata = read_json(output/"experiment.json")
        require(metadata["kind"] == settings["kind"] and all(metadata[key] == value for key, value in scope.items()),
            "Resume source/settings/library/scope/PPO evidence changed")
        require(metadata["nominal_bindings"] == nominal_bindings, "Resume nominal evidence binding changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        jobs = []
        for mu in ordered_grid(selected, settings):
            for label in settings["beta_conditions"]:
                reused = mu == .7
                directory = DEFAULT_NOMINAL/label if reused else output/job_id(label, mu)
                job = {"id": job_id(label, mu), "controller": "MPC-CEM", "condition": label,
                    "mu": mu, "status": "reused_completed_nominal" if reused else "pending",
                    "directory": str(directory.resolve()), "raw_file": "rollout.npz", "report_file": "report.json",
                    "reused_nominal": reused, "new_episode": not reused}
                if reused:
                    files = nominal_bindings[label]["file_sha256"]
                    job.update(report_sha256=files["report.json"], recording_sha256=files["rollout.npz"],
                        solver_log_sha256=files["solver_log.json"], native_samples=2793,
                        completed_policy_steps=349, duration_s=6.98)
                jobs.append(job)
        metadata = {"schema_version": 1, "kind": settings["kind"], "status": "running",
            "started_utc": utc_now(), **scope, "jobs": jobs, "nominal_bindings": nominal_bindings,
            "new_PPO_training": False, "new_GPU_job": False, "remote_job": False,
            "N_per_condition": 1, "max_concurrent_local_CPU_episodes": 2,
            "prediction_mu_source": "explicit plant friction oracle; not identification/adaptation",
            "comparison_is_optimizer_only": False, "PPO_episode_returns_available": False,
            "stage_rewards_copied_from_completed_PPO": True, "failure_criteria_change": False,
            "automatic_failed_job_retry": False, "completed_scope_includes_full17grid": False}
        write_json(output/"experiment.json", metadata)
    expected_jobs = [(job_id(label, mu), label, mu) for mu in ordered_grid(selected, settings)
                     for label in settings["beta_conditions"]]
    require([(job["id"], job["condition"], job["mu"]) for job in metadata["jobs"]] == expected_jobs,
        "Resume job schedule changed")
    pending = []
    for job in metadata["jobs"]:
        directory = Path(job["directory"])
        require(directory == ((DEFAULT_NOMINAL/job["condition"]).resolve() if job["reused_nominal"]
                              else output/job["id"]), "Job directory escaped fixed schedule")
        if job["reused_nominal"]:
            require(job["mu"] == .7 and job["status"] == "reused_completed_nominal", "Nominal reuse mislabeled")
            files = nominal_bindings[job["condition"]]["file_sha256"]
            require(job["report_sha256"] == files["report.json"]
                and job["recording_sha256"] == files["rollout.npz"]
                and job["solver_log_sha256"] == files["solver_log.json"], "Reused nominal job ledger changed")
            continue
        if directory.exists():
            # A process interrupted before a terminal report is diagnostic data,
            # not an implicit instruction to retry the episode.
            report_path = directory/"report.json"
            report = read_json(report_path) if report_path.is_file() else {}
            if report.get("status") in ("completed", "smoke_completed"):
                try:
                    job.update(validate_completed(directory, job["condition"], job["mu"], settings,
                        initials[job["condition"]], max_steps, library_sha))
                except Exception as error:
                    job.update(status="failed_invalid_evidence", error_type=type(error).__name__, error=str(error))
            elif report.get("status") == "failed":
                job.update(status="failed", error_type=report.get("error_type"), error=report.get("error"))
            else:
                job.update(status="interrupted_preserved", error="No completed terminal report; no automatic retry")
        elif job["status"] == "pending":
            pending.append(job)
        else:
            job.update(status="failed_missing_evidence", error="Previously started job directory is missing")
    metadata.update(status="running", last_update_utc=utc_now(), resumed=resume)
    write_json(output/"experiment.json", metadata)

    def submit(executor, job):
        job.update(status="running", started_utc=utc_now())
        future = executor.submit(run_episode, Path(job["directory"]), job["condition"], settings,
            contracts[job["condition"]], initials[job["condition"]], library, job["mu"], max_steps)
        return future

    # Two in flight only. Each worker owns a separate plant, predictor native
    # subprocess, CEM RNG(seed42), history, previous action and warm start.
    queue = iter(pending)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="CEM400_mu") as executor:
        active = {}
        for _ in range(2):
            job = next(queue, None)
            if job is not None:
                active[submit(executor, job)] = job
        write_json(output/"experiment.json", metadata)
        while active:
            future = next(as_completed(tuple(active)))
            job = active.pop(future)
            try:
                future.result()
                job.update(validate_completed(job["directory"], job["condition"], job["mu"], settings,
                    initials[job["condition"]], max_steps, library_sha))
            except BaseException as error:
                job.update(status="failed", error_type=type(error).__name__, error=str(error))
            job["finished_utc"] = utc_now()
            next_job = next(queue, None)
            if next_job is not None:
                active[submit(executor, next_job)] = next_job
            metadata["last_update_utc"] = utc_now()
            write_json(output/"experiment.json", metadata)
    valid = all(job["status"] in ("completed", "reused_completed_nominal") for job in metadata["jobs"])
    smoke_valid = all(job["status"] in ("smoke_completed", "reused_completed_nominal") for job in metadata["jobs"])
    metadata.update(status=("completed" if valid else "smoke_completed" if smoke_valid else "completed_with_failures"),
        finished_utc=utc_now(), completed_scope_includes_full17grid=valid and selected == list(GRID),
        completed_full_episode_count=sum(job["status"] in ("completed", "reused_completed_nominal") for job in metadata["jobs"]))
    write_json(output/"experiment.json", metadata)
    return metadata


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    result.add_argument("--native-library", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-policy-steps", type=int)
    result.add_argument("--mu", type=float, nargs="+", help="Explicit grid subset; not a full17-point sweep")
    result.add_argument("--resume", action="store_true", help="Validate/reuse completed jobs, preserve failed prefixes; no auto retries")
    return result


if __name__ == "__main__":
    run(parser().parse_args())

