"""New nominal400Hz CEM runs with the existing completed PPO stage reward.

The SAME native MuJoCo plant and recorded initial state are used. This does
not make CEM's privileged state/reference/model access or finite horizon equal
to PPO's observation interface and learned long-term value. Prior PPO records
do not contain all body contact histories, so no exact PPO episode-return
recomputation is claimed. No GPU, remote write or policy inference occurs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import copy
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import render_ppo_beta_physx_to_mujoco_400hz as paired
from controller_references import get_reference
from eval_ppo_backend_400hz_mujoco import audit_materials
from nmpc_shooting import CEMConfig, ShootingNMPC
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState
from ppo_backend_400hz_contract import require, read_json, sha256
from run_nmpc_tracking import MujocoTrackingPredictor, build_provider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ROOT/"config/ppo_cem_matched_reward_400hz_v1.json"


def write_json(path, value):
    """Update this runner's own status under its fresh-output guard only."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def inputs(settings_file=DEFAULT_SETTINGS):
    settings = read_json(settings_file)
    require(settings["schema_version"] == 1
            and settings["kind"] == "same_stage_reward_400hz_ppo_cem_nominal_comparison"
            and settings["reference_id"] == "drift_drl_traj4"
            and settings["validation_backend"] == "native CPU MuJoCo"
            and settings["physics_dt_s"] == .0025 and settings["decimation"] == 8
            and settings["policy_dt_s"] == .02 and settings["plant_mu"] == .7
            and settings["prediction_mu_scenarios"] == [.7]
            and settings["action_rate_weight"] == -.5 and settings["grounded_reward_weight"] == .4
            and settings["discount_factor"] == .99 and settings["beta_conditions"] == ["beta_on", "beta_off"],
            "Require frozen400Hz nominal same-stage-reward comparison")
    traces, reference, evidence = paired.load_comparison()
    require(evidence["reference_sha256"] == sha256(paired.DEFAULT_REFERENCE)
            and evidence["model_sha256"] == sha256(paired.DEFAULT_MODEL), "Strict PPO asset binding differs")
    require(all(row["checkpoint_sha256"] == paired.ON_CHECKPOINT if row["id"] == "beta_on"
                else row["checkpoint_sha256"] == paired.OFF["checkpoint"] for row in evidence["rows"]),
            "Require completed PhysX-trained PPO actors, not alternate training origins")
    # This contract is already SHA-bound by the strict completed-PPO loader.
    source = paired.DEFAULT_ON/"cases/physx/physx/policy_contract.json"
    on_contract = read_json(source)
    off_contract = read_json(paired.DEFAULT_OFF/"off/evaluation_inputs/contract.json")
    contracts = {"beta_on": on_contract, "beta_off": off_contract}
    on_profile = copy.deepcopy(on_contract["trained_task_profile"])
    off_profile = copy.deepcopy(off_contract["trained_task_profile"])
    compare = copy.deepcopy(off_profile); compare["reward_weights"]["sideslip"] = 2.
    require(compare == on_profile, "Only the direct beta stage-reward weight may differ")
    for label, contract in contracts.items():
        require(contract["physics_dt"] == .0025 and contract["decimation"] == 8
                and contract["task_profile"]["contact_friction"] == .7
                and contract["task_profile"]["action_rate_weight"] == -.5
                and contract["task_profile"]["reward_weights"]["grounded"] == .4,
                "PPO trained/deployed task does not match400Hz comparison: "+label)
    return settings, contracts, traces, reference, evidence


def configure_native_model(model, expected_materials):
    """Use the actual PPO native material mapping and verify compiled arrays."""
    audit = audit_materials(model, {}, {"mu": .7})
    expected_readback = expected_materials["native_plant_readback"]
    for name in ("colliding_geom_ids", "colliding_geom_names", "wheel_geom_ids", "geom_friction",
                 "geom_priority", "geom_condim", "geom_solref", "geom_solimp"):
        require(audit[name] == expected_materials[name], "Native PPO/CEM contact material mismatch: "+name)
    for name, value in expected_readback["readbacks"].items():
        observed = np.asarray(getattr(model, name))
        expected = np.asarray(value["values"])
        require(observed.shape == tuple(value["shape"])
                and np.array_equal(observed, expected), "Native PPO/CEM plant nuisance mismatch: "+name)
    for name, expected in expected_readback["solver_configuration"].items():
        require(float(getattr(model.opt, name)) == expected, "Native solver mismatch: "+name)
    return audit


def run_episode(output, label, settings, contract, raw_policy, native_library, max_steps=None):
    from nmpc_batch_objective_400hz import NativeBatchTrackingPredictor400Hz
    require(not output.exists(), "Preserve existing episode; fresh output required")
    # Full simulator contract is copied, including the actual eight-substep
    # actuator mapping; provider ignores old Q/R and does not execute PPO.
    profile = copy.deepcopy(contract["task_profile"])
    report_source = (paired.DEFAULT_ON/"cases/physx/mujoco/report.json" if label == "beta_on"
                     else paired.DEFAULT_OFF/"off/report.json")
    policy_report = read_json(report_source)
    expected = policy_report["contact_material_audit"]
    provider = build_provider(paired.DEFAULT_MODEL, paired.DEFAULT_REFERENCE, contract, settings, (.7,))
    scalar = MujocoTrackingPredictor(provider, paired.DEFAULT_MODEL, profile, (.7,))
    # Exact native wheel priority/contact law is important even for equal mu:
    # wheel-priority also selects solref/solimp when contacts are unequal.
    prediction_materials = configure_native_model(scalar.models[.7], expected)
    scalar.measure_model = scalar.models[.7]
    model, _ = provider.api.load_model(paired.DEFAULT_MODEL, no_visual_mesh=True)
    provider.api.configure_model(model, provider.contract)
    model.geom_friction[:, 0] = .7
    plant_materials = configure_native_model(model, expected)
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
    report = {"schema_version": 1, "kind": "400hz_CEM_with_completed_PPO_stage_reward",
        "status": "running", "condition": label, "engine": "native CPU MuJoCo", "mujoco_version": mujoco.__version__,
        "reference_id": settings["reference_id"], "reference_spec": reference_spec,
        "model_sha256": sha256(paired.DEFAULT_MODEL), "motion_sha256": sha256(paired.DEFAULT_REFERENCE),
        "matched_PPO_report_sha256": sha256(report_source), "task_profile": profile,
        "objective_contract": scalar.objective.contract(), "physics_dt_s": .0025, "policy_dt_s": .02, "decimation": 8,
        "plant_mu": .7, "prediction_mu_scenarios": [.7], "prediction_friction_known": True,
        "CEM_config": asdict(cem), "discount_factor": settings["discount_factor"],
        "discounted_objective": "sum_h gamma**h * dt * (reward_upper_bound - total_stage_reward); fixed horizon, no terminal value",
        "plant_contact_material_audit": plant_materials, "prediction_contact_material_audit": prediction_materials,
        "prediction_state_contract": scalar.layout.contract(),
        "prediction_reset_caveat": "Predictor cold-resets solver warmstart/time each20ms stage; validation plant integrates continuously. Full hidden-contact-state equivalence is NOT claimed even within MuJoCo.",
        "initial_state": initial, "PPO_actions_or_future_states_used_for_initial_plan": False,
        "new_training_or_policy_inference": False, "new_GPU_job": False, "native_contact_substeps": 8,
        "termination_or_reset_after_initialization": False, "height_contact_tilt_failure_gate": False,
        "native_plant_arrays_match_PPO_validation_exactly": True, "state_access": "Full qpos/qvel, objective memory and future reference; PPO actor has a51dim observation",
        "PPO_total_reward_recomputed": False, "PPO_total_reward_unavailable_reason": "Saved PPO traces lack all-body contact force/history; do not manufacture an exact episode-return comparison.",
        "reward_parity_scope": "Matched mathematical15term stage function and recorded weights/stds; native contact observations and runtime softlimits need not be bitwise identical to PhysX training.",
        "soft_joint_limits_source": extractor.soft_limits_source,
        "soft_joint_pos_limits": extractor.soft_limits.tolist(),
        "N_per_condition": 1, "max_policy_steps": steps, "truncated_smoke": steps != 349,
        "concurrent_local_CPU_episodes": settings["concurrent_local_CPU_episodes"],
        "timing_caveat": "End-to-end measured solve time includes IPC and two local CPU episode jobs; not isolated hardware-matched PPO inference latency.",
        "real_time_verified": False, "controller_tuned_on_this_episode": False}
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
            controller = ShootingNMPC(predictor, cem, action_low=-np.ones(8), action_high=np.ones(8), mu_scenarios=(.7,))
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
                    print(f"{label} CEM frame={frame}/{steps} compute_ms={1000*compute_s:.2f}", flush=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), completed_policy_steps=len(decisions))
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
    write_json(output/"report.json", report)
    print(json.dumps({name: report[name] for name in ("condition", "status", "duration_s", "compute_ms_median_p95_max", "solver_failures")}), flush=True)
    return report


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing evidence; fresh output directory required")
    settings, contracts, traces, reference, evidence = inputs(args.settings)
    require(Path(args.native_library).is_file(), "An explicitly built400Hz native library is required")
    sources = [Path(__file__), Path(args.settings)] + [ROOT/"scripts"/name for name in
        ("nmpc_shooting.py", "nmpc_tracking_objective.py", "nmpc_batch_objective.py", "nmpc_batch_objective_400hz.py",
         "nmpc_native_batch_400hz.py", "run_nmpc_tracking.py", "classical_tracking.py", "eval_ppo_backend_400hz_mujoco.py",
         "render_ppo_beta_physx_to_mujoco_400hz.py")]+[ROOT/"native/nmpc_batch_400hz.cpp"]
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "kind": settings["kind"], "status": "running", "settings": settings,
        "new_PPO_training": False, "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "PPO_input_evidence": evidence, "stage_rewards_copied_from_completed_PPO": True,
        "PPO_episode_returns_available": False, "comparison_is_optimizer_only": False}
    write_json(output/"experiment.json", metadata)
    require(settings["concurrent_local_CPU_episodes"] == 2, "Declared localCPU execution budget differs")
    jobs = []
    for label in settings["beta_conditions"]:
        raw_path = (paired.DEFAULT_ON/"cases/physx/mujoco/rollout.npz" if label == "beta_on"
                    else paired.DEFAULT_OFF/"off/rollout.npz")
        with np.load(raw_path, allow_pickle=False) as raw:
            initial = {name: raw[name].copy() for name in ("initial_qpos", "initial_qvel")}
        jobs.append((output/label, label, settings, contracts[label], initial, args.native_library, args.max_policy_steps))
    # Separate native subprocesses and MjModel/MjData own each episode; no GPU
    # or shared simulator state. Two nominal conditions run concurrently only.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="CEM400_episode") as executor:
        futures = [executor.submit(run_episode, *job) for job in jobs]
        reports = [future.result() for future in futures]
    metadata.update(status="completed" if all(row["status"] == "completed" for row in reports) else "smoke_completed",
        reports={row["condition"]: sha256(output/row["condition"]/"report.json") for row in reports})
    write_json(output/"experiment.json", metadata)
    return metadata


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    result.add_argument("--native-library", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-policy-steps", type=int)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
