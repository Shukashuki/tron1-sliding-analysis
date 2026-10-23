"""New fixed nominal CEM experiment with opt-in V2f any-wheel ground support.

No new PPO training/inference, tuning, remote writes, GPU, retries or implicit
native builds. Historical PPO records supply ONLY the original reset state.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import copy
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import run_ppo_cem_matched_reward_400hz as nominal
from run_ppo_cem_matched_reward_400hz import configure_native_model, write_json
import render_ppo_beta_physx_to_mujoco_400hz as paired
from controller_references import get_reference
from ground_contact_reward_v2f import reward_numpy, formula_metadata, validate_settings
from nmpc_native_ground_contact_400hz import ground_contact_readout, model_weight_n, resolve_floor_geom, _validated_library
from nmpc_shooting import CEMConfig, ShootingNMPC
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState, HeightSmoothTrackingObjective
from ppo_backend_400hz_contract import require, read_json, sha256
from run_nmpc_tracking import MujocoTrackingPredictor, build_provider

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "ceded934984b00341d375ebb1e8c888d42fae203b2b58680d29bd8f5cefdf42b"


def replace_actual_grounded(legacy, ground_window_w, mg):
    """New actual-plant reward, never mislabel the legacy grounded reward."""
    result = copy.deepcopy(legacy)
    require(len(result["raw_terms"]) == len(result["weighted_terms"]) == 15, "Require exact legacy15-term base")
    force = np.asarray(ground_window_w, dtype=float)
    require(force.shape == (8, 2, 3) and np.isfinite(force).all()
            and np.max(np.abs(force[..., :2])) <= 1e-7, "Need all eight actual floor NORMAL samples")
    components = reward_numpy(force[..., 2], mg, result["raw_terms"]["motion_global_anchor_pos"],
                              result["raw_terms"]["drift_speed"])
    raw = float(components["raw_reward"])
    result["raw_terms"]["drift_grounded"] = raw
    result["weighted_terms"]["drift_grounded"] = .4*raw
    if "term_costs" in result:
        result["term_costs"]["drift_grounded"] = .02*(.4-.4*raw)
    result["total_reward"] = sum(result["weighted_terms"].values())
    result["stage_cost"] = .02*(result["reward_upper_bound"]-result["total_reward"])
    result["ground_contact_v2"] = {name: np.asarray(value).tolist() for name, value in components.items()}
    result["grounded_reward_replaced_not_legacy"] = True
    return result


def recompute_actual_rewards(raw, report, decisions, reference_file=paired.DEFAULT_REFERENCE):
    """Reconstruct all15 actual stage terms including eight-sample V2 replacement."""
    steps = len(decisions)
    require(raw["qpos"].shape == (steps*8+1, 15), "Incomplete native rewards cannot be audited")
    require(np.allclose(raw["time_s"], np.arange(steps*8+1)*.0025, rtol=0, atol=1e-10), "Reward audit clock differs")
    require(report["ground_contact_reward_v2"] == formula_metadata(), "Reward audit V2 formula differs")
    objective = HeightSmoothTrackingObjective(report["task_profile"], dt=.02,
        soft_joint_limits=report["soft_joint_pos_limits"])
    with np.load(reference_file, allow_pickle=False) as archive:
        names = tuple(archive["body_names"].tolist())
        ids = [names.index(name) for name in BODY_NAMES]
        reference = {name: archive[name][:, ids].copy() for name in
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    def state(index):
        value = {name: np.asarray(raw[name][index]) for name in
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_net_forces_w", "contact_force_history_w")}
        value.update(body_names=BODY_NAMES, joint_pos=raw["joint_pos"][index])
        return value
    previous = np.zeros(8); previous_vz = float(raw["body_lin_vel_w"][0, 0, 2])
    stages, rewards = [], []
    for frame, decision in enumerate(decisions):
        require(decision["frame"] == frame and np.array_equal(decision["action"], raw["actions"][frame]), "Action/frame reward audit differs")
        start, end = frame*8, (frame+1)*8
        force = np.asarray(raw["ground_normal_force_w"][start+1:end+1])
        require(np.array_equal(force, raw["ground_normal_force_window_w"][frame])
                and np.array_equal(raw["ground_contact_count"][start+1:end+1], raw["ground_contact_count_window"][frame]),
                "Stored actual ground windows differ from native records")
        held = objective.hold_reference({name: value[frame] for name, value in reference.items()}, state(start))
        legacy = objective.evaluate(state(end), held, raw["actions"][frame], previous, previous_vz, reset_sample=frame == 0)
        recomputed = replace_actual_grounded(legacy, force, report["model_weight_n"])
        expected = copy.deepcopy(decision["objective"])
        for value in (recomputed, expected):
            value.pop("surrogate_flags", None)
        require(recomputed == expected, "Actual fifteen-term V2 stage reward does not exactly reproduce")
        stages.append(recomputed["stage_cost"]); rewards.append(.02*recomputed["total_reward"])
        previous, previous_vz = raw["actions"][frame], recomputed["base_com_vz"]
    require(np.array_equal(raw["stage_cost"], stages) and np.array_equal(raw["stage_reward"], rewards), "Saved V2 cost/reward arrays differ")
    return {"status": "verified_exact", "stages": steps, "reward_terms": 15,
            "V2_grounded_used_for_actual_plant_and_planner": True, "native_substeps_per_stage": 8,
            "actual_dt_scaled_reward_sum": float(np.sum(rewards)), "PPO_return_recomputed": False}


def run_episode(output, label, settings, contract, raw_policy, native_library, max_steps=None, *, source_ledger=None):
    from nmpc_batch_ground_contact_v2f_400hz import NativeGroundContactTrackingPredictor400Hz
    require(not output.exists(), "Preserve existing episode; fresh output required")
    # Full simulator contract is copied, including the actual eight-substep
    # actuator mapping; provider ignores old Q/R and does not execute PPO.
    require(label == "beta_on", "V2 nominal runner is betaON only")
    profile = copy.deepcopy(contract["task_profile"])
    profile["ground_contact_reward_v2"] = validate_settings()
    contract = copy.deepcopy(contract)
    contract["task_profile"] = copy.deepcopy(profile)
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
    floor = resolve_floor_geom(model)
    mg = model_weight_n(model)
    require(mg == model_weight_n(scalar.models[.7]), "Prediction/plant native weight differs")
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
    report = {"schema_version": 1, "kind": "400hz_CEM_ground_contact_reward_v2f_anywheel",
        "source_sha256": copy.deepcopy(source_ledger or {}),
        "reward_definition_version": "ground_contact_v2f_any_wheel",
        "protocol_sha256": sha256(ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json"),
        "native_library_sha256": sha256(native_library),
        "status": "running", "condition": label, "engine": "native CPU MuJoCo", "mujoco_version": mujoco.__version__,
        "reference_id": settings["reference_id"], "reference_spec": reference_spec,
        "model_sha256": sha256(paired.DEFAULT_MODEL), "motion_sha256": sha256(paired.DEFAULT_REFERENCE),
        "matched_PPO_report_sha256": sha256(report_source), "task_profile": profile,
        "base_objective_contract": scalar.objective.contract(),
        "ground_contact_reward_v2": formula_metadata(), "ground_reward_external_weight": .4,
        "reward_term_count": 15, "other_reward_terms_unchanged": 14, "PPO_V2_trained_policy_claimed": False,
        "model_weight_n": mg, "ground_floor_geom_id": floor, "ground_floor_geom_name": model.geom(floor).name,
        "floor_filter_verified": True, "ground_force_kind": "floor-only NORMAL world force acting on each wheel",
        "ground_count_semantics": "floor pair normal force >=1e-6N; not legacy>5N", "physics_dt_s": .0025, "policy_dt_s": .02, "decimation": 8,
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
        "reward_parity_scope": "NEW grounded replacement; fourteen other legacy stage terms unchanged. Not the old PPO trained reward and not bitwise cross-engine force parity.",
        "soft_joint_limits_source": extractor.soft_limits_source,
        "soft_joint_pos_limits": extractor.soft_limits.tolist(),
        "N_per_condition": 1, "max_policy_steps": steps, "truncated_smoke": steps != 349,
        "concurrent_local_CPU_episodes": 1,
        "timing_caveat": "End-to-end measured solve time includes IPC; no realtime or hardware-matched PPO inference claim.",
        "real_time_verified": False, "controller_tuned_on_this_episode": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"report.json", report)
    records, decisions = [], []
    held_action = np.zeros(8)
    held_frame = 0

    def record(current):
        state = extractor.extract(data, contact_force_history=history, body_net_forces_w=current)
        ground = ground_contact_readout(model, data)
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
            "nonwheel_ground_contact_count": count, "nonwheel_ground_contact_force_norm_n": force, **slip, **ground})
        return state

    previous = np.zeros(8)
    pre = record(np.zeros((len(BODY_NAMES), 3)))
    previous_vz = float(pre["body_lin_vel_w"][0, 2])
    started = time.perf_counter()
    try:
        with NativeGroundContactTrackingPredictor400Hz(scalar, native_library, discount_factor=settings["discount_factor"]) as predictor:
            controller = ShootingNMPC(predictor, cem, action_low=-np.ones(8), action_high=np.ones(8), mu_scenarios=(.7,))
            predictor.record_cem_wiring(controller)
            report["native_backend"] = predictor.metadata
            report["objective_contract"] = predictor.objective.contract()
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
                legacy_objective = scalar.objective.evaluate(pre, held, action, previous, previous_vz, reset_sample=frame == 0)
                ground_window = np.asarray([row["ground_normal_force_w"] for row in records[-8:]])
                objective = replace_actual_grounded(legacy_objective, ground_window, mg)
                decisions.append({"frame": frame, "time_s": float(data.time), "action": action.tolist(),
                    "raw_CEM_action": raw_action.tolist(), "compute_s": compute_s, "solver": solver, "objective": objective})
                previous = action.copy(); previous_vz = objective["base_com_vz"]
                if frame % 25 == 0:
                    np.savez_compressed(output/"checkpoint.npz", **{name: np.asarray([row[name] for row in records]) for name in records[0]})
                    report.update(completed_policy_steps=len(decisions), native_samples=len(records), duration_s=float(data.time),
                        checkpoint_sha256=sha256(output/"checkpoint.npz"))
                    write_json(output/"report.json", report)
                    print(f"V2f {label} CEM frame={frame}/{steps} compute_ms={1000*compute_s:.2f}", flush=True)
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
        action_joint_names=np.array(JOINT_NAMES), body_names=np.array(BODY_NAMES),
        ground_normal_force_window_w=np.asarray([row["ground_normal_force_w"] for row in records[1:]]).reshape(steps, 8, 2, 3),
        ground_contact_count_window=np.asarray([row["ground_contact_count"] for row in records[1:]]).reshape(steps, 8, 2))
    np.savez_compressed(output/"rollout.npz", **trace)
    write_json(output/"solver_log.json", decisions)
    report.update(status="completed" if steps == 349 else "smoke_completed", completed_policy_steps=steps,
        native_samples=len(records), duration_s=float(data.time), wall_s=time.perf_counter()-started,
        compute_ms_median_p95_max=(1000*np.quantile(trace["compute_s"], [.5, .95, 1.])).tolist(),
        compute_over_control_dt_fraction=float(np.mean(trace["compute_s"] > .02)),
        solver_failures=sum(bool(row["solver"]["optimization_failed"]) for row in decisions),
        plant_warning_counts=[int(item.number) for item in data.warning],
        actual_CEM_reward_sum=float(trace["stage_reward"].sum()), actual_reward_uses_ground_contact_v2=True,
        smoke_only=steps != 349, completed_full_reference=steps == 349,
        rollout_sha256=sha256(output/"rollout.npz"), solver_log_sha256=sha256(output/"solver_log.json"))
    report["actual_reward_audit"] = recompute_actual_rewards(trace, report, decisions)
    write_json(output/"report.json", report)
    print(json.dumps({name: report[name] for name in ("condition", "status", "duration_s", "compute_ms_median_p95_max", "solver_failures")}), flush=True)
    return report


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve evidence; fresh output directory required")
    maximum = getattr(args, "max_policy_steps", None)
    require(maximum is None or type(maximum) is int and 1 <= maximum <= 349, "Explicit smoke length must be1..349")
    require(sha256(ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json") == PROTOCOL_SHA256,
            "Frozen F A-bonus plus any-wheel support-gap protocol differs")
    library, library_metadata, _ = _validated_library(args.native_library)
    settings, contracts, traces, reference, evidence = nominal.inputs()
    expected = {"horizon_steps": 30, "control_blocks": 10, "candidates": 64,
                "iterations": 3, "elite_count": 8, "seed": 42}
    require(all(settings["nmpc"][key] == value for key, value in expected.items()), "Frozen CEM search budget differs")
    require(settings["nmpc"]["initial_std"] == [.04]*6+[.35]*2 and settings["discount_factor"] == .99,
            "Frozen search standard deviations/discount differ")
    paths = [Path(__file__), nominal.DEFAULT_SETTINGS, ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json"] + [ROOT/"scripts"/name for name in
        ("ground_contact_reward_v2.py", "ground_contact_reward_v2b.py", "ground_contact_reward_v2c.py", "ground_contact_reward_v2d.py", "ground_contact_reward_v2f.py", "nmpc_batch_ground_contact_v2f_400hz.py", "nmpc_native_ground_contact_400hz.py",
         "run_ppo_cem_matched_reward_400hz.py", "nmpc_shooting.py", "nmpc_tracking_objective.py",
         "nmpc_batch_objective.py", "nmpc_batch_objective_400hz.py", "nmpc_native_batch_400hz.py",
         "run_nmpc_tracking.py", "classical_tracking.py", "eval_ppo_backend_400hz_mujoco.py",
         "render_ppo_beta_physx_to_mujoco_400hz.py")] + [ROOT/"native/nmpc_batch_ground_contact_400hz.cpp"]
    source_ledger = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    raw_path = paired.DEFAULT_ON/"cases/physx/mujoco/rollout.npz"
    with np.load(raw_path, allow_pickle=False) as raw:
        initial = {name: raw[name].copy() for name in ("initial_qpos", "initial_qvel")}
    report = run_episode(output, "beta_on", settings, contracts["beta_on"], initial, library, maximum,
                         source_ledger=source_ledger)
    require(all(sha256(ROOT/name) == digest for name, digest in source_ledger.items()), "V2 executed sources changed during episode")
    report.update(source_sha256=source_ledger, baseline_reset_recording_sha256=sha256(raw_path),
        native_library_sha256=sha256(library), native_build_manifest_sha256=sha256(library.with_suffix(".json")),
        V2_same_as_old_PPO_training_reward=False, PPO_V2_trained_policy_claimed=False)
    write_json(output/"report.json", report)
    write_json(output/"experiment.json", {"kind": "ground_contact_reward_v2f_CEM_nominal_only", "status": report["status"],
        "source_sha256": source_ledger, "native_abi": library_metadata["abi_version"], "sample_size": 1,
        "new_policy_training_or_inference": False, "automatic_retry": False,
        "report_sha256": sha256(output/"report.json"), "baseline_PPO_input_evidence": evidence})
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--native-library", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-policy-steps", type=int, help="Explicit short smoke; omitted means349decisions")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
