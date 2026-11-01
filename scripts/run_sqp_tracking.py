"""Run a fixed-friction MuJoCo SQP development episode on the shared TRON1 task.

This runner reuses the CEM/MPPI reference, physical actuator, prediction and
15-term score.  It does not train a policy, use an expert action replay, or
claim Isaac/real-time performance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT
from nmpc_batch_objective import NativeBatchTrackingPredictor
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState
from run_nmpc_tracking import MujocoTrackingPredictor, _metrics, build_provider, selected_reference, sha
from sqp_shooting import SQPConfig, SQPController


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    if args.plant_mu != 0.7 or args.prediction_mu != 0.7:
        raise ValueError("This frozen pilot requires plant and prediction mu=0.7")
    if args.max_policy_steps is not None and args.max_policy_steps < 1:
        raise ValueError("max-policy-steps must be positive")
    if not args.native_library.is_file() or not args.native_library.with_suffix(".json").is_file():
        raise FileNotFoundError("Require an existing native library and its build manifest")
    reference_spec = selected_reference(args)
    settings = json.loads((ROOT/"config/nmpc_height_smooth_residual.json").read_text())
    task = json.loads(Path(args.task_profile).read_text())
    contract = json.loads(Path(args.contract).read_text())
    cfg = SQPConfig(horizon_steps=30, control_blocks=10, max_iterations=args.max_iterations,
                    finite_difference_step=args.finite_difference_step)
    with ExitStack() as resources:
        provider = build_provider(args.model, args.motion, contract, settings, (.7,))
        if provider.reference.frames != reference_spec["frames"]:
            raise ValueError("Reference length mismatch")
        predictor = MujocoTrackingPredictor(provider, args.model, task, (.7,))
        batch = resources.enter_context(NativeBatchTrackingPredictor(predictor, args.native_library))
        controller = SQPController(batch, cfg, action_low=-np.ones(8),
                                   action_high=np.ones(8), mu_scenarios=(.7,))
        model, _ = provider.api.load_model(args.model, no_visual_mesh=True)
        provider.api.configure_model(model, provider.contract)
        model.geom_friction[:, 0] = args.plant_mu
        data = mujoco.MjData(model)
        physical = provider.api.MujocoState(model, data, provider.reference)
        physical.initialize()
        np.testing.assert_allclose(np.r_[data.qpos, data.qvel], provider.targets[0], atol=1e-10)
        extractor = MujocoObjectiveState(model)
        history = np.zeros(predictor.layout.history_shape)
        steps = provider.reference.frames if args.max_policy_steps is None else min(
            args.max_policy_steps, provider.reference.frames)
        report = {
            "kind": "full_body_height_smooth_sqp", "engine": "MuJoCo", "status": "running",
            "mujoco_version": mujoco.__version__, "reference_id": reference_spec["reference_id"],
            "role": reference_spec["role"], "stratum": reference_spec["stratum"],
            "model_sha256": sha(args.model), "motion_sha256": sha(args.motion),
            "contract_sha256": sha(args.contract), "reference_catalog_sha256": sha(DEFAULT_CATALOG),
            "task_profile_sha256": sha(args.task_profile),
            "source_sha256": {name: sha(ROOT/name) for name in
                ("scripts/sqp_shooting.py", "scripts/run_sqp_tracking.py",
                 "scripts/run_nmpc_tracking.py", "scripts/nmpc_tracking_objective.py",
                 "scripts/nmpc_batch_objective.py")},
            "native_library_sha256": sha(args.native_library),
            "native_manifest_sha256": sha(args.native_library.with_suffix(".json")),
            "plant_mu": .7, "prediction_mu_scenarios": [.7],
            "prediction_receives_plant_mu": False,
            "settings": asdict(cfg), "common_15_term_physical_rollout_score_unchanged": True,
            "optimization_objective": "unregularized physical task score, bounded residual SLSQP",
            "prediction_backend": "native_batch_mujoco", "trained_policy": False,
            "learned_policy_used": False, "tvlqr_used": False, "action_replay": False,
            "real_time_verified": False, "sim2sim_comparison": False,
            "total_reference_policy_steps": provider.reference.frames,
            "evaluated_policy_steps": steps, "truncated_smoke": steps < provider.reference.frames,
            "reference_evaluation_core_interval_s": reference_spec["evaluation_core_interval_s"],
            "state_access": "Full qpos/qvel, contact history and reference preview",
            "solver_success_is_not_tracking_success": True,
        }
        output.mkdir(parents=True)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        records, decisions, surrogates = [], [], set()
        started = time.perf_counter()
        held_action = np.zeros(8)
        held_frame = 0

        def record(current=None):
            measured = extractor.extract(data, contact_force_history=history,
                                         body_net_forces_w=current)
            slip = physical.ground_contact_slip(force_threshold_n=5.)
            count, force = physical.nonwheel_ground_contacts()
            records.append({
                "time_s": float(data.time),
                "root_pos": measured["body_pos_w"][0].copy(),
                "root_quaternion_wxyz": measured["body_quat_w"][0].copy(),
                "base_com_velocity_w": measured["body_lin_vel_w"][0].copy(),
                "base_angular_velocity_w": measured["body_ang_vel_w"][0].copy(),
                "wheel_pos": measured["body_pos_w"][-2:].copy(),
                "joint_pos": measured["joint_pos"].copy(),
                "joint_vel": data.qvel[physical.vids].copy(),
                "applied_joint_torque_nm": data.ctrl[physical.aids].copy(),
                "held_action": held_action.copy(),
                "reference_frame_for_action": held_frame,
                "action_was_applied": bool(records),
                "nonwheel_ground_contact_count": count,
                "nonwheel_ground_contact_force_norm_n": force,
                "body_net_forces_w": measured["body_net_forces_w"].copy(),
                "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), **slip,
            })
            return measured

        previous = np.zeros(8)
        pre = record(np.zeros((len(BODY_NAMES), 3)))
        previous_vz = float(pre["body_lin_vel_w"][0, 2])
        try:
            for k in range(steps):
                packed = predictor.layout.pack(data.qpos, data.qvel, history, k == 0)
                held = predictor.objective.hold_reference(predictor.reference_frame(k), pre)
                tic = time.perf_counter()
                action, info = controller.action(packed, k, previous_action=previous,
                                                 previous_vz=previous_vz)
                elapsed = time.perf_counter()-tic
                if action.shape != (8,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.+1e-9:
                    raise FloatingPointError("SQP returned an invalid action")
                raw_action = action.copy()
                action = action.astype(np.float32).astype(float)
                held_action, held_frame = action.copy(), min(k, provider.reference.frames-1)
                for _ in range(provider.contract["decimation"]):
                    data.ctrl[physical.aids] = provider.torque(action, data.qpos, data.qvel)
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                        raise RuntimeError("Unexpected external force")
                    history, current = extractor.update_contact_history(data, history)
                    pre = record(current)
                    expected = (len(records)-1)*model.opt.timestep
                    if not np.isclose(data.time, expected, atol=1e-8, rtol=0):
                        raise FloatingPointError("Physical clock mismatch")
                objective = predictor.objective.evaluate(pre, held, action, previous, previous_vz,
                                                         reset_sample=k == 0)
                surrogates.update(objective["surrogate_flags"])
                decisions.append({"frame": k, "action": action.tolist(),
                    "raw_sqp_action": raw_action.tolist(), "compute_s": elapsed,
                    "solver": info, "objective": objective})
                previous, previous_vz = action.copy(), objective["base_com_vz"]
                if k % 25 == 0:
                    print(f"sqp frame={k}/{steps} ms={1000*elapsed:.1f} cost={info['best_cost']}", flush=True)
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error),
                          completed_policy_steps=len(decisions), wall_s=time.perf_counter()-started)
            (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            raise
        trace = {name: np.asarray([row[name] for row in records]) for name in records[0]}
        trace.update(actions=np.asarray([row["action"] for row in decisions]),
                     compute_s=np.asarray([row["compute_s"] for row in decisions]),
                     stage_cost=np.asarray([row["objective"]["stage_cost"] for row in decisions]),
                     reference_time_s=np.arange(provider.reference.frames)*provider.dt,
                     reference_root_xy_m=provider.reference.archive["body_pos_w"][:, 0, :2],
                     body_names=np.array(BODY_NAMES), action_joint_names=np.array(JOINT_NAMES),
                     reference_id=np.array(reference_spec["reference_id"]),
                     physics_dt_s=np.array(model.opt.timestep), policy_dt_s=np.array(provider.dt))
        report.update(_metrics(trace, provider.reference, reference_spec))
        np.savez_compressed(output/"rollout.npz", **trace)
        (output/"solver_log.json").write_text(json.dumps(decisions, indent=2, allow_nan=False)+"\n")
        timing = trace["compute_s"]
        report.update(status="smoke_completed" if steps < provider.reference.frames else "completed",
                      completed_policy_steps=steps, duration_s=float(data.time),
                      wall_s=time.perf_counter()-started, physics_samples=len(records),
                      compute_ms_median_p95_max=(1000*np.quantile(timing, [.5, .95, 1.])).tolist(),
                      compute_over_control_dt_fraction=float(np.mean(timing > provider.dt)),
                      candidate_evaluations=sum(row["solver"]["candidate_evaluations"] for row in decisions),
                      sqp_successes=sum(row["solver"]["slsqp_success"] for row in decisions),
                      solver_failures=sum(row["solver"]["optimization_failed"] for row in decisions),
                      total_stage_cost=float(trace["stage_cost"].sum()),
                      surrogate_flags=sorted(surrogates),
                      rollout_sha256=sha(output/"rollout.npz"),
                      solver_log_sha256=sha(output/"solver_log.json"))
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key: report[key] for key in
            ("status", "completed_policy_steps", "core_coverage_complete", "core_xy_rmse_m",
             "compute_ms_median_p95_max", "candidate_evaluations", "sqp_successes")}), flush=True)
        return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "motion", "contract", "native-library", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--reference-id", default=PRIMARY_REFERENCE_ID)
    p.add_argument("--task-profile", type=Path,
                   default=ROOT/"config/car_drift_height_smooth_task.json")
    p.add_argument("--plant-mu", type=float, default=.7)
    p.add_argument("--prediction-mu", type=float, default=.7)
    p.add_argument("--max-policy-steps", type=int)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--finite-difference-step", type=float, default=.002)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
