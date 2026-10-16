"""Bounded audit of EXISTING nominal NMPC actions, not controller retuning.

Sixteen independent one-control-interval predictions (64 physics substeps)
are compared with a continuously simulated, already recorded NMPC plant.
No PPO examples, held-out friction cases, closed-loop episode, new CEM search,
native build, fake ground-contact labels, or new controller parameters enter.
Unrecorded qacc_warmstart prevents identifying cold solver state separately
from omitted simulator time; local agreement is not a multi-step certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from nmpc_native_batch import NativeBatchTransition
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState
from run_nmpc_tracking import MujocoTrackingPredictor, build_provider


SELECTION = {
    0: "initial interval",
    19: "early wheel-mode changes around 0.385-0.390 s",
    43: "early wheel-mode changes around 0.865-0.885 s",
    56: "after first logged 0.10 m XY-error crossing",
    72: "turn-core entry and first core wheel-mode change",
    78: "core wheel-contact loss around 1.565 s",
    87: "near first 0.20 m XY-error crossing",
    100: "early core, before nonwheel impact",
    125: "mid core, before nonwheel impact",
    145: "late pre-impact core",
    150: "3.00 s pre-impact high-speed interval",
    154: "immediately before first nonwheel-ground impact interval",
    155: "first nonwheel-ground impact at 3.110 s within this interval",
    156: "immediately after logged beta sign/branch change",
    175: "3.50 s post-impact low-speed state",
    225: "4.50 s post-core tracking loss",
}
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_history_at(forces, sample):
    """Exact logger memory: latest 3 post-forward forces; initial history zero."""
    forces = np.asarray(forces, dtype=float)
    if (forces.ndim != 3 or forces.shape[1:] != (9, 3) or not np.isfinite(forces).all()
            or isinstance(sample, bool) or not isinstance(sample, (int, np.integer))
            or not 0 <= sample < len(forces)):
        raise ValueError("Invalid logged contact-force history/sample")
    result = np.zeros((3, 9, 3))
    if sample:
        values = forces[max(1, sample-2):sample+1]
        result[-len(values):] = values
    return result


def selected_indices(policy_steps):
    if isinstance(policy_steps, bool) or not isinstance(policy_steps, int) or policy_steps <= max(SELECTION):
        raise ValueError("Complete archived primary episode is required for declared sampling")
    indices = np.array(list(SELECTION), dtype=int)
    assert len(indices) == 16 and len(set(indices)) == 16
    return indices


def tangent_error(model, predicted, actual):
    result = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, result, 1., actual, predicted)
    return result


def max_abs(value):
    return float(np.max(np.abs(value)))


def fk_state(model, data, extractor, qpos, qvel, history, logged_forces):
    """Kinematics only; objective forces/history remain measured log evidence."""
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_fwdPosition(model, data)
    return extractor.extract(data, contact_force_history=history, body_net_forces_w=logged_forces)


def analyze(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Preserving existing probe report: {output}")
    episode = Path(args.episode)
    report_file, rollout_file, solver_file = [episode / name for name in ("report.json", "rollout.npz", "solver_log.json")]
    report, solver = json.loads(report_file.read_text()), json.loads(solver_file.read_text())
    with np.load(rollout_file, allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    if (report["status"] != "completed" or report["reference_id"] != "drift_drl_traj4"
            or report["engine"] != "MuJoCo" or report["plant_mu"] != .7
            or report["prediction_mu_scenarios"] != [.7] or report["learned_policy_used"]
            or report["prediction_backend"] != "native"):
        raise ValueError("Only completed primary NOMINAL MuJoCo NMPCv1 is in scope")
    if sha(rollout_file) != report["rollout_sha256"] or sha(solver_file) != report["solver_log_sha256"]:
        raise ValueError("Archived raw trace/log hashes differ from the original report")
    if sha(args.model) != report["model_sha256"] or sha(args.motion) != report["motion_sha256"]:
        raise ValueError("Probe model/motion differs from the archived experiment")
    settings = json.loads(Path(args.settings).read_text())
    task = json.loads(Path(args.task_profile).read_text())
    contract = json.loads(Path(args.contract).read_text())
    required_sources = ("scripts/nmpc_tracking_objective.py", "scripts/nmpc_batch_objective.py",
                        "scripts/nmpc_native_batch.py", "native/nmpc_batch.cpp", "scripts/classical_tracking.py",
                        "scripts/height_smoothness.py", "config/classical_tracking.json")
    matched_sources = {}
    for relative in required_sources:
        expected = [value for name, value in report["source_sha256"].items() if name.endswith("/" + relative)]
        actual = sha(ROOT / relative)
        if expected != [actual]:
            raise ValueError("Physical/objective source revision differs: " + relative)
        matched_sources[relative] = actual
    for path in (args.settings, args.task_profile, args.contract):
        expected = report["source_sha256"].get(str(Path(path).resolve()))
        if expected != sha(path):
            raise ValueError("Frozen settings/profile/contract hash differs")
    if settings != report["settings"]:
        raise ValueError("Nominal settings differ from recorded settings")
    steps = len(solver)
    selected = selected_indices(steps)
    if (steps != 349 or report["completed_policy_steps"] != steps
            or tuple(trace["body_names"].tolist()) != BODY_NAMES
            or tuple(trace["action_joint_names"].tolist()) != JOINT_NAMES
            or trace["qpos"].shape != (4*steps+1, 15)
            or trace["qvel"].shape != (4*steps+1, 14)
            or float(trace["physics_dt_s"]) != .005 or float(trace["policy_dt_s"]) != .02):
        raise ValueError("Unexpected archived physical clock/body/joint layout")
    np.testing.assert_allclose(trace["time_s"], np.arange(4*steps+1)*.005, atol=1e-10, rtol=0)
    if not np.isfinite(trace["actions"]).all() or np.max(np.abs(trace["actions"])) > 1.:
        raise ValueError("Archived applied actions are not finite normalized controls")
    np.testing.assert_array_equal(trace["actions"], trace["actions"].astype(np.float32).astype(float))
    for k in selected:
        if solver[k]["frame"] != k or solver[k]["reference_frame_for_reward"] != k:
            raise ValueError("Logged objective reference must be HELD at action frame k")
        np.testing.assert_array_equal(trace["held_action"][4*k+1:4*k+5], np.tile(trace["actions"][k], (4, 1)))
        np.testing.assert_array_equal(trace["reference_frame_for_action"][4*k+1:4*k+5], [k]*4)
    provider = build_provider(args.model, args.motion, contract, settings, (.7,))
    predictor = MujocoTrackingPredictor(provider, args.model, task, (.7,))
    model = predictor.models[.7]
    measurement = mujoco.MjData(model)
    extractor = MujocoObjectiveState(model)
    initial_sample = 4*selected
    histories = np.array([contact_history_at(trace["body_net_forces_w"], int(i)) for i in initial_sample])
    with NativeBatchTransition(model, provider.q0, contract, args.native_library) as native:
        prediction = native.rollout(trace["qpos"][initial_sample], trace["qvel"][initial_sample],
                                    trace["actions"][selected, None, :], initial_history=histories)
        native_metadata = native.metadata.copy()
    rows = []
    for row, k in enumerate(selected):
        k = int(k); begin, end = 4*k, 4*k+4
        actual_history = contact_history_at(trace["body_net_forces_w"], end)
        action = trace["actions"][k]
        previous = np.zeros(8) if k == 0 else trace["actions"][k-1]
        previous_vz = float(trace["base_com_velocity_w"][begin, 2])
        pre = fk_state(model, measurement, extractor, trace["qpos"][begin], trace["qvel"][begin], histories[row], trace["body_net_forces_w"][begin])
        actual = fk_state(model, measurement, extractor, trace["qpos"][end], trace["qvel"][end], actual_history, trace["body_net_forces_w"][end])
        predicted = fk_state(model, measurement, extractor, prediction["qpos"][row, 0, -1], prediction["qvel"][row, 0, -1],
                             prediction["contact_force_history_w"][row, 0], prediction["body_net_forces_w"][row, 0, -1])
        held = predictor.objective.hold_reference(predictor.reference_frame(k), pre)
        actual_objective = predictor.objective.evaluate(actual, held, action, previous, previous_vz, reset_sample=k == 0)
        predicted_objective = predictor.objective.evaluate(predicted, held, action, previous, previous_vz, reset_sample=k == 0)
        logged = solver[k]["objective"]
        raw_difference = {name: float(predicted_objective["raw_terms"][name] - logged["raw_terms"][name]) for name in logged["raw_terms"]}
        weighted_difference = {name: float(predicted_objective["weighted_terms"][name] - logged["weighted_terms"][name]) for name in logged["weighted_terms"]}
        reconstructed_error = {name: max_abs(np.array([actual_objective[name][term] - logged[name][term] for term in logged[name]]))
                               for name in ("raw_terms", "weighted_terms", "diagnostics")}
        reconstructed_error["stage_cost"] = abs(actual_objective["stage_cost"] - logged["stage_cost"])
        tangent = np.array([tangent_error(model, prediction["qpos"][row, 0, sub], trace["qpos"][begin+sub+1]) for sub in range(4)])
        velocity_error = prediction["qvel"][row, 0] - trace["qvel"][begin+1:end+1]
        torque_error = prediction["torques"][row, 0] - trace["applied_joint_torque_nm"][begin+1:end+1]
        actual_torque, predicted_torque = trace["applied_joint_torque_nm"][begin+1:end+1], prediction["torques"][row, 0]
        if np.any(np.abs(predicted_torque) > np.r_[[80.]*6, [12.]*2] + 1e-9):
            raise AssertionError("Native torque exceeded original 80/12 Nm actuator bounds")
        rows.append({
            "policy_frame": k, "selection": SELECTION[k],
            "actual_interval_s": [float(trace["time_s"][begin]), float(trace["time_s"][end])],
            "held_reference_frame_for_reward": k,
            "cem_plan_warm_start_used": bool(solver[k]["solver"]["warm_start_used"]),
            "actual_root_xy_reference_error_before_after_m": trace["xy_error_m"][[begin, end]].tolist(),
            "measured_ground_wheel_counts_before_after": trace["wheel_ground_contact_count"][[begin, end]].tolist(),
            "measured_nonwheel_ground_count_max_during_interval": int(trace["nonwheel_ground_contact_count"][begin+1:end+1].max()),
            "qpos_tangent_error_each_substep": tangent.tolist(),
            "qpos_tangent_final_l2_mixed_units": float(np.linalg.norm(tangent[-1])),
            "root_xy_error_each_substep_m": np.linalg.norm(prediction["qpos"][row, 0, :, :2]-trace["qpos"][begin+1:end+1, :2], axis=-1).tolist(),
            "root_rotation_tangent_error_final_rad": float(np.linalg.norm(tangent[-1, 3:6])),
            "qvel_error_max_each_substep_mixed_units": np.max(np.abs(velocity_error), axis=-1).tolist(),
            "qvel_error_final_max_mixed_units": max_abs(velocity_error[-1]),
            "root_com_world_velocity_error_final_m_s": (predicted["body_lin_vel_w"][0]-trace["base_com_velocity_w"][end]).tolist(),
            "all_body_link_position_error_final_max_m": max_abs(predicted["body_pos_w"]-actual["body_pos_w"]),
            "all_body_com_world_velocity_error_final_max_m_s": max_abs(predicted["body_lin_vel_w"]-actual["body_lin_vel_w"]),
            "contact_force_history_error_final_max_n": max_abs(prediction["contact_force_history_w"][row, 0]-actual_history),
            "all_contact_net_body_force_error_each_substep_max_n": np.max(np.abs(prediction["body_net_forces_w"][row, 0]-trace["body_net_forces_w"][begin+1:end+1]), axis=(1, 2)).tolist(),
            "torque_error_each_substep_max_nm": np.max(np.abs(torque_error), axis=-1).tolist(),
            "native_max_abs_torque_leg_wheel_nm": [max_abs(predicted_torque[:, :6]), max_abs(predicted_torque[:, 6:])],
            "actual_max_abs_torque_leg_wheel_nm": [max_abs(actual_torque[:, :6]), max_abs(actual_torque[:, 6:])],
            "raw_cem_vs_applied_float32_action_max": max_abs(np.asarray(solver[k]["raw_cem_action"])-action),
            "logged_actual_cost": float(logged["stage_cost"]),
            "predicted_cost_same_applied_action": float(predicted_objective["stage_cost"]),
            "predicted_minus_actual_cost": float(predicted_objective["stage_cost"]-logged["stage_cost"]),
            "predicted_minus_actual_raw_terms": raw_difference,
            "predicted_minus_actual_weighted_terms": weighted_difference,
            "logged_actual_reconstruction_max_errors": reconstructed_error,
        })
    aggregate = {
        "maximum_root_xy_prediction_error_m": max(max(row["root_xy_error_each_substep_m"]) for row in rows),
        "maximum_root_rotation_prediction_error_rad": max(row["root_rotation_tangent_error_final_rad"] for row in rows),
        "maximum_qvel_prediction_error_mixed_units": max(max(row["qvel_error_max_each_substep_mixed_units"]) for row in rows),
        "maximum_root_com_velocity_prediction_error_m_s": max(max_abs(row["root_com_world_velocity_error_final_m_s"]) for row in rows),
        "maximum_contact_history_prediction_error_n": max(row["contact_force_history_error_final_max_n"] for row in rows),
        "maximum_torque_prediction_error_nm": max(max(row["torque_error_each_substep_max_nm"]) for row in rows),
        "maximum_first_substep_torque_error_nm": max(row["torque_error_each_substep_max_nm"][0] for row in rows),
        "maximum_absolute_stage_cost_prediction_error": max(abs(row["predicted_minus_actual_cost"]) for row in rows),
        "maximum_logged_actual_reconstruction_error": max(max(row["logged_actual_reconstruction_max_errors"].values()) for row in rows),
        "maximum_weighted_term_prediction_error": max(max_abs(list(row["predicted_minus_actual_weighted_terms"].values())) for row in rows),
    }
    result = {
        "schema_version": 1, "kind": "bounded_existing_nominal_nmpc_one_step_model_audit",
        "reference_id": "drift_drl_traj4", "mu": .7, "physical_transition_calls": 16,
        "predicted_control_intervals": 16, "predicted_physics_substeps": 64,
        "new_closed_loop_episode": False, "new_cem_search": False, "learned_policy_data_used": False,
        "sampling_rule": "Declared 16 distinct intervals from initial/early wheel modes/core entry/pre-impact/impact/postloss strata of this EXISTING NMPC trace; selected before one-step prediction, not optimized after seeing model error.",
        "selected_policy_frames": selected.tolist(), "aggregate": aggregate, "rows": rows,
        "measurement_contract": {"body_names": list(BODY_NAMES), "joint_names": list(JOINT_NAMES),
                                 "history_shape": [3, 9, 3], "physics_dt_s": .005, "control_dt_s": .02,
                                 "action": "Actual recorded float32-rounded applied normalized action, NOT raw CEM search output",
                                 "forces": "Recorded all-contact net link forces; ground-only contact counts retained separately as measured diagnostics",
                                 "reference": "reference[k] held against actual PRE-action body state; no phase k+1 target",
                                 "plant_clock": "Continuous recorded clock", "prediction_clock": "Cold reset at EACH interval, 0 to .02 s",
                                 "qpos_tangent_units": "Root translations m; root rotations and eight hinge differences rad. Mixed L2 is diagnostic, not a physical norm.",
                                 "qvel_error_units": "Generalized root linear components m/s; rotation/joints rad/s, reported mixed maximum with this caveat."},
        "caveats": [
            "The archived plant trace does NOT contain qacc_warmstart or full mjSTATE_INTEGRATION. Solver warm-start and clock effects cannot be independently isolated or recreated.",
            "CEM shifted-plan warm-start metadata is not MuJoCo solver warm-start state.",
            "One-step agreement on 16 observed actions does not prove multi-step accuracy, counterfactual candidate correctness, local optimality, successful tracking or absence of contact chaos.",
            "No ground contact labels were inferred from FK or net-body forces, and flight is not a task-failure gate.",
            "This probe does not rerun CEM or measure controller performance; no causal attribution to horizon/cost/search is established by local prediction error alone."
        ],
        "archive_sha256": {"report.json": sha(report_file), "rollout.npz": sha(rollout_file), "solver_log.json": sha(solver_file)},
        "model_sha256": sha(args.model), "motion_sha256": sha(args.motion), "contract_sha256": sha(args.contract),
        "verified_physical_objective_source_sha256": matched_sources,
        "archived_runner_sha256": next(value for name, value in report["source_sha256"].items() if name.endswith("/scripts/run_nmpc_tracking.py")),
        "probe_runner_sha256": sha(ROOT / "scripts/run_nmpc_tracking.py"),
        "probe_source_sha256": sha(__file__), "native_backend": native_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "audited", "selected_policy_frames": selected.tolist(), **aggregate}, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--settings", type=Path, default=ROOT / "config/nmpc_height_smooth.json")
    p.add_argument("--task-profile", type=Path, default=ROOT / "config/car_drift_height_smooth_task.json")
    p.add_argument("--native-library", type=Path, required=True, help="Existing hash-verified build; never auto-build")
    p.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    analyze(parser().parse_args())
