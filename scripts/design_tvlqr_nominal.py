"""Explicit offline TVLQR design around an audited MODEL-ONLY NMPC episode.

The quadratic surrogate tracks the model-generated nominal, not the original
15-term task reward. The original reference is reserved for plant evaluation.
No PPO/SAC actions, inverse-kinematic playbacks or root corrections are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
               "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")
EPSILON, STEPS, NX, NU = 1e-5, 349, 28, 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite real {shape}")
    return value.astype(float, copy=True)


def load_nominal_donor(donor_dir, model, motion, contract):
    """Validate original bytes, clocks and applied actions without physics."""
    directory = Path(donor_dir)
    report_path, trace_path, solver_path = [directory/name for name in ("report.json", "rollout.npz", "solver_log.json")]
    report = json.loads(report_path.read_text())
    required = {"kind": "full_body_height_smooth_cem_nmpc", "engine": "MuJoCo", "status": "completed",
                "reference_id": "drift_drl_traj4", "completed_policy_steps": STEPS,
                "evaluated_policy_steps": STEPS, "plant_mu": .7, "prediction_mu_scenarios": [.7],
                "learned_policy_used": False, "root_state_prescribed_after_reset": False,
                "external_root_forces_used": False, "truncated_smoke": False}
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Donor must be a completed primary nominal model-only NMPC episode")
    if report.get("model_generated_nominal_plan") != {"enabled": False}:
        raise ValueError("Donor cannot use an additional model-plan initialization artifact")
    hashes = {name: sha(path) for name, path in (("report.json", report_path), ("rollout.npz", trace_path), ("solver_log.json", solver_path))}
    if (hashes["rollout.npz"] != report.get("rollout_sha256") or hashes["solver_log.json"] != report.get("solver_log_sha256")
            or sha(model) != report.get("model_sha256") or sha(motion) != report.get("motion_sha256")):
        raise ValueError("Donor model/motion/trace/solver byte binding differs")
    contract_hashes = [value for path, value in report["source_sha256"].items()
                       if Path(path).name in ("policy_contract.json", "nominal_actuator_contract.json")]
    if contract_hashes != [sha(contract)]:
        raise ValueError("Shared actuator contract differs from donor")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    qpos = _array(trace["qpos"], (1397, 15), "qpos")
    qvel = _array(trace["qvel"], (1397, 14), "qvel")
    actions = _array(trace["actions"], (STEPS, NU), "actions")
    clock = _array(trace["time_s"], (1397,), "time_s")
    if (tuple(trace["action_joint_names"].tolist()) != JOINT_NAMES or float(trace["physics_dt_s"]) != .005
            or float(trace["policy_dt_s"]) != .02 or np.max(np.abs(actions)) > 1.
            or not np.array_equal(actions, actions.astype(np.float32).astype(float))
            or not np.allclose(clock, np.arange(1397)*.005, atol=1e-10, rtol=0)
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5, rtol=0)):
        raise ValueError("Donor physical clock/quaternion/joint/applied-action contract differs")
    if (not np.array_equal(trace["held_action"][1:], np.repeat(actions, 4, axis=0))
            or not np.array_equal(trace["reference_frame_for_action"][1:], np.repeat(np.arange(STEPS), 4))):
        raise ValueError("Donor does not hold the actual action for four substeps")
    solver = json.loads(solver_path.read_text())
    if len(solver) != STEPS or any(row["frame"] != k or row["action"] != actions[k].tolist() for k, row in enumerate(solver)):
        raise ValueError("Donor decisions do not match actual applied actions")
    nominal = np.concatenate((qpos[::4], qvel[::4]), axis=1)
    previous = np.vstack((np.zeros(NU), actions[:-1]))
    evaluations = sum(row["solver"]["scenario_evaluations"] for row in solver)
    prediction_steps = sum(row["solver"]["scenario_evaluations"] * row["solver"]["horizon_steps"] * 4 for row in solver)
    identity = {"archive_sha256": hashes, "learned_policy_used": False, "initialization": "Existing pure-model full-body NMPC simulated states and actual float32 controls, not PPO demonstrations.",
                "generation_budget": {"scenario_evaluations": evaluations, "predicted_physics_substeps": prediction_steps,
                                      "plant_physics_substeps": 1396, "wall_s": report["wall_s"]},
                "effective_cem_config": report["effective_cem_config"], "source_sha256": report["source_sha256"]}
    return nominal, actions, previous, identity


def physical_state_cost(vids, joint_names):
    vids = np.asarray(vids)
    if (vids.shape != (8,) or vids.dtype.kind not in "iu" or set(vids.tolist()) != set(range(6, 14))
            or tuple(joint_names) != JOINT_NAMES):
        raise ValueError("Named joint tangent layout must match the audited full-body model")
    diagonal = np.r_[[400., 400., 0., 500., 500., 120.], np.zeros(8),
                     [20., 20., 0., 5., 5., 5.], np.zeros(8)]
    for name, vid in zip(joint_names, vids):
        wheel = name.startswith("wheel_")
        diagonal[vid] = 0. if wheel else 5.
        diagonal[14 + vid] = .005 if wheel else .05
    return np.diag(diagonal)


def augment_dynamics(A, B, defect):
    A = _array(A, (NX, NX), "A")
    B = _array(B, (NX, NU), "B")
    defect = _array(defect, (NX,), "defect")
    augmented_A, augmented_B, augmented_d = np.zeros((36, 36)), np.zeros((36, 8)), np.zeros(36)
    augmented_A[:28, :28], augmented_B[:28], augmented_d[:28] = A, B, defect
    augmented_B[28:] = np.eye(8)
    return augmented_A, augmented_B, augmented_d


def quadratic_stage_cost(Qphysical, nominal_action, nominal_previous, smooth_state, smooth_action, smooth_defect):
    """Cost .5 zQz + .5 vRv + zNv + q'z + r'v, constants omitted."""
    Qphysical = _array(Qphysical, (NX, NX), "Qphysical")
    delta = _array(nominal_action, (8,), "nominal_action") - _array(nominal_previous, (8,), "nominal_previous")
    gx = np.r_[_array(smooth_state, (28,), "smooth_state"), np.zeros(8)]
    gu = _array(smooth_action, (8,), "smooth_action")
    dz = _array(np.asarray(smooth_defect), (), "smooth_defect").item()
    Q, R, N, q, r = np.zeros((36, 36)), np.diag([10.]*6 + [.5]*2), np.zeros((36, 8)), np.zeros(36), np.zeros(8)
    Q[:28, :28] = Qphysical
    Q[28:, 28:] += .2*np.eye(8)
    R += .2*np.eye(8)
    N[28:] = -.2*np.eye(8)
    q[28:], r[:] = -.2*delta, .2*delta
    # .1 * (Delta_vz / (.02*5))^2 = 10 * Delta_vz^2.
    Q += 20*np.outer(gx, gx)
    R += 20*np.outer(gu, gu)
    N += 20*np.outer(gx, gu)
    q += 20*dz*gx
    r += 20*dz*gu
    return Q, R, N, q, r


def design_objective_contract():
    return {"kind": "quadratic_model_generated_nominal_surrogate_not_original_15_term_reward",
            "nominal_source": "Pure-model NMPC simulated trajectory and actual normalized actions; original reference used only for evaluation.",
            "quadratic_convention": ".5*z.T@Q*z + .5*v.T@R*v + z.T@N*v + q.T@z + r.T@v; omitted state-independent constants.",
            "state_error": "MuJoCo position tangent28 (14 position/rotation/joints +14 velocities), then eight actual previous-action deviations.",
            "position_weights": {"xy": 400., "z": 0., "roll_pitch": 500., "yaw": 120., "leg": 5., "wheel_angle": 0.},
            "velocity_weights": {"xy": 20., "vz": 0., "angular": 5., "leg": .05, "wheel": .005},
            "action_deviation_weights": {"leg": 10., "wheel": .5},
            "actual_action_rate_cost": ".1*sum((u_actual-u_previous_actual)**2), includes nominal action increments via linear terms.",
            "vertical_smoothness_cost": ".1*((COM_vz_next-COM_vz_current)/(.02*5))**2, locally linearized; no absolute height/vz target.",
            "terminal": "5*physical_state_Q, zero action-memory terminal cost.",
            "absolute_height_tracking": False, "absolute_vz_tracking": False,
            "sideslip_zero_target": False, "flight_failure_gate": False, "original_reward_matched": False}


def design(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Preserving existing TVLQR design: " + str(output))
    nominal, actions, previous, donor = load_nominal_donor(args.donor_dir, args.model, args.motion, args.contract)
    # Physics imports/construction occur only in this explicit design action.
    from run_nmpc_tracking import build_provider
    from tvlqr_core import generalized_affine_riccati
    import mujoco
    settings = json.loads(Path(args.settings).read_text())
    contract = json.loads(Path(args.contract).read_text())
    for relative in ("scripts/classical_tracking.py", "upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py"):
        saved = [digest for path, digest in donor["source_sha256"].items() if path.endswith("/" + relative)]
        if saved != [sha(ROOT/relative)]:
            raise ValueError("Physical predictor source differs from donor: " + relative)
    provider = build_provider(args.model, args.motion, contract, settings, (.7,))
    if (provider.model.nq, provider.model.nv, provider.model.nu) != (15, 14, 8) or provider.n != STEPS:
        raise ValueError("Unexpected nominal full-body model size/reference length")
    Qphysical = physical_state_cost(provider.vids, provider.reference.action_names)
    source_paths = [Path(__file__), ROOT/"scripts/tvlqr_core.py", ROOT/"scripts/classical_tracking.py",
                    ROOT/"scripts/run_nmpc_tracking.py", ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py",
                    ROOT/"config/classical_tracking.json", Path(args.settings)]
    report = {"kind": "model_nominal_initialized_height_smooth_tvlqr_design", "status": "running",
              "model_sha256": sha(args.model), "motion_sha256": sha(args.motion), "actuator_contract_sha256": sha(args.contract),
              "nominal_prediction_mu": .7, "design_objective_contract": design_objective_contract(), "donor_identity": donor,
              "source_sha256": {str(path.resolve()): sha(path) for path in source_paths},
              "finite_difference_epsilon": EPSILON, "stages": STEPS, "physical_tangent_dimension": 28,
              "augmented_dimension": 36, "physics_dt_s": .005, "control_dt_s": .02,
              "expected_transition_calls": STEPS*73, "expected_predicted_physics_substeps": STEPS*73*4,
              "learned_policy_initialization": False, "automatic_regularization": False,
              "certified_constraints": False, "robustness_certificate": False, "real_time_verified": False,
              "warnings": ["Cold qacc/time reset per prediction interval; full solver integration state was not logged.",
                           "Central finite differences cross possible contact and action-saturation boundaries; local gains are not a global validity or stability certificate.",
                           "Affine nominal transition defects are retained, not removed by prescribing root state.",
                           "Online action clipping enforces only normalized command bounds, not certified state/contact constraints."]}
    output.mkdir(parents=True)
    report_path = output/"report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    start = time.perf_counter()
    completed = attempted = 0

    def transition(x, u):
        nonlocal attempted, completed
        attempted += 1
        result = provider.transition(x, u)
        completed += 1
        return result

    def vz(x):
        provider.data.qpos[:], provider.data.qvel[:] = x[:15], x[15:]
        mujoco.mj_fwdPosition(provider.model, provider.data)
        jacobian, _ = provider.measure.root_jacobians()
        result = float(jacobian[2] @ x[15:])
        if not np.isfinite(result):
            raise FloatingPointError("Nonfinite original-base COM vertical velocity")
        return result

    try:
        A, B, defects = np.empty((STEPS, 28, 28)), np.empty((STEPS, 28, 8)), np.empty((STEPS, 28))
        smooth_x, smooth_u, smooth_d = np.empty((STEPS, 28)), np.empty((STEPS, 8)), np.empty(STEPS)
        augmented_A, augmented_B, augmented_d, costs = [], [], [], []
        for k in range(STEPS):
            x, u, target = nominal[k], actions[k], nominal[k+1]
            baseline = transition(x, u)
            defects[k] = provider.error(target, baseline)
            smooth_d[k] = vz(baseline) - vz(x)
            for j in range(28):
                plus_x, minus_x = provider.perturb(x, j, EPSILON), provider.perturb(x, j, -EPSILON)
                plus, minus = transition(plus_x, u), transition(minus_x, u)
                A[k, :, j] = (provider.error(target, plus)-provider.error(target, minus))/(2*EPSILON)
                smooth_x[k, j] = (vz(plus)-vz(minus)-vz(plus_x)+vz(minus_x))/(2*EPSILON)
            for j in range(8):
                delta = np.zeros(8); delta[j] = EPSILON
                plus, minus = transition(x, u+delta), transition(x, u-delta)
                B[k, :, j] = (provider.error(target, plus)-provider.error(target, minus))/(2*EPSILON)
                smooth_u[k, j] = (vz(plus)-vz(minus))/(2*EPSILON)
            aa, bb, dd = augment_dynamics(A[k], B[k], defects[k])
            augmented_A.append(aa); augmented_B.append(bb); augmented_d.append(dd)
            costs.append(quadratic_stage_cost(Qphysical, u, previous[k], smooth_x[k], smooth_u[k], smooth_d[k]))
            if k % 25 == 0:
                print(f"tvlqr linearize={k}/{STEPS} completed_transitions={completed}", flush=True)
        Q, R, N, q, r = [np.stack(values) for values in zip(*costs)]
        terminal = np.zeros((36, 36)); terminal[:28, :28] = 5*Qphysical
        K, feed, P, p = generalized_affine_riccati(np.stack(augmented_A), np.stack(augmented_B), np.stack(augmented_d),
                                                Q, R, N, q, r, terminal, np.zeros(36))
        arrays = dict(nominal_states=nominal, nominal_actions=actions, nominal_previous_actions=previous, K=K, feed=feed,
                      physical_A=A, physical_B=B, physical_defect=defects, Q=Q, R=R, N=N, q=q, r=r, P=P, p=p,
                      smooth_state_gradient=smooth_x, smooth_action_gradient=smooth_u, smooth_defect=smooth_d)
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite TVLQR design arrays")
        design_path = output/"design.npz"
        np.savez_compressed(design_path, **arrays)
        report.update(status="completed", design_sha256=sha(design_path),
                      physical_defect_component_max_abs=np.max(np.abs(defects), axis=0).tolist(),
                      nominal_command_margin_min=float(np.min(1-np.abs(actions))),
                      nominal_commands_at_bound=int(np.sum(np.abs(actions) == 1)),
                      max_abs_gain=float(np.max(np.abs(K))), max_abs_affine_feed=float(np.max(np.abs(feed))),
                      max_abs_unclipped_nominal_plus_feed=float(np.max(np.abs(actions+feed))))
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(attempted_transition_calls=attempted, completed_transition_calls=completed,
                      predicted_physics_substeps=completed*4, wall_s=time.perf_counter()-start,
                      substep_count_scope="Exactly four per completed transition; partial failed-transition steps, if any, are not inferred.")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "motion", "contract", "donor-dir", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--settings", type=Path, default=ROOT/"config/nmpc_height_smooth_residual.json")
    return result


if __name__ == "__main__":
    design(parser().parse_args())
