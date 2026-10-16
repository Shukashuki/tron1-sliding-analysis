"""MuJoCo-only TVLQR pilot initialized by a hash-bound model-generated nominal.

Feedback is evaluated from actual state, never an applied-action playback.
The design's quadratic surrogate differs from the common 15-term evaluation
objective; both are reported explicitly. No pose correction or height gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controller_references import DEFAULT_CATALOG, ROOT, get_reference
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState
from run_nmpc_tracking import build_provider, MujocoTrackingPredictor, _metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design(folder, *, model, motion, contract):
    folder = Path(folder)
    report = json.loads((folder/"report.json").read_text())
    if (report.get("kind") != "model_nominal_initialized_height_smooth_tvlqr_design"
            or report.get("status") != "completed" or report.get("nominal_prediction_mu") != .7
            or report.get("design_sha256") != sha(folder/"design.npz")
            or report.get("model_sha256") != sha(model) or report.get("motion_sha256") != sha(motion)
            or report.get("actuator_contract_sha256") != sha(contract)):
        raise ValueError("TVLQR design/model/reference/actuator identity mismatch")
    physical_sources = report.get("source_sha256",{})
    for relative in ("scripts/classical_tracking.py","scripts/run_nmpc_tracking.py",
                     "upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py"):
        expected_path=(ROOT/relative).resolve()
        recorded=[digest for path,digest in physical_sources.items() if Path(path).resolve()==expected_path]
        if recorded != [sha(expected_path)]:
            raise ValueError("TVLQR design physical-predictor source identity mismatch: "+relative)
    with np.load(folder/"design.npz", allow_pickle=False) as archive:
        design = {key: archive[key].copy() for key in archive.files}
    for key, shape in {"nominal_states": (350,29), "nominal_actions": (349,8),
                       "nominal_previous_actions": (349,8), "K": (349,8,36), "feed": (349,8)}.items():
        if key not in design or design[key].shape != shape or not np.isfinite(design[key]).all():
            raise ValueError("Malformed TVLQR design: "+key)
    controls = design["nominal_actions"]
    if np.max(np.abs(controls)) > 1 or not np.array_equal(controls, controls.astype(np.float32).astype(float)):
        raise ValueError("Nominal actions must be the bounded actual float32 commands")
    if not np.array_equal(design["nominal_previous_actions"], np.vstack((np.zeros((1,8)), controls[:-1]))):
        raise ValueError("Nominal action-rate memory has incorrect phase")
    return report, design


def feedback_action(design, frame, physical_error, previous_action):
    error = np.asarray(physical_error, dtype=float)
    previous = np.asarray(previous_action, dtype=float)
    if (type(frame) is not int or not 0 <= frame < 349 or error.shape != (28,)
            or previous.shape != (8,) or not np.isfinite(error).all() or not np.isfinite(previous).all()
            or np.max(np.abs(previous)) > 1
            or not np.array_equal(previous, previous.astype(np.float32).astype(float))):
        raise ValueError("TVLQR requires finite tangent error and actual applied action memory")
    augmented = np.r_[error, previous-design["nominal_previous_actions"][frame]]
    raw = design["nominal_actions"][frame]+design["feed"][frame]+design["K"][frame]@augmented
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite TVLQR feedback command")
    applied = np.clip(raw, -1., 1.).astype(np.float32).astype(float)
    return applied, {"raw_action": raw.tolist(),
        "clipped_action_components": int(np.count_nonzero(np.abs(raw) > 1.)),
        "physical_tangent_error_norm_mixed_units": float(np.linalg.norm(error)),
        "feedback_delta_l2": float(np.linalg.norm(design["K"][frame]@augmented)),
        "online_optimization": False, "action_replay": False}


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Preserve existing TVLQR output: "+str(output))
    if not np.isfinite(args.plant_mu) or args.plant_mu < 0:
        raise ValueError("Plant coefficient must be finite and nonnegative")
    spec = get_reference("drift_drl_traj4")
    if sha(args.motion) != spec["motion_npz_sha256"]:
        raise ValueError("TVLQR pilot requires the locked primary reference")
    design_report, design = load_design(args.design_dir, model=args.model, motion=args.motion, contract=args.contract)
    settings = json.loads(Path(args.settings).read_text())
    contract = json.loads(Path(args.contract).read_text())
    task = json.loads(Path(args.task_profile).read_text())
    provider = build_provider(args.model,args.motion,contract,settings,(.7,))
    prediction = MujocoTrackingPredictor(provider,args.model,task,(.7,))
    model,_ = provider.api.load_model(args.model,no_visual_mesh=True)
    provider.api.configure_model(model,provider.contract)
    model.geom_friction[:,0] = args.plant_mu
    data = mujoco.MjData(model)
    physical = provider.api.MujocoState(model,data,provider.reference)
    physical.initialize()
    np.testing.assert_allclose(np.r_[data.qpos,data.qvel],design["nominal_states"][0],atol=1e-10,rtol=0)
    extractor = MujocoObjectiveState(model)
    history = np.zeros(prediction.layout.history_shape)
    steps = 349 if args.max_policy_steps is None else args.max_policy_steps
    if type(steps) is not int or not 1 <= steps <= 349:
        raise ValueError("Use between one and 349 control decisions")
    source = [Path(__file__), ROOT/"scripts/tvlqr_core.py", ROOT/"scripts/design_tvlqr_nominal.py",
        ROOT/"scripts/run_nmpc_tracking.py",ROOT/"scripts/classical_tracking.py",
        ROOT/"scripts/nmpc_tracking_objective.py",ROOT/"scripts/height_smoothness.py",DEFAULT_CATALOG,
        Path(args.settings),Path(args.task_profile),Path(args.contract)]
    for relative in ("scripts/eval_tracking_mujoco.py","scripts/eval_tracking.py",
                     "training/tron1_domain_randomization.py","training/tron1_contact_slip.py",
                     "training/tron1_tracking.py","training/tron1_drift.py",
                     "training/tron1_runtime_device.py","training/tron1_physics_clock.py"):
        source.append(ROOT/"upstream/tron1-learning-from-human"/relative)
    report = {"kind":"model_nominal_initialized_height_smooth_tvlqr", "status":"running", "engine":"MuJoCo",
        "mujoco_version":mujoco.__version__, "reference_id":spec["reference_id"],"reference_spec":spec,
        "role":spec["role"],"stratum":spec["stratum"],"reference_catalog_sha256":sha(DEFAULT_CATALOG),
        "model_sha256":sha(args.model),"motion_sha256":sha(args.motion),"actuator_contract_sha256":sha(args.contract),
        "design_sha256":sha(Path(args.design_dir)/"design.npz"),"design_report_sha256":sha(Path(args.design_dir)/"report.json"),
        "design_objective_contract":design_report["design_objective_contract"],
        "donor_identity":design_report["donor_identity"],"source_sha256":{str(p.resolve()):sha(p) for p in source},
        "offline_design_budget":{key:design_report[key] for key in (
            "attempted_transition_calls","completed_transition_calls","predicted_physics_substeps","wall_s")},
        "objective_contract":prediction.objective.contract(),"design_cost_equals_15_term_objective":False,
        "plant_mu":args.plant_mu,"nominal_prediction_mu":.7,"prediction_receives_plant_mu":False,
        "plant_material_readback":{"compiled_sliding_friction_range":[float(model.geom_friction[:,0].min()),float(model.geom_friction[:,0].max())]},
        "physics_dt_s":.005,"control_dt_s":.02,"total_reference_policy_steps":349,"evaluated_policy_steps":steps,
        "truncated_smoke":steps<349,"pose_height_failure_gate":False,"height_or_flight_failure_gate":False,
        "heading_offset_is_mismatch":False,"learned_policy_used":False,"action_replay":False,
        "root_state_prescribed_after_reset":False,"external_root_forces_used":False,
        "real_time_verified":False,"robustness_certificate":False,"sim2sim_comparison":False,
        "state_access":"28-D full physical tangent state plus 8 previous applied actions; time-indexed offline gains for the whole nominal clip.",
        "timing_scope":"Online tangent-error/feedback call only; offline NMPC donor and TVLQR design costs are separately charged, no delay injection.",
        "contact_slip_definition":"Actual force-weighted wheel-ground contact-point lateral-speed RMS; contact-free samples unavailable, not rolling."}
    output.mkdir(parents=True)
    def save_report():
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save_report()
    records=[]; decisions=[]; held_action=np.zeros(8); held_frame=0
    def record(current):
        state=extractor.extract(data,contact_force_history=history,body_net_forces_w=current)
        contact=physical.ground_contact_slip(force_threshold_n=5.)
        count,force=physical.nonwheel_ground_contacts()
        records.append({"time_s":float(data.time),"root_pos":state["body_pos_w"][0].copy(),
            "root_quaternion_wxyz":state["body_quat_w"][0].copy(),"base_com_velocity_w":state["body_lin_vel_w"][0].copy(),
            "base_angular_velocity_w":state["body_ang_vel_w"][0].copy(),"wheel_pos":state["body_pos_w"][-2:].copy(),
            "joint_pos":state["joint_pos"].copy(),"joint_vel":data.qvel[physical.vids].copy(),
            "applied_joint_torque_nm":data.ctrl[physical.aids].copy(),"held_action":held_action.copy(),
            "reference_frame_for_action":held_frame,"action_was_applied":bool(records),**contact,
            "nonwheel_ground_contact_count":count,"nonwheel_ground_contact_force_norm_n":force,
            "body_net_forces_w":state["body_net_forces_w"].copy(),"qpos":data.qpos.copy(),"qvel":data.qvel.copy()})
        return state
    previous=np.zeros(8); pre=record(np.zeros((9,3))); previous_vz=float(pre["body_lin_vel_w"][0,2])
    started=time.perf_counter()
    try:
        for k in range(steps):
            held=prediction.objective.hold_reference(prediction.reference_frame(k),pre)
            before=np.r_[data.qpos,data.qvel]
            now=time.perf_counter()
            error=provider.error(design["nominal_states"][k],before)
            action,info=feedback_action(design,k,error,previous)
            elapsed=time.perf_counter()-now
            held_action=action.copy();held_frame=k
            for _ in range(4):
                data.ctrl[physical.aids]=provider.torque(action,data.qpos,data.qvel)
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                    raise RuntimeError("Unexpected external plant force")
                history,current=extractor.update_contact_history(data,history)
                pre=record(current)
                if (not np.isclose(data.time,(len(records)-1)*.005,atol=1e-8,rtol=0)
                        or not np.isfinite(np.r_[data.qpos,data.qvel]).all()):
                    raise FloatingPointError("Nonfinite plant state or hidden clock reset")
            objective=prediction.objective.evaluate(pre,held,action,previous,previous_vz,reset_sample=k==0)
            decisions.append({"frame":k,"time_s":float(data.time),"reference_frame_for_reward":k,
                "action":action.tolist(),"compute_s":elapsed,"feedback":info,"objective":objective})
            previous=action.copy();previous_vz=objective["base_com_vz"]
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error),completed_policy_steps=len(decisions))
        save_report();raise
    trace={key:np.asarray([row[key] for row in records]) for key in records[0]}
    trace.update(actions=np.asarray([row["action"] for row in decisions]),compute_s=np.asarray([row["compute_s"] for row in decisions]),
        stage_cost=np.asarray([row["objective"]["stage_cost"] for row in decisions]),physics_dt_s=np.array(.005),policy_dt_s=np.array(.02),
        qpos_joint_names=np.array([model.joint(i).name for i in range(1,model.njnt)]),body_names=np.array(BODY_NAMES),action_joint_names=np.array(JOINT_NAMES),
        reference_id=np.array(spec["reference_id"]),reference_time_s=np.arange(349)*.02,
        reference_root_xy_m=provider.reference.archive["body_pos_w"][:,0,:2])
    report.update(_metrics(trace,provider.reference,spec))
    np.savez_compressed(output/"rollout.npz",**trace)
    (output/"feedback_log.json").write_text(json.dumps(decisions,indent=2,allow_nan=False)+"\n")
    report.update(status="completed" if steps==349 else "smoke_completed",completed_policy_steps=len(decisions),duration_s=float(data.time),
        wall_s=time.perf_counter()-started,physics_samples=len(records),rollout_sha256=sha(output/"rollout.npz"),
        feedback_log_sha256=sha(output/"feedback_log.json"),total_stage_cost=float(trace["stage_cost"].sum()),
        compute_ms_median_p95_max=(1000*np.quantile(trace["compute_s"],[.5,.95,1.])).tolist(),
        clipped_action_component_fraction=sum(row["feedback"]["clipped_action_components"] for row in decisions)/(steps*8))
    save_report()
    print(json.dumps({key:report[key] for key in ("status","duration_s","core_xy_rmse_m","clipped_action_component_fraction")}))
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("model","motion","contract","design-dir","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--plant-mu",type=float,required=True)
    p.add_argument("--settings",type=Path,default=ROOT/"config/nmpc_height_smooth_residual.json")
    p.add_argument("--task-profile",type=Path,default=ROOT/"config/car_drift_height_smooth_task.json")
    p.add_argument("--max-policy-steps",type=int,default=None)
    return p


if __name__=="__main__":
    run(parser().parse_args())
