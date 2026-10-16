"""Independent MuJoCo MPPI pilot using the archived full-body plant interface.

The existing CEM/adaptive/TVLQR runners remain untouched. This runner imports
their common reference, actuator, predictor and reward services and copies the
same physical recorder/held-action loop; parity is checked without simulation.
MPPI's Gaussian latent-action prior is additional to the unchanged 15-term
physical rollout score, so its optimization is not a pure task-cost CEM clone.
"""
from __future__ import annotations

from contextlib import ExitStack
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT
from mppi_shooting import MPPIConfig, MPPIController
from nmpc_batch_objective import NativeBatchTrackingPredictor
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, MujocoObjectiveState
from run_nmpc_tracking import (MujocoTrackingPredictor, _metrics, build_provider,
                               selected_reference, sha)


class NativeMPPIPredictor(NativeBatchTrackingPredictor):
    """Audit actual MPPI attachment without asserting historical CEM wiring."""

    def record_mppi_wiring(self, controller):
        if not isinstance(controller, MPPIController) or controller.predictor is not self:
            raise ValueError("MPPI wiring audit requires the actual controller using this adapter")
        if not callable(getattr(controller.predictor, "rollout_batch", None)):
            raise ValueError("Actual MPPI predictor lacks the batch interface")
        if self.metadata.get("production_cem_wired") is not False:
            raise ValueError("Independent MPPI adapter must not claim CEM wiring")
        self.metadata["production_mppi_wired"] = True
        self.metadata["objective"]["production_mppi_wired"] = True
        self.metadata["objective"]["production_cem_wired"] = False
        self.metadata["runtime_mppi_population_path"] = "callable predictor.rollout_batch"
        self.metadata["worker_build_manifest_flag_semantics"] = (
            "Nested worker production_cem_wired is historical build-time metadata, not MPPI runtime attachment.")


def effective_settings(args):
    """Load every pilot knob explicitly; never derive prediction mu from plant."""
    settings = json.loads(Path(args.settings).read_text())
    required = {"schema_version", "task", "policy_dt_s", "physics_dt_s",
                "prediction_mu_scenarios", "mppi"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Explicit MPPI settings must declare exactly the documented schema fields")
    if (type(settings["schema_version"]) is not int or settings["schema_version"] != 1
            or settings["task"] != "car_drift_height_smooth"
            or settings["policy_dt_s"] != .02 or settings["physics_dt_s"] != .005):
        raise ValueError("Expected schema 1 car_drift_height_smooth MPPI settings with .02/.005 clocks")
    config = settings["mppi"]
    if not isinstance(config, dict) or set(config) != set(MPPIConfig.__dataclass_fields__):
        raise ValueError("Explicit MPPI settings must declare every MPPIConfig field")
    config = dict(config)
    if isinstance(config.get("noise_std"), list):
        config["noise_std"] = tuple(config["noise_std"])
    mppi = MPPIConfig(**config)
    values = settings["prediction_mu_scenarios"]
    if (not isinstance(values, list) or len(values) != 1
            or type(values[0]) not in (int, float) or not np.isfinite(values[0]) or values[0] < 0):
        raise ValueError("MPPI pilot requires one explicit finite nonnegative prediction friction")
    return settings, mppi, tuple(float(mu) for mu in values)


def source_files(args, backend):
    files = [Path(__file__), ROOT/"scripts/mppi_shooting.py", ROOT/"scripts/run_nmpc_tracking.py",
             ROOT/"scripts/controller_references.py", DEFAULT_CATALOG,
             ROOT/"scripts/nmpc_shooting.py", ROOT/"scripts/nmpc_tracking_objective.py",
             ROOT/"scripts/height_smoothness.py", ROOT/"scripts/classical_tracking.py",
             ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py",
             ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking.py",
             ROOT/"upstream/tron1-learning-from-human/training/tron1_domain_randomization.py",
             ROOT/"upstream/tron1-learning-from-human/training/tron1_contact_slip.py",
             ROOT/"upstream/tron1-learning-from-human/training/tron1_tracking.py",
             ROOT/"upstream/tron1-learning-from-human/training/tron1_drift.py",
             ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
             ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
             ROOT/"config/classical_tracking.json", Path(args.settings), Path(args.task_profile), Path(args.contract)]
    if backend == "native":
        files.extend((ROOT/"scripts/nmpc_native_batch.py", ROOT/"scripts/nmpc_batch_objective.py",
                      ROOT/"native/nmpc_batch.cpp"))
    return files


def run(args):
    with ExitStack() as resources:
        return _run(args, resources)


def _run(args, resources):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Preserving existing output directory: {output}")
    if (type(args.plant_mu) not in (int, float) or not np.isfinite(args.plant_mu)
            or args.plant_mu < 0):
        raise ValueError("Plant mu must be finite and nonnegative")
    if args.max_policy_steps is not None and (type(args.max_policy_steps) is not int or args.max_policy_steps < 1):
        raise ValueError("max-policy-steps must be a positive integer")
    backend = args.prediction_backend
    library = args.native_library
    if backend not in ("python", "native") or (backend == "native") != (library is not None):
        raise ValueError("Native prediction is explicit opt-in and requires exactly one prebuilt --native-library")
    reference_spec = selected_reference(args)
    settings, mppi, scenarios = effective_settings(args)
    task = json.loads(Path(args.task_profile).read_text())
    contract = json.loads(Path(args.contract).read_text())
    provider = build_provider(args.model,args.motion,contract,settings,scenarios)
    if provider.reference.frames != reference_spec["frames"] or not np.isclose(provider.reference.fps,reference_spec["fps"]):
        raise ValueError("Selected reference frame count/fps differs from locked catalog")
    predictor = MujocoTrackingPredictor(provider,args.model,task,scenarios)
    controller_predictor = predictor
    native_metadata = None
    if backend == "native":
        controller_predictor = resources.enter_context(NativeMPPIPredictor(predictor,library))
        native_metadata = controller_predictor.metadata
    controller = MPPIController(controller_predictor,mppi,action_low=-np.ones(8),action_high=np.ones(8),mu_scenarios=scenarios)
    if backend == "native":
        controller_predictor.record_mppi_wiring(controller)
    model,_=provider.api.load_model(args.model,no_visual_mesh=True)
    provider.api.configure_model(model,provider.contract)
    model.geom_friction[:,0]=args.plant_mu
    if not np.allclose(model.geom_friction[:,0],args.plant_mu,atol=1e-12,rtol=0):
        raise AssertionError("Compiled validation plant friction does not match the fixed coefficient")
    data=mujoco.MjData(model)
    physical=provider.api.MujocoState(model,data,provider.reference)
    physical.initialize()
    np.testing.assert_allclose(np.r_[data.qpos,data.qvel],provider.targets[0],atol=1e-10)
    extractor=MujocoObjectiveState(model)
    history=np.zeros(predictor.layout.history_shape)
    steps=provider.reference.frames if args.max_policy_steps is None else min(args.max_policy_steps,provider.reference.frames)
    truncated=steps<provider.reference.frames
    report = {"kind":"full_body_height_smooth_mppi", "engine":"MuJoCo", "mujoco_version":mujoco.__version__,
        "status":"running", "controller_label":"Latent Gaussian-prior sampling MPPI pilot",
        "model_sha256":sha(args.model), "motion_sha256":sha(args.motion),
        "reference_id":reference_spec["reference_id"], "role":reference_spec["role"],
        "stratum":reference_spec["stratum"], "reference_spec":reference_spec,
        "reference_catalog_sha256":sha(DEFAULT_CATALOG), "pooled_success_statistics":False,
        "source_sha256":{str(p.resolve()):sha(p) for p in source_files(args,backend)},
        "settings":settings, "effective_mppi_config":asdict(mppi), "plant_mu":args.plant_mu,
        "prediction_mu_scenarios":list(scenarios), "prediction_receives_plant_mu":False,
        "prediction_backend":backend, "native_prediction_backend":native_metadata,
        "action_parameterization":controller.parameterization_contract(), "algorithm_contract":controller.algorithm_contract(),
        "model_generated_nominal_plan":{"enabled":False},
        "plant_material_readback":{"compiled_sliding_friction_range":[float(model.geom_friction[:,0].min()),float(model.geom_friction[:,0].max())],
            "note":"Equal numeric contact coefficient, not a calibrated tire model."},
        "packed_state_contract":predictor.layout.contract(), "objective_contract":predictor.objective.contract(),
        "common_15_term_physical_rollout_score_unchanged":True,
        "optimization_equals_pure_task_cost_cem":False,
        "optimization_difference":"Latent Gaussian action prior and importance correction supplement the common physical rollout score; see algorithm_contract.",
        "physics_dt_s":model.opt.timestep, "control_dt_s":provider.dt, "contact_threshold_n":5.,
        "wheel_lateral_contact_slip_evidence_kind":"actual_ground_contact_points",
        "contact_slip_definition":"Per-wheel force-weighted contact-point lateral-speed RMS; >threshold normal force, ground-filtered. Zero with no active contact is unavailable, not rolling.",
        "total_reference_policy_steps":provider.reference.frames, "evaluated_policy_steps":steps,
        "truncated_smoke":truncated, "pose_height_failure_gate":False, "height_or_flight_failure_gate":False,
        "heading_offset_is_mismatch":False, "learned_policy_used":False, "tvlqr_used":False,
        "old_qr_used":False, "adaptive_model_selection_used":False, "action_replay":False,
        "root_state_prescribed_after_reset":False, "external_root_forces_used":False,
        "initial_plan":"Bounded bias-force joint feedforward from kinematic reference; not a feasible trajectory certificate.",
        "state_access":"Full simulator qpos/qvel, objective memory and future reference preview; unlike the actor-only observation interface.",
        "plant_recorder_scope":"Independent MPPI runner; same imported actuator/predictor/objective services and AST-parity-checked physical recorder loop, not a modified archived CEM runner.",
        "robustness_certificate":False, "real_time_verified":False, "isaac_result":False,
        "sim2sim_comparison":False, "surrogate_flags":[]}
    output.mkdir(parents=True)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    records=[];decisions=[];surrogates=set();started=time.perf_counter()
    held_action=np.zeros(8);held_frame=0

    def record(current=None):
        s=extractor.extract(data,contact_force_history=history,body_net_forces_w=current)
        contact=physical.ground_contact_slip(force_threshold_n=5.)
        nonwheel_count,nonwheel_force=physical.nonwheel_ground_contacts()
        records.append({"time_s":float(data.time),"root_pos":s["body_pos_w"][0].copy(),
                        "root_quaternion_wxyz":s["body_quat_w"][0].copy(),
                        "base_com_velocity_w":s["body_lin_vel_w"][0].copy(),
                        "base_angular_velocity_w":s["body_ang_vel_w"][0].copy(),
                        "wheel_pos":s["body_pos_w"][-2:].copy(),"joint_pos":s["joint_pos"].copy(),
                        "joint_vel":data.qvel[physical.vids].copy(),
                        "applied_joint_torque_nm":data.ctrl[physical.aids].copy(),
                        "held_action":held_action.copy(),"reference_frame_for_action":held_frame,
                        "action_was_applied":bool(records),
                        **contact,"nonwheel_ground_contact_count":nonwheel_count,
                        "nonwheel_ground_contact_force_norm_n":nonwheel_force,
                        "body_net_forces_w":s["body_net_forces_w"].copy(),
                        "qpos":data.qpos.copy(),"qvel":data.qvel.copy()})
        return s

    previous=np.zeros(8)
    pre=record(np.zeros((len(BODY_NAMES),3)))
    previous_vz=float(pre["body_lin_vel_w"][0,2])
    try:
        for k in range(steps):
            packed=predictor.layout.pack(data.qpos,data.qvel,history,k==0)
            held=predictor.objective.hold_reference(predictor.reference_frame(k),pre)
            now=time.perf_counter()
            action,info=controller.action(packed,k,previous_action=previous,previous_vz=previous_vz)
            elapsed=time.perf_counter()-now
            if action.shape!=(8,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.+1e-9:
                raise FloatingPointError("MPPI returned an invalid bounded action")
            raw_action=action.copy()
            # Identical finite action interface to the archived CEM/Isaac plants.
            action=action.astype(np.float32).astype(float)
            held_action=action.copy();held_frame=min(k,provider.reference.frames-1)
            for _ in range(provider.contract["decimation"]):
                data.ctrl[physical.aids]=provider.torque(action,data.qpos,data.qvel)
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                    raise RuntimeError("Unexpected external force in validation plant")
                history,current=extractor.update_contact_history(data,history)
                pre=record(current)
                expected=(len(records)-1)*model.opt.timestep
                if not np.isclose(data.time,expected,atol=1e-8,rtol=0) or not np.isfinite(np.r_[data.qpos,data.qvel]).all():
                    raise FloatingPointError("Nonfinite plant state or unexpected simulator clock reset")
            objective=predictor.objective.evaluate(pre,held,action,previous,previous_vz,reset_sample=k==0)
            surrogates.update(objective["surrogate_flags"])
            decisions.append({"frame":k,"reference_frame_for_reward":min(k,provider.reference.frames-1),
                              "time_s":float(data.time),"action":action.tolist(),"raw_mppi_action":raw_action.tolist(),"compute_s":elapsed,
                              "solver":info,"objective":objective})
            previous=action.copy();previous_vz=objective["base_com_vz"]
            if k%25==0:
                print(f"mppi frame={k}/{steps} plant_mu={args.plant_mu} prediction={controller.mu_scenarios} compute_ms={1000*elapsed:.2f}",flush=True)
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error),
                      completed_policy_steps=len(decisions),wall_s=time.perf_counter()-started)
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        raise
    trace={name:np.asarray([r[name] for r in records]) for name in records[0]}
    trace.update(actions=np.asarray([r["action"] for r in decisions]),
                 compute_s=np.asarray([r["compute_s"] for r in decisions]),
                 stage_cost=np.asarray([r["objective"]["stage_cost"] for r in decisions]),
                 reference_time_s=np.arange(provider.reference.frames)*provider.dt,
                 reference_root_xy_m=provider.reference.archive["body_pos_w"][:,0,:2],
                 qpos_joint_names=np.array([model.joint(i).name for i in range(1,model.njnt)]),
                 body_names=np.array(BODY_NAMES),action_joint_names=np.array(JOINT_NAMES))
    trace.update(physics_dt_s=np.array(model.opt.timestep),policy_dt_s=np.array(provider.dt))
    trace.update(reference_id=np.array(reference_spec["reference_id"]),
                 reference_role=np.array(reference_spec["role"]),reference_stratum=np.array(reference_spec["stratum"]),
                 reference_evaluation_core_interval_s=np.array(reference_spec["evaluation_core_interval_s"]),
                 reference_supplementary_event_windows_s=np.array(reference_spec["supplementary_event_windows_s"]).reshape(-1,2))
    report.update(_metrics(trace,provider.reference,reference_spec))
    np.savez_compressed(output/"rollout.npz",**trace)
    (output/"solver_log.json").write_text(json.dumps(decisions,indent=2,allow_nan=False)+"\n")
    timing=trace["compute_s"]
    report.update(status="smoke_completed" if truncated else "completed",completed_policy_steps=len(decisions),
                  duration_s=float(data.time),wall_s=time.perf_counter()-started,physics_samples=len(records),
                  compute_ms_median_p95_max=(1000*np.quantile(timing,[.5,.95,1.])).tolist(),
                  compute_over_control_dt_fraction=float(np.mean(timing>provider.dt)),
                  action_bound_fraction=float(np.mean(np.abs(trace["actions"])>=1.-1e-8)),
                  solver_failures=sum(bool(r["solver"]["optimization_failed"]) for r in decisions),
                  total_stage_cost=float(trace["stage_cost"].sum()),surrogate_flags=sorted(surrogates),
                  rollout_sha256=sha(output/"rollout.npz"),solver_log_sha256=sha(output/"solver_log.json"))
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({name:report[name] for name in ("status","duration_s","completed_policy_steps",
                     "core_coverage_complete","core_xy_rmse_m","compute_ms_median_p95_max","solver_failures")}),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("model","motion","contract","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reference-id",default=PRIMARY_REFERENCE_ID)
    p.add_argument("--task-profile",type=Path,default=ROOT/"config/car_drift_height_smooth_task.json")
    p.add_argument("--settings",type=Path,default=ROOT/"config/mppi_height_smooth.json")
    p.add_argument("--plant-mu",type=float,required=True)
    p.add_argument("--max-policy-steps",type=int,default=None)
    p.add_argument("--prediction-backend",choices=("python","native"),default="python")
    p.add_argument("--native-library",type=Path,default=None,
                   help="Explicit existing hash-verified library; no automatic build/restart")
    return p


if __name__=="__main__":
    run(parser().parse_args())
