"""Run one full-body MuJoCo height-smooth NMPC episode, or a short smoke.

The plant is initialized once and remains entirely physics-driven. Prediction
friction scenarios come only from the settings/explicit CLI override, or an
explicit adaptive model bank, never from the validation --plant-mu. No learned policy, TVLQR design, old Q/R,
root-force correction, pose reset, or absolute-height failure gate is used.
The cold-start qacc/time predictor-state limitation is explicit in the report.
"""
from __future__ import annotations

from contextlib import ExitStack
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from classical_tracking import Predictor, ROOT
from controller_references import DEFAULT_CATALOG,PRIMARY_REFERENCE_ID,get_reference
from nmpc_shooting import CEMConfig, ShootingNMPC
from nmpc_tracking_objective import (
    BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective, MujocoObjectiveState,
)


@dataclass(frozen=True)
class PackedState:
    """qpos/qvel plus objective memory, not additional physical actuation."""

    nq: int = 15
    nv: int = 14
    bodies: int = 9
    history_length: int = 3

    @property
    def physics_dimension(self):
        return self.nq+self.nv

    @property
    def history_shape(self):
        return (self.history_length,self.bodies,3)

    @property
    def dimension(self):
        return self.physics_dimension+int(np.prod(self.history_shape))+1

    def pack(self,qpos,qvel,history,first_reward_after_reset=False):
        qpos,qvel,history=(np.asarray(v,dtype=float) for v in (qpos,qvel,history))
        if qpos.shape!=(self.nq,) or qvel.shape!=(self.nv,) or history.shape!=self.history_shape:
            raise ValueError("Packed state qpos/qvel/contact-history shape mismatch")
        if not isinstance(first_reward_after_reset,(bool,np.bool_)):
            raise ValueError("Reset history flag must be boolean")
        packed=np.r_[qpos,qvel,history.reshape(-1),float(first_reward_after_reset)]
        if not np.isfinite(packed).all():
            raise FloatingPointError("Nonfinite packed state")
        return packed

    def unpack(self,state):
        state=np.asarray(state,dtype=float)
        if state.shape!=(self.dimension,) or not np.isfinite(state).all():
            raise ValueError("Invalid packed predictor state")
        if state[-1] not in (0.,1.):
            raise ValueError("Packed reset flag must be zero or one")
        return (state[:self.nq].copy(),state[self.nq:self.physics_dimension].copy(),
                state[self.physics_dimension:-1].reshape(self.history_shape).copy(),bool(state[-1]))

    def contract(self):
        return {"dimension":self.dimension,"qpos_dimension":self.nq,"qvel_dimension":self.nv,
                "contact_force_history_shape":list(self.history_shape),
                "algorithmic_memory":"Physical-substep contact-force history plus first-reward reset flag; not extra robot state or actuators.",
                "excluded_integration_state":"qacc warm-start and simulator time are cold-reset per prediction interval; no actuator activation state exists in this model.",
                "external_applied_forces":"Required zero in both predictor and plant; no prescribed root correction."}


class MujocoTrackingPredictor:
    """Isolated per-scenario MuJoCo dynamics and the new full reward objective."""

    def __init__(self,reference_provider,model_path,task_profile,mu_scenarios):
        self.provider=reference_provider
        self.api=reference_provider.api
        self.reference=reference_provider.reference
        self.contract=reference_provider.contract
        self.mu_scenarios=tuple(float(mu) for mu in mu_scenarios)
        if (not self.mu_scenarios or len(set(self.mu_scenarios))!=len(self.mu_scenarios)
                or not np.isfinite(self.mu_scenarios).all() or min(self.mu_scenarios)<0):
            raise ValueError("Declare unique finite nonnegative prediction friction scenarios")
        if tuple(self.reference.action_names)!=JOINT_NAMES:
            raise ValueError("Reference joint order differs from objective/action order")
        if any(name not in self.reference.body_names for name in BODY_NAMES):
            raise ValueError("Reference lacks a configured tracking body")
        # The exporter can include limx_imu and use MJCF-native ordering;
        # match the command's explicit nine tracking bodies by NAME, not index.
        self.reference_body_order=np.array([self.reference.body_names.index(name) for name in BODY_NAMES])
        self.objective=HeightSmoothTrackingObjective(task_profile,dt=reference_provider.dt)
        self.layout=PackedState()
        self.models={};self.datas={};self.extractors={}
        for mu in self.mu_scenarios:
            model,_=self.api.load_model(model_path,no_visual_mesh=True)
            self.api.configure_model(model,self.contract)
            model.geom_friction[:,0]=mu
            if model.na or model.nmocap or model.nuserdata:
                raise ValueError("This bridge does not support hidden activation/mocap/user state")
            self.models[mu]=model;self.datas[mu]=mujoco.MjData(model)
            self.extractors[mu]=MujocoObjectiveState(model)
        # A separate measurement workspace never changes rollout data/plant.
        self.measure_model=self.models[self.mu_scenarios[0]]
        self.measure_data=mujoco.MjData(self.measure_model)
        self.measure_extractor=MujocoObjectiveState(self.measure_model)
        self.full_references=[{
            key:self.reference.archive[key][k,self.reference_body_order].copy() for key in
            ("body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w")
        } for k in range(self.reference.frames)]

    def reference_frame(self,index):
        if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or index<0:
            raise ValueError("Reference index must be nonnegative")
        return self.full_references[min(index,self.reference.frames-1)]

    def measure(self,state):
        qpos,qvel,history,_=self.layout.unpack(state)
        d=self.measure_data;m=self.measure_model
        mujoco.mj_resetData(m,d);d.qpos[:]=qpos;d.qvel[:]=qvel
        mujoco.mj_forward(m,d)
        return self.measure_extractor.extract(d,contact_force_history=history,
                                              body_net_forces_w=history[-1])

    def transition(self,state,action,*,mu):
        if mu not in self.models:
            raise ValueError("Transition friction was not declared before validation")
        action=np.asarray(action,dtype=float)
        if action.shape!=(8,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.+1e-9:
            raise ValueError("Expected eight finite bounded normalized actions")
        qpos,qvel,history,_=self.layout.unpack(state)
        m,d,extractor=self.models[mu],self.datas[mu],self.extractors[mu]
        mujoco.mj_resetData(m,d);d.qpos[:]=qpos;d.qvel[:]=qvel
        for _ in range(self.contract["decimation"]):
            d.ctrl[self.provider.aids]=self.provider.torque(action,d.qpos,d.qvel)
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            history,_=extractor.update_contact_history(d,history)
            if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
                raise RuntimeError("Unexpected external force in isolated prediction")
        if not np.isclose(d.time,self.provider.dt,atol=1e-10,rtol=0):
            raise FloatingPointError("Prediction clock mismatch")
        return self.layout.pack(d.qpos,d.qvel,history,False)

    def stage_cost(self,state,action,next_state,reference_index,*,previous_action,previous_vz):
        pre=self.measure(state);post=self.measure(next_state)
        held=self.objective.hold_reference(self.reference_frame(reference_index),pre)
        reset=self.layout.unpack(state)[-1]
        return self.objective.evaluate(post,held,action,previous_action,previous_vz,
                                       reset_sample=reset)["stage_cost"]

    def vertical_velocity(self,state):
        return float(self.measure(state)["body_lin_vel_w"][0,2])

    def initial_candidate(self,state,start_index,*,previous_action,previous_vz,horizon_steps):
        # Bias-force feedforward only; no TVLQR K/feed/P or PPO action is read.
        indices=np.minimum(start_index+np.arange(horizon_steps),self.reference.frames-1)
        return self.provider.nominal_action[indices].copy()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_settings(args):
    settings=json.loads(Path(args.settings).read_text())
    if settings.get("schema_version")!=1 or settings.get("task")!="car_drift_height_smooth":
        raise ValueError("Expected schema 1 car_drift_height_smooth NMPC settings")
    config=dict(settings["nmpc"])
    for name in ("horizon_steps","control_blocks","candidates","iterations","elite_count"):
        value=getattr(args,name,None)
        if value is not None:
            config[name]=value
    for name in ("initial_std","min_std"):
        if isinstance(config.get(name),list):
            config[name]=tuple(config[name])
    cem=CEMConfig(**config)
    scenarios=tuple(settings["prediction_mu_scenarios"] if args.prediction_mu_scenarios is None
                    else args.prediction_mu_scenarios)
    if not scenarios or not np.isfinite(scenarios).all() or min(scenarios)<0 or len(set(scenarios))!=len(scenarios):
        raise ValueError("Prediction scenarios must be unique finite nonnegative mu values")
    return settings,cem,scenarios


def build_provider(model_path,motion_path,contract,settings,scenarios):
    # The old constructor still builds unused Q/R; only immutable actuator,
    # reference and bias-only nominal-action services are used by this runner.
    legacy=json.loads((ROOT/"config/classical_tracking.json").read_text())
    legacy.update(nominal_prediction_mu=scenarios[0],
                  policy_dt_s=settings["policy_dt_s"],physics_dt_s=settings["physics_dt_s"])
    return Predictor(model_path,motion_path,contract,legacy)


def selected_reference(args):
    """Fail before simulation if caller-supplied motion differs from its label."""
    spec=get_reference(getattr(args,"reference_id",PRIMARY_REFERENCE_ID))
    if sha(args.motion)!=spec["motion_npz_sha256"]:
        raise ValueError(f"motion_npz_sha256 mismatch for selected reference_id={spec['reference_id']}")
    return spec


def attach_model_nominal_plan(args, controller_predictor, scalar_predictor, reference_spec):
    """Opt-in ADDITIONAL model-only initial candidate; never action playback.

    The default imports no artifact loader and changes no candidate, score or
    random draw. Attach to the final Python/native predictor so both engines
    use the same CEM hook and actual-current-state scenario evaluation.
    """
    path=getattr(args,"model_nominal_plan",None)
    if path is None:
        return {"enabled":False}
    if hasattr(controller_predictor,"initial_candidates"):
        raise ValueError("Refusing to overwrite an existing additional-initial-candidate hook")
    from model_nominal_plan import load_model_nominal_plan
    model_path=getattr(args,"model",None) or getattr(args,"mujoco_model",None)
    plan=load_model_nominal_plan(path,
        expected_reference_id=reference_spec["reference_id"],
        expected_motion_sha256=sha(args.motion),
        expected_robot_reference_sha256=reference_spec["robot_reference_npz_sha256"],
        expected_model_sha256=sha(model_path),
        expected_actuator_contract_sha256=sha(args.contract),
        expected_objective_contract=scalar_predictor.objective.contract(),
        control_dt_s=scalar_predictor.provider.dt,action_joint_names=JOINT_NAMES)
    metadata=plan.report_contract()
    if metadata.get("enabled") is not True or metadata.get("artifact_sha256")!=sha(path):
        raise ValueError("Loaded model-generated nominal plan has inconsistent provenance")
    controller_predictor.initial_candidates=plan.initial_candidates
    return metadata


def _metrics(trace,reference,reference_spec=None):
    reference_spec=get_reference() if reference_spec is None else reference_spec
    t=trace["time_s"];positions=trace["root_pos"]
    clock=np.arange(reference.frames)/reference.fps
    desired=np.column_stack([np.interp(t,clock,reference.archive["body_pos_w"][:,0,i]) for i in range(2)])
    error=np.linalg.norm(positions[:,:2]-desired,axis=1)
    speed=np.linalg.norm(trace["base_com_velocity_w"][:,:2],axis=1)
    target_speed=np.linalg.norm(reference.archive["body_lin_vel_w"][:,0,:2],axis=1)
    def window_metrics(interval):
        start,end=interval
        mask=(t>=start-1e-9)&(t<=end+1e-9)
        complete=bool(t[0]<=start+1e-9 and t[-1]>=end-1e-9 and np.any(mask))
        result={"window_s":[start,end],"coverage_complete":complete,
                "xy_rmse_m":None,"xy_peak_m":None,"speed_mean_m_s":None,
                "reference_speed_mean_m_s":None,"speed_retention_mean_ratio":None}
        if complete:
            desired_speed=np.interp(t[mask],clock,target_speed)
            result.update(xy_rmse_m=float(np.sqrt(np.mean(error[mask]**2))),
                          xy_peak_m=float(error[mask].max()),speed_mean_m_s=float(speed[mask].mean()),
                          reference_speed_mean_m_s=float(desired_speed.mean()),
                          speed_retention_mean_ratio=float(speed[mask].mean()/desired_speed.mean()) if desired_speed.mean()>1e-12 else None)
        return result

    core=window_metrics(reference_spec["evaluation_core_interval_s"])
    events=[{**window_metrics(interval),"descriptive_not_success_gate":True}
            for interval in reference_spec["supplementary_event_windows_s"]]
    metrics={"reference_id":reference_spec["reference_id"],
             "role":reference_spec["role"],"stratum":reference_spec["stratum"],
             "pooled_success_statistics":False,
             "core_window_s":core["window_s"],"core_coverage_complete":core["coverage_complete"],
             **{"core_"+name:core[name] for name in ("xy_rmse_m","xy_peak_m","speed_mean_m_s",
                  "reference_speed_mean_m_s","speed_retention_mean_ratio")},
             "supplementary_event_metrics":events,
             "observed_window_xy_rmse_m":float(np.sqrt(np.mean(error**2))),
             "observed_window_speed_mean_m_s":float(speed.mean())}
    trace.update(reference_xy_m=desired,xy_error_m=error,speed_m_s=speed)
    return metrics


def run(args):
    # Native workers are owned by this run and close on every failure path.
    with ExitStack() as resources:
        return _run(args, resources)


def _run(args, resources):
    output=Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Preserving existing output directory: {output}")
    if not np.isfinite(args.plant_mu) or args.plant_mu<0:
        raise ValueError("Plant mu must be finite and nonnegative")
    if args.max_policy_steps is not None and args.max_policy_steps<1:
        raise ValueError("max-policy-steps must be positive")
    reference_spec=selected_reference(args)
    settings,cem,scenarios=effective_settings(args)
    adaptation_path=getattr(args,"adaptive_contact_config",None)
    adaptation=None
    if adaptation_path is not None:
        from adaptive_mpc_runtime import load_adaptation
        adaptation=load_adaptation(adaptation_path,args.prediction_mu_scenarios)
        scenarios=adaptation.mu_hypotheses
    task=json.loads(Path(args.task_profile).read_text())
    contract=json.loads(Path(args.contract).read_text())
    provider=build_provider(args.model,args.motion,contract,settings,scenarios)
    if provider.reference.frames!=reference_spec["frames"] or not np.isclose(provider.reference.fps,reference_spec["fps"]):
        raise ValueError("Selected reference frame count/fps differs from locked catalog")
    predictor=MujocoTrackingPredictor(provider,args.model,task,scenarios)
    backend=getattr(args,"prediction_backend","python")
    library=getattr(args,"native_library",None)
    physx_config=getattr(args,"physx_prediction_config",None)
    if backend not in ("python","native","physx") or (backend=="native") != (library is not None):
        raise ValueError("Native prediction is explicit opt-in and requires exactly one prebuilt --native-library")
    if (backend=="physx") != (physx_config is not None):
        raise ValueError("PhysX prediction requires exactly one explicit --physx-prediction-config")
    controller_predictor=predictor
    native_metadata=None
    physx_metadata=None
    if backend=="native":
        from nmpc_batch_objective import NativeBatchTrackingPredictor
        controller_predictor=resources.enter_context(NativeBatchTrackingPredictor(predictor,library))
        native_metadata=controller_predictor.metadata
    elif backend=="physx":
        from physx_batch_predictor import PhysXBatchTrackingPredictor
        controller_predictor=resources.enter_context(PhysXBatchTrackingPredictor(
            predictor,json.loads(Path(physx_config).read_text())))
        physx_metadata=controller_predictor.metadata
    model_nominal_plan=attach_model_nominal_plan(args,controller_predictor,predictor,reference_spec)
    if adaptation is None:
        controller=ShootingNMPC(controller_predictor,cem,action_low=-np.ones(8),action_high=np.ones(8),mu_scenarios=scenarios)
    else:
        from adaptive_nmpc import AdaptiveShootingNMPC
        from adaptive_mpc_runtime import ScalarIdentificationModel,transition_context,adaptation_summary
        controller=AdaptiveShootingNMPC(controller_predictor,cem,adaptation_config=adaptation,
            identification_model=ScalarIdentificationModel(predictor),action_low=-np.ones(8),action_high=np.ones(8))
    if backend in ("native","physx"):
        controller_predictor.record_cem_wiring(controller)
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
    source_files=[Path(__file__),ROOT/"scripts/controller_references.py",DEFAULT_CATALOG,
                  ROOT/"scripts/nmpc_shooting.py",ROOT/"scripts/nmpc_tracking_objective.py",
                  ROOT/"scripts/height_smoothness.py",ROOT/"scripts/classical_tracking.py",
                  ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py",
                  ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking.py",
                  ROOT/"upstream/tron1-learning-from-human/training/tron1_domain_randomization.py",
                  ROOT/"upstream/tron1-learning-from-human/training/tron1_contact_slip.py",
                  ROOT/"upstream/tron1-learning-from-human/training/tron1_tracking.py",
                  ROOT/"upstream/tron1-learning-from-human/training/tron1_drift.py",
                  ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
                  ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
                  ROOT/"config/classical_tracking.json",Path(args.settings),Path(args.task_profile),Path(args.contract)]
    if backend=="native":
        source_files.extend((ROOT/"scripts/nmpc_native_batch.py",ROOT/"scripts/nmpc_batch_objective.py",ROOT/"native/nmpc_batch.cpp"))
    elif backend=="physx":
        source_files.extend((Path(physx_config),ROOT/"scripts/physx_batch_predictor.py",
                             ROOT/"scripts/physx_batch_predictor_worker.py",ROOT/"scripts/nmpc_batch_objective.py"))
    if model_nominal_plan["enabled"]:
        source_files.extend((ROOT/"scripts/model_nominal_plan.py",Path(args.model_nominal_plan)))
    if adaptation is not None:
        source_files.extend((Path(adaptation_path),ROOT/"scripts/adaptive_contact_model.py",
                             ROOT/"scripts/adaptive_nmpc.py",ROOT/"scripts/adaptive_mpc_runtime.py"))
    report={"kind":"full_body_height_smooth_cem_nmpc","engine":"MuJoCo","mujoco_version":mujoco.__version__,
            "status":"running","model_sha256":sha(args.model),"motion_sha256":sha(args.motion),
            "reference_id":reference_spec["reference_id"],"role":reference_spec["role"],
            "stratum":reference_spec["stratum"],"reference_spec":reference_spec,
            "reference_catalog_sha256":sha(DEFAULT_CATALOG),"pooled_success_statistics":False,
            "source_sha256":{str(p.resolve()):sha(p) for p in source_files},
            "settings":settings,"effective_cem_config":asdict(cem),"plant_mu":args.plant_mu,
            "prediction_mu_scenarios":list(scenarios),"prediction_receives_plant_mu":False,
            "prediction_backend":backend,"native_prediction_backend":native_metadata,
            "prediction_engine":"Isaac/PhysX" if backend=="physx" else "MuJoCo",
            "physx_prediction_backend":physx_metadata,
            "action_parameterization":controller.parameterization_contract(),
            "model_generated_nominal_plan":model_nominal_plan,
            "plant_material_readback":{"compiled_sliding_friction_range":[float(model.geom_friction[:,0].min()),float(model.geom_friction[:,0].max())],
                                       "note":"Equal numeric contact coefficient, not a calibrated tire model."},
            "packed_state_contract":(controller_predictor.state_contract() if backend=="physx" else predictor.layout.contract()),
            "objective_contract":predictor.objective.contract(),
            "physics_dt_s":model.opt.timestep,"control_dt_s":provider.dt,
            "contact_threshold_n":5.,
            "wheel_lateral_contact_slip_evidence_kind":"actual_ground_contact_points",
            "contact_slip_definition":"Per-wheel force-weighted contact-point lateral-speed RMS; >threshold normal force, ground-filtered. Zero with no active contact is unavailable, not rolling.",
            "total_reference_policy_steps":provider.reference.frames,"evaluated_policy_steps":steps,
            "truncated_smoke":truncated,"pose_height_failure_gate":False,
            "learned_policy_used":False,"tvlqr_used":False,"old_qr_used":False,
            "root_state_prescribed_after_reset":False,"external_root_forces_used":False,
            "initial_plan":"Bounded bias-force joint feedforward from kinematic reference; not a feasible trajectory certificate.",
            "state_access":"Full simulator qpos/qvel, objective memory and future reference preview; unlike the actor-only observation interface.",
            "robustness_certificate":False,"real_time_verified":False,
            "isaac_result":False,"surrogate_flags":[]}
    if adaptation is not None:
        report.update(kind="full_body_height_smooth_adaptive_cem_mpc",
            controller_label="Finite-model contact-adaptive MPC (not certified robust MPC)",
            adaptation_contract=controller.adaptation_contract(),prediction_mu_model_bank=list(scenarios),
            prediction_mu_scenarios=None,prediction_selection="Causal singleton per solve; see solver log",
            identification_state_channels=["base_COM_vx_world_m_s","base_COM_vy_world_m_s","base_omega_z_world_rad_s"])
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
            if adaptation is None:
                action,info=controller.action(packed,k,previous_action=previous,previous_vz=previous_vz)
            else:
                context=None if k==0 else transition_context(records[-4:])
                action,info=controller.action(packed,k,previous_action=previous,previous_vz=previous_vz,
                                              transition_context=context)
            elapsed=time.perf_counter()-now
            if action.shape!=(8,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.+1e-9:
                raise FloatingPointError("NMPC returned an invalid bounded action")
            raw_action=action.copy()
            # Same finite action interface as the Isaac plant, not extra float64
            # actuator precision available only to the MuJoCo controller.
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
                              "time_s":float(data.time),"action":action.tolist(),"raw_cem_action":raw_action.tolist(),"compute_s":elapsed,
                              "solver":info,"objective":objective})
            previous=action.copy();previous_vz=objective["base_com_vz"]
            if k%25==0:
                print(f"nmpc frame={k}/{steps} plant_mu={args.plant_mu} prediction={controller.mu_scenarios} compute_ms={1000*elapsed:.2f}",flush=True)
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
                  duration_s=float(data.time),wall_s=time.perf_counter()-started,
                  compute_ms_median_p95_max=(1000*np.quantile(timing,[.5,.95,1.])).tolist(),
                  compute_over_control_dt_fraction=float(np.mean(timing>provider.dt)),
                  action_bound_fraction=float(np.mean(np.abs(trace["actions"])>=1.-1e-8)),
                  solver_failures=sum(bool(r["solver"]["optimization_failed"]) for r in decisions),
                  total_stage_cost=float(trace["stage_cost"].sum()),surrogate_flags=sorted(surrogates),
                  rollout_sha256=sha(output/"rollout.npz"),solver_log_sha256=sha(output/"solver_log.json"))
    if adaptation is not None:
        report["adaptation_summary"]=adaptation_summary(decisions)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({name:report[name] for name in ("status","duration_s","completed_policy_steps",
                     "core_coverage_complete","core_xy_rmse_m","compute_ms_median_p95_max","solver_failures")}),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--motion",type=Path,required=True)
    p.add_argument("--reference-id",default=PRIMARY_REFERENCE_ID,
                   help="Hash-locked primary or separately scored supplementary reference")
    p.add_argument("--contract",type=Path,required=True)
    p.add_argument("--task-profile",type=Path,default=ROOT/"config/car_drift_height_smooth_task.json")
    p.add_argument("--settings",type=Path,default=ROOT/"config/nmpc_height_smooth.json")
    p.add_argument("--plant-mu",type=float,required=True)
    p.add_argument("--prediction-mu-scenarios",type=float,nargs="+",default=None,
                   help="Predeclare predictor scenarios independently of plant mu")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--max-policy-steps",type=int,default=None)
    p.add_argument("--prediction-backend",choices=("python","native","physx"),default="python",
                   help="Native requires explicit prebuilt, hash-verified isolated CPU worker library")
    p.add_argument("--native-library",type=Path,default=None)
    p.add_argument("--physx-prediction-config",type=Path,default=None,
                   help="Explicit isolated Isaac/PhysX candidate-worker configuration")
    p.add_argument("--model-nominal-plan",type=Path,default=None,
                   help="Optional audited model-generated plan metadata; an extra rescored initial candidate, never applied-action replay")
    p.add_argument("--adaptive-contact-config",type=Path,default=None,
                   help="Explicit causal contact-model adaptation; fixed-model NMPC is unchanged when omitted")
    for name in ("horizon-steps","control-blocks","candidates","iterations","elite-count"):
        p.add_argument("--"+name,type=int,default=None,help="Explicit CEM override, saved in report")
    return p


if __name__=="__main__":
    run(parser().parse_args())
