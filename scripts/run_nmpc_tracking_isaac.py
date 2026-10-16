"""Replan the shared CEM controller on an Isaac plant using MuJoCo prediction.

This is NOT replay of MuJoCo actions, a learned actor, or same-engine NMPC.
The controller receives privileged measured state and reference preview. Only
the ordinary initial env.reset writes the Isaac root; subsequent root motion
is exclusively simulated physics. Every 5 ms physical sample is recorded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class SourceModelProvider:
    """Audited source-root helpers and existing actuator/bias-only methods.

    Does not instantiate legacy Q/R, TVLQR gains or a learned policy. The source
    checkout is explicit, so curated remote snapshots need no hidden upstream
    symlink. Changing the source helper is visible in the run's source hashes.
    """

    def __init__(self,source_root,model_path,motion_path,contract,settings,scenarios):
        import mujoco
        from classical_tracking import Predictor
        source_root=Path(source_root).resolve()
        for directory in (source_root,source_root/"scripts"):
            sys.path.insert(0,str(directory))
        path=source_root/"scripts/eval_tracking_mujoco.py"
        spec=importlib.util.spec_from_file_location("_isaac_nmpc_source_evaluator",path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load evaluator: {path}")
        self.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.api)
        self.contract=self.api.validate_contract(contract)
        expected={"leg_kp":500.,"leg_kd":10.,"leg_position_scale":1.,"wheel_torque_scale_nm":12.,
                  "leg_saturation_effort_nm":80.,"wheel_saturation_effort_nm":12.,
                  "leg_torque_limit_nm":80.,"wheel_torque_limit_nm":12.,
                  "leg_motor_velocity_limit_rad_s":15.,"wheel_motor_velocity_limit_rad_s":100.,
                  "physics_dt":.005,"decimation":4}
        if (self.contract.get("actuator_model")!="dc_motor"
                or self.contract.get("domain_parameters") is not None
                or self.contract.get("raw_action_clip")!=[-1.,1.]
                or any(not np.isclose(self.contract.get(k,np.nan),v) for k,v in expected.items())):
            raise ValueError("Isaac NMPC requires the shared nominal 80/12 Nm DC-motor action contract")
        if not np.isclose(settings["physics_dt_s"],.005) or not np.isclose(settings["policy_dt_s"],.02):
            raise ValueError("Isaac NMPC bridge supports exactly 200 Hz physics / 50 Hz control")
        self.model,_=self.api.load_model(model_path,no_visual_mesh=True)
        self.api.configure_model(self.model,self.contract)
        self.model.geom_friction[:,0]=scenarios[0]
        self.reference=self.api.MotionReference(motion_path,self.contract)
        self.data=mujoco.MjData(self.model)
        self.measure=self.api.MujocoState(self.model,self.data,self.reference)
        self.qids,self.vids,self.aids=self.measure.qids,self.measure.vids,self.measure.aids
        self.nv,self.n,self.dt=self.model.nv,self.reference.frames,.02
        self.q0=self.reference.q0
        self.effort=np.r_[np.full(6,80.),np.full(2,12.)]
        self.no_load=np.r_[np.full(6,15.),np.full(2,100.)]
        self._torque=Predictor.torque
        self.targets,self.nominal_action=Predictor.make_targets(self)

    def torque(self,action,qpos,qvel):
        return self._torque(self,action,qpos,qvel)


class IsaacToMujocoState:
    """Read a measured LINK pose/COM velocity; write ONLY isolated MjData."""

    def __init__(self,provider):
        import mujoco
        from run_nmpc_tracking import PackedState
        self.mj,self.provider,self.model=mujoco,provider,provider.model
        self.data=mujoco.MjData(self.model)
        self.measured=provider.api.MujocoState(self.model,self.data,provider.reference)
        self.layout=PackedState()
        self.base=self.model.body("base_Link").id
        self.com_offset=np.asarray(provider.api.ROOT_COM_LINK,dtype=float).copy()
        if not np.allclose(self.model.body_ipos[self.base],self.com_offset,atol=2e-7,rtol=0):
            raise ValueError("Prediction model does not retain the original unmerged base COM")

    def convert(self,measurement,contact_history,*,environment_origin=None,first_reward_after_reset=False):
        d,m=self.data,self.model
        def vector(name,width):
            value=np.asarray(measurement[name],dtype=float)
            if value.shape!=(width,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid measured {name}")
            return value
        position=vector("root_pos",3)
        quat=vector("root_quat_wxyz",4)
        if not np.isclose(np.linalg.norm(quat),1.,atol=1e-5,rtol=0):
            raise ValueError("Measured root quaternion is not normalized")
        quat=quat/np.linalg.norm(quat)
        origin=np.zeros(3) if environment_origin is None else np.asarray(environment_origin,dtype=float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError("Invalid environment origin")
        self.mj.mj_resetData(m,d)
        root_q=self.measured.root_qid
        d.qpos[root_q:root_q+7]=np.r_[position-origin,quat]
        d.qpos[self.provider.qids]=vector("joint_pos",8)
        d.qvel[:]=0.;d.qvel[self.provider.vids]=vector("joint_vel",8)
        self.mj.mj_forward(m,d)
        jp,jr=self.measured.root_jacobians()
        jac=np.vstack((jp,jr))
        desired=np.r_[vector("root_com_lin_vel_w",3),vector("root_ang_vel_w",3)]
        columns=slice(self.measured.root_vid,self.measured.root_vid+6)
        d.qvel[columns]=np.linalg.solve(jac[:,columns],desired-jac@d.qvel)
        self.mj.mj_forward(m,d)
        error=float(np.max(np.abs(jac@d.qvel-desired)))
        if error>1e-8:
            raise AssertionError("Measured COM velocity cannot be reconstructed in MuJoCo coordinates")
        return self.layout.pack(d.qpos,d.qvel,contact_history,first_reward_after_reset),error


def configure_fixed_evaluation(cfg,mu):
    """Remove termination/reset triggers; retain only fixed startup material/cleanup."""
    if not np.isfinite(mu) or mu<0:
        raise ValueError("Plant friction must be finite and nonnegative")
    removed=[]
    for name,term in list(vars(cfg.terminations).items()):
        if hasattr(term,"func") or (isinstance(term,dict) and "func" in term):
            setattr(cfg.terminations,name,None);removed.append(name)
    disabled=[]
    for name,term in list(vars(cfg.events).items()):
        if hasattr(term,"func") and name not in ("physics_material","legacy_joint_friction"):
            setattr(cfg.events,name,None);disabled.append(name)
    event=cfg.events.physics_material
    if event is None or event.mode!="startup":
        raise ValueError("Expected startup material assignment event")
    event.params.update(static_friction_range=(mu,mu),dynamic_friction_range=(mu,mu),
                        restitution_range=(0.,0.),num_buckets=1)
    for material in (cfg.scene.terrain.physics_material,cfg.sim.physics_material):
        material.static_friction=mu;material.dynamic_friction=mu;material.restitution=0.
    return {"removed_termination_terms":removed,"disabled_random_events":disabled,
            "all_termination_terms_removed":True,"static_dynamic_mu":mu,"restitution":0.}


class SubstepRecorder:
    """Only accept consecutive real simulator-counter increments inside actions."""

    def __init__(self,start_counter,physics_dt,initial_snapshot):
        self.start=self.last=int(start_counter);self.dt=float(physics_dt)
        self.records=[];self.in_step=False
        self.action=np.zeros(8);self.frame=0
        self._append(initial_snapshot,False)

    def _append(self,snapshot,applied):
        values={name:np.asarray(value).copy() for name,value in snapshot.items()}
        if any(value.dtype.kind in "fci" and not np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError("Nonfinite native physical snapshot")
        self.records.append({**values,"time_s":(self.last-self.start)*self.dt,
                             "simulation_step_counter":self.last,"held_action":self.action.copy(),
                             "reference_frame_for_action":self.frame,"action_was_applied":bool(applied)})

    def begin(self,counter,action,reference_frame):
        if self.in_step or int(counter)!=self.last:
            raise RuntimeError("Unexpected simulation step outside the controller clock")
        self.start_step=self.last;self.in_step=True
        self.action=np.asarray(action).copy();self.frame=int(reference_frame)

    def observe(self,counter,snapshot_factory):
        counter=int(counter)
        if not self.in_step or counter==self.last:
            return False
        if counter!=self.last+1:
            raise RuntimeError("Physics counter skipped or moved backwards")
        self.last=counter;self._append(snapshot_factory(),True)
        return True

    def finish(self,counter,decimation):
        if not self.in_step or int(counter)!=self.last or self.last-self.start_step!=decimation:
            raise RuntimeError("Physics recorder missed a decimation substep")
        self.in_step=False

    def arrays(self):
        return {key:np.asarray([row[key] for row in self.records]) for key in self.records[0]}


def scene_update_observer(original_update,counter,recorder,snapshot):
    def update(*args,**kwargs):
        result=original_update(*args,**kwargs)
        recorder.observe(counter(),snapshot)
        return result
    return update


def audit_materials(env,mu):
    """Native robot coefficients plus authored bound ground material readback."""
    from pxr import UsdPhysics,UsdShade
    import omni.usd
    robot=env.scene["robot"]
    material=robot.root_physx_view.get_material_properties().detach().cpu().numpy()
    if (not np.isfinite(material).all() or not np.allclose(material[...,:2],mu,atol=1e-6,rtol=0)
            or not np.allclose(material[...,2],0.,atol=1e-6,rtol=0)):
        raise AssertionError("Native Isaac robot friction/restitution differs from requested fixed material")
    ground=[]
    for prim in omni.usd.get_context().get_stage().Traverse():
        if str(prim.GetPath()).startswith("/World/ground") and prim.HasAPI(UsdPhysics.CollisionAPI):
            binding,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if not binding:
                raise AssertionError("Ground collider has no bound physics material")
            api=UsdPhysics.MaterialAPI(binding.GetPrim())
            values=[api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),api.GetRestitutionAttr().Get()]
            if None in values or not np.allclose(values,[mu,mu,0.],atol=1e-6,rtol=0):
                raise AssertionError("Ground bound material differs from requested fixed material")
            ground.append({"collider":str(prim.GetPath()),"material":str(binding.GetPath()),"coefficients":values})
    if not ground:
        raise AssertionError("No audited ground collider")
    paths={entry["collider"] for entry in ground}
    for name,sensor in env.scene.sensors.items():
        if name in ("wheel_L_ground","wheel_R_ground") or name.startswith("nonwheel_ground_"):
            if not set(sensor.cfg.filter_prim_paths_expr)<=paths:
                raise AssertionError(f"Contact filter does not resolve to ground collider: {name}")
    return {"robot_static_dynamic_range":[float(material[...,:2].min()),float(material[...,:2].max())],
            "robot_restitution_range":[float(material[...,2].min()),float(material[...,2].max())],
            "ground_bound_materials":ground,"expected_mu":mu}


def audit_actuators(robot):
    """Read the realized PD/DC envelope, not only its requested config."""
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)
    result={}
    for name,kp,kd,effort,speed in (("legs",500.,10.,80.,15.),("wheels",0.,0.,12.,100.)):
        motor=robot.actuators[name]
        values={"stiffness":array(motor.stiffness),"damping":array(motor.damping),
                "effort_limit_nm":array(motor.effort_limit),"velocity_limit_rad_s":array(motor.velocity_limit),
                "saturation_effort_nm":array(motor._saturation_effort),
                "velocity_at_effort_limit_rad_s":array(motor._vel_at_effort_lim)}
        expected={"stiffness":kp,"damping":kd,"effort_limit_nm":effort,
                  "velocity_limit_rad_s":speed,"saturation_effort_nm":effort,
                  "velocity_at_effort_limit_rad_s":2.*speed}
        for key,value in values.items():
            if not np.isfinite(value).all() or not np.allclose(value,expected[key],atol=1e-6,rtol=1e-6):
                raise AssertionError(f"Realized {name} actuator {key} differs from nominal shared contract")
        result[name]={key:[float(value.min()),float(value.max())] for key,value in values.items()}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","motion","asset-path","mujoco-model","contract","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--reference-id",default="drift_drl_traj4")
    parser.add_argument("--task-profile",type=Path,default=ROOT/"config/car_drift_height_smooth_task.json")
    parser.add_argument("--settings",type=Path,default=ROOT/"config/nmpc_height_smooth.json")
    parser.add_argument("--plant-mu",type=float,required=True)
    parser.add_argument("--prediction-mu-scenarios",type=float,nargs="+",default=None)
    parser.add_argument("--max-policy-steps",type=int,default=None)
    parser.add_argument("--num-envs",type=int,default=1)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--contact-threshold-n",type=float,default=5.)
    for name in ("horizon-steps","control-blocks","candidates","iterations","elite-count"):
        parser.add_argument("--"+name,type=int,default=None)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.num_envs!=1 or (args.max_policy_steps is not None and args.max_policy_steps<1):
        parser.error("Use one finite validation environment and a positive optional smoke length")
    if (not np.isfinite(args.plant_mu) or args.plant_mu<0
            or not np.isfinite(args.contact_threshold_n) or args.contact_threshold_n<0):
        parser.error("Invalid friction/contact threshold")
    output=args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Preserving prior Isaac NMPC output: "+str(output))
    for path in (args.motion,args.asset_path,args.mujoco_model,args.contract,args.settings,args.task_profile,
                 args.source_root/"scripts/eval_tracking_mujoco.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    from controller_references import DEFAULT_CATALOG,get_reference
    from hashlib import sha256
    def sha(path):
        return sha256(Path(path).read_bytes()).hexdigest()
    reference_spec=get_reference(args.reference_id)
    if sha(args.motion)!=reference_spec["motion_npz_sha256"]:
        raise ValueError("motion_npz_sha256 mismatch for selected reference_id")
    output.mkdir(parents=True)
    status={"kind":"receding_horizon_nmpc","status":"initializing",
            "engine":"Isaac/PhysX","prediction_engine":"MuJoCo",
            "controller_label":"Isaac plant / MuJoCo prediction model",
            "trained_policy":False,"learned_policy_used":False,"tvlqr_used":False,"action_replay":False,
            "root_state_prescribed_after_reset":False,"external_root_forces_used":False,
            "reference_spec":reference_spec,"reference_id":args.reference_id,"role":reference_spec["role"],
            "stratum":reference_spec["stratum"],"reference_catalog_sha256":sha(DEFAULT_CATALOG),
            "pooled_success_statistics":False,"robustness_certificate":False,"real_time_verified":False}
    def write_status():
        (output/"report.json").write_text(json.dumps(status,indent=2,allow_nan=False)+"\n")
    app=env=None;original_scene_update=None;started=time.perf_counter()
    try:
        app=AppLauncher(args).app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        sys.path.insert(0,str(args.source_root.resolve()))
        from training.tron1_tracking import make_env_cfg
        from training.tron1_drift import load_task_profile
        from training.tron1_runtime_device import validate_simulation_device
        from training.tron1_contact_slip import sample_raw_isaac_ground_contacts,sample_nonwheel_ground_force_norm
        from run_nmpc_tracking import MujocoTrackingPredictor,effective_settings,_metrics
        from nmpc_shooting import ShootingNMPC
        from nmpc_tracking_objective import BODY_NAMES,JOINT_NAMES,HeightSmoothTrackingObjective,quaternion_rotate,quaternion_inverse
        from height_smoothness import install_height_smoothness
        settings,cem,scenarios=effective_settings(args)
        profile=load_task_profile(args.task_profile)
        contract=json.loads(args.contract.read_text())
        provider=SourceModelProvider(args.source_root,args.mujoco_model,args.motion,contract,settings,scenarios)
        if provider.reference.frames!=reference_spec["frames"] or not np.isclose(provider.reference.fps,reference_spec["fps"]):
            raise ValueError("Reference frame count/fps differs from catalog")
        prediction=MujocoTrackingPredictor(provider,args.mujoco_model,profile,scenarios)
        conversion=IsaacToMujocoState(provider)
        cfg=make_env_cfg(args.motion,args.asset_path,1,args.device,eval_mode=True,task_profile=profile)
        cfg.seed=args.seed
        height_contract=install_height_smoothness(cfg,profile["height_regularization"])
        fixed_contract=configure_fixed_evaluation(cfg,args.plant_mu)
        if cfg.sim.dt!=.005 or cfg.decimation!=4:
            raise AssertionError("Isaac source clock is not the shared 200 Hz / 50 Hz contract")
        env=ManagerBasedRLEnv(cfg=cfg)
        if env.termination_manager.active_terms:
            raise AssertionError("A hidden auto-reset/termination gate remains active")
        device_audit=validate_simulation_device(env,requested_device=args.device)
        material_audit=audit_materials(env,args.plant_mu)
        if not np.isclose(env.sim.get_physics_dt(),.005,atol=1e-12,rtol=0) or not np.isclose(env.step_dt,.02,atol=1e-12,rtol=0):
            raise AssertionError("Runtime Isaac physics/control clock mismatch")
        robot=env.scene["robot"];sensor=env.scene.sensors["contact_forces"]
        actuator_audit=audit_actuators(robot)
        legacy_friction_audit=getattr(env,"legacy_joint_friction_audit",None)
        if not legacy_friction_audit or not legacy_friction_audit.get("all_environments_verified_zero"):
            raise AssertionError("Legacy PhysX joint friction has no verified-zero startup readback")
        body_ids=[robot.body_names.index(name) for name in BODY_NAMES]
        joint_ids=[robot.joint_names.index(name) for name in JOINT_NAMES]
        force_ids=[sensor.body_names.index(name) for name in BODY_NAMES]
        command=env.command_manager.get_term("motion")
        origin=env.scene.env_origins[0].detach().cpu().numpy().copy()
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        def measured_state():
            data=robot.data
            state={"body_names":BODY_NAMES,"body_pos_w":cpu(data.body_pos_w[0,body_ids])-origin,
                   "body_quat_w":cpu(data.body_quat_w[0,body_ids]),
                   "body_lin_vel_w":cpu(data.body_lin_vel_w[0,body_ids]),
                   "body_ang_vel_w":cpu(data.body_ang_vel_w[0,body_ids]),
                   "joint_pos":cpu(data.joint_pos[0,joint_ids]),
                   "soft_joint_pos_limits":cpu(data.soft_joint_pos_limits[0,joint_ids]),
                   "soft_joint_limits_source":"isaac_runtime",
                   "body_net_forces_w":cpu(sensor.data.net_forces_w[0,force_ids])}
            # Use [:,force_ids] after dropping env dim to avoid advanced-index
            # axis reordering; the fixed contract is (history=3,bodies=9,xyz=3).
            state["contact_force_history_w"]=cpu(sensor.data.net_forces_w_history[0][:,force_ids])
            measurement={"root_pos":cpu(data.body_pos_w[0,body_ids[0]]),
                         "root_quat_wxyz":state["body_quat_w"][0],
                         "root_com_lin_vel_w":state["body_lin_vel_w"][0],
                         "root_ang_vel_w":state["body_ang_vel_w"][0],
                         "joint_pos":state["joint_pos"],"joint_vel":cpu(data.joint_vel[0,joint_ids])}
            return measurement,state
        obs,_=env.reset()  # The sole post-construction normal physical reset.
        initial,pre=measured_state()
        if int(command.time_steps[0])!=0:
            raise AssertionError("Isaac validation did not start at reference frame zero")
        initial_pack,initial_conversion_error=conversion.convert(initial,pre["contact_force_history_w"],environment_origin=origin,first_reward_after_reset=True)
        reference_initial=provider.reference.sample(0)
        status["initial_state_audit"]={
            "reference_frame":int(command.time_steps[0]),
            "measured_mujoco_qpos":initial_pack[:15].tolist(),"measured_mujoco_qvel":initial_pack[15:29].tolist(),
            "reference_mujoco_qpos":provider.targets[0,:15].tolist(),"reference_mujoco_qvel":provider.targets[0,15:].tolist(),
            "qpos_max_abs_error":float(np.max(np.abs(initial_pack[:15]-provider.targets[0,:15]))),
            "qvel_max_abs_error":float(np.max(np.abs(initial_pack[15:29]-provider.targets[0,15:]))),
            "measured_base_com_velocity_world_m_s":initial["root_com_lin_vel_w"].tolist(),
            "reference_base_com_velocity_world_m_s":reference_initial["root_com_lin_vel_w"].tolist(),
            "base_com_velocity_max_abs_error":float(np.max(np.abs(initial["root_com_lin_vel_w"]-reference_initial["root_com_lin_vel_w"]))),
            "measured_base_angular_velocity_world_rad_s":initial["root_ang_vel_w"].tolist(),
            "reference_base_angular_velocity_world_rad_s":reference_initial["root_ang_vel_w"].tolist(),
        }
        actual_offset=quaternion_rotate(quaternion_inverse(pre["body_quat_w"][0]),cpu(robot.data.body_com_pos_w[0,body_ids[0]])-initial["root_pos"])
        actual_mass=float(cpu(robot.root_physx_view.get_masses())[0,body_ids[0]])
        status["base_mass_and_com_audit"]={"isaac_mass_kg":actual_mass,"mujoco_mass_kg":float(provider.model.body_mass[conversion.base]),
                                           "isaac_local_com_offset_m":actual_offset.tolist(),"mujoco_local_com_offset_m":conversion.com_offset.tolist()}
        np.testing.assert_allclose(initial_pack[:29],provider.targets[0],atol=2e-5,rtol=2e-5)
        if not np.allclose(actual_offset,conversion.com_offset,atol=2e-6,rtol=0) or not np.isclose(actual_mass,provider.model.body_mass[conversion.base],atol=2e-5,rtol=0):
            raise AssertionError("Isaac base mass/COM differs from original unmerged MuJoCo link")
        prediction.objective=HeightSmoothTrackingObjective(profile,dt=env.step_dt,soft_joint_limits=pre["soft_joint_pos_limits"])
        controller=ShootingNMPC(prediction,cem,action_low=-np.ones(8),action_high=np.ones(8),mu_scenarios=scenarios)
        nonwheel_sensors=tuple(name for name in env.scene.sensors if name.startswith("nonwheel_ground_"))
        if len(nonwheel_sensors)!=7:
            raise AssertionError("Expected seven name-filtered nonwheel ground sensors")
        def snapshot():
            measurement,state=measured_state()
            wheel=sample_raw_isaac_ground_contacts(env,force_threshold_n=args.contact_threshold_n)
            nonwheel=sample_nonwheel_ground_force_norm(env,nonwheel_sensors,force_threshold_n=1.)
            return {"root_pos":measurement["root_pos"]-origin,"root_quaternion_wxyz":measurement["root_quat_wxyz"],
                    "base_com_velocity_w":measurement["root_com_lin_vel_w"],"base_angular_velocity_w":measurement["root_ang_vel_w"],
                    "joint_pos":measurement["joint_pos"],"joint_vel":measurement["joint_vel"],
                    "wheel_pos":state["body_pos_w"][-2:],"body_net_forces_w":state["body_net_forces_w"],
                    "applied_joint_torque_nm":cpu(robot.data.applied_torque[0,joint_ids]),
                    **{key:value[0] for key,value in {**wheel,**nonwheel}.items()}}
        recorder=SubstepRecorder(env._sim_step_counter,.005,snapshot())
        original_scene_update=env.scene.update
        env.scene.update=scene_update_observer(original_scene_update,lambda:env._sim_step_counter,recorder,snapshot)
        source_paths=[Path(__file__),DEFAULT_CATALOG,args.settings,args.task_profile,args.contract]
        for filename in ("run_nmpc_tracking.py","nmpc_shooting.py","nmpc_tracking_objective.py","classical_tracking.py","controller_references.py","height_smoothness.py"):
            source_paths.append(Path(__file__).with_name(filename))
        for relative in ("scripts/eval_tracking_mujoco.py","scripts/eval_tracking.py","training/tron1_tracking.py",
                         "training/tron1_drift.py","training/tron1_domain_randomization.py","training/tron1_contact_slip.py",
                         "training/tron1_runtime_device.py","training/tron1_physics_clock.py"):
            source_paths.append(args.source_root/relative)
        for func in (make_env_cfg,sample_raw_isaac_ground_contacts,type(env).step):
            path=inspect.getsourcefile(func)
            if path is not None:
                source_paths.append(Path(path))
        for name in env.reward_manager.active_terms:
            func=env.reward_manager.get_term_cfg(name).func
            path=inspect.getsourcefile(func if inspect.isfunction(func) else type(func))
            if path is not None:
                source_paths.append(Path(path))
        steps=provider.reference.frames if args.max_policy_steps is None else min(args.max_policy_steps,provider.reference.frames)
        status.update(status="running",source_sha256={str(path.resolve()):sha(path) for path in source_paths},
                      motion_sha256=sha(args.motion),asset_sha256=sha(args.asset_path),mujoco_model_sha256=sha(args.mujoco_model),
                      effective_cem_config=asdict(cem),settings=settings,plant_seed=args.seed,
                      plant_mu=args.plant_mu,prediction_mu_scenarios=list(scenarios),prediction_receives_plant_mu=False,
                      fixed_evaluation=fixed_contract,height_contract=height_contract,
                      device_audit=device_audit,material_audit=material_audit,
                      actuator_audit=actuator_audit,legacy_joint_friction_audit=legacy_friction_audit,
                      objective_contract=prediction.objective.contract(),packed_state_contract=prediction.layout.contract(),
                      privileged_state_and_reference_preview=True,action_order=list(JOINT_NAMES),body_order=list(BODY_NAMES),
                      runtime_soft_joint_pos_limits=pre["soft_joint_pos_limits"].tolist(),
                      prediction_mujoco_version=conversion.mj.__version__,torch_version=str(torch.__version__),
                      initial_conversion_max_velocity_error=initial_conversion_error,
                      physics_dt_s=.005,control_dt_s=.02,physics_sampling="Every actual scene update after its original refresh; no reset/warmup callbacks.",
                      total_reference_policy_steps=provider.reference.frames,evaluated_policy_steps=steps,
                      truncated_smoke=steps<provider.reference.frames,pose_height_failure_gate=False,
                      contact_slip_definition="Per-wheel force-weighted contact-point lateral-speed RMS; >threshold normal force, ground-filtered. Zero with no active contact is unavailable, not rolling.",
                      timing_scope="Replanning call only; CUDA extraction, logging and plant stepping additionally cost time; no real-time deadline enforced.")
        write_status()
        previous_action=np.zeros(8);previous_vz=float(pre["body_lin_vel_w"][0,2]);decisions=[];packed_samples=[]
        for k in range(steps):
            if not app.is_running():
                raise RuntimeError("Isaac app closed before requested finite rollout finished")
            expected_frame=min(k,provider.reference.frames-1)
            if int(command.time_steps[0])!=expected_frame or int(env.episode_length_buf[0])!=k:
                raise AssertionError("Reference phase or episode length changed unexpectedly; possible hidden reset")
            measurement,pre=measured_state()
            packed,error=conversion.convert(measurement,pre["contact_force_history_w"],environment_origin=origin,first_reward_after_reset=k==0)
            held=prediction.objective.hold_reference(prediction.reference_frame(k),pre)
            now=time.perf_counter()
            action,info=controller.action(packed,k,previous_action=previous_action,previous_vz=previous_vz)
            elapsed=time.perf_counter()-now
            raw_action=action.copy()
            if action.shape!=(8,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.+1e-9:
                raise FloatingPointError("Controller produced an invalid bounded action")
            # Record/use the actual float32 action received by Isaac, including
            # the previous-action regularizer, not an unrounded CPU surrogate.
            action=action.astype(np.float32).astype(float)
            recorder.begin(env._sim_step_counter,action,expected_frame)
            obs,reward,terminated,truncated,_=env.step(torch.as_tensor(action[None],device=env.device,dtype=torch.float32))
            recorder.finish(env._sim_step_counter,4)
            if bool(terminated.any()) or bool(truncated.any()) or int(env.episode_length_buf[0])!=k+1:
                raise AssertionError("Unexpected Isaac termination/reset during finite NMPC rollout")
            if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                raise FloatingPointError("Nonfinite Isaac observation/reward")
            _,post=measured_state()
            objective=prediction.objective.evaluate(post,held,action,previous_action,previous_vz,reset_sample=k==0)
            decisions.append({"frame":k,"reference_frame_for_reward":expected_frame,"compute_s":elapsed,
                              "action":action.tolist(),"raw_cem_action":raw_action.tolist(),
                              "solver":info,"actual_isaac_objective":objective,
                              "actual_dt_scaled_reward":float(reward[0]),
                              "same_state_dt_reward_abs_error":abs(float(reward[0])-objective["total_reward"]*env.step_dt),
                              "coordinate_conversion_max_velocity_error":error})
            packed_samples.append(packed)
            previous_action=action.copy();previous_vz=objective["base_com_vz"]
            if k%25==0:
                print(f"Isaac/MuJoCo NMPC frame={k}/{steps} plant_mu={args.plant_mu} prediction={scenarios} compute_ms={1000*elapsed:.2f}",flush=True)
        env.scene.update=original_scene_update;original_scene_update=None
        trace=recorder.arrays()
        trace.update(actions=np.asarray([row["action"] for row in decisions]),
                     compute_s=np.asarray([row["compute_s"] for row in decisions]),packed_controller_states=np.asarray(packed_samples),
                     reference_id=np.array(args.reference_id),reference_role=np.array(reference_spec["role"]),reference_stratum=np.array(reference_spec["stratum"]),
                     action_joint_names=np.array(JOINT_NAMES),body_names=np.array(BODY_NAMES),physics_dt_s=np.array(.005),policy_dt_s=np.array(.02),
                     reference_time_s=np.arange(provider.reference.frames)*.02,reference_root_xy_m=provider.reference.archive["body_pos_w"][:,0,:2])
        status.update(_metrics(trace,provider.reference,reference_spec))
        np.savez_compressed(output/"rollout.npz",**trace)
        (output/"solver_log.json").write_text(json.dumps(decisions,indent=2,allow_nan=False)+"\n")
        timing=trace["compute_s"]
        status.update(status="smoke_completed" if steps<provider.reference.frames else "completed",completed_policy_steps=steps,
                      duration_s=float(trace["time_s"][-1]),physics_samples=len(trace["time_s"]),
                      compute_ms_median_p95_max=(1000*np.quantile(timing,[.5,.95,1.])).tolist(),
                      compute_over_control_dt_fraction=float(np.mean(timing>.02)),
                      action_bound_fraction=float(np.mean(np.abs(trace["actions"])>=1.-1e-8)),
                      solver_failures=sum(bool(row["solver"]["optimization_failed"]) for row in decisions),
                      maximum_coordinate_conversion_error=max(row["coordinate_conversion_max_velocity_error"] for row in decisions),
                      maximum_same_state_dt_reward_abs_error=max(row["same_state_dt_reward_abs_error"] for row in decisions),
                      rollout_sha256=sha(output/"rollout.npz"),solver_log_sha256=sha(output/"solver_log.json"),elapsed_s=time.perf_counter()-started)
        write_status()
        print(json.dumps({key:status[key] for key in ("status","controller_label","duration_s","core_xy_rmse_m","solver_failures")}),flush=True)
    except BaseException as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}",elapsed_s=time.perf_counter()-started)
        raise
    finally:
        write_status()
        if env is not None:
            if original_scene_update is not None:
                env.scene.update=original_scene_update
            env.close()
        if app is not None:
            app.close(skip_cleanup=True)


if __name__=="__main__":
    main()
