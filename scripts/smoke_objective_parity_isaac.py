"""Same-state NumPy/Isaac reward arithmetic check, not sim2sim or training.

Capture happens inside RewardManager.compute, after physics and before any
automatic reset. Both implementations see exactly the SAME Isaac state,
contact sensor samples, runtime soft joint limits, held reference and actions.
A normal partial environment reset exercises per-environment smoothness reset.
No root-state writes occur beyond the simulator's normal reset implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","motion-file","asset-path","task-profile","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--num-envs",type=int,default=2)
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--atol",type=float,default=5e-5)
    parser.add_argument("--rtol",type=float,default=1e-5)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.num_envs!=2 or not 4<=args.steps<=16:
        parser.error("This parity smoke requires two environments and 4..16 control steps")
    if args.atol<=0 or args.rtol<0:
        parser.error("Invalid parity tolerances")
    output=args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Preserving previous parity output: "+str(output))
    output.mkdir(parents=True)
    status={"kind":"same_isaac_state_numpy_reward_parity","status":"initializing",
            "trained_policy":False,"closed_loop_tracking_success_claimed":False,
            "sim2sim_physics_parity_claimed":False}
    app=env=None;original_compute=None;captures=[];samples=[];started=time.monotonic()
    try:
        app=AppLauncher(args).app
        import numpy as np
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        sys.path.insert(0,str(args.source_root.resolve()))
        from training.tron1_tracking import make_env_cfg
        from training.tron1_drift import load_task_profile
        from training.tron1_runtime_device import validate_simulation_device
        from train_beyondmimic_style import install_beyondmimic_events,audit_material_draw
        from height_smoothness import install_height_smoothness
        from nmpc_tracking_objective import BODY_NAMES,JOINT_NAMES,HeightSmoothTrackingObjective

        def cpu(value):
            return value.detach().cpu().numpy().copy()

        profile=load_task_profile(args.task_profile)
        cfg=make_env_cfg(args.motion_file,args.asset_path,args.num_envs,args.device,task_profile=profile)
        cfg.seed=args.seed
        recipe=install_beyondmimic_events(cfg)
        height_contract=install_height_smoothness(cfg,profile["height_regularization"])
        env=ManagerBasedRLEnv(cfg=cfg)
        device_audit=validate_simulation_device(env,requested_device=args.device)
        material_audit=audit_material_draw(env,recipe)
        objective=HeightSmoothTrackingObjective(profile,dt=env.step_dt)
        robot=env.scene["robot"]
        command=env.command_manager.get_term("motion")
        sensor=env.scene.sensors["contact_forces"]
        body_ids=[robot.body_names.index(name) for name in BODY_NAMES]
        reference_ids=[command.cfg.body_names.index(name) for name in BODY_NAMES]
        joint_ids=[robot.joint_names.index(name) for name in JOINT_NAMES]
        force_ids=[sensor.body_names.index(name) for name in BODY_NAMES]
        smooth=env.reward_manager.get_term_cfg("height_vertical_acceleration").func
        term_names=list(env.reward_manager.active_terms)
        if set(term_names)!=set(objective.weights):
            raise AssertionError(f"Active reward terms differ: Isaac={term_names}, NumPy={list(objective.weights)}")
        weights=np.array([env.reward_manager.get_term_cfg(name).weight for name in term_names])
        if any(not np.isclose(weight,objective.weights[name]) for name,weight in zip(term_names,weights)):
            raise AssertionError("NumPy/Isaac objective weights differ")
        if np.any(weights==0):
            raise AssertionError("Parity raw-value reconstruction requires nonzero active weights")
        joint_limit_cfg=env.reward_manager.get_term_cfg("joint_limit").params["asset_cfg"]
        actual_limit_joint_names=[robot.joint_names[i] for i in joint_limit_cfg.joint_ids]
        if actual_limit_joint_names!=list(JOINT_NAMES[:6]):
            raise AssertionError("Joint-limit reward is not the expected six leg joints")

        def state_arrays():
            data=robot.data
            return {"body_pos_w":cpu(data.body_pos_w[:,body_ids]),
                    "body_quat_w":cpu(data.body_quat_w[:,body_ids]),
                    # Match exactly the alias used by the pinned command/reward.
                    "body_lin_vel_w":cpu(data.body_lin_vel_w[:,body_ids]),
                    "body_ang_vel_w":cpu(data.body_ang_vel_w[:,body_ids]),
                    "joint_pos":cpu(data.joint_pos[:,joint_ids]),
                    "soft_joint_pos_limits":cpu(data.soft_joint_pos_limits[:,joint_ids]),
                    "body_net_forces_w":cpu(sensor.data.net_forces_w[:,force_ids]),
                    "contact_force_history_w":cpu(sensor.data.net_forces_w_history[:,:,force_ids])}

        def state_row(arrays,index):
            return {**{name:value[index] for name,value in arrays.items()},
                    "body_names":BODY_NAMES,"soft_joint_limits_source":"isaac_runtime"}

        def reference_arrays():
            return {"body_pos_w":cpu(command.body_pos_w[:,reference_ids]),
                    "body_quat_w":cpu(command.body_quat_w[:,reference_ids]),
                    "body_lin_vel_w":cpu(command.body_lin_vel_w[:,reference_ids]),
                    "body_ang_vel_w":cpu(command.body_ang_vel_w[:,reference_ids])}

        source_paths=[Path(__file__),Path(__file__).with_name("nmpc_tracking_objective.py"),
                      Path(__file__).with_name("height_smoothness.py"),
                      Path(__file__).with_name("train_beyondmimic_style.py"),args.task_profile]
        for relative in ("training/tron1_tracking.py","training/tron1_drift.py",
                         "training/tron1_runtime_device.py","training/tron1_domain_randomization.py",
                         "training/tron1_contact_slip.py"):
            source_paths.append(args.source_root/relative)
        # Record the actual installed implementations, not assumed release IDs.
        source_paths.extend(Path(inspect.getsourcefile(func)) for func in (
            type(env.reward_manager).compute,
            env.reward_manager.get_term_cfg("joint_limit").func,
            env.reward_manager.get_term_cfg("undesired_contacts").func))
        for name in term_names:
            func=env.reward_manager.get_term_cfg(name).func
            path=inspect.getsourcefile(func if inspect.isfunction(func) else type(func))
            if path is not None:
                source_paths.append(Path(path))
        status.update(source_sha256={str(p.resolve()):sha(p) for p in source_paths},
                      motion_sha256=sha(args.motion_file),asset_sha256=sha(args.asset_path),
                      objective_contract=objective.contract(),height_contract=height_contract,
                      device_audit=device_audit,material_draw_audit=material_audit,
                      recipe=recipe,term_names=term_names,weights=weights.tolist(),
                      atol=args.atol,rtol=args.rtol,seed=args.seed,
                      body_names=list(BODY_NAMES),joint_names=list(JOINT_NAMES),
                      joint_limit_joint_names=actual_limit_joint_names,
                      native_body_ids=body_ids,command_body_ids=reference_ids,
                      native_joint_ids=joint_ids,contact_sensor_body_ids=force_ids,
                      environment_origins_m=cpu(env.scene.env_origins).tolist(),
                      measurement_semantics="Link-origin poses, original-link COM linear velocities and world angular velocities; exact same Isaac tensors as training rewards.")
        write_json(output/"run_status.json",status)
        env.reset()
        context={"held":None,"reference":None,"frames":None,"step":None}
        original_compute=env.reward_manager.compute

        def capture_reward(dt):
            if context["held"] is None:
                raise RuntimeError("Reward computed without a pre-action held reference")
            post=state_arrays()
            old_vz=cpu(smooth.previous_vz)
            reset=~cpu(smooth.valid).astype(bool)
            actions=cpu(env.action_manager.action)
            previous_actions=cpu(env.action_manager.prev_action)
            held_relative_pos=cpu(command.body_pos_relative_w[:,reference_ids])
            held_relative_quat=cpu(command.body_quat_relative_w[:,reference_ids])
            actual=original_compute(dt)
            # Installed IsaacLab stores weighted reward DENSITIES in _step_reward.
            step_weighted=cpu(env.reward_manager._step_reward)
            total=cpu(actual)
            if not np.allclose(step_weighted.sum(-1)*dt,total,atol=1e-6,rtol=1e-5):
                raise AssertionError("RewardManager._step_reward density semantics changed")
            rows=[]
            for index in range(args.num_envs):
                held=context["held"][index]
                np.testing.assert_allclose(held.relative_body_pos_w,held_relative_pos[index],atol=1e-5,rtol=1e-5)
                # q and -q describe the same orientation; compare antipodal-safe.
                agreement=np.abs(np.sum(held.relative_body_quat_w*held_relative_quat[index],axis=-1))
                if np.max(np.abs(agreement-1.))>1e-5:
                    raise AssertionError("Frozen relative-body quaternion targets changed before reward")
                result=objective.evaluate(state_row(post,index),held,actions[index],previous_actions[index],
                                           old_vz[index],reset_sample=bool(reset[index]))
                raw_actual=step_weighted[index]/weights
                raw_expected=np.array([result["raw_terms"][name] for name in term_names])
                weighted_expected=np.array([result["weighted_terms"][name] for name in term_names])
                raw_error=np.abs(raw_expected-raw_actual)
                weighted_error=np.abs(weighted_expected-step_weighted[index])
                expected_total=result["total_reward"]*dt
                okay=bool(np.allclose(raw_expected,raw_actual,atol=args.atol,rtol=args.rtol)
                          and np.allclose(weighted_expected,step_weighted[index],atol=args.atol,rtol=args.rtol)
                          and np.isclose(expected_total,total[index],atol=args.atol*dt,rtol=args.rtol))
                row={"step":context["step"],"environment":index,
                     "held_reference_frame":int(context["frames"][index]),
                     "reset_smoothness_sample":bool(reset[index]),"passed":okay,
                     "raw_actual":dict(zip(term_names,raw_actual.tolist())),
                     "raw_numpy":result["raw_terms"],
                     "raw_abs_error":dict(zip(term_names,raw_error.tolist())),
                     "weighted_abs_error":dict(zip(term_names,weighted_error.tolist())),
                     "dt_scaled_reward_actual":float(total[index]),
                     "dt_scaled_reward_numpy":expected_total,
                     "dt_scaled_reward_abs_error":abs(expected_total-float(total[index])),
                     "surrogate_flags":result["surrogate_flags"]}
                rows.append(row)
            samples.extend(rows)
            captures.append({**post,**{"reference_"+k:v for k,v in context["reference"].items()},
                             "actions":actions,"previous_actions":previous_actions,
                             "previous_base_com_vz":old_vz,"reset_sample":reset,
                             "held_relative_body_pos_w":held_relative_pos,
                             "held_relative_body_quat_w":held_relative_quat,
                             "held_reference_frames":context["frames"],
                             "actual_weighted_terms":step_weighted,"actual_dt_reward":total})
            return actual

        env.reward_manager.compute=capture_reward
        partial_step=args.steps//2
        for step in range(args.steps):
            if step==partial_step:
                # This is the simulator's normal per-environment reset routine,
                # not a manually prescribed robot state or target correction.
                env._reset_idx(torch.tensor([0],device=env.device,dtype=torch.long))
            pre=state_arrays();reference=reference_arrays()
            context.update(held=[objective.hold_reference(state_row(reference,i),state_row(pre,i))
                                 for i in range(args.num_envs)],
                           reference=reference,frames=cpu(command.time_steps),step=step)
            obs,reward,terminated,truncated,_=env.step(torch.zeros(args.num_envs,8,device=env.device))
            if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                raise FloatingPointError("Nonfinite parity smoke observation/reward")
        if len(samples)!=args.num_envs*args.steps:
            raise AssertionError("Missing pre-reset reward captures")
        if not all(s["reset_smoothness_sample"] for s in samples if s["step"]==0):
            raise AssertionError("Initial reward was not reset-suppressed")
        manual=[s for s in samples if s["step"]==partial_step and s["environment"]==0]
        if len(manual)!=1 or not manual[0]["reset_smoothness_sample"]:
            raise AssertionError("Partial reset did not suppress environment-zero smoothness history")
        arrays={name:np.stack([capture[name] for capture in captures]) for name in captures[0]}
        arrays.update(term_names=np.array(term_names),body_names=np.array(BODY_NAMES),
                      joint_names=np.array(JOINT_NAMES),dt_s=np.array(env.step_dt))
        np.savez_compressed(output/"same_state_snapshots.npz",**arrays)
        write_json(output/"samples.json",samples)
        maxima={name:{"raw":max(s["raw_abs_error"][name] for s in samples),
                      "weighted":max(s["weighted_abs_error"][name] for s in samples)} for name in term_names}
        passed=all(s["passed"] for s in samples)
        status.update(status="parity_passed" if passed else "parity_failed",steps=args.steps,num_envs=args.num_envs,
                      compared_samples=len(samples),partial_manual_reset_step=partial_step,
                      max_abs_error_by_term=maxima,
                      max_dt_scaled_reward_abs_error=max(s["dt_scaled_reward_abs_error"] for s in samples),
                      first_reward_smoothness_suppressed=True,partial_reset_smoothness_suppressed=True,
                      failed_samples=[{"step":s["step"],"environment":s["environment"]} for s in samples if not s["passed"]],
                      snapshots_sha256=sha(output/"same_state_snapshots.npz"),samples_sha256=sha(output/"samples.json"),
                      elapsed_s=time.monotonic()-started)
        write_json(output/"run_status.json",status)
        print(json.dumps({key:status[key] for key in ("status","compared_samples","max_dt_scaled_reward_abs_error","failed_samples")}),flush=True)
        if not passed:
            raise AssertionError("Same-state NumPy/Isaac reward parity failed; inspect per-term samples")
    except BaseException as error:
        if status.get("status")!="parity_failed":
            status["status"]="failed"
        status.update(error=f"{type(error).__name__}: {error}",elapsed_s=time.monotonic()-started)
        raise
    finally:
        write_json(output/"run_status.json",status)
        if env is not None:
            if original_compute is not None:
                env.reward_manager.compute=original_compute
            env.close()
        if app is not None:
            app.close(skip_cleanup=True)


if __name__=="__main__":
    main()
