#!/usr/bin/env python3
"""Preview raw GVHMR -> pinned GMR -> TRON1 WF, including each wheel's axle.

Kinematics only. Monocular world scale is not externally calibrated; this
output is not admitted for training or a certificate of contact/sliding.
GMR internally solves its optimization using Mink; no standalone IK baseline
is substituted. Wheel spin is held at a placeholder zero, not inferred.
"""
import argparse
import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hdm05_gmr import load_gmr, rolling_geometry, sha, wrap, axial

ROOT = Path(__file__).resolve().parents[1]


def wheel_headings_from_axles(axles):
    """Axial rolling-plane heading, with an explicit per-frame/per-wheel shape."""
    axles = np.asarray(axles, dtype=float)
    if axles.ndim != 3 or axles.shape[1:] != (2, 3):
        raise ValueError("Expected (frames, left/right, xyz) axles")
    if np.any(np.linalg.norm(axles[..., :2], axis=-1) < 1e-8):
        raise ValueError("A vertical wheel axle has no ground-plane heading")
    return np.degrees(np.arctan2(-axles[..., 0], axles[..., 1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--smplx-neutral", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory to preserve previous results")
    source = np.load(args.world, allow_pickle=False)
    if bool(source["postproc"]):
        raise ValueError("The primary reference must use postproc=False")
    t = source["time_s"]
    settings = json.loads((ROOT / "config/hdm05_gmr_experiment.json").read_text())
    GMR, params, _, _ = load_gmr(settings)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if (model.nq, model.nv, model.nu) != (15,14,8):
        raise ValueError("Expected TRON1 WF 8-joint model")
    data = mujoco.MjData(model)
    seed = model.qpos0.copy()
    seed[:7] = [0,0,0,1,0,0,0]
    data.qpos[:] = seed;mujoco.mj_forward(model,data)
    seed[2] -= np.mean(rolling_geometry(model,data)[1])
    data.qpos[:] = seed;mujoco.mj_forward(model,data)
    wheel_ids = [model.body(f"wheel_{s}_Link").id for s in ("L","R")]
    wheels0 = data.xpos[wheel_ids].copy()
    robot_leg_lengths=[]
    for side in ("left","right"):
        chain=data.site_xpos[[model.site(side+"_"+k).id for k in ("hip","knee","wheel")]]
        robot_leg_lengths.append(np.linalg.norm(np.diff(chain,axis=0),axis=-1).sum())
    j = source["joints_world_estimated_m"]
    human_leg_lengths = [np.linalg.norm(j[0,hip]-j[0,knee])+np.linalg.norm(j[0,knee]-j[0,ankle])
                         for hip,knee,ankle in ((1,4,7),(2,5,8))]
    scale = float(np.mean(robot_leg_lengths)/np.mean(human_leg_lengths))
    # Official GVHMR/GMR convention: Y-up -> Z-up. Robot forward +X is
    # mapped from SMPL local +Z, robot left +Y from SMPL local +X.
    conversion=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    axes=np.array([[0,1,0],[0,0,1],[1,0,0.]])
    parents=np.load(args.smplx_neutral,allow_pickle=False)["kintree_table"][0,:22].astype(int)
    local=Rotation.from_rotvec(np.concatenate([source["global_orient"],source["body_pose"]],axis=-1).reshape(-1,3)).as_matrix().reshape(len(t),22,3,3)
    global_rot=local.copy()
    for k in range(1,22):global_rot[:,k]=global_rot[:,parents[k]]@local[:,k]
    target_rot=conversion@global_rot@axes
    yaw0=np.arctan2(target_rot[0,0,1,0],target_rot[0,0,0,0])
    align=Rotation.from_rotvec([0,0,-yaw0]).as_matrix()
    target_rot=align@target_rot
    human=j@(align@conversion).T
    # Preserve initial crouch/root-to-ankle height at robot leg scale. Use
    # constant initial per-foot XY offsets to match robot morphology; never
    # zero subsequent ankle displacement or lock a foot to the floor.
    initial_ankle_height=np.mean(human[0,[7,8],2])
    seed[2]=np.mean(wheels0[:,2])+scale*(human[0,0,2]-initial_ankle_height)
    root=seed[:3]+scale*(human[:,0]-human[0,0])
    rel=scale*(human[:,[7,8]]-human[:,[0]])
    offset=wheels0[:,:2]-seed[None,:2]-rel[0,:,:2]
    offset=np.pad(offset,((0,0),(0,1)))
    relative_root_rot=target_rot[:,0]@target_rot[0,0].T
    wheel_target=root[:,None]+rel+np.einsum('tij,kj->tki',relative_root_rot,offset)
    match={"base_Link":["Pelvis",10,10,[0,0,0],[1,0,0,0]],
           "wheel_L_Link":["LeftAnkle",100,[10,0,10],[0,0,0],[1,0,0,0]],
           "wheel_R_Link":["RightAnkle",100,[10,0,10],[0,0,0],[1,0,0,0]]}
    cfg={"robot_root_name":"base_Link","human_root_name":"Pelvis","ground_height":0,
         "human_height_assumption":1.8,"use_ik_match_table1":True,"use_ik_match_table2":False,
         "human_scale_table":{s:1. for s in ("Pelvis","LeftAnkle","RightAnkle")},
         "ik_match_table1":match,"ik_match_table2":{}}
    args.output.mkdir(parents=True,exist_ok=True)
    cfg_path=(args.output/"gmr_task.json").resolve();cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
    params.ROBOT_XML_DICT["tron1_hockey_wf"]=args.model.resolve()
    params.IK_CONFIG_DICT["hockey_gvhmr_raw"]={"tron1_hockey_wf":cfg_path}
    solver=GMR("hockey_gvhmr_raw","tron1_hockey_wf",solver=settings["solver"],damping=settings["damping"],verbose=False,use_velocity_limit=False)
    solver.max_iter=settings["max_iter_per_stage"]
    solver.model.opt.timestep=float(np.median(np.diff(t)))
    solver.configuration.update(seed);solver.set_ground_offset(0.)
    solver.ik_limits.append(mink.VelocityLimit(solver.model,{"wheel_L_Joint":0.,"wheel_R_Joint":0.}))
    qpos=[];centers=[];headings=[];bottoms=[];axles=[];body_heading=[]
    for i in range(len(t)):
        targets={"Pelvis":[root[i].copy(),Rotation.from_matrix(target_rot[i,0]).as_quat(scalar_first=True)]}
        for k,name in enumerate(("LeftAnkle","RightAnkle")):
            targets[name]=[wheel_target[i,k].copy(),Rotation.from_matrix(target_rot[i,[7,8][k]]).as_quat(scalar_first=True)]
        q=solver.retarget(targets,offset_to_ground=False)
        qpos.append(q);data.qpos[:]=q;mujoco.mj_forward(model,data)
        centers.append(data.xpos[wheel_ids].copy());h,b=rolling_geometry(model,data)
        headings.append(h);bottoms.append(b)
        axles.append([data.xmat[wid].reshape(3,3)@model.jnt_axis[model.joint(f"wheel_{side}_Joint").id] for wid,side in zip(wheel_ids,("L","R"))])
        body_heading.append(np.degrees(np.arctan2(data.xmat[model.body("base_Link").id].reshape(3,3)[1,0],data.xmat[model.body("base_Link").id].reshape(3,3)[0,0])))
    qpos,centers,headings,bottoms,axles=map(np.asarray,(qpos,centers,headings,bottoms,axles))
    desired_axles=target_rot[:,[7,8]][...,:,1]
    desired_heading=wheel_headings_from_axles(desired_axles)
    heading_error=axial(headings-desired_heading)
    axle_error=np.degrees(np.arccos(np.clip(np.abs(np.sum(axles*desired_axles,axis=-1)),0,1)))
    velocity=np.gradient(qpos[:,:2],t,axis=0);speed=np.linalg.norm(velocity,axis=-1)
    course=np.degrees(np.arctan2(velocity[:,1],velocity[:,0]))
    reference_velocity=np.gradient(root[:,:2],t,axis=0)
    reference_course=np.degrees(np.arctan2(reference_velocity[:,1],reference_velocity[:,0]))
    center_velocity=np.gradient(centers[...,:2],t,axis=0)
    center_course=np.degrees(np.arctan2(center_velocity[...,1],center_velocity[...,0]))
    np.savez_compressed(args.output/"retarget_trace.npz",time_s=t,qpos=qpos,
                        human_joints_zup_m=seed[None,None,:3]+scale*(human-human[:1,:1]),
                        root_target=root,wheel_target=wheel_target,wheel_center=centers,
                        wheel_heading_deg=headings,wheel_camber_deg=np.degrees(np.arcsin(np.clip(axles[...,2],-1,1))),
                        wheel_bottom_proxy_m=bottoms,body_heading_deg=body_heading,
                        desired_wheel_heading_deg=desired_heading,wheel_heading_error_axial_deg=heading_error,
                        wheel_axle_error_axial_deg=axle_error,
                        body_speed_m_s=speed,course_minus_wheel_heading_deg=wrap(course[:,None]-headings),
                        reference_course_minus_foot_heading_deg=wrap(reference_course[:,None]-desired_heading),
                        wheel_center_course_minus_heading_deg=wrap(center_course-headings),
                        wheel_center_speed_m_s=np.linalg.norm(center_velocity,axis=-1),
                        near_floor_geometry_proxy=(bottoms<=.03)&(bottoms>=-.02),
                        beta_valid=speed>=.25,metric_world_trajectory_verified=False)
    report={"status":"kinematic_preview_only_not_admitted_for_training","source_sha256":sha(args.world),
            "runner_sha256":sha(__file__),"model_sha256":sha(args.model),"gmr_commit":settings["gmr_commit"],
            "model_local_path":str(args.model.resolve()),"frame_count":len(t),"robot_leg_scale":scale,
            "wheel_spin":"zero placeholder; not observed motion or dynamic control",
            "contact_and_friction_forces_available":False,"physics_run":False,
            "saturated_loaded_sliding_measured":False,
            "root_XY_RMSE_m":float(np.sqrt(np.mean(np.sum((qpos[:,:2]-root[:,:2])**2,axis=-1)))),
            "per_wheel_position_RMSE_m":np.sqrt(np.mean(np.sum((centers-wheel_target)**2,axis=-1),axis=0)).tolist(),
            "per_wheel_heading_axial_RMSE_deg":np.sqrt(np.mean(heading_error**2,axis=0)).tolist(),
            "per_wheel_axle_axial_RMSE_deg":np.sqrt(np.mean(axle_error**2,axis=0)).tolist(),
            "body_speed_peak_robot_scale_estimate_m_s":float(speed.max()),
            "minimum_wheel_bottom_proxy_m":bottoms.min(axis=0).tolist(),
            "automatic_floor_snapping":False,"wheel_spin_axis_orientation_cost":0.,
            "free_base_absolute_height_limit":False}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=="__main__":main()
