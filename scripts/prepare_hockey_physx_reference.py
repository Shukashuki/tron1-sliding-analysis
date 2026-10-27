#!/usr/bin/env python3
"""Build a motion-inspired stop command from GMR posture and designed root speed.

Not a measured global human trajectory or a dynamically feasible reference.
Keep the old actor's exact initial joint offsets; expose the changed commands
as out-of-distribution inputs to a PPO-assisted physics probe, not a new policy.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation, Slerp

from hdm05_gmr import rolling_geometry, sha


def stopping_speed(time, entry_speed, brake_start, brake_end):
    if not (entry_speed > 0 and 0 <= brake_start < brake_end):
        raise ValueError("Invalid stopping profile")
    u = np.clip((np.asarray(time)-brake_start)/(brake_end-brake_start), 0, 1)
    return entry_speed*(1-u*u*(3-2*u))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retarget", type=Path, required=True)
    p.add_argument("--actor-reference", type=Path, required=True)
    p.add_argument("--seed-trace", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    g=np.load(args.retarget/"retarget_trace.npz",allow_pickle=False)
    actor=np.load(args.actor_reference,allow_pickle=False)
    seed=np.load(args.seed_trace,allow_pickle=False)
    model=mujoco.MjModel.from_xml_path(str(args.model.resolve()));data=mujoco.MjData(model)
    names=actor["joint_names"].astype(str).tolist();bodies=actor["body_names"].astype(str).tolist()
    qids=[int(model.jnt_qposadr[model.joint(k).id]) for k in names]
    # The Isaac reference has a separate coincident IMU link; this MJCF uses a
    # base-attached site. Verify the locked reference before aliasing that link.
    if "limx_imu" in bodies:
        base,imu=bodies.index("base_Link"),bodies.index("limx_imu")
        np.testing.assert_allclose(actor["body_pos_w"][:,imu],actor["body_pos_w"][:,base],atol=1e-7)
        delta=Rotation.from_quat(actor["body_quat_w"][:,imu],scalar_first=True)*Rotation.from_quat(actor["body_quat_w"][:,base],scalar_first=True).inv()
        np.testing.assert_allclose(delta.magnitude(),0,atol=1e-7)
    bids=[model.body("base_Link" if k=="limx_imu" else k).id for k in bodies]
    t=np.arange(139)*.02  # 2.76 s, including an explicitly extended stop hold.
    source_t=g["time_s"];sample_t=np.minimum(t,source_t[-1])
    q=np.zeros((len(t),model.nq));q[:,3]=1
    blend_u=np.clip(t/.3,0,1);blend=blend_u**3*(10+blend_u*(-15+6*blend_u))
    for k,qid in enumerate(qids):
        pose=np.interp(sample_t,source_t,g["qpos"][:,qid])
        q[:,qid]=actor["joint_pos"][0,k]+blend*(pose-actor["joint_pos"][0,k])
        if names[k].startswith("wheel_"):q[:,qid]=actor["joint_pos"][0,k]
    rotation=Slerp(source_t,Rotation.from_quat(g["qpos"][:,3:7],scalar_first=True))(sample_t)
    q[:,3:7]=Rotation.from_rotvec(rotation.as_rotvec()*blend[:,None]).as_quat(scalar_first=True)
    entry_vector=seed["root_linear_velocity_w_m_s"][0,0,:2].astype(float)
    entry_speed=float(np.linalg.norm(entry_vector));direction=entry_vector/entry_speed
    speed=stopping_speed(t,entry_speed,.4,1.8)
    distance=cumulative_trapezoid(speed,t,initial=0)
    q[:,:2]=seed["root_position_m"][0,0,:2]+distance[:,None]*direction
    positions=[];quats=[];coms=[];bottoms=[];headings=[]
    for frame in q:
        # One-wheel support projection is a command-construction assumption.
        # No pose writes inside the later physical rollout are permitted.
        frame[2]=0.;data.qpos[:]=frame;mujoco.mj_forward(model,data)
        frame[2]-=rolling_geometry(model,data)[1].min()
        data.qpos[:]=frame;mujoco.mj_forward(model,data)
        positions.append(data.xpos[bids].copy());quats.append(data.xquat[bids].copy())
        coms.append(data.xipos[bids].copy())
        h,b=rolling_geometry(model,data);headings.append(h);bottoms.append(b)
    positions,quats,coms=map(np.asarray,(positions,quats,coms))
    link_velocity=np.gradient(positions,t,axis=0)
    omega=np.zeros_like(positions)
    for j in range(len(bodies)):
        R=Rotation.from_quat(quats[:,j],scalar_first=True)
        w=(R[1:]*R[:-1].inv()).as_rotvec()/.02
        omega[0,j]=w[0];omega[-1,j]=w[-1];omega[1:-1,j]=(w[1:]+w[:-1])/2
    com_velocity=link_velocity+np.cross(omega,coms-positions)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"motion.npz",fps=50.,joint_pos=q[:,qids],joint_vel=np.gradient(q[:,qids],t,axis=0),
        body_pos_w=positions,body_quat_w=quats,body_lin_vel_w=com_velocity,body_ang_vel_w=omega,
        joint_names=np.array(names),body_names=np.array(bodies))
    np.testing.assert_array_equal(q[0,qids],actor["joint_pos"][0])
    np.savez_compressed(args.output/"command_diagnostics.npz",time_s=t,qpos=q,
        designed_root_link_speed_m_s=speed,root_link_velocity_w_m_s=link_velocity[:,0],
        wheel_bottom_proxy_m=bottoms,wheel_heading_deg=headings,source_sample_time_s=sample_t,
        posture_blend=blend)
    report=dict(kind="human_pose_inspired_designed_hockey_stop_command_v1",
        runner_sha256=sha(__file__),retarget_sha256=sha(args.retarget/"retarget_trace.npz"),
        actor_reference_sha256=sha(args.actor_reference),seed_trace_sha256=sha(args.seed_trace),
        model_sha256=sha(args.model),motion_sha256=sha(args.output/"motion.npz"),
        coincident_imu_link_verified_in_actor_reference=True,
        entry_speed_m_s=entry_speed,entry_speed_origin="existing PhysX seed COM planar speed; designed command, not measured human speed",
        brake_start_s=.4,brake_end_s=1.8,stop_hold_s=float(t[-1]-1.8),
        designed_root_link_stop_distance_m=float(distance[-1]),
        source_pose_timing_unchanged_until_s=float(source_t[-1]),
        initial_posture_blend_s=.3,actor_initial_joint_offsets_preserved_exactly=True,
        source_world_XY_used=False,one_wheel_ground_projection=True,
        root_height_is_geometry_derived_not_fixed=True,
        wheel_spin_reference="zero placeholder, excluded from actor reference command",
        body_linear_velocity_semantics="link velocity plus omega cross COM offset",
        existing_actor_has_explicit_velocity_or_braking_command=False,
        controller_label="existing car PPO used out of distribution as stabilizer; MPPI must evaluate complete dynamic rollouts",
        physics_run=False,admitted_for_training=False,task_success_claimed=False)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
