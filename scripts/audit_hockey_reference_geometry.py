#!/usr/bin/env python3
"""Audit GVHMR root motion and anatomical foot frames before dynamic imitation.

Camera-rotation compensation is an explicit alternative with a fixed camera
center assumption. It is not metric calibration or a replacement for observed
camera translation. Heel/toe surface landmarks are not measured skate blades.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from reconstruct_hockey_stop_gvhmr import ROOT, GVHMR, sha


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate foot geometry")
    return x / n


def sole_frames(points, lateral_sign):
    """Points are (..., left/right, big toe/small toe/heel, xyz)."""
    forward = unit(points[..., :2, :].mean(-2) - points[..., 2, :])
    lateral = (points[..., 1, :] - points[..., 0, :]) * np.asarray(lateral_sign)[..., None]
    lateral = unit(lateral - (forward * lateral).sum(-1, keepdims=True) * forward)
    normal = unit(np.cross(forward, lateral))
    return np.stack((forward, lateral, normal), axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--retarget", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(GVHMR))
    import torch
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.body_model.smplx_lite import SmplxLite, batch_rigid_transform_v2
    from smplx.vertex_ids import vertex_ids
    from pytorch3d.transforms import axis_angle_to_matrix
    torch.set_num_threads(4)
    torch.manual_seed(0)
    old = np.load(args.run / "world_raw.npz")
    time = old["time_s"]
    n = len(time)
    retarget = np.load(args.retarget / "retarget_trace.npz")
    retarget_report = json.loads((args.retarget / "report.json").read_text())
    scale = retarget_report["robot_leg_scale"]
    boxes = torch.load(args.run / "bbx.pt", map_location="cpu")["bbx_xys"]
    kpts = torch.load(args.run / "vitpose.pt", map_location="cpu")
    R_w2c = torch.from_numpy(torch.load(args.run / "simplevo.pt", map_location="cpu")[:, :3, :3]).float()
    K = torch.load(args.run / "hmr4d_raw.pt", map_location="cpu")["K_fullimg"]
    batch = dict(length=torch.tensor([n]), obs=normalize_kp2d(kpts, boxes)[None],
                 bbx_xys=boxes[None], K_fullimg=K[None], cam_angvel=compute_cam_angvel(R_w2c)[None],
                 f_imgseq=torch.load(args.run / "vit_features.pt", map_location="cpu")[None])
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=hockey_geometry_audit"])
    with torch.inference_mode():
        model = instantiate(cfg.model, _recursive_=False).cpu().eval()
        model.load_pretrained_model(args.checkpoint)
        result = model.pipeline.forward(batch, train=False, postproc=False, static_cam=False)
        raw = {k: v[0].cpu() for k, v in result["pred_smpl_params_global"].items()}
        incam = {k: v[0].cpu() for k, v in result["pred_smpl_params_incam"].items()}
        for k, v in raw.items():
            np.testing.assert_allclose(v.numpy(), old[k], atol=1e-6, rtol=1e-5)
        body = SmplxLite()
        ids = np.asarray([[vertex_ids["smplx"][s + k] for k in ("BigToe", "SmallToe", "Heel")] for s in ("L", "R")])

        def surface(params):
            parts = []
            for start in range(0, len(params["transl"]), 8):
                v = body(**{k: x[start:start + 8] for k, x in params.items()})
                parts.append(v[:, ids.ravel()].numpy().reshape(-1, 2, 3, 3))
            return np.concatenate(parts)

        foot = surface(raw)
        neutral = {k: v[:1].clone() for k, v in raw.items()}
        for k in ("body_pose", "global_orient", "transl"):
            neutral[k].zero_()
        neutral_foot = surface(neutral)
        trial = sole_frames(neutral_foot, [1, 1])
        sign = np.where(trial[0, :, 1, 2] > 0, 1, -1)
        frames = sole_frames(foot, sign)
        neutral_frames = sole_frames(neutral_foot, sign)
        # Source global axes are Y-up. Align the first camera frame to the
        # original global orientation, and compensate only observed rotation.
        Rg = Rotation.from_rotvec(raw["global_orient"][0].numpy()).as_matrix()
        Rc = Rotation.from_rotvec(incam["global_orient"].numpy()).as_matrix()
        initial = Rg @ Rc[0].T
        world_from_camera = initial @ R_w2c.numpy().transpose(0, 2, 1)
        translated = np.einsum("tij,tj->ti", world_from_camera, incam["transl"].numpy())
        translated += raw["transl"][0].numpy() - translated[0]
        alt = {k: v.clone() for k, v in raw.items()}
        alt["transl"] = torch.from_numpy(translated).float()
        alt["global_orient"] = torch.from_numpy(Rotation.from_matrix(world_from_camera @ Rc).as_rotvec()).float()
        pose = torch.cat((alt["global_orient"], alt["body_pose"], body.other_default_pose.expand(n, -1)), dim=-1)
        alt_joints, _ = batch_rigid_transform_v2(axis_angle_to_matrix(pose.reshape(n, 55, 3)), body.get_skeleton(alt["betas"]), body.parents)
        alt_joints = (alt_joints + alt["transl"][:, None]).numpy()
        alt_foot = surface(alt)
    A = np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    C = np.array([[0,1,0],[0,0,1],[1,0,0.]])
    root_R0 = A @ Rg @ C
    align = Rotation.from_rotvec([0,0,-np.arctan2(root_R0[1,0], root_R0[0,0])]).as_matrix() @ A
    sole = align @ frames
    sole_heading = np.degrees(np.arctan2(sole[..., 1, 0], sole[..., 0, 0]))
    axle_heading = np.degrees(np.arctan2(-sole[..., 0, 1], sole[..., 1, 1]))
    axial = lambda x: (x+90)%180-90
    raw_root = old["joints_world_estimated_m"][:,0] @ align.T * scale
    alt_root = alt_joints[:,0] @ align.T * scale
    raw_v = np.gradient(raw_root[:,:2], time, axis=0)
    alt_v = np.gradient(alt_root[:,:2], time, axis=0)
    raw_speed = np.linalg.norm(raw_v,axis=-1)
    alt_speed = np.linalg.norm(alt_v,axis=-1)
    course = np.degrees(np.arctan2(raw_v[:,1],raw_v[:,0]))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"foot_geometry.npz",time_s=time,sole_points_yup_m=foot,
                        sole_frame_yup=frames,sole_frame_neutral_yup=neutral_frames,
                        sole_frame_zup=sole,sole_heading_deg=sole_heading,
                        desired_axle_heading_deg=axle_heading,
                        beta_ref_per_foot_deg=axial(course[:,None]-axle_heading),
                        beta_valid=raw_speed>=.25,vertex_ids=ids,
                        raw_root_zup_robot_scale_m=raw_root,alternative_root_zup_robot_scale_m=alt_root,
                        raw_speed_robot_scale_m_s=raw_speed,alternative_speed_robot_scale_m_s=alt_speed,
                        alternative_sole_points_yup_m=alt_foot)
    np.savez_compressed(args.output/"world_rotation_compensated_camera.npz",time_s=time,
                        joints_world_estimated_m=alt_joints,postproc=False,metric_world_trajectory_verified=False,
                        **{k:v.numpy() for k,v in alt.items()})
    report={"source_sha256":sha(args.run/"world_raw.npz"),"script_sha256":sha(__file__),
            "raw_reinference_matches_frozen_reference":True,
            "sole_definition":"heel to midpoint(big toe, small toe); lateral orthogonalized; normal sign fixed from neutral shape",
            "sole_is_measured_blade":False,"camera_alternative_assumption":"camera center fixed; SimpleVO rotation and model depth only; not externally calibrated",
            "raw_peak_speed_robot_scale_m_s":float(raw_speed.max()),
            "alternative_peak_speed_robot_scale_m_s":float(alt_speed.max()),
            "raw_path_length_robot_scale_m":float(np.linalg.norm(np.diff(raw_root[:,:2],axis=0),axis=-1).sum()),
            "alternative_path_length_robot_scale_m":float(np.linalg.norm(np.diff(alt_root[:,:2],axis=0),axis=-1).sum()),
            "ankle_axis_vs_sole_axle_heading_RMSE_deg":np.sqrt(np.mean(axial(retarget["desired_wheel_heading_deg"]-axle_heading)**2,axis=0)).tolist(),
            "sole_left_right_heading_separation_p50_p95_max_deg":np.quantile(abs(axial(axle_heading[:,0]-axle_heading[:,1])),[.5,.95,1]).tolist(),
            "metric_world_trajectory_verified":False,"physics_run":False,"admitted_for_training":False}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
