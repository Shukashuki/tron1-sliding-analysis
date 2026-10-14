"""Attach bilateral human foot-heading offsets to a GMR TRON1 reference.

The output is an auxiliary target/diagnostic, not a wheel-yaw joint command:
TRON1 WF has no independent wheel-yaw DOF. Source markers must already share
the GMR time clock; --source-to-robot-yaw-deg supplies the spatial alignment.
No simulation or physical contact inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def wrap_deg(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def wheel_headings(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError("GMR qpos does not match the supplied TRON1 model")
    wheel_ids = [model.body(name).id for name in ("wheel_L_Link", "wheel_R_Link")]
    root_id = model.body("base_Link").id
    data = mujoco.MjData(model)
    heading = np.empty((len(qpos), 2))
    root_xy = np.empty((len(qpos), 2))
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        root_xy[i] = data.xpos[root_id, :2]
        for j, body_id in enumerate(wheel_ids):
            axle = data.xmat[body_id].reshape(3, 3)[:, 1]
            roll = np.cross(axle, [0.0, 0.0, 1.0])[:2]
            if np.linalg.norm(roll) < 1e-6:
                raise ValueError("Wheel rolling-plane heading is undefined")
            heading[i, j] = np.degrees(np.arctan2(roll[1], roll[0]))
    return heading, root_xy


def attach(time: np.ndarray, wheel_heading: np.ndarray, root_xy: np.ndarray,
           heel: np.ndarray, toe: np.ndarray, alignment_deg: float,
           min_speed: float = 0.8) -> dict[str, np.ndarray]:
    time = np.asarray(time, dtype=float)
    length = len(time)
    if (length < 3 or np.any(np.diff(time) <= 0)
            or wheel_heading.shape != (length, 2) or root_xy.shape != (length, 2)
            or heel.shape != (length, 2, 3) or toe.shape != (length, 2, 3)
            or not all(np.isfinite(v).all() for v in (time, wheel_heading, root_xy, heel, toe))):
        raise ValueError("GMR/marker arrays must be finite and frame-aligned")
    axis = toe[:, :, :2] - heel[:, :, :2]
    norm = np.linalg.norm(axis, axis=-1)
    if np.any(norm < 0.08):
        raise ValueError("Heel-to-toe marker axis is too short to define heading")
    foot_heading = np.degrees(np.arctan2(axis[..., 1], axis[..., 0])) + alignment_deg
    velocity = np.gradient(root_xy, time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    course = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    # This is a marker-height proxy only. It is not an observed contact flag.
    foot_low = np.minimum(heel[..., 2], toe[..., 2])
    ground_proxy = foot_low <= np.quantile(foot_low, 0.05, axis=0)[None, :] + 0.10
    offset = wrap_deg(foot_heading - wheel_heading)
    source_course_offset = wrap_deg(course[:, None] - foot_heading)
    gmr_course_offset = wrap_deg(course[:, None] - wheel_heading)
    axial_source_offset = (source_course_offset + 90.0) % 180.0 - 90.0
    valid = (speed[:, None] >= min_speed) & ground_proxy
    return {"time_s": time, "root_xy_m": root_xy, "robot_course_deg": course,
            "robot_speed_m_s": speed, "gmr_wheel_heading_deg": wheel_heading,
            "human_foot_heading_aligned_deg": wrap_deg(foot_heading),
            "human_to_gmr_wheel_heading_offset_deg": offset,
            "human_course_foot_offset_deg": source_course_offset,
            "human_course_foot_axial_offset_deg": axial_source_offset,
            "gmr_course_wheel_offset_deg": gmr_course_offset,
            "near_ground_height_proxy": ground_proxy,
            "analysis_valid_mask": valid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmr-reference", required=True, type=Path)
    parser.add_argument("--foot-markers", required=True, type=Path,
                        help="NPZ with time_s, heel_xyz_m and toe_xyz_m of shape [T,2,3]")
    parser.add_argument("--model", required=True, type=Path, help="Exact TRON1 WF MJCF used by GMR")
    parser.add_argument("--source-to-robot-yaw-deg", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.source_to_robot_yaw_deg):
        parser.error("Alignment angle must be finite")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Preserving nonempty output directory")
    gmr, markers = read(args.gmr_reference), read(args.foot_markers)
    if not np.array_equal(gmr["time_s"], markers["time_s"]):
        raise ValueError("Human markers and GMR motion require identical sampled clocks")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    heading, root_xy = wheel_headings(model, np.asarray(gmr["qpos"], dtype=float))
    result = attach(np.asarray(gmr["time_s"], dtype=float), heading, root_xy,
                    np.asarray(markers["heel_xyz_m"], dtype=float),
                    np.asarray(markers["toe_xyz_m"], dtype=float), args.source_to_robot_yaw_deg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "wheel_heading_offset.npz", **result)
    valid = result["analysis_valid_mask"]
    report = {"kind": "GMR_reference_plus_bilateral_human_heading_sidecar",
              "gmr_reference_sha256": sha(args.gmr_reference),
              "foot_markers_sha256": sha(args.foot_markers), "model_sha256": sha(args.model),
              "source_to_robot_yaw_deg": args.source_to_robot_yaw_deg,
              "valid_samples_L_R": valid.sum(axis=0).tolist(),
              "offset_p90_abs_deg_L_R": [float(np.quantile(
                  np.abs(result["human_to_gmr_wheel_heading_offset_deg"][valid[:, i], i]), 0.9))
                  if np.any(valid[:, i]) else None for i in range(2)],
              "limitations": ["No independent wheel-yaw DOF; offset is a target sidecar, not realized wheel motion.",
                              "Near-ground mask uses marker height, not contact forces.",
                              "No policy, Isaac rollout, MuJoCo rollout, or sim2sim claim."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
