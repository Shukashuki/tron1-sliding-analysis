"""Recompute bilateral angle/contact-proxy metrics from saved GMR traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from hdm05_gmr import heading_features, rolling_geometry, sha, summarize_features, summary, wrap


def compare(expected, actual, context="metrics"):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(context + ": key mismatch")
        for key in expected:
            compare(expected[key], actual[key], context + "." + key)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(context + ": list length mismatch")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, context + f"[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, err_msg=context)
    elif expected != actual:
        raise ValueError(context + ": value mismatch")


def audit(directory):
    experiment = json.loads((directory / "experiment.json").read_text())
    settings = experiment["settings"]
    rows = []
    for report in experiment["clips"]:
        path = directory / Path(report["clip"]).stem / "retarget_trace.npz"
        if sha(path) != report["trace_sha256"]:
            raise ValueError("Trace SHA-256 differs from recorded result")
        with np.load(path, allow_pickle=False) as data:
            trace = {key: data[key].copy() for key in data.files}
        names = list(trace["human_joint_names"])
        feet = [names.index(name) for name in settings["bvh_ankle_nodes"]]
        toes = [names.index(name) for name in settings["bvh_toe_base_nodes"]]
        recomputed = heading_features(trace["time_s"], trace["root_target_m"], trace["qpos"][:, :3],
                                      trace["human_positions_robot_scale_m"][:, feet],
                                      trace["human_positions_robot_scale_m"][:, toes],
                                      trace["robot_wheel_heading_deg"], trace["robot_wheel_bottom_m"], settings,
                                      wheel_centers=trace["robot_site_positions_m"][:, 5:7])
        for key, value in recomputed.items():
            np.testing.assert_allclose(trace[key], value, rtol=1e-10, atol=1e-10, err_msg=key)
        compare(report["angle_analysis"], summarize_features(trace["time_s"], recomputed, settings))
        compare(report["wheel_center_error_m"], summary(np.linalg.norm(
            trace["robot_site_positions_m"][:, 5:7] - trace["wheel_target_m"], axis=-1)))
        compare(report["root_xy_error_m"], summary(np.linalg.norm(
            trace["qpos"][:, :2] - trace["root_target_m"][:, :2], axis=-1)))
        rows.append({"clip": report["clip"], "status": "trace_metrics_reproduced",
                     "paired_slow_samples_L_R": report["angle_analysis"]["slow_sensitivity"]["paired_samples_L_R"]})
    return rows


def geometry_audit(directory, model_path):
    experiment = json.loads((directory / "experiment.json").read_text())
    if sha(model_path) != experiment["model_sha256"]:
        raise ValueError("Geometry model differs from GMR model")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    wheel_geoms = {model.geom("wheel_" + side + "_collision").id for side in ("L", "R")}
    rows = []
    for report in experiment["clips"]:
        trace_path = directory / Path(report["clip"]).stem / "retarget_trace.npz"
        if sha(trace_path) != report["trace_sha256"]:
            raise ValueError("Geometry trace hash mismatch")
        trace = np.load(trace_path, allow_pickle=False)
        max_self = max_other_floor = 0.
        self_frames = 0
        for frame, pose in enumerate(trace["qpos"]):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            heading, bottom = rolling_geometry(model, data)
            np.testing.assert_allclose(wrap(heading - trace["robot_wheel_heading_deg"][frame]), 0, atol=1e-9)
            np.testing.assert_allclose(bottom, trace["robot_wheel_bottom_m"][frame], atol=1e-9)
            frame_self = 0.
            for contact in data.contact:
                if contact.dist >= 0:
                    continue
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1, b2 = model.geom_bodyid[[g1, g2]]
                if b1 and b2:
                    frame_self = max(frame_self, -float(contact.dist))
                elif (g2 if b1 == 0 else g1) not in wheel_geoms:
                    max_other_floor = max(max_other_floor, -float(contact.dist))
            max_self = max(max_self, frame_self)
            self_frames += frame_self > .001
        trace.close()
        rows.append({"clip": report["clip"], "wheel_heading_and_bottom_reproduced_from_qpos": True,
                     "max_self_penetration_m": max_self,
                     "frames_with_self_penetration_gt_1mm": self_frames,
                     "max_nonwheel_ground_penetration_m": max_other_floor})
    return {"kind": "kinematic_geometry_replay_audit", "model_sha256": sha(model_path),
            "experiment_sha256": sha(directory / "experiment.json"), "clips": rows,
            "dynamics_stepped": False, "contact_forces_used": False,
            "limitations": "Collision distances are kinematic diagnostics with this model's collision filters, not dynamic feasibility or an XY/height failure gate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Optional licensed model for independent geometry replay")
    parser.add_argument("--geometry-output", type=Path, help="Write a new geometry audit; requires --model")
    args = parser.parse_args()
    print(json.dumps(audit(args.experiment_dir), indent=2))
    if args.geometry_output:
        if args.model is None:
            parser.error("--geometry-output requires --model")
        result = geometry_audit(args.experiment_dir, args.model)
        with args.geometry_output.open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
