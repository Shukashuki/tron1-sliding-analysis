"""Real pinned GMR retargeting and bilateral near-ground heading diagnostics.

No standalone IK baseline, mj_step, contact forces, policy training or sim2sim.
The source BVH is an explicitly labeled human-skeleton-standardized derivative.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(angle):
    return (np.asarray(angle) + 180) % 360 - 180


def axial(angle):
    return (np.asarray(angle) + 90) % 180 - 90


def summary(value):
    value = np.asarray(value)
    return {"rmse": float(np.sqrt(np.mean(value ** 2))),
            "p95_abs": float(np.quantile(np.abs(value), .95)),
            "max_abs": float(np.max(np.abs(value)))}


def load_gmr(settings):
    repository = ROOT / "upstream/GMR"
    source = repository / "general_motion_retargeting"
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != settings["gmr_commit"]:
        raise ValueError("GMR commit does not match experiment lock")
    for filename, expected in settings["source_locks"].items():
        if sha(source / filename) != expected:
            raise ValueError("Modified upstream source: " + filename)
    package_name = "_hdm05_pinned_gmr"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    params = importlib.import_module(package_name + ".params")
    core = importlib.import_module(package_name + ".motion_retarget")
    extract = importlib.import_module(package_name + ".utils.lafan_vendor.extract")
    utils = importlib.import_module(package_name + ".utils.lafan_vendor.utils")

    class MinkAPI:
        def __getattr__(self, name):
            return getattr(mink, name)

        @staticmethod
        def solve_ik(configuration, tasks, dt, solver, damping, limits):
            return mink.solve_ik(configuration, tasks, dt, solver,
                                 damping=damping, limits=limits, safety_break=True,
                                 primal_tol=1e-10, dual_tol=1e-10)

    if importlib.metadata.version("mink") != "1.2.0" or "safety_break" not in inspect.signature(mink.solve_ik).parameters:
        raise ValueError("This audited GMR compatibility adapter requires Mink 1.2.0")
    # Only GMR's local Mink reference changes; the upstream algorithm is unmodified.
    core.mink = MinkAPI()
    return core.GeneralMotionRetargeting, params, extract, utils


def read_source(path, extract, utils, settings):
    text = path.read_text()
    dt = float(next(line.split(":", 1)[1] for line in text.splitlines() if line.startswith("Frame Time:")))
    frame_count = int(next(line.split(":", 1)[1] for line in text.splitlines() if line.startswith("Frames:")))
    # This pinned vendor reader uses one Euler order for all joints. Reject
    # any source outside that supported subset rather than silently misreading it.
    orders = []
    channel_count = 0
    for line in text.splitlines():
        if line.strip().startswith("CHANNELS "):
            channels = line.split()[2:]
            channel_count += len(channels)
            orders.append([channel for channel in channels if channel.endswith("rotation")])
    if any(order != orders[0] for order in orders) or len(orders[0]) != 3:
        raise ValueError("BVH has mixed or unsupported joint Euler orders")
    anim = extract.read_bvh(str(path))
    quats, positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    if len(positions) != frame_count or not np.isfinite(positions).all() or dt <= 0:
        raise ValueError("Invalid BVH frames")
    np.testing.assert_allclose(positions[:, 0], anim.pos[:, 0], atol=1e-12)
    # Independently validate the vendor's Euler composition against SciPy.
    first_frame = text.split("Frame Time:", 1)[1].splitlines()[1]
    root_values = np.array(list(map(float, first_frame.split())))
    order = "".join(channel[0] for channel in orders[0])  # intrinsic declared BVH order
    expected = Rotation.from_euler(order, root_values[3:6], degrees=True).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(quats[0, 0, [1, 2, 3, 0]]).as_matrix(), expected, atol=1e-10)
    index = {name: i for i, name in enumerate(anim.bones)}
    for side in range(2):
        hip, knee, ankle = [index[settings[key][side]] for key in
                            ("bvh_hip_nodes", "bvh_knee_nodes", "bvh_ankle_nodes")]
        if anim.parents[knee] != hip or anim.parents[ankle] != knee:
            raise ValueError("Unexpected BVH hip/knee/ankle pivot semantics")
    conversion = np.asarray(settings["source_to_z_up"], float)
    np.testing.assert_allclose(conversion @ conversion.T, np.eye(3))
    if np.linalg.det(conversion) < .999:
        raise ValueError("Source conversion must be a proper rotation")
    source_rotation = Rotation.from_quat(quats[:, 0, [1, 2, 3, 0]]).as_matrix()
    first_forward = conversion @ source_rotation[0] @ np.array([0., 0., 1.])
    yaw = np.arctan2(first_forward[1], first_forward[0])
    alignment = Rotation.from_euler("z", -yaw).as_matrix()
    positions = positions @ (alignment @ conversion).T
    # Remove a constant initial mounting rotation, not subsequent motion.
    root_rotation = alignment @ conversion @ source_rotation @ source_rotation[0].T @ conversion.T @ alignment.T
    root_quat = Rotation.from_matrix(root_rotation).as_quat(scalar_first=True)
    return {"time_s": np.arange(frame_count) * dt, "dt": dt,
            "positions_native": positions, "root_quat_wxyz": root_quat,
            "index": index, "joint_names": np.array(anim.bones),
            "parents": anim.parents, "heading_alignment_rad": float(-yaw),
            "bvh_channel_count": channel_count}


def transport_targets(ankles, human_root, robot_ankles0, robot_root0, scale, rotation):
    root = robot_root0 + scale * (human_root - human_root[0])
    offset = robot_ankles0 - robot_root0 - scale * (ankles[0] - human_root[0])
    relative = rotation @ rotation[0].T
    targets = root[:, None] + scale * (ankles - human_root[:, None]) + np.einsum("tij,kj->tki", relative, offset)
    return root, targets


def rolling_geometry(model, data):
    headings, bottoms = [], []
    for side in ("L", "R"):
        body = model.body("wheel_" + side + "_Link").id
        geom = model.geom("wheel_" + side + "_collision").id
        joint = model.joint("wheel_" + side + "_Joint").id
        axle = data.xmat[body].reshape(3, 3) @ model.jnt_axis[joint]
        roll = np.cross(axle, [0, 0, 1])[:2]
        if np.linalg.norm(roll) < 1e-8:
            raise ValueError("Wheel rolling-plane heading is undefined")
        headings.append(np.degrees(np.arctan2(roll[1], roll[0])))
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("Expected cylindrical wheel collision geometry")
        cylinder_axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        radius, half_width = model.geom_size[geom, :2]
        extent = radius * np.sqrt(max(0., 1 - cylinder_axis[2] ** 2)) + half_width * abs(cylinder_axis[2])
        bottoms.append(data.geom_xpos[geom, 2] - extent)
    return np.asarray(headings), np.asarray(bottoms)


def heading_features(times, human_root, robot_root, ankles, toes, wheel_heading, wheel_bottom, settings,
                     wheel_centers=None):
    velocity_h = np.gradient(human_root[:, :2], times, axis=0)
    velocity_r = np.gradient(robot_root[:, :2], times, axis=0)
    speed_h, speed_r = [np.linalg.norm(value, axis=1) for value in (velocity_h, velocity_r)]
    course_h, course_r = [np.degrees(np.arctan2(value[:, 1], value[:, 0])) for value in (velocity_h, velocity_r)]
    foot_axis = toes[:, :, :2] - ankles[:, :, :2]
    axis_valid = np.linalg.norm(foot_axis, axis=-1) >= .01
    foot_heading = np.degrees(np.arctan2(foot_axis[..., 1], foot_axis[..., 0]))
    human_offset = wrap(course_h[:, None] - foot_heading)
    robot_offset = wrap(course_r[:, None] - wheel_heading)
    human_low = np.minimum(ankles[..., 2], toes[..., 2])
    # One shared constant floor proxy: a persistently raised foot must not get
    # its own elevated "ground" merely because its personal minimum is high.
    human_floor_proxy = np.quantile(human_low, .05)
    human_ground = human_low <= human_floor_proxy + settings["human_near_ground_height_m"]
    robot_ground = ((wheel_bottom <= settings["robot_near_ground_height_m"])
                    & (wheel_bottom >= -settings["robot_max_contact_proxy_penetration_m"]))
    foot_velocity = np.gradient(ankles[:, :, :2], times, axis=0)
    foot_speed = np.linalg.norm(foot_velocity, axis=-1)
    foot_course = np.degrees(np.arctan2(foot_velocity[..., 1], foot_velocity[..., 0]))
    features = {"human_root_speed_robot_scale_m_s": speed_h, "robot_root_speed_m_s": speed_r,
                "human_root_course_deg": course_h, "robot_root_course_deg": course_r,
                "human_foot_heading_deg": foot_heading, "robot_wheel_heading_deg": wheel_heading,
                "human_course_foot_offset_deg": human_offset, "human_course_foot_axial_deg": axial(human_offset),
                "robot_course_wheel_offset_deg": robot_offset, "robot_course_wheel_axial_deg": axial(robot_offset),
                "human_to_robot_heading_sidecar_deg": wrap(foot_heading - wheel_heading),
                "human_axis_valid": axis_valid, "human_near_ground_proxy": human_ground,
                "robot_near_ground_proxy": robot_ground, "human_foot_low_robot_scale_m": human_low,
                "robot_wheel_bottom_m": wheel_bottom, "human_floor_proxy_robot_scale_m": human_floor_proxy,
                "human_foot_velocity_offset_deg": wrap(foot_course - foot_heading),
                "human_foot_velocity_valid": (foot_speed >= .15) & axis_valid & human_ground,
                "human_foot_speed_robot_scale_m_s": foot_speed}
    if wheel_centers is not None:
        wheel_velocity = np.gradient(wheel_centers[:, :, :2], times, axis=0)
        wheel_speed = np.linalg.norm(wheel_velocity, axis=-1)
        wheel_course = np.degrees(np.arctan2(wheel_velocity[..., 1], wheel_velocity[..., 0]))
        features["robot_wheel_center_velocity_offset_deg"] = wrap(wheel_course - wheel_heading)
        features["robot_wheel_center_speed_m_s"] = wheel_speed
        features["robot_wheel_center_velocity_valid"] = (wheel_speed >= .15) & robot_ground
    for name, threshold in (("common", settings["min_speed_m_s"]),
                            ("slow_sensitivity", settings["slow_speed_sensitivity_m_s"])):
        features["human_valid_" + name] = axis_valid & human_ground & (speed_h[:, None] >= threshold)
        features["robot_valid_" + name] = robot_ground & (speed_r[:, None] >= threshold)
        features["paired_valid_" + name] = features["human_valid_" + name] & features["robot_valid_" + name]
    return features


def events(times, mask, angle, settings):
    high = mask & (np.abs(angle) >= settings["high_axial_offset_deg"])
    boundaries = np.flatnonzero(np.diff(np.r_[False, high, False]))
    result = []
    for start, stop in boundaries.reshape(-1, 2):
        duration = float(times[stop - 1] - times[start])
        if duration >= settings["min_high_offset_event_duration_s"]:
            result.append({"start_s": float(times[start]), "end_s": float(times[stop - 1]),
                           "duration_s": duration, "samples": int(stop - start),
                           "mean_abs_axial_deg": float(np.abs(angle[start:stop]).mean()),
                           "peak_abs_axial_deg": float(np.abs(angle[start:stop]).max())})
    return result


def summarize_features(times, features, settings):
    result = {}
    for threshold in ("common", "slow_sensitivity"):
        subsets = {}
        for kind, key in (("human", "human_course_foot_axial_deg"),
                          ("robot", "robot_course_wheel_axial_deg")):
            per_side = []
            for side in range(2):
                mask = features[kind + "_valid_" + threshold][:, side]
                angle = features[key][:, side]
                per_side.append({"samples": int(mask.sum()), "fraction_of_frames": float(mask.mean()),
                                 "mean_abs_axial_deg": float(np.abs(angle[mask]).mean()) if mask.any() else None,
                                 "p90_abs_axial_deg": float(np.quantile(np.abs(angle[mask]), .9)) if mask.any() else None,
                                 "high_offset_fraction_of_valid": float((np.abs(angle[mask]) >= settings["high_axial_offset_deg"]).mean()) if mask.any() else None,
                                 "high_offset_events": events(times, mask, angle, settings)})
            subsets[kind + "_L_R"] = per_side
        subsets["paired_samples_L_R"] = features["paired_valid_" + threshold].sum(axis=0).tolist()
        subsets["paired_mean_abs_axial_L_R"] = []
        for side in range(2):
            mask = features["paired_valid_" + threshold][:, side]
            subsets["paired_mean_abs_axial_L_R"].append({
                key: float(np.abs(value[:, side][mask]).mean()) if mask.any() else None
                for key, value in {
                    "human_deg": features["human_course_foot_axial_deg"],
                    "robot_deg": features["robot_course_wheel_axial_deg"],
                    "human_to_robot_heading_sidecar_deg": axial(features["human_to_robot_heading_sidecar_deg"])
                }.items()})
        result[threshold] = subsets
    return result


def run_clip(path, output, model_path, settings, GMR, params, extract, utils):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserving existing clip result: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    source = read_source(path, extract, utils, settings)
    human = source["positions_native"]
    index = source["index"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.nu) != (15, 14, 8):
        raise ValueError("Expected TRON1 WF free base and eight joint actuators")
    seed = model.qpos0.copy()
    seed[:7] = [0, 0, 0, 1, 0, 0, 0]
    data = mujoco.MjData(model)
    data.qpos[:] = seed
    mujoco.mj_forward(model, data)
    seed[2] -= float(np.mean(rolling_geometry(model, data)[1]))
    data.qpos[:] = seed
    mujoco.mj_forward(model, data)
    site_names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_wheel", "right_wheel"]
    site_ids = [model.site(name).id for name in site_names]
    wheels0 = data.site_xpos[[model.site(name).id for name in ("left_wheel", "right_wheel")]].copy()
    robot_lengths, source_lengths = [], []
    for side, side_index in (("left", 0), ("right", 1)):
        robot_chain = data.site_xpos[[model.site(side + "_" + part).id for part in ("hip", "knee", "wheel")]]
        robot_lengths.append(np.linalg.norm(np.diff(robot_chain, axis=0), axis=1).sum())
        human_chain = human[:, [index[settings[key][side_index]] for key in ("bvh_hip_nodes", "bvh_knee_nodes", "bvh_ankle_nodes")]]
        lengths = np.linalg.norm(np.diff(human_chain, axis=1), axis=-1).sum(axis=1)
        np.testing.assert_allclose(lengths, lengths[0], atol=1e-8)
        source_lengths.append(lengths[0])
    scale = float(np.mean(robot_lengths) / np.mean(source_lengths))
    ankles_native = human[:, [index[name] for name in settings["bvh_ankle_nodes"]]]
    root_quat = source["root_quat_wxyz"]
    rotations = Rotation.from_quat(root_quat, scalar_first=True).as_matrix()
    root_targets, wheel_targets = transport_targets(ankles_native, human[:, 0], wheels0, seed[:3], scale, rotations)
    human_display = seed[None, None, :3] + scale * (human - human[:1, :1])
    match1 = {"base_Link": ["Pelvis", 0, 10, [0, 0, 0], [1, 0, 0, 0]],
              "wheel_L_Link": ["LeftAnkle", 50, 0, [0, 0, 0], [1, 0, 0, 0]],
              "wheel_R_Link": ["RightAnkle", 50, 0, [0, 0, 0], [1, 0, 0, 0]]}
    match2 = {key: [value[0], 10 if key == "base_Link" else 100, value[2], value[3], value[4]]
              for key, value in match1.items()}
    gmr_config = {"robot_root_name": "base_Link", "human_root_name": "Pelvis", "ground_height": 0,
                  "human_height_assumption": 1.8, "use_ik_match_table1": True, "use_ik_match_table2": True,
                  "human_scale_table": {name: 1.0 for name in ("Pelvis", "LeftAnkle", "RightAnkle")},
                  "ik_match_table1": match1, "ik_match_table2": match2}
    config_path = output / "gmr_task.json"
    config_path.write_text(json.dumps(gmr_config, indent=2) + "\n")
    params.ROBOT_XML_DICT["tron1_wf_hdm05"] = model_path
    params.IK_CONFIG_DICT["hdm05_anchored"] = {"tron1_wf_hdm05": config_path}
    retargeter = GMR("hdm05_anchored", "tron1_wf_hdm05", solver=settings["solver"],
                     damping=settings["damping"], verbose=False, use_velocity_limit=False)
    retargeter.max_iter = settings["max_iter_per_stage"]
    retargeter.model.opt.timestep = source["dt"]
    retargeter.configuration.update(seed)
    retargeter.set_ground_offset(0.)
    retargeter.ik_limits.append(mink.VelocityLimit(retargeter.model, {"wheel_L_Joint": 0., "wheel_R_Joint": 0.}))
    n = len(human)
    qpos, sites, headings, bottoms, errors = [np.empty(shape) for shape in
        ((n, model.nq), (n, len(site_ids), 3), (n, 2), (n, 2), (n, 2))]
    start = time.perf_counter()
    for frame in range(n):
        targets = {"Pelvis": [root_targets[frame].copy(), root_quat[frame].copy()],
                   "LeftAnkle": [wheel_targets[frame, 0].copy(), np.array([1., 0., 0., 0.])],
                   "RightAnkle": [wheel_targets[frame, 1].copy(), np.array([1., 0., 0., 0.])]}
        qpos[frame] = retargeter.retarget(targets, offset_to_ground=False)
        for name, expected in (("Pelvis", root_targets[frame]), ("LeftAnkle", wheel_targets[frame, 0]), ("RightAnkle", wheel_targets[frame, 1])):
            np.testing.assert_allclose(retargeter.scaled_human_data[name][0], expected, atol=1e-12)
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        sites[frame] = data.site_xpos[site_ids]
        headings[frame], bottoms[frame] = rolling_geometry(model, data)
        errors[frame] = np.linalg.norm(sites[frame, 5:7] - wheel_targets[frame], axis=-1)
        if frame % 200 == 0:
            print(f"{path.stem}: GMR {frame + 1}/{n}", flush=True)
    elapsed = time.perf_counter() - start
    wheel_qids = [model.joint(name).qposadr[0] for name in ("wheel_L_Joint", "wheel_R_Joint")]
    if not np.isfinite(qpos).all() or np.max(np.abs(qpos[:, wheel_qids])) > 1e-8:
        raise ValueError("Nonfinite pose or unfrozen wheel spin")
    intervals = np.empty((n - 1, model.nv))
    for frame, dt in enumerate(np.diff(source["time_s"])):
        mujoco.mj_differentiatePos(model, intervals[frame], dt, qpos[frame], qpos[frame + 1])
    qids = [model.joint(i).qposadr[0] for i in range(1, model.njnt)]
    ranges = model.jnt_range[1:]
    violation = float(max(0., np.max(np.maximum(ranges[:, 0] - qpos[:, qids], qpos[:, qids] - ranges[:, 1]))))
    if violation > 1e-6:
        raise ValueError("GMR joint-limit violation")
    features = heading_features(source["time_s"], root_targets, qpos[:, :3],
                                human_display[:, [index[name] for name in settings["bvh_ankle_nodes"]]],
                                human_display[:, [index[name] for name in settings["bvh_toe_base_nodes"]]],
                                headings, bottoms, settings, wheel_centers=sites[:, 5:7])
    robot_vids = [model.joint(i).dofadr[0] for i in range(1, model.njnt)
                  if not model.joint(i).name.startswith("wheel_")]
    report = {"clip": path.name, "method": "pinned_upstream_GMR_two_stage", "gmr_commit": settings["gmr_commit"],
              "upstream_core_modified": False, "upstream_retarget_calls": n, "both_stages": True,
              "frames": n, "duration_s": float(source["time_s"][-1]), "elapsed_s": elapsed,
              "model_sha256": sha(model_path), "source_sha256": sha(path), "task_sha256": sha(config_path),
              "robot_scale_m_per_source_unit": scale, "robot_leg_length_m_L_R": list(map(float, robot_lengths)),
              "human_leg_length_native_L_R": list(map(float, source_lengths)),
              "heading_alignment_rad": source["heading_alignment_rad"], "bvh_frame_time_s": source["dt"],
              "bvh_ankle_pivots": settings["bvh_ankle_nodes"], "foot_heading_axis": "BVH ankle joint pivot to toe-base pivot; not heel markers or measured sole axis",
              "root_xy_error_m": summary(np.linalg.norm(qpos[:, :2] - root_targets[:, :2], axis=-1)),
              "wheel_center_error_m": summary(errors), "joint_limit_violation_rad": violation,
              "max_sampled_leg_speed_rad_s": float(np.max(np.abs(intervals[:, robot_vids]))),
              "max_wheel_spin_rad": float(np.max(np.abs(qpos[:, wheel_qids]))),
              "max_wheel_ground_penetration_m": float(max(0., -bottoms.min())),
              "robot_near_ground_fraction_L_R": features["robot_near_ground_proxy"].mean(axis=0).tolist(),
              "human_near_ground_fraction_L_R": features["human_near_ground_proxy"].mean(axis=0).tolist(),
              "angle_analysis": summarize_features(source["time_s"], features, settings),
              "physics_validated": False, "policy_trained": False, "root_prescribed": False,
              "limitations": ["Source BVH has pre-existing human-to-human Jacobian IK skeleton standardization.",
                              "Uniform leg normalization: human distances/speeds are at robot scale, not measured anatomical SI units.",
                              "Three landmarks only: pelvis and ankles, no human knee or foot orientation task.",
                              "GMR itself uses Mink IK; this is the upstream GMR algorithm, not the historical custom Mink baseline.",
                              "Wheel spin is frozen; heading sidecar is diagnostic, not an independent wheel-yaw command.",
                              "No rolling, contact-force, collision avoidance, leg-speed or momentum constraint.",
                              "Near-ground masks are geometry proxies, not observed stance or contact forces.",
                              "Course-to-heading offsets are features, not a tracking mismatch or proof of sliding.",
                              "No training, friction validation or Isaac/MuJoCo dynamics rollout in this step."]}
    trace = {"time_s": source["time_s"], "human_positions_robot_scale_m": human_display,
             "human_joint_names": source["joint_names"], "human_parent_indices": source["parents"],
             "robot_site_positions_m": sites, "robot_site_names": np.array(site_names),
             "root_target_m": root_targets, "wheel_target_m": wheel_targets, "qpos": qpos,
             "qvel_intervals": intervals, "wheel_center_error_m": errors, **features}
    np.savez_compressed(output / "retarget_trace.npz", **trace)
    report["trace_sha256"] = sha(output / "retarget_trace.npz")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"clip": path.name, "wheel_error_m": report["wheel_center_error_m"],
                      "root_xy_error_m": report["root_xy_error_m"], "elapsed_s": elapsed}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/hdm05_gmr_experiment.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/data/hdm05_walking")
    parser.add_argument("--source-audit", type=Path, default=ROOT / "results/hdm05_downloads/bvh_audit.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip", nargs="+", help="Optional subset of the predeclared clip filenames")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Preserving existing experiment output; choose a fresh directory")
    settings = json.loads(args.config.read_text())
    clips = args.clip or settings["clips"]
    if len(clips) != len(set(clips)) or set(clips) - set(settings["clips"]):
        raise ValueError("Expected distinct, predeclared experiment clips")
    audit = json.loads(args.source_audit.read_text())
    hashes = {row["file"]: row["sha256"] for row in audit["files"]}
    for clip in clips:
        if hashes[clip] != sha(args.data_dir / clip):
            raise ValueError("Source differs from the acquisition audit: " + clip)
    GMR, params, extract, utils = load_gmr(settings)
    reports = [run_clip(args.data_dir / clip, output / Path(clip).stem,
                        args.model.resolve(), settings, GMR, params, extract, utils) for clip in clips]
    overview = {"kind": "HDM05_GMR_kinematic_contact_heading_experiment", "settings": settings,
                "pipeline_sha256": sha(Path(__file__)),
                "config_sha256": sha(args.config), "source_audit_sha256": sha(args.source_audit),
                "model_sha256": sha(args.model), "model_local_path": str(args.model.resolve()),
                "versions": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "mujoco", "mink", "qpsolvers", "daqp")},
                "clips": reports, "policy_trained": False, "dynamics_validated": False}
    (output / "experiment.json").write_text(json.dumps(overview, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
