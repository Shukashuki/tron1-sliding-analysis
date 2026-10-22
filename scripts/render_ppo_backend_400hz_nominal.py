"""Four horizontal opaque full-body replays of the completed nominal PPO cells.

Recorded400Hz poses are selected at50Hz, never integrated, filtered, aligned or
corrected. All four panels share one camera per frame plus fixed world-XY maps.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from analyze_ppo_backend_400hz_matrix import COLORS, heading_geometry_masks
from nmpc_tracking_objective import quaternion_rotate
from ppo_backend_400hz_contract import DEFAULT_PROTOCOL, INPUT_PINS, require, sha256, write_json
from render_physx_ppo_fullbody import DEFAULT_MODEL, build_qpos, create_runtime, mesh_provenance, validate_fk, _pose

ROOT = Path(__file__).resolve().parents[1]
ORDER = (("physx", "physx"), ("physx", "mujoco"), ("mjlab", "physx"), ("mjlab", "mujoco"))
TITLES = {("physx", "physx"): "PhysX-trained → PhysX",
    ("physx", "mujoco"): "PhysX-trained → native CPU MuJoCo",
    ("mjlab", "physx"): "MJWarp-trained → PhysX",
    ("mjlab", "mujoco"): "MJWarp-trained → native CPU MuJoCo"}
CANVAS_SIZE = (2560, 720)
FRAMES, FPS = 350, 50
CAMERA = {"azimuth_deg": 135., "elevation_deg": -45., "minimum_distance_m": 1.3,
          "vertical_fov_deg": 45., "robot_extent_margin_m": .03, "fit_margin": 1.08}


def native_indices():
    return np.arange(FRAMES, dtype=np.int64) * 8


def load_data(input_root, reference, model=DEFAULT_MODEL, legacy_evaluator=None, protocol=DEFAULT_PROTOCOL):
    from summarize_ppo_backend_400hz_nominal import load_nominal
    require(sha256(model) == INPUT_PINS["model"], "Original opaque display model SHA differs")
    traces, reference_values, summary = load_nominal(input_root, reference, model, legacy_evaluator, protocol)
    require(set(traces) == set(ORDER) and summary["nominal_only"] is True
            and summary["full_matrix_complete"] is False, "Require the four audited nominal cells, not a claimed full matrix")
    expected = np.arange(2793) * .0025
    for key in ORDER:
        trace = traces[key]
        require(trace["physics_dt_s"] == .0025 and trace["time_s"].shape == (2793,)
                and np.allclose(trace["time_s"], expected, rtol=0, atol=1e-10), "Native400Hz clock differs")
        require(trace["wheel_quaternion_wxyz"].shape == (2793, 2, 4), "Missing audited wheel FK rotations")
    time_difference = max(float(np.max(np.abs(traces[key]["time_s"][native_indices()] - expected[native_indices()]))) for key in ORDER)
    summary = copy.deepcopy(summary)
    completed = summary["matrix_status"]["counts"]["formal_completed"]
    queue_state = summary["matrix_status"]["status"]
    summary.update(kind="recorded_nominal_new400hz_ppo_four_horizontal_fullbody_replay",
        graphics_engine="MuJoCo graphics/kinematics only", case_order=[list(key) for key in ORDER],
        panel_titles={key[0] + "_to_" + key[1]: TITLES[key] for key in ORDER},
        native_samples_per_case=2793, native_sample_stride=8, video_frames=FRAMES, video_fps=FPS,
        first_original_time_s=0., last_original_time_s=6.98, encoded_duration_s=7.,
        selected_clock_max_absolute_difference_s=time_difference, video_size_pixels=list(CANVAS_SIZE),
        robot_mesh_alpha=1., physical_pose_corrected=False, root_height_corrected=False,
        pose_alignment=False, state_interpolation=False, state_filtering=False, dynamics_stepping=False,
        contact_solver_called=False, new_policy_inference=False, new_training=False, visual_reviewed=False,
        rendering_limitations=[f"Four completed nominal mu=0.7 cells only. Full 17-mu matrix is partial ({completed}/68 observed); queue status: {queue_state}. This replay does not restart it.",
            "N=1 per cell. Displayed 50 Hz frames cannot establish continuous contact or resolve all 400 Hz chatter.",
            "The common 3D camera fits all four original robot meshes, never recenters one robot independently.",
            "Fixed world-XY maps show all four full trajectories with shared bounds; no drift or fall is removed.",
            "Wheel glyphs are axial rolling planes from local +Y axle cross world +Z; wheel spin does not define heading.",
            "Counts/Fz are native backend-specific telemetry. Zero count is not flight or a success/failure label.",
            "Reference ends 6.96 s. Camera may hold the final knot for display at 6.98 s; XY error is unavailable beyond real reference.",
            "The same opaque display meshes/light settings are used; source physical inertias/contact laws are not replaced."])
    return traces, reference_values, summary


def camera_schedule(traces, reference, width=640, height=360, *, geometry=None, basis=None):
    """One perspective-fit camera for all original poses, without pose edits."""
    az, el = np.radians([CAMERA["azimuth_deg"], CAMERA["elevation_deg"]])
    outward = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), -np.sin(el)])
    right = np.cross(-outward, np.array([0., 0., 1.])); right /= np.linalg.norm(right)
    up = np.cross(outward, right)
    if basis is not None:
        right, up, outward = basis
    tangent_y = np.tan(np.radians(CAMERA["vertical_fov_deg"]) / 2.)
    tangent_x = tangent_y * width / height
    lookats, distances = [], []
    for frame, index in enumerate(native_indices()):
        time = min(frame / FPS, float(reference["time_s"][-1]))
        ref = np.array([np.interp(time, reference["time_s"], reference["position"][:, j]) for j in range(3)])
        points = [ref]
        if geometry is not None:
            points.extend(geometry[frame])
        for key in ORDER:
            points.append(traces[key]["position"][index])
            points.extend(traces[key]["wheel_pos"][index])
        points = np.asarray(points)
        axes = np.column_stack((right, up, outward))
        projected = points @ axes
        middle = (projected.min(axis=0) + projected.max(axis=0)) / 2.
        center = axes @ middle
        relative = points - center
        required = relative @ outward + np.maximum(np.abs(relative @ right) / tangent_x, np.abs(relative @ up) / tangent_y)
        distance = max(CAMERA["minimum_distance_m"], (float(required.max()) + CAMERA["robot_extent_margin_m"]) * CAMERA["fit_margin"])
        lookats.append(center)
        distances.append(distance)
    return np.asarray(lookats), np.asarray(distances)


def geometry_schedule(runtime, qposes):
    """Bound actual opaque mesh vertices, transformed only by recorded FK."""
    import mujoco
    model, data = runtime.model, mujoco.MjData(runtime.model)
    templates = []
    for geom in range(model.ngeom):
        if not model.geom_bodyid[geom] or not runtime.option.geomgroup[int(model.geom_group[geom])]:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start+count].astype(float)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        templates.append((geom, corners))
    require(bool(templates), "No visible opaque robot mesh geometry to fit")
    result = []
    for index in native_indices():
        points = []
        for key in ORDER:
            _pose(model, data, qposes[key][index])
            points.extend(np.concatenate([corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                                          for geom, corners in templates]))
        result.append(np.asarray(points))
    return result, len(templates)


def actual_camera_basis(runtime, qpos):
    """Read native visual-camera axes, not an assumed azimuth convention."""
    camera = runtime.cameras[0]
    camera.azimuth, camera.elevation = CAMERA["azimuth_deg"], CAMERA["elevation_deg"]
    camera.distance, camera.lookat[:] = 1., 0.
    _pose(runtime.model, runtime.data, qpos)
    runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
    visual = runtime.renderer.scene.camera[0]
    forward, up = np.asarray(visual.forward, float), np.asarray(visual.up, float)
    forward /= np.linalg.norm(forward); up /= np.linalg.norm(up)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    return right, up, -forward


def camera_fit_audit(geometry, lookats, distances, basis):
    right, up, outward = basis
    ty = np.tan(np.radians(CAMERA["vertical_fov_deg"])/2.); tx = ty*640/360
    maximum = 0.
    for points, center, distance in zip(geometry, lookats, distances):
        relative = points - center
        depth = distance - relative @ outward
        require(np.all(depth > 0.), "A recorded mesh lies behind the common camera")
        maximum = max(maximum, float(np.max(np.maximum(np.abs(relative @ right)/(depth*tx), np.abs(relative @ up)/(depth*ty)))))
    require(maximum < 1., "A recorded robot mesh would be clipped by the common camera")
    return {"frames_checked": 350, "panels_per_frame": 4, "maximum_normalized_frustum_extent": maximum,
        "all_visible_mesh_AABB_corners_inside": True, "mesh_bounds_margin_m": CAMERA["robot_extent_margin_m"],
        "body_pose_changes": False, "basis_source": "native MjvGLCamera forward/up after visual scene update"}


def world_bounds(traces, reference):
    points = np.concatenate([reference["position"][:, :2], *[traces[key]["position"][:, :2] for key in ORDER],
                             *[traces[key]["wheel_pos"][:, :, :2].reshape(-1, 2) for key in ORDER]])
    require(np.isfinite(points).all(), "Nonfinite original world-XY map geometry")
    return points.min(axis=0) - .25, points.max(axis=0) + .25


def replay_fk_audit(model, trace, qpos):
    """Retain the original opaque renderer's strict 2µm all-sample FK guard."""
    audit = validate_fk(model, trace, qpos)
    require(np.array_equal(qpos[:, :3], trace["position"]) and np.array_equal(qpos[:, 3:7], trace["quaternion"]),
            "Recorded original root pose was changed")
    return {**audit, "root_pose_copied_exactly": True}


def diagnostics(trace, reference, index):
    time = float(trace["time_s"][index])
    target = None
    if time <= reference["time_s"][-1] + 1e-10:
        target = np.array([np.interp(time, reference["time_s"], reference["position"][:, j]) for j in range(2)])
    up = quaternion_rotate(trace["quaternion"][index], np.array([0., 0., 1.]))
    return {"xy_error_m": float(np.linalg.norm(trace["position"][index, :2] - target)) if target is not None else None,
        "speed_m_s": float(np.linalg.norm(trace["com_velocity"][index, :2])),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1., 1.)))),
        "height_m": float(trace["position"][index, 2]), "count": trace["contact_count"][index],
        "fz": trace["contact_force"][index, :, 2]}


def _map_panel(draw, trace, reference, index, box, bounds, font):
    low, high = np.asarray(bounds, dtype=float)
    left, top, right, bottom = box
    scale = min((right-left-20) / (high[0]-low[0]), (bottom-top-28) / (high[1]-low[1]))
    center = (low+high) / 2.
    pixel = lambda point: ((left+right)/2. + (point[0]-center[0])*scale,
                           (top+bottom)/2. - (point[1]-center[1])*scale)
    draw.rectangle(box, fill="#f7f9fa", outline="#c9cdd1")
    draw.text((left+8, top+3), "Fixed world-XY: ref grey | actual green | body blue | L red / R purple ↔", font=font, fill="#333333")
    draw.text((left+8, bottom-18), f"X→ [{low[0]:.2f},{high[0]:.2f}]m | Y↑ [{low[1]:.2f},{high[1]:.2f}]m | same bounds/scale in all panels", font=font, fill="#555555")
    draw.line([pixel(point) for point in reference["position"][:, :2]], fill=COLORS["reference"], width=2)
    draw.line([pixel(point) for point in trace["position"][:index+1:8, :2]], fill=COLORS["path"], width=2)
    root = trace["position"][index, :2]
    x, y = pixel(root)
    draw.ellipse((x-3, y-3, x+3, y+3), fill=COLORS["path"])
    body_valid, wheel_valid = trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"]
    signals = trace["signals"]
    arrows = [(root, signals["body_heading_rad"][index], "body", body_valid[index], False)]
    for wheel, color in enumerate(("left", "right")):
        arrows.append((trace["wheel_pos"][index, wheel, :2], signals["wheel_rolling_plane_heading_rad"][index, wheel], color, wheel_valid[index, wheel], True))
    for point, heading, color, valid, axial in arrows:
        if not valid:
            continue
        direction = .18 * np.array([np.cos(heading), np.sin(heading)])
        start = point-direction/2. if axial else point
        stop = point+direction/2. if axial else point+direction
        draw.line((pixel(start), pixel(stop)), fill=COLORS[color], width=3)
        for tip, sign in ((stop, 1.), (start, -1.)) if axial else ((stop, 1.),):
            tip_px = np.asarray(pixel(tip))
            axis = np.array([np.cos(heading), -np.sin(heading)]) * sign
            normal = np.array([-axis[1], axis[0]])
            draw.polygon([tuple(tip_px), tuple(tip_px-axis*6.+normal*3.), tuple(tip_px-axis*6.-normal*3.)], fill=COLORS[color])


def render_frame(runtime, traces, qposes, reference, summary, frame):
    from PIL import Image, ImageDraw
    require(isinstance(frame, int) and 0 <= frame < FRAMES, "Invalid native50Hz replay frame")
    require((runtime.width, runtime.height) == (640, 360), "Fixed horizontal canvas requires640x360 views")
    index = int(native_indices()[frame])
    canvas = Image.new("RGB", CANVAS_SIZE, "#fafafa")
    draw = ImageDraw.Draw(canvas)
    observed = summary.get("matrix_status", {})
    completed = observed.get("counts", {}).get("formal_completed", "?")
    state = observed.get("status", "not provided")
    draw.text((12, 5), f"NEW 400 Hz PPO | μ=0.7 | t={traces[ORDER[0]]['time_s'][index]:.2f} s | PARTIAL grid {completed}/68; queue: {state} | all 4 share the same 3D camera and world-XY scale", font=runtime.font, fill="#111111")
    camera = runtime.cameras[0]
    camera.azimuth, camera.elevation = CAMERA["azimuth_deg"], CAMERA["elevation_deg"]
    camera.lookat[:] = summary["camera_lookat_m"][frame]
    camera.distance = float(summary["camera_distance_m"][frame])
    for column, key in enumerate(ORDER):
        trace = traces[key]
        _pose(runtime.model, runtime.data, qposes[key][index])
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        canvas.paste(Image.fromarray(runtime.renderer.render().copy()), (column*640, 58))
        x = column*640
        draw.rectangle((x, 30, x+640, 57), fill="#f0f2f4")
        draw.text((x+9, 34), TITLES[key], font=runtime.font, fill="#111111")
        value = diagnostics(trace, reference, index)
        xy = f"{value['xy_error_m']:.3f} m" if value["xy_error_m"] is not None else "n/a (ref ends 6.96 s)"
        draw.text((x+9, 424), f"XY {xy} | COM speed {value['speed_m_s']:.2f} m/s | tilt {value['tilt_deg']:.1f} deg", font=runtime.font, fill="#111111")
        draw.text((x+9, 447), f"Original height {value['height_m']:.3f} m | native 400 Hz / video 50 Hz", font=runtime.font, fill="#111111")
        draw.text((x+9, 470), f"L/R ground count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f} N", font=runtime.font, fill="#111111")
        _map_panel(draw, trace, reference, index, (x+5, 500, x+635, 672), summary["world_bounds"], runtime.small)
    draw.text((12, 683), "N=1 per nominal cell. Recorded opaque meshes only: no new simulation, inference, pose correction or interpolation. Contact telemetry is diagnostic, not a flight/failure label.", font=runtime.small, fill="#333333")
    draw.text((12, 702), "Body → | left/right axial rolling planes ↔ from +Y axle × world +Z. All 4 use the same camera; no individual recentering.", font=runtime.small, fill="#333333")
    return canvas


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh directory")
    # The audited loader performs FK and therefore imports MuJoCo before
    # create_runtime. Select CPU OSMesa BEFORE that first import, not after.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    require(os.environ["MUJOCO_GL"] == "osmesa", "This local replay requires CPU OSMesa, not a GPU graphics context")
    traces, reference, summary = load_data(args.input_root, args.reference, args.model,
                                          args.legacy_mujoco_evaluator, args.cohort_protocol)
    runtime = create_runtime(args.model, width=640, height=360)
    try:
        runtime.model.vis.global_.fovy = CAMERA["vertical_fov_deg"]
        qposes, audits, replay_arrays = {}, {}, {}
        for key in ORDER:
            qposes[key], mapping = build_qpos(runtime.model, traces[key])
            audit = replay_fk_audit(runtime.model, traces[key], qposes[key])
            body_valid, wheel_valid = heading_geometry_masks(traces[key])
            traces[key]["body_heading_geometry_valid"] = body_valid
            traces[key]["wheel_heading_geometry_valid"] = wheel_valid
            name = key[0] + "_to_" + key[1]
            audits[name] = {**audit, "named_joint_qpos_addresses": mapping.tolist()}
            replay_arrays.update({name+"_native_time_s": traces[key]["time_s"], name+"_qpos": qposes[key],
                name+"_selected_native_indices": native_indices(), name+"_recorded_ground_count": traces[key]["contact_count"],
                name+"_recorded_ground_force_w_n": traces[key]["contact_force"],
                name+"_original_root_position_m": traces[key]["position"],
                name+"_original_root_quaternion_wxyz": traces[key]["quaternion"]})
        geometry, visible_meshes = geometry_schedule(runtime, qposes)
        basis = actual_camera_basis(runtime, qposes[ORDER[0]][0])
        lookats, distances = camera_schedule(traces, reference, geometry=geometry, basis=basis)
        fit_audit = camera_fit_audit(geometry, lookats, distances, basis)
        runtime.model.vis.map.zfar = max(float(runtime.model.vis.map.zfar),
            2. * (float(distances.max()) + 2.) / float(runtime.model.stat.extent))
        bounds = world_bounds(traces, reference)
        summary.update(camera_lookat_m=lookats.tolist(), camera_distance_m=distances.tolist(), world_bounds=[value.tolist() for value in bounds],
            camera_settings=CAMERA, camera_same_across_panels=True, camera_individual_recenter=False,
            camera_fit_includes_all_four_original_root_and_wheel_positions=True,
            camera_fit_audit=fit_audit, visible_robot_mesh_geoms_per_panel=visible_meshes,
            camera_distance_range_m=[float(distances.min()), float(distances.max())],
            graphics_far_clip_m=float(runtime.model.vis.map.zfar) * float(runtime.model.stat.extent),
            fk_validation_by_panel=audits, mesh_provenance=mesh_provenance(args.model),
            model_sha256=sha256(args.model), renderer_source_sha256=sha256(__file__))
        output.mkdir(parents=True)
        np.savez_compressed(output / "replay_states.npz", **replay_arrays,
                            common_camera_lookat_m=lookats, common_camera_distance_m=distances)
        render_frame(runtime, traces, qposes, reference, summary, 150).save(output / "comparison.png")
        files = ["replay_states.npz", "comparison.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output / "fullbody_comparison.mp4"), CANVAS_SIZE,
                fps=FPS, codec="libx264", quality=7, macro_block_size=2,
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for frame in range(FRAMES):
                    stream.send(np.asarray(render_frame(runtime, traces, qposes, reference, summary, frame)))
            finally:
                stream.close()
            files.append("fullbody_comparison.mp4")
        summary["video_generated"] = not args.skip_video
        summary["render_helpers_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in
            ("summarize_ppo_backend_400hz_nominal.py", "analyze_ppo_backend_400hz_matrix.py", "render_physx_ppo_fullbody.py",
             "controller_trace.py", "controller_metrics.py", "nmpc_tracking_objective.py")}
        summary["artifacts"] = {name: sha256(output / name) for name in files}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-root", type=Path, required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, required=True)
    result.add_argument("--cohort-protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
