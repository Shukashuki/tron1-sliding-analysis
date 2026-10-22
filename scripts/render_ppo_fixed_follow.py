"""Fixed-distance, fixed-world-orientation camera for recorded-pose replays.

Only camera translation follows each original root plus a constant offset.
One distance is selected ONCE from all displayed cases/frames: never zoom.
"""
from __future__ import annotations

import numpy as np

from ppo_backend_400hz_contract import require
from render_physx_ppo_fullbody import _pose


def mesh_templates(runtime):
    import mujoco
    model, result = runtime.model, []
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
        result.append((geom, corners))
    require(bool(result), "No visible robot mesh geometry")
    return result


def visual_basis(runtime, qpos, azimuth_deg, elevation_deg):
    camera = runtime.cameras[0]
    camera.azimuth, camera.elevation = azimuth_deg, elevation_deg
    camera.distance, camera.lookat[:] = 1., 0.
    _pose(runtime.model, runtime.data, qpos)
    runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
    native = runtime.renderer.scene.camera[0]
    forward, up = np.asarray(native.forward, float), np.asarray(native.up, float)
    forward /= np.linalg.norm(forward); up /= np.linalg.norm(up)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    return right, up, -forward


def fit_fixed_distance(geometry, targets, basis, *, width=640, height=360, vertical_fov_deg=45.,
                       margin_m=.03, fit_margin=1.08, minimum_distance_m=1.3):
    """Pure camera math; geometry/targets are case -> existing-frame arrays."""
    right, up, outward = basis
    tangent_y = np.tan(np.radians(vertical_fov_deg)/2.)
    tangent_x = tangent_y * width/height
    require(set(geometry) == set(targets) and bool(geometry), "Camera cases differ")
    maximum_required = 0.
    for key in geometry:
        require(len(geometry[key]) == len(targets[key]), "Camera frame count differs")
        for points, target in zip(geometry[key], targets[key]):
            relative = np.asarray(points) - target
            require(relative.ndim == 2 and relative.shape[1] == 3 and np.isfinite(relative).all(), "Invalid mesh bounds")
            required = relative @ outward + np.maximum(np.abs(relative @ right)/tangent_x, np.abs(relative @ up)/tangent_y)
            maximum_required = max(maximum_required, float(required.max()))
    distance = max(minimum_distance_m, (maximum_required + margin_m) * fit_margin)
    extent, minimum_depth = 0., float("inf")
    for key in geometry:
        for points, target in zip(geometry[key], targets[key]):
            relative = points - target
            depth = distance - relative @ outward
            require(np.all(depth > 0.), "Mesh behind fixed camera")
            extent = max(extent, float(np.max(np.maximum(np.abs(relative @ right)/(depth*tangent_x), np.abs(relative @ up)/(depth*tangent_y)))))
            minimum_depth = min(minimum_depth, float(depth.min()))
    require(extent < 1., "Visible robot mesh would clip in fixed follow camera")
    return distance, {"all_mesh_bounds_inside": True, "maximum_normalized_frustum_extent": extent,
        "minimum_depth_m": minimum_depth, "distance_changed_with_time": False,
        "distance_differs_between_cases": False, "margin_m": margin_m}


def prepare_camera(runtime, traces, qposes, order, indices, *, offset=(0., 0., -.28),
                   azimuth_deg=135., elevation_deg=-30.):
    """Graphics/FK only. Inputs and all original physical poses stay unchanged."""
    import mujoco
    indices = np.asarray(indices)
    require(indices.ndim == 1 and indices.dtype.kind in "iu" and len(indices) > 0, "Invalid original native indices")
    require(set(traces) == set(qposes) == set(order), "Replay cases differ")
    runtime.model.vis.global_.fovy = 45.
    templates = mesh_templates(runtime)
    basis = visual_basis(runtime, qposes[order[0]][int(indices[0])], azimuth_deg, elevation_deg)
    targets, geometry = {}, {}
    data = mujoco.MjData(runtime.model)
    for key in order:
        targets[key] = traces[key]["position"][indices].copy() + np.asarray(offset)
        geometry[key] = []
        for index in indices:
            _pose(runtime.model, data, qposes[key][int(index)])
            geometry[key].append(np.concatenate([corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                                                 for geom, corners in templates]))
    distance, audit = fit_fixed_distance(geometry, targets, basis, width=runtime.width, height=runtime.height)
    runtime.model.vis.map.zfar = max(float(runtime.model.vis.map.zfar), 2.*(distance+2.)/float(runtime.model.stat.extent))
    audit.update(frames_per_case=len(indices), cases=len(order), visible_mesh_geoms_per_case=len(templates),
        fixed_root_relative_offset_m=list(offset), orientation_follows_yaw=False, original_root_pose_changed=False,
        basis_source="native MjvGLCamera forward/up", physics_stepping=False)
    return {"lookat_by_case": targets, "distance_m": distance, "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg, "vertical_fov_deg": 45., "fit_audit": audit}


def apply_camera(runtime, plan, key, frame):
    camera = runtime.cameras[0]
    camera.lookat[:] = plan["lookat_by_case"][key][frame]
    camera.distance = plan["distance_m"]
    camera.azimuth, camera.elevation = plan["azimuth_deg"], plan["elevation_deg"]
    return camera
