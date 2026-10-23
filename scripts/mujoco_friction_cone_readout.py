"""Read current solved MuJoCo wheel/floor contacts without changing the plant.

No stepping, forward computation, reset, model edits or inferred wheel axes.
The caller must supply a current solved MjData with valid position Jacobians.
Contact-frame rows are [normal, tangent1, tangent2] in world coordinates;
mj_contactForce is force:torque acting ON geom2. Relative velocities therefore
use geom2 minus geom1 in the SAME original engine frame, including moving-floor
and angular point velocities. Per-contact arrays have a ragged leading C axis.

References: https://mujoco.readthedocs.io/en/stable/computation/index.html#contact
and the mj_contactForce/mj_jac API. This reader does not evaluate a friction
cone: condim4/6 include torsional/rolling torque, not just translational Ft.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _index(value, upper, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + " must be an explicit integer id")
    result = int(value)
    if not (int(positive) <= result < upper):
        raise ValueError(name + " is outside the model")
    return result


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + " must have the expected finite shape")
    return result.copy()


def read_wheel_floor_contacts(model, data, *, floor_geom_id, wheel_body_ids,
                              wheel_lateral_axes_local=None):
    """Return owned arrays from active rigid contacts with ONE exact floor geom.

    wheel_body_ids supplies two distinct actual model body ids, in caller order;
    neither must be world or the floor body. Sensor semantics do not require a
    cylinder: any rigid wheel body geom can be diagnosed. Active means a real
    solver contact with efc_address>=0 and exclude==0; zero-force active contacts
    are retained and must later be masked by the cone-math load criterion.

    Optional lateral axes are explicit nonzero local vectors, shape(2,3). They
    are normalized and projected into each contact tangent plane. No axes are
    guessed when omitted; the lateral fields are then NaN with validity=False.
    Forces are N, contact torques Nm, point velocities m/s and angular rates
    rad/s. friction[0:2] is dimensionless; torsional/rolling coefficients have
    units of length. Full six-component wrench and torque moments are retained.

    Contact forces refer to the CURRENT cached solution, not a recomputed
    post-forward or next-state solution. Only mj_contactForce and mj_jac are
    called; both write solely into this function's temporary output buffers.
    """
    import mujoco

    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("Require an actual MuJoCo MjModel and current solved MjData")
    floor = _index(floor_geom_id, model.ngeom, "floor_geom_id")
    try:
        supplied = tuple(wheel_body_ids)
    except TypeError as error:
        raise ValueError("wheel_body_ids must contain two explicit body ids") from error
    if len(supplied) != 2:
        raise ValueError("wheel_body_ids must contain two explicit body ids")
    wheels = tuple(_index(v, model.nbody, "wheel_body_ids", positive=True) for v in supplied)
    floor_body = int(model.geom_bodyid[floor])
    if wheels[0] == wheels[1] or floor_body in wheels:
        raise ValueError("Wheel bodies must be distinct and exclude the floor body")
    velocity = _array(data.qvel, (model.nv,), "current qvel")
    if not np.isfinite(data.time):
        raise ValueError("Current simulator time is not finite")
    cone = int(model.opt.cone)
    if cone not in (int(mujoco.mjtCone.mjCONE_PYRAMIDAL), int(mujoco.mjtCone.mjCONE_ELLIPTIC)):
        raise ValueError("Unknown MuJoCo friction-cone type")
    axes = None
    if wheel_lateral_axes_local is not None:
        axes = _array(wheel_lateral_axes_local, (2, 3), "wheel_lateral_axes_local")
        norm = np.linalg.norm(axes, axis=-1)
        if np.any(norm <= 1.e-12):
            raise ValueError("Explicit wheel lateral axes must be nonzero")
        axes /= norm[:, None]
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    wrench = np.empty(6)
    records, inactive = [], []

    def point_velocity(point, body):
        # World body naturally yields zero Jacobians; do not assume the floor
        # belongs to world, because a sliding/rotating ground body is allowed.
        mujoco.mj_jac(model, data, jacp, jacr, point, body)
        return jacp @ velocity, jacr @ velocity

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if floor not in (geom1, geom2) or geom1 < 0 or geom2 < 0:
            continue
        other = geom2 if geom1 == floor else geom1
        body = int(model.geom_bodyid[other])
        if body not in wheels:
            continue  # Nonwheel/floor and wheel/wheel self-contacts excluded.
        if int(contact.efc_address) < 0 or int(contact.exclude) != 0:
            inactive.append(index)
            continue
        dim = int(contact.dim)
        if dim not in (1, 3, 4, 6):
            raise ValueError("Unsupported active contact dimensionality")
        frame = _array(contact.frame, (9,), "contact.frame").reshape(3, 3)
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1.e-6, rtol=0):
            raise ValueError("Contact frame is not orthonormal")
        point = _array(contact.pos, (3,), "contact.pos")
        friction = _array(contact.friction, (5,), "contact.friction")
        if np.any(friction < 0) or not np.isfinite(contact.dist):
            raise ValueError("Invalid contact friction or distance")
        mujoco.mj_contactForce(model, data, index, wrench)
        if not np.isfinite(wrench).all():
            raise ValueError("Current solved contact wrench is not finite")
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        v1, w1 = point_velocity(point, body1)
        v2, w2 = point_velocity(point, body2)
        relative = v2-v1
        relative_cf, angular_cf = frame @ relative, frame @ (w2-w1)
        wheel_is_geom2 = geom2 == other
        sign = 1. if wheel_is_geom2 else -1.
        force_world = sign * (frame.T @ wrench[:3])
        torque_world = sign * (frame.T @ wrench[3:])
        wheel_relative = sign * relative
        wheel = wheels.index(body)
        lateral_world, lateral_speed, lateral_valid = np.full(3, np.nan), np.nan, False
        if axes is not None:
            rotation = _array(data.xmat[body], (9,), "wheel xmat").reshape(3, 3)
            lateral = rotation @ axes[wheel]
            lateral -= np.dot(lateral, frame[0]) * frame[0]
            length = float(np.linalg.norm(lateral))
            if length > 1.e-10:
                lateral_world, lateral_valid = lateral/length, True
                lateral_speed = float(np.dot(wheel_relative, lateral_world))
        records.append({
            "contact_index": index, "wheel_index": wheel, "wheel_body_id": body,
            "floor_body_id": floor_body, "geom1_id": geom1, "geom2_id": geom2,
            "body1_id": body1, "body2_id": body2, "condim": dim,
            "efc_address": int(contact.efc_address), "wheel_is_geom2": wheel_is_geom2,
            "wheel_force_sign": sign, "contact_distance_m": float(contact.dist),
            "contact_position_world_m": point, "contact_frame_world": frame,
            "friction_coefficients": friction, "wrench_on_geom2_contact_frame": wrench.copy(),
            "fn_n": float(wrench[0]), "ft_contact_frame_n": wrench[1:3].copy(),
            "torque_on_geom2_contact_frame_nm": wrench[3:].copy(),
            "force_on_wheel_world_n": force_world,
            "normal_force_on_wheel_world_n": sign * wrench[0] * frame[0],
            "tangent_force_on_wheel_world_n": sign * (frame[1:].T @ wrench[1:3]),
            "torque_on_wheel_at_contact_world_nm": torque_world,
            "contact_moment_about_wheel_com_world_nm": torque_world + np.cross(point-data.xipos[body], force_world),
            "geom1_point_velocity_world_m_s": v1, "geom2_point_velocity_world_m_s": v2,
            "relative_point_velocity_geom2_minus_geom1_world_m_s": relative,
            "relative_point_velocity_geom2_minus_geom1_contact_frame_m_s": relative_cf,
            "relative_tangent_velocity_m_s": relative_cf[1:].copy(),
            "relative_angular_velocity_geom2_minus_geom1_contact_frame_rad_s": angular_cf,
            "wheel_relative_point_velocity_world_m_s": wheel_relative,
            "tangent_slip_speed_m_s": float(np.linalg.norm(relative_cf[1:])),
            "tangent_contact_power_w": float(np.dot(wrench[1:3], relative_cf[1:])),
            "rotational_contact_power_w": float(np.dot(wrench[3:], angular_cf)),
            "wheel_lateral_axis_world": lateral_world,
            "wheel_lateral_slip_m_s": lateral_speed, "wheel_lateral_axis_valid": lateral_valid})

    integers = ("contact_index", "wheel_index", "wheel_body_id", "floor_body_id", "geom1_id", "geom2_id", "body1_id", "body2_id", "condim", "efc_address")
    booleans = ("wheel_is_geom2", "wheel_lateral_axis_valid")
    scalars = ("wheel_force_sign", "contact_distance_m", "fn_n", "tangent_slip_speed_m_s", "tangent_contact_power_w", "rotational_contact_power_w", "wheel_lateral_slip_m_s")
    vectors2 = ("ft_contact_frame_n", "relative_tangent_velocity_m_s")
    vectors3 = ("contact_position_world_m", "torque_on_geom2_contact_frame_nm", "force_on_wheel_world_n", "normal_force_on_wheel_world_n", "tangent_force_on_wheel_world_n", "torque_on_wheel_at_contact_world_nm", "contact_moment_about_wheel_com_world_nm", "geom1_point_velocity_world_m_s", "geom2_point_velocity_world_m_s", "relative_point_velocity_geom2_minus_geom1_world_m_s", "relative_point_velocity_geom2_minus_geom1_contact_frame_m_s", "relative_angular_velocity_geom2_minus_geom1_contact_frame_rad_s", "wheel_relative_point_velocity_world_m_s", "wheel_lateral_axis_world")
    result = {}
    for fields, shape, dtype in ((integers, (), np.int64), (booleans, (), bool), (scalars, (), np.float64),
                                  (vectors2, (2,), np.float64), (vectors3, (3,), np.float64),
                                  (("friction_coefficients",), (5,), np.float64),
                                  (("wrench_on_geom2_contact_frame",), (6,), np.float64),
                                  (("contact_frame_world",), (3, 3), np.float64)):
        for field in fields:
            result[field] = np.asarray([record[field] for record in records], dtype=dtype).reshape((len(records), *shape)).copy()
    result["metadata"] = {
        "schema_version": 1, "kind": "current_solved_mujoco_wheel_floor_contact_readout",
        "mujoco_version": mujoco.__version__, "time_s": float(data.time),
        "physics_dt_s": float(model.opt.timestep), "ncon_total": int(data.ncon),
        "contact_rows": len(records), "floor_geom_id": floor, "wheel_body_ids": list(wheels),
        "skipped_inactive_floor_wheel_contact_indices": inactive,
        "opt_cone": cone, "cone_type": "pyramidal" if cone == 0 else "elliptic",
        "contact_wrench_convention": "force:torque ON geom2, original engine [normal,tangent1,tangent2] frame",
        "relative_velocity_convention": "point velocity geom2 minus geom1 in SAME original contact frame",
        "wheel_world_force_convention": "actual force on the wheel; sign reversed when wheel is geom1",
        "force_units": "N", "torque_units": "Nm", "velocity_units": "m/s", "angular_velocity_units": "rad/s",
        "friction_coefficient_units": ["1", "1", "m", "m", "m"],
        "wheel_lateral_axes_explicitly_supplied": axes is not None,
        "normalized_wheel_lateral_axes_local": None if axes is None else axes.tolist(),
        "invalid_lateral_projection_storage": "NaN with wheel_lateral_axis_valid=False; never a guessed axis",
        "full_contact_wrench_cone_evaluated": False,
        "condim4_6_translational_ratio_is_not_full_cone_utilization": True,
        "mixed_condim_math": "Group condim3 rows before requesting an engine boundary; dim1/4/6 may use only an explicitly labeled translational reference, never a full-wrench cone claim",
        "cone_math_input_fields": {"fn": "fn_n", "ft": "ft_contact_frame_n", "mu": "friction_coefficients[:,:2]", "slip_m_s": "relative_tangent_velocity_m_s", "condim": "condim"},
        "caller_supplies_current_solved_data_freshness_not_recomputed_or_certified": True,
        "no_step_forward_reset_or_state_mutation": True, "outputs_are_owned_copies": True}
    return result
