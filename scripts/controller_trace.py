"""Offline adapters for learned-policy and NMPC native physical traces.

Only an isolated kinematic MuJoCo workspace is written. No mj_step, contact
re-simulation, retarget, root correction of the recorded plant, or inferred
contact labels. Wheel-center velocities are analytic link-origin Jacobians,
not finite differences across discontinuous contact transitions.
"""
from __future__ import annotations

import numpy as np

from controller_metrics import heading_signals
from nmpc_tracking_objective import JOINT_NAMES


def native_trace(data, kind):
    """Convert explicit existing schemas; reject multi-env or reset-spliced data."""
    if kind not in ("policy_isaac", "policy_mujoco", "nmpc"):
        raise ValueError("Select policy_isaac, policy_mujoco or nmpc explicitly")
    time = np.asarray(data["time_s"], dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Invalid increasing native physical clock")
    if "physics_dt_s" not in data:
        raise ValueError("Missing explicit physical sample interval")
    raw_dt = np.asarray(data["physics_dt_s"])
    if raw_dt.shape != () or not np.isfinite(raw_dt) or float(raw_dt) <= 0:
        raise ValueError("Invalid scalar physical sample interval")
    dt = float(raw_dt)
    if not np.allclose(np.diff(time), dt, atol=1e-9, rtol=0):
        raise ValueError("Native clock is not consecutive physical sampling")
    selected = np.ones(len(time), dtype=bool)
    if kind == "policy_isaac":
        valid = np.asarray(data["valid_mask"])
        if valid.shape != (len(time), 1) or valid.dtype.kind != "b":
            raise ValueError("Require one Isaac environment with a boolean first-episode mask")
        selected = valid[:, 0]
        if not selected[0] or np.any(np.diff(selected.astype(int)) > 0) or selected.sum() < 2:
            raise ValueError("First episode must be a single nonempty prefix, never reset-spliced")
        fields = {"position": "root_position_m", "quaternion": "root_quaternion_wxyz",
                  "com_velocity": "root_linear_velocity_w_m_s", "angular_velocity": "root_angular_velocity_w_rad_s",
                  "joint_pos": "joint_position_rad", "joint_vel": "joint_velocity_rad_s", "wheel_pos": "wheel_position_m"}
        names_key = "joint_names"
    elif kind == "policy_mujoco":
        fields = {"position": "root_pos", "quaternion": "root_quat_wxyz",
                  "com_velocity": "root_com_lin_vel_w", "angular_velocity": "root_ang_vel_w",
                  "joint_pos": "joint_pos", "joint_vel": "joint_vel", "wheel_pos": "wheel_pos"}
        names_key = "joint_names"
    else:
        fields = {"position": "root_pos", "quaternion": "root_quaternion_wxyz",
                  "com_velocity": "base_com_velocity_w", "angular_velocity": "base_angular_velocity_w",
                  "joint_pos": "joint_pos", "joint_vel": "joint_vel", "wheel_pos": "wheel_pos"}
        names_key = "action_joint_names"
    fields.update(contact_count="wheel_ground_contact_count", contact_force="wheel_ground_contact_force_w_n",
                  contact_slip="wheel_lateral_contact_slip_m_s")
    names = tuple(np.asarray(data[names_key]).tolist())
    if len(names) != 8 or set(names) != set(JOINT_NAMES):
        raise ValueError("Trace must name the eight unique physical joints")
    order = [names.index(name) for name in JOINT_NAMES]
    result = {"time_s": time[selected].copy(), "physics_dt_s": dt, "joint_names": JOINT_NAMES}
    widths = {"position": (3,), "quaternion": (4,), "com_velocity": (3,), "angular_velocity": (3,),
              "joint_pos": (8,), "joint_vel": (8,), "wheel_pos": (2, 3),
              "contact_count": (2,), "contact_force": (2, 3), "contact_slip": (2,)}
    for name, source in fields.items():
        value = np.asarray(data[source], dtype=float)
        expected = (len(time), *((1,) if kind == "policy_isaac" else ()), *widths[name])
        if value.shape != expected:
            raise ValueError(f"{source}: expected native shape {expected}")
        value = value[:, 0] if kind == "policy_isaac" else value
        value = value[selected]
        if not np.isfinite(value).all():
            raise ValueError(f"{source}: nonfinite first-episode measurements")
        result[name] = value[:, order].copy() if name in ("joint_pos", "joint_vel") else value.copy()
    if not np.allclose(np.linalg.norm(result["quaternion"], axis=-1), 1., atol=1e-5, rtol=0):
        raise ValueError("Recorded link quaternions must be normalized wxyz")
    return result


class WheelKinematics:
    """Original unmerged model FK from measured link pose and COM velocities."""

    ROOT_COM = np.array([.04576, .00014, -.16398])

    def __init__(self, model):
        import mujoco
        self.mj, self.model = mujoco, model
        if model.nq != 15 or model.nv != 14:
            raise ValueError("Require original TRON1 floating base and eight hinge joints")
        root = model.joint("root")
        if (int(root.type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)
                or int(root.qposadr[0]) != 0 or int(root.dofadr[0]) != 0):
            raise ValueError("Unexpected TRON1 root coordinate mapping")
        self.base = int(model.body("base_Link").id)
        if not np.allclose(model.body_ipos[self.base], self.ROOT_COM, atol=2e-7, rtol=0):
            raise ValueError("Merged or changed base COM would corrupt measured velocity reconstruction")
        self.qids = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
        self.vids = np.array([int(model.joint(name).dofadr[0]) for name in JOINT_NAMES])
        self.wheels = np.array([int(model.body(name).id) for name in ("wheel_L_Link", "wheel_R_Link")])
        self.data = mujoco.MjData(model)

    def reconstruct(self, trace, *, wheel_position_tolerance_m=2e-4):
        if not np.isfinite(wheel_position_tolerance_m) or wheel_position_tolerance_m <= 0:
            raise ValueError("Wheel FK tolerance must be finite and positive")
        time = np.asarray(trace["time_s"])
        n = len(time)
        if tuple(trace["joint_names"]) != JOINT_NAMES:
            raise ValueError("Canonical trace must already be in named physical action order")
        shapes = {"position": (n, 3), "quaternion": (n, 4), "com_velocity": (n, 3),
                  "angular_velocity": (n, 3), "joint_pos": (n, 8), "joint_vel": (n, 8), "wheel_pos": (n, 2, 3)}
        for name, shape in shapes.items():
            value = np.asarray(trace[name], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid measured {name} for FK")
        norm = np.linalg.norm(trace["quaternion"], axis=-1)
        if not np.allclose(norm, 1., atol=1e-5, rtol=0):
            raise ValueError("Measured root quaternion is not normalized")
        wheel_quat, wheel_velocity, wheel_position = [], [], []
        max_velocity_error = 0.
        m, d, mj = self.model, self.data, self.mj
        jp, jr = np.empty((3, m.nv)), np.empty((3, m.nv))
        for i in range(n):
            mj.mj_resetData(m, d)
            d.qpos[:3] = trace["position"][i]
            d.qpos[3:7] = trace["quaternion"][i] / norm[i]
            d.qpos[self.qids] = trace["joint_pos"][i]
            d.qvel[self.vids] = trace["joint_vel"][i]
            mj.mj_fwdPosition(m, d)
            # Match the original base link COM, NOT root-origin/subtree velocity.
            mj.mj_jac(m, d, jp, jr, d.xipos[self.base], self.base)
            jac = np.vstack((jp, jr))
            desired = np.r_[trace["com_velocity"][i], trace["angular_velocity"][i]]
            d.qvel[:6] = np.linalg.solve(jac[:, :6], desired - jac @ d.qvel)
            max_velocity_error = max(max_velocity_error, float(np.max(np.abs(jac @ d.qvel - desired))))
            wheel_quat.append(d.xquat[self.wheels].copy())
            wheel_position.append(d.xpos[self.wheels].copy())
            velocity = []
            for body in self.wheels:
                mj.mj_jac(m, d, jp, None, d.xpos[body], int(body))
                velocity.append(jp @ d.qvel)
            wheel_velocity.append(velocity)
        wheel_quat, wheel_velocity, wheel_position = map(np.asarray, (wheel_quat, wheel_velocity, wheel_position))
        error = float(np.max(np.linalg.norm(wheel_position - trace["wheel_pos"], axis=-1)))
        if error > wheel_position_tolerance_m or max_velocity_error > 1e-8:
            raise ValueError(f"Measured/model FK mismatch: wheel position={error}, COM velocity={max_velocity_error}")
        signals = heading_signals(trace["quaternion"], trace["com_velocity"], wheel_quat, wheel_velocity)
        return {"signals": signals, "wheel_quaternion_wxyz": wheel_quat,
                "wheel_center_velocity_w": wheel_velocity,
                "audit": {"maximum_wheel_link_position_error_m": error,
                          "wheel_position_tolerance_m": float(wheel_position_tolerance_m),
                          "maximum_root_com_velocity_reconstruction_error": max_velocity_error,
                          "dynamics_stepping": False, "measured_contact_replaced": False,
                          "wheel_velocity_kind": "analytic link-origin Jacobian; not patch velocity or body COM velocity"}}
