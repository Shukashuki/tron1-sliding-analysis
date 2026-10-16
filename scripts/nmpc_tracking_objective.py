"""NumPy reconstruction of the new height-smooth TRON1 tracking objective.

This is an objective/measurement bridge, not an NMPC solver. Reference frames
and relative-body targets are held BEFORE the physical action interval, as in
IsaacLab's reward-before-command-update ordering. Positions are link origins;
linear velocities are original, unmerged link COM velocities in world axes.
The floating base is measured, never overwritten or directly actuated here.

The stage cost is ``dt * (sum_positive_weights - weighted_reward)``. For a fixed
horizon it differs from negative dt-scaled reward only by a constant. Contact
and soft-limit surrogate differences are explicit; this is not a claim of
bitwise Isaac/MuJoCo reward parity or of robust-control guarantees.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import math

import numpy as np

from height_smoothness import validate_settings


BODY_NAMES = (
    "base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
    "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link",
)
JOINT_NAMES = (
    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
    "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint",
)


def _array(value, shape, name):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return out


def _quaternion(value, shape, name):
    q = _array(value, shape, name)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"{name} must contain normalized wxyz quaternions")
    return q / norms


def quaternion_multiply(a, b):
    """Hamilton multiplication of wxyz quaternions, with broadcasting."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def quaternion_inverse(q):
    q = np.asarray(q)
    return q * np.array([1., -1., -1., -1.])


def quaternion_rotate(q, vector):
    q, vector = np.asarray(q), np.asarray(vector)
    cross = 2. * np.cross(q[..., 1:], vector)
    return vector + q[..., :1] * cross + np.cross(q[..., 1:], cross)


def quaternion_angle(a, b):
    """Shortest SO(3) angle, invariant to the quaternion sign."""
    difference = quaternion_multiply(a, quaternion_inverse(b))
    return 2. * np.arctan2(np.linalg.norm(difference[..., 1:], axis=-1),
                          np.abs(difference[..., 0]))


def yaw_quaternion(q):
    w, x, y, z = np.asarray(q)
    yaw = np.arctan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    return np.array([np.cos(.5*yaw), 0., 0., np.sin(.5*yaw)])


def append_contact_history(history, current, length=3):
    """Return the latest physical-substep force history, without hidden state.

    ``current`` is net world contact force on each tracked body, including
    self-contact. Missing pre-reset history is zero-filled, like reset sensor
    buffers. Call after EACH physics substep, not just each action interval.
    """
    current = np.asarray(current, dtype=float)
    if current.ndim != 2 or current.shape[1] != 3 or not np.isfinite(current).all():
        raise ValueError("current contact forces must have shape (bodies,3)")
    if not isinstance(length, int) or length < 1:
        raise ValueError("contact history length must be positive")
    if history is None:
        history = np.zeros((length, *current.shape))
    history = _array(history, (length, *current.shape), "contact_force_history_w")
    return np.concatenate((history[1:], current[None]), axis=0)


@dataclass(frozen=True)
class HeldReference:
    """The current command, including its pre-action relative-body targets."""

    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    relative_body_pos_w: np.ndarray
    relative_body_quat_w: np.ndarray


class HeightSmoothTrackingObjective:
    def __init__(self, task_profile, body_names=BODY_NAMES, dt=.02,
                 soft_joint_limits=None):
        self.body_names = tuple(body_names)
        if len(set(self.body_names)) != len(self.body_names) or not self.body_names or self.body_names[0] != "base_Link":
            raise ValueError("Unique tracked body names must start with base_Link")
        self.wheels = np.array([self.body_names.index(n) for n in ("wheel_L_Link", "wheel_R_Link")])
        self.nonwheels = np.array([i for i,n in enumerate(self.body_names) if not n.startswith("wheel_")])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.dt = float(dt)
        self.profile = copy.deepcopy(task_profile)
        if self.profile.get("schema_version") != 1 or self.profile.get("task") != "car_drift":
            raise ValueError("Expected schema 1 car_drift height-smooth profile")
        self.smooth = validate_settings(self.profile.get("height_regularization"))
        if "wheel_ground_load_reserve" in self.profile:
            raise ValueError("Height-smooth task excludes historical ground-load reserve")
        drift = self.profile.get("reward_weights", {})
        if set(drift) != {"sideslip", "yaw_rate", "speed", "grounded", "tilt"}:
            raise ValueError("Expected all five explicit drift reward weights")
        self.weights = {
            "motion_global_anchor_pos": 2., "motion_global_anchor_ori": 2.,
            "motion_body_pos": 1., "motion_body_ori": 1.,
            "motion_body_lin_vel": 1., "motion_body_ang_vel": 1.,
            "action_rate_l2": self.profile.get("action_rate_weight", -.1),
            "joint_limit": -10., "undesired_contacts": -.1,
            **{"drift_"+k: float(v) for k,v in drift.items()},
            "height_vertical_acceleration": self.smooth["weight"],
        }
        if "leg_target_rate_weight" in self.profile:
            self.weights["drift_leg_target_rate"] = self.profile["leg_target_rate_weight"]
        if not all(math.isfinite(w) for w in self.weights.values()):
            raise ValueError("All objective weights must be finite")
        if any(drift[k] < 0 for k in drift if k != "tilt") or drift["tilt"] > 0:
            raise ValueError("Drift reward signs differ from the training task")
        self.stds = {"anchor_position_m": .3, "anchor_orientation_rad": .4,
                     "body_position_m": .3, "body_orientation_rad": .4,
                     "body_linear_velocity_m_s": .5, "body_angular_velocity_rad_s": .8,
                     "sideslip_rad": .12, "yaw_rate_rad_s": .5,
                     "speed_m_s": float(self.profile.get("speed_std_m_s", .35))}
        if self.stds["speed_m_s"] <= 0:
            raise ValueError("speed standard deviation must be positive")
        self.soft_joint_limits = (None if soft_joint_limits is None else
                                  self._limits(soft_joint_limits))
        self.reward_upper_bound = sum(w for w in self.weights.values() if w > 0)

    @staticmethod
    def _limits(value):
        limits = _array(value, (8, 2), "soft_joint_pos_limits")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Soft joint lower limits exceed upper limits")
        return limits.copy()

    def contract(self):
        return {
            "kind": "numpy_height_smooth_tracking_objective",
            "body_names": list(self.body_names), "joint_names": list(JOINT_NAMES),
            "dt_s": self.dt, "weights": dict(self.weights), "stds": dict(self.stds),
            "height_regularization": copy.deepcopy(self.smooth),
            "absolute_height_target": False, "reference_vertical_velocity_target": False,
            "sideslip_target": "held reference body-frame course angle, NOT zero",
            "reference_order": "Freeze reference[k] and relative-body targets at pre-action state; evaluate after decimation, before command advances.",
            "pose_semantics": "World link-origin position and link wxyz quaternion",
            "velocity_semantics": "World original-unmerged-link COM linear velocity and world angular velocity",
            "cost": "dt * (sum_positive_reward_weights - sum(weight * raw_term)); no terminal term supplied",
            "contact_thresholds_n": {"grounded_each_wheel_net_force": 5., "undesired_contact_body_force": 1.},
            "joint_limit_joints": list(JOINT_NAMES[:6]),
            "surrogate_differences": [
                "MuJoCo net contact forces are not identical to PhysX ContactSensor forces; all contacts, including self-contact, are included.",
                "undesired_contacts uses max norm over supplied physical-substep history; omitted history is explicitly an instantaneous surrogate.",
                "MuJoCo-derived 0.95 joint ranges are an approximation unless replaced by Isaac runtime soft_joint_pos_limits in matching joint order.",
                "Body COM offsets/inertias must match the original unmerged USD links; random COM offsets require the same realized model parameters.",
                "Task tilt safety termination and policy action/observation privileges are outside this stage objective.",
            ],
        }

    def _body_state(self, state):
        if state.get("body_names") is not None and tuple(state["body_names"]) != self.body_names:
            raise ValueError("State body order differs from objective body order")
        n = len(self.body_names)
        return (_array(state["body_pos_w"], (n,3), "body_pos_w"),
                _quaternion(state["body_quat_w"], (n,4), "body_quat_w"),
                _array(state["body_lin_vel_w"], (n,3), "body_lin_vel_w"),
                _array(state["body_ang_vel_w"], (n,3), "body_ang_vel_w"))

    def hold_reference(self, reference: Mapping, pre_action_state: Mapping):
        positions, quaternions, velocity, angular = self._body_state(reference)
        robot_positions, robot_quaternions, _, _ = self._body_state(pre_action_state)
        delta_quat = yaw_quaternion(quaternion_multiply(robot_quaternions[0],
                                                       quaternion_inverse(quaternions[0])))
        origin = robot_positions[0].copy()
        origin[2] = positions[0,2]
        relative_positions = origin + quaternion_rotate(delta_quat, positions-positions[0])
        relative_quaternions = quaternion_multiply(delta_quat, quaternions)
        return HeldReference(positions.copy(), quaternions.copy(), velocity.copy(), angular.copy(),
                             relative_positions, relative_quaternions)

    def evaluate(self, state: Mapping, reference: HeldReference, action,
                 previous_action, previous_base_com_vz, *, reset_sample=False):
        if not isinstance(reference, HeldReference):
            raise TypeError("Use hold_reference with the PRE-action state before evaluating")
        p, q, velocity, angular = self._body_state(state)
        action = _array(action, (8,), "action")
        previous_action = _array(previous_action, (8,), "previous_action")
        if np.max(np.abs(action)) > 1.+1e-9 or np.max(np.abs(previous_action)) > 1.+1e-9:
            raise ValueError("Objective requires clipped, applied normalized actions")
        n = len(self.body_names)
        forces = _array(state["body_net_forces_w"], (n,3), "body_net_forces_w")
        history = state.get("contact_force_history_w")
        surrogate = []
        if history is None:
            history = forces[None]
            surrogate.append("instantaneous_undesired_contacts_without_sensor_history")
        else:
            history = np.asarray(history, dtype=float)
            if history.shape != (3,n,3) or not np.isfinite(history).all():
                raise ValueError("Contact sensor history must have shape (3,bodies,3)")
        limits = self.soft_joint_limits
        if limits is None:
            limits = self._limits(state["soft_joint_pos_limits"])
            if state.get("soft_joint_limits_source") != "isaac_runtime":
                surrogate.append("mujoco_or_unspecified_soft_joint_limits")
        joints = _array(state["joint_pos"], (8,), "joint_pos")
        joint_error = np.maximum(limits[:,0]-joints, 0.) + np.maximum(joints-limits[:,1], 0.)
        body_error = np.mean(np.sum((reference.relative_body_pos_w[:,:2]-p[:,:2])**2, axis=-1))
        relative_angle = quaternion_angle(reference.relative_body_quat_w[self.nonwheels], q[self.nonwheels])
        body_v_error = np.mean(np.sum((reference.body_lin_vel_w[:,:2]-velocity[:,:2])**2,axis=-1))
        body_w_error = np.mean(np.sum((reference.body_ang_vel_w[self.nonwheels]-angular[self.nonwheels])**2,axis=-1))
        actual_body_v = quaternion_rotate(quaternion_inverse(q[0]), velocity[0])
        target_body_v = quaternion_rotate(quaternion_inverse(reference.body_quat_w[0]), reference.body_lin_vel_w[0])
        beta, target_beta = np.arctan2(actual_body_v[1],actual_body_v[0]), np.arctan2(target_body_v[1],target_body_v[0])
        beta_error = np.arctan2(np.sin(beta-target_beta), np.cos(beta-target_beta))
        speed, target_speed = np.linalg.norm(velocity[0,:2]), np.linalg.norm(reference.body_lin_vel_w[0,:2])
        gravity_b = quaternion_rotate(quaternion_inverse(q[0]), np.array([0.,0.,-1.]))
        if reset_sample:
            acceleration_cost = 0.
        else:
            if previous_base_com_vz is None or not np.isfinite(previous_base_com_vz):
                raise ValueError("Non-reset reward requires previous_base_com_vz")
            scaled_acceleration = (velocity[0,2]-previous_base_com_vz)/(self.dt*self.smooth["acceleration_scale_m_s2"])
            acceleration_cost = -np.expm1(-scaled_acceleration**2)
        raw = {
            "motion_global_anchor_pos": np.exp(-np.sum((reference.body_pos_w[0,:2]-p[0,:2])**2)/self.stds["anchor_position_m"]**2),
            "motion_global_anchor_ori": np.exp(-quaternion_angle(reference.body_quat_w[0],q[0])**2/self.stds["anchor_orientation_rad"]**2),
            "motion_body_pos": np.exp(-body_error/self.stds["body_position_m"]**2),
            "motion_body_ori": np.exp(-np.mean(relative_angle**2)/self.stds["body_orientation_rad"]**2),
            "motion_body_lin_vel": np.exp(-body_v_error/self.stds["body_linear_velocity_m_s"]**2),
            "motion_body_ang_vel": np.exp(-body_w_error/self.stds["body_angular_velocity_rad_s"]**2),
            "action_rate_l2": np.sum((action-previous_action)**2),
            "joint_limit": np.sum(joint_error[:6]),
            "undesired_contacts": np.sum(np.max(np.linalg.norm(history[:,self.nonwheels],axis=-1),axis=0)>1.),
            "drift_sideslip": np.exp(-(beta_error/self.stds["sideslip_rad"])**2)*float(target_speed>.3)*float(speed>.25),
            "drift_speed": np.exp(-((speed-target_speed)/self.stds["speed_m_s"])**2),
            "drift_yaw_rate": np.exp(-((angular[0,2]-reference.body_ang_vel_w[0,2])/self.stds["yaw_rate_rad_s"])**2),
            "drift_grounded": float(np.all(np.linalg.norm(forces[self.wheels],axis=-1)>5.)),
            "drift_tilt": np.sum(gravity_b[:2]**2),
            "height_vertical_acceleration": acceleration_cost,
        }
        if "drift_leg_target_rate" in self.weights:
            raw["drift_leg_target_rate"] = np.sum(((action[:6]-previous_action[:6])/self.dt)**2)
        raw = {name:float(value) for name,value in raw.items()}
        weighted = {name:self.weights[name]*value for name,value in raw.items()}
        reward = sum(weighted.values())
        return {"raw_terms":raw, "weighted_terms":weighted, "total_reward":reward,
                "reward_upper_bound":self.reward_upper_bound,
                "stage_cost":self.dt*(self.reward_upper_bound-reward),
                "base_com_vz":float(velocity[0,2]),
                "diagnostics":{"beta_rad":float(beta), "target_beta_rad":float(target_beta),
                               "beta_error_rad":float(beta_error), "speed_m_s":float(speed),
                               "target_speed_m_s":float(target_speed)},
                "surrogate_flags":surrogate}


class MujocoObjectiveState:
    """Cache IDs/Jacobian storage; extract, never integrate or correct a state."""

    def __init__(self, model, body_names=BODY_NAMES, joint_names=JOINT_NAMES,
                 *, soft_joint_limits=None):
        import mujoco
        self.mj, self.model = mujoco, model
        self.body_names, self.joint_names = tuple(body_names), tuple(joint_names)
        if self.joint_names != JOINT_NAMES:
            raise ValueError("Objective joints must use the established action order")
        self.bids = np.array([model.body(n).id for n in self.body_names])
        self.body_to_index = {int(b):i for i,b in enumerate(self.bids)}
        self.qids = np.array([int(model.joint(n).qposadr[0]) for n in self.joint_names])
        self.jids = np.array([model.joint(n).id for n in self.joint_names])
        self.jp = np.empty((3,model.nv)); self.jr = np.empty((3,model.nv))
        self.contact_wrench = np.empty(6)
        if soft_joint_limits is None:
            limits = model.jnt_range[self.jids].copy()
            midpoint = limits.mean(axis=1)
            half_range = .475*(limits[:,1]-limits[:,0])
            self.soft_limits = np.column_stack((midpoint-half_range,midpoint+half_range))
            # Unbounded joints should never acquire an artificial range cost.
            self.soft_limits[~model.jnt_limited[self.jids].astype(bool)] = [-np.finfo(float).max,np.finfo(float).max]
            self.soft_limits_source = "mujoco_ranges_times_0.95_surrogate"
        else:
            self.soft_limits = HeightSmoothTrackingObjective._limits(soft_joint_limits)
            self.soft_limits_source = "isaac_runtime"

    def contact_forces(self, data):
        forces = np.zeros((len(self.bids),3))
        for i,contact in enumerate(data.contact):
            self.mj.mj_contactForce(self.model,data,i,self.contact_wrench)
            world_force = contact.frame.reshape(3,3).T@self.contact_wrench[:3]
            for geom, sign in ((contact.geom1,-1.),(contact.geom2,1.)):
                index = self.body_to_index.get(int(self.model.geom_bodyid[geom]))
                if index is not None:
                    forces[index] += sign*world_force
        return forces

    def update_contact_history(self, data, history=None):
        current = self.contact_forces(data)
        return append_contact_history(history,current), current

    def extract(self, data, *, contact_force_history=None, body_net_forces_w=None):
        velocity, angular = [], []
        for bid in self.bids:
            # xipos is the world COM of this specific link, not subtree COM.
            self.mj.mj_jac(self.model,data,self.jp,self.jr,data.xipos[bid],int(bid))
            velocity.append(self.jp@data.qvel); angular.append(self.jr@data.qvel)
        state = {"body_names":self.body_names,
                 "body_pos_w":data.xpos[self.bids].copy(),
                 "body_quat_w":data.xquat[self.bids].copy(),
                 "body_lin_vel_w":np.asarray(velocity),
                 "body_ang_vel_w":np.asarray(angular),
                 "joint_pos":data.qpos[self.qids].copy(),
                 "body_net_forces_w":self.contact_forces(data) if body_net_forces_w is None else np.asarray(body_net_forces_w).copy(),
                 "soft_joint_pos_limits":self.soft_limits.copy(),
                 "soft_joint_limits_source":self.soft_limits_source}
        if contact_force_history is not None:
            state["contact_force_history_w"] = np.asarray(contact_force_history).copy()
        return state
