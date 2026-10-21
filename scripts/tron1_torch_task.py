"""Device-resident TRON1 observations and height-smooth tracking objective.

All hot-path arithmetic is batched Torch: no scalar reads, NumPy conversion,
host copies, RNG changes or global Torch settings. State positions are original
link origins and velocities are original, unmerged link COM world velocities.
The public helpers preserve tensor dtype/device (the runner chooses float32).
Validation of finite physical states belongs to the caller's batched health
check, rather than introducing a synchronization point in every reward term.
"""
from __future__ import annotations

import copy
import math

import torch

from height_smoothness import validate_settings


BODY_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def quaternion_inverse(q):
    """Conjugate a unit wxyz quaternion, retaining the input dtype/device."""
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quaternion_multiply(a, b):
    """Broadcast Hamilton multiplication of wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def quaternion_rotate(q, vector):
    """Broadcast rotation by unit quaternions, including shared vectors."""
    qvector, vector = torch.broadcast_tensors(q[..., 1:], vector)
    cross = 2.*torch.linalg.cross(qvector, vector, dim=-1)
    return vector + q[..., :1]*cross + torch.linalg.cross(qvector, cross, dim=-1)


def quaternion_angle(a, b):
    """Shortest SO(3) angle, invariant to the quaternion's sign."""
    difference = quaternion_multiply(a, quaternion_inverse(b))
    return 2.*torch.atan2(torch.linalg.vector_norm(difference[..., 1:], dim=-1),
                         torch.abs(difference[..., 0]))


def rotation_matrix(q):
    """wxyz rotation matrices, with the original row-major column slicing."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)),
                       dim=-1).reshape((*q.shape[:-1], 3, 3))


def build_observations(state, reference, phases, q0s, previous_actions, frames,
                       *, encoder_bias=None, critic_q0s=None):
    """Exact noise-free actor51/critic120 without a device/host round trip.

    ``q0s`` is the realized motor neutral position, including the already
    applied ``-encoder_bias`` from mjlab. Optional ``encoder_bias`` therefore
    corrects only the critic's six joint-position observations; actor bias is
    already represented by ``joint_pos - q0s``. Alternatively ``critic_q0s``
    explicitly supplies the privileged unbiased neutral positions.
    """
    if frames < 2:
        raise ValueError("Reference must contain at least two frames")
    if encoder_bias is not None and critic_q0s is not None:
        raise ValueError("Choose encoder_bias or critic_q0s, not both")
    p, q, v, w = (state[key] for key in BODY_FIELDS)
    batch = p.shape[0]
    root_inv = quaternion_inverse(q[:, 0])
    phase = (2*math.pi/(frames-1))*phases.to(dtype=p.dtype)
    relative_q = quaternion_multiply(root_inv, reference["body_quat_w"][:, 0])
    actor = torch.cat((reference["joint_pos"][:, :6], reference["joint_vel"][:, :6],
                       torch.sin(phase)[:, None], torch.cos(phase)[:, None],
                       quaternion_rotate(root_inv, reference["body_pos_w"][:, 0]-p[:, 0]),
                       rotation_matrix(relative_q)[..., :2].reshape(batch, 6),
                       quaternion_rotate(root_inv, v[:, 0]),
                       quaternion_rotate(root_inv, w[:, 0]),
                       state["joint_pos"][:, :6]-q0s[:, :6],
                       state["joint_vel"], previous_actions), dim=-1)
    relative_p = quaternion_rotate(root_inv[:, None], p-p[:, :1])
    nonwheel_q = quaternion_multiply(root_inv[:, None], q[:, :7])
    critic_actor = actor
    if encoder_bias is not None or critic_q0s is not None:
        privileged_joint = (actor[:, 29:35]-encoder_bias[:, :6]
                            if encoder_bias is not None else
                            state["joint_pos"][:, :6]-critic_q0s[:, :6])
        critic_actor = torch.cat((actor[:, :29], privileged_joint, actor[:, 35:]), dim=-1)
    critic = torch.cat((critic_actor, relative_p.reshape(batch, 27),
                        rotation_matrix(nonwheel_q)[..., :2].reshape(batch, 42)), dim=-1)
    if actor.shape != (batch, 51) or critic.shape != (batch, 120):
        raise ValueError("TRON1 observation layout requires nine bodies and eight joints")
    return {"policy": actor, "critic": critic}


def _normalized(q):
    return q/torch.linalg.vector_norm(q, dim=-1, keepdim=True)


class TorchHeightSmoothObjective:
    """Exact (B,...) Torch counterpart of the existing 15-term NumPy objective.

    ``total_reward`` is the unscaled sum of weighted terms. The PPO environment
    applies its control dt exactly once, as in the original manager-based task.
    Grounded uses the last physical-substep net force, while undesired contact
    uses the last three physical-substep force norms, including self-contact.
    """

    def __init__(self, profile, soft_limits, *, device=None, dtype=None, dt=.02):
        self.profile = copy.deepcopy(profile)
        if profile.get("schema_version") != 1 or profile.get("task") != "car_drift":
            raise ValueError("Expected schema 1 car_drift height-smooth profile")
        if "wheel_ground_load_reserve" in profile:
            raise ValueError("Height-smooth task excludes historical ground-load reserve")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.dt = float(dt)
        self.smooth = validate_settings(profile.get("height_regularization"))
        drift = profile.get("reward_weights", {})
        if set(drift) != {"sideslip", "yaw_rate", "speed", "grounded", "tilt"}:
            raise ValueError("Expected all five explicit drift reward weights")
        if any(drift[key] < 0 for key in drift if key != "tilt") or drift["tilt"] > 0:
            raise ValueError("Drift reward signs differ from the training task")
        self.weights = {
            "motion_global_anchor_pos": 2., "motion_global_anchor_ori": 2.,
            "motion_body_pos": 1., "motion_body_ori": 1.,
            "motion_body_lin_vel": 1., "motion_body_ang_vel": 1.,
            "action_rate_l2": profile.get("action_rate_weight", -.1),
            "joint_limit": -10., "undesired_contacts": -.1,
            **{"drift_"+key: float(value) for key, value in drift.items()},
            "height_vertical_acceleration": self.smooth["weight"],
        }
        if "leg_target_rate_weight" in profile:
            self.weights["drift_leg_target_rate"] = profile["leg_target_rate_weight"]
        if not all(math.isfinite(value) for value in self.weights.values()):
            raise ValueError("All objective weights must be finite")
        self.stds = {"anchor_position_m": .3, "anchor_orientation_rad": .4,
                     "body_position_m": .3, "body_orientation_rad": .4,
                     "body_linear_velocity_m_s": .5, "body_angular_velocity_rad_s": .8,
                     "sideslip_rad": .12, "yaw_rate_rad_s": .5,
                     "speed_m_s": float(profile.get("speed_std_m_s", .35))}
        if not math.isfinite(self.stds["speed_m_s"]) or self.stds["speed_m_s"] <= 0:
            raise ValueError("speed standard deviation must be positive and finite")
        self.soft_limits = torch.as_tensor(soft_limits, device=device, dtype=dtype)
        if self.soft_limits.shape != (8, 2):
            raise ValueError("Soft joint limits must have shape (8,2)")
        # The only tensor checks in this module are constructor-time checks.
        if (not bool(torch.isfinite(self.soft_limits).all())
                or bool((self.soft_limits[:, 0] > self.soft_limits[:, 1]).any())):
            raise ValueError("Soft joint limits must be finite and ordered")
        self.gravity = self.soft_limits.new_tensor([0., 0., -1.])
        self.reward_upper_bound = sum(value for value in self.weights.values() if value > 0)

    def hold_reference(self, reference, pre_action_state):
        """Freeze the current reference against the PRE-action base pose."""
        p = reference["body_pos_w"]
        q = _normalized(reference["body_quat_w"])
        robot_p = pre_action_state["body_pos_w"]
        robot_q = _normalized(pre_action_state["body_quat_w"])
        difference = quaternion_multiply(robot_q[:, 0], quaternion_inverse(q[:, 0]))
        w, x, y, z = difference.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        zero = torch.zeros_like(yaw)
        delta_q = torch.stack((torch.cos(.5*yaw), zero, zero, torch.sin(.5*yaw)), dim=-1)
        origin = torch.cat((robot_p[:, :1, :2], p[:, :1, 2:]), dim=-1)
        return {
            "body_pos_w": p.clone(), "body_quat_w": q.clone(),
            "body_lin_vel_w": reference["body_lin_vel_w"].clone(),
            "body_ang_vel_w": reference["body_ang_vel_w"].clone(),
            "relative_body_pos_w": origin+quaternion_rotate(delta_q[:, None], p-p[:, :1]),
            "relative_body_quat_w": quaternion_multiply(delta_q[:, None], q),
        }

    def evaluate(self, state, held, actions, previous_actions, previous_vz,
                 first_reward, contact_substeps=None):
        p, q, velocity, angular = (state[key] for key in BODY_FIELDS)
        q = _normalized(q)
        target_p, target_q = held["body_pos_w"], _normalized(held["body_quat_w"])
        target_v, target_w = held["body_lin_vel_w"], held["body_ang_vel_w"]
        relative_p = held["relative_body_pos_w"]
        relative_q = _normalized(held["relative_body_quat_w"])
        if contact_substeps is None:
            forces = state["body_net_forces_w"]
            history = state.get("contact_force_history_w")
            history = forces[:, None] if history is None else history
        else:
            forces = contact_substeps[:, -1]
            history = contact_substeps[:, -3:]
        joint_error = (torch.clamp_min(self.soft_limits[:, 0]-state["joint_pos"], 0.)
                       + torch.clamp_min(state["joint_pos"]-self.soft_limits[:, 1], 0.))
        body_error = ((relative_p[..., :2]-p[..., :2])**2).sum(-1).mean(-1)
        relative_angle = quaternion_angle(relative_q[:, :7], q[:, :7])
        body_v_error = ((target_v[..., :2]-velocity[..., :2])**2).sum(-1).mean(-1)
        body_w_error = ((target_w[:, :7]-angular[:, :7])**2).sum(-1).mean(-1)
        actual_body_v = quaternion_rotate(quaternion_inverse(q[:, 0]), velocity[:, 0])
        target_body_v = quaternion_rotate(quaternion_inverse(target_q[:, 0]), target_v[:, 0])
        beta = torch.atan2(actual_body_v[:, 1], actual_body_v[:, 0])
        target_beta = torch.atan2(target_body_v[:, 1], target_body_v[:, 0])
        beta_error = torch.atan2(torch.sin(beta-target_beta), torch.cos(beta-target_beta))
        speed = torch.linalg.vector_norm(velocity[:, 0, :2], dim=-1)
        target_speed = torch.linalg.vector_norm(target_v[:, 0, :2], dim=-1)
        gravity_b = quaternion_rotate(quaternion_inverse(q[:, 0]), self.gravity)
        if previous_vz is None:
            previous_vz = torch.zeros_like(velocity[:, 0, 2])
        delta_vz = torch.where(first_reward, torch.zeros_like(previous_vz),
                              velocity[:, 0, 2]-previous_vz)
        scaled_acceleration = delta_vz/(self.dt*self.smooth["acceleration_scale_m_s2"])
        acceleration_cost = -torch.expm1(-scaled_acceleration**2)
        raw = {
            "motion_global_anchor_pos": torch.exp(-((target_p[:, 0, :2]-p[:, 0, :2])**2).sum(-1)/self.stds["anchor_position_m"]**2),
            "motion_global_anchor_ori": torch.exp(-quaternion_angle(target_q[:, 0], q[:, 0])**2/self.stds["anchor_orientation_rad"]**2),
            "motion_body_pos": torch.exp(-body_error/self.stds["body_position_m"]**2),
            "motion_body_ori": torch.exp(-(relative_angle**2).mean(-1)/self.stds["body_orientation_rad"]**2),
            "motion_body_lin_vel": torch.exp(-body_v_error/self.stds["body_linear_velocity_m_s"]**2),
            "motion_body_ang_vel": torch.exp(-body_w_error/self.stds["body_angular_velocity_rad_s"]**2),
            "action_rate_l2": ((actions-previous_actions)**2).sum(-1),
            "joint_limit": joint_error[:, :6].sum(-1),
            "undesired_contacts": (torch.linalg.vector_norm(history[:, :, :7], dim=-1).amax(dim=1)>1.).to(dtype=p.dtype).sum(-1),
            "drift_sideslip": torch.exp(-(beta_error/self.stds["sideslip_rad"])**2)*(target_speed>.3)*(speed>.25),
            "drift_speed": torch.exp(-((speed-target_speed)/self.stds["speed_m_s"])**2),
            "drift_yaw_rate": torch.exp(-((angular[:, 0, 2]-target_w[:, 0, 2])/self.stds["yaw_rate_rad_s"])**2),
            "drift_grounded": (torch.linalg.vector_norm(forces[:, 7:9], dim=-1)>5.).all(-1).to(dtype=p.dtype),
            "drift_tilt": (gravity_b[:, :2]**2).sum(-1),
            "height_vertical_acceleration": acceleration_cost,
        }
        if "drift_leg_target_rate" in self.weights:
            raw["drift_leg_target_rate"] = (((actions[:, :6]-previous_actions[:, :6])/self.dt)**2).sum(-1)
        weighted = {name: self.weights[name]*value for name, value in raw.items()}
        total = sum(weighted.values())
        return {"total_reward": total, "raw_terms": raw, "weighted_terms": weighted,
                "stage_cost": self.dt*(self.reward_upper_bound-total),
                "base_com_vz": velocity[:, 0, 2].clone(),
                "diagnostics": {"beta_rad": beta, "target_beta_rad": target_beta,
                                "beta_error_rad": beta_error, "speed_m_s": speed,
                                "target_speed_m_s": target_speed}}
