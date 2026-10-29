"""Opt-in planar tracking with vertical smoothness, not reference-height tracking.

This adapter leaves the pinned source and archived policy unchanged. Importing
it does not require Isaac or Torch; runtime imports happen after AppLauncher.
"""
from __future__ import annotations

import copy
import math


MODE = "vertical_acceleration_only"


def validate_settings(settings):
    required = {"mode", "weight", "acceleration_scale_m_s2"}
    if not isinstance(settings, dict) or set(settings) != required or settings["mode"] != MODE:
        raise ValueError("Expected explicit vertical_acceleration_only height regularization")
    for name in ("weight", "acceleration_scale_m_s2"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid height regularization {name}")
    if not -10. <= settings["weight"] < 0. or settings["acceleration_scale_m_s2"] <= 0.:
        raise ValueError("Height smoothness requires a negative weight and positive acceleration scale")
    return copy.deepcopy(settings)


def planar_squared_error(target, actual):
    """Works with NumPy arrays or Torch tensors; Z is deliberately unused."""
    return ((target[..., :2] - actual[..., :2]) ** 2).sum(-1)


def acceleration_cost(current_vz, previous_vz, dt, scale, exp):
    """Bounded cost in [0,1]; constant vertical speed has zero cost.

    This is a control-interval mean acceleration, not instantaneous substep
    acceleration or jerk. No position, height target or zero-vz target is used.
    """
    return 1. - exp(-((current_vz - previous_vz) / (dt * scale)) ** 2)


def anchor_xy_tracking(env, command_name, std):
    import torch
    command = env.command_manager.get_term(command_name)
    return torch.exp(-planar_squared_error(command.anchor_pos_w, command.robot_anchor_pos_w) / std**2)


def body_xy_tracking(env, command_name, std, body_names=None):
    import torch
    from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes
    command = env.command_manager.get_term(command_name)
    ids = _get_body_indexes(command, body_names)
    error = planar_squared_error(command.body_pos_relative_w[:, ids], command.robot_body_pos_w[:, ids])
    return torch.exp(-error.mean(-1) / std**2)


def body_xy_velocity_tracking(env, command_name, std, body_names=None):
    import torch
    from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes
    command = env.command_manager.get_term(command_name)
    ids = _get_body_indexes(command, body_names)
    error = planar_squared_error(command.body_lin_vel_w[:, ids], command.robot_body_lin_vel_w[:, ids])
    return torch.exp(-error.mean(-1) / std**2)


def vertical_acceleration_term():
    """Create an Isaac-managed term with per-environment reset-safe history."""
    import torch
    from isaaclab.managers import ManagerTermBase

    class VerticalAcceleration(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.previous_vz = torch.zeros(env.num_envs, device=env.device)
            self.valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            self.cost = torch.zeros(env.num_envs, device=env.device)
            self.last_step = None

        def reset(self, env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            self.valid[ids] = False
            self.cost[ids] = 0.

        def __call__(self, env, acceleration_scale_m_s2):
            step = env.common_step_counter
            if self.last_step != step:
                # root_lin_vel_w is base_Link COM velocity in world axes.
                vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
                value = acceleration_cost(vz, self.previous_vz, env.step_dt,
                                          acceleration_scale_m_s2, torch.exp)
                self.cost = torch.where(self.valid, value, torch.zeros_like(value))
                self.previous_vz.copy_(vz)
                self.valid[:] = True
                self.last_step = step
            return self.cost

    return VerticalAcceleration


def install_height_smoothness(cfg, settings):
    """Install after make_env_cfg and before constructing the training env."""
    from isaaclab.managers import RewardTermCfg
    settings = validate_settings(settings)
    replacements = {"motion_global_anchor_pos": anchor_xy_tracking,
                    "motion_body_pos": body_xy_tracking,
                    "motion_body_lin_vel": body_xy_velocity_tracking}
    for name in replacements:
        if getattr(cfg.rewards, name, None) is None:
            raise ValueError(f"Missing expected motion reward: {name}")
    for name in ("anchor_pos", "ee_body_pos"):
        if getattr(cfg.terminations, name, None) is None:
            raise ValueError(f"Missing expected height termination: {name}")
    if getattr(cfg.rewards, "drift_wheel_ground_load_reserve", None) is not None:
        raise ValueError("Height-smoothness task excludes the historical cached load-reserve reward")
    for name, func in replacements.items():
        getattr(cfg.rewards, name).func = func
    # Do not silently retain the optional zero-vertical-speed drift penalty.
    if hasattr(cfg.rewards, "drift_wheel_vertical_velocity"):
        cfg.rewards.drift_wheel_vertical_velocity = None
    cfg.terminations.anchor_pos = None
    cfg.terminations.ee_body_pos = None
    cfg.rewards.height_vertical_acceleration = RewardTermCfg(
        func=vertical_acceleration_term(), weight=settings["weight"],
        params={"acceleration_scale_m_s2": settings["acceleration_scale_m_s2"]})
    return {**settings, "absolute_height_reward": False, "height_termination": False,
            "reference_vertical_velocity_reward": False,
            "removed_termination_terms": ["anchor_pos", "ee_body_pos"],
            "position_and_linear_velocity_tracking_axes": ["x", "y"],
            "smoothness_signal": "base_Link COM world vz difference / control dt",
            "reset_history": "first reward sample after each reset is unpenalized",
            "actor_observation_layout": "unchanged; reference-height observation retained, not constrained",
            "retained": ["orientation_tracking", "tilt_safety", "action_rate", "joint_limits", "contact_rewards"]}


def require_height_free_validation(task, training, full_horizon):
    """The historical evaluators still carry Z gates unless diagnostic mode is on."""
    settings = task.get("height_regularization")
    if settings is None:
        return
    settings = validate_settings(settings)
    if not full_horizon:
        raise ValueError("Height-smoothness policies require --full-horizon-diagnostic; historical evaluators otherwise restore Z termination gates")
    recorded = training.get("height_regularization", {})
    if (not all(recorded.get(key) == value for key, value in settings.items())
            or recorded.get("absolute_height_reward") is not False
            or recorded.get("height_termination") is not False
            or not training.get("height_adapter_sha256")):
        raise ValueError("Checkpoint lacks height-smoothness training provenance; do not relabel the archived actor")
