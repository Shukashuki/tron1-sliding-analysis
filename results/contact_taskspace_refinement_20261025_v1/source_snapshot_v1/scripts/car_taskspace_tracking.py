"""Car-reference objectives without artificial intermediate-leg imitation.

This changes four tracking terms, not the contact formula or task acceptance.
Wheel heading is axial, invariant to wheel spin, and world-referenced. Root
orientation and directed beta retain their original separate objectives.
"""
from __future__ import annotations

import copy
import importlib

SETTINGS = {
    "mode": "base_wheel_targets_free_leg_links_v1",
    "position_bodies": ["base_Link", "wheel_L_Link", "wheel_R_Link"],
    "linear_velocity_bodies": ["base_Link", "wheel_L_Link", "wheel_R_Link"],
    "angular_velocity_bodies": ["base_Link"],
    "orientation_term": "world_axial_wheel_plane_heading",
    "wheel_bodies": ["wheel_L_Link", "wheel_R_Link"],
    "wheel_axle_local": [0., 1., 0.],
    "wheel_heading_std_rad": .4,
    "wheel_heading_min_projection": 1e-6,
    "all_reward_weights_unchanged": True,
    "root_global_orientation_and_nonzero_beta_retained": True,
    "intermediate_leg_reference_observations_retained": True,
    "contact_reward_formula_changed": False,
}
CHANGED = ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")


def settings(value=None):
    if value is not None and value != SETTINGS:
        raise ValueError("Require the explicit task-space target-selection protocol")
    return copy.deepcopy(SETTINGS)


def heading_quality(reference_quat, actual_quat, std=.4):
    """Torch batch (..., two wheels, WXYZ); no host transfers or wheel spin.

    Y is the authored wheel joint axis. Horizontal rolling direction is
    (axle_y,-axle_x). Opposite axle signs describe the same wheel plane.
    Singular horizontal projections receive zero reward. Nonfinite physical
    inputs remain nonfinite so the trainer rejects them rather than hiding it.
    """
    import torch
    if std != SETTINGS["wheel_heading_std_rad"]:
        raise ValueError("The task-space wheel heading scale is frozen at .4 rad")
    headings = []
    valid = []
    for q in (reference_quat, actual_quat):
        length = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        w, x, y, z = (q / length.clamp_min(1e-12)).unbind(-1)
        h = torch.stack((1-2*(x*x+z*z), -2*(x*y-w*z)), dim=-1)
        hn = torch.linalg.vector_norm(h, dim=-1)
        headings.append(h / hn.clamp_min(1e-12).unsqueeze(-1))
        valid.append((length[..., 0] > 1e-6) & (hn > SETTINGS["wheel_heading_min_projection"]))
    a, b = headings
    dot = (a*b).sum(-1)
    cross = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    error = torch.atan2(cross.abs(), dot.abs())
    quality = torch.exp(-error.square().mean(-1)/(std*std))
    quality = quality * (valid[0] & valid[1]).all(-1)
    return quality


def wheel_plane_heading_tracking(env, command_name, std, body_names):
    if body_names != SETTINGS["wheel_bodies"]:
        raise ValueError("Require both explicitly named wheel bodies")
    command = env.command_manager.get_term(command_name)
    ids = [command.cfg.body_names.index(n) for n in body_names]
    return heading_quality(command.body_quat_w[:, ids], command.robot_body_quat_w[:, ids], std)


def configure(cfg, value=None):
    settings(value)
    terms = cfg.rewards
    before = {name: copy.deepcopy(getattr(terms, name).params) for name in CHANGED}
    assert all(getattr(terms, name).weight == 1. for name in CHANGED)
    # Require the already installed XY-only height adapter. No Z target added.
    height = importlib.import_module("height_smoothness")
    assert terms.motion_body_pos.func is height.body_xy_tracking
    assert terms.motion_body_lin_vel.func is height.body_xy_velocity_tracking
    assert before["motion_body_pos"].get("body_names") is None
    assert before["motion_body_lin_vel"].get("body_names") is None
    expected = ["base_Link", "abad_L_Link", "hip_L_Link", "abad_R_Link", "hip_R_Link", "knee_L_Link", "knee_R_Link"]
    assert before["motion_body_ori"]["body_names"] == expected
    assert before["motion_body_ang_vel"]["body_names"] == expected
    assert before["motion_body_ori"]["std"] == .4
    terms.motion_body_pos.params["body_names"] = SETTINGS["position_bodies"][:]
    terms.motion_body_lin_vel.params["body_names"] = SETTINGS["linear_velocity_bodies"][:]
    terms.motion_body_ang_vel.params["body_names"] = SETTINGS["angular_velocity_bodies"][:]
    terms.motion_body_ori.func = wheel_plane_heading_tracking
    terms.motion_body_ori.params = {"command_name": "motion", "std": .4,
                                     "body_names": SETTINGS["wheel_bodies"][:]}
    return {"settings": settings(), "original_tracking_params": before,
            "four_tracking_terms_changed": list(CHANGED),
            "contact_reward_and_other_eleven_terms_unchanged": True}


def runtime_audit(env, configured):
    import torch
    expected_names = {
        "motion_body_pos": SETTINGS["position_bodies"],
        "motion_body_lin_vel": SETTINGS["linear_velocity_bodies"],
        "motion_body_ang_vel": SETTINGS["angular_velocity_bodies"],
        "motion_body_ori": SETTINGS["wheel_bodies"],
    }
    terms = {n: env.reward_manager.get_term_cfg(n) for n in env.reward_manager.active_terms}
    assert len(terms) == 15
    for n, names in expected_names.items():
        assert terms[n].params["body_names"] == names and terms[n].weight == 1.
    assert terms["motion_body_ori"].func is wheel_plane_heading_tracking
    assert terms["motion_global_anchor_ori"].weight == 2.
    assert terms["motion_global_anchor_pos"].weight == 2.
    assert terms["drift_sideslip"].weight == 2. and terms["drift_grounded"].weight == .4
    assert terms["action_rate_l2"].weight == -.5
    assert env.cfg.terminations.anchor_pos is None and env.cfg.terminations.ee_body_pos is None
    value = wheel_plane_heading_tracking(env, **terms["motion_body_ori"].params)
    assert value.shape == (env.num_envs,) and value.device == torch.device(env.device)
    assert bool(torch.isfinite(value).all()) and bool(((value >= 0.) & (value <= 1.)).all())
    return {"status": "live_target_selection_verified", "configured": copy.deepcopy(configured),
            "active_term_weights": {n: float(t.weight) for n, t in terms.items()},
            "active_tracking_params": {n: copy.deepcopy(terms[n].params) for n in CHANGED},
            "live_heading_reward_range": [float(value.min()), float(value.max())],
            "live_heading_reward_shape": list(value.shape), "num_envs": env.num_envs,
            "extra_simulation_steps": 0, "absolute_height_gate": False,
            "observation_layout_changed": False, "task_success_claimed": False}
