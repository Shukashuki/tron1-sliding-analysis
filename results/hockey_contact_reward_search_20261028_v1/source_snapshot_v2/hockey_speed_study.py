"""Designed speed scaling and paired, explicit initial-state trials.

Scale world translation only. GMR joint/posture clocks, height, wheel axes and
angular velocities stay unchanged. These are designed, not measured, speeds.
"""
from __future__ import annotations

import numpy as np


def scale_reference(motion, diagnostics, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Speed scale must be positive and finite")
    m = {k: np.array(v, copy=True) for k, v in motion.items()}
    d = {k: np.array(v, copy=True) for k, v in diagnostics.items()}
    root = diagnostics["qpos"][:, :2]
    delta = (scale-1)*(root-root[0])
    delta_v = (scale-1)*diagnostics["root_link_velocity_w_m_s"][:, :2]
    m["body_pos_w"][..., :2] += delta[:, None]
    m["body_lin_vel_w"][..., :2] += delta_v[:, None]
    d["qpos"][:, :2] += delta
    d["root_link_velocity_w_m_s"][:, :2] += delta_v
    d["designed_root_link_speed_m_s"] *= scale
    return m, d


def evaluation_cases(seed=20261028, count=32):
    """One exact nominal case followed by IID, paired small reset perturbations."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("Positive random trial count required")
    fields = ("x_m", "y_m", "yaw_rad", "speed_fraction", "lateral_m_s", "yaw_rate_rad_s")
    bounds = np.array([.02, .02, np.deg2rad(2.), .05, .05, .1])
    values = np.random.default_rng(seed).uniform(-1, 1, (count, len(fields)))*bounds
    rows = [dict(id="nominal", kind="nominal", **dict.fromkeys(fields, 0.))]
    rows += [dict(id=f"trial_{i:03d}", kind="perturbed", **dict(zip(fields, map(float, row))))
             for i, row in enumerate(values)]
    return rows


def perturbed_root(pose, velocity, case):
    """World base-link pose (wxyz), world COM linear/angular velocity.

    A world-Z rotation preserves all body heights on the horizontal floor.
    Reference commands are not shifted/rotated with the perturbed robot.
    """
    pose = np.asarray(pose, dtype=float).copy()
    velocity = np.asarray(velocity, dtype=float).copy()
    if pose.shape != (7,) or velocity.shape != (6,):
        raise ValueError("Require a seven-field pose and six-field COM velocity")
    fields = ("x_m", "y_m", "yaw_rad", "speed_fraction", "lateral_m_s", "yaw_rate_rad_s")
    if not np.isfinite([case[k] for k in fields]).all():
        raise ValueError("Nonfinite reset perturbation")
    pose[:2] += [case["x_m"], case["y_m"]]
    c, s = np.cos(case["yaw_rad"]/2), np.sin(case["yaw_rad"]/2)
    w, x, y, z = pose[3:]
    pose[3:] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
    speed = np.linalg.norm(velocity[:2])
    if speed <= 0:
        raise ValueError("A moving initial reference is required")
    lateral = np.array([-velocity[1], velocity[0]])/speed
    velocity[:2] += case["speed_fraction"]*velocity[:2]+case["lateral_m_s"]*lateral
    velocity[5] += case["yaw_rate_rad_s"]
    return pose, velocity


def wilson_interval(successes, trials, z=1.959963984540054):
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid binomial counts")
    p = successes/trials
    denom = 1+z*z/trials
    center = (p+z*z/(2*trials))/denom
    half = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denom
    return [max(0., float(center-half)), min(1., float(center+half))]
