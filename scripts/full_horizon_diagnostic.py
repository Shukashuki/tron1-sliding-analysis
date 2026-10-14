"""Observe pose termination predicates while retaining a full finite rollout.

This helper is installed only in an isolated evaluation source snapshot.
Motion-end/time limits stay active; training is unaffected.
"""
from functools import wraps

import numpy as np


POSE_TERMS = ("anchor_pos", "anchor_ori", "ee_body_pos")


def summarize_flags(times, flags):
    times = np.asarray(times, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    if times.ndim != 1 or flags.shape != (len(times), len(POSE_TERMS)):
        raise ValueError("Pose diagnostic times/flags are not aligned")
    return {name: {"active_policy_steps": int(flags[:, j].sum()),
                   "first_trigger_time_s": float(times[np.flatnonzero(flags[:, j])[0]])
                   if flags[:, j].any() else None,
                   "active_fraction": float(flags[:, j].mean()) if len(times) else 0.}
            for j, name in enumerate(POSE_TERMS)}


def reset_isaac_diagnostics(env):
    env.full_horizon_pose_log = {"counter_start": env._sim_step_counter,
                                 "times": [], "flags": []}


def install_isaac_diagnostics(cfg):
    for name in POSE_TERMS:
        term = getattr(cfg.terminations, name)
        if term is None:
            raise ValueError(f"Missing original pose predicate: {name}")
        original = term.func

        def wrap_predicate(func, label):
            @wraps(func)
            def observe(env, **params):
                result = func(env, **params)
                flags = result.detach().cpu().numpy() if hasattr(result, "detach") else np.asarray(result)
                if env.num_envs != 1 or flags.shape != (1,):
                    raise ValueError("Full-horizon diagnostic requires one environment")
                if not hasattr(env, "full_horizon_pose_log"):
                    reset_isaac_diagnostics(env)
                log = env.full_horizon_pose_log
                time_s = (env._sim_step_counter - log["counter_start"]) * env.physics_dt
                if not log["times"] or time_s != log["times"][-1]:
                    log["times"].append(float(time_s))
                    log["flags"].append([False] * len(POSE_TERMS))
                log["flags"][-1][POSE_TERMS.index(label)] = bool(flags[0])
                return result & False
            return observe

        term.func = wrap_predicate(original, name)


def isaac_diagnostic_report(env):
    log = env.full_horizon_pose_log
    flags = np.asarray(log["flags"], dtype=bool).reshape(-1, len(POSE_TERMS))
    return summarize_flags(log["times"], flags)


def validate_snapshot(snapshot):
    keys = ("root_position_m", "root_quaternion_wxyz", "root_linear_velocity_w_m_s",
            "root_angular_velocity_w_rad_s", "joint_position_rad", "joint_velocity_rad_s")
    if not all(np.isfinite(snapshot[key]).all() for key in keys):
        raise FloatingPointError("Nonfinite state during full-horizon diagnostic")
