"""O: K contact reward plus a bounded command cost on a previously loaded leg.

Use actual clipped targets; left actions0/2/4, right1/3/5. Minimum over loaded
legs permits the other leg to move freely. No preceding support means no new
cost, allowing recovery; K's physical support-gap cost remains. No zero-slip,
absolute height, bilateral support or actuator filtering is introduced.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2k as previous

base = previous.base
BASE_F_DEFINITION_SHA256 = previous.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256 = previous.BASE_G_DEFINITION_SHA256
BASE_I_DEFINITION_SHA256 = previous.BASE_I_DEFINITION_SHA256
BASE_K_DEFINITION_SHA256 = "dbea1148c00a394b990964f7e7b26d3a7e9bdbe75e43975b3e9e79e7e0ebe18f"
LEFT, RIGHT = (0, 2, 4), (1, 3, 5)
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, "candidate": "ground_contact_v2o",
    "stance_command_delta_std_rad": .1, "stance_command_penalty_weight": 1.,
    "command_clip_lower": -1., "command_clip_upper": 1., "leg_command_scale_rad": 1.,
    "stance_command_cost_mode": "minimum_previously_loaded_leg_clipped_target_delta_rms"}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_K_DEFINITION_SHA256:
        raise ValueError("Frozen K arithmetic differs")
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require complete frozen O settings")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else: valid = isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_)) and math.isfinite(float(actual)) and actual == expected
        if not valid: raise ValueError("Frozen O setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings), raw_reward_bounds=[-2., 1.],
        immutable_v2k_definition_sha256=BASE_K_DEFINITION_SHA256,
        formula=result["formula"]+"; da=clip(a,-1,1)-clip(previous_a,-1,1); "
            "e_side=mean3(da_side**2); e=min_over_previous_Fz>5N(e_side), or0 if none; "
            "O_command_cost=1-exp(-e/.1**2); raw_O=raw_K-O_command_cost",
        only_causal_change="Add bounded preceding-load-conditioned clipped leg-target-change cost to K",
        stance_command_penalty_weight=1., stance_command_delta_std_rad=.1,
        left_leg_action_indices=list(LEFT), right_leg_action_indices=list(RIGHT),
        preceding_force_is_before_current_control_command=True, no_preceding_support_cost=0.,
        wheel_torque_actions_in_new_cost=False, other_leg_free_when_one_is_loaded=True,
        no_action_filter_or_actuator_change=True, other14_reward_terms_changed=False,
        continuous_contact_or_task_success_certified=False)
    return result


def command_numpy(preceding_Fz, action, previous_action):
    values = [np.asarray(v) for v in (preceding_Fz, action, previous_action)]
    if any(v.dtype.kind not in "fiu" or v.size == 0 or v.ndim < 1 or v.shape[-1] != size
           for v, size in zip(values, (2, 8, 8))):
        raise ValueError("Require real force[...,2], actions[...,8] with explicit joint order")
    dtype = np.result_type(*(v.dtype for v in values), np.float32)
    force, current, old = [v.astype(dtype, copy=False) for v in values]
    shape = np.broadcast_shapes(force.shape[:-1], current.shape[:-1], old.shape[:-1])
    force, current, old = [np.broadcast_to(v, (*shape, size)) for v, size in zip((force, current, old), (2, 8, 8))]
    if any(not np.isfinite(v).all() for v in (force, current, old)) or np.any(force < -1e-4):
        raise ValueError("Nonfinite commands or invalid preceding floor-normal force")
    delta = np.clip(current, -1., 1.)-np.clip(old, -1., 1.)
    square = np.stack([np.mean(delta[..., list(ids)]**2, axis=-1) for ids in (LEFT, RIGHT)], axis=-1)
    bearing = force > 5.
    minimum = np.min(np.where(bearing, square, np.inf), axis=-1)
    minimum = np.where(np.any(bearing, axis=-1), minimum, np.zeros_like(minimum))
    penalty = -np.expm1(-minimum/_FROZEN_SETTINGS["stance_command_delta_std_rad"]**2)
    return {"stance_command_penalty": penalty, "stance_command_side_delta_mean_square_rad2": square,
        "stance_command_minimum_bearing_delta_mean_square_rad2": minimum,
        "stance_command_preceding_bearing": bearing.astype(dtype)}


def command_torch(preceding_Fz, action, previous_action, *, validate=True):
    import torch
    force = preceding_Fz
    if not torch.is_tensor(force) or force.dtype not in (torch.float32, torch.float64):
        raise ValueError("Torch preceding force must be a real float32/64 tensor")
    def value(v):
        raw = torch.as_tensor(v, device=force.device)
        if raw.dtype == torch.bool or raw.is_complex(): raise ValueError("Real nonboolean commands required")
        return (raw.to(dtype=force.dtype) if torch.is_tensor(v)
                else torch.as_tensor(v, device=force.device, dtype=force.dtype))
    current, old = value(action), value(previous_action)
    values = (force, current, old)
    if any(v.numel() == 0 or v.ndim < 1 or v.shape[-1] != size for v, size in zip(values, (2, 8, 8))):
        raise ValueError("Require force[...,2], action[...,8], previous_action[...,8]")
    shape = torch.broadcast_shapes(*(v.shape[:-1] for v in values))
    force, current, old = [torch.broadcast_to(v, (*shape, size)) for v, size in zip(values, (2, 8, 8))]
    valid = torch.isfinite(force).all(-1) & (force >= -1e-4).all(-1) & torch.isfinite(current).all(-1) & torch.isfinite(old).all(-1)
    if validate and not bool(valid.all()): raise ValueError("Invalid preceding force or commands")
    delta = current.clamp(-1., 1.)-old.clamp(-1., 1.)
    square = torch.stack([delta[..., list(ids)].square().mean(-1) for ids in (LEFT, RIGHT)], dim=-1)
    bearing = force > 5.
    minimum = torch.where(bearing, square, torch.full_like(square, float("inf"))).amin(-1)
    minimum = torch.where(bearing.any(-1), minimum, torch.zeros_like(minimum))
    penalty = -torch.expm1(-minimum/_FROZEN_SETTINGS["stance_command_delta_std_rad"]**2)
    result = {"stance_command_penalty": penalty, "stance_command_side_delta_mean_square_rad2": square,
        "stance_command_minimum_bearing_delta_mean_square_rad2": minimum,
        "stance_command_preceding_bearing": bearing.to(force.dtype)}
    if not validate:
        for name, v in result.items():
            mask = valid[(...,)+(None,)*(v.ndim-valid.ndim)]
            result[name] = torch.where(mask, v, torch.full_like(v, float("nan")))
    return result


def _replace(result, command):
    if result["raw_reward"].shape != command["stance_command_penalty"].shape:
        raise ValueError("Command batch shape must match actual reward batch")
    result["K_raw_reward_before_command_cost"] = result["raw_reward"]
    result.update(command)
    result["raw_reward"] = result["raw_reward"]-result["stance_command_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 *, preceding_Fz, action, previous_action, settings=None):
    validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    return _replace(result, command_numpy(preceding_Fz, action, previous_action))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 *, preceding_Fz, action, previous_action, settings=None, validate=True):
    validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, validate=validate)
    return _replace(result, command_torch(preceding_Fz, action, previous_action, validate=validate))
