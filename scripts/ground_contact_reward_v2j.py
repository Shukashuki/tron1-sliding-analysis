"""J: immutable I minus bounded loaded-contact angular-velocity increments.

Input angular[...,9,2] contains the ACTUAL start-state omegaXY and all eight
native post-step omegaXY samples, in world rad/s. Differences are of measured
velocities, never finite-differenced poses. No action/state filter, absolute
height or angular-rate target is added. Constant angular motion has zero cost.
Only drift_grounded changes; arithmetic alone proves no native task success.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2i as previous

base = previous.base
BASE_F_DEFINITION_SHA256 = previous.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256 = previous.BASE_G_DEFINITION_SHA256
BASE_I_DEFINITION_SHA256 = "2d805c5d878541da7a7ccc1f6af421ebe4b425fb9006756081f786ae79bf5101"
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, "candidate": "ground_contact_v2j",
    "loaded_angular_increment_deadband_rad_s": .05,
    "loaded_angular_increment_scale_rad_s": .2,
    "loaded_angular_increment_cap": 1., "loaded_angular_increment_weight": .5}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_I_DEFINITION_SHA256:
        raise ValueError("Frozen I arithmetic source differs")
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen J settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else:
            valid = (isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_))
                     and math.isfinite(float(actual)) and float(actual) == expected)
        if not valid: raise ValueError("Frozen J setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"]+"; dOmega=omega_worldXY[1:9]-omega_worldXY[0:8]; "
            "angular_cost=mean8(loaded*min(meanXY((max(abs(dOmega)-.05,0)/.2)**2),1)); "
            "raw_J=raw_I-.5*angular_cost",
        raw_reward_bounds=[-1.5, 1.], immutable_v2i_definition_sha256=BASE_I_DEFINITION_SHA256,
        only_causal_change="I plus bounded loaded-contact physical-step angularXY increment cost",
        velocity_samples_per_window=9, current_start_state_plus8_actual_substeps_required=True,
        angular_velocity_source_units="world rad/s; increment per actual2.5ms physics step",
        velocity_is_pose_finite_difference=False, end_state_repetition_allowed=False,
        no_action_or_state_filter_or_absolute_height_target=True,
        constant_angular_rate_unpenalized=True, other14_reward_terms_changed=False,
        no_velocity_proxy_or_absolute_height_input=False,
        new_input_is_true_native_angular_velocity_not_height_proxy=True,
        arithmetic_alone_does_not_certify_runtime_or_task_success=True)
    return result


def _numeric(value, dtype):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu": raise ValueError("Angular samples must be real nonboolean data")
    return value.astype(dtype, copy=False)


def angular_increment_numpy(Fz_ground, angular_world_xy, *, settings=None):
    cfg = validate_settings(settings)
    forces = np.asarray(Fz_ground)
    if forces.dtype.kind not in "fiu" or forces.ndim < 2 or forces.shape[-2:] != (8, 2):
        raise ValueError("Require native forces[...,8,2]")
    dtype = np.result_type(forces.dtype, np.float32)
    forces = forces.astype(dtype, copy=False)
    angular = _numeric(angular_world_xy, dtype)
    if angular.shape != (*forces.shape[:-2], 9, 2):
        raise ValueError("Require actual start plus8 angular samples[...,9,2]")
    if (not np.isfinite(forces).all() or np.any(forces < -cfg["negative_force_tolerance_n"])
            or not np.isfinite(angular).all()):
        raise ValueError("Invalid native force/angular sample, even when unloaded")
    delta = np.diff(angular, axis=-2)
    excess = np.maximum(np.abs(delta)-cfg["loaded_angular_increment_deadband_rad_s"], 0.)
    channel = (excess/cfg["loaded_angular_increment_scale_rad_s"])**2
    per_sample = np.minimum(channel.mean(-1), cfg["loaded_angular_increment_cap"])
    per_sample = per_sample*(forces.max(-1) > cfg["support_force_threshold_n"])
    cost = per_sample.mean(-1)
    return {"angular_velocity_increments": delta, "loaded_angular_increment_sample_cost": per_sample,
            "loaded_angular_increment_cost": cost,
            "loaded_angular_increment_penalty": np.asarray(cfg["loaded_angular_increment_weight"], dtype=dtype)*cost}


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 angular_world_xy, *, settings=None):
    validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    result.update(angular_increment_numpy(Fz_ground, angular_world_xy, settings=settings))
    result["raw_reward"] = result["raw_reward"]-result["loaded_angular_increment_penalty"]
    return result


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 angular_world_xy, *, settings=None, validate=True):
    import torch
    cfg = validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, validate=validate)
    raw = result["raw_reward"]
    angular = torch.as_tensor(angular_world_xy, device=raw.device)
    if angular.dtype == torch.bool or angular.is_complex(): raise ValueError("Require real nonboolean angular samples")
    angular = angular.to(dtype=raw.dtype)
    if tuple(angular.shape) != (*raw.shape, 9, 2): raise ValueError("Require actual start plus8 angular samples[...,9,2]")
    valid = torch.isfinite(raw) & torch.isfinite(angular).all(dim=-1).all(dim=-1)
    if validate and not bool(valid.all()): raise ValueError("Invalid angular sample, even when unloaded")
    forces = torch.as_tensor(Fz_ground, device=raw.device, dtype=raw.dtype)
    delta = angular[..., 1:, :]-angular[..., :-1, :]
    excess = (delta.abs()-cfg["loaded_angular_increment_deadband_rad_s"]).clamp_min(0.)
    per_sample = ((excess/cfg["loaded_angular_increment_scale_rad_s"])**2).mean(-1).clamp_max(cfg["loaded_angular_increment_cap"])
    per_sample = per_sample*(forces.amax(-1) > cfg["support_force_threshold_n"])
    cost = per_sample.mean(-1)
    result.update(angular_velocity_increments=delta, loaded_angular_increment_sample_cost=per_sample,
                  loaded_angular_increment_cost=cost, loaded_angular_increment_penalty=cfg["loaded_angular_increment_weight"]*cost)
    result["raw_reward"] = raw-result["loaded_angular_increment_penalty"]
    if not validate:
        for name, value in result.items():
            mask = valid[(...,)+(None,)*(value.ndim-valid.ndim)]
            result[name] = torch.where(mask, value, torch.full_like(value, float("nan")))
    return result
