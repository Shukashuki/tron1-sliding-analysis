"""L: preserve K and penalize bounded within-window net normal-force changes.

Uses the same eight actual floor-filtered samples and live post-DR mg as K.
No extra sensor, body-height/angular-rate input, action filter or bilateral target.
A left/right load transfer with constant summed force has zero added cost.
Zero-load windows also have zero slew cost, so K's independent support-margin
cost is retained exactly. Only actual closed-loop tests can establish benefit.
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
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, "candidate": "ground_contact_v2l",
    "normal_force_slew_weight": 1., "normal_force_slew_mg_scale": 1.,
    "normal_force_slew_cap": 1.}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_K_DEFINITION_SHA256:
        raise ValueError("Frozen K arithmetic source differs")
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen L settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str):
            valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        else:
            valid = (isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_))
                     and math.isfinite(float(actual)) and float(actual) == expected)
        if not valid:
            raise ValueError("Frozen L setting differs: " + name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"] + "; netF=sum_wheels(max(Fz,0)); "
            "delta=netF[1:8]-netF[0:7]; "
            "slew_samples=min((delta/(1*mg))^2,1); "
            "slew_cost=mean7(slew_samples); raw_L=raw_K-1*slew_cost",
        raw_reward_bounds=[-2., 1.], immutable_v2k_definition_sha256=BASE_K_DEFINITION_SHA256,
        only_causal_change="Subtract bounded mean7 squared net-normal-force increments from immutable K reward",
        normal_force_slew_uses_existing_eight_actual_samples=True,
        normal_force_slew_cross_window_boundary_sample_invented=False,
        normal_force_slew_difference_count=7,
        normal_force_slew_is_per_native400Hz_increment_not_force_derivative=True,
        normal_force_slew_tracking_or_contact_gated=False,
        constant_total_load_wheel_transfer_has_zero_added_cost=True,
        zero_load_slew_zero_does_not_remove_K_support_margin_cost=True,
        K_support_margin_and_positive_reference_beta_bonus_unchanged=True,
        external_grounded_weight_unchanged=True, other14_reward_terms_changed=False,
        new_sensor_observation_action_or_physics_input=False,
        no_body_height_angular_rate_or_velocity_proxy=True,
        continuous_contact_or_task_success_certified=False)
    return result


def _replace(result, net_force, increment, samples, *, torch_backend=False):
    result["K_raw_reward_before_normal_force_slew"] = result["raw_reward"]
    cost = samples.mean(dim=-1) if torch_backend else samples.mean(axis=-1)
    cost = cost + 0.*result["raw_reward"]  # Retain parent's whole-window NaN poison.
    result["net_ground_normal_force_n"] = net_force
    result["net_ground_normal_force_increment_n"] = increment
    result["normal_force_slew_samples"] = samples
    result["normal_force_slew_cost"] = cost
    result["normal_force_slew_penalty"] = _FROZEN_SETTINGS["normal_force_slew_weight"]*cost
    result["raw_reward"] = result["K_raw_reward_before_normal_force_slew"]-result["normal_force_slew_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None):
    cfg = validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    raw = result["raw_reward"]
    force = np.maximum(np.asarray(Fz_ground, dtype=raw.dtype), 0.)
    weight = np.broadcast_to(np.asarray(mg, dtype=raw.dtype), raw.shape)
    net_force = force.sum(axis=-1)
    increment = net_force[..., 1:]-net_force[..., :-1]
    with np.errstate(over="ignore", invalid="ignore"):
        samples = np.minimum((increment/(cfg["normal_force_slew_mg_scale"]*weight[..., None]))**2,
                             cfg["normal_force_slew_cap"])
    return _replace(result, net_force, increment, samples)


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 *, settings=None, validate=True):
    import torch
    cfg = validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed,
                                   beta_quality, reference_speed, validate=validate)
    raw = result["raw_reward"]
    force = torch.as_tensor(Fz_ground, device=raw.device, dtype=raw.dtype).clamp_min(0.)
    weight = torch.broadcast_to(torch.as_tensor(mg, device=raw.device, dtype=raw.dtype), raw.shape)
    net_force = force.sum(dim=-1)
    increment = net_force[..., 1:]-net_force[..., :-1]
    samples = ((increment/(cfg["normal_force_slew_mg_scale"]*weight[..., None]))**2).clamp_max(
        cfg["normal_force_slew_cap"])
    return _replace(result, net_force, increment, samples, torch_backend=True)

