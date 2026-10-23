"""H arithmetic prototype: G plus loaded-contact vertical COM-motion damping.

Only the contact term changes. WorldZ is the floor normal ONLY on the verified
flat floor. This is BASE COM velocity, not contact-point slip, force-cone
utilization or an absolute-height target. The beta target stays nonzero.
Runtime must measure all eight physical substeps; no end-state repetition or
finite-difference velocity is accepted as runtime evidence. No H policy exists.
"""
from __future__ import annotations
import copy
import hashlib
import math
from numbers import Real
from pathlib import Path
import numpy as np
import ground_contact_reward_v2g as base

BASE_G_DEFINITION_SHA256 = "81056b23fbf2052f2e0630d1888aa6b71556027ca108ab3b08dd768a912e493a"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2h",
    "loaded_com_vz_deadband_m_s": .3, "loaded_com_vz_scale_m_s": .3,
    "loaded_com_vz_cost_cap": 1., "loaded_com_vz_weight": .5}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)

def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_G_DEFINITION_SHA256:
        raise ValueError("Frozen G arithmetic source differs")
    base.verify_base()

verify_base()

def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require complete frozen H settings")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else:
            valid = (isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_))
                and math.isfinite(float(actual)) and float(actual) == expected)
        if not valid: raise ValueError("Frozen H setting differs: "+name)
    return copy.deepcopy(value)

def formula_metadata(settings=None):
    value = copy.deepcopy(base.formula_metadata())
    value.update(settings=validate_settings(settings),
        formula=value["formula"]+"; loaded=max_wheel(Fz)>5N; "
            "motion=mean8(loaded*min((max(abs(base_COM_vz)-.3,0)/.3)**2,1)); "
            "raw_H=raw_G-.5*motion",
        immutable_v2g_definition_sha256=BASE_G_DEFINITION_SHA256,
        only_causal_change="Add bounded loaded-contact vertical COM velocity cost to G",
        existing_beta_reward_weight_or_std_changed=False,
        absolute_height_target_added=False, bilateral_support_required=False,
        velocity_source="measured base COM worldZ velocity at EACH of eight native refreshes",
        runtime_velocity_samples_verified=False,
        contact_point_slip_or_friction_cone_measurement_claimed=False,
        arithmetic_alone_does_not_certify_runtime_or_task_success=True)
    return value

def _numpy(value, dtype, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or value.dtype.kind == "b":
        raise ValueError(name+" must be real numeric data")
    return value.astype(dtype, copy=False)

def normal_motion_penalty_numpy(Fz_ground, base_com_vz, *, settings=None):
    settings = validate_settings(settings)
    force = np.asarray(Fz_ground)
    if force.dtype.kind not in "fiu" or force.dtype.kind == "b":
        raise ValueError("Force must be real numeric data")
    dtype = np.result_type(force.dtype, np.float32)
    force = force.astype(dtype, copy=False)
    velocity = _numpy(base_com_vz, dtype, "Base COM velocity")
    if force.ndim < 2 or force.shape[-2:] != (8, 2) or velocity.shape != force.shape[:-1]:
        raise ValueError("Require measured eight-substep forces[...,8,2] and velocities[...,8]")
    if (not np.isfinite(force).all() or np.any(force < -settings["negative_force_tolerance_n"])
            or not np.isfinite(velocity).all()):
        raise ValueError("Invalid native force or COM velocity, even when unloaded")
    loaded = np.max(force, axis=-1) > settings["support_force_threshold_n"]
    excess = np.maximum(np.abs(velocity)-settings["loaded_com_vz_deadband_m_s"], 0.)
    per_sample = np.where(loaded, np.minimum((excess/settings["loaded_com_vz_scale_m_s"])**2,
                                             settings["loaded_com_vz_cost_cap"]), 0.)
    cost = per_sample.mean(-1)
    return {"loaded_com_vz_sample_cost": per_sample, "loaded_com_vz_cost": cost,
            "loaded_com_vz_penalty": np.asarray(settings["loaded_com_vz_weight"],dtype=dtype)*cost}

def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 base_com_vz, *, settings=None):
    validate_settings(settings)
    result = base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    components = normal_motion_penalty_numpy(Fz_ground, base_com_vz, settings=settings)
    result.update(components)
    result["raw_reward"] = result["raw_reward"]-components["loaded_com_vz_penalty"]
    return result

def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 base_com_vz, *, settings=None, validate=True):
    import torch
    settings = validate_settings(settings)
    result = base.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                              validate=validate)
    raw = result["raw_reward"]
    force = torch.as_tensor(Fz_ground, device=raw.device, dtype=raw.dtype)
    velocity = torch.as_tensor(base_com_vz, device=raw.device)
    if velocity.dtype == torch.bool or velocity.is_complex():
        raise ValueError("Base COM velocity must be real nonboolean data")
    velocity = velocity.to(dtype=raw.dtype)
    if tuple(velocity.shape) != (*raw.shape, 8):
        raise ValueError("Require all eight measured COM velocity samples")
    valid = torch.isfinite(raw) & torch.isfinite(velocity).all(-1)
    if validate and not bool(valid.all()):
        raise ValueError("Invalid native COM velocity, even when unloaded")
    loaded = force.amax(-1) > settings["support_force_threshold_n"]
    excess = torch.clamp(velocity.abs()-settings["loaded_com_vz_deadband_m_s"], min=0.)
    per_sample = torch.where(loaded, torch.clamp((excess/settings["loaded_com_vz_scale_m_s"])**2,
                                                max=settings["loaded_com_vz_cost_cap"]),
                             torch.zeros_like(velocity))
    cost = per_sample.mean(-1)
    result.update(loaded_com_vz_sample_cost=per_sample, loaded_com_vz_cost=cost,
                  loaded_com_vz_penalty=settings["loaded_com_vz_weight"]*cost)
    result["raw_reward"] = raw-result["loaded_com_vz_penalty"]
    if not validate:
        for name, value in result.items():
            mask = valid[(...,)+(None,)*(value.ndim-valid.ndim)]
            result[name] = torch.where(mask, value, torch.full_like(value, float("nan")))
    return result

