"""G arithmetic candidate: gate ONLY F's positive bonus by reference-beta quality.

Caller must supply the EXISTING unweighted sideslip_tracking quality for the
same held motion reference (std .12rad, ref speed>.3 and actual speed>.25).
Reference speed<=.3 makes this additional gate neutral1, not a zero-beta target.
F's ungated any-wheel support-gap cost, clocks, loads and all14 other terms stay
unchanged. Arithmetic tests do not verify a collector, trained policy or task.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2f as base

BASE_F_DEFINITION_SHA256 = "a9e561a09ca2f0cbcd6c412a8e1860817aeccaff50660e9796866996a844d3e1"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2g",
                    "beta_gate_low_reference_speed_m_s": .3}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_F_DEFINITION_SHA256:
        raise ValueError("Frozen V2f arithmetic source differs")
    base.verify_base()


verify_base()  # Import/startup only, no reward hot-path filesystem operations.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2g settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else:
            valid = (isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_))
                     and math.isfinite(float(actual)) and float(actual) == expected)
        if not valid: raise ValueError("Frozen V2g setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(base.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("raw=A_bonus-1*support_gap",
            "beta_gate=where(reference_speed>.3,existing_reference_beta_quality,1); "
            "raw=A_bonus*beta_gate-1*support_gap"),
        immutable_v2f_definition_sha256=BASE_F_DEFINITION_SHA256,
        only_causal_change="F positive bonus multiplied by existing reference-beta quality; neutral1 at low reference speed",
        existing_beta_reward_weight_or_std_changed=False, support_gap_cost_unchanged_and_ungated=True,
        reference_beta_not_zero_beta_is_intended=True, low_reference_speed_beta_gate_neutral=True,
        caller_must_verify_existing_beta_quality_and_held_reference_phase=True,
        arithmetic_alone_does_not_certify_runtime_or_task_success=True)
    return result


def _apply_gate(result, gate):
    result["beta_contact_gate"] = gate
    result["G_gated_support_bonus"] = result["A_gated_support_bonus"]*gate
    result["raw_reward"] = result["G_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None):
    validate_settings(settings)
    result = base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed)
    shape, dtype = result["raw_reward"].shape, result["raw_reward"].dtype
    quality = np.broadcast_to(base.base._numpy_float(beta_quality, dtype, "beta_quality"), shape)
    speed = np.broadcast_to(base.base._numpy_float(reference_speed, dtype, "reference_speed"), shape)
    if (not np.isfinite(quality).all() or np.any((quality < 0.) | (quality > 1.))
            or not np.isfinite(speed).all() or np.any(speed < 0.)):
        raise ValueError("Require finite existing beta quality in[0,1] and nonnegative reference speed")
    gate = np.where(speed > _FROZEN_SETTINGS["beta_gate_low_reference_speed_m_s"], quality, np.ones_like(quality))
    return _apply_gate(result, gate)


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None, validate=True):
    import torch
    validate_settings(settings)
    result = base.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, validate=validate)
    raw = result["raw_reward"]
    def broadcast(value):
        tensor = torch.as_tensor(value, device=raw.device)
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise ValueError("Beta quality/reference speed must be real nonboolean data")
        tensor = (tensor.to(dtype=raw.dtype) if torch.is_tensor(value)
                  else torch.as_tensor(value, device=raw.device, dtype=raw.dtype))
        return torch.broadcast_to(tensor, raw.shape)
    quality, speed = broadcast(beta_quality), broadcast(reference_speed)
    valid = (torch.isfinite(raw) & torch.isfinite(quality) & (quality >= 0.) & (quality <= 1.)
             & torch.isfinite(speed) & (speed >= 0.))
    if validate and not bool(valid.all()):
        raise ValueError("Require finite existing beta quality in[0,1] and nonnegative reference speed")
    gate = torch.where(speed > _FROZEN_SETTINGS["beta_gate_low_reference_speed_m_s"], quality, torch.ones_like(quality))
    result = _apply_gate(result, gate)
    if not validate:
        for name, value in result.items():
            mask = valid[(...,)+(None,)*(value.ndim-valid.ndim)]
            result[name] = torch.where(mask, value, torch.full_like(value, float("nan")))
    return result
