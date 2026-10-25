"""Frozen V2a ground-contact shaping arithmetic, without simulator side effects.

The caller supplies *ground-filtered normal forces*, in upward world-Z newtons,
at all eight actual 400 Hz substeps. These functions do not infer contact from
height, total body net force, wheel headings, contact counts, or zero sideslip.
V2a is an initial experiment, not evidence of stable contact or task success.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

import numpy as np


_FROZEN_SETTINGS = {
    "schema_version": 1,
    "candidate": "ground_contact_v2a",
    "physics_substeps": 8,
    "support_load_fraction": .50,
    "mean_weight": .75,
    "minimum_weight": .25,
    "impact_multiple": 3.,
    "negative_force_tolerance_n": 1.e-4,
    "external_reward_weight": .4,
    "reward_dt_s": .02,
}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def validate_settings(settings=None):
    """Reject missing/extra fields and retuning; return independent JSON data."""
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2a settings dictionary")
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
            raise ValueError("Frozen V2a setting differs: " + name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    return {
        "settings": validate_settings(settings),
        "force_input": "ground-filtered upward world-Z normal force N; [...,8,2]",
        "mass_normalization": "mg is actual post-DR total mass times gravity magnitude in N",
        "formula": "q=clip(sum_wheels(max(Fz,0))/(.50*mg),0,1); "
                   "support=.75*mean_substeps(q)+.25*min_substeps(q); "
                   "excess=max(Fz/(3*mg)-1,0); impact_gate=exp(-mean_substeps_wheels(excess**2)); "
                   "raw=anchor_xy*drift_speed*support*impact_gate",
        "raw_reward_bounds": [0., 1.],
        "ground_filter_and_force_units_verified_by_caller": True,
        "contact_count_required_by_arithmetic": False,
        "support_requirement": "At least one supporting wheel; single-wheel lift or alternating load is allowed",
        "bilateral_support_required": False,
        "height_or_zero_sideslip_target": False,
        "cross_window_streak_memory": False,
        "continuous_contact_or_task_success_certified": False,
    }


def _numpy_float(value, dtype, name):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or result.dtype.kind == "f" and result.dtype.itemsize not in (4, 8):
        raise ValueError(name + " must be real float32/float64 or integer numeric data")
    return result.astype(dtype, copy=False)


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None):
    """Return bounded raw reward and every shaping component; never mutate inputs.

    mg and the two existing raw tracking qualities broadcast to the force array's
    leading batch dimensions, *not* to the eight-substep or wheel dimensions.
    Invalid input raises ValueError; tiny tolerated negative forces clamp to zero.
    """
    cfg = validate_settings(settings)
    original = np.asarray(Fz_ground)
    dtype = original.dtype if original.dtype in (np.dtype("float32"), np.dtype("float64")) else np.dtype("float64")
    force = _numpy_float(original, dtype, "Fz_ground")
    if force.ndim < 2 or force.shape[-2:] != (8, 2) or force.size == 0:
        raise ValueError("Fz_ground must have nonempty shape [...,8,2]")
    shape = force.shape[:-2]
    try:
        gravity_load = np.broadcast_to(_numpy_float(mg, dtype, "mg"), shape)
        anchor = np.broadcast_to(_numpy_float(anchor_xy, dtype, "anchor_xy"), shape)
        speed = np.broadcast_to(_numpy_float(drift_speed, dtype, "drift_speed"), shape)
    except ValueError as error:
        raise ValueError("mg and tracking qualities must broadcast to the force batch shape") from error
    if not np.isfinite(force).all() or np.any(force < -cfg["negative_force_tolerance_n"]):
        raise ValueError("Ground forces must be finite upward normal N; invalid negative force")
    if not np.isfinite(gravity_load).all() or np.any(gravity_load <= 0):
        raise ValueError("mg must be finite and strictly positive")
    if (not np.isfinite(anchor).all() or not np.isfinite(speed).all()
            or np.any((anchor < 0) | (anchor > 1)) or np.any((speed < 0) | (speed > 1))):
        raise ValueError("Existing raw tracking qualities must be finite in [0,1]")
    force = np.maximum(force, np.asarray(0., dtype=dtype))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        quality = np.clip(np.sum(force, axis=-1) / (cfg["support_load_fraction"] * gravity_load[..., None]), 0., 1.)
        mean, minimum = np.mean(quality, axis=-1), np.min(quality, axis=-1)
        support = cfg["mean_weight"] * mean + cfg["minimum_weight"] * minimum
        excess = np.maximum(force / (cfg["impact_multiple"] * gravity_load[..., None, None]) - 1., 0.)
        impact = np.exp(-np.mean(excess ** 2, axis=(-2, -1)))
        gate = anchor * speed
        raw = gate * support * impact
    return {"per_step_support": quality, "support_mean": mean, "support_min": minimum,
            "support": support, "impact_excess": excess, "impact_gate": impact,
            "tracking_gate": gate, "raw_reward": raw}


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None, validate=True):
    """Torch counterpart, preserving float32/64 dtype/device and autograd graph.

    validate=True rejects invalid values (and synchronizes on CUDA). For a GPU
    hot path, validate=False avoids host scalar reads and poisons the entire
    invalid window with NaN instead; the owner must fail its runtime health audit.
    Shape, settings and dtype validation always remain enabled.
    """
    import torch

    cfg = validate_settings(settings)
    if type(validate) is not bool:
        raise ValueError("validate must be boolean")
    force = torch.as_tensor(Fz_ground)
    if force.dtype not in (torch.float32, torch.float64):
        raise ValueError("Torch Fz_ground must be float32 or float64")
    if force.ndim < 2 or tuple(force.shape[-2:]) != (8, 2) or force.numel() == 0:
        raise ValueError("Fz_ground must have nonempty shape [...,8,2]")
    shape = force.shape[:-2]
    def broadcast(value, name):
        tensor = torch.as_tensor(value, device=force.device)
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise ValueError(name + " must be real numeric data")
        if not torch.is_tensor(value):
            # Python floats must not first round through Torch's float32 default
            # when the force arithmetic and NumPy counterpart are float64.
            tensor = torch.as_tensor(value, device=force.device, dtype=force.dtype)
        try:
            return torch.broadcast_to(tensor.to(dtype=force.dtype), shape)
        except RuntimeError as error:
            raise ValueError(name + " must broadcast to the force batch shape") from error
    gravity_load = broadcast(mg, "mg")
    anchor, speed = broadcast(anchor_xy, "anchor_xy"), broadcast(drift_speed, "drift_speed")
    valid = (torch.isfinite(force).all(dim=(-2, -1))
             & (force >= -cfg["negative_force_tolerance_n"]).all(dim=(-2, -1))
             & torch.isfinite(gravity_load) & (gravity_load > 0)
             & torch.isfinite(anchor) & (anchor >= 0) & (anchor <= 1)
             & torch.isfinite(speed) & (speed >= 0) & (speed <= 1))
    if validate and not bool(valid.all()):
        raise ValueError("Invalid ground forces, positive mg, or raw tracking qualities in [0,1]")
    force = force.clamp_min(0.)
    quality = (force.sum(dim=-1) / (cfg["support_load_fraction"] * gravity_load[..., None])).clamp(0., 1.)
    mean, minimum = quality.mean(dim=-1), quality.amin(dim=-1)
    support = cfg["mean_weight"] * mean + cfg["minimum_weight"] * minimum
    excess = (force / (cfg["impact_multiple"] * gravity_load[..., None, None]) - 1.).clamp_min(0.)
    impact = (-(excess ** 2).mean(dim=(-2, -1))).exp()
    gate, raw = anchor * speed, anchor * speed * support * impact
    result = {"per_step_support": quality, "support_mean": mean, "support_min": minimum,
              "support": support, "impact_excess": excess, "impact_gate": impact,
              "tracking_gate": gate, "raw_reward": raw}
    if not validate:
        for name, value in result.items():
            mask = valid[(...,) + (None,) * (value.ndim - valid.ndim)]
            result[name] = torch.where(mask, value, torch.full_like(value, float("nan")))
    return result
