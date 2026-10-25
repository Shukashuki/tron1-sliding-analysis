"""K arithmetic: I positive bonus with worst-substep any-wheel load margin.

The existing5N support criterion and external+.4 remain fixed. Replace only I's
binary any-gap cost with a continuous deficit between5N and the existing .5mg
load scale. Any unsupported substep still receives cost1; large force peaks in
other substeps cannot compensate. One or alternating supporting wheels work.
No velocity, action filter, absolute height or bilateral target is introduced.
Arithmetic tests alone establish no collector, trained policy or task success.
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
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, "candidate": "ground_contact_v2k",
    "support_gap_cost_mode": "worst_substep_continuous_anywheel_load_margin"}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_I_DEFINITION_SHA256:
        raise ValueError("Frozen I arithmetic source differs")
    previous.verify_base()


verify_base()  # Import/startup only; no reward hot-path file reads.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen K settings dictionary")
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
            raise ValueError("Frozen K setting differs: " + name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"] + "; target=.5mg>5N; "
            "margin_samples=clip((target-max_wheel(max(Fz,0)))/(target-5N),0,1); "
            "K_cost=max8(margin_samples); raw_K=G_gated_support_bonus-1*K_cost",
        raw_reward_bounds=[-1., 1.], immutable_v2i_definition_sha256=BASE_I_DEFINITION_SHA256,
        only_causal_change="I binary any-gap cost replaced by worst-substep continuous any-wheel load margin",
        support_gap_cost_mode=_FROZEN_SETTINGS["support_gap_cost_mode"],
        support_gap_is_window_event_not_duration_fraction=False,
        support_gap_is_continuous_load_margin=True, supported_light_load_penalized=True,
        any_single_unsupported_substep_receives_full_gap_cost=True,
        support_gap_cost_ungated=True, support_gap_coefficient_unchanged=True,
        positive_bonus_and_existing_reference_beta_gate_unchanged=True,
        load_margin_uses_existing_support_load_fraction=True,
        mass_weight_must_exceed_threshold_divided_by_load_fraction=True,
        original_support_acceptance_threshold_n=5.,
        other14_reward_terms_changed=False, new_angular_velocity_input=False,
        no_velocity_proxy_or_absolute_height_input=True,
        continuous_contact_or_task_success_certified=False)
    return result


def _replace(result, samples, target, minimum, *, torch_backend=False):
    result["I_binary_support_gap_samples"] = result["support_gap_samples"]
    result["I_binary_support_gap"] = result["support_gap"]
    result["I_mean_substep_binary_gap"] = result["mean_substep_support_gap"]
    gap = samples.amax(dim=-1) if torch_backend else samples.max(axis=-1)
    gap = gap + 0.*result["raw_reward"]  # Preserve parent whole-window NaN poison.
    result["support_gap_samples"] = samples
    result["mean_substep_support_gap"] = samples.mean(dim=-1) if torch_backend else samples.mean(axis=-1)
    result["support_gap"] = gap
    result["support_gap_penalty"] = _FROZEN_SETTINGS["support_gap_weight"]*gap
    result["load_margin_target_n"] = target
    result["minimum_anywheel_support_force_n"] = minimum
    result["raw_reward"] = result["G_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None):
    cfg = validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    raw = result["raw_reward"]
    force = np.asarray(Fz_ground, dtype=raw.dtype)
    weight = np.broadcast_to(np.asarray(mg, dtype=raw.dtype), raw.shape)
    target = cfg["support_load_fraction"]*weight
    threshold = cfg["support_force_threshold_n"]
    if not np.isfinite(target).all() or np.any(target <= threshold):
        raise ValueError("K requires actual .5mg load scale strictly above5N")
    any_force = np.maximum(force, 0.).max(axis=-1)
    samples = np.clip((target[..., None]-any_force)/(target-threshold)[..., None], 0., 1.)
    return _replace(result, samples, target, any_force.min(axis=-1))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 *, settings=None, validate=True):
    import torch
    cfg = validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed,
                                   beta_quality, reference_speed, validate=validate)
    raw = result["raw_reward"]
    force = torch.as_tensor(Fz_ground, device=raw.device, dtype=raw.dtype)
    weight = torch.broadcast_to(torch.as_tensor(mg, device=raw.device, dtype=raw.dtype), raw.shape)
    target = cfg["support_load_fraction"]*weight
    threshold = cfg["support_force_threshold_n"]
    valid = torch.isfinite(raw) & torch.isfinite(target) & (target > threshold)
    if validate and not bool(valid.all()):
        raise ValueError("K requires actual .5mg load scale strictly above5N")
    denominator = (target-threshold).clamp_min(torch.finfo(raw.dtype).tiny)
    any_force = force.clamp_min(0.).amax(dim=-1)
    samples = ((target.unsqueeze(-1)-any_force)/denominator.unsqueeze(-1)).clamp(0., 1.)
    result = _replace(result, samples, target, any_force.amin(dim=-1), torch_backend=True)
    if not validate:
        for name, value in result.items():
            mask = valid[(...,)+(None,)*(value.ndim-valid.ndim)]
            result[name] = torch.where(mask, value, torch.full_like(value, float("nan")))
    return result
