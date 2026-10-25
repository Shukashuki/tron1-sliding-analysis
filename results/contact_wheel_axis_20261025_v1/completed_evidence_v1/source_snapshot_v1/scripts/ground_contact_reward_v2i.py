"""I: charge any unsupported physical substep, not mean gap occupancy.

Branch ONLY from immutable G. Its reference-beta-gated A positive bonus stays
unchanged; its ungated gap coefficient stays 1 (external reward weight +.4).
The gap aggregation changes from mean8 to max8. One or alternating supporting
wheels are allowed. No torso/wheel velocity, height or bilateral target exists.
Arithmetic alone does not certify native collection, learned behavior or success.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2g as previous

# Preserve the F/A ancestry interface used by the frozen G trainer/evaluator.
base = previous.base
BASE_F_DEFINITION_SHA256 = previous.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256 = "81056b23fbf2052f2e0630d1888aa6b71556027ca108ab3b08dd768a912e493a"
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, "candidate": "ground_contact_v2i",
                    "support_gap_aggregation": "max8"}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_G_DEFINITION_SHA256:
        raise ValueError("Frozen V2g arithmetic source differs")
    previous.verify_base()


verify_base()  # Import/startup only. Never do filesystem I/O per reward.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2i settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else:
            valid = (isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_))
                     and math.isfinite(float(actual)) and float(actual) == expected)
        if not valid: raise ValueError("Frozen V2i setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("support_gap=mean_substeps(", "support_gap=max_substeps("),
        immutable_v2g_definition_sha256=BASE_G_DEFINITION_SHA256,
        only_causal_change="G ungated support-gap aggregation mean8->max8; all other inputs and coefficients unchanged",
        support_gap_cost_unchanged_and_ungated=False,
        support_gap_cost_ungated=True, support_gap_coefficient_unchanged=True,
        support_gap_is_window_event_not_duration_fraction=True,
        negative_term_independent_of_mean_or_min_window_packing=False,
        any_single_unsupported_substep_receives_full_gap_cost=True,
        positive_bonus_and_existing_reference_beta_gate_unchanged=True,
        no_velocity_proxy_or_absolute_height_input=True,
        continuous_contact_or_task_success_certified=False)
    return result


def _apply(result, *, torch_backend=False):
    result["mean_substep_support_gap"] = result["support_gap"]
    samples = result["support_gap_samples"]
    gap = samples.amax(dim=-1) if torch_backend else samples.max(axis=-1)
    # Comparisons alone could hide NaN forces. Keep G's whole-window poison.
    gap = gap + 0.*result["raw_reward"]
    result["support_gap"] = gap
    result["support_gap_penalty"] = _FROZEN_SETTINGS["support_gap_weight"]*gap
    result["raw_reward"] = result["G_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None):
    validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    return _apply(result)


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None, validate=True):
    validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, validate=validate)
    return _apply(result, torch_backend=True)
