"""M: only raise K's continuous any-wheel load-margin cost weight from1 to4.

Keep K's exact eight-sample force inputs, margin mapping, positive support bonus
and held-reference beta gate. No force-slew, angular/vertical velocity proxy,
new observation/action/physics input, body-height or bilateral target is added.
External grounded weight+.4, other14 rewards and original acceptance are fixed.
This arithmetic module is not a trained or validated controller.
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
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, "candidate": "ground_contact_v2m",
                    "support_gap_weight": 4.}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_K_DEFINITION_SHA256:
        raise ValueError("Frozen K arithmetic source differs")
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen M settings dictionary")
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
            raise ValueError("Frozen M setting differs: " + name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"] + "; raw_M=G_gated_support_bonus-4*K_cost",
        raw_reward_bounds=[-4., 1.], immutable_v2k_definition_sha256=BASE_K_DEFINITION_SHA256,
        only_causal_change="K continuous any-wheel load-margin cost coefficient1->4",
        support_gap_coefficient_unchanged=False,
        support_gap_coefficient_previous=1., support_gap_coefficient_current=4.,
        K_support_margin_mapping_and_positive_reference_beta_bonus_unchanged=True,
        external_grounded_weight_unchanged=True, other14_reward_terms_changed=False,
        force_slew_cost_added=False, no_new_force_or_velocity_inputs=True,
        continuous_contact_or_task_success_certified=False)
    return result


def _replace(result):
    result["K_raw_reward_before_gap_weight_change"] = result["raw_reward"]
    result["K_support_gap_penalty"] = result["support_gap_penalty"]
    result["support_gap_penalty"] = _FROZEN_SETTINGS["support_gap_weight"]*result["support_gap"]
    result["raw_reward"] = result["G_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None):
    validate_settings(settings)
    return _replace(previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed,
                 *, settings=None, validate=True):
    validate_settings(settings)
    return _replace(previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed,
                                          beta_quality, reference_speed, validate=validate))

