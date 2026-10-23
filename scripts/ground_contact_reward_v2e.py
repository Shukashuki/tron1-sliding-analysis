"""E: ONLY disable D's ungated deficit; retain its gated mean8 support bonus.

raw=g*impact*mean8(q). Frozen D/C/B/A preserve all force validation, q,
normalization, tracking gates and impact terms. Zero times an invalid deficit
intentionally remains NaN, so zero weighting cannot conceal bad force windows.
This reward-level factorial cell is not an empirical interaction result.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import ground_contact_reward_v2d as base

BASE_DEFINITION_SHA256 = "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5"
BASE_B_DEFINITION_SHA256 = "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c"
BASE_C_DEFINITION_SHA256 = "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb"
BASE_D_DEFINITION_SHA256 = "1f6245ca3ca08483704d4f42a044ba38b8d09f01d81dc6e71bbf423d206898a7"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2e", "support_deficit_weight": 0}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_D_DEFINITION_SHA256:
        raise ValueError("Frozen V2d arithmetic source differs")
    base.verify_base()
    if (base.BASE_DEFINITION_SHA256 != BASE_DEFINITION_SHA256
            or base.BASE_B_DEFINITION_SHA256 != BASE_B_DEFINITION_SHA256
            or base.BASE_C_DEFINITION_SHA256 != BASE_C_DEFINITION_SHA256):
        raise ValueError("Frozen V2a/V2b/V2c arithmetic identities differ")


verify_base()  # Import/startup only, never in the GPU reward hot path.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2e settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else: valid = isinstance(actual, Real) and not isinstance(actual, bool) and math.isfinite(float(actual)) and float(actual) == expected
        if not valid: raise ValueError("Frozen V2e setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(base.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("raw=gated_support_bonus-1*support_deficit", "raw=gated_support_bonus-0*support_deficit"),
        raw_reward_bounds=[0., 1.], immutable_v2a_definition_sha256=BASE_DEFINITION_SHA256,
        immutable_v2b_definition_sha256=BASE_B_DEFINITION_SHA256,
        immutable_v2c_definition_sha256=BASE_C_DEFINITION_SHA256,
        immutable_v2d_definition_sha256=BASE_D_DEFINITION_SHA256,
        only_causal_change="D support_deficit_weight1->0; complete gated mean8(q) positive bonus and all inputs unchanged",
        support_deficit_active=False, zero_weight_preserves_invalid_force_poison=True,
        cross_window_streak_memory=False, continuous_contact_or_task_success_certified=False,
        reward_level_factorial_interaction_zero=True, empirical_performance_interaction_claimed=False)
    return result


def _disable_deficit(result):
    result["support_deficit_penalty"] = _FROZEN_SETTINGS["support_deficit_weight"]*result["support_deficit"]
    result["raw_reward"] = result["gated_support_bonus"]-result["support_deficit_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None):
    validate_settings(settings)
    return _disable_deficit(base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None, validate=True):
    validate_settings(settings)
    return _disable_deficit(base.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, validate=validate))
