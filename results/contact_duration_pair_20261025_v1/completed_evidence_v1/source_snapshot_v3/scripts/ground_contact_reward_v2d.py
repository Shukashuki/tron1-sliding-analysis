"""D: ONLY replace C's positive min8 bonus by mean8; keep dense deficit.

raw=g*impact*mean8(q)-mean8(1-q). This stateless current-window hypothesis
removes a conditional packing/min-cliff incentive; it does not certify support.
Frozen C/B/A supply every input check, q, impact, tracking and deficit component.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import ground_contact_reward_v2c as base

BASE_DEFINITION_SHA256 = "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5"
BASE_B_DEFINITION_SHA256 = "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c"
BASE_C_DEFINITION_SHA256 = "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2d", "mean_weight": 1., "minimum_weight": 0.}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_C_DEFINITION_SHA256:
        raise ValueError("Frozen V2c arithmetic source differs")
    base.verify_base()
    if (base.BASE_DEFINITION_SHA256 != BASE_DEFINITION_SHA256
            or base.BASE_B_DEFINITION_SHA256 != BASE_B_DEFINITION_SHA256):
        raise ValueError("Frozen V2a/V2b arithmetic identities differ")


verify_base()  # Import/startup only, never in the GPU reward hot path.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2d settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else: valid = isinstance(actual, Real) and not isinstance(actual, bool) and math.isfinite(float(actual)) and float(actual) == expected
        if not valid: raise ValueError("Frozen V2d setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(base.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("support=min_substeps(q)", "support=mean_substeps(q)"),
        immutable_v2a_definition_sha256=BASE_DEFINITION_SHA256,
        immutable_v2b_definition_sha256=BASE_B_DEFINITION_SHA256,
        immutable_v2c_definition_sha256=BASE_C_DEFINITION_SHA256,
        only_causal_change="C positive bonus min8(q)->mean8(q); entire ungated deficit and gates unchanged",
        cross_window_streak_memory=False, continuous_contact_or_task_success_certified=False)
    return result


def _mean_positive_bonus(result):
    result["support"] = result["support_mean"]
    result["gated_support_bonus"] = result["tracking_gate"]*result["support"]*result["impact_gate"]
    result["raw_reward"] = result["gated_support_bonus"]-result["support_deficit_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None):
    validate_settings(settings)
    return _mean_positive_bonus(base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None, validate=True):
    validate_settings(settings)
    return _mean_positive_bonus(base.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, validate=validate))
