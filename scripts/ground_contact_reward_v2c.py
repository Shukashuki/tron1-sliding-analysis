"""C: preserve full frozen B bonus and subtract an UNGATED dense load deficit.

raw = B_gated_impact_min8_bonus - mean8(1-q), weight fixed1. Every force,
load-normalization, tracking/impact gate and input check comes from frozen B/A.
Single or alternating wheel support is allowed; no height/zero-beta/streak term.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import ground_contact_reward_v2b as base

BASE_DEFINITION_SHA256 = "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5"
BASE_B_DEFINITION_SHA256 = "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2c", "support_deficit_weight": 1.}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_B_DEFINITION_SHA256:
        raise ValueError("Frozen V2b arithmetic source differs")
    base.verify_base()
    if base.BASE_DEFINITION_SHA256 != BASE_DEFINITION_SHA256:
        raise ValueError("Frozen V2a arithmetic identity differs")


verify_base()  # Startup/import only; no file access or host scalar reads in reward hot path.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2c settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str):
            valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        else:
            valid = isinstance(actual, Real) and not isinstance(actual, bool) and math.isfinite(float(actual)) and float(actual) == expected
        if not valid: raise ValueError("Frozen V2c setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(base.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("raw=anchor_xy*drift_speed*support*impact_gate",
            "gated_support_bonus=anchor_xy*drift_speed*support*impact_gate; support_deficit=mean_substeps(1-q); raw=gated_support_bonus-1*support_deficit"),
        raw_reward_bounds=[-1., 1.], immutable_v2a_definition_sha256=BASE_DEFINITION_SHA256,
        immutable_v2b_definition_sha256=BASE_B_DEFINITION_SHA256,
        only_causal_change="Subtract ungated dense mean8(1-q), coefficient1; preserve entire B positive bonus",
        support_deficit_tracking_gated=False, support_deficit_impact_gated=False)
    return result


def _add_deficit(result, *, torch_backend=False):
    deficit_samples = 1.-result["per_step_support"]
    deficit = deficit_samples.mean(dim=-1) if torch_backend else deficit_samples.mean(axis=-1)
    result["gated_support_bonus"] = result["raw_reward"]
    result["support_deficit"] = deficit
    result["support_deficit_penalty"] = _FROZEN_SETTINGS["support_deficit_weight"]*deficit
    result["raw_reward"] = result["gated_support_bonus"]-result["support_deficit_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None):
    validate_settings(settings)
    return _add_deficit(base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None, validate=True):
    validate_settings(settings)
    return _add_deficit(base.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, validate=validate), torch_backend=True)
