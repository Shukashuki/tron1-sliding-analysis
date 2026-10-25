"""V2b: only replace V2a's .75 mean + .25 minimum support with min8(q).

The immutable V2a implementation supplies every input check, force/load
component and impact/tracking gate. No bilateral or cross-window constraint is
introduced: one supporting wheel at each of the eight substeps is sufficient.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2 as base


BASE_DEFINITION_SHA256 = "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2b",
                    "mean_weight": 0., "minimum_weight": 1.}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_DEFINITION_SHA256:
        raise ValueError("Frozen V2a arithmetic source differs")


verify_base()  # Import/startup only: never perform filesystem work in the GPU reward hot path.


def validate_settings(settings=None):
    """Require exact B settings; neither A settings nor hidden retuning passes."""
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2b settings dictionary")
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
            raise ValueError("Frozen V2b setting differs: " + name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(base.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("support=.75*mean_substeps(q)+.25*min_substeps(q)",
                                           "support=min_substeps(q)"),
        immutable_v2a_definition_sha256=BASE_DEFINITION_SHA256,
        only_causal_change="support mean/min weights .75/.25 -> 0/1; all other components unchanged")
    return result


def _minimum_support(result):
    result["support"] = result["support_min"]
    result["raw_reward"] = result["tracking_gate"] * result["support"] * result["impact_gate"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None):
    validate_settings(settings)
    return _minimum_support(base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None, validate=True):
    validate_settings(settings)
    return _minimum_support(base.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, validate=validate))
