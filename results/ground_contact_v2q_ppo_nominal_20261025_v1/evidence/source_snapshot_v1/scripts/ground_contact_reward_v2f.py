"""F: frozen A positive bonus minus actual any-wheel support-gap occupancy.

This branches from A, not E: A's .75*mean8(q)+.25*min8(q), gates and impact
term are unchanged. Only subtract mean8(max_wheel(Fz)<=5N), coefficient1.
No penalty is applied merely because a supported sample carries less than
.5mg. One supporting wheel and alternating wheel support remain valid.
The finite reward is an experiment, not a guarantee of continuous support.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import ground_contact_reward_v2 as base

BASE_DEFINITION_SHA256 = "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5"
# Historical source identities retained for the existing strict training ledger.
# B/C/D are verified as immutable records, NOT F's computational parent.
BASE_B_DEFINITION_SHA256 = "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c"
BASE_C_DEFINITION_SHA256 = "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb"
BASE_D_DEFINITION_SHA256 = "1f6245ca3ca08483704d4f42a044ba38b8d09f01d81dc6e71bbf423d206898a7"
_FROZEN_SETTINGS = {**base.DEFAULT_SETTINGS, "candidate": "ground_contact_v2f",
                    "support_gap_weight": 1., "support_force_threshold_n": 5.}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_DEFINITION_SHA256:
        raise ValueError("Frozen V2a arithmetic source differs")
    for suffix, digest in (("b", BASE_B_DEFINITION_SHA256), ("c", BASE_C_DEFINITION_SHA256),
                           ("d", BASE_D_DEFINITION_SHA256)):
        source = Path(__file__).with_name("ground_contact_reward_v2"+suffix+".py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen historical V2"+suffix+" arithmetic source differs")


verify_base()  # Startup only: no filesystem access in the reward hot path.


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError("Require the complete frozen V2f settings dictionary")
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str):
            valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        else:
            valid = (isinstance(actual, Real) and not isinstance(actual, bool)
                     and math.isfinite(float(actual)) and float(actual) == expected)
        if not valid:
            raise ValueError("Frozen V2f setting differs: " + name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(base.formula_metadata())
    result.update(settings=validate_settings(settings),
        formula=result["formula"].replace("raw=anchor_xy*drift_speed*support*impact_gate",
            "A_bonus=anchor_xy*drift_speed*support*impact_gate; "
            "support_gap=mean_substeps(max_wheels(max(Fz,0))<=5N); raw=A_bonus-1*support_gap"),
        raw_reward_bounds=[-1., 1.], immutable_v2a_definition_sha256=BASE_DEFINITION_SHA256,
        only_causal_change="Branch from A: subtract ungated actual any-wheel support-gap occupancy, coefficient1",
        support_gap_tracking_gated=False, support_gap_impact_gated=False,
        supported_light_load_penalized=False, per_step_threshold_strict_greater=True,
        support_threshold_is_world_normal_newtons_not_mass_fraction=True,
        negative_term_independent_of_mean_or_min_window_packing=True,
        continuous_contact_or_task_success_certified=False)
    return result


def _add_gap(result, forces, *, torch_backend=False):
    threshold = _FROZEN_SETTINGS["support_force_threshold_n"]
    if torch_backend:
        gap_samples = (forces.clamp_min(0.).amax(dim=-1) <= threshold).to(forces.dtype)
        gap = gap_samples.mean(dim=-1)
        # Invalid forces must poison the whole window even with validate=False.
        # Do not permit comparisons with NaN to conceal invalid sensor data.
        gap = gap + 0.*result["support"]
    else:
        import numpy as np
        gap_samples = (np.maximum(forces, 0.).max(axis=-1) <= threshold).astype(result["support"].dtype)
        gap = gap_samples.mean(axis=-1)
    result["A_gated_support_bonus"] = result["raw_reward"]
    result["support_gap_samples"] = gap_samples
    result["support_gap"] = gap
    result["support_gap_penalty"] = _FROZEN_SETTINGS["support_gap_weight"]*gap
    result["raw_reward"] = result["A_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None):
    import numpy as np
    validate_settings(settings)
    result = base.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed)
    return _add_gap(result, np.asarray(Fz_ground))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, *, settings=None, validate=True):
    import torch
    validate_settings(settings)
    forces = torch.as_tensor(Fz_ground)
    result = base.reward_torch(forces, mg, anchor_xy, drift_speed, validate=validate)
    return _add_gap(result, forces, torch_backend=True)
