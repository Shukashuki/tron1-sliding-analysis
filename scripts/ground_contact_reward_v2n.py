"""N: keep K's soft load margin; charge three more only for an actual any-gap.

If every physical sample has a wheel above5N this equals K exactly. If any
sample loses both wheels, its total cost equals M exactly. The positive support
bonus, intended reference-beta gate and external+.4 stay fixed. No height,
velocity, bilateral support, action/observation or physical model change.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2k as previous

base=previous.base
BASE_F_DEFINITION_SHA256=previous.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256=previous.BASE_G_DEFINITION_SHA256
BASE_I_DEFINITION_SHA256=previous.BASE_I_DEFINITION_SHA256
BASE_K_DEFINITION_SHA256="dbea1148c00a394b990964f7e7b26d3a7e9bdbe75e43975b3e9e79e7e0ebe18f"
_FROZEN_SETTINGS={**previous.DEFAULT_SETTINGS,"candidate":"ground_contact_v2n",
    "support_gap_weight":1.,"binary_any_gap_weight":3.,
    "support_gap_cost_mode":"worst_substep_margin_plus_binary_anygap"}
DEFAULT_SETTINGS=copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest()!=BASE_K_DEFINITION_SHA256:
        raise ValueError("Frozen K arithmetic source differs")
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value=_FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value,dict) or value.keys()!=_FROZEN_SETTINGS.keys():
        raise ValueError("Require complete frozen N settings")
    for name,expected in _FROZEN_SETTINGS.items():
        actual=value[name]
        if isinstance(expected,str):valid=isinstance(actual,str) and actual==expected
        elif isinstance(expected,int):valid=type(actual) is int and actual==expected
        else:valid=isinstance(actual,Real) and not isinstance(actual,(bool,np.bool_)) and math.isfinite(float(actual)) and float(actual)==expected
        if not valid:raise ValueError("Frozen N setting differs: "+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result=copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings),formula=result["formula"]+
        "; N_binary_gap=max8(max_wheels(max(Fz,0))<=5N); raw_N=G_gated_support_bonus-1*K_cost-3*N_binary_gap",
        raw_reward_bounds=[-4.,1.],immutable_v2k_definition_sha256=BASE_K_DEFINITION_SHA256,
        only_causal_change="Keep K margin coefficient1; add ungated coefficient3 for an actual any-wheel unsupported window",
        support_gap_cost_mode=_FROZEN_SETTINGS["support_gap_cost_mode"],
        support_gap_coefficient_previous=1.,support_gap_coefficient_current=1.,binary_any_gap_coefficient=3.,
        hybrid_margin_plus_binary_gap=True,same_as_K_when_all_samples_supported=True,
        same_total_unsupported_window_cost_as_M=True,force_slew_cost_added=False,
        no_new_force_or_velocity_inputs=True,other14_reward_terms_changed=False,
        continuous_contact_or_task_success_certified=False)
    return result


def _replace(result):
    result["K_raw_reward_before_binary_gap_cost"]=result["raw_reward"]
    result["K_support_gap_penalty"]=result["support_gap_penalty"]
    result["binary_support_gap"]=result["I_binary_support_gap"]
    result["binary_gap_penalty"]=_FROZEN_SETTINGS["binary_any_gap_weight"]*result["binary_support_gap"]
    result["support_gap_penalty"]=result["K_support_gap_penalty"]+result["binary_gap_penalty"]
    result["raw_reward"]=result["G_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed,*,settings=None):
    validate_settings(settings)
    return _replace(previous.reward_numpy(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed))


def reward_torch(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed,*,settings=None,validate=True):
    validate_settings(settings)
    return _replace(previous.reward_torch(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed,validate=validate))
