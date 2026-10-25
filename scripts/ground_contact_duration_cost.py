"""Matched-scale temporal support costs around immutable K force arithmetic.

S penalizes worst8+mean8 deficit,T penalizes2*worst8. Both cost ranges[0,2]
and all uniform-force windows match exactly. No new physical input or gate.
"""
from __future__ import annotations
import copy
import hashlib
from numbers import Real
from pathlib import Path

import numpy as np
import ground_contact_reward_v2k as previous

BASE_K_SHA="dbea1148c00a394b990964f7e7b26d3a7e9bdbe75e43975b3e9e79e7e0ebe18f"
base=previous.base
BASE_F_DEFINITION_SHA256=previous.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256=previous.BASE_G_DEFINITION_SHA256
BASE_I_DEFINITION_SHA256=previous.BASE_I_DEFINITION_SHA256


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest()!=BASE_K_SHA:
        raise ValueError("Immutable K force arithmetic changed")
    previous.verify_base()


verify_base()


def settings(variant,value=None):
    if variant not in ("S","T"):raise ValueError("Select matched S/T support cost")
    expected={**previous.validate_settings(),"candidate":"ground_contact_v2"+variant.lower(),
        "support_gap_weight":1. if variant=="S" else 2.,
        "support_gap_aggregation":"max8_plus_mean8" if variant=="S" else "max8",
        "support_gap_cost_mode":"worst_plus_mean_continuous_anywheel_load_margin" if variant=="S" else "double_worst_continuous_anywheel_load_margin"}
    d=expected if value is None else value
    if not isinstance(d,dict) or d.keys()!=expected.keys():raise ValueError("Require complete matched support settings")
    for k,v in expected.items():
        a=d[k]
        valid=(isinstance(a,str) and a==v) if isinstance(v,str) else (type(a) is int and a==v) if isinstance(v,int) else (isinstance(a,Real) and not isinstance(a,(bool,np.bool_)) and np.isfinite(a) and a==v)
        if not valid:raise ValueError("Frozen support setting differs:"+k)
    return copy.deepcopy(d)


def metadata(variant,value=None):
    result=copy.deepcopy(previous.formula_metadata())
    result.update(settings=settings(variant,value),raw_reward_bounds=[-2.,1.],
        immutable_v2k_definition_sha256=BASE_K_SHA,
        formula=result["formula"]+"; matched_T_cost=2*max8(margin);matched_S_cost=max8(margin)+mean8(margin)",
        support_gap_coefficient_unchanged=False,cost_upper_bound_unchanged=False,
        matched_penalty_upper_bound=2.,uniform_force_windows_equal_between_S_T=True,
        other14_reward_terms_changed=False,external_grounded_weight_unchanged=True,
        only_causal_comparison="At matched[0,2]cost scale,replace one worst-deficit component by the mean-deficit component",
        new_physical_inputs_or_height_or_bilateral_target=False,
        any_single_unsupported_substep_saturates_full_penalty_bound=variant=="T",
        support_gap_aggregation=settings(variant)["support_gap_aggregation"],
        continuous_contact_or_task_success_certified=False)
    return result


def replace(result,variant):
    worst=result["support_gap"];mean=result["mean_substep_support_gap"]
    result["K_raw_reward_before_duration_change"]=result["raw_reward"]
    result["K_support_gap_penalty"]=result["support_gap_penalty"]
    result["worst_support_margin_component"]=worst
    result["mean_support_margin_component"]=mean
    result["support_gap_penalty"]=worst+mean if variant=="S" else 2.*worst
    result["raw_reward"]=result["G_gated_support_bonus"]-result["support_gap_penalty"]
    return result


def reward_numpy(variant,*args,settings_value=None):
    settings(variant,settings_value)
    return replace(previous.reward_numpy(*args),variant)


def reward_torch(variant,*args,settings_value=None,validate=True):
    settings(variant,settings_value)
    return replace(previous.reward_torch(*args,validate=validate),variant)
