"""R: M contact reward minus actual six-leg velocity cost at all8 substeps.

raw_R=raw_M-.1*mean_8x6((dq_leg/(1rad/s))^2). Wheel spin is excluded; the
penalty is ungated so lifting the wheels cannot avoid it. All M support/beta
arithmetic and the external+.4 remain. No absolute pose or height target.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import ground_contact_reward_v2m as previous

BASE_M_SHA = "83912fa9a4a44b997fe241a1bd99d9ecf1ad866e1f8e07932439363825da4542"
LEGS = ["abad_L_Joint","abad_R_Joint","hip_L_Joint","hip_R_Joint","knee_L_Joint","knee_R_Joint"]
_FROZEN_SETTINGS = {**previous.validate_settings(), "candidate":"ground_contact_v2r",
                    "leg_velocity_penalty_weight":.1, "leg_velocity_scale_rad_s":1.,
                    "leg_velocity_sampling":"all8_actual_mean_square", "leg_velocity_joint_names":LEGS[:]}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_M_SHA:
        raise ValueError("Frozen M arithmetic changed")
    previous.verify_base()


verify_base()


def validate_settings(value=None):
    d = _FROZEN_SETTINGS if value is None else value
    if not isinstance(d,dict) or d.keys()!=_FROZEN_SETTINGS.keys() or d.get("candidate")!="ground_contact_v2r":
        raise ValueError("Require complete R velocity/contact settings")
    base = {k:d[k] for k in previous.DEFAULT_SETTINGS};base["candidate"]="ground_contact_v2m"
    previous.validate_settings(base)
    for k in ("leg_velocity_penalty_weight","leg_velocity_scale_rad_s"):
        if isinstance(d[k],bool) or not isinstance(d[k],(int,float)) or d[k]!=_FROZEN_SETTINGS[k]:
            raise ValueError("Frozen R velocity setting differs: "+k)
    for k in ("leg_velocity_sampling","leg_velocity_joint_names"):
        if d[k]!=_FROZEN_SETTINGS[k]:raise ValueError("Frozen R velocity setting differs: "+k)
    return copy.deepcopy(d)


def formula_metadata(settings=None):
    return {**previous.formula_metadata(), "settings":validate_settings(settings),
            "immutable_v2m_definition_sha256":BASE_M_SHA,
            "formula":previous.formula_metadata()["formula"]+"; raw_R=raw_M-.1*mean8_mean6((actual_leg_dq/1rad_s)^2)",
            "raw_reward_bounds":[None,1.], "leg_velocity_cost_is_ungated":True,
            "leg_velocity_is_measured_at_each_actual_physical_sample":True,
            "wheel_spin_in_leg_velocity_cost":False,"absolute_height_or_pose_target_added":False,
            "other14_reward_terms_changed":False,"continuous_contact_or_task_success_certified":False}


def reward_numpy(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed,joint_velocity_rad_s,*,settings=None):
    cfg = validate_settings(settings)
    result = previous.reward_numpy(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed)
    velocity = np.asarray(joint_velocity_rad_s)
    if velocity.dtype.kind not in "fiu" or velocity.shape!=(*np.shape(Fz_ground)[:-2],8,6):
        raise ValueError("Require named actual leg velocities[...,8,6]")
    if not np.isfinite(velocity).all():raise ValueError("Nonfinite actual leg velocity")
    v = velocity.astype(result["raw_reward"].dtype,copy=False)
    square = np.mean((v/cfg["leg_velocity_scale_rad_s"])**2,axis=(-2,-1))
    result.update(M_raw_reward_before_leg_velocity=result["raw_reward"],leg_velocity_mean_square_rad2_s2=square,
                  leg_velocity_cost=cfg["leg_velocity_penalty_weight"]*square)
    result["raw_reward"] = result["raw_reward"]-result["leg_velocity_cost"]
    return result


def reward_torch(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed,joint_velocity_rad_s,*,settings=None,validate=True):
    import torch
    cfg = validate_settings(settings)
    result = previous.reward_torch(Fz_ground,mg,anchor_xy,drift_speed,beta_quality,reference_speed,validate=validate)
    velocity = torch.as_tensor(joint_velocity_rad_s,device=Fz_ground.device)
    if velocity.dtype==torch.bool or velocity.is_complex() or tuple(velocity.shape)!=(*Fz_ground.shape[:-2],8,6):
        raise ValueError("Require named actual leg velocities[...,8,6]")
    velocity = velocity.to(dtype=Fz_ground.dtype)
    finite = torch.isfinite(velocity).all(dim=(-2,-1))
    if validate and not bool(finite.all()):raise ValueError("Nonfinite actual leg velocity")
    square = (velocity/cfg["leg_velocity_scale_rad_s"]).square().mean(dim=(-2,-1))
    result.update(M_raw_reward_before_leg_velocity=result["raw_reward"],leg_velocity_mean_square_rad2_s2=square,
                  leg_velocity_cost=cfg["leg_velocity_penalty_weight"]*square)
    result["raw_reward"] = result["raw_reward"]-result["leg_velocity_cost"]
    # Propagate an invalid physical velocity to the whole environment reward.
    result["raw_reward"] = torch.where(finite,result["raw_reward"],torch.full_like(result["raw_reward"],float("nan")))
    return result
