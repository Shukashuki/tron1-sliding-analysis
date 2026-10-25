import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import train_contact_taskspace_leg_velocity as training


def test_R_adds_only_predeclared_physical_leg_velocity_cost_without_easier_gates():
    read=lambda n:json.loads((ROOT/"config"/n).read_text())
    old=read("contact_taskspace_margin_400hz_v1.json")
    new=read("contact_taskspace_leg_velocity_400hz_v1.json")
    original=read("ppo_ground_contact_v2k_400hz_v1.json")
    validate=training.namespace()["validate_protocol"]
    assert validate(new,original)==new
    for key in ("algorithm","warm_start","refinement_optimizer","task_space_tracking","training_randomization",
                "nominal_acceptance_before_training","validation","leg_command_execution"):
        assert old[key]==new[key],key
    before,after=old["variant_contact_settings"]["M"],new["variant_contact_settings"]["R"]
    assert {k for k in before if before[k]!=after[k]}=={"candidate"}
    assert set(after)-set(before)=={"leg_velocity_penalty_weight","leg_velocity_scale_rad_s","leg_velocity_sampling","leg_velocity_joint_names"}
    assert after["leg_velocity_penalty_weight"]==.1 and after["leg_velocity_scale_rad_s"]==1.
    for field,value in [("leg_velocity_penalty_weight",.2),("support_force_threshold_n",1.)]:
        bad=copy.deepcopy(new);bad["variant_contact_settings"]["R"][field]=value
        with pytest.raises(AssertionError):validate(bad,original)
    bad=copy.deepcopy(new);bad["nominal_acceptance_before_training"]["any_wheel_support_fraction_min"]=.9
    with pytest.raises(AssertionError):validate(bad,original)
