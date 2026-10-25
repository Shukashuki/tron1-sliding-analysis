import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import train_contact_taskspace_margin as training


def read(name):
    return json.loads((ROOT/"config"/name).read_text())


def test_margin_arm_changes_only_contact_coefficient_and_keeps_targets_and_budget():
    control = read("contact_taskspace_refinement_400hz_v1.json")
    candidate = read("contact_taskspace_margin_400hz_v1.json")
    validate = training.namespace()["validate_protocol"]
    original = read("ppo_ground_contact_v2k_400hz_v1.json")
    assert validate(candidate, original) == candidate
    for key in ("algorithm", "warm_start", "refinement_optimizer", "task_space_tracking", "training_randomization",
                "nominal_acceptance_before_training", "validation", "leg_command_execution"):
        assert control[key] == candidate[key], key
    old, new = control["variant_contact_settings"]["K"], candidate["variant_contact_settings"]["M"]
    assert {k for k in old if old[k] != new[k]} == {"candidate", "support_gap_weight"}
    assert old["support_gap_weight"] == 1. and new["support_gap_weight"] == 4.
    bad = copy.deepcopy(candidate);bad["variant_contact_settings"]["M"]["support_force_threshold_n"] = 1.
    with pytest.raises(ValueError): validate(bad, original)
    bad = copy.deepcopy(candidate);bad["nominal_acceptance_before_training"]["any_wheel_support_fraction_min"] = .85
    with pytest.raises(AssertionError): validate(bad, original)
