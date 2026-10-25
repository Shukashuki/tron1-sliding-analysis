"""Counted integration and matched-protocol checks; no physics success claim."""
from pathlib import Path
import hashlib
import json
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import train_ppo_k_native_slices9 as training
import ppo_native_slices9_model as model

ROOT=Path(__file__).resolve().parents[1]


def test_parent_is_immutable_and_model_changes_are_explicit_hooks():
    source=(ROOT/"scripts/train_ppo_ground_contact_v2k.py").read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==training.PARENT_SHA
    result=training.materialize_parent(source)
    compile(result,"<test_model_training>","exec")
    assert result.count('protocol["cohort_id"] == "'+model.COHORT_ID+'"')==1
    assert result.count("_model_configure(cfg, args, current)")==1
    assert result.count("_model_before_learn(env, args, manifest, current)")==1
    assert result.count("_model_before_complete(args, manifest, current)")==1
    assert result.count("_model_seal(args, manifest, report, current)")==1
    assert 'cfg.rewards.drift_grounded = contact.reward_term(.4)' in result
    assert 'owner = contact.install(env, current["contact_settings"])' in result
    assert 'EXPECTED_INITIAL_POLICY_SHA256' in result


def test_changed_parent_is_not_silently_adapted():
    source=(ROOT/"scripts/train_ppo_ground_contact_v2k.py").read_text()
    with pytest.raises(AssertionError):training.materialize_parent(source+"\n")


def test_learning_recipe_reward_and_nominal_acceptance_match_K():
    original=json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    new=json.loads((ROOT/"config/ppo_k_native_slices9_400hz_v1.json").read_text())
    assert new["cohort_id"]!=original["cohort_id"] and new["cohort_id"]==model.COHORT_ID
    for name in ("algorithm","smoke","training_randomization","ground_contact_reward_v2", "reward_definition_version",
                 "physics_dt_s","decimation","policy_dt_s","action_rate_weight","grounded_reward_weight",
                 "nominal_acceptance_before_training"):
        assert new[name]==original[name]
    assert new["wheel_contact_model"]["model_id"]==model.MODEL_ID
    assert new["wheel_contact_model"]["contact_offset_m"]==.001
    assert new["wheel_contact_model"]["rest_offset_m"]==0.
