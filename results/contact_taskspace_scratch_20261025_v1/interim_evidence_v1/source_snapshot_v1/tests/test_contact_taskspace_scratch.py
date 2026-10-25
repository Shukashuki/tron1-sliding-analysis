"""Reject inherited warm-start semantics and preserve the frozen PPO recipe."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import train_contact_taskspace_scratch as training


def inputs():
    return (training.read(ROOT/"config/contact_taskspace_scratch_400hz_v1.json"),
            training.read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json"))


def test_pair_is_from_zero_and_only_contact_coefficient_differs():
    p, b = inputs(); assert training.validate_protocol(p, b) == p
    k, m = copy.deepcopy(p["variant_contact_settings"]["K"]), copy.deepcopy(p["variant_contact_settings"]["M"])
    assert k.pop("candidate") == "ground_contact_v2k" and m.pop("candidate") == "ground_contact_v2m"
    assert k.pop("support_gap_weight") == 1. and m.pop("support_gap_weight") == 4.
    assert k == m
    assert p["algorithm"]["transitions"] == 512*24*1000


@pytest.mark.parametrize("field,value", [
    ("from_scratch", False), ("historical_checkpoint_used", True),
    ("warm_start", {}), ("refinement_optimizer", {}), ("variants", ["M"]),
    ("leg_command_execution", "ramped"),
])
def test_resume_and_unmatched_execution_rejected(field, value):
    p, b = inputs(); p[field] = value
    with pytest.raises(AssertionError): training.validate_protocol(p, b)


@pytest.mark.parametrize("field", ["algorithm", "validation", "training_randomization", "nominal_acceptance_before_training"])
def test_budget_DR_and_gates_cannot_change(field):
    p, b = inputs(); p[field] = {}
    with pytest.raises(AssertionError): training.validate_protocol(p, b)


def test_materialized_frozen_trainer_has_no_load_or_optimizer_override():
    cohort = training.frozen.load_base(ROOT)
    source = (ROOT/"scripts/train_ppo_backend.py").read_text()
    generated = training.materialize(source, cohort)
    assert "runner.load(" not in generated and "torch.load(" not in generated
    assert generated.count("_scratch_pre_export(env, args, manifest, runner, _cohort_plan)") == 1
    assert generated.count("_scratch_install(env, _cohort_plan)") == 1
    assert 'config = ppo_config(args.seed, args.save_interval)' in generated
    assert generated.index('torch.manual_seed(args.seed)') < generated.index('runner = OnPolicyRunner(')
    assert 'init_at_random_ep_len=False' in generated
    with pytest.raises(ValueError): training.materialize(source+'\n', cohort)


def test_launch_template_cannot_supply_a_checkpoint_or_warm_snapshot():
    a = training.read(ROOT/"config/contact_taskspace_scratch_launch_template.json")["argv"]
    assert "--initial-checkpoint" not in a and "--k-snapshot" not in a
    assert "--baseline-protocol" in a and a[a.index("--iterations")+1] == "2"
    assert a[a.index("--reward-variant")+1] == "K"
