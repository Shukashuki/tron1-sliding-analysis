from pathlib import Path
import hashlib
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import eval_contact_warm_refinement as evaluation
import train_contact_warm_refinement as training


def test_native_adapter_admits_declared_refinement_config_without_mutating_legacy_physics():
    source=(ROOT/'scripts/eval_ppo_ground_contact_v2k.py').read_text()
    new=evaluation.materialize_parent_source(source)
    assert new.count('_refine_config(ppo_config(42, config["save_interval"]))')==1
    assert 'contact_warm_refinement_evaluation' in new
    assert source.count('native.materialize_source')==new.count('native.materialize_source')
    with pytest.raises(AssertionError):evaluation.materialize_parent_source(source+'\n')


def test_warm_start_provenance_does_not_claim_from_scratch_or_a_12M_total_budget():
    parent=evaluation.load_parent(ROOT,ROOT)
    protocol=ROOT/'config/contact_warm_refinement_400hz_v1.json'
    c=dict(protocol=training.read(protocol),protocol_path=protocol,training_backend='physx',smoke_only=False,
        completion=dict(run_kind='formal',training_environment_steps=2457600,cumulative_training_environment_steps=14745600),
        variant='M',contract=dict(reward_definition_version='ground_contact_v2m_any_wheel'),
        input_sha256={'model_final.pt':'a'*64,'policy.pt':'b'*64})
    p=parent.provenance(c)
    assert not p['from_scratch'] and p['historical_checkpoint_used']
    assert p['reward_variant']=='M' and p['initial_policy_state_sha256']==training.WARM_STATE_SHA
    assert p['additional_refinement_transitions']==2457600 and p['cumulative_training_transitions']==14745600
