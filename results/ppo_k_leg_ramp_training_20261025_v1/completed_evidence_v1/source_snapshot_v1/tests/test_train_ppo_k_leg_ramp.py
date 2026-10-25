from pathlib import Path
import json
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_ppo_k_leg_ramp as trainer


def test_ramp_is_installed_before_learning_without_replacing_reward_or_step_owner():
    parent=(ROOT/'scripts/train_ppo_ground_contact_v2k.py').read_text()
    source=trainer.materialize_parent(parent)
    assert source.count('_ramp_install(env, owner, current)')==1
    assert source.count('owner = contact.install(env, current["contact_settings"])')==1
    assert source.count('cfg.rewards.drift_grounded = contact.reward_term(.4)')==1
    assert '_ramp_before_complete(args, manifest, current)' in source
    assert 'contract["leg_command_execution"]' in source
    with pytest.raises(AssertionError):trainer.materialize_parent(parent+'\n')


def test_generated_preserves_live_audit_and_independent_export_precision_guard():
    source=('def main():\n    if True:\n'
        '        runner.save(str(args.output_dir/"model_final.pt"))\n'
        '        with context:\n            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})\n')
    result=trainer.materialize_generated(source)
    assert result.index('_ramp_pre_export(')<result.index('runner.save(')
    assert result.count('_ramp_export(')==1
    with pytest.raises(AssertionError):trainer.materialize_generated('pass\n')


def test_original_K_rewards_DR_budget_and_nominal_gates_preserved():
    old=json.loads((ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json').read_text())
    new=json.loads((ROOT/'config/ppo_k_leg_ramp_400hz_v1.json').read_text())
    assert new['cohort_id']==trainer.COHORT_ID!=old['cohort_id']
    for k in ['algorithm','smoke','ground_contact_reward_v2','training_randomization',
              'reward_definition_version','nominal_acceptance_before_training','physics_dt_s','decimation','policy_dt_s']:
        assert new[k]==old[k]
    assert new['leg_command_execution']==trainer.EXECUTION
    assert 'wheel_contact_model' not in new
