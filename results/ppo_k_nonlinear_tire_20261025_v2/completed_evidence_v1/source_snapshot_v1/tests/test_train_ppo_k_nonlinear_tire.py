from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import train_ppo_k_nonlinear_tire as training


def test_counted_hooks_preserve_K_and_replace_both_native_material_audits():
    parent = (ROOT/"scripts/train_ppo_ground_contact_v2k.py").read_text()
    source = training.materialize_parent(parent)
    assert source.count('_tire_install(env, owner, current)') == 1
    assert source.count('_tire_before_complete(args, manifest, current)') == 1
    assert source.count('cfg.rewards.drift_grounded = contact.reward_term(.4)') == 1
    assert source.count('owner = contact.install(env, current["contact_settings"])') == 1
    assert source.count('"_tire_audit": _tire_recipe_audit') == 1
    with pytest.raises(AssertionError): training.materialize_parent(parent+"\n")
    generated = ('def body():\n    if True:\n'
        '        x = a if b else recipe_audit(env)\n        y = a if b else recipe_audit(env)\n'
        '        runner.save(str(args.output_dir/"model_final.pt"))\n'
        '        with mode:\n            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})\n')
    modified=training.materialize_generated(generated)
    assert modified.count('_tire_audit(env)') == 2
    assert modified.count('_tire_pre_export_audit(args, manifest, _cohort_plan)')==1
    assert modified.count('_tire_export_expected(')==1
    with pytest.raises(AssertionError): training.materialize_generated(generated.splitlines()[0])


def test_matched_learning_recipe_and_unchanged_acceptance_with_explicit_new_plant():
    original = json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    new = json.loads((ROOT/"config/ppo_k_nonlinear_tire_400hz_v1.json").read_text())
    assert new['cohort_id'] == training.COHORT_ID != original['cohort_id']
    for key in ('algorithm','smoke','training_randomization','ground_contact_reward_v2',
                'reward_definition_version','nominal_acceptance_before_training'):
        assert original[key] == new[key]
    m = new['wheel_contact_model']
    assert m['model_id'] == training.MODEL_ID and m['native_wheel_static_dynamic_mu'] == 0
    assert m['shared_wheel_mu_range'] == [.3,1.2] and m['normal_load_delay_physical_steps'] == 1
    assert m['parameters_identified_from_real_tire'] is False
