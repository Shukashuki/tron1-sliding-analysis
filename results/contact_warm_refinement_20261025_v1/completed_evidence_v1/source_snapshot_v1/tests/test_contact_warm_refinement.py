from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_contact_warm_refinement as t
import train_ppo_backend_400hz as frozen
from train_ppo_backend import ppo_config


def test_new_protocol_keeps_DR_task_and_all_original_gates_with_honest_extra_budget():
    old=t.read(ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json')
    p=t.read(ROOT/'config/contact_warm_refinement_400hz_v1.json')
    assert t.validate_protocol(p,old)==p
    for key in ['nominal_acceptance_before_training','training_randomization','validation','smoke']:
        assert old[key]==p[key]
    bad=copy.deepcopy(p);bad['warm_start']['load_optimizer']=True
    with pytest.raises(AssertionError):t.validate_protocol(bad,old)
    bad=copy.deepcopy(p);bad['nominal_acceptance_before_training']['any_wheel_support_fraction_min']=.9
    with pytest.raises(AssertionError):t.validate_protocol(bad,old)


def test_same_conservative_optimizer_changes_only_three_declared_fields():
    old=ppo_config();new=t.refinement_config(old)
    assert old['algorithm']['learning_rate']==.001 and old['algorithm']['schedule']=='adaptive'
    expected=copy.deepcopy(old);expected['algorithm'].update(t.REFINEMENT)
    assert new==expected and new['algorithm']['clip_param']==.1


def test_checkpoint_loading_retains_all_tensors_and_normalizers_but_not_Adam_or_stage_counter(tmp_path,monkeypatch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Linear(2,1);self.critic=torch.nn.Linear(2,1)
            self.actor_obs_normalizer=torch.nn.Module();self.actor_obs_normalizer.register_buffer('count',torch.tensor(t.BASE_TRANSITIONS))
            self.critic_obs_normalizer=torch.nn.Module();self.critic_obs_normalizer.register_buffer('count',torch.tensor(t.BASE_TRANSITIONS))
    policy=Policy();saved=copy.deepcopy(policy.state_dict());saved['actor.weight']+=1;saved['critic.bias']-=2
    checkpoint=tmp_path/'trusted.pt';torch.save(dict(iter=999,model_state_dict=saved,optimizer_state_dict={'discarded':True}),checkpoint)
    monkeypatch.setattr(t,'CHECKPOINT_SHA',t.sha(checkpoint))
    monkeypatch.setattr(t,'WARM_STATE_SHA',t.state_sha(saved));monkeypatch.setattr(t,'RANDOM_STATE_SHA',t.state_sha(policy.state_dict()))
    optimizer=torch.optim.Adam(policy.parameters(),lr=.0001)
    runner=NS(alg=NS(policy=policy,optimizer=optimizer,learning_rate=.0001,schedule='fixed',clip_param=.1),
        device='cpu',current_learning_iteration=0,tot_timesteps=0)
    def load(path,*,load_optimizer,map_location):
        assert load_optimizer is False and map_location=='cpu'
        d=torch.load(path,map_location='cpu',weights_only=True);policy.load_state_dict(d['model_state_dict'])
        runner.current_learning_iteration=d['iter']
    runner.load=load
    audit=t.initialize_runner(runner,checkpoint)
    assert runner.current_learning_iteration==0 and audit['historical_checkpoint_used'] and not audit['from_scratch']
    assert all(torch.equal(v,saved[k]) for k,v in policy.state_dict().items())
    assert not optimizer.state


def test_counted_integration_initializes_before_manifest_and_preserves_live_audit_before_export():
    source=(ROOT/'scripts/train_ppo_backend.py').read_text();new=t.materialize(source,frozen)
    assert new.index('_refine_initialize(runner, args, _cohort_plan)')<new.index('manifest["initial_policy_state_sha256"] = policy_state_hash(')
    assert new.index('_refine_pre_export(env, args, manifest, runner, _cohort_plan)')<new.index('runner.save(str(args.output_dir/"model_final.pt"))')
    assert new.count('_refine_install(env, _cohort_plan)')==1
    with pytest.raises(ValueError):t.materialize(source+'\n',frozen)
