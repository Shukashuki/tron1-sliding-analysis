"""Paired conservative PPO refinement of the trusted original K checkpoint.

Both arms load actor, critic and observation normalizers identically, discard
old Adam moments and use the same fixed learning rate/clip. Only K versus M
load-margin weight differs between arms. This is NOT from-scratch 12M training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

COHORT_ID='tron1_contact_warm_refinement_400hz_v1'
K_TRAINER_SHA='65e6d41695219b676200f61d6d6941ca546fb8f1e95fbd6c60ca89347cf1d1e8'
K_PROTOCOL_SHA='855ca5a08e2b228c5146c65fb76f55cf6c9152d70480dc43a32a875aa4ff17d9'
CHECKPOINT_SHA='8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a'
ACTOR_SHA='fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912'
WARM_STATE_SHA='1c4894073146a9bd1abd131c60de7efdba493fc33d75599c679792435a110db8'
RANDOM_STATE_SHA='7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d'
BASE_TRANSITIONS=12288000
RUNNER_SHA='e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5'
PPO_SHA='deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c'
REFINEMENT=dict(learning_rate=.0001,schedule='fixed',clip_param=.1)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def state_sha(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_module(path,name,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def validate_protocol(protocol,k_protocol):
    assert protocol['cohort_id']==COHORT_ID and protocol['schema_version']==1
    for key in ['physics_dt_s','decimation','policy_dt_s','policy_fps','action_rate_weight','grounded_reward_weight',
            'shared_dr_recipe','mjlab_commit','mjlab_task_implementation','training_randomization',
            'nominal_acceptance_before_training','validation','smoke']:
        assert protocol[key]==k_protocol[key],key
    expected={**k_protocol['algorithm'],'iterations':200,'transitions':2457600}
    assert protocol['algorithm']==expected
    assert protocol['refinement_optimizer']==REFINEMENT
    assert protocol['warm_start']==dict(checkpoint_sha256=CHECKPOINT_SHA,actor_sha256=ACTOR_SHA,
        policy_state_sha256=WARM_STATE_SHA,base_training_transitions=BASE_TRANSITIONS,
        load_actor_critic_normalizers=True,load_optimizer=False,reset_stage_iteration_counter=True)
    assert protocol['variants']==['K','M'] and protocol['leg_command_execution']=='original_position_target_hold'
    for variant in ['K','M']:
        definition=importlib.import_module('ground_contact_reward_v2'+variant.lower())
        assert protocol['variant_contact_settings'][variant]==definition.validate_settings()
    return protocol


def refinement_config(base_config):
    c=copy.deepcopy(base_config);c['algorithm'].update(REFINEMENT)
    return c


def initialize_runner(runner,checkpoint,*,torch_module=None):
    t=torch_module or importlib.import_module('torch')
    assert sha(checkpoint)==CHECKPOINT_SHA
    construction=state_sha(runner.alg.policy.state_dict());assert construction==RANDOM_STATE_SHA
    saved=t.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['iter']==999 and state_sha(saved['model_state_dict'])==WARM_STATE_SHA
    runner.load(str(checkpoint),load_optimizer=False,map_location=runner.device)
    assert runner.current_learning_iteration==999 and runner.tot_timesteps==0
    actual=runner.alg.policy.state_dict()
    assert actual.keys()==saved['model_state_dict'].keys()
    assert all(t.equal(v.detach().cpu(),saved['model_state_dict'][n]) for n,v in actual.items())
    assert state_sha(actual)==WARM_STATE_SHA
    assert not runner.alg.optimizer.state_dict()['state'],'Adam moments must be fresh in both arms'
    assert runner.alg.learning_rate==.0001 and runner.alg.schedule=='fixed' and runner.alg.clip_param==.1
    assert all(g['lr']==.0001 for g in runner.alg.optimizer.param_groups)
    assert int(actual['actor_obs_normalizer.count'])==int(actual['critic_obs_normalizer.count'])==BASE_TRANSITIONS
    runner.current_learning_iteration=0  # New stage labels only; learned tensors and counts retained.
    return dict(checkpoint_sha256=CHECKPOINT_SHA,source_actor_sha256=ACTOR_SHA,loaded_policy_state_sha256=WARM_STATE_SHA,
        original_checkpoint_iteration=999,new_stage_initial_iteration=0,source_training_transitions=BASE_TRANSITIONS,
        constructed_random_state_sha256=construction,actor_critic_normalizers_bit_identical=True,
        normalizer_initial_count=BASE_TRANSITIONS,optimizer_state_loaded=False,optimizer_initial_state_empty=True,
        refinement_optimizer=REFINEMENT,historical_checkpoint_used=True,from_scratch=False)


def materialize(source,cohort):
    result=cohort.materialize_source(source)
    pairs=[('            accumulator = install_drift_reward_sampling(env, task)\n',
        '            accumulator = install_drift_reward_sampling(env, task)\n'
        '            if accumulator is not None:\n                raise ValueError("Do not stack reward owners")\n'
        '            accumulator = _refine_install(env, _cohort_plan)\n'),
       ('        config = ppo_config(args.seed, args.save_interval)\n',
        '        config = _refine_config(ppo_config(args.seed, args.save_interval))\n'),
       ('        runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=str(args.output_dir), device=args.device)\n',
        '        runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=str(args.output_dir), device=args.device)\n'
        '        _refine_initialize(runner, args, _cohort_plan)\n'),
       ('        runner.save(str(args.output_dir/"model_final.pt"))\n',
        '        _refine_pre_export(env, args, manifest, runner, _cohort_plan)\n'
        '        runner.save(str(args.output_dir/"model_final.pt"))\n'),
       ('            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})\n',
        '            expected = _refine_export(runner.alg.policy, exported, probe, args, manifest, _cohort_plan)\n')]
    for old,new in pairs:
        assert result.count(old)==1,'Missing/ambiguous refinement anchor: '+old[:90]
        result=result.replace(old,new,1)
    compile(result,'<contact_warm_refinement>','exec');return result


def run(argv):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for n in ['base-snapshot','k-snapshot','initial-checkpoint','cohort-protocol','declared-input-dir','isaac-device-audit-source']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reward-variant',choices=['K','M'],required=True)
    p.add_argument('--cohort-run-kind',choices=['smoke','formal'],required=True)
    own,inherited=p.parse_known_args(argv);previous=list(sys.path),sys.argv
    sys.path.insert(1,str(own.base_snapshot.resolve()/'scripts'))
    sys.path.append(str(own.k_snapshot.resolve()/'scripts'))
    try:
        k=load_module(own.k_snapshot/'scripts/train_ppo_ground_contact_v2k.py','_warm_K_parent',K_TRAINER_SHA)
        cohort=k.load_base(own.base_snapshot.resolve())
        kp=own.k_snapshot/'config/ppo_ground_contact_v2k_400hz_v1.json';assert sha(kp)==K_PROTOCOL_SHA
        protocol=validate_protocol(read(own.cohort_protocol),read(kp))
        letter=own.reward_variant.lower()
        definition=importlib.import_module('ground_contact_reward_v2'+letter)
        settings=definition.validate_settings()
        version='ground_contact_v2'+letter+'_any_wheel'
        here=Path(__file__).resolve().parent
        files=[Path(__file__),here/'verify_contact_warm_refinement.py',here/'policy_export_precision.py',
            here/'ground_contact_reward_v2m.py',here/'ppo_ground_contact_v2m.py',own.cohort_protocol.resolve()]
        ledger={**k.source_ledger(own.base_snapshot.resolve()),**{str(f.resolve()):sha(f) for f in files}}
        plan=dict(protocol=protocol,protocol_path=own.cohort_protocol.resolve(),protocol_sha256=sha(own.cohort_protocol),
            run_kind=own.cohort_run_kind,variant=own.reward_variant,source_hashes={n:sha(own.base_snapshot/n) for n in cohort.SOURCE_FILES},
            device_audit_source=own.isaac_device_audit_source.resolve(),warm_start_checkpoint=own.initial_checkpoint.resolve())
        generated=materialize((own.base_snapshot/'scripts/train_ppo_backend.py').read_text(),cohort)
        plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()

        def before_start(args,current):
            d=protocol['algorithm'] if own.cohort_run_kind=='formal' else protocol['smoke']
            assert (args.num_envs,args.iterations,args.seed,args.smoke_steps)==(d['num_envs'],d['iterations'],42,8)
            assert args.backend=='physx' and args.mjlab_task_implementation=='gpu'
            assert re.fullmatch(r'cuda:[0-7]',args.device) and os.environ.get('CUDA_VISIBLE_DEVICES') is None
            os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID');assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
            assert sha(args.contract)==cohort.NOMINAL_SHA256 and sha(args.task_profile)==cohort.PROFILE_SHA256
            assert sha(own.isaac_device_audit_source)==cohort.DEVICE_AUDIT_SHA256
            assert sha(own.initial_checkpoint)==CHECKPOINT_SHA
            for n,h in cohort.INPUT_PINS.items():assert sha(getattr(args,n))==h
            out=own.declared_input_dir.resolve();assert not out.exists() and not args.output_dir.exists()
            assert not out.is_relative_to(args.output_dir.resolve()) and not args.output_dir.resolve().is_relative_to(out)
            contract,task=read(args.contract),read(args.task_profile)
            original=dict(contract=sha(args.contract),task_profile=sha(args.task_profile))
            for n in cohort.STALE_FIELDS:contract.pop(n,None)
            task.update(action_rate_weight=-.5,ground_contact_reward_v2=copy.deepcopy(settings),
                description='Paired warm-start refinement of original K; '+own.reward_variant+' contact reward, fixed small PPO updates; no new actuator filtering.')
            contract.update(physics_dt=.0025,decimation=8,policy_fps=50.,task_profile=copy.deepcopy(task),trained_task_profile=copy.deepcopy(task),
                training_domain_randomization=copy.deepcopy(protocol['training_randomization']),domain_randomization=protocol['shared_dr_recipe'],
                cohort_id=COHORT_ID,cohort_variant=own.reward_variant,contract_status='declared_warm_refinement_interface_not_exported',
                physics_clock_origin='native400Hz_warm_refinement',reward_definition_version=version,ground_contact_reward_v2=copy.deepcopy(settings),
                historical_checkpoint_used=True,warm_start=copy.deepcopy(protocol['warm_start']),leg_command_execution='original_position_target_hold')
            out.mkdir(parents=True);write(out/'contract.json',contract);write(out/'task_profile.json',task)
            derived=dict(contract=sha(out/'contract.json'),task_profile=sha(out/'task_profile.json'))
            write(out/'derivation.json',dict(cohort_id=COHORT_ID,variant=own.reward_variant,protocol_sha256=sha(own.cohort_protocol),
                original_input_sha256=original,derived_input_sha256=derived,historical_checkpoint_used=True,source_files_modified=False))
            current.update(expected_transitions=d['transitions'],original_input_sha256=original,derived_input_sha256=derived)
            args.contract,args.task_profile=out/'contract.json',out/'task_profile.json'

        def configure(cfg,args,current,recipe):
            cohort.configure_physx(cfg,args,current,recipe)
            contact=importlib.import_module('ppo_ground_contact_v2'+letter)
            contact.configure_sensors(cfg.scene);cfg.rewards.drift_grounded=contact.reward_term(.4)

        def install(env,current):
            owner=importlib.import_module('ppo_ground_contact_v2'+letter).install(env,settings)
            current['contact_owner']=owner;return owner

        def initialize(runner,args,current):
            current['warm_initialization']=initialize_runner(runner,own.initial_checkpoint)

        def live_audit(env,args,manifest,current):
            owner=current['contact_owner'];a=owner.audit(health=True)
            assert a['device_finite_normal_force_health'] and a['device_finite_reward_health']
            assert a['steps_started']==a['steps_completed']==a['unique_reward_windows']
            assert a['physics_samples']==8*a['steps_completed'] and not a['cpu_fixture_only']
            assert not hasattr(env,'_tron1_batched_leg_ramp')
            term=env.action_manager.get_term('joint_pos')
            assert term.apply_actions.__func__ is type(term).apply_actions,'Original held action writer required'
            assert sha(own.initial_checkpoint)==CHECKPOINT_SHA and all(sha(f)==h for f,h in ledger.items())
            return dict(contact=a,runtime=cohort.runtime_audit(env,args,manifest['environment'],current))

        def before_learn(env,args,manifest,runner,current):
            live=live_audit(env,args,manifest,current)
            assert live['contact']['physics_samples']==64
            assert manifest['initial_policy_state_sha256']==WARM_STATE_SHA
            assert manifest['algorithm_source_sha256']==dict(runner=RUNNER_SHA,ppo=PPO_SHA)
            manifest.update(warm_start=copy.deepcopy(current['warm_initialization']),reward_definition_version=version,
                reward_variant=own.reward_variant,contact_settings=settings,smoke_only=own.cohort_run_kind=='smoke',
                trained_policy_claimed=own.cohort_run_kind=='formal')
            manifest['cohort']=dict(id=COHORT_ID,variant=own.reward_variant,run_kind=own.cohort_run_kind,
                protocol_sha256=sha(own.cohort_protocol),from_scratch=False,historical_checkpoint_used=True,
                expected_additional_transitions=current['expected_transitions'],source_training_transitions=BASE_TRANSITIONS,
                initial_policy_state_sha256=WARM_STATE_SHA,source_sha256=ledger,live_startup=live,
                generated_source_sha256=current['generated_sha256'],refinement_optimizer=REFINEMENT)
            write(args.output_dir/'manifest.json',manifest);write(args.output_dir/'cohort_startup_audit.json',manifest['cohort'])

        def pre_export(env,args,manifest,runner,current):
            live=live_audit(env,args,manifest,current)
            assert live['contact']['steps_completed']==8+args.iterations*24
            assert runner.tot_timesteps==current['expected_transitions']
            state=runner.alg.policy.state_dict();count=BASE_TRANSITIONS+current['expected_transitions']
            import torch
            assert all(bool(torch.isfinite(v).all()) for v in state.values())
            assert int(state['actor_obs_normalizer.count'])==int(state['critic_obs_normalizer.count'])==count
            manifest['cohort']['live_final']=live
            manifest['cohort']['normalizer_final_count']=count
            write(args.output_dir/'pre_export_runtime_audit.json',live)

        def export(policy,exported,probe,args,manifest,current):
            from policy_export_precision import verify_export
            expected,audit=verify_export(policy,exported,probe)
            manifest['cohort']['export_precision']=audit;write(args.output_dir/'export_precision_audit.json',audit)
            return expected

        def finalize(args,manifest,runner,exported,probe,expected,current):
            cohort.finalize_contract(args,manifest,runner,exported,probe,expected,current)
            c=manifest['contract'];c['cumulative_training_environment_transitions']=BASE_TRANSITIONS+runner.tot_timesteps
            c['from_scratch']=False;c['historical_checkpoint_used']=True
            write(args.output_dir/'policy_contract.json',c);write(args.output_dir/'final_policy_contract.json',c)
            write(args.output_dir/'manifest.json',manifest)

        def complete(args,manifest,report,current):
            assert report['status']=='training_completed' and report['training_environment_steps']==current['expected_transitions']
            assert all(sha(own.base_snapshot/n)==h for n,h in current['source_hashes'].items())
            assert sha(own.cohort_protocol)==current['protocol_sha256']
            smoke=own.cohort_run_kind=='smoke'
            report.update(cohort_id=COHORT_ID,reward_variant=own.reward_variant,run_kind=own.cohort_run_kind,smoke_only=smoke,
                trained_policy_claimed=not smoke,initial_policy_state_sha256=WARM_STATE_SHA,historical_checkpoint_used=True,
                cumulative_training_environment_steps=BASE_TRANSITIONS+current['expected_transitions'])
            audit=dict(status='completed_verified',cohort_id=COHORT_ID,variant=own.reward_variant,run_kind=own.cohort_run_kind,
                smoke_only=smoke,trained_policy_claimed=not smoke,from_scratch=False,historical_checkpoint_used=True,
                initial_policy_state_sha256=WARM_STATE_SHA,training_environment_steps=current['expected_transitions'],
                cumulative_training_environment_steps=BASE_TRANSITIONS+current['expected_transitions'],
                protocol_sha256=sha(own.cohort_protocol),cohort=manifest['cohort'],warm_start=manifest['warm_start'],
                source_sha256=ledger,task_success_claimed=False)
            for key,name in [('manifest','manifest.json'),('policy_contract','policy_contract.json'),('actor','policy.pt'),
                    ('checkpoint','model_final.pt'),('pre_export_runtime_audit','pre_export_runtime_audit.json'),
                    ('export_precision_audit','export_precision_audit.json'),('startup_audit','cohort_startup_audit.json')]:
                audit[key+'_sha256']=sha(args.output_dir/name)
            write(args.output_dir/'cohort_completion_audit.json',audit)

        namespace=dict(__name__='_warm_refinement_frozen_base',__file__=str(own.base_snapshot/'scripts/train_ppo_backend.py'),
            _cohort_plan=plan,_cohort_before_start=before_start,_cohort_launch_app=cohort.launch_app,
            _cohort_set_torch_device=cohort.set_torch_device,_cohort_configure_physx=configure,
            _cohort_before_learn=before_learn,_cohort_finalize_contract=finalize,_cohort_complete=complete,
            _refine_install=install,_refine_config=refinement_config,_refine_initialize=initialize,
            _refine_pre_export=pre_export,_refine_export=export)
        sys.argv=[str(Path(__file__).resolve()),*inherited]
        exec(compile(generated,'<contact_warm_refinement>','exec'),namespace);namespace['main']()
    finally:sys.path[:],sys.argv=previous


if __name__=='__main__':run(sys.argv[1:])
