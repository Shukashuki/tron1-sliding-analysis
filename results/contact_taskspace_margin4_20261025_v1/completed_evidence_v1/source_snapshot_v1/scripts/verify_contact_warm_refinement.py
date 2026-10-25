"""Strict warm-start accounting, native contact audit and independent export."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import train_contact_warm_refinement as training


def verify_training(directory,protocol_path,*,allow_smoke=False,cpu_inference=False):
    root=Path(directory).resolve();protocol_path=Path(protocol_path).resolve()
    names=['manifest.json','cohort_completion_audit.json','cohort_startup_audit.json','policy_contract.json',
        'final_policy_contract.json','runner_config.json','run_status.json','policy.pt','model_final.pt',
        'pre_export_runtime_audit.json','export_precision_audit.json','environment_final.json']
    paths={n:root/n for n in names}
    assert all(p.is_file() and not p.is_symlink() for p in paths.values())
    digest={n:training.sha(p) for n,p in paths.items()}
    read=training.read;manifest=read(paths['manifest.json']);complete=read(paths['cohort_completion_audit.json'])
    contract=read(paths['policy_contract.json']);report=read(paths['run_status.json']);protocol=read(protocol_path)
    assert protocol['cohort_id']==complete['cohort_id']==contract['cohort_id']==training.COHORT_ID
    variant=complete['variant'];assert variant in ['K','M'] and variant==contract['cohort_variant']==manifest['reward_variant']
    smoke=complete['smoke_only'];assert type(smoke) is bool and (not smoke or allow_smoke)
    mode='smoke' if smoke else 'formal';d=protocol['smoke'] if smoke else protocol['algorithm']
    assert complete['run_kind']==contract['run_kind']==mode
    assert complete['status']=='completed_verified' and report['status']=='training_completed'
    assert not complete['from_scratch'] and complete['historical_checkpoint_used'] and not contract['from_scratch']
    assert complete['trained_policy_claimed']==contract['trained_policy_claimed']==(not smoke)
    assert complete['initial_policy_state_sha256']==report['initial_policy_state_sha256']==training.WARM_STATE_SHA
    assert manifest['initial_policy_state_sha256']==training.WARM_STATE_SHA
    assert manifest['iterations']==d['iterations'] and manifest['num_envs']==d['num_envs'] and manifest['seed']==42
    assert complete['training_environment_steps']==report['training_environment_steps']==contract['training_environment_transitions']==d['transitions']
    assert complete['cumulative_training_environment_steps']==report['cumulative_training_environment_steps']==training.BASE_TRANSITIONS+d['transitions']
    assert contract['cumulative_training_environment_transitions']==training.BASE_TRANSITIONS+d['transitions']
    assert complete['protocol_sha256']==contract['training_cohort_protocol_sha256']==training.sha(protocol_path)
    assert complete['cohort']==manifest['cohort'] and complete['warm_start']==manifest['warm_start']
    c=manifest['cohort'];assert c['from_scratch'] is False and c['historical_checkpoint_used'] is True
    assert c['refinement_optimizer']==training.REFINEMENT
    initial=manifest['warm_start']
    assert initial['loaded_policy_state_sha256']==training.WARM_STATE_SHA
    assert initial['constructed_random_state_sha256']==training.RANDOM_STATE_SHA
    assert initial['checkpoint_sha256']==training.CHECKPOINT_SHA and initial['source_actor_sha256']==training.ACTOR_SHA
    assert initial['normalizer_initial_count']==training.BASE_TRANSITIONS and initial['actor_critic_normalizers_bit_identical']
    assert initial['original_checkpoint_iteration']==999 and initial['new_stage_initial_iteration']==0
    assert not initial['optimizer_state_loaded'] and initial['optimizer_initial_state_empty']
    for key,name in [('manifest','manifest.json'),('policy_contract','policy_contract.json'),('actor','policy.pt'),
            ('checkpoint','model_final.pt'),('startup_audit','cohort_startup_audit.json'),
            ('pre_export_runtime_audit','pre_export_runtime_audit.json'),('export_precision_audit','export_precision_audit.json')]:
        assert complete[key+'_sha256']==digest[name]
    assert paths['policy_contract.json'].read_bytes()==paths['final_policy_contract.json'].read_bytes()
    assert manifest['contract']==contract
    assert contract['checkpoint_sha256']==digest['model_final.pt'] and contract['exported_policy_sha256']==digest['policy.pt']
    assert manifest['algorithm_source_sha256']==dict(runner=training.RUNNER_SHA,ppo=training.PPO_SHA)
    config=read(paths['runner_config.json'])
    for key,value in training.REFINEMENT.items():assert config['algorithm'][key]==value
    assert config['num_steps_per_env']==24 and config['policy']['actor_hidden_dims']==[256,128,128]
    assert config['policy']['critic_hidden_dims']==[256,128,128] and config['policy']['noise_std_type']=='log'
    assert config['algorithm']['num_learning_epochs']==5 and config['algorithm']['num_mini_batches']==4
    assert config['algorithm']['entropy_coef']==.005 and config['algorithm']['gamma']==.99 and config['algorithm']['lam']==.95
    for key,value in dict(physics_dt=.0025,decimation=8,policy_fps=50.,actor_dim=51,critic_dim=120,action_dim=8,
            leg_kp=500.,leg_kd=10.,leg_position_scale=1.,wheel_torque_scale_nm=12.,actuator_model='dc_motor',
            raw_action_clip=[-1.,1.],leg_torque_limit_nm=80.,wheel_torque_limit_nm=12.,
            leg_motor_velocity_limit_rad_s=15.,wheel_motor_velocity_limit_rad_s=100.,
            leg_command_execution='original_position_target_hold').items():assert contract[key]==value,key
    assert contract['training_domain_randomization']==protocol['training_randomization']
    assert contract['task_profile']==contract['trained_task_profile']==manifest['task_profile']
    task=contract['task_profile'];assert task['reward_weights']['sideslip']==2. and task['reward_weights']['grounded']==.4
    assert task['action_rate_weight']==-.5 and task['height_regularization']['mode']=='vertical_acceleration_only'
    assert contract['ground_contact_reward_v2']==manifest['contact_settings']==task['ground_contact_reward_v2']
    assert contract['ground_contact_reward_v2']['support_gap_weight']==(1 if variant=='K' else 4)
    assert contract['reward_definition_version']=='ground_contact_v2'+variant.lower()+'_any_wheel'
    expected_windows=8+d['iterations']*24;n=d['num_envs']
    for name,count in [('live_startup',8),('live_final',expected_windows)]:
        live=c[name];a=live['contact'];runtime=live['runtime']
        assert a['steps_completed']==a['steps_started']==a['unique_reward_windows']==count
        assert a['physics_samples']==count*8 and a['sampled_environment_substeps']==count*8*n
        assert a['device_finite_normal_force_health'] and a['device_finite_reward_health'] and not a['cpu_fixture_only']
        assert a['settings']==contract['ground_contact_reward_v2']
        assert runtime['status']=='verified' and runtime['fields']['reward_term_count']==15
        assert runtime['fields']['absolute_height_termination'] is False and runtime['fields']['absolute_xy_termination'] is False
        assert runtime['shared_DR_recipe']==protocol['shared_dr_recipe']
    assert read(paths['pre_export_runtime_audit.json'])==c['live_final']
    assert c['normalizer_final_count']==training.BASE_TRANSITIONS+d['transitions']
    precision=read(paths['export_precision_audit.json']);assert precision==c['export_precision']
    assert precision['CPU_source_module_export_bit_identical'] and precision['CPU_max_abs_error']==0.
    assert precision['exported_weights_and_normalizer_bit_identical']
    assert precision['GPU_FP32_max_abs_error']<=precision['GPU_FP32_absolute_tolerance']==1e-4
    assert precision['original_precision_restored'] and precision['GPU_reference_TF32'] is False
    assert all(training.sha(p)==h for p,h in complete['source_sha256'].items())
    result=dict(directory=root,protocol_path=protocol_path,protocol=protocol,manifest=manifest,contract=contract,
        completion=complete,training_backend='physx',smoke_only=smoke,checkpoint=paths['model_final.pt'],actor=paths['policy.pt'],
        paths=paths,input_sha256=digest,variant=variant)
    if cpu_inference:
        import torch
        checkpoint=torch.load(paths['model_final.pt'],map_location='cpu',weights_only=True)
        state=checkpoint['model_state_dict'];assert checkpoint['iter']==d['iterations']-1
        assert all(bool(torch.isfinite(v).all()) for v in state.values())
        assert int(state['actor_obs_normalizer.count'])==int(state['critic_obs_normalizer.count'])==training.BASE_TRANSITIONS+d['transitions']
        actor=torch.jit.load(str(paths['policy.pt']),map_location='cpu')
        exported=actor.state_dict()
        expected={**{'0.'+k.removeprefix('actor_obs_normalizer.'):v for k,v in state.items() if k.startswith('actor_obs_normalizer.')},
                  **{'1.'+k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')}}
        assert exported.keys()==expected.keys() and all(torch.equal(v,expected[k]) for k,v in exported.items())
        with torch.inference_mode():a=actor(torch.zeros(1,51))
        assert tuple(a.shape)==(1,8) and torch.isfinite(a).all()
        result['CPU_actor_output_shape']=[1,8]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--training-dir',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--cpu-inference',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=verify_training(a.training_dir,a.protocol,allow_smoke=a.allow_smoke,cpu_inference=a.cpu_inference)
    d=c['completion'];proof=dict(kind='strict_contact_warm_refinement_training_verification',status='completed_verified',
        cohort_id=training.COHORT_ID,variant=c['variant'],run_kind=d['run_kind'],transitions=d['training_environment_steps'],
        cumulative_transitions=d['cumulative_training_environment_steps'],actor_sha256=c['input_sha256']['policy.pt'],
        checkpoint_sha256=c['input_sha256']['model_final.pt'],completion_audit_sha256=c['input_sha256']['cohort_completion_audit.json'],
        initial_policy_state_sha256=training.WARM_STATE_SHA,CPU_actor_output_shape=c.get('CPU_actor_output_shape'),
        from_scratch=False,historical_checkpoint_used=True,verifier_sha256=training.sha(__file__),task_success_claimed=False)
    with a.output.open('x') as f:json.dump(proof,f,indent=2,allow_nan=False)
    print(json.dumps(proof))


if __name__=='__main__':main()
