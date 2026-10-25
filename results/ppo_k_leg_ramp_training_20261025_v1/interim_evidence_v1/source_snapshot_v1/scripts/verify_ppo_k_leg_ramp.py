"""Verify exact K learning budget, reset-safe ramp execution and CPU export."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import train_ppo_k_leg_ramp as training


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def verify_training(directory,protocol_path,k_snapshot,*,allow_smoke=False):
    directory,protocol_path=Path(directory).resolve(),Path(protocol_path).resolve()
    parent=Path(k_snapshot).resolve()/'scripts/eval_ppo_ground_contact_v2k.py'
    assert sha(parent)=='cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c'
    source=parent.read_text();old='COHORT_ID = "'+training.OLD_ID+'"';assert source.count(old)==1
    source=source.replace(old,'COHORT_ID = "'+training.COHORT_ID+'"')
    namespace=dict(__name__='_K_leg_ramp_verifier',__file__=str(parent))
    sys.path.append(str(parent.parent));exec(compile(source,'<K_leg_ramp_verifier>','exec'),namespace)
    cohort=namespace['verify_training'](directory,protocol_path,allow_smoke=allow_smoke)
    protocol=read(protocol_path);manifest=cohort['manifest']
    audit=read(directory/'leg_ramp_training_audit.json');payload=manifest['leg_command_training']
    assert audit['kind']=='K_PPO_trained_with_batched_leg_ramp' and audit['status']=='completed_verified'
    assert audit['execution_audit']==payload and payload['cohort_id']==training.COHORT_ID
    assert audit['protocol_sha256']==sha(protocol_path)
    assert protocol['leg_command_execution']==payload['settings']==cohort['contract']['leg_command_execution']==training.EXECUTION
    assert manifest['cohort']['leg_command_execution']==training.EXECUTION
    assert payload['from_scratch'] and payload['all15_rewards_and_original_DR_unchanged']
    startup=read(directory/'leg_ramp_startup_audit.json')
    assert all(payload[k]==v for k,v in startup.items())
    for key,name in [('K_training_audit','ground_contact_v2k_training_audit.json'),('startup_audit','leg_ramp_startup_audit.json'),
            ('manifest','manifest.json'),('policy_contract','policy_contract.json'),('actor','policy.pt'),('checkpoint','model_final.pt'),
            ('pre_export_runtime_audit','pre_export_runtime_audit.json'),('export_precision_audit','export_precision_audit.json')]:
        assert audit[key+'_sha256']==sha(directory/name)
    assert audit['actor_sha256']==cohort['input_sha256']['policy.pt'] and audit['checkpoint_sha256']==cohort['input_sha256']['model_final.pt']
    assert audit['initial_policy_state_sha256']=='7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d'
    assert audit['smoke_only']==cohort['smoke_only'] and audit['trained_policy_claimed']==(not cohort['smoke_only'])
    assert audit['training_environment_steps']==cohort['training_audit']['training_environment_steps']
    precision=read(directory/'export_precision_audit.json')
    assert payload['export_precision']==precision and precision['CPU_source_module_export_bit_identical']
    assert precision['exported_weights_and_normalizer_bit_identical'] and precision['CPU_max_abs_error']==0.
    assert precision['GPU_FP32_max_abs_error']<=precision['GPU_FP32_absolute_tolerance']==1e-4
    assert precision['original_precision_restored'] and not precision['GPU_reference_TF32']
    pre=read(directory/'pre_export_runtime_audit.json')
    assert pre['status']=='live_native_health_verified_before_export' and pre['leg_ramp']==payload['final_runtime']
    windows=56 if cohort['smoke_only'] else 24008;n=manifest['num_envs']
    for key,count in [('startup_runtime',64),('final_runtime',windows*8)]:
        d=payload[key]
        assert d['mode']==training.EXECUTION['mode'] and d['physical_applications']==count
        assert d['environment_physical_applications']==count*n and d['num_envs']==n
        assert d['finite_and_held_target_and_native_application_health'] and d['exact_window_endpoint']
        assert d['original_PD_and_DC_motor_limits_retained'] and not d['wheel_effort_interpolation']
        assert not d['host_transfers_in_physical_hot_path'] and not d['per_world_Python_loop']
        assert not d['extra_environment_or_scene_step_owner']
    parent_source=Path(k_snapshot).resolve()/'scripts/train_ppo_ground_contact_v2k.py'
    assert hashlib.sha256(training.materialize_parent(parent_source.read_text()).encode()).hexdigest()==payload['materialized_parent_sha256']
    assert all(sha(p)==h for p,h in payload['source_sha256'].items())
    cohort['ramp_training_audit']=audit;return cohort


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('training-dir','protocol','k-snapshot'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--cpu-inference',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cohort=verify_training(a.training_dir,a.protocol,a.k_snapshot,allow_smoke=a.allow_smoke)
    shape=None
    if a.cpu_inference:
        import torch
        actor=torch.jit.load(str(cohort['actor']),map_location='cpu')
        with torch.inference_mode():v=actor(torch.zeros(1,51))
        assert list(v.shape)==[1,8] and torch.isfinite(v).all();shape=list(v.shape)
    d=cohort['ramp_training_audit']
    proof=dict(kind='strict_K_PPO_leg_ramp_training_verification',cohort_id=training.COHORT_ID,
        formal_or_smoke='smoke' if cohort['smoke_only'] else 'formal',transitions=d['training_environment_steps'],
        actor_sha256=d['actor_sha256'],checkpoint_sha256=d['checkpoint_sha256'],CPU_actor_output_shape=shape,
        initial_policy_state_sha256=d['initial_policy_state_sha256'],verifier_sha256=sha(__file__),task_success_claimed=False)
    with a.output.open('x') as f:json.dump(proof,f,indent=2,allow_nan=False)
    print(json.dumps(proof))


if __name__=='__main__':main()
