"""Post-hoc CPU-only physics diagnostic of the export-rejected first actor.

Its CPU weights/normalizer/export identity was separately verified. Its final
training-period native contact audit is missing, so this episode is explicitly
ineligible for the qualified formal comparison. It never changes that record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','The diagnostic must not create a GPU context'
    root=a.first_attempt.resolve();output=a.output.resolve();inputs=output.with_name(output.name+'_inputs')
    assert not output.exists() and not inputs.exists()
    actor=root/'formal_v1/policy.pt';checkpoint=root/'formal_v1/model_final.pt'
    assert sha(actor)=='3cfa0cf2707e9c9363cf9b4e6c0954dbb86a218d59675ff2b427c7e687aa37f1'
    assert sha(checkpoint)=='fe7dccab416f09a4ed35002448054632d1928999be83d720705dcb1ae847ad1f'
    diagnostic=json.loads((root/'export_precision_diagnostic_v1.json').read_text())
    assert diagnostic['CPU_module_and_TorchScript_bit_identical']
    assert diagnostic['export_all_weights_and_normalizer_equal_checkpoint']
    assert diagnostic['checkpoint_sha256']==sha(checkpoint) and diagnostic['actor_sha256']==sha(actor)
    assert diagnostic['normalizer_samples']==12288000 and diagnostic['checkpoint_iteration']==999
    assert not (root/'formal_v1/nonlinear_tire_training_audit.json').exists()
    assert sha(a.motion)=='da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254'
    assert sha(a.model)=='bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513'
    scripts=a.k_snapshot.resolve()/'scripts';sys.path.append(str(scripts))
    import eval_ppo_backend_400hz_mujoco as native
    import ppo_backend_400hz_contract as recording
    assert sha(native.__file__)=='9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4'
    evaluator,legacy_root=recording.verify_legacy(a.legacy_evaluator,native.BASE_SHA256)
    manifest=json.loads((root/'formal_v1/manifest.json').read_text())
    contract=copy.deepcopy(manifest['contract'])
    assert (contract['physics_dt'],contract['decimation'],contract['actor_dim'],contract['action_dim'])==(.0025,8,51,8)
    assert (contract['leg_kp'],contract['leg_kd'],contract['wheel_torque_scale_nm'])==(500.,10.,12.)
    contract.update(contract_status='posthoc_recovered_checkpoint_diagnostic_not_qualified_formal_export',
        full_horizon_diagnostic=True,domain_randomization='disabled during CPU diagnostic',
        policy_algorithm='PPO',policy_training_backend='physx',actor_includes_observation_normalizer=True,
        actor_output_is_unclipped=True,source_actor_sha256=sha(actor),source_checkpoint_sha256=sha(checkpoint),
        formal_training_audit_verified=False,first_attempt_final_native_health_available=False,
        evaluation_backend='native CPU MuJoCo',posthoc_diagnostic_only=True)
    contract['task_profile']['contact_friction']=.7
    inputs.mkdir()
    write(inputs/'contract.json',contract)
    generated=native.materialize_source(evaluator.read_text(),legacy_root)
    plan=dict(mu=.7)

    def finish(report,arrays,current):
        proof=recording.verify_recording(output/'rollout.npz','native CPU MuJoCo')
        assert report['status']=='completed' and report['completed_full_reference']
        assert report['hidden_resets']==0 and report['root_state_writes']==1
        assert report['policy_steps']==349 and report['physics_steps']==2792
        assert report['policy_sha256']==sha(actor)
        metadata=dict(kind='posthoc_recovered_checkpoint_CPU_MuJoCo_diagnostic',status='completed_requires_analysis',
            formally_qualified_training_cohort=False,first_attempt_final_live_native_health_missing=True,
            qualified_formal_comparison_admission=False,new_training_or_GPU_job=False,
            checkpoint_sha256=sha(checkpoint),actor_sha256=sha(actor),normalizer_samples=12288000,
            checkpoint_iteration=999,CPU_source_export_identity_verified=True,mu=.7,
            physics_dt_s=.0025,policy_dt_s=.02,native_states=2793,policy_decisions=349,
            source_sha256={str(Path(p).resolve()):sha(p) for p in (__file__,native.__file__,evaluator)},
            materialized_evaluator_sha256=hashlib.sha256(generated.encode()).hexdigest(),
            recording_proof=proof,task_success_claimed=False)
        report['posthoc_recovered_checkpoint_diagnostic']=metadata
        write(output/'posthoc_diagnostic_status.json',metadata)

    namespace=dict(__name__='_posthoc_recovered_checkpoint_native_CPU',__file__=str(Path(__file__).resolve()),
        _new_plan=plan,_new_audit_materials=native.audit_materials,_new_finish_report=finish)
    old=list(sys.path)
    try:
        sys.path[:0]=[str(evaluator.parent),str(legacy_root)]
        exec(compile(generated,'<posthoc_recovered_checkpoint_native_CPU>','exec'),namespace)
        policy=namespace['load_torchscript'](actor)
        return namespace['run_evaluation'](policy,a.motion,a.model,output,policy_path=actor,
            no_visual_mesh=True,contract_path=inputs/'contract.json')
    finally:sys.path[:]=old


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('first-attempt','output','k-snapshot','motion','model','legacy-evaluator'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
