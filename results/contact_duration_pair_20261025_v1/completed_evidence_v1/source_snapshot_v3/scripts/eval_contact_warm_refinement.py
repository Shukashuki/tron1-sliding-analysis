"""Original held-action native validation of explicitly warm-refined K/M PPO.

Reuse only the pinned native plan/recording lifecycle. Training admission uses
the new strict warm-start verifier, never the old from-scratch 12M verifier.
No training reward owner, motor ramp or extra simulation step is installed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import train_contact_warm_refinement as training
import verify_contact_warm_refinement as verification


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize_parent_source(source):
    assert hashlib.sha256(source.encode()).hexdigest()=='cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c'
    pairs=[('COHORT_ID = "tron1_ground_contact_v2k_any_wheel_400hz_v1"','COHORT_ID = "'+training.COHORT_ID+'"'),
        ('require(config == ppo_config(42, config["save_interval"]), "PPO runner configuration differs")',
         'require(config == _refine_config(ppo_config(42, config["save_interval"])), "PPO refinement runner configuration differs")')]
    for old,new in pairs:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    assert source.count('ground_contact_v2k_evaluation')==3
    source=source.replace('ground_contact_v2k_evaluation','contact_warm_refinement_evaluation')
    compile(source,'<warm_refinement_native_parent>','exec');return source


def load_parent(k_snapshot,base_snapshot):
    path=Path(k_snapshot).resolve()/'scripts/eval_ppo_ground_contact_v2k.py'
    source=materialize_parent_source(path.read_text())
    namespace=dict(__name__='_warm_refinement_native_parent',__file__=str(path),_refine_config=training.refinement_config)
    sys.path.insert(1,str(Path(base_snapshot).resolve()/'scripts'))
    sys.path.append(str(path.parent));exec(compile(source,'<warm_refinement_native_parent>','exec'),namespace)
    namespace['__file__']=str(Path(__file__).resolve())
    namespace['verify_training']=lambda d,p,allow_smoke=False:verification.verify_training(d,p,allow_smoke=allow_smoke)
    common=namespace['common']
    def provenance(c):
        p=common.provenance(c)
        p.update(kind='warm_contact_refinement_native_evaluation',cohort_id=training.COHORT_ID,
            reward_variant=c['variant'],reward_definition_version=c['contract']['reward_definition_version'],
            initial_policy_state_sha256=training.WARM_STATE_SHA,source_K_checkpoint_sha256=training.CHECKPOINT_SHA,
            source_K_actor_sha256=training.ACTOR_SHA,from_scratch=False,historical_checkpoint_used=True,
            additional_refinement_transitions=c['completion']['training_environment_steps'],
            cumulative_training_transitions=c['completion']['cumulative_training_environment_steps'],
            normalization_applications_per_inference=1,extra_normalization=False,
            single_wheel_support_allowed=True,absolute_height_gate=False,airborne_failure=False,
            leg_command_execution='original_position_target_hold',hardware_ready=False)
        return p
    namespace['provenance']=provenance
    return SimpleNamespace(**namespace)


def main(argv=None):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--k-snapshot',type=Path,required=True);p.add_argument('--base-snapshot',type=Path,required=True)
    own,rest=p.parse_known_args(argv)
    parent=load_parent(own.k_snapshot,own.base_snapshot);plan=parent.prepare_plan(rest);assert plan['mu']==.7
    plan['wrapper_sha256']=sha(__file__)
    ledger={str(Path(f).resolve()):sha(f) for f in [__file__,training.__file__,verification.__file__]}
    def metadata(report,current,deployed):
        c=current['cohort'];info=parent.provenance(c)
        backend=current['args'].backend;name='rollout.npz' if backend=='mujoco' else 'physics_trajectory.npz'
        info.update(status='completed_requires_analysis',validation_backend=backend,evaluation_mu=.7,
            native_states=2793,policy_decisions=349,recording_sha256=sha(current['output']/name),
            source_sha256=ledger,training_completion_audit_sha256=c['input_sha256']['cohort_completion_audit.json'],
            task_success_claimed=False)
        assert all(sha(f)==h for f,h in ledger.items())
        report['warm_contact_evaluation']=info;deployed['warm_contact_evaluation']=copy.deepcopy(info)
        with (current['output']/'warm_contact_evaluation_status.json').open('x') as f:json.dump(info,f,indent=2,allow_nan=False)
    if plan['args'].backend=='mujoco':
        native=parent.native;generated=native.materialize_source(plan['evaluator'].read_text(),plan['legacy_root'])
        plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()
        def finish(report,arrays,current):
            parent._finish(report,arrays,current);metadata(report,current,report['contract'])
        namespace=dict(__name__='_warm_contact_native_mujoco',__file__=str(Path(__file__).resolve()),
            _new_plan=plan,_new_audit_materials=native.audit_materials,_new_finish_report=finish)
        previous=list(sys.path)
        try:
            sys.path[:0]=[str(plan['evaluator'].parent),str(plan['legacy_root'])]
            exec(compile(generated,'<warm_contact_native_mujoco>','exec'),namespace)
            policy=namespace['load_torchscript'](plan['cohort']['actor'])
            return namespace['run_evaluation'](policy,plan['motion'],plan['args'].model,plan['output'],
                policy_path=plan['cohort']['actor'],no_visual_mesh=True,contract_path=plan['contract_path'])
        finally:sys.path[:]=previous
    import eval_ppo_backend_400hz_isaac as physx
    generated=physx.materialize_source(plan['evaluator'].read_text(),plan['legacy_root'])
    plan['generated_sha256']=hashlib.sha256(generated.encode()).hexdigest()
    def materials(env,current):
        import torch
        torch.cuda.set_device(current['args'].physical_gpu_index)
        return physx.audit_materials(env,current)
    def runner(cohort,device):
        r=parent.common.StrictPPOExportRunner(cohort,device);r.inference_provenance=parent.provenance(cohort);return r
    def finish(report,contract,current,raw):
        parent._finish(report,raw,current,contract=contract);metadata(report,current,contract)
    namespace=dict(__name__='_warm_contact_native_physx',__file__=str(Path(__file__).resolve()),
        _new_plan=plan,_new_configure=physx.configure,_new_audit_materials=materials,
        _new_runner=runner,_new_finish_report=finish)
    previous=sys.argv
    try:
        os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID');sys.argv=[str(Path(__file__).resolve()),*plan['inherited_argv']]
        exec(compile(generated,'<warm_contact_native_physx>','exec'),namespace);namespace['main']()
    finally:sys.argv=previous


if __name__=='__main__':main(sys.argv[1:])
