"""Deterministic SAC in the same frozen 400 Hz PhysX/CPU-MuJoCo plants.

Reuse only material, actuator, recording and lifecycle adapters. SAC provenance
is checked here; no PPO cohort/export checker is bypassed or impersonated.
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)


def cohort(path,protocol_path,allow_smoke):
    directory=path.resolve()
    names=('run_status.json','manifest.json','policy_contract.json','runner_config.json','model_final.pt','policy.pt')
    paths={n:directory/n for n in names};hashes={n:sha(p) for n,p in paths.items()}
    status,manifest,contract=(read(paths[n]) for n in ('run_status.json','manifest.json','policy_contract.json'))
    smoke=manifest['smoke_only']
    require(not smoke or allow_smoke,'Smoke policy needs explicit pipeline-only mode')
    require(status['status']==('smoke_passed' if smoke else 'training_completed'),'Training incomplete')
    require(status['actor_sha256']==hashes['policy.pt'] and status['checkpoint_sha256']==hashes['model_final.pt']
            and status['manifest_sha256']==hashes['manifest.json']
            and status['policy_contract_sha256']==hashes['policy_contract.json'],'Training evidence changed')
    require(contract['policy_algorithm']=='SAC' and manifest['algorithm']=='SAC'
            and read(paths['runner_config.json'])['algorithm']=='SAC','Not a SAC cohort')
    require(contract['physics_dt']==.0025 and contract['decimation']==8
            and contract['actor_output_transform']=='tanh','Clock/action ABI differs')
    require(status['transitions']==(1536 if smoke else 12288000),'Incomplete transition budget')
    require(sha(protocol_path)==manifest['task_protocol_sha256'],'Shared task protocol differs')
    require(manifest['runtime_audit']['status']=='verified','Training runtime unverified')
    for source,expected in manifest['source_sha256'].items():
        require(sha(source)==expected,'Training source changed: '+source)
    return dict(directory=directory,paths=paths,input_sha256=hashes,checkpoint=paths['model_final.pt'],
                actor=paths['policy.pt'],contract=contract,manifest=manifest,protocol=read(protocol_path),
                training_backend=manifest['backend'],smoke_only=smoke)


class SACExportRunner:
    def __init__(self,cohort,device):
        self.cohort,self.device,self.alg=cohort,device,None
        self.inference_provenance=dict(inference_algorithm='SAC',actor_sha256=cohort['input_sha256']['policy.pt'],
            observation_normalization='identity affine embedded in scripted actor',output='deterministic tanh',
            training_backend=cohort['training_backend'],smoke_only=cohort['smoke_only'])
    def load(self,checkpoint,*,load_optimizer=False,map_location=None):
        import torch
        require(self.alg is None and not load_optimizer,'Inference load once; no optimizer')
        require(Path(checkpoint).resolve()==self.cohort['checkpoint'] and sha(checkpoint)==self.cohort['input_sha256']['model_final.pt'],
                'Wrong SAC checkpoint')
        require(sha(self.cohort['actor'])==self.cohort['input_sha256']['policy.pt'],'SAC actor changed')
        device=map_location or self.device
        actor=torch.jit.load(str(self.cohort['actor']),map_location=device).eval()
        class SACView(torch.nn.Module):
            is_recurrent=False
            obs_groups={'policy':['policy']}
            def __init__(self,actor):
                super().__init__();self.actor=actor;self.actor_obs_normalizer=torch.nn.Identity()
            def get_actor_obs(self,obs):
                x=obs['policy'];require(x.ndim==2 and x.shape[1]==51 and torch.isfinite(x).all(),'Invalid actor observation')
                return x
            def forward(self,obs):
                y=self.actor(self.get_actor_obs(obs))
                require(y.ndim==2 and y.shape[1]==8 and torch.isfinite(y).all() and (y.abs()<=1).all(),'Invalid SAC action')
                return y
        self.alg=SimpleNamespace(policy=SACView(actor).to(device).eval())
    def get_inference_policy(self,device=None):return self.alg.policy


def run(a):
    if a.backend=='physx':
        require(a.beyondmimic_source is not None and a.beyondmimic_source.is_dir(),'PhysX needs explicit existing BeyondMimic source')
        os.environ['TRON1_BEYONDMIMIC_ROOT']=str(a.beyondmimic_source.resolve())
    sys.path.insert(0,str(a.validation_source/'scripts'))
    import ppo_backend_400hz_contract as common
    import eval_ppo_backend_400hz_isaac as physx
    import eval_ppo_backend_400hz_mujoco as mj
    protocol=a.validation_source/'config/ppo_backend_400hz_comparison_v1.json'
    common.verify_protocol(protocol)
    c=cohort(a.cohort_dir,protocol,a.allow_smoke_policy)
    engine='PhysX' if a.backend=='physx' else 'native CPU MuJoCo'
    plan=common.prepare_inputs(c,a.motion,a.output_dir,a.mu,engine,
         **({'asset':a.asset_path} if a.backend=='physx' else {'model':a.model}))
    module=physx if a.backend=='physx' else mj
    legacy=a.legacy_source/'scripts'/('eval_tracking_friction_isaac.py' if a.backend=='physx' else 'eval_tracking_mujoco.py')
    evaluator,legacy_root=common.verify_legacy(legacy,module.BASE_SHA256)
    plan.update(args=a,evaluator=evaluator,legacy_root=legacy_root)
    generated=module.materialize_source(evaluator.read_text(),legacy_root)
    audit=dict(kind='matched_task400hz_SAC_evaluation',algorithm='SAC',training_backend=c['training_backend'],
        validation_backend=engine,smoke_only=c['smoke_only'],task_protocol_sha256=sha(protocol),
        actor_sha256=c['input_sha256']['policy.pt'],checkpoint_sha256=c['input_sha256']['model_final.pt'],
        cohort_files_sha256=c['input_sha256'],wrapper_sha256=sha(__file__),
        physics_adapter_sha256=sha(Path(module.__file__)),legacy_evaluator_sha256=sha(evaluator),
        generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        physics_dt_s=.0025,policy_dt_s=.02,mu=a.mu,additional_training_updates=0,
        contact_law_equivalence_claimed=False,hardware_ready=False)
    def finish(report,second,plan,arrays=None):
        for name,p in c['paths'].items():require(sha(p)==c['input_sha256'][name],'Cohort changed during evaluation')
        recording=plan['output']/('physics_trajectory.npz' if a.backend=='physx' else 'rollout.npz')
        proof=common.verify_recording(recording,engine)
        if a.backend=='physx':
            require(report['status']=='evaluated' and report['deterministic_start_frame']==0,'Incomplete PhysX episode')
            require(report['physics_clock_readback']['status']=='verified','Unverified PhysX clock')
            require(second['policy_algorithm']=='SAC','Export mislabeled')
            audit['deployment_actor_sha256']=sha(plan['output']/'actor_normalized.pt')
        else:
            require(report['status']=='completed' and report['completed_full_reference']
                    and report['hidden_resets']==0 and report['root_state_writes']==1,'Incomplete or reset-spliced MuJoCo episode')
        audit.update(proof)
        report['matched_sac_backend_evaluation']=copy.deepcopy(audit)
        common.write_json(plan['input_dir']/'evaluation_source_audit.json',audit)
    namespace={'__name__':'_matched_task_sac_evaluator','__file__':str(Path(__file__).resolve()),
               '_new_plan':plan,'_new_finish_report':finish}
    previous_path,previous_argv=list(sys.path),sys.argv
    try:
        sys.path[:0]=[str(evaluator.parent),str(legacy_root)]
        if a.backend=='physx':
            require('CUDA_VISIBLE_DEVICES' not in os.environ,'Use unmasked GPU')
            os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
            index=a.physical_gpu_index
            kit=f'--/renderer/activeGpu={index} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={index} --/physics/sceneMultiGPUMode=0'
            args=['--motion-file',str(a.motion),'--asset-path',str(a.asset_path),'--checkpoint',str(c['checkpoint']),
                '--runner-config',str(c['paths']['runner_config.json']),'--output-dir',str(plan['output']),
                '--task-profile',str(plan['task_profile_path']),'--evaluation-contact-friction',str(a.mu),
                '--num-envs','1','--seed','42','--record-physics','--verify-unmasked-device',
                '--device',f'cuda:{index}','--headless','--kit_args',kit]
            if a.allow_smoke_policy:args.append('--allow-smoke-policy')
            namespace.update(_new_configure=physx.configure,_new_audit_materials=physx.audit_materials,_new_runner=SACExportRunner)
            sys.argv=[str(Path(__file__)),*args]
            exec(compile(generated,'<matched_task_sac_physx>','exec'),namespace)
            namespace['main']()
        else:
            namespace['_new_audit_materials']=mj.audit_materials
            exec(compile(generated,'<matched_task_sac_mujoco>','exec'),namespace)
            policy=namespace['load_torchscript'](c['actor'])
            namespace['run_evaluation'](policy,plan['motion'],a.model,plan['output'],policy_path=c['actor'],
                                        no_visual_mesh=True,contract_path=plan['contract_path'])
    finally:
        sys.path[:]=previous_path;sys.argv=previous_argv


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('cohort-dir','validation-source','legacy-source','motion','model','asset-path','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backend',choices=('physx','mujoco'),required=True)
    p.add_argument('--mu',type=float,required=True)
    p.add_argument('--physical-gpu-index',type=int,choices=(3,4,5),default=3)
    p.add_argument('--allow-smoke-policy',action='store_true')
    p.add_argument('--beyondmimic-source',type=Path)
    run(p.parse_args())
