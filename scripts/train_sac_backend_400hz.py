"""SAC on the frozen 400 Hz PPO task, with explicit algorithm provenance.

Reuses the immutable environment/DR/runtime audit. SAC owns its replay,
optimizers, entropy and deterministic tanh actor; no PPO export impersonation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def run(a):
    if a.output_dir.exists() or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Fresh output and unmasked explicit GPU required')
    if a.seed != 42 or (a.num_envs,a.policy_steps) != ((32,48) if a.smoke_only else (512,24000)):
        raise ValueError('Frozen seed42 budget: smoke1536 or formal12288000 transitions')
    os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    if os.environ['CUDA_DEVICE_ORDER']!='PCI_BUS_ID':
        raise ValueError('Require PCI_BUS_ID CUDA ordering')
    sys.path.insert(0,str(a.frozen_source/'scripts'))
    import train_ppo_backend_400hz as shared
    if sha(a.frozen_source/'scripts/train_ppo_backend_400hz.py')!='5234070be219e59c61e6c4e4435e0dd0b2ab71df60d9c23886e507ed73692390':
        raise ValueError('Frozen 400 Hz task wrapper changed')
    protocol=a.frozen_source/'config/ppo_backend_400hz_comparison_v1.json'
    if sha(protocol)!='c54cd530884cffa22811ca2232cb2588eba3582deefc91d4d9675e4fca7fe30f':
        raise ValueError('Frozen task protocol changed')
    for key,expected in shared.INPUT_PINS.items():
        if sha(getattr(a,key))!=expected:raise ValueError('Input differs: '+key)
    prepared=shared.prepare_declared_inputs(protocol,a.output_dir.with_name(a.output_dir.name+'_inputs'))
    a.contract,a.task_profile=prepared['contract'],prepared['task_profile']
    plan=dict(protocol=shared.validate_protocol(json.loads(protocol.read_text())),device_audit_source=a.device_audit_source)
    source_files=[Path(__file__),Path(__file__).with_name('sac_tracking.py'),
                  Path(__file__).with_name('terminal_observation.py'),Path(__file__).with_name('matched_sac_terminal.py')]
    source_files += [a.frozen_source/name for name in shared.SOURCE_FILES]
    frozen={str(p):sha(p) for p in source_files}
    a.output_dir.mkdir(parents=True)
    status=dict(status='initializing',algorithm='SAC',backend=a.backend,smoke_only=a.smoke_only,
                trained_policy_claimed=False,transitions_requested=a.num_envs*a.policy_steps,seed=a.seed)
    write(a.output_dir/'run_status.json',status)
    start=time.monotonic();app=env=accumulator=None
    try:
        if a.backend=='physx':
            os.environ['TRON1_BEYONDMIMIC_ROOT']=str(a.beyondmimic_source)
            sys.path.insert(0,str(a.tron1_source))
            from isaaclab.app import AppLauncher
            app=shared.launch_app(AppLauncher,a,plan)
        import torch
        from sac_tracking import SACConfig,SACAgent,VectorReplay
        from train_ppo_backend import policy_state_hash
        shared.set_torch_device(torch,a,plan)
        if a.backend=='mjlab':
            pin=subprocess.check_output(['git','-C',str(a.mjlab_source),'rev-parse','HEAD'],text=True).strip()
            if pin!=shared.MJLAB_PIN:raise ValueError('mjlab source pin differs')
            sys.path.insert(0,str(a.mjlab_source/'src'))
            from mjlab_gpu_ppo_env import GpuTron1PPOEnv,TensorRslWrapper
            from matched_sac_terminal import make_gpu_terminal_class
            env=make_gpu_terminal_class(GpuTron1PPOEnv)(a.model,a.motion,a.contract,a.task_profile,
                num_envs=a.num_envs,seed=a.seed,training=True,dr_mode='mjlab_v1p4',device=a.device)
            if a.smoke_only:env.max_episode_length=4
            wrapped=TensorRslWrapper(env)
        else:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from training.tron1_tracking import make_env_cfg
            from training.tron1_drift import load_task_profile,install_drift_reward_sampling
            from physx_mjlab_recipe import apply_mjlab_recipe,recipe_audit
            from height_smoothness import install_height_smoothness
            from terminal_observation import make_terminal_observation_env_class
            task=load_task_profile(a.task_profile)
            cfg=make_env_cfg(a.motion,a.asset_path,a.num_envs,a.device,task_profile=task)
            cfg.seed=a.seed
            height=install_height_smoothness(cfg,json.loads(a.task_profile.read_text())['height_regularization'])
            shared.configure_physx(cfg,a,plan,apply_mjlab_recipe)
            if a.smoke_only:cfg.episode_length_s=.08
            env=make_terminal_observation_env_class()(cfg=cfg)
            accumulator=install_drift_reward_sampling(env,task)
            wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
        obs,_=wrapped.reset()
        for _ in range(8):
            obs,reward,done,extra=wrapped.step(torch.zeros(a.num_envs,8,device=a.device))
            if not all(torch.isfinite(obs[k]).all() for k in ('policy','critic')) or not torch.isfinite(reward).all():
                raise FloatingPointError('Nonfinite setup smoke')
        obs,_=wrapped.reset()
        if obs['policy'].shape!=(a.num_envs,51) or obs['critic'].shape!=(a.num_envs,120):
            raise ValueError('Observation ABI differs')
        metadata=env.metadata() if a.backend=='mjlab' else recipe_audit(env)
        runtime=shared.runtime_audit(env,a,metadata,plan)
        torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        config=SACConfig(initial_alpha=.2*env.step_dt)
        agent=SACAgent(config,a.device)
        capacity=4096 if a.smoke_only else 500000
        warmup=64 if a.smoke_only else 10000
        batch_size=64 if a.smoke_only else 1024
        replay=VectorReplay(capacity,51,120,8,a.device)
        initial_hash=policy_state_hash(agent.actor)
        manifest=dict(algorithm='SAC',algorithm_implementation='independent asymmetric twin-Q SAC in sac_tracking.py',
            backend=a.backend,training_backend_label='PhysX' if a.backend=='physx' else 'MuJoCo-Warp',
            seed=a.seed,num_envs=a.num_envs,policy_steps=a.policy_steps,smoke_only=a.smoke_only,
            source_sha256=frozen,environment=metadata,runtime_audit=runtime,
            task_profile=json.loads(a.task_profile.read_text()),contract=json.loads(a.contract.read_text()),
            task_protocol_path=str(protocol),task_protocol_sha256=sha(protocol),
            reference_sha256=sha(a.motion),model_sha256=sha(a.model),
            transitions_requested=a.num_envs*a.policy_steps,initial_actor_state_sha256=initial_hash,
            policy_initialization_seeded_after_setup=True,network_config=asdict(config),
            replay_capacity=capacity,batch_size=batch_size,warmup_transitions=warmup,updates_per_policy_step=1,
            gamma=.99,reward_scaling='unchanged dt-scaled 15-term environment reward',
            normalization='SAC fixed identity affine; historical matched PPO uses online RSL normalization',
            algorithm_differences=['replay vs on-policy batches','entropy optimization','tanh vs clipped Gaussian actor','fixed vs running observation normalization'],
            bootstrap='actual pre-reset physical successor; reference advanced/clamped; truncation bootstraps, termination masks',
            smoke_timeout_steps=4 if a.smoke_only else None,
            host=platform.node(),versions={n:importlib.metadata.version(n) for n in ('torch','mujoco','mujoco-warp','warp-lang')},
            evaluation_backend_warning='native CPU MuJoCo evaluation is not MuJoCo-Warp')
        write(a.output_dir/'manifest.json',manifest)
        write(a.output_dir/'runner_config.json',dict(algorithm='SAC',sac=asdict(config)))
        status.update(status='training',runtime_verified=True)
        write(a.output_dir/'run_status.json',status)
        transitions=terminated_count=truncated_count=captured_count=0;history=[];losses=None
        for step in range(a.policy_steps):
            actor_obs,critic_obs=obs['policy'].clone(),obs['critic'].clone()
            action=2*torch.rand((a.num_envs,8),device=a.device)-1 if transitions<warmup else agent.act(actor_obs)
            obs,reward,terminated,truncated,extra=env.step(action)
            next_obs=extra['bootstrap_observation']
            replay.add(actor_obs=actor_obs,critic_obs=critic_obs,actions=action,rewards=reward,
                bootstrap_actor_obs=next_obs['policy'],bootstrap_critic_obs=next_obs['critic'],
                terminated=terminated.bool(),truncated=truncated.bool(),discounts=torch.full_like(reward,.99))
            transitions+=a.num_envs
            terminated_count+=int(terminated.sum());truncated_count+=int(truncated.sum())
            captured_count+=len(extra['final_observation_env_ids'])
            if transitions>=warmup and replay.size>=batch_size:losses=agent.update(replay.sample(batch_size))
            if step==0 or (step+1)%100==0 or step+1==a.policy_steps:
                row=dict(policy_steps=step+1,transitions=transitions,updates=agent.updates,
                    reward_mean=float(reward.mean()),terminated=terminated_count,truncated=truncated_count,
                    captured_final_observations=captured_count,losses=losses,elapsed_s=time.monotonic()-start)
                history.append(row);status.update(row)
                write(a.output_dir/'run_status.json',status);write(a.output_dir/'training_log.json',history)
                print(json.dumps(row),flush=True)
            if not a.smoke_only and (step+1)%2400==0:agent.save(a.output_dir/f'model_step_{step+1:06d}.pt')
        if agent.updates==0 or (a.smoke_only and (truncated_count==0 or captured_count<truncated_count)):
            raise RuntimeError('Learner/truncation bootstrap integration was not exercised')
        if transitions!=a.num_envs*a.policy_steps:raise RuntimeError('Transition budget mismatch')
        checkpoint=a.output_dir/'model_final.pt';actor=a.output_dir/'policy.pt'
        agent.save(checkpoint);agent.export_actor(actor)
        exported=torch.jit.load(str(actor),map_location=a.device)
        with torch.no_grad():
            error=float((exported(obs['policy'])-agent.act(obs['policy'],deterministic=True)).abs().max())
        if error>1e-6:raise RuntimeError('Export differs from deterministic SAC actor')
        contract=json.loads(a.contract.read_text())
        contract.update(policy_algorithm='SAC',policy_training_backend=a.backend,
            cohort_id='sac_matched400_seed42_20261031_v1',
            smoke_only=a.smoke_only,trained_policy_claimed=not a.smoke_only,actor_file='policy.pt',
            actor_includes_observation_normalizer=True,actor_output_is_unclipped=False,
            actor_output_transform='tanh',actor_observation_groups=['policy'],
            checkpoint_sha256=sha(checkpoint),exported_policy_sha256=sha(actor),
            export_validation_max_abs_error=error,training_environment_transitions=transitions,
            contract_status='smoke_actor' if a.smoke_only else 'trained_completed_sac_actor',
            reference_file_sha256=sha(a.motion),hardware_ready=False)
        write(a.output_dir/'policy_contract.json',contract)
        if any(sha(p)!=h for p,h in frozen.items()):raise RuntimeError('Source changed during training')
        write(a.output_dir/'environment_final.json',env.metadata() if a.backend=='mjlab' else recipe_audit(env))
        status.update(status='smoke_passed' if a.smoke_only else 'training_completed',
            trained_policy_claimed=not a.smoke_only,checkpoint_sha256=sha(checkpoint),actor_sha256=sha(actor),
            policy_contract_sha256=sha(a.output_dir/'policy_contract.json'),manifest_sha256=sha(a.output_dir/'manifest.json'),
            export_max_abs_error=error,elapsed_s=time.monotonic()-start)
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_s=time.monotonic()-start)
        raise
    finally:
        write(a.output_dir/'run_status.json',status)
        if accumulator is not None:accumulator.close()
        if env is not None:env.close()
        if app is not None:app.close(skip_cleanup=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ('frozen-source','model','motion','asset-path','tron1-source','beyondmimic-source','mjlab-source','device-audit-source','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=('physx','mjlab'),required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--policy-steps',type=int,default=24000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke-only',action='store_true')
    run(p.parse_args())
