"""Serial matched SAC smoke/train/evaluate sequence after contact-model gates."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def run(a):
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    status=dict(status='waiting_for_wheel_factor_gate',pid=os.getpid(),jobs=[],
        one_simulation_job_at_a_time=True,process_signals_sent=False,automatic_retry=False,
        training_budget=dict(algorithm='SAC',backends=['PhysX','MuJoCo-Warp'],seed=42,
            transitions_each=12288000,num_envs=512,policy_steps=24000),
        evaluation=dict(backends=['PhysX','native CPU MuJoCo'],formal_episodes=68,smoke_episodes=4,
            sample_size_per_cell=1,statistical_success_rate_claimed=False))
    def save():(a.output_dir/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    source_hashes={str(p):sha(p) for p in sorted(a.source.glob('*.py'))}
    status['source_sha256']=source_hashes;save()
    def load(path,name):
        previous=list(sys.path)
        try:
            sys.path.insert(0,str(path.parent))
            spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        finally:sys.path[:]=previous
    helper=load(a.continuation_helper,'sac_queue_helper')
    probe=load(a.gpu_probe,'sac_queue_gpu').query_gpu
    previous=None
    if a.previous_sequence is not None:
        previous=read(a.previous_sequence/'status.json')
        assert previous['status']=='stopped_without_retry'
        assert previous['training_budget']==status['training_budget']
        status['reused_preflight_source']=str(a.previous_sequence)
        status['reused_preflight_status_sha256']=sha(a.previous_sequence/'status.json')
    pair=read(a.training_pair/'backend_400hz_pair_status.json')
    command=pair['children'][0]['command']
    def original(flag):return command[command.index('--'+flag)+1]
    frozen=Path(command[2]).parents[1]
    common=['--frozen-source',str(frozen),'--model',original('model'),'--motion',original('motion'),
        '--asset-path',original('asset-path'),'--tron1-source',original('tron1-source'),
        '--beyondmimic-source',original('beyondmimic-source'),'--mjlab-source',original('mjlab-source'),
        '--device-audit-source',original('isaac-device-audit-source'),'--seed','42','--device',f'cuda:{a.physical_gpu_index}']
    evaluation=['--validation-source',str(a.validation_source),'--legacy-source',str(a.legacy_source),
        '--motion',original('motion'),'--model',original('model'),'--asset-path',original('asset-path'),
        '--beyondmimic-source',original('beyondmimic-source'),
        '--physical-gpu-index',str(a.physical_gpu_index)]
    def child(name,argv,gpu):
        if any(sha(p)!=h for p,h in source_hashes.items()):raise RuntimeError('Frozen queue source changed')
        row=dict(name=name,status='pending',command=argv)
        status['jobs'].append(row)
        if gpu:
            def waiting(observation):
                status.update(status='waiting_for_idle_gpu',latest_gpu_observation=observation);save()
            row['gpu_gate']=helper.wait_idle(probe,a.physical_gpu_index,time.monotonic()+86400,waiting)
        env=os.environ.copy();env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        else:
            if 'CUDA_VISIBLE_DEVICES' in env:raise ValueError('Do not mask evaluation/training GPU')
        status['status']='running_'+name;row['status']='running';save()
        with (a.output_dir/(name+'.log')).open('xb') as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
            row['pid']=process.pid;save();code=process.wait()
        row['returncode']=code
        if code:raise RuntimeError(name+' failed; outputs retained, no retry')
        row['status']='completed_process_pending_verification';save()
        return row
    def train(backend,smoke):
        name=backend+('_smoke' if smoke else '_formal');folder=a.output_dir/name
        if smoke and previous is not None:
            candidates=[j for j in previous['jobs'] if j['name']==name and j['status']=='completed_verified']
            if len(candidates)==1:
                source=a.previous_sequence/name
                verifier=load(a.source/'eval_sac_backend_400hz.py','sac_reuse_verifier')
                proof=verifier.cohort(source,a.validation_source/'config/ppo_backend_400hz_comparison_v1.json',True)
                assert proof['smoke_only'] and proof['training_backend']==backend
                assert candidates[0]['run_status_sha256']==sha(source/'run_status.json')
                hashes=helper.copy_verified_tree(source,folder)
                copied=verifier.cohort(folder,a.validation_source/'config/ppo_backend_400hz_comparison_v1.json',True)
                assert proof['input_sha256']==copied['input_sha256']
                status['jobs'].append(dict(name=name,status='completed_verified',reused=True,
                    reused_from=str(source),reused_file_sha256=hashes,run_status_sha256=sha(folder/'run_status.json')))
                save();return folder
        argv=[str(a.python),'-u',str(a.source/'train_sac_backend_400hz.py'),*common,
            '--backend',backend,'--num-envs','32' if smoke else '512','--policy-steps','48' if smoke else '24000',
            '--output-dir',str(folder)]
        if smoke:argv.append('--smoke-only')
        row=child(name,argv,True);proof=read(folder/'run_status.json')
        assert proof['status']==('smoke_passed' if smoke else 'training_completed')
        assert proof['transitions']==(1536 if smoke else 12288000)
        assert proof['actor_sha256']==sha(folder/'policy.pt') and proof['checkpoint_sha256']==sha(folder/'model_final.pt')
        row.update(status='completed_verified',run_status_sha256=sha(folder/'run_status.json'));save()
        return folder
    cases=[]
    def evaluate(origin,engine,mu,smoke):
        phase='pipeline_smoke' if smoke else 'formal'
        training=a.output_dir/(origin+('_smoke' if smoke else '_formal'))
        name=f'{phase}_{origin}_{engine}_mu{mu:.3f}'.replace('.','p')
        folder=a.output_dir/'evaluations'/phase/origin/engine/f'mu{mu:.3f}'.replace('.','p')
        argv=[str(a.python),'-u',str(a.source/'eval_sac_backend_400hz.py'),*evaluation,
            '--cohort-dir',str(training),'--backend',engine,'--mu',str(mu),'--output-dir',str(folder)]
        if smoke:argv.append('--allow-smoke-policy')
        row=child(name,argv,engine=='physx')
        report=read(folder/'report.json');proof=report['matched_sac_backend_evaluation']
        recording=folder/('physics_trajectory.npz' if engine=='physx' else 'rollout.npz')
        assert proof['status']=='verified' and proof['recording_sha256']==sha(recording) and proof['algorithm']=='SAC'
        assert proof['actor_sha256']==sha(training/'policy.pt') and proof['mu']==mu and proof['smoke_only']==smoke
        row.update(status='completed_verified',report_sha256=sha(folder/'report.json'));save()
        cases.append(dict(phase=phase,training_backend=origin,engine_tag=engine,mu=mu,directory=str(folder),
            report_sha256=sha(folder/'report.json'),recording_sha256=sha(recording),actor_sha256=proof['actor_sha256'],status='completed_verified'))
        (a.output_dir/'evaluation_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    try:
        deadline=time.monotonic()+86400
        while True:
            prior=read(a.wheel_suite/'status.json')
            if prior['status']=='completed_requires_analysis':break
            if 'failed' in prior['status'] or 'stopped' in prior['status'] or time.monotonic()>deadline:
                raise RuntimeError('Wheel-factor predecessor has not passed; no automatic training')
            time.sleep(15)
        analysis=a.output_dir/'wheel_factor_analysis'
        row=child('wheel_factor_analysis',[str(a.python),str(a.source/'analyze_wheel_model_factors.py'),
            '--suite',str(a.wheel_suite),'--output-dir',str(analysis)],False)
        report=read(analysis/'report.json');assert report['status']=='completed_verified' and report['formal_cases']==18
        row.update(status='completed_verified',report_sha256=sha(analysis/'report.json'));save()
        for backend in ('physx','mjlab'):train(backend,True)
        for origin in ('physx','mjlab'):
            for engine in ('physx','mujoco'):evaluate(origin,engine,.7,True)
        for backend in ('physx','mjlab'):train(backend,False)
        initial=[read(a.output_dir/(b+'_formal')/'manifest.json')['initial_actor_state_sha256'] for b in ('physx','mjlab')]
        assert initial[0]==initial[1],'SAC backends must have identical initialized actors'
        grid=read(a.validation_source/'config/ppo_backend_400hz_comparison_v1.json')['validation']['friction_grid']
        for mu in [.7]+[m for m in grid if m!=.7]:
            for origin in ('physx','mjlab'):
                for engine in ('physx','mujoco'):evaluate(origin,engine,mu,False)
        assert len(cases)==72
        row=child('matched_algorithm_analysis',[str(a.python),str(a.source/'analyze_matched_ppo_sac.py'),
            '--sac-root',str(a.output_dir),'--ppo-root',str(a.ppo_matrix),'--validation-source',str(a.validation_source),
            '--reference',original('motion'),'--model',original('model'),'--legacy-source',str(a.legacy_source),
            '--output-dir',str(a.output_dir/'comparison_analysis')],False)
        assert read(a.output_dir/'comparison_analysis/report.json')['status']=='completed_verified'
        row.update(status='completed_verified',report_sha256=sha(a.output_dir/'comparison_analysis/report.json'))
        status['status']='completed_with_analysis'
    except BaseException as error:
        status.update(status='stopped_without_retry',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['finished_unix_time']=time.time();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('output-dir','source','python','continuation-helper','gpu-probe','training-pair','validation-source',
              'legacy-source','wheel-suite','ppo-matrix'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--physical-gpu-index',type=int,choices=(3,4,5),required=True)
    p.add_argument('--previous-sequence',type=Path)
    run(p.parse_args())
