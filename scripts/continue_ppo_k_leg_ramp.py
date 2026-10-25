"""Wait for the exact ramp training driver, then two nominal episodes.

One GPU evaluation at a time on fresh fully idle GPU3/5/4; native CPU MuJoCo
uses the same ramp and remains a cross-engine transfer. No retries, tuning or friction sweeps.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def run(a):
    root=a.workspace.resolve();snapshot=a.evaluation_snapshot.resolve()
    with (root/'evaluation_continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=root/'nominal_v1';out.mkdir(exist_ok=False)
        status_path=out/'status.json'
        status=dict(kind='K_leg_ramp_PPO_nominal_two_engines',status='preflight',cases=[],task_success_claimed=False)
        write(status_path,status)
        try:
            assert sha(a.idle_audit_source)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
            spec=importlib.util.spec_from_file_location('_leg_ramp_validation_idle_audit',a.idle_audit_source)
            audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
            launch=read(root/'driver_v1_launch.json')
            assert launch['pid']==a.expected_driver_pid
            assert Path(launch['argv'][3]).resolve()==a.training_snapshot.resolve()/'scripts/run_ppo_k_leg_ramp.py'
            observed=audit.process_snapshot(a.expected_driver_pid)
            if observed is not None and observed['state']!='Z':
                assert observed['argv']==launch['argv']
                assert observed['start_ticks']==launch['start_ticks']
            expected=observed
            deadline=time.monotonic()+a.deadline_seconds
            m=read(root/(snapshot.name+'.manifest.json'))
            ledger={str(snapshot/x['path']):x['sha256'] for x in m['files']}
            def bound():
                assert time.monotonic()<deadline,'Monitoring deadline reached; all owned jobs preserved'
                assert all(sha(p)==h for p,h in ledger.items())
            status.update(status='waiting_for_exact_training_driver',driver_process_at_start=expected)
            write(status_path,status)
            while True:
                bound();observed=audit.process_snapshot(a.expected_driver_pid)
                if observed is None or observed['state']=='Z':break
                assert expected is not None
                assert all(observed[k]==expected[k] for k in ('pid','uid','start_ticks','argv'))
                time.sleep(15.)
            driver=read(root/'driver_status.json')
            assert driver['status']=='formal_completed_verified_nominal_validation_pending'
            assert driver['formal_proof']['transitions']==12288000
            assert all(row['status']=='completed_strictly_verified' for row in driver['children'])
            status['formal_training_proof']=driver['formal_proof']
            env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
            env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(a.training_snapshot.resolve()/'scripts')+os.pathsep+env.get('PYTHONPATH',''))
            template=read(a.training_snapshot/'config/ppo_k_leg_ramp_launch_template.json')['argv']
            option=lambda f:template[template.index(f)+1]
            for case,backend in [('physx','physx'),('mujoco','mujoco')]:
                bound();gpus=[];selected=None
                if backend=='physx':
                    status['status']='waiting_for_fully_idle_GPU_'+case;write(status_path,status)
                    while True:
                        bound();gpus=audit.query_gpus();selected=next((g for g in gpus if g['idle']),None)
                        if selected:break
                        time.sleep(15.)
                command=[str(a.python),'-B','-u',str(snapshot/'scripts/eval_ppo_k_leg_ramp.py'),
                    '--k-snapshot',option('--k-snapshot'),'--training-dir',str(root/'formal_v1'),
                    '--protocol',str(a.training_snapshot/'config/ppo_k_leg_ramp_400hz_v1.json'),
                    '--backend',backend,'--mu','.7','--motion-file',option('--motion'),
                    '--output-dir',str(out/case),'--legacy-evaluator',str(a.legacy_physx if backend=='physx' else a.legacy_mujoco)]
                child_env=env.copy()
                if backend=='physx':
                    command+=['--asset-path',option('--asset-path'),'--physical-gpu-index',str(selected['physical_index'])]
                    child_env['TRON1_BEYONDMIMIC_ROOT']=option('--beyondmimic-source')
                else:
                    command+=['--model',option('--model')];child_env['CUDA_VISIBLE_DEVICES']=''
                with (out/(case+'.log')).open('xb') as log:
                    child=subprocess.Popen(command,cwd=snapshot,env=child_env,stdout=log,stderr=subprocess.STDOUT)
                    row=dict(id=case,pid=child.pid,argv=command,status='running',gpu_audit=gpus,
                        selected_gpu=selected,process_at_launch=audit.process_snapshot(child.pid))
                    status['cases'].append(row);status['status']='running_'+case;write(status_path,status)
                    while child.poll() is None:bound();time.sleep(2.)
                row['returncode']=child.returncode
                assert child.returncode==0,('Nominal episode failed; no retry',case,child.returncode)
                metadata=read(out/case/'leg_ramp_evaluation_status.json')
                report=read(out/case/'report.json')
                assert report['leg_ramp_evaluation']==metadata
                assert metadata['native_states']==2793 and metadata['policy_decisions']==349
                assert metadata['validation_backend']==backend
                assert metadata['normalized_actor_sha256']==driver['formal_proof']['actor_sha256']
                name='physics_trajectory.npz' if backend=='physx' else 'rollout.npz'
                assert sha(out/case/name)==metadata['recording_sha256']
                assert metadata['command_audit']['native_target_applications']==2792
                assert sha(out/case/'leg_command_trajectory.npz')==metadata['command_audit']['command_recording_sha256']
                row.update(status='completed_requires_independent_analysis',metadata_sha256=sha(out/case/'leg_ramp_evaluation_status.json'))
                write(status_path,status)
            status['status']='two_nominal_episodes_completed_require_analysis_and_render';write(status_path,status)
        except BaseException as error:
            status.update(status='failed_no_retry_or_process_signal',error=repr(error));write(status_path,status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('workspace','evaluation-snapshot','training-snapshot','idle-audit-source','python','legacy-physx','legacy-mujoco'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-driver-pid',type=int,required=True)
    p.add_argument('--deadline-seconds',type=int,default=7200)
    run(p.parse_args())
