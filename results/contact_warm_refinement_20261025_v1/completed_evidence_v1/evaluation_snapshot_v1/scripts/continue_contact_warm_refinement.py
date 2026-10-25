"""Exact paired-driver wait, then four serial native nominal validations."""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def run(a):
    root=a.workspace.resolve();snapshot=a.evaluation_snapshot.resolve()
    with (root/'nominal_continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=root/'nominal_v1';out.mkdir(exist_ok=False);record=out/'status.json'
        state=dict(kind='paired_warm_K_M_four_native_nominal_episodes',status='preflight',cases=[],task_success_claimed=False)
        write(record,state)
        try:
            assert sha(a.idle_audit_source)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
            spec=importlib.util.spec_from_file_location('_warm_nominal_idle_and_handle',a.idle_audit_source)
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            launch=read(root/'driver_v1_launch.json');assert launch['pid']==a.expected_driver_pid
            assert Path(launch['argv'][3]).resolve()==a.training_snapshot.resolve()/'scripts/run_contact_warm_refinement.py'
            deadline=time.monotonic()+7200
            manifest=read(root/(snapshot.name+'.manifest.json'))
            ledger={str(snapshot/r['path']):r['sha256'] for r in manifest['files']}
            def bound():
                assert time.monotonic()<deadline,'Observation deadline; all compute is preserved'
                assert all(sha(p)==h for p,h in ledger.items())
            state['status']='waiting_for_exact_paired_training_driver';write(record,state)
            while True:
                bound();p=helper.process_snapshot(launch['pid'])
                if p is None or p['state']=='Z':break
                assert p['start_ticks']==launch['start_ticks'] and p['argv']==launch['argv']
                time.sleep(15.)
            driver=read(root/'driver_status.json')
            assert driver['status']=='both_formal_arms_completed_verified_nominal_pending'
            assert all(r['status']=='completed_strictly_verified' and r['returncode']==0 for r in driver['children'])
            for variant in ['K','M']:
                proof=driver['proofs']['formal_'+variant]
                assert proof['transitions']==2457600 and proof['cumulative_transitions']==14745600
                assert not proof['from_scratch'] and proof['historical_checkpoint_used']
            state['paired_formal_proofs']={v:driver['proofs']['formal_'+v] for v in ['K','M']}
            template=read(a.training_snapshot/'config/contact_warm_refinement_launch_template.json')['argv']
            option=lambda f:template[template.index(f)+1]
            env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
            env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(a.training_snapshot.resolve()/'scripts')+os.pathsep+env.get('PYTHONPATH',''))
            for variant in ['K','M']:
                for backend in ['physx','mujoco']:
                    bound();case=variant+'_'+backend;selected=None;gpus=[]
                    if backend=='physx':
                        state['status']='waiting_for_fully_idle_GPU_'+case;write(record,state)
                        while True:
                            bound();gpus=helper.query_gpus();selected=next((g for g in gpus if g['idle']),None)
                            if selected:break
                            time.sleep(15.)
                    command=[str(a.python),'-B','-u',str(snapshot/'scripts/eval_contact_warm_refinement.py'),
                        '--base-snapshot',option('--base-snapshot'),'--k-snapshot',option('--k-snapshot'),
                        '--training-dir',str(root/('formal_'+variant)),
                        '--protocol',str(a.training_snapshot/'config/contact_warm_refinement_400hz_v1.json'),
                        '--backend',backend,'--mu','.7','--motion-file',option('--motion'),'--output-dir',str(out/case),
                        '--legacy-evaluator',str(a.legacy_physx if backend=='physx' else a.legacy_mujoco)]
                    child_env=env.copy()
                    if backend=='physx':
                        command+=['--asset-path',option('--asset-path'),'--physical-gpu-index',str(selected['physical_index'])]
                        child_env['TRON1_BEYONDMIMIC_ROOT']=option('--beyondmimic-source')
                    else:
                        command+=['--model',option('--model')];child_env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    with (out/(case+'.log')).open('xb') as log:
                        p=subprocess.Popen(command,cwd=snapshot,env=child_env,stdout=log,stderr=subprocess.STDOUT)
                        row=dict(id=case,variant=variant,backend=backend,pid=p.pid,argv=command,status='running',
                            process_at_launch=helper.process_snapshot(p.pid),selected_gpu=selected,gpu_audit=gpus)
                        state['cases'].append(row);state['status']='running_'+case;write(record,state)
                        while p.poll() is None:bound();time.sleep(2.)
                    row['returncode']=p.returncode;assert p.returncode==0,('Nominal process failed; no retry',case,p.returncode)
                    meta=read(out/case/'warm_contact_evaluation_status.json');report=read(out/case/'report.json')
                    assert meta==report['warm_contact_evaluation'] and meta['reward_variant']==variant
                    assert meta['normalized_actor_sha256']==driver['proofs']['formal_'+variant]['actor_sha256']
                    assert meta['native_states']==2793 and meta['policy_decisions']==349 and meta['evaluation_mu']==.7
                    assert meta['validation_backend']==backend and not meta['smoke_only']
                    name='physics_trajectory.npz' if backend=='physx' else 'rollout.npz'
                    assert sha(out/case/name)==meta['recording_sha256']
                    row.update(status='completed_require_independent_analysis',metadata_sha256=sha(out/case/'warm_contact_evaluation_status.json'))
                    write(record,state)
            state['status']='four_native_nominal_episodes_completed_require_analysis_and_render';write(record,state)
        except BaseException as error:
            state.update(status='failed_no_retry_or_process_signal',error=repr(error));write(record,state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ['workspace','evaluation-snapshot','training-snapshot','idle-audit-source','python','legacy-physx','legacy-mujoco']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-driver-pid',type=int,required=True);run(p.parse_args())
