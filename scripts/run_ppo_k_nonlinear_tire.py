"""Serial native smoke, strict CPU verification, then matched-budget training.

Uses only freshly fully idle GPUs3/5/4. No retries, process signals, installs,
compilation or altered training budget. A completed training is not task success.
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


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def run(a):
    root,snapshot=a.workspace.resolve(),a.snapshot.resolve()
    status_path=root/'driver_status.json'
    assert not status_path.exists()
    status=dict(kind='K_nonlinear_tire_smoke_then_formal',status='preflight',children=[],task_success_claimed=False)
    write(status_path,status)
    with (root/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert sha(a.idle_audit_source)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
            spec=importlib.util.spec_from_file_location('_nonlinear_full_idle_GPU_audit',a.idle_audit_source)
            audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
            for pid in (528887,528956,529379,530031,547996,550510,587951,588047,594180,594183,596504,596511,597440):
                process=audit.process_snapshot(pid)
                assert process is None or process['state']=='Z',('Prior owned process still active',process)
            manifest=json.loads((root/(snapshot.name+'.manifest.json')).read_text())
            ledger={str(snapshot/r['path']):r['sha256'] for r in manifest['files']}
            def sources(): assert all(sha(p)==digest for p,digest in ledger.items())
            sources()
            template=json.loads((snapshot/'config/ppo_k_nonlinear_tire_launch_template.json').read_text())['argv']
            environment=os.environ.copy();environment.pop('CUDA_VISIBLE_DEVICES',None)
            environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
            def child(command,case,env,extra=None):
                with (root/(case+'.log')).open('xb') as log:
                    p=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
                    row=dict(case=case,pid=p.pid,argv=command,status='running',started_unix_time=time.time(),
                             process_at_launch=audit.process_snapshot(p.pid),**(extra or {}))
                    status['children'].append(row);status['status']='running_'+case;write(status_path,status)
                    while p.poll() is None: time.sleep(2.)
                row['returncode']=p.returncode
                assert p.returncode==0,('Owned job failed; no retry',case,p.pid,p.returncode)
                row['status']='process_terminal_require_artifact_verification';write(status_path,status)
                return row
            for mode,n,iterations in [('smoke',32,2),('formal',512,1000)]:
                if mode=='formal' and a.smoke_only: break
                status['status']='waiting_for_fully_idle_GPU_'+mode;write(status_path,status)
                while True:
                    gpus=audit.query_gpus();selected=next((g for g in gpus if g['idle']),None)
                    if selected: break
                    time.sleep(15.)
                sources()
                folder=root/(mode+'_v1')
                command=template[:];command[0]=str(a.python);command[3]=str(snapshot/'scripts/train_ppo_k_nonlinear_tire.py')
                replacements={'--cohort-protocol':str(snapshot/'config/ppo_k_nonlinear_tire_400hz_v1.json'),
                    '--cohort-run-kind':mode,'--num-envs':str(n),'--iterations':str(iterations),
                    '--output-dir':str(folder),'--declared-input-dir':str(root/(mode+'_v1_declared_inputs')),
                    '--device':'cuda:'+str(selected['physical_index'])}
                for flag,value in replacements.items():
                    assert command.count(flag)==1;command[command.index(flag)+1]=value
                assert not folder.exists()
                row=child(command,mode,environment,dict(selected_gpu=selected,gpu_audit=gpus))
                verify=[str(a.python),'-B',str(snapshot/'scripts/verify_ppo_k_nonlinear_tire.py'),
                    '--training-dir',str(folder),'--protocol',replacements['--cohort-protocol'],
                    '--k-snapshot',command[command.index('--k-snapshot')+1],
                    '--cpu-inference','--output',str(root/(mode+'_v1_strict_verification.json'))]
                if mode=='smoke':verify.append('--allow-smoke')
                cpu=environment.copy();cpu['CUDA_VISIBLE_DEVICES']=''
                verification=child(verify,mode+'_CPU_verification',cpu)
                proof=json.loads((root/(mode+'_v1_strict_verification.json')).read_text())
                assert proof['formal_or_smoke']==mode and proof['transitions']==n*24*iterations
                assert proof['CPU_actor_output_shape']==[1,8]
                row['status']='completed_strictly_verified';verification['status']='completed_strictly_verified'
                status[mode+'_proof']=proof;write(status_path,status)
            status['status']='smoke_completed_verified' if a.smoke_only else 'formal_completed_verified_nominal_validation_pending'
            sources();write(status_path,status)
        except BaseException as error:
            status.update(status='failed_no_retry_or_process_signal',error=repr(error));write(status_path,status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('workspace','snapshot','idle-audit-source','python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true');run(p.parse_args())
