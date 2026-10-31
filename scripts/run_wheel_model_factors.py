"""Run the controlled 2 geometry x 3 law x 3 timestep study serially."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    if a.output_dir.exists() or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Fresh output and unmasked CUDA ordering required')
    a.output_dir.mkdir(parents=True)
    status = dict(status='waiting_for_predecessor', cases=[], process_signals_sent=False,
                  automatic_retry=False, policy_training=False, pid=os.getpid())
    sources = {str(p): sha(p) for p in (Path(__file__), a.probe, a.probe.with_name('nonlinear_tire_contact.py'),
               a.continuation_helper, a.gpu_probe)}
    status['source_sha256'] = sources
    def save():
        (a.output_dir/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def load(path, name):
        previous=list(sys.path)
        try:
            sys.path.insert(0,str(path.parent))
            spec=importlib.util.spec_from_file_location(name,path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            return mod
        finally:sys.path[:]=previous
    save()
    try:
        if a.after_status:
            deadline=time.monotonic()+86400
            while True:
                prior=json.loads(a.after_status.read_text())
                state=prior.get('status','')
                if state=='completed_with_analysis':
                    status['predecessor_status_sha256']=sha(a.after_status)
                    break
                if 'stopped' in state or 'failed' in state or time.monotonic()>deadline:
                    raise RuntimeError('Predecessor did not complete; no overlap or silent skip')
                time.sleep(15)
        continuation=load(a.continuation_helper,'factor_continuation')
        query=load(a.gpu_probe,'factor_gpu_probe').query_gpu
        laws=('native_coulomb','external_coulomb_regularized','combined_slip_magic_formula')
        jobs=[(f'smoke_s{s}_{law}',s,law,.0025,.04) for s in (1,9) for law in laws]
        jobs += [(f's{s}_{law}_{hz}hz',s,law,1/hz,2.) for hz in (200,400,800) for s in (1,9) for law in laws]
        identity=None
        for name,slices,law,dt,duration in jobs:
            if any(sha(p)!=h for p,h in sources.items()):
                raise RuntimeError('Frozen experiment source changed')
            def waiting(gpu):
                status.update(status='waiting_for_idle_gpu',latest_gpu_observation=gpu);save()
            gpu=continuation.wait_idle(query,a.physical_gpu_index,time.monotonic()+86400,waiting)
            if identity is not None and gpu['uuid']!=identity:
                raise RuntimeError('GPU identity changed')
            identity=gpu['uuid']
            folder=a.output_dir/name
            command=[str(a.python),'-u',str(a.probe),'--slices',str(slices),'--law',law,
                     '--device',f'cuda:{a.physical_gpu_index}','--dt',str(dt),'--duration',str(duration),
                     '--output-dir',str(folder)]
            row=dict(name=name,status='running',gpu_gate=gpu,command=command)
            status['cases'].append(row);status['status']='running_'+name;save()
            env=os.environ.copy();env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            with (a.output_dir/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=child.pid;save();code=child.wait()
            row['returncode']=code
            if code:raise RuntimeError(name+' failed; no automatic retry')
            proof=json.loads((folder/'status.json').read_text())
            if proof.get('status')!='completed_requires_analysis' or proof['native_trace_sha256']!=sha(folder/'native_trace.npz'):
                raise RuntimeError('Incomplete or modified evidence')
            if max(proof['momentum_residual_rms_by_axis_n'])>.002:
                raise RuntimeError('Native force/momentum consistency gate failed')
            row.update(status='completed_verified',status_sha256=sha(folder/'status.json'),trace_sha256=proof['native_trace_sha256'])
            save();time.sleep(2)
        status['status']='completed_requires_analysis'
    except BaseException as error:
        status.update(status='stopped_without_retry',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['finished_unix_time']=time.time();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('output-dir','probe','python','continuation-helper','gpu-probe'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--after-status',type=Path)
    p.add_argument('--physical-gpu-index',type=int,choices=(3,4,5),required=True)
    run(p.parse_args())
