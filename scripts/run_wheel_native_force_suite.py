"""Serial ten-case diagnostic, with a fresh process/resource gate per case.

Requires no GPU process. Never signals any existing process or waits
indefinitely for resources.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def gate(physical_index):
    def query(args):
        output=subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True)
        return [[x.strip() for x in row] for row in csv.reader(io.StringIO(output))]
    gpus=query(['--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu'])
    target=next(g for g in gpus if int(g[0])==physical_index)
    processes=query(['--query-compute-apps=gpu_uuid,pid'])
    pids={int(p[1]) for p in processes if p and p[0]==target[1]}
    if pids: raise RuntimeError('GPU has an existing CUDA process')
    if int(target[3])<16000 or int(target[4])>2:
        raise RuntimeError('GPU lacks required free memory or is active')
    ordinal=[g[1] for g in sorted(gpus,key=lambda g:g[2])].index(target[1])
    return dict(physical_index=physical_index, cuda_ordinal=ordinal, uuid=target[1],
                free_MiB=int(target[3]), utilization_percent=int(target[4]), pids=sorted(pids),
                observed_unix_time=time.time())


def settled_gate(index):
    # nvidia-smi utilization is averaged over a recent window, even after the
    # just-completed child has exited. Wait only for that transient, never run
    # a second simulation or admit a foreign process to work around a failure.
    deadline=time.monotonic()+60
    while True:
        try: return gate(index)
        except RuntimeError as error:
            if 'existing CUDA process' in str(error) or time.monotonic()>=deadline: raise
            time.sleep(2)


def run(a):
    if a.output_dir.exists(): raise FileExistsError('Fresh suite output required')
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Do not remap CUDA visibility for this existing Isaac environment')
    source=a.probe.resolve()
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    settled_gate(a.physical_gpu_index)  # Before creating outputs or launching anything.
    a.output_dir.mkdir(parents=True)
    status=dict(status='initializing', probe_sha256=digest,
                process_signals_sent=False, policy_training=False, cases=[])
    jobs=[('cold_smoke_400hz','cold',.0025,.04)]
    modes=('cold',) if a.controls_only else ('legacy','cold_replay','cold')
    jobs += [(f'{mode}_{hz}hz',mode,1/hz,2.) for hz in (200,400,800) for mode in modes]
    try:
        for name,mode,dt,duration in jobs:
            gpu=settled_gate(a.physical_gpu_index)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest: raise RuntimeError('Probe source changed')
            argv=[str(a.python),'-B','-u',str(source),'--initialization','cold' if mode=='cold_replay' else mode,'--dt',str(dt),
                  '--duration',str(duration),'--device',f"cuda:{gpu['cuda_ordinal']}",
                  '--output-dir',str(a.output_dir/name),'--floor-y-offset-m',str(a.floor_y_offset_m)]
            if mode=='cold_replay':
                argv+=['--initial-state-from',str(a.output_dir/f'legacy_{round(1/dt)}hz'/'status.json')]
            env=os.environ.copy(); env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            row=dict(name=name,status='running',gpu_gate=gpu,argv=argv)
            status['cases'].append(row); status['status']='running'
            write(a.output_dir/'status.json',status)
            with (a.output_dir/f'{name}.log').open('xb') as log:
                child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=child.pid; write(a.output_dir/'status.json',status)
                code=child.wait()
            row['returncode']=code
            if code: raise RuntimeError(f'{name} failed; no automatic retry')
            proof=json.loads((a.output_dir/name/'status.json').read_text())
            assert proof['status']=='completed_requires_analysis' and proof['source_sha256']==digest
            trace=a.output_dir/name/'native_trace.npz'
            assert hashlib.sha256(trace.read_bytes()).hexdigest()==proof['native_trace_sha256']
            row['status']='completed'; row['trace_sha256']=proof['native_trace_sha256']
            write(a.output_dir/'status.json',status)
        status['status']='completed_requires_analysis'
    except BaseException as error:
        status.update(status='stopped_no_automatic_retry',error=str(error))
        write(a.output_dir/'status.json',status)
        raise
    write(a.output_dir/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--physical-gpu-index',type=int,required=True)
    p.add_argument('--floor-y-offset-m',type=float,choices=(0.,-5.),default=0.)
    p.add_argument('--controls-only',action='store_true')
    run(p.parse_args())
