"""One bounded1.5x serial MPPI rescue, then fresh full-telemetry validation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    assert not a.output.exists() and 'CUDA_VISIBLE_DEVICES' not in os.environ
    assert sha(a.audit_helper)=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
    spec=importlib.util.spec_from_file_location('_rescue_idle',a.audit_helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.GPU_ORDER=(3,5,4,0,1,2,6,7)
    prior=json.loads(a.probe_status.read_text());assert prior['status']=='completed_all_serial_single_world_probes'
    record=next(c for c in prior['cases'] if c['id']=='speed1p5')
    process=h.process_snapshot(record['pid']);assert process is None or process['state']=='Z' or process['start_ticks']!=record['identity']['start_ticks']
    initial=record['argv'];a.output.mkdir();status=dict(status='starting',cases=[],driver_sha256=sha(__file__),policy_training=False)
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    for name in ('smoke_v1','search_v1','validation_v1'):
        source_manifest=json.loads((a.source/'source_manifest.json').read_text())
        assert all(sha(a.source/r['path'])==r['sha256'] for r in source_manifest['files'])
        idle=next((g for g in h.query_gpus() if g['idle']),None)
        if idle is None:raise RuntimeError('No fully idle RTX; no process termination, sharing or automatic retry')
        argv=list(initial);argv[3]=str(a.source/'scripts/generate_physx_policy_mppi_reference.py')
        for flag,value in [('--config',a.source/'config/physx_speed_mppi_rescue_v1.json'),('--output-dir',a.output/name),('--device','cuda:'+str(idle['physical_index']))]:argv[argv.index(flag)+1]=str(value)
        index=argv.index('--validate-mean');del argv[index:index+2]
        if name=='validation_v1':argv+=['--validate-mean',str(a.output/'search_v1/mean.npz')]
        else:argv+=['--serial-candidates']
        if name=='smoke_v1':argv+=['--steps','16','--samples','2','--iterations','0']
        with (a.output/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID'),shell=False)
        identity=h.process_snapshot(child.pid);assert identity and identity['uid']==os.getuid() and identity['argv']==argv
        row=dict(id=name,pid=child.pid,argv=argv,identity=identity,gpu_idle_before_launch=idle,status='running');status['cases'].append(row);status['status']='running_serial';save()
        code=child.wait();row['returncode']=code
        assert code==0 and not (a.output/name/'failure.json').exists(),name+' failed; inspect preserved evidence; no retry'
        report=json.loads((a.output/name/'report.json').read_text());assert report['worlds']==1
        row.update(status='completed',report_sha256=sha(a.output/name/'report.json'));save()
        print(name,'completed',report['metrics'],flush=True)
    status.update(status='completed_search_and_fresh_single_world_validation',process_signals_sent=False,automatic_retry=False);save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('source','probe-status','audit-helper','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
