"""Serial single-world PhysX speed probes, freshly idle GPU before each job."""
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
    s=importlib.util.spec_from_file_location('_idle_gpu',a.audit_helper);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
    h.GPU_ORDER=(3,5,4,0,1,2,6,7)
    manifest=json.loads((a.source/'source_manifest.json').read_text())
    assert all(sha(a.source/r['path'])==r['sha256'] for r in manifest['files'])
    refs=a.source/'references';cases=json.loads((refs/'manifest.json').read_text())['cases']
    previous=json.loads(a.prior_launch.read_text());argv0=previous['argv']
    # Do not overlap an owned prior run, and never signal a foreign process.
    p=h.process_snapshot(previous['pid'])
    assert p is None or p['state']=='Z' or p['start_ticks']!=previous['identity']['start_ticks']
    a.output.mkdir();records=[]
    jobs=[('smoke_'+cases[0]['id'],cases[0],True)]+[(c['id'],c,False) for c in cases]
    for name,case,smoke in jobs:
        current=json.loads((a.source/'source_manifest.json').read_text())
        assert all(sha(a.source/r['path'])==r['sha256'] for r in current['files'])
        folder=refs/case['id']
        assert sha(folder/'config.json')==case['config_sha256']
        idle=next((g for g in h.query_gpus() if g['idle']),None)
        if idle is None:raise RuntimeError('No completely idle RTX GPU; no sharing, termination or automatic retry')
        argv=list(argv0);argv[3]=str(a.source/'scripts/generate_physx_policy_mppi_reference.py')
        for flag,value in [('--config',folder/'config.json'),('--motion',folder/'motion.npz'),
                ('--original-motion',folder/'car_reference.npz'),('--output-dir',a.output/name),
                ('--device','cuda:'+str(idle['physical_index'])),('--validate-mean',refs/'zero_mean.npz')]:
            argv[argv.index(flag)+1]=str(value)
        if smoke:argv+=['--steps','16']
        with (a.output/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID'),shell=False)
        identity=h.process_snapshot(child.pid);assert identity and identity['uid']==os.getuid() and identity['argv']==argv
        record=dict(id=name,pid=child.pid,argv=argv,identity=identity,gpu_idle_before_launch=idle,status='running')
        records.append(record);(a.output/'status.json').write_text(json.dumps(dict(status='running_serial',cases=records),indent=2)+'\n')
        code=child.wait();record['returncode']=code
        assert code==0 and not (a.output/name/'failure.json').exists(),name+' failed; evidence preserved; no retry'
        report=json.loads((a.output/name/'report.json').read_text())
        assert report['single_world_validation'] and report['worlds']==1 and report['actor_sha256']=='e5b94e87ed070f28c391cc7b0a11b216ce6b6d887141a28e7c83157085a9d8d0'
        assert report['steps']==(16 if smoke else case['frames'])
        record.update(status='completed',report_sha256=sha(a.output/name/'report.json'))
        print(name,'completed',report['metrics'],flush=True)
    (a.output/'status.json').write_text(json.dumps(dict(status='completed_all_serial_single_world_probes',cases=records,
        driver_sha256=sha(__file__),process_signals_sent=False,automatic_retry=False,policy_training=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('source','prior-launch','audit-helper','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
