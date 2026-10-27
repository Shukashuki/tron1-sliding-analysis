#!/usr/bin/env python3
"""Smoke then one full stop probe; one freshly idle RTX, one world, no retries."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bundle','engine-scripts','prior-launch','audit-helper','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or 'CUDA_VISIBLE_DEVICES' in os.environ:raise ValueError('Unsafe output/device state')
    if sha(a.audit_helper)!='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866':raise ValueError('GPU audit helper changed')
    spec=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.GPU_ORDER=(3,5,4,0,1,2,6,7)
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    if not all(sha(a.bundle/r['path'])==r['sha256'] for r in manifest['files']):raise ValueError('Payload changed')
    old=json.loads(a.prior_launch.read_text());python=old['argv'][0]
    live=helper.process_snapshot(old['pid'])
    if live and live['state']!='Z' and live['start_ticks']==old['identity']['start_ticks']:raise RuntimeError('Prior owned job still live')
    a.output.mkdir();records=[]
    for name,steps in [('smoke',16),('baseline',0)]:
        idle=next((g for g in helper.query_gpus() if g['idle']),None)
        if idle is None:raise RuntimeError('No completely idle RTX GPU; nothing launched')
        argv=[python,'-B','-u',str(a.bundle/'hockey_physx_probe.py'),'--engine-scripts',str(a.engine_scripts),
            '--prior-launch',str(a.prior_launch),'--bundle',str(a.bundle),'--output-dir',str(a.output/name),
            '--device','cuda:'+str(idle['physical_index']),'--steps',str(steps)]
        with (a.output/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID'),shell=False)
        record=dict(id=name,pid=child.pid,argv=argv,identity=helper.process_snapshot(child.pid),gpu_before_launch=idle,status='running')
        records.append(record);(a.output/'status.json').write_text(json.dumps(records,indent=2)+'\n')
        code=child.wait();record['returncode']=code
        if code:raise RuntimeError(name+' failed; no retry')
        report=json.loads((a.output/name/'report.json').read_text())
        assert report['worlds']==1 and report['single_world_validation']
        record.update(status='completed',report_sha256=sha(a.output/name/'report.json'))
        (a.output/'status.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
