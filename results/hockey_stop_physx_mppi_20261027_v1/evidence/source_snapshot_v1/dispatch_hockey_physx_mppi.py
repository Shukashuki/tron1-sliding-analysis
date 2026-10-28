"""Bounded serial native-PhysX hockey search, with fresh idle-GPU checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','prior-launch','audit-helper','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or 'CUDA_VISIBLE_DEVICES' in os.environ:raise ValueError('Unsafe output/device state')
    if sha(a.audit_helper)!='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866':raise ValueError('GPU audit helper changed')
    spec=importlib.util.spec_from_file_location('gpu_audit',a.audit_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.GPU_ORDER=(3,5,4,0,1,2,6,7)
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    if not all(sha(a.bundle/r['path'])==r['sha256'] for r in manifest['files']):raise ValueError('Snapshot changed')
    cfg=json.loads((a.bundle/'config.json').read_text())
    if cfg['samples']!=63 or cfg['iterations']!=6 or cfg['knots']!=12:raise ValueError('Unreviewed search scope')
    old=json.loads(a.prior_launch.read_text())['argv']
    get=lambda name:old[old.index('--'+name)+1]
    a.output.mkdir();records=[]
    for mode in ('smoke','search','validation'):
        idle=next((g for g in helper.query_gpus() if g['idle']),None)
        if idle is None:raise RuntimeError('No fully idle RTX; next stage not started')
        argv=[old[0],'-B','-u',str(a.bundle/'generate_hockey_physx_mppi.py'),
              '--config',str(a.bundle/'config.json'),'--motion',str(a.bundle/'motion.npz'),
              '--original-motion',str(a.bundle/'motion.npz'),
              '--conditioning-table',str(a.bundle/'nominal_conditioning.npz'),
              '--initial-mean',str(a.bundle/'initial_mean.npz'),'--output-dir',str(a.output/mode),
              '--device','cuda:'+str(idle['physical_index']),'--steps','16' if mode=='smoke' else '139']
        for name in ('actor','asset','seed-report','seed-trace','tracking-root','beyondmimic-root'):
            argv+=['--'+name,get(name)]
        if mode!='search':
            plan=a.bundle/'initial_mean.npz' if mode=='smoke' else a.output/'search/mean.npz'
            argv+=['--iterations','0','--validate-mean',str(plan)]
        with (a.output/(mode+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID'),shell=False)
        row=dict(id=mode,pid=child.pid,identity=helper.process_snapshot(child.pid),argv=argv,
                 gpu_before_launch=idle,status='running')
        records.append(row);(a.output/'status.json').write_text(json.dumps(records,indent=2)+'\n')
        code=child.wait();row['returncode']=code
        if code:
            row['status']='failed';(a.output/'status.json').write_text(json.dumps(records,indent=2)+'\n')
            raise RuntimeError(mode+' failed; no retry or next job')
        report=json.loads((a.output/mode/'report.json').read_text())
        if report['worlds']!=(64 if mode=='search' else 1) or report['steps']!=(16 if mode=='smoke' else 139):raise ValueError('Unexpected completed scope')
        row.update(status='completed',report_sha256=sha(a.output/mode/'report.json'))
        (a.output/'status.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
