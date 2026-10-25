"""Capture source-bound launch/smoke evidence without collecting actor weights."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time


def sha(data): return hashlib.sha256(data).hexdigest()


def process(pid):
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,uid=path.stat().st_uid,state=stat[0],start_ticks=int(stat[19]),
            argv=[v.decode() for v in (path/'cmdline').read_bytes().split(b'\0') if v])
    except FileNotFoundError:return None


def run(args):
    root=args.workspace.resolve()
    names=['driver_launch.json','driver_status.json','smoke_v1_launch.json',
        'smoke_v1_strict_verification.json','formal_v1_launch.json',
        'nominal_driver_launch.json','nominal_v1/status.json','source_snapshot_v1.manifest.json']
    for mode in ('smoke_v1','formal_v1'):
        names.extend(mode+'/'+name for name in ('manifest.json','runner_config.json',
            'cohort_startup_audit.json','ground_contact_v2q_startup_audit.json'))
    names.extend('smoke_v1/'+name for name in ('ground_contact_v2q_training_audit.json',
        'cohort_completion_audit.json','policy_contract.json','final_policy_contract.json','run_status.json'))
    payload={}
    for name in names:
        path=root/name
        assert path.is_file() and not path.is_symlink(),name
        payload[name]=path.read_bytes()
    observed=dict(kind='Q_launch_and_native_smoke_observation',observed_unix_time=time.time(),processes={})
    for key,name in (('driver','driver_launch.json'),('formal','formal_v1_launch.json'),('nominal','nominal_driver_launch.json')):
        launch=json.loads(payload[name]);actual=process(launch['pid'])
        if actual and actual['state']!='Z':
            assert actual['uid']==os.getuid() and actual['argv']==launch['argv'],key
            if 'process_at_launch' in launch:assert actual['start_ticks']==launch['process_at_launch']['start_ticks']
        observed['processes'][key]=actual
    log=(root/'formal_v1.log').read_text(errors='replace')
    iterations=re.findall(r'Learning iteration\s+(\d+)/(\d+)',re.sub(r'\x1b\[[0-9;]*m','',log))
    observed['last_logged_iteration']=list(map(int,iterations[-1])) if iterations else None
    observed['training_driver_status']=json.loads(payload['driver_status.json'])['status']
    observed['nominal_driver_status']=json.loads(payload['nominal_v1/status.json'])['status']
    observed['task_success_claimed']=False
    payload['process_observation.json']=(json.dumps(observed,indent=2)+'\n').encode()
    payload['formal_log_tail.txt']=('\n'.join(log.splitlines()[-50:])+'\n').encode()
    manifest=dict(kind=observed['kind'],models_weights_credentials_binaries_included=False,
        observer_source_sha256=sha(Path(__file__).read_bytes()),
        files={n:dict(sha256=sha(v),bytes=len(v)) for n,v in payload.items()})
    payload['observation_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    assert not args.output.exists()
    with tarfile.open(args.output,'x:gz') as tar:
        for name,data in payload.items():
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    print(json.dumps(dict(archive=str(args.output),bytes=args.output.stat().st_size,
        sha256=sha(args.output.read_bytes()),observation=observed)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
