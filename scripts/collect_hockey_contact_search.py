"""Read approved hockey screens without copying weights or changing remote jobs.

Completed artifacts are immutable locally; live status is explicitly an
observation, accompanied by a /proc identity check rather than trusted alone.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE_READER = r'''
import base64,hashlib,json,pathlib,time
root=pathlib.Path(REMOTE_ROOT)
bundle=root/'source_snapshot_v2';run=root/'run_v2'
plan=json.loads((bundle/'plan.json').read_text())
paths=['launch_v2.json','run_v2/status.json']
for cell in plan['cells']:
    directory=pathlib.PurePosixPath('run_v2')/cell['id']
    assert len(pathlib.PurePosixPath(cell['id']).parts)==1
    for stage in ('smoke','formal'):
        paths.append(str(directory/stage/'report.json'))
        paths.append(str(directory/stage/'run_status.json'))
    for stage in ('parent_transfer','nominal'):
        paths.extend((str(directory/(stage+'_qualification.json')),str(directory/stage/'report.json')))
        if INCLUDE_ROLLOUTS:
            for name in ('final_rollout.npz','native_contact_buffers.npz'):
                paths.append(str(directory/stage/name))
files={}
for relative in paths:
    p=(root/relative).resolve()
    assert p.is_relative_to(root.resolve()) and p.suffix in ('.json','.npz')
    if not p.exists():continue
    if p.suffix=='.npz' and not (p.parent/'report.json').exists():continue
    payload=p.read_bytes()
    assert len(payload)<=64*1024*1024
    files[relative]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),data=base64.b64encode(payload).decode())
status=json.loads(base64.b64decode(files['run_v2/status.json']['data']))
current=status.get('current') or {};pid=current.get('pid');probe=dict(pid=pid,live=False,identity_matches=False)
if pid is not None:
    p=pathlib.Path('/proc')/str(pid)
    if p.exists():
        fields=(p/'stat').read_text().split();argv=(p/'cmdline').read_bytes().split(b'\0')[:-1]
        probe.update(live=fields[2]!='Z',state=fields[2],start_ticks=fields[21],
            identity_matches=str(current.get('identity',{}).get('start_ticks'))==fields[21]
                and [s.decode() for s in argv]==current.get('argv'))
log_tail=None
if current.get('stage'):
    p=(run/(current['stage']+'.log')).resolve()
    assert p.is_relative_to(run.resolve())
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-18000));log_tail=f.read().decode(errors='replace')
print(json.dumps(dict(observed_unix_s=time.time(),current_process=probe,log_tail=log_tail,files=files)))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote-root',default='/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1')
    p.add_argument('--helper',type=Path,default=Path('/home/roy/.codex/skills/remote-compute-ssh/scripts/remote_compute.sh'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-rollouts',action='store_true')
    a=p.parse_args()
    if a.remote_root!='/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1':
        raise ValueError('Collector is scoped to the approved study')
    code=REMOTE_READER.replace('REMOTE_ROOT',repr(a.remote_root)).replace('INCLUDE_ROLLOUTS',repr(a.include_rollouts))
    r=subprocess.run([str(a.helper),'exec','4090d','--','python3','-'],input=code,text=True,
                     capture_output=True,timeout=60,check=True)
    response=json.loads(r.stdout);files=response.pop('files')
    a.output.mkdir(parents=True,exist_ok=True)
    for relative,row in files.items():
        dest=(a.output/relative).resolve()
        if not dest.is_relative_to(a.output.resolve()):raise ValueError('Unsafe remote artifact path')
        data=base64.b64decode(row['data'],validate=True)
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Corrupt transfer')
        mutable=dest.name in ('status.json','run_status.json')
        if dest.exists() and not mutable and dest.read_bytes()!=data:
            raise ValueError('Completed artifact changed: '+relative)
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix(dest.suffix+'.download')
        temporary.write_bytes(data);temporary.replace(dest)
    response['file_sha256']={relative:row['sha256'] for relative,row in files.items()}
    (a.output/'latest_observation.json').write_text(json.dumps(response,indent=2)+'\n')
    s=json.loads((a.output/'run_v2/status.json').read_text())
    print(json.dumps(dict(files_received=len(files),status=s['status'],
        stage=(s.get('current') or {}).get('stage'),current_process=response['current_process'],
        completed_stages=len(s['completed']),observed_unix_s=response['observed_unix_s'])))


if __name__=='__main__':main()
