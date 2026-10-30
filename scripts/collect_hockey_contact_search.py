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
import re
import subprocess

REMOTE_READER = r'''
import base64,hashlib,json,pathlib,re,time
root=pathlib.Path(REMOTE_ROOT)
bundle=root/SNAPSHOT_NAME;run=root/RUN_NAME
plan=json.loads((bundle/'plan.json').read_text())
paths=[RUN_NAME+'/status.json']
if ROBUSTNESS:
    state=json.loads((run/'status.json').read_text())
    assert state['source_manifest_sha256']==hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()
    cells=[cell for cell in plan['cells'] if cell['id']==state['candidate']]
    assert len(cells)==1
    protocol=(bundle/cells[0]['protocol']).resolve()
    assert protocol.is_relative_to(bundle.resolve())
    cfg=json.loads(protocol.read_text());cases=cfg['evaluation_cases']
    ids=[case['id'] for case in cases]
    assert len(ids)==33 and len(set(ids))==33 and ids[0]=='nominal'
    assert all(re.fullmatch(r'[a-zA-Z0-9_]+',name) for name in ids)
    paths.extend(RUN_NAME+'/'+name for name in ('nominal_requalification.json','eval_suite/report.json','qualification/report.json'))
    for name in ids:
        paths.append(RUN_NAME+'/qualification/'+name+'.json')
        directory=run/'eval_suite'/name
        # case.json is written at reset; only a completed report makes its
        # trajectory eligible for download. Never copy a partial NPZ.
        if not (directory/'report.json').is_file():continue
        recorded=json.loads((directory/'report.json').read_text())
        assert recorded['case_id']==name
        assert set(recorded['artifacts'])=={'case.json','final_rollout.npz','native_contact_buffers.npz'}
        paths.append(RUN_NAME+'/eval_suite/'+name+'/report.json')
        for file,digest in recorded['artifacts'].items():
            if file.endswith('.npz') and not INCLUDE_ROLLOUTS:continue
            assert hashlib.sha256((directory/file).read_bytes()).hexdigest()==digest
            paths.append(RUN_NAME+'/eval_suite/'+name+'/'+file)
else:
    paths.append(LAUNCH_NAME)
    for cell in plan['cells']:
        directory=pathlib.PurePosixPath(RUN_NAME)/cell['id']
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
status=json.loads(base64.b64decode(files[RUN_NAME+'/status.json']['data']))
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
    p.add_argument('--version',default='v2',help='Snapshot/run version within the same authorized remote study')
    p.add_argument('--robustness-run',help='Optional robustness_vN_vM directory for the same snapshot version; collects declared trials only')
    p.add_argument('--include-rollouts',action='store_true')
    p.add_argument('--timeout-s',type=float,default=60,help='Read-only transfer timeout; complete 33-case raw suites may need longer')
    a=p.parse_args()
    if not 5<=a.timeout_s<=600:
        raise ValueError('Transfer timeout must be between 5 and 600 seconds')
    if a.remote_root!='/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1':
        raise ValueError('Collector is scoped to the approved study')
    if not re.fullmatch(r'v[1-9][0-9]*',a.version):raise ValueError('Invalid study version')
    if a.robustness_run is not None and not re.fullmatch('robustness_'+a.version+r'_v[1-9][0-9]*',a.robustness_run):
        raise ValueError('Robustness directory must identify the selected snapshot version')
    run_name=a.robustness_run or 'run_'+a.version
    code=REMOTE_READER.replace('REMOTE_ROOT',repr(a.remote_root)).replace('INCLUDE_ROLLOUTS',repr(a.include_rollouts))
    code=code.replace('SNAPSHOT_NAME',repr('source_snapshot_'+a.version)).replace('RUN_NAME',repr(run_name))
    code=code.replace('LAUNCH_NAME',repr('launch_'+a.version+'.json'))
    code=code.replace('ROBUSTNESS',repr(a.robustness_run is not None))
    r=subprocess.run([str(a.helper),'exec','4090d','--','python3','-'],input=code,text=True,
                     capture_output=True,timeout=a.timeout_s,check=True)
    response=json.loads(r.stdout);files=response.pop('files')
    a.output.mkdir(parents=True,exist_ok=True)
    for relative,row in files.items():
        dest=(a.output/relative).resolve()
        if not dest.is_relative_to(a.output.resolve()):raise ValueError('Unsafe remote artifact path')
        data=base64.b64decode(row['data'],validate=True)
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Corrupt transfer')
        mutable=dest.name in ('status.json','run_status.json') or relative==run_name+'/eval_suite/report.json'
        if dest.exists() and not mutable and dest.read_bytes()!=data:
            raise ValueError('Completed artifact changed: '+relative)
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix(dest.suffix+'.download')
        temporary.write_bytes(data);temporary.replace(dest)
    response['file_sha256']={relative:row['sha256'] for relative,row in files.items()}
    (a.output/'latest_observation.json').write_text(json.dumps(response,indent=2)+'\n')
    s=json.loads((a.output/run_name/'status.json').read_text())
    print(json.dumps(dict(files_received=len(files),status=s['status'],
        stage=(s.get('current') or {}).get('stage'),current_process=response['current_process'],
        completed_stages=len(s['completed']),observed_unix_s=response['observed_unix_s'])))


if __name__=='__main__':main()
