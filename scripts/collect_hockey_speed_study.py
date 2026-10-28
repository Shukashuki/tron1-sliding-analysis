"""Read status / collect completed study evidence, without modifying remote jobs."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

REMOTE = '/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1'
HELPER = '/home/roy/.codex/skills/remote-compute-ssh/scripts/remote_compute.sh'
PYTHON = '/data/zengxiong/.conda/envs/regrind/bin/python'


def remote(source):
    return subprocess.check_output([HELPER, 'exec', '4090d', '--', PYTHON, '-c', source])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status-only', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--representative-traces', action='store_true',
                   help='Also retrieve nominal and worst failed XY trial for completed analyzed cells')
    p.add_argument('--case', action='append', default=[],
                   help='Also retrieve this declared case from every completed analyzed cell; repeatable')
    a = p.parse_args()
    probe = r'''from pathlib import Path
import json,re
root=Path(REMOTE)
s=json.loads((root/'runs_v1/status.json').read_text())
c=s.get('current')
live=None;progress=None
if c and 'pid' in c:
 p=Path('/proc')/str(c['pid'])/'stat'
 if p.exists():
  raw=p.read_text();fields=raw[raw.rfind(')')+2:].split()
  live=fields[0]!='Z' and int(fields[19])==c['identity']['start_ticks']
 else:live=False
 cell,mode=c['id'].split('/')
 log=root/'runs_v1'/cell/(mode+'.log')
 if log.exists():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-500000));txt=f.read().decode(errors='replace')
  txt=re.sub(r'\x1b\[[0-9;]*m','',txt)
  matches=re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)',txt)
  if matches:progress={'iteration_index':int(matches[-1][0]),'total_iterations':int(matches[-1][1])}
brief=dict(status=s['status'],current=c['id'] if c else None,current_pid=c.get('pid') if c else None,
 original_process_live=live,progress=progress,idle_wait_s=s['idle_wait_s'],
 completed_stages=[r['id'] for r in s['completed']],error=s.get('error'),updated_unix_s=s['updated_unix_s'])
print(json.dumps(brief))
'''.replace('REMOTE', repr(REMOTE))
    status = json.loads(remote(probe))
    if a.status_only:
        print(json.dumps(status, indent=2)); return
    if a.output is None:
        p.error('--output required for collection')
    if a.output.exists():
        raise FileExistsError(a.output)
    collect = '''from pathlib import Path
import sys,tarfile,json,hashlib,io
root=Path(REMOTE)
s=json.loads((root/'runs_v1/status.json').read_text())
files=['launch.json','runs_v1/status.json']
for row in s['completed']:
 cell,mode=row['id'].split('/')
 directory=root/'runs_v1'/cell/mode
 assert hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()==row['report_sha256']
 files.append(str((directory/'report.json').relative_to(root)))
 for name in ('protocol.json','runner_config.json'):
  if (directory/name).exists():files.append(str((directory/name).relative_to(root)))
 if mode=='analyze':
  report=json.loads((directory/'report.json').read_text())
  for case in report['cases']:
   f=directory/(case['id']+'.json')
   assert hashlib.sha256(f.read_bytes()).hexdigest()==case['report_sha256']
   files.append(str(f.relative_to(root)))
  if INCLUDE_TRACES or EXTRA_CASES:
   failed=[c for c in report['cases'][1:] if not c['success']]
   chosen=(['nominal']+([max(failed,key=lambda c:c['xy_rmse_m'])['id']] if failed else [])) if INCLUDE_TRACES else []
   declared={c['id'] for c in report['cases']}
   if set(EXTRA_CASES)-declared:raise ValueError('Requested case is not in the audited suite')
   chosen=list(dict.fromkeys(chosen+EXTRA_CASES))
   for case_id in chosen:
    for name in ('report.json','case.json','final_rollout.npz','native_contact_buffers.npz'):
     files.append(str((root/'runs_v1'/cell/'eval-suite'/case_id/name).relative_to(root)))
    files.append(str((directory/(case_id+'.signals.npz')).relative_to(root)))
files=list(dict.fromkeys(files))
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for name in files:
  path=root/name
  raw=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(raw)
  tar.addfile(info,io.BytesIO(raw))
  manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 raw=(json.dumps(manifest,indent=2)+'\\n').encode();info=tarfile.TarInfo('collection_manifest.json');info.size=len(raw)
 tar.addfile(info,io.BytesIO(raw))
'''.replace('REMOTE', repr(REMOTE)).replace('INCLUDE_TRACES', repr(a.representative_traces)).replace('EXTRA_CASES', repr(a.case))
    raw = remote(collect)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as tar:
        members = tar.getmembers()
        for member in members:
            target = (a.output/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(a.output.resolve()) or target.suffix in ('.pt','.pth','.ckpt'):
                raise ValueError('Unexpected public-evidence entry')
        a.output.mkdir(parents=True)
        for member in members:
            target = a.output/member.name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    (a.output/'verified_process_status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), files=len(members), status=status['status'])))


if __name__ == '__main__':
    main()
