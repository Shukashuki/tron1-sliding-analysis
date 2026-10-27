#!/usr/bin/env python3
"""Wait on the approved queue, retrieve completed evidence and analyze/render locally.

This program never launches/retries physics, changes the uploaded source, trains
a policy or stops a remote process. A queue identity mismatch is an error.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/data/zengxiong/workspaces/tron1-hockey-stop-20261027-v1'
SSH=['ssh','-i','/home/roy/.ssh/id_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','zengxiong@14.103.233.29']
CASES=('reference_dq','nominal_command')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_completed_status(records):
    if len(records)!=2 or [r['id'] for r in records]!=list(CASES):raise ValueError('Both approved cases are required')
    for row in records:
        if row['status']!='completed' or row.get('returncode')!=0 or not row['gpu_before_launch']['idle']:
            raise ValueError('Both approved cases must complete on initially idle GPUs')


def remote_python(source):
    result=subprocess.run(SSH+['python3 -'],input=source,text=True,capture_output=True,timeout=45,check=True)
    return json.loads(result.stdout)


POLL=f'''
from pathlib import Path
import json,time
r=Path({REMOTE!r});launch=json.loads((r/'queue.launch.json').read_text());s=json.loads((r/'queue.status.json').read_text())
assert launch['pid']==2485141 and launch['identity']['start_ticks']==2052218368
p=Path('/proc')/str(launch['pid'])/'stat';live=False
if p.exists():
 f=p.read_text().rsplit(')',1)[1].split();live=int(f[19])==launch['identity']['start_ticks'] and f[0]!='Z'
jobs=json.loads((r/'conditioning_probes_v1/status.json').read_text()) if (r/'conditioning_probes_v1/status.json').exists() else []
print(json.dumps(dict(observed_unix=time.time(),queue=s,queue_identity_verified_live=live,jobs=jobs)))
'''


COLLECT=f'''
from pathlib import Path
import json,hashlib,tarfile
r=Path({REMOTE!r});sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r.parent/'hockey_conditioning_probe_bundle_20261027_v1.tar.gz')=='94555eff1ae646cb3edd1166a8ef378f1ef573d6570f0962f61e98fa1f8a487c'
rows=json.loads((r/'conditioning_probes_v1/status.json').read_text())
assert [x['id'] for x in rows]==['reference_dq','nominal_command']
files=[Path('queue.launch.json'),Path('queue.status.json'),Path('queue.log'),Path('dispatch.log'),Path('conditioning_probes_v1/status.json')]
manifest=json.loads((r/'source_snapshot_v1/manifest.json').read_text());files.append(Path('source_snapshot_v1/manifest.json'))
for row in manifest['files']:
 p=Path('source_snapshot_v1')/row['path'];assert sha(r/p)==row['sha256'];files.append(p)
for row in rows:
 assert row['status']=='completed' and row['returncode']==0 and row['gpu_before_launch']['idle']
 p=Path('conditioning_probes_v1')/row['id']/'report.json';report=json.loads((r/p).read_text())
 assert sha(r/p)==row['report_sha256'];assert report['steps']==139 and report['worlds']==1 and report['iterations']==0
 assert report['conditioning_mode']==row['id'] and report['actual_actor_input_archived']
 assert report['hockey_wrapper_sha256']=='51447598d9c840de94df203dfa21679c22f99a8d5d5b2876de7d9c671ab9d266'
 assert set(report['artifacts'])=={{'final_rollout.npz','native_contact_buffers.npz','mean.npz','reference_candidate.npz'}}
 assert not (r/p.parent/'failure.json').exists()
 files.extend([p,Path('conditioning_probes_v1')/(row['id']+'.log')])
 for name,digest in report['artifacts'].items():
  q=p.parent/name;assert sha(r/q)==digest;files.append(q)
out=r/'evidence_manifest.json';archive=r/'conditioning_evidence_v1.tar.gz'
assert not out.exists() and not archive.exists()
out.write_text(json.dumps(dict(files=[dict(path=str(p),bytes=(r/p).stat().st_size,sha256=sha(r/p)) for p in files]),indent=2)+'\\n')
with tarfile.open(archive,'w:gz') as tar:
 for p in files+[Path(out.name)]:
  assert not (r/p).is_symlink();tar.add(r/p,arcname=str(p),recursive=False)
print(json.dumps(dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    state=dict(status='waiting_for_approved_remote_jobs',collector_pid=os.getpid(),script_sha256=sha(__file__),physics_jobs_launched=0)
    def save():
        temp=a.output/'collector.status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'collector.status.json')
    save();previous=None;failures=0
    try:
        while True:
            try:snapshot=remote_python(POLL);failures=0
            except (subprocess.SubprocessError,json.JSONDecodeError) as e:
                failures+=1;state.update(last_poll_error=str(e),consecutive_poll_errors=failures);save()
                if failures>=3:raise RuntimeError('Polling failed; remote job state is unknown, no restart attempted') from e
                time.sleep(30);continue
            state['remote_snapshot']=snapshot;save();status=snapshot['queue']['status']
            if status!=previous:print(status,flush=True);previous=status
            if status=='dispatch_completed':validate_completed_status(snapshot['jobs']);break
            if status in ('expired_without_idle_GPU','dispatch_failed_no_retry'):
                state['status']=status;save();return
            if not snapshot['queue_identity_verified_live']:raise RuntimeError('Expected queue handle is absent; no replacement launched')
            time.sleep(30)
        state['status']='collecting_completed_evidence';save()
        archive=remote_python(COLLECT);local=a.output/'conditioning_evidence_v1.tar.gz'
        subprocess.run(['scp']+SSH[1:-1]+[SSH[-1]+':'+archive['path'],str(local)],check=True,timeout=120)
        if sha(local)!=archive['sha256']:raise ValueError('Evidence transfer hash mismatch')
        evidence=a.output/'evidence';evidence.mkdir()
        with tarfile.open(local) as tar:tar.extractall(evidence,filter='data')
        for row in json.loads((evidence/'evidence_manifest.json').read_text())['files']:
            q=evidence/row['path']
            if q.stat().st_size!=row['bytes'] or sha(q)!=row['sha256']:raise ValueError('Collected artifact changed')
        if sha(evidence/'source_snapshot_v1/manifest.json')!=sha(ROOT/'results/hockey_stop_conditioning_audit_20261027_v1/probe_payload_manifest.json'):
            raise ValueError('Collected source manifest differs from approved payload')
        state.update(status='analyzing_locally',returned_archive=archive);save()
        old=ROOT/'results/hockey_stop_physx_probe_20261026_v2'
        command=ROOT/'results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2'
        runs={'baseline':old/'evidence/probes_v2/baseline',**{k:evidence/'conditioning_probes_v1'/k for k in CASES}}
        table=evidence/'source_snapshot_v1/nominal_conditioning.npz'
        def run_local(argv,log):
            with log.open('xb') as stream:subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                env=dict(os.environ,MUJOCO_GL='osmesa',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'))
        summary={};all_xy=[np.load(command/'command_diagnostics.npz')['qpos'][:,:2]]
        for mode,run in runs.items():
            qdir=a.output/'qualification'/mode
            flags=['--execution-evidence',old/'evidence'] if mode=='baseline' else ['--conditioning-table',table]
            run_local(['scripts/evaluate_hockey_physx.py','--run-dir',run,'--command-dir',command,'--output',qdir,*flags],a.output/(mode+'_evaluation.log'))
            run_local(['scripts/analyze_hockey_posture_fidelity.py','--run-dir',run,'--qualification',qdir,'--command-dir',command,'--output',a.output/'posture'/mode],a.output/(mode+'_posture.log'))
            d=json.loads((qdir/'report.json').read_text())
            summary[mode]=dict(qualification=d['qualification'],first_45deg_tilt_s=d['first_45deg_tilt_diagnostic_s'],loaded_beta=d['loaded_beta_summary'],imitation=d['imitation'])
            with np.load(run/'final_rollout.npz',allow_pickle=False) as z:
                names=z['body_names'].astype(str).tolist();ids=[names.index('wheel_'+k+'_Link') for k in ('L','R')]
                all_xy.extend([z['root_pos'][:,:2],z['body_pos'][:,ids,:2].reshape(-1,2)])
        xy=np.concatenate(all_xy);center=(xy.min(0)+xy.max(0))/2;extent=max(float(np.ptp(xy,axis=0).max()*1.3),2.)
        for mode,run in runs.items():
            run_local(['scripts/render_hockey_physx_probe.py','--result',a.output,'--run-dir',run,'--qualification',a.output/'qualification'/mode,
                '--command-dir',command,'--model',a.model,'--output',a.output/'render'/mode,'--ffmpeg',a.ffmpeg,
                '--xy-center',*center,'--xy-extent',extent],a.output/(mode+'_render.log'))
        (a.output/'comparison.json').write_text(json.dumps(dict(cases=summary,baseline_is_historical=True,new_physics_jobs=2,
            physics_source='approved remote queue only; no physics launched by collector',common_XY_center=center.tolist(),common_XY_extent=extent,
            automatic_goal_completion_claimed=False),indent=2)+'\n')
        state['status']='collected_evaluated_rendered_requires_review';save();print(state['status'],flush=True)
    except BaseException as e:
        state.update(status='local_pipeline_failed_no_remote_restart',error_type=type(e).__name__,error=str(e));save();raise


if __name__=='__main__':main()
