"""Collect completed gain transfers and preserved setup failures, without weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def terminal(pid):
    p=Path('/proc')/str(pid)
    try:return (p/'stat').read_text().rsplit(') ',1)[1].split()[0]=='Z'
    except FileNotFoundError:return True


def run(args):
    root=args.workspace.resolve();run_id='episodes_v3'
    state=read(root/run_id/'status.json')
    assert state['status']=='both_explicit_PD_transfers_completed_require_analysis'
    assert [c['id'] for c in state['cases']]==['baseline_500_10','soft_40_2p5']
    assert all(c['status']=='completed_verified' and c['returncode']==0 for c in state['cases'])
    launch=read(root/'driver_v3_launch.json')
    pids=[launch['pid'],*[c['pid'] for c in state['cases']],587951,588047,594180,594183]
    assert all(terminal(pid) for pid in pids),'Wait for exact owned processes to end'
    names=['driver_v3_launch.json','driver_v3.log',run_id+'/status.json']
    for version in (1,2,3):
        names.extend([f'source_snapshot_v{version}.manifest.json',f'source_snapshot_v{version}.tar.gz'])
    names.extend(['driver_launch.json','driver.log','episodes_v1/status.json','episodes_v1/baseline_500_10.log',
        'driver_v2_launch.json','driver_v2.log','episodes_v2/status.json','episodes_v2/baseline_500_10.log'])
    for case in state['cases']:
        prefix=run_id+'/'+case['id']+'/'
        proof=read(root/prefix/'leg_pd_status.json')
        assert proof['kind']=='frozen_K_PhysX_explicit_leg_PD_transfer'
        for n in ('physics_trajectory','tire_trajectory','leg_pd_trajectory'):
            assert sha(root/prefix/(n+'.npz'))==proof[n+'_sha256']
        names.extend(prefix+n for n in ('report.json','policy_contract.json','wheel_contact_model_status.json',
            'leg_pd_status.json','physics_trajectory.npz','tire_trajectory.npz','leg_pd_trajectory.npz'))
        names.extend([run_id+'/'+case['id']+'.log',run_id+'/'+case['id']+'_inputs/contract.json',run_id+'/'+case['id']+'_inputs/task_profile.json'])
    latest=read(root/'source_snapshot_v3.manifest.json')
    assert sha(root/'source_snapshot_v3.tar.gz')==latest['archive_sha256']
    for item in latest['files']:
        n='source_snapshot_v3/'+item['path'];assert sha(root/n)==item['sha256'];names.append(n)
    assert len(set(names))==len(names)
    for n in names:
        p=root/n
        assert p.is_file() and not p.is_symlink() and p.suffix in ('.json','.log','.py','.npz','.gz')
    manifest=dict(kind='completed_K_leg_PD_probe_native_evidence',run_id=run_id,
        files={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in names},
        collector_source_sha256=sha(__file__),all_owned_jobs_terminal=True,terminal_owned_pids=pids,
        observed_unix_time=time.time(),models_weights_credentials_native_binaries_included=False)
    target=root/'completed_collection_v3.manifest.json';archive=root/'completed_collection_v3.tar.gz'
    assert not target.exists() and not archive.exists()
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(archive,'x:gz') as tar:
        for n in names:tar.add(root/n,arcname=n,recursive=False)
        tar.add(target,arcname='collection_manifest.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(names))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--workspace',type=Path,required=True);run(p.parse_args())
