"""Collect completed K/ramp learning and two native nominal episodes only.

Live jobs, incomplete training and smoke-policy rollouts are rejected. Native
recordings and source/audits are allowed; assets, weights and credentials are
never packaged. Task success is determined later from actual native signals.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import time

COHORT='tron1_ppo_k_leg_ramp_400hz_v1'
CASES=('physx','mujoco')


def sha(data):return hashlib.sha256(data).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def layout(root):
    p=Path(root)/'collection_layout.json'
    return read(p) if p.exists() else dict(nominal_directory='nominal_v1',evaluation_snapshot='evaluation_snapshot_v1',
        continuation_launch='evaluation_continuation_v1_launch.json')


def verify_completed(root,*,nominal_name=None):
    root=Path(root);nominal_name=nominal_name or layout(root)['nominal_directory']
    assert nominal_name.replace('_','').isalnum()
    driver=read(root/'driver_status.json');status=read(root/nominal_name/'status.json')
    assert driver['status']=='formal_completed_verified_nominal_validation_pending'
    assert status['status']=='two_nominal_episodes_completed_require_analysis_and_render'
    proof=read(root/'formal_v1_strict_verification.json')
    assert proof==driver['formal_proof']==status['formal_training_proof']
    assert proof['cohort_id']==COHORT and proof['formal_or_smoke']=='formal'
    assert proof['transitions']==12288000 and proof['CPU_actor_output_shape']==[1,8]
    assert [r['id'] for r in status['cases']]==list(CASES)
    assert all(r['status']=='completed_strictly_verified' and r['returncode']==0 for r in driver['children'])
    verifier=root/'source_snapshot_v1/scripts/verify_ppo_k_leg_ramp.py'
    assert sha(verifier.read_bytes())==proof['verifier_sha256']
    folder=root/'formal_v1';audit=read(folder/'leg_ramp_training_audit.json')
    assert audit['status']=='completed_verified' and audit['cohort_id']==COHORT and not audit['smoke_only']
    assert audit['training_environment_steps']==12288000 and audit['actor_sha256']==proof['actor_sha256']
    assert audit['checkpoint_sha256']==proof['checkpoint_sha256']
    for key,name in [('manifest','manifest.json'),
            ('K_training_audit','ground_contact_v2k_training_audit.json'),('policy_contract','policy_contract.json'),
            ('startup_audit','leg_ramp_startup_audit.json'),('pre_export_runtime_audit','pre_export_runtime_audit.json'),
            ('export_precision_audit','export_precision_audit.json')]:
        assert audit[key+'_sha256']==sha((folder/name).read_bytes())
    for row in status['cases']:
        assert row['status']=='completed_requires_independent_analysis' and row['returncode']==0
        f=root/nominal_name/row['id'];m=read(f/'leg_ramp_evaluation_status.json');report=read(f/'report.json')
        assert sha((f/'leg_ramp_evaluation_status.json').read_bytes())==row['metadata_sha256']
        assert m==report['leg_ramp_evaluation'] and m['normalized_actor_sha256']==proof['actor_sha256']
        assert not m['smoke_policy_only'] and m['training_cohort_id']==COHORT
        assert m['validation_backend']==row['id'] and m['native_states']==2793 and m['policy_decisions']==349 and m['mu']==.7
        assert m['leg_command_execution']==audit['execution_audit']['settings']
        name='rollout.npz' if row['id']=='mujoco' else 'physics_trajectory.npz'
        assert sha((f/name).read_bytes())==m['recording_sha256']
        assert sha((f/'leg_command_trajectory.npz').read_bytes())==m['command_audit']['command_recording_sha256']
        assert m['command_audit']['native_target_applications']==2792
    return driver,status,proof


def collect(workspace,idle_audit_source,output_prefix='completed_evidence_v1',*,nominal_name='nominal_v1',
            evaluation_snapshot_name='evaluation_snapshot_v1',continuation_launch_name='evaluation_continuation_v1_launch.json'):
    root=Path(workspace).resolve();assert output_prefix.replace('_','').isalnum()
    assert nominal_name.replace('_','').isalnum() and evaluation_snapshot_name.replace('_','').isalnum()
    assert Path(continuation_launch_name).name==continuation_launch_name and continuation_launch_name.endswith('.json')
    driver,status,proof=verify_completed(root,nominal_name=nominal_name)
    # Read exact final neural files on the execution host; do not package them.
    actual_weights={name:sha((root/'formal_v1'/name).read_bytes()) for name in ['policy.pt','model_final.pt']}
    assert actual_weights=={'policy.pt':proof['actor_sha256'],'model_final.pt':proof['checkpoint_sha256']}
    helper=Path(idle_audit_source)
    assert sha(helper.read_bytes())=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
    spec=importlib.util.spec_from_file_location('_ramp_collection_process_audit',helper)
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    launches=['driver_v1_launch.json',continuation_launch_name]
    pids={read(root/n)['pid'] for n in launches}
    pids.update(r['pid'] for r in driver['children']);pids.update(r['pid'] for r in status['cases'])
    processes={str(p):observer.process_snapshot(p) for p in sorted(pids)}
    assert all(p is None or p['state']=='Z' for p in processes.values()),'Owned compute or continuation still live'
    names=launches+['driver_status.json','formal_v1_strict_verification.json','smoke_v1_strict_verification.json',
        'formal_CPU_verification.log',nominal_name+'/status.json']
    names.extend(str(p.relative_to(root)) for p in sorted((root/'formal_v1').glob('*.json')))
    for snapshot in ['source_snapshot_v1',evaluation_snapshot_name]:
        names.append(snapshot+'.manifest.json')
        for row in read(root/(snapshot+'.manifest.json'))['files']:
            p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts and p.suffix in ['.py','.json']
            assert sha((root/snapshot/p).read_bytes())==row['sha256'];names.append(snapshot+'/'+str(p))
    collector_name=str(Path(__file__).resolve().relative_to(root))
    if collector_name not in names:names.append(collector_name)
    for case in CASES:
        f=root/nominal_name/case
        names.extend(str(p.relative_to(root)) for p in sorted(f.rglob('*.json')))
        for name in ['rollout.npz' if case=='mujoco' else 'physics_trajectory.npz','leg_command_trajectory.npz']:
            names.append(str((f/name).relative_to(root)))
        names.append(nominal_name+'/'+case+'.log')
    if nominal_name!='nominal_v1':
        # Preserve the failed pre-rollout integration attempt, never relabel it
        # as a task trajectory or replace its source/status/log bytes.
        prior=read(root/'nominal_v1/status.json')
        assert prior['status']=='failed_no_retry_or_process_signal'
        names+=['nominal_v1/status.json','nominal_v1/physx.log','evaluation_continuation_v1.log',
            'evaluation_continuation_v1_launch.json','collection_waiter_status.json','collection_waiter_v1.log',
            'evaluation_snapshot_v1.manifest.json']
        names.extend('evaluation_snapshot_v1/'+r['path'] for r in read(root/'evaluation_snapshot_v1.manifest.json')['files'])
    assert len(names)==len(set(names))
    payload={}
    for name in names:
        p=root/name;assert p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink()
        assert p.suffix in ['.json','.py','.log','.npz'] and p.stat().st_size<=64*1024**2
        if p.suffix=='.npz':assert p.name in ['rollout.npz','physics_trajectory.npz','leg_command_trajectory.npz']
        payload[name]=p.read_bytes()
    assert sum(map(len,payload.values()))<=256*1024**2
    assert read(root/'driver_status.json')==driver and read(root/nominal_name/'status.json')==status
    payload['collection_layout.json']=(json.dumps(dict(nominal_directory=nominal_name,evaluation_snapshot=evaluation_snapshot_name,
        continuation_launch=continuation_launch_name),indent=2)+'\n').encode()
    payload['completion_process_audit.json']=(json.dumps(dict(processes=processes,observed_unix_time=time.time(),
        final_neural_files_read_and_hash_verified_on_execution_host_only=actual_weights,
        all_owned_compute_and_continuation_terminal=True),indent=2)+'\n').encode()
    archive=root/(output_prefix+'.tar.gz');manifest_path=root/(output_prefix+'.manifest.json')
    assert not archive.exists() and not manifest_path.exists()
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tf:
            for name,data in payload.items():
                item=tarfile.TarInfo(name);item.size=len(data);tf.addfile(item,io.BytesIO(data))
    result=dict(kind='completed_K_leg_ramp_training_two_native_episodes',archive=archive.name,
        archive_sha256=sha(archive.read_bytes()),archive_size_bytes=archive.stat().st_size,
        files=[dict(path=n,sha256=sha(b),bytes=len(b)) for n,b in payload.items()],
        actor_sha256=proof['actor_sha256'],collector_sha256=sha(Path(__file__).read_bytes()),
        models_weights_credentials_native_binaries_included=False,task_success_claimed=False)
    with manifest_path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--idle-audit-source',type=Path,required=True)
    p.add_argument('--output-prefix',default='completed_evidence_v1')
    p.add_argument('--nominal-name',default='nominal_v1');p.add_argument('--evaluation-snapshot-name',default='evaluation_snapshot_v1')
    p.add_argument('--continuation-launch-name',default='evaluation_continuation_v1_launch.json');a=p.parse_args()
    collect(a.workspace,a.idle_audit_source,a.output_prefix,nominal_name=a.nominal_name,
        evaluation_snapshot_name=a.evaluation_snapshot_name,continuation_launch_name=a.continuation_launch_name)
