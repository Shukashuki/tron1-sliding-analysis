"""Collect completed paired warm refinement and four native nominal records.

Refuses incomplete or mixed-policy results and live owned compute. Rehash final
neural files on the execution host but omit their bytes and all physical assets.
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

COHORT='tron1_contact_warm_refinement_400hz_v1'
INITIAL='1c4894073146a9bd1abd131c60de7efdba493fc33d75599c679792435a110db8'
SOURCE_CHECKPOINT='8cb011e4c5e2a33a685bc58280df410141ab99aa245892042762b8956595fe5a'
CASES=('K_physx','K_mujoco','M_physx','M_mujoco')


def sha(b):return hashlib.sha256(b).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def verify_completed(root):
    root=Path(root);driver=read(root/'driver_status.json');status=read(root/'nominal_v1/status.json')
    assert driver['status']=='both_formal_arms_completed_verified_nominal_pending'
    assert status['status']=='four_native_nominal_episodes_completed_require_analysis_and_render'
    assert all(r['status']=='completed_strictly_verified' and r['returncode']==0 for r in driver['children'])
    assert [r['id'] for r in status['cases']]==list(CASES)
    for variant in ['K','M']:
        folder=root/('formal_'+variant);proof=read(root/('formal_'+variant+'_verification.json'))
        assert proof==driver['proofs']['formal_'+variant]==status['paired_formal_proofs'][variant]
        assert proof['cohort_id']==COHORT and proof['variant']==variant and proof['run_kind']=='formal'
        assert proof['transitions']==2457600 and proof['cumulative_transitions']==14745600
        assert proof['initial_policy_state_sha256']==INITIAL and proof['CPU_actor_output_shape']==[1,8]
        assert not proof['from_scratch'] and proof['historical_checkpoint_used']
        assert sha((folder/'cohort_completion_audit.json').read_bytes())==proof['completion_audit_sha256']
        audit=read(folder/'cohort_completion_audit.json')
        assert audit['initial_policy_state_sha256']==INITIAL and audit['warm_start']['checkpoint_sha256']==SOURCE_CHECKPOINT
        assert audit['training_environment_steps']==2457600 and audit['cumulative_training_environment_steps']==14745600
        assert audit['actor_sha256']==proof['actor_sha256'] and audit['checkpoint_sha256']==proof['checkpoint_sha256']
        for key,name in [('manifest','manifest.json'),('policy_contract','policy_contract.json'),
                ('pre_export_runtime_audit','pre_export_runtime_audit.json'),('export_precision_audit','export_precision_audit.json'),
                ('startup_audit','cohort_startup_audit.json')]:assert audit[key+'_sha256']==sha((folder/name).read_bytes())
    for row in status['cases']:
        assert row['status']=='completed_require_independent_analysis' and row['returncode']==0
        folder=root/'nominal_v1'/row['id'];m=read(folder/'warm_contact_evaluation_status.json');report=read(folder/'report.json')
        assert m==report['warm_contact_evaluation'] and sha((folder/'warm_contact_evaluation_status.json').read_bytes())==row['metadata_sha256']
        proof=driver['proofs']['formal_'+row['variant']]
        assert m['reward_variant']==row['variant'] and m['validation_backend']==row['backend']
        assert m['normalized_actor_sha256']==proof['actor_sha256']
        assert m['training_completion_audit_sha256']==proof['completion_audit_sha256']
        assert m['additional_refinement_transitions']==2457600 and m['cumulative_training_transitions']==14745600
        assert m['native_states']==2793 and m['policy_decisions']==349 and m['evaluation_mu']==.7
        assert not m['smoke_only'] and not m['from_scratch']
        name='physics_trajectory.npz' if row['backend']=='physx' else 'rollout.npz'
        assert sha((folder/name).read_bytes())==m['recording_sha256']
    return driver,status


def collect(root,helper_path):
    root=Path(root).resolve();driver,status=verify_completed(root)
    assert sha(Path(helper_path).read_bytes())=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
    spec=importlib.util.spec_from_file_location('_warm_collection_process',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    launch_names=['driver_v1_launch.json','nominal_continuation_v1_launch.json']
    pids={read(root/n)['pid'] for n in launch_names}
    pids.update(r['pid'] for r in driver['children']);pids.update(r['pid'] for r in status['cases'])
    processes={str(pid):helper.process_snapshot(pid) for pid in sorted(pids)}
    assert all(p is None or p['state']=='Z' for p in processes.values()),'Owned training or evaluation still live'
    weights={}
    names=launch_names+['driver_status.json','nominal_v1/status.json']
    for variant in ['K','M']:
        proof=driver['proofs']['formal_'+variant];folder=root/('formal_'+variant)
        actual={n:sha((folder/n).read_bytes()) for n in ['model_final.pt','policy.pt']}
        assert actual=={'model_final.pt':proof['checkpoint_sha256'],'policy.pt':proof['actor_sha256']}
        weights[variant]=actual
        names += ['formal_'+variant+'_verification.json','smoke_'+variant+'_verification.json','formal_'+variant+'_CPU_verification.log']
        names += [str(p.relative_to(root)) for p in sorted(folder.glob('*.json'))]
    for snapshot in ['source_snapshot_v1','evaluation_snapshot_v1','postprocessing_snapshot_v1']:
        names.append(snapshot+'.manifest.json')
        for row in read(root/(snapshot+'.manifest.json'))['files']:
            p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts and p.suffix in ['.py','.json']
            assert sha((root/snapshot/p).read_bytes())==row['sha256'];names.append(snapshot+'/'+str(p))
    for case in CASES:
        folder=root/'nominal_v1'/case
        names += [str(p.relative_to(root)) for p in sorted(folder.rglob('*.json'))]
        names += ['nominal_v1/'+case+'.log',str((folder/('physics_trajectory.npz' if case.endswith('physx') else 'rollout.npz')).relative_to(root))]
    assert len(names)==len(set(names))
    payload={}
    for name in names:
        p=root/name;assert p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink()
        assert p.suffix in ['.py','.json','.log','.npz'] and p.stat().st_size<=64*1024**2
        if p.suffix=='.npz':assert p.name in ['physics_trajectory.npz','rollout.npz']
        payload[name]=p.read_bytes()
    assert sum(map(len,payload.values()))<256*1024**2
    assert read(root/'driver_status.json')==driver and read(root/'nominal_v1/status.json')==status
    payload['completion_process_audit.json']=(json.dumps(dict(processes=processes,observed_unix_time=time.time(),
        all_owned_training_and_evaluation_terminal=True,final_neural_hashes_checked_on_execution_host_only=weights),indent=2)+'\n').encode()
    archive=root/'completed_evidence_v1.tar.gz';manifest=root/'completed_evidence_v1.manifest.json'
    assert not archive.exists() and not manifest.exists()
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tf:
            for name,b in payload.items():
                item=tarfile.TarInfo(name);item.size=len(b);tf.addfile(item,io.BytesIO(b))
    result=dict(kind='completed_paired_warm_contact_refinement_four_native_episodes',archive_sha256=sha(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,files=[dict(path=n,sha256=sha(b),bytes=len(b)) for n,b in payload.items()],
        actors={v:driver['proofs']['formal_'+v]['actor_sha256'] for v in ['K','M']},collector_sha256=sha(Path(__file__).read_bytes()),
        models_weights_credentials_native_binaries_included=False,task_success_claimed=False)
    with manifest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--idle-audit-source',type=Path,required=True)
    a=p.parse_args();collect(a.workspace,a.idle_audit_source)
