"""Allowlist completed Q native evidence; omit checkpoints, models and large logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def run(args):
    root = args.workspace.resolve()
    state = read(root/'nominal_v1/status.json')
    assert state['status'] == 'both_nominal_engines_completed_require_analysis'
    assert len(state['children']) == 2 and all(c['status'] == 'completed_verified' and c['returncode'] == 0 for c in state['children'])
    assert read(root/'driver_status.json')['status'] == 'smoke_and_formal_completed_require_nominal_validation'
    names = ['driver_status.json','driver_launch.json','driver.log',
        'source_snapshot_v1.manifest.json','source_snapshot_v1.tar.gz',
        'smoke_v1_launch.json','smoke_v1_strict_verification.json',
        'formal_v1_launch.json','formal_v1_strict_verification.json',
        'nominal_driver_launch.json','nominal_v1/status.json']
    for mode in ('smoke_v1', 'formal_v1'):
        for name in ('ground_contact_v2q_training_audit.json', 'cohort_completion_audit.json',
                'manifest.json', 'policy_contract.json', 'final_policy_contract.json', 'run_status.json',
                'runner_config.json', 'cohort_startup_audit.json', 'ground_contact_v2q_startup_audit.json'):
            names.append(mode+'/'+name)
    for engine in ('physx', 'mujoco'):
        prefix = 'nominal_v1/'+engine+'/'
        proof = read(root/prefix/'strict_verification.json')
        recording = 'physics_trajectory.npz' if engine == 'physx' else 'rollout.npz'
        assert sha(root/prefix/recording) == proof['recording_sha256']
        assert sha(root/prefix/'report.json') == proof['report_sha256']
        names.extend(prefix+name for name in ('strict_verification.json', 'report.json', recording))
        names.append('nominal_v1/'+engine+'.log')
    source = read(root/'source_snapshot_v1.manifest.json')
    assert sha(root/'source_snapshot_v1.tar.gz') == source['archive_sha256']
    for row in source['files']:
        name = 'source_snapshot_v1/'+row['path']
        assert sha(root/name) == row['sha256']; names.append(name)
    assert len(set(names)) == len(names)
    for name in names:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts and (root/p).is_file() and not (root/p).is_symlink()
        assert p.suffix in ('.py', '.json', '.log', '.npz', '.gz')
    manifest = dict(kind='completed_Q_dual_engine_native_evidence',
        models_weights_credentials_binaries_included=False, formal_transitions=12288000,
        actor_sha256=state['normalized_actor_sha256'], collector_source_sha256=sha(__file__),
        files={n: dict(bytes=(root/n).stat().st_size, sha256=sha(root/n)) for n in names})
    target, archive = root/'completed_collection_v1.manifest.json', root/'completed_collection_v1.tar.gz'
    assert not target.exists() and not archive.exists()
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    with tarfile.open(archive, 'w:gz') as t:
        for name in names: t.add(root/name, arcname=name, recursive=False)
        t.add(target, arcname='collection_manifest.json', recursive=False)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(names))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--workspace', type=Path, required=True)
    run(p.parse_args())
