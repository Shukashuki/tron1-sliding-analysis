"""One read-only K episode, source/hash audit and fresh fully idle GPU preflight."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, path)


def run(args):
    root = args.workspace.resolve(); state_file = root/'execution_status.json'
    assert not state_file.exists()
    state = dict(kind='K_native_contact_manifold_single_episode', status='preflight',
        driver_pid=os.getpid(), driver_sha256=sha(__file__), task_success_claimed=False, other_jobs_stopped=False)
    write(state_file, state)
    try:
        assert sha(args.idle_audit_source) == '6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
        spec = importlib.util.spec_from_file_location('_manifold_idle_audit', args.idle_audit_source)
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        for pid in (463064, 463139, 464400, 466721, 499759, 502008):
            p = audit.process_snapshot(pid); assert p is None or p['state'] == 'Z', ('Prior owned job active', p)
        manifest = read(root/'source_snapshot_v1.manifest.json')
        assert sha(root/'source_snapshot_v1.tar.gz') == manifest['archive_sha256']
        snapshot = root/'source_snapshot_v1'; snapshot.mkdir(exist_ok=False)
        expected = {r['path']:r for r in manifest['files']}
        with tarfile.open(root/'source_snapshot_v1.tar.gz') as t:
            members = t.getmembers(); assert len(members) == len(expected) and {m.name for m in members} == set(expected)
            for m in members:
                p = Path(m.name); assert m.isfile() and not p.is_absolute() and '..' not in p.parts and p.suffix in ('.py', '.json')
                data = t.extractfile(m).read(); assert len(data) == expected[m.name]['bytes'] and hashlib.sha256(data).hexdigest() == expected[m.name]['sha256']
                target = snapshot/p; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        ledger = {str(snapshot/n): r['sha256'] for n, r in expected.items()}
        deadline = time.monotonic()+1800.
        state['status'] = 'waiting_for_fully_idle_GPU'; write(state_file, state)
        while True:
            observed = audit.query_gpus(); gpu = next((g for g in observed if g['idle']), None)
            if gpu: break
            assert time.monotonic() < deadline, 'No fully idle GPU before declared wait deadline'
            time.sleep(15.)
        assert all(sha(p) == digest for p, digest in ledger.items())
        command = read(snapshot/'config/parent_launch.json')['argv']
        for flag in ('--command-mode', '--probe-protocol'):
            assert command.count(flag) == 1; i = command.index(flag); del command[i:i+2]
        command[3] = str(snapshot/'scripts/eval_physx_contact_manifold_probe.py')
        output = root/'physx_v1'
        command[command.index('--output-dir')+1] = str(output)
        command[command.index('--physical-gpu-index')+1] = str(gpu['physical_index'])
        environment = os.environ.copy(); environment.pop('CUDA_VISIBLE_DEVICES', None)
        environment.update(PYTHONDONTWRITEBYTECODE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            TRON1_BEYONDMIMIC_ROOT='/data/zengxiong/workspaces/tron1-beyondmimic-20261002/third_party/whole_body_tracking')
        with (root/'physx_v1.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', pid=child.pid, argv=command, gpu_audit=observed, selected_gpu=gpu,
                child_process_at_launch=audit.process_snapshot(child.pid), source_sha256=ledger)
            write(state_file, state)
            while child.poll() is None: time.sleep(2.)
        state['returncode'] = child.returncode
        assert child.returncode == 0, ('Native episode failed; no retry', child.returncode)
        proof = read(output/'manifold_status.json')
        assert proof['kind'] == 'K_readonly_PhysX_native_contact_manifold_probe'
        assert proof['physics_trajectory_sha256'] == 'cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a'
        assert sha(output/'physics_trajectory.npz') == proof['physics_trajectory_sha256']
        assert sha(output/'manifold_trajectory.npz') == proof['manifold_probe']['native_trace_sha256']
        assert proof['manifold_probe']['native_pre_step_samples'] == 2792
        assert read(output/'report.json')['status'] == 'evaluated'
        state.update(status='completed_verified_requires_native_contact_analysis', proof_sha256=sha(output/'manifold_status.json'))
        write(state_file, state)
    except BaseException as error:
        state.update(status='failed_no_retry_or_process_signal', error=repr(error)); write(state_file, state); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('workspace', 'idle-audit-source'): p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
