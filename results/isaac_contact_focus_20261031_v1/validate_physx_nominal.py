"""One nominal PhysX validation of the preserved, completed SAC learner."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/zengxiong/workspaces/tron1-foundation-completion-20261031-v1')
OUT = ROOT / 'isaac_contact_focus_v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p, name):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(p.parent))
        spec = importlib.util.spec_from_file_location(name, p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def main():
    sequence = read(ROOT / 'sac_sequence_v3/status.json')
    assert sequence['status'] == 'deferred_after_completed_physx_training'
    proof = read(OUT / 'finalization.json')
    assert proof['status'] == 'completed_verified'
    for name, expected in sequence['source_sha256'].items():
        assert sha(name) == expected
    case = next(j for j in sequence['jobs'] if j['name'] == 'pipeline_smoke_physx_physx_mu0p700')
    command = list(case['command'])
    folder = OUT / 'nominal_physx_mu0p700'
    assert not folder.exists()
    command[command.index('--cohort-dir') + 1] = str(ROOT / 'sac_sequence_v3/physx_formal')
    command[command.index('--output-dir') + 1] = str(folder)
    command.remove('--allow-smoke-policy')
    assert command[command.index('--backend') + 1] == 'physx'
    assert command[command.index('--mu') + 1] == '0.7'
    helper = load(ROOT / 'source_v1/continue_ppo_backend_400hz_matrix.py', 'focus_queue_helper')
    probe_path = Path('/data/zengxiong/workspaces/tron1-ppo-backend-comparison-20261022/source_snapshot_v2/scripts/run_ppo_leg_target_filter.py')
    probe = load(probe_path, 'focus_gpu_probe').query_gpu
    status = dict(status='waiting_for_idle_gpu', command=command, algorithm='SAC',
                  physics_backend='PhysX', mu=.7, single_environment=True,
                  source_sha256=sequence['source_sha256'], no_training=True)
    def save():
        (OUT / 'nominal_status.json').write_text(json.dumps(status, indent=2) + '\n')
    def waiting(observation):
        status['latest_gpu_observation'] = observation
        save()
    save()
    status['gpu_gate'] = helper.wait_idle(probe, 3, time.monotonic() + 600, waiting)
    env = os.environ.copy()
    assert 'CUDA_VISIBLE_DEVICES' not in env
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    status['status'] = 'running'; save()
    with (OUT / 'nominal_physx.log').open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        status['pid'] = child.pid; save()
        status['returncode'] = child.wait()
    assert status['returncode'] == 0
    report = read(folder / 'report.json')
    audit = report['matched_sac_backend_evaluation']
    assert audit['status'] == 'verified' and not audit['smoke_only']
    assert audit['actor_sha256'] == proof['actor_sha256']
    assert audit['recording_sha256'] == sha(folder / 'physics_trajectory.npz')
    status.update(status='completed_verified', report_sha256=sha(folder / 'report.json'),
                  actor_sha256=audit['actor_sha256'], recording_sha256=audit['recording_sha256'])
    save()
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = OUT / 'nominal_status.json'
        if path.is_file():
            status = read(path)
            status.update(status='stopped_without_retry', error=f'{type(error).__name__}: {error}')
            path.write_text(json.dumps(status, indent=2) + '\n')
        raise
