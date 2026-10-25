"""After exact Q training driver exits, validate the same actor in both engines."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(tmp, path)


def run(args):
    root, snapshot = args.workspace.resolve(), args.snapshot.resolve()
    output = root/'nominal_v1'; output.mkdir(exist_ok=False)
    state = dict(kind='Q_nominal_dual_engine', status='preflight', children=[], task_success_claimed=False)
    write(output/'status.json', state)
    try:
        assert sha(args.idle_audit_source) == '6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
        spec = importlib.util.spec_from_file_location('_Q_nominal_audit', args.idle_audit_source)
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        launch = read(root/'driver_launch.json')
        assert launch['pid'] == args.training_driver_pid
        pinned = None
        while True:
            process = audit.process_snapshot(args.training_driver_pid)
            if process is None or process['state'] == 'Z': break
            assert process['uid'] == os.getuid() and process['argv'] == launch['argv']
            if pinned is None: pinned = process['start_ticks']
            assert process['start_ticks'] == pinned
            state.update(status='waiting_for_exact_training_driver', live_training_driver=process)
            write(output/'status.json', state); time.sleep(5.)
        completed = read(root/'driver_status.json')
        assert completed['status'] == 'smoke_and_formal_completed_require_nominal_validation'
        assert len(completed['children']) == 2 and all(c['status'] == 'completed_verified' for c in completed['children'])
        for c in completed['children']:
            observed = audit.process_snapshot(c['pid']); assert observed is None or observed['state'] == 'Z'
        manifest = read(root/'source_snapshot_v1.manifest.json')
        ledger = {str(snapshot/v['path']): v['sha256'] for v in manifest['files']}
        ledger[str(Path(__file__).resolve())] = sha(__file__)
        assert all(sha(p) == digest for p, digest in ledger.items())
        state.update(source_sha256=ledger, training_driver_status_sha256=sha(root/'driver_status.json'))
        sys.path.insert(0, str(snapshot/'scripts'))
        import eval_ppo_ground_contact_v2q as evaluator
        protocol = snapshot/'config/ppo_ground_contact_v2q_400hz_v1.json'
        cohort = evaluator.verify_training(root/'formal_v1', protocol)
        state['normalized_actor_sha256'] = cohort['input_sha256']['policy.pt']
        environment = os.environ.copy(); environment.pop('CUDA_VISIBLE_DEVICES', None)
        environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1',
            TRON1_BEYONDMIMIC_ROOT='/data/zengxiong/workspaces/tron1-beyondmimic-20261002/third_party/whole_body_tracking')
        for engine in ('mujoco', 'physx'):
            gpu, observed = None, None
            if engine == 'physx':
                state['status'] = 'waiting_for_fully_idle_GPU'; write(output/'status.json', state)
                while True:
                    observed = audit.query_gpus(); gpu = next((g for g in observed if g['idle']), None)
                    if gpu: break
                    time.sleep(15.)
            assert all(sha(p) == digest for p, digest in ledger.items())
            legacy = '/data/zengxiong/workspaces/tron1-controller-comparison-20261015/controller_evaluation_source_v2/scripts/'
            command = [sys.executable, '-B', '-u', str(snapshot/'scripts/eval_ppo_ground_contact_v2q.py'),
                '--training-dir', str(root/'formal_v1'), '--protocol', str(protocol), '--legacy-evaluator',
                legacy+('eval_tracking_mujoco.py' if engine == 'mujoco' else 'eval_tracking_friction_isaac.py'),
                '--motion-file', '/data/zengxiong/workspaces/tron1-drift-turn-worst-load-20261014/motion.npz',
                '--output-dir', str(output/engine), '--mu', '.7', '--backend', engine]
            if engine == 'mujoco': command += ['--model', '/data/zengxiong/workspaces/tron1-sliding-analysis-20261014/mujoco_model.xml']
            else: command += ['--asset-path', '/data/zengxiong/workspaces/tron1-beyondmimic-20261002/outputs/training-asset/WF_TRON1A.usda', '--physical-gpu-index', str(gpu['physical_index'])]
            with (output/(engine+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
                row = dict(engine=engine, pid=child.pid, argv=command, status='running', gpu_audit=observed, selected_gpu=gpu)
                state['children'].append(row); state['status'] = 'running_'+engine; write(output/'status.json', state)
                while child.poll() is None: time.sleep(2.)
            row['returncode'] = child.returncode; assert child.returncode == 0, ('Validation failed; no retry', engine)
            proof = evaluator.validate_output(output/engine, root/'formal_v1', protocol, backend=engine)
            assert proof['checkpoint_sha256'] == cohort['input_sha256']['model_final.pt']
            write(output/engine/'strict_verification.json', proof)
            row.update(status='completed_verified', verification=proof); write(output/'status.json', state)
        state['status'] = 'both_nominal_engines_completed_require_analysis'; write(output/'status.json', state)
    except BaseException as error:
        state.update(status='failed_no_retry_or_process_signal', error=repr(error)); write(output/'status.json', state); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('workspace', 'snapshot', 'idle-audit-source'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--training-driver-pid', type=int, required=True)
    run(p.parse_args())
