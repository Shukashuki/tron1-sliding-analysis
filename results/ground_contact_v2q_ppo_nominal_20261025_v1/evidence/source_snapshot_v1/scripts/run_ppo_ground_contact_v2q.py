"""Sequential Q smoke then matched formal PPO; source-bound proofs, idle GPU only."""
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


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def run(args):
    root, snapshot = args.workspace.resolve(), args.snapshot.resolve()
    assert not (root/'driver_status.json').exists()
    state = dict(kind='Q_unsupported_geometry_gap_PPO_training', status='preflight', children=[],
        from_scratch=True, task_success_claimed=False, other_jobs_stopped=False)
    write(root/'driver_status.json', state)
    try:
        assert sha(args.idle_audit_source) == '6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
        spec = importlib.util.spec_from_file_location('_Q_idle_audit', args.idle_audit_source)
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        for pid in (463064, 463139, 464400, 466721, 499759, 502008, 510210, 510215, 516431, 516432):
            observed = audit.process_snapshot(pid)
            assert observed is None or observed['state'] == 'Z', ('Previous owned job remains', observed)
        manifest = json.loads((root/'source_snapshot_v1.manifest.json').read_text())
        ledger = {str(snapshot/v['path']): v['sha256'] for v in manifest['files']}
        assert all(sha(p) == digest for p, digest in ledger.items())
        template = json.loads((snapshot/'config/ground_contact_v2q_formal_parent_launch_template.json').read_text())['argv']
        protocol = snapshot/'config/ppo_ground_contact_v2q_400hz_v1.json'
        environment = os.environ.copy(); environment.pop('CUDA_VISIBLE_DEVICES', None)
        environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1')
        sys.path.insert(0, str(snapshot/'scripts'))
        import eval_ppo_ground_contact_v2q as verify
        for mode, envs, iterations in (('smoke', 32, 2), ('formal', 512, 1000)):
            state['status'] = 'waiting_for_fully_idle_GPU_'+mode; write(root/'driver_status.json', state)
            while True:
                observed = audit.query_gpus()
                gpu = next((g for g in observed if g['idle']), None)
                if gpu: break
                time.sleep(15.)
            assert all(sha(p) == digest for p, digest in ledger.items())
            command = list(template)
            command[2] = str(snapshot/'scripts/train_ppo_ground_contact_v2q.py')
            for option, value in {'--cohort-protocol': str(protocol), '--cohort-run-kind': mode,
                    '--declared-input-dir': str(root/(mode+'_v1_declared_inputs')),
                    '--output-dir': str(root/(mode+'_v1')), '--num-envs': str(envs),
                    '--iterations': str(iterations), '--device': 'cuda:'+str(gpu['physical_index'])}.items():
                assert command.count(option) == 1; command[command.index(option)+1] = value
            with (root/(mode+'_v1.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
                row = dict(mode=mode, pid=child.pid, argv=command, status='running',
                    gpu_audit=observed, selected_gpu=gpu, started_unix_time=time.time(),
                    process_at_launch=audit.process_snapshot(child.pid))
                state['children'].append(row); state['status'] = 'running_'+mode
                write(root/'driver_status.json', state); write(root/(mode+'_v1_launch.json'), row)
                while child.poll() is None: time.sleep(2.)
            row['returncode'] = child.returncode
            assert child.returncode == 0, ('Training failed, no retry', mode, child.returncode)
            cohort = verify.verify_training(root/(mode+'_v1'), protocol, allow_smoke=mode == 'smoke')
            proof = dict(mode=mode, status='completed_verified', transitions=1536 if mode == 'smoke' else 12288000,
                normalized_actor_sha256=cohort['input_sha256']['policy.pt'],
                training_audit_sha256=sha(root/(mode+'_v1/ground_contact_v2q_training_audit.json')),
                initial_policy_state_sha256=cohort['training_audit']['initial_policy_state_sha256'],
                contact_runtime_audit=cohort['training_audit']['contact_runtime_audit'])
            write(root/(mode+'_v1_strict_verification.json'), proof)
            row.update(status='completed_verified', proof_sha256=sha(root/(mode+'_v1_strict_verification.json')))
            write(root/'driver_status.json', state)
        state['status'] = 'smoke_and_formal_completed_require_nominal_validation'
        write(root/'driver_status.json', state)
    except BaseException as error:
        state.update(status='failed_no_retry_or_process_signal', error=repr(error))
        write(root/'driver_status.json', state)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('workspace', 'snapshot', 'idle-audit-source'): p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
