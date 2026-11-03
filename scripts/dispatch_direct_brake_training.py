"""Serial direct-brake PPO queue: fresh seeds, checked stages, idle RTX only.

There is no termination of existing processes, policy selection, outcome-based
early exit, package install, or automatic retry after a failed simulator run.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_rows(xml):
    rows = []
    for index, gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        processes = [dict(pid=int(p.findtext('pid')), type=p.findtext('type'))
                     for p in gpu.findall('processes/process_info')]
        def number(name):
            try:
                return float(gpu.findtext(name).split()[0])
            except (ValueError, AttributeError):
                return None
        memory = number('fb_memory_usage/used')
        utilization = number('utilization/gpu_util')
        name = gpu.findtext('product_name') or ''
        rows.append(dict(index=index, uuid=gpu.findtext('uuid'), name=name,
                         memory_used_mib=memory, utilization_percent=utilization,
                         processes=processes,
                         eligible=('4090' in name and not processes and memory is not None
                                   and memory < 256 and utilization == 0)))
    if not rows:
        raise ValueError('Missing GPU telemetry; fail closed')
    return rows


def query_gpus():
    return gpu_rows(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=20))


def inspect(bundle, *, verify_runtime=True):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Require physical unmasked GPU identities')
    manifest = read(bundle/'manifest.json')
    for item in manifest['files']:
        path = (bundle/item['path']).resolve()
        if (not path.is_relative_to(bundle.resolve()) or path.is_symlink()
                or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']):
            raise ValueError('Snapshot identity changed: '+str(path))
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
    plan = read(bundle/'plan.json')
    if plan['seeds'] != [43, 44, 45, 46, 47] or plan['unique_formal_transitions_total'] != 184320000:
        raise ValueError('Predeclared seed cohort/budget changed')
    if not verify_runtime:
        return plan, None
    runtime = plan['runtime']
    versions = {key: importlib.metadata.version(key) for key in plan['expected_versions']}
    if versions != plan['expected_versions']:
        raise ValueError('Runtime version lock differs: '+str(versions))
    protocol = read(bundle/'protocols/seed43_parent.json')
    asset_hash = sha(runtime['asset'])
    if asset_hash != protocol['asset_sha256']:
        raise ValueError('Robot asset changed')
    upstream = read(bundle/'upstream_provenance.json')
    for name, digest in upstream.items():
        if name.endswith('scripts/train_tracking.py'):
            continue
        if name.startswith('upstream/whole_body_tracking/'):
            path = Path(runtime['beyondmimic_root'])/name.removeprefix('upstream/whole_body_tracking/')
        else:
            path = Path(runtime['tracking_root'])/name.removeprefix('upstream/tron1-learning-from-human/')
        if sha(path) != digest:
            raise ValueError('Upstream runtime source changed: '+str(path))
    return plan, dict(python=sys.executable, versions=versions, upstream=upstream,
                      asset_sha256=asset_hash, GPU_allocated=False,
                      manifest_sha256=sha(bundle/'manifest.json'))


def completed_training(folder, protocol, mode):
    report = read(folder/'report.json')
    cfg = read(protocol)
    expected = cfg[mode]['num_envs']*cfg[mode]['iterations']*cfg['ppo']['steps_per_env']
    if (report['status'] != 'training_completed_not_task_success'
            or report['transitions'] != expected or report['config_sha256'] != sha(protocol)
            or report.get('bilateral_audit', {}).get('fault_seen', False)
            or report['checkpoint_sha256'] != sha(folder/'model_final.pt')
            or report['actor_sha256'] != sha(folder/'actor_normalized.pt')):
        raise ValueError('Training did not complete its exact finite budget: '+str(folder))
    return report


def bind_parent(protocol, completed, source_protocol, *, continuation=False):
    cfg = copy.deepcopy(protocol)
    report = completed_training(completed, source_protocol, 'formal')
    cfg['warm_start'] = dict(checkpoint=str(completed/'model_final.pt'),
                            checkpoint_sha256=report['checkpoint_sha256'],
                            actor=str(completed/'actor_normalized.pt'), actor_sha256=report['actor_sha256'])
    if continuation:
        cfg['formal']['iterations'] = 750
        cfg['training_continuation'] = dict(kind='hockey_optimizer_continuation_v1',
            source_report=str(completed/'report.json'), source_report_sha256=sha(completed/'report.json'),
            source_protocol=str(source_protocol), source_protocol_sha256=sha(source_protocol))
    return cfg


def run_log_path(directory):
    # A decimal material value is part of the directory name, not an extension.
    return directory.parent/(directory.name+'.log')


def write_protocol_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Refuse to replace an existing runtime protocol: '+str(path))
    else:
        write(path, value)


def completed_evaluation(directory, protocol, mode, actor):
    report = read(directory/'report.json')
    expected = ('validation_suite_completed_requires_independent_qualification' if mode == 'eval-suite'
                else 'completed_requires_independent_qualification')
    if (report['status'] != expected or report['actor_sha256'] != sha(actor)
            or report['config_sha256'] != sha(protocol)):
        raise ValueError('Evaluation did not complete: '+str(directory))
    cases = read(protocol)['evaluation_cases'] if mode == 'eval-suite' else []
    if cases and [r['id'] for r in report['cases']] != [r['id'] for r in cases]:
        raise ValueError('Evaluation case list differs')
    locations = [directory/case['id'] for case in cases] if cases else [directory]
    for index, location in enumerate(locations):
        item = read(location/'report.json') if cases else report
        if cases and sha(location/'report.json') != report['cases'][index]['report_sha256']:
            raise ValueError('Case report identity changed')
        for name, digest in item['artifacts'].items():
            if sha(location/name) != digest:
                raise ValueError('Evaluation artifact changed: '+str(location/name))
        audit = read(location/'common_reset_audit.json')
        if audit['maximum_state_error'] > 5e-5 or not audit['previous_actions_zero']:
            raise ValueError('Common physical reset validation failed')
    return report


def verify_completed_record(record, directory, protocol, mode, actor):
    argv = record['argv']
    get = lambda key: argv[argv.index('--'+key)+1]
    if (record['returncode'] != 0 or record['config_sha256'] != sha(protocol)
            or get('protocol') != str(protocol) or get('output-dir') != str(directory)
            or get('mode') != mode or record['report_sha256'] != sha(directory/'report.json')):
        raise ValueError('Completed stage identity changed')
    if mode in ('formal', 'smoke'):
        completed_training(directory, protocol, mode)
    else:
        if get('actor') != str(actor):
            raise ValueError('Completed stage actor changed')
        completed_evaluation(directory, protocol, mode, actor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume-unlaunched-stage', action='store_true',
                        help='Verify and reuse completed stages after a pre-launch dispatcher failure')
    a = parser.parse_args()
    a.bundle, a.output = a.bundle.resolve(), a.output.resolve()
    plan, runtime_lock = inspect(a.bundle)
    if a.preflight_only:
        print(json.dumps(runtime_lock, indent=2))
        return
    a.output.mkdir(parents=True, exist_ok=a.resume_unlaunched_stage)
    queue_lock = (a.output/'.dispatcher.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.resume_unlaunched_stage:
        state = read(a.output/'status.json')
        if (state['status'] != 'stopped' or (state.get('current') or {}).get('pid') is not None
                or not state.get('error', '').startswith('FileExistsError:')
                or state['manifest_sha256'] != sha(a.bundle/'manifest.json')
                or read(a.output/'runtime_lock.json') != runtime_lock):
            raise ValueError('Only an unlaunched stage with intact runtime can resume')
        stages = [r['stage'] for r in state['completed']]
        accounted = sum(r.get('transitions', 0) for r in state['completed'] if r['stage'].endswith('/formal'))
        if len(stages) != len(set(stages)) or accounted != state['formal_transitions_completed']:
            raise ValueError('Completed-stage accounting changed')
        archive = a.output/'resume_after_log_collision_20261103.json'
        if archive.exists():
            raise FileExistsError('This explicit recovery already has an audit record')
        write(archive, dict(previous_status=state, dispatcher_sha256=sha(__file__),
                            source_bundle_unchanged=True, reward_or_budget_changes=False))
        state.pop('error', None)
        state.update(status='resuming_verified_stages', current=None,
                     recovery_dispatcher_sha256=sha(__file__), resumed_unix_s=time.time())
    else:
        write(a.output/'runtime_lock.json', runtime_lock)
        state = dict(kind=plan['kind'], status='preparing', completed=[], current=None,
                     manifest_sha256=sha(a.bundle/'manifest.json'), started_unix_s=time.time(),
                     formal_transitions_completed=0, idle_wait_s=0, task_success_claimed=False)

    def save():
        state['updated_unix_s'] = time.time()
        write(a.output/'status.json', state)

    def idle_device(label):
        while True:
            rows = query_gpus()
            state.update(status='waiting_for_idle_gpu', current={'stage': label}, last_gpu_probe=rows)
            save()
            for row in rows:
                if not row['eligible']:
                    continue
                lock = open(a.output.parent.parent/f'.tron1-direct-brake-gpu-{row["index"]}.lock', 'a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                checks = [row]
                for _ in range(2):
                    time.sleep(1)
                    checks.append(query_gpus()[row['index']])
                if all(c['eligible'] and c['uuid'] == row['uuid'] for c in checks):
                    return row, checks, lock
                lock.close()
            if state['idle_wait_s'] >= plan['idle_wait_limit_s']:
                raise RuntimeError('Idle-GPU waiting budget exhausted; no workload displaced')
            time.sleep(plan['poll_seconds'])
            state['idle_wait_s'] += plan['poll_seconds']

    def run(label, directory, protocol, mode, source, actor=None):
        inspect(a.bundle)
        prior = next((r for r in state['completed'] if r['stage'] == label), None)
        if prior is not None:
            if not a.resume_unlaunched_stage:
                raise ValueError('Unexpected completed stage in a fresh queue')
            verify_completed_record(prior, directory, protocol, mode, actor)
            return
        if directory.exists():
            raise FileExistsError('Uncompleted stage directory exists; no automatic retry: '+str(directory))
        row, checks, lock = idle_device(label)
        script = 'train_hockey_ppo.py' if source == 'parent' else 'train_hockey_contact_search.py'
        runtime = plan['runtime']
        argv = [runtime['python'], '-B', '-u', str(a.bundle/source/script),
                '--protocol', str(protocol), '--motion', str(a.bundle/'motion.npz'),
                '--variant', 'tracking', '--mode', mode, '--output-dir', str(directory),
                '--device', 'cuda:'+str(row['index'])]
        for key in ('asset', 'tracking_root', 'beyondmimic_root'):
            argv.extend(['--'+key.replace('_', '-'), runtime[key]])
        if actor is not None:
            argv.extend(['--actor', str(actor)])
        directory.parent.mkdir(parents=True, exist_ok=True)
        log_path = run_log_path(directory)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        with log_path.open('xb') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=a.output)
            record = dict(stage=label, pid=child.pid, argv=argv, config_sha256=sha(protocol),
                          gpu_before_launch=checks, started_unix_s=time.time(), log=str(log_path))
            state.update(status='running', current=record)
            save()
            try:
                code = child.wait()
            finally:
                lock.close()
        record.update(returncode=code, finished_unix_s=time.time())
        if code:
            raise RuntimeError('Run failed without retry: '+label)
        if mode in ('formal', 'smoke'):
            report = completed_training(directory, protocol, mode)
            if mode == 'formal':
                state['formal_transitions_completed'] += report['transitions']
            record['transitions'] = report['transitions']
        else:
            report = completed_evaluation(directory, protocol, mode, actor)
        record['report_sha256'] = sha(directory/'report.json')
        state['completed'].append(record)
        state.update(status='between_stages', current=None)
        save()

    save()
    try:
        for seed in plan['seeds']:
            seed_dir = a.output/f'seed{seed}'
            seed_dir.mkdir(exist_ok=a.resume_unlaunched_stage)
            parent_cfg = a.bundle/f'protocols/seed{seed}_parent.json'
            run(f'seed{seed}/parent/smoke', seed_dir/'parent/smoke', parent_cfg, 'smoke', 'parent')
            run(f'seed{seed}/parent/reset_probe', seed_dir/'parent/reset_probe', parent_cfg,
                'eval', 'parent', seed_dir/'parent/smoke/actor_normalized.pt')
            run(f'seed{seed}/parent/formal', seed_dir/'parent/formal', parent_cfg, 'formal', 'parent')
            for arm in plan['material_arms']:
                arm_dir = seed_dir/arm
                template = read(a.bundle/f'protocols/seed{seed}_{arm}.json')
                cfg = bind_parent(template, seed_dir/'parent/formal', parent_cfg)
                stage_cfg = arm_dir/'contact_protocol.json'
                write_protocol_once(stage_cfg, cfg)
                for mode in ('smoke', 'formal'):
                    run(f'seed{seed}/{arm}/contact/{mode}', arm_dir/'contact'/mode, stage_cfg, mode, 'contact')
                continuation = bind_parent(cfg, arm_dir/'contact/formal', stage_cfg, continuation=True)
                continuation_cfg = arm_dir/'continuation_protocol.json'
                write_protocol_once(continuation_cfg, continuation)
                for mode in ('smoke', 'formal'):
                    run(f'seed{seed}/{arm}/continuation/{mode}', arm_dir/'continuation'/mode,
                        continuation_cfg, mode, 'contact')
                actor = arm_dir/'continuation/formal/actor_normalized.pt'
                run(f'seed{seed}/{arm}/evaluation', arm_dir/'evaluation', continuation_cfg, 'eval-suite', 'contact', actor)
                own_mu = (round(.7*continuation.get('static_dynamic_friction_ratio', 1.), 5), .7)
                for ms, md in plan['cross_material_nominal']:
                    if (ms, md) == own_mu:
                        continue  # already included as the suite's nominal episode
                    cross = copy.deepcopy(continuation)
                    cross.update(evaluation_mu=md, static_dynamic_friction_ratio=ms/md)
                    label = f'cross_s{ms:.2f}_d{md:.2f}'
                    cross_cfg = arm_dir/(label+'_protocol.json')
                    write_protocol_once(cross_cfg, cross)
                    run(f'seed{seed}/{arm}/{label}', arm_dir/label, cross_cfg, 'eval', 'contact', actor)
        if state['formal_transitions_completed'] != plan['unique_formal_transitions_total']:
            raise ValueError('Final unique transition accounting differs')
        state.update(status='training_and_recording_completed_requires_independent_qualification', current=None)
        save()
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
