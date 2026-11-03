"""Run paired speed levels serially, finishing the authorized predecessor first."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    import runtime_helpers as H
except ImportError:
    import dispatch_direct_brake_training as H


def inspect(bundle, verify_runtime=True):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Require physical unmasked GPU identities')
    for item in H.read(bundle/'manifest.json')['files']:
        raw = bundle/item['path']
        p = raw.resolve()
        if (raw.is_symlink() or not p.is_relative_to(bundle.resolve())
                or p.stat().st_size != item['bytes'] or H.sha(p) != item['sha256']):
            raise ValueError('Snapshot changed: '+str(p))
        if p.suffix == '.py':
            compile(p.read_text(), str(p), 'exec')
    plan = H.read(bundle/'plan.json')
    if (plan['pilot_seed'] != 43 or plan['confirmation_seeds'] != [44,45]
            or [r['scale'] for r in plan['levels']] != [1.5,2.,2.5,3.]
            or plan['maximum_formal_transitions'] != 294912000):
        raise ValueError('Predeclared study changed')
    if not verify_runtime:
        return plan, None
    runtime = plan['runtime']
    versions = {k:importlib.metadata.version(k) for k in plan['expected_versions']}
    if versions != plan['expected_versions']:
        raise ValueError('Runtime versions changed')
    asset = H.sha(runtime['asset'])
    if asset != H.read(bundle/'protocols/s150_direct_parent.json')['asset_sha256']:
        raise ValueError('Asset changed')
    upstream = H.read(bundle/'upstream_provenance.json')
    for name,digest in upstream.items():
        if name.endswith('scripts/train_tracking.py'):
            continue
        key = 'upstream/whole_body_tracking/'
        path = (Path(runtime['beyondmimic_root'])/name.removeprefix(key) if name.startswith(key)
                else Path(runtime['tracking_root'])/name.removeprefix('upstream/tron1-learning-from-human/'))
        if H.sha(path) != digest:
            raise ValueError('Upstream source changed: '+str(path))
    return plan, dict(python=sys.executable,versions=versions,asset_sha256=asset,
                      upstream=upstream,manifest_sha256=H.sha(bundle/'manifest.json'),GPU_allocated=False)


def process_identity(pid):
    path = Path('/proc')/str(pid)
    stat = (path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,uid=path.stat().st_uid,state=stat[0],ppid=int(stat[1]),
                start_ticks=int(stat[19]),exit_code=int(stat[49]),
                argv=[v.decode() for v in (path/'cmdline').read_bytes().split(b'\0') if v])


def check_identity(actual, expected, *, zombie=False):
    for key in ('pid','uid','start_ticks'):
        if actual[key] != expected[key]:
            raise ValueError('Process identity changed: '+key)
    if actual['uid'] != os.getuid():
        raise ValueError('Refuse to signal another user process')
    if not (zombie and actual['state']=='Z') and actual['argv'] != expected['argv']:
        raise ValueError('Process argv changed')


def handoff(plan, output, state, save):
    cfg = plan['predecessor']; root = Path(cfg['root'])
    path = root/cfg['handoff_file']
    if H.sha(path) != cfg['handoff_sha256']:
        raise ValueError('Predecessor handoff changed')
    old = H.read(path)
    expected_parent, expected_child = old['parent_identity'], old['child_identity']
    script = root/'recovery_dispatcher_20261103_v1.py'
    if H.sha(script) != old['previous_status']['recovery_dispatcher_sha256']:
        raise ValueError('Predecessor dispatcher changed')
    prior = old['previous_status']; record = copy.deepcopy(prior['current'])
    argv = record['argv']
    argument = lambda key: Path(argv[argv.index('--'+key)+1])
    while True:
        parent, child = process_identity(expected_parent['pid']), process_identity(expected_child['pid'])
        check_identity(parent,expected_parent);check_identity(child,expected_child,zombie=True)
        if parent['state'] != 'T' or child['ppid'] != parent['pid']:
            raise ValueError('Paused dispatcher/worker relationship changed')
        if child['state']=='Z':
            break
        state.update(status='waiting_for_current_low_speed_training',
                     current=dict(stage=record['stage'],pid=child['pid'],predecessor=True))
        save();time.sleep(plan['poll_seconds'])
    if child['exit_code'] != 0:
        # Let its own dispatcher reap and record the failed worker; no next stage.
        os.kill(parent['pid'],signal.SIGCONT)
        raise RuntimeError('Predecessor worker failed; resumed its error recorder')
    report = H.completed_training(argument('output-dir'),argument('protocol'),'formal')
    old_status = H.read(root/'runs_v1/status.json')
    if old_status != prior:
        raise ValueError('Predecessor state changed while paused')
    evidence = dict(parent=parent,child=child,worker_returncode=0,
                    checkpoint_sha256=report['checkpoint_sha256'],
                    actor_sha256=report['actor_sha256'],observed_unix_s=time.time(),
                    training_worker_signalled=False,handoff_sha256=H.sha(path))
    H.write(output/'predecessor_completion_before_release.json',evidence)
    # Only the stopped CPU dispatcher is retired, after its GPU child exited.
    # Revalidate immediately before signalling; never signal the training child.
    check_identity(process_identity(parent['pid']),expected_parent)
    check_identity(process_identity(child['pid']),expected_child,zombie=True)
    if process_identity(child['pid'])['state']!='Z':
        raise ValueError('Worker has not exited')
    os.kill(parent['pid'],signal.SIGKILL)
    record.update(returncode=0,completion_observed_unix_s=time.time(),
                  transitions=report['transitions'],report_sha256=H.sha(argument('output-dir')/'report.json'))
    old_status['completed'].append(record)
    old_status['formal_transitions_completed'] += report['transitions']
    old_status.update(status='deferred_for_high_speed',current=None,
                      handoff_completion=evidence,updated_unix_s=time.time())
    H.write(root/'runs_v1/status.json',old_status)
    H.write(root/'handoff_completed_20261103_v1.json',evidence)
    state.update(status='predecessor_completed_remaining_low_speed_deferred',current=None,
                 predecessor_completion=evidence);save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    a = parser.parse_args();a.bundle=a.bundle.resolve();a.output=a.output.resolve()
    plan,runtime_lock=inspect(a.bundle)
    if a.preflight_only:
        import json
        print(json.dumps(runtime_lock,indent=2));return
    a.output.mkdir(parents=True,exist_ok=False)
    queue_lock=(a.output/'.dispatcher.lock').open('a')
    fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    H.write(a.output/'runtime_lock.json',runtime_lock)
    state=dict(kind=plan['kind'],status='preparing',completed=[],current=None,
        manifest_sha256=H.sha(a.bundle/'manifest.json'),started_unix_s=time.time(),
        formal_transitions_completed=0,idle_wait_s=0,pairs=[],task_success_claimed=False)

    def save():
        state['updated_unix_s']=time.time();H.write(a.output/'status.json',state)

    def idle_device(label):
        while True:
            rows=H.query_gpus()
            state.update(status='waiting_for_idle_gpu',current={'stage':label},last_gpu_probe=rows);save()
            for row in rows:
                if not row['eligible']:continue
                lock=(a.output.parent.parent/f'.tron1-direct-brake-gpu-{row["index"]}.lock').open('a')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lock.close();continue
                checks=[row]
                for _ in range(2):
                    time.sleep(1);checks.append(H.query_gpus()[row['index']])
                if all(c['eligible'] and c['uuid']==row['uuid'] for c in checks):return row,checks,lock
                lock.close()
            if state['idle_wait_s']>=plan['idle_wait_limit_s']:
                raise RuntimeError('Idle GPU wait budget exhausted; no jobs displaced')
            time.sleep(plan['poll_seconds']);state['idle_wait_s']+=plan['poll_seconds']

    def run(label,directory,protocol,mode,source,motion,actor=None):
        inspect(a.bundle)
        if directory.exists():raise FileExistsError('No automatic overwrite/retry: '+str(directory))
        row,checks,lock=idle_device(label)
        runtime=plan['runtime']
        script='train_hockey_ppo.py' if source=='parent' else 'train_hockey_contact_search.py'
        argv=[runtime['python'],'-B','-u',str(a.bundle/source/script),
            '--protocol',str(protocol),'--motion',str(motion),'--variant','tracking',
            '--mode',mode,'--output-dir',str(directory),'--device','cuda:'+str(row['index'])]
        for key in ('asset','tracking_root','beyondmimic_root'):
            argv.extend(['--'+key.replace('_','-'),runtime[key]])
        if actor is not None:argv.extend(['--actor',str(actor)])
        directory.parent.mkdir(parents=True,exist_ok=True)
        log_path=H.run_log_path(directory)
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with log_path.open('xb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=a.output)
            record=dict(stage=label,pid=child.pid,argv=argv,config_sha256=H.sha(protocol),
                gpu_before_launch=checks,started_unix_s=time.time(),log=str(log_path))
            state.update(status='running',current=record);save()
            try:code=child.wait()
            finally:lock.close()
        record.update(returncode=code,finished_unix_s=time.time())
        if code:raise RuntimeError('Stage failed without retry: '+label)
        if mode in ('smoke','formal'):
            report=H.completed_training(directory,protocol,mode)
            record['transitions']=report['transitions']
            if mode=='formal':state['formal_transitions_completed']+=report['transitions']
        else:H.completed_evaluation(directory,protocol,mode,actor)
        record['report_sha256']=H.sha(directory/'report.json')
        state['completed'].append(record);state.update(status='between_stages',current=None);save()

    def train_pair(level,seed,phase):
        # Counterbalance serial ordering, never choose a controller by its score.
        order=plan['behaviors'] if seed%2 else list(reversed(plan['behaviors']))
        results={};pair_root=a.output/level['id']/f'seed{seed}'
        for behavior in order:
            base=pair_root/behavior;label=f'{level["id"]}/seed{seed}/{behavior}'
            ref_dir=a.bundle/level['reference_dir'];motion=ref_dir/(behavior+'_motion.npz')
            cfg=H.read(a.bundle/f'protocols/{level["id"]}_{behavior}_parent.json');cfg['seed']=seed
            parent=base/'parent_protocol.json';H.write_protocol_once(parent,cfg)
            run(label+'/parent/smoke',base/'parent/smoke',parent,'smoke','parent',motion)
            run(label+'/parent/reset_probe',base/'parent/reset_probe',parent,'eval','parent',motion,
                base/'parent/smoke/actor_normalized.pt')
            run(label+'/parent/formal',base/'parent/formal',parent,'formal','parent',motion)
            template=H.read(a.bundle/f'protocols/{level["id"]}_{behavior}_contact.json')
            template['seed']=seed;template['bilateral_sliding']['seed']=seed
            contact_cfg=H.bind_parent(template,base/'parent/formal',parent)
            contact=base/'contact_protocol.json';H.write_protocol_once(contact,contact_cfg)
            for mode in ('smoke','formal'):run(label+'/contact/'+mode,base/'contact'/mode,contact,mode,'contact',motion)
            continuation_cfg=H.bind_parent(contact_cfg,base/'contact/formal',contact,continuation=True)
            continuation=base/'continuation_protocol.json';H.write_protocol_once(continuation,continuation_cfg)
            for mode in ('smoke','formal'):
                run(label+'/continuation/'+mode,base/'continuation'/mode,continuation,mode,'contact',motion)
            run(label+'/evaluation',base/'evaluation',continuation,'eval-suite','contact',motion,
                base/'continuation/formal/actor_normalized.pt')
            command=[plan['runtime']['python'],'-B',str(a.bundle/'analysis/score_paired_stop_speed.py'),
                '--suite',str(base/'evaluation'),'--reference',str(motion),
                '--reference-report',str(ref_dir/'report.json'),'--output',str(base/'score.json')]
            with (base/'score.log').open('xb') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg'))
            results[behavior]=H.read(base/'score.json')
        import numpy as np
        initial={b:np.load(pair_root/b/'evaluation/nominal/final_rollout.npz',allow_pickle=False)
                 for b in plan['behaviors']}
        errors={k:float(abs(initial['direct'][k][0]-initial['hockey'][k][0]).max())
                for k in ('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel')}
        for values in initial.values():values.close()
        if max(errors.values())>5e-5:raise ValueError('Nominal initial physical states do not match')
        pair=dict(level=level['id'],seed=seed,phase=phase,initial_state_max_errors=errors,
                  passes={b:results[b]['passes'] for b in plan['behaviors']},
                  score_sha256={b:H.sha(pair_root/b/'score.json') for b in plan['behaviors']})
        H.write(pair_root/'comparison.json',pair);state['pairs'].append(pair);save()
        return pair

    save()
    try:
        handoff(plan,a.output,state,save)
        selected=None
        for level in plan['levels']:
            pair=train_pair(level,plan['pilot_seed'],'exploratory_speed_selection')
            selected=level
            if min(pair['passes'].values())<plan['minimum_passes_before_next_speed']:break
        state['selected_level']=selected;save()
        for seed in plan['confirmation_seeds']:train_pair(selected,seed,'paired_confirmation')
        expected=len(state['pairs'])*2*plan['formal_transitions_per_policy']
        if state['formal_transitions_completed']!=expected or expected>plan['maximum_formal_transitions']:
            raise ValueError('Transition accounting mismatch')
        state.update(status='paired_training_evaluation_and_scoring_completed',current=None);save()
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
