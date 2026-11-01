"""Close only the held queue after its preserved PhysX child has exited.

Operational audit for this workspace: no signal is sent to the learner, and
no policy/source file is changed. Refuses running or unverified learners.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/data/zengxiong/workspaces/tron1-foundation-completion-20261031-v1')
FOCUS = ROOT / 'isaac_contact_focus_v1'
SEQUENCE = ROOT / 'sac_sequence_v3'


def inspect(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    assert path.stat().st_uid == os.getuid()
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, argv=(path / 'cmdline').read_bytes().decode().split('\0'),
                state=fields[0], start_ticks=int(fields[19]))


def same_process(current, expected):
    assert current is not None and current['start_ticks'] == expected['start_ticks']
    assert current['argv'] == expected['argv']


def main():
    target = FOCUS / 'finalization.json'
    assert not target.exists()
    hold = json.loads((FOCUS / 'queue_hold.json').read_text())
    child = inspect(hold['preserved_physx_child']['pid'])
    if child is not None:
        assert child['start_ticks'] == hold['preserved_physx_child']['start_ticks']
        assert child['state'] == 'Z', 'Preserve the learner until it exits'
    source = ROOT / 'sac_source_v3/eval_sac_backend_400hz.py'
    spec = importlib.util.spec_from_file_location('focus_verifier', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = Path('/data/zengxiong/workspaces/tron1-ppo-backend-comparison-20261022/validation_source_v1/config/ppo_backend_400hz_comparison_v1.json')
    proof = module.cohort(SEQUENCE / 'physx_formal', protocol, False)
    assert proof['training_backend'] == 'physx' and not proof['smoke_only']
    assert not (SEQUENCE / 'mjlab_formal').exists(), 'Unexpected backend learner started'
    parent = inspect(hold['parent_before']['pid'])
    same_process(parent, hold['parent_before'])
    assert parent['state'] == 'T'
    old = json.loads((SEQUENCE / 'status.json').read_text())
    assert old['jobs'][-1]['name'] == 'physx_formal'
    (FOCUS / 'prior_sequence_status.json').write_text(json.dumps(old, indent=2) + '\n')
    # TERM is pending while stopped; CONT delivers it without advancing the queue.
    os.kill(parent['pid'], signal.SIGTERM)
    os.kill(parent['pid'], signal.SIGCONT)
    for _ in range(50):
        after = inspect(parent['pid'])
        if after is None or after['state'] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Queue termination not confirmed')
    # Stop the obsolete offline waiter only after matching its exact script and
    # dependency. This PID is from the read-only process audit, not a name kill.
    waiter = inspect(2965861)
    if waiter is not None and waiter['state'] != 'Z':
        assert str(ROOT / 'contact_analysis_source_v1/analyze_matched_contact_geometry.py') in waiter['argv']
        assert str(SEQUENCE / 'status.json') in waiter['argv']
        same_process(inspect(waiter['pid']), waiter)
        os.kill(waiter['pid'], signal.SIGTERM)
        for _ in range(50):
            after = inspect(waiter['pid'])
            if after is None or after['state'] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Offline waiter termination not confirmed')
    old['jobs'][-1].update(status='completed_verified_after_dispatcher_hold',
                          run_status_sha256=module.sha(SEQUENCE / 'physx_formal/run_status.json'))
    old.update(status='deferred_after_completed_physx_training',
               user_direction='Prioritize contact-model analysis and Isaac/PhysX training',
               finalization_audit=str(target), process_signals_sent=True,
               signals_to_training_child=[], finished_unix_time=time.time())
    (SEQUENCE / 'status.json').write_text(json.dumps(old, indent=2) + '\n')
    audit = dict(status='completed_verified', observed_unix_time=time.time(),
                 parent_pid=parent['pid'], parent_signals=['SIGTERM', 'SIGCONT'],
                 obsolete_analysis_waiter_pid=None if waiter is None else waiter['pid'],
                 learner_signals=[], actor_sha256=proof['input_sha256']['policy.pt'],
                 cohort_files_sha256=proof['input_sha256'],
                 training=json.loads((SEQUENCE / 'physx_formal/run_status.json').read_text()),
                 mjlab_formal_started=False, extra_training_budget=0)
    target.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
