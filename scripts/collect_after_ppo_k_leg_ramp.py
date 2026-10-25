"""Read-only exact-handle wait, then collect verified terminal ramp experiments."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import time

import collect_ppo_k_leg_ramp as collector


def run(a):
    root=a.workspace.resolve();assert Path(a.status_name).name==a.status_name
    record=root/a.status_name
    assert not record.exists()
    state=dict(status='preflight',task_success_claimed=False)
    def save():record.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        assert collector.sha(a.idle_audit_source.read_bytes())=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
        spec=importlib.util.spec_from_file_location('_ramp_collection_waiter_process',a.idle_audit_source)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        assert Path(a.continuation_launch_name).name==a.continuation_launch_name
        launch=collector.read(root/a.continuation_launch_name)
        assert launch['pid']==a.expected_continuation_pid
        deadline=time.monotonic()+7200
        while True:
            p=helper.process_snapshot(launch['pid'])
            if p is None or p['state']=='Z':break
            assert p['start_ticks']==launch['start_ticks'] and p['argv']==launch['argv']
            assert time.monotonic()<deadline,'Observation deadline; all jobs preserved'
            state.update(status='waiting_for_exact_evaluation_continuation',last_process_snapshot=p,
                last_observed_unix_time=time.time());save();time.sleep(15.)
        result=collector.collect(root,a.idle_audit_source,nominal_name=a.nominal_name,
            evaluation_snapshot_name=a.evaluation_snapshot_name,continuation_launch_name=a.continuation_launch_name)
        state.update(status='completed_collection_not_task_success',archive_sha256=result['archive_sha256'],
            archive_bytes=result['archive_size_bytes'],completed_unix_time=time.time());save()
    except BaseException as error:
        state.update(status='failed_no_retry_or_process_signal',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--idle-audit-source',type=Path,required=True)
    p.add_argument('--expected-continuation-pid',type=int,required=True)
    p.add_argument('--status-name',default='collection_waiter_status.json')
    p.add_argument('--nominal-name',default='nominal_v1');p.add_argument('--evaluation-snapshot-name',default='evaluation_snapshot_v1')
    p.add_argument('--continuation-launch-name',default='evaluation_continuation_v1_launch.json');run(p.parse_args())
