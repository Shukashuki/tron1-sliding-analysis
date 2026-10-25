"""Wait on the exact nominal continuation; never restart or signal compute."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import time

import collect_contact_warm_refinement as collection


def run(a):
    root=a.workspace.resolve();record=root/'collection_waiter_status.json';assert not record.exists()
    state=dict(status='preflight',task_success_claimed=False)
    def save():record.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        assert collection.sha(a.idle_audit_source.read_bytes())=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
        spec=importlib.util.spec_from_file_location('_warm_collection_waiter_handle',a.idle_audit_source)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        launch=collection.read(root/'nominal_continuation_v1_launch.json')
        assert launch['pid']==a.expected_continuation_pid
        deadline=time.monotonic()+7200
        while True:
            p=helper.process_snapshot(launch['pid'])
            if p is None or p['state']=='Z':break
            assert p['start_ticks']==launch['start_ticks'] and p['argv']==launch['argv']
            assert time.monotonic()<deadline,'Observation timeout; all compute is preserved'
            state.update(status='waiting_for_exact_nominal_continuation',process_snapshot=p,last_observed_unix_time=time.time())
            save();time.sleep(15.)
        result=collection.collect(root,a.idle_audit_source)
        state.update(status='completed_collection_not_task_success',archive_sha256=result['archive_sha256'],archive_bytes=result['archive_bytes'])
        save()
    except BaseException as error:
        state.update(status='failed_no_retry_or_signal',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--idle-audit-source',type=Path,required=True)
    p.add_argument('--expected-continuation-pid',type=int,required=True);run(p.parse_args())
