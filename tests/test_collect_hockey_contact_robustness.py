"""Only completed, declared robustness trajectories may leave the remote host."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_hockey_contact_search import REMOTE_READER


def fixture(tmp_path):
    def write(relative,value):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    bundle='source_snapshot_v7';run='robustness_v7_v1'
    manifest=write(bundle+'/manifest.json',{'files':[]})
    write(bundle+'/plan.json',{'cells':[{'id':'candidate','protocol':'protocol.json'}]})
    cases=[{'id':'nominal'}]+[{'id':f'perturb_{i:02d}'} for i in range(32)]
    write(bundle+'/protocol.json',{'evaluation_cases':cases})
    write(run+'/status.json',dict(status='running',candidate='candidate',current=None,completed=[],
        source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    write(run+'/eval_suite/report.json',dict(status='validation_suite_running'))
    directory=tmp_path/run/'eval_suite/nominal';directory.mkdir()
    for name in ('case.json','final_rollout.npz','native_contact_buffers.npz'):
        (directory/name).write_bytes(b'fixture-data')
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    write(run+'/eval_suite/nominal/report.json',dict(case_id='nominal',artifacts=artifacts))
    # These files must never enter the collection, even if they exist.
    (directory/'model_final.pt').write_bytes(b'not-for-transfer')
    write(run+'/eval_suite/perturb_00/case.json',{'partial':True})
    (tmp_path/run/'eval_suite/perturb_00/final_rollout.npz').write_bytes(b'in-progress')
    return bundle,run


def collect(tmp_path,bundle,run,raw):
    code=REMOTE_READER
    for token,value in dict(REMOTE_ROOT=str(tmp_path),SNAPSHOT_NAME=bundle,RUN_NAME=run,
        LAUNCH_NAME='launch_v7.json',INCLUDE_ROLLOUTS=raw,ROBUSTNESS=True).items():
        code=code.replace(token,repr(value))
    stream=io.StringIO()
    with redirect_stdout(stream):exec(compile(code,'remote-reader-test','exec'),{})
    return json.loads(stream.getvalue())


def test_partial_case_and_policy_weights_are_excluded(tmp_path):
    bundle,run=fixture(tmp_path);response=collect(tmp_path,bundle,run,True)
    assert not any('perturb_00' in path or path.endswith('.pt') for path in response['files'])
    assert run+'/eval_suite/nominal/final_rollout.npz' in response['files']
    compact=collect(tmp_path,bundle,run,False)
    assert not any(path.endswith('.npz') for path in compact['files'])


def test_changed_completed_recording_is_rejected(tmp_path):
    bundle,run=fixture(tmp_path)
    (tmp_path/run/'eval_suite/nominal/final_rollout.npz').write_bytes(b'changed')
    with pytest.raises(AssertionError):collect(tmp_path,bundle,run,True)


def test_unsafe_declared_case_cannot_select_arbitrary_files(tmp_path):
    bundle,run=fixture(tmp_path);path=tmp_path/bundle/'protocol.json';cfg=json.loads(path.read_text())
    cfg['evaluation_cases'][1]['id']='../../outside'
    path.write_text(json.dumps(cfg))
    with pytest.raises(AssertionError):collect(tmp_path,bundle,run,True)
