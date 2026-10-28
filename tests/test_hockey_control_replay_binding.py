"""Bind real historical replay arrays; metadata here is a verifier fixture only."""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_hockey_physx import load,sha,verify_control_replay


def fixture(tmp_path):
    root=Path(__file__).resolve().parents[1]
    replay=root/'results/hockey_stop_mppi_preflight_20261027_v1/control_replay/reference_dq'
    run_dir=root/'results/hockey_stop_conditioning_physx_20261027_v1/evidence/conditioning_probes_v1/reference_dq'
    run=json.loads((run_dir/'report.json').read_text())
    report=json.loads((replay/'report.json').read_text())
    # The admission helper is tested using an existing physical zero-residual
    # trace. This is not a new MPPI execution record or search qualification.
    source_audit={'fixture':'historical zero-residual control replay binding'}
    report['source_audit']=source_audit
    report['plan_sha256']=sha(run_dir/'mean.npz')
    p=tmp_path/'report.json';p.write_text(json.dumps(report))
    shutil.copyfile(replay/'replay.npz',tmp_path/'replay.npz')
    return p,run_dir,run,source_audit


def test_replay_evidence_is_bound_to_exact_trace_actor_and_plan(tmp_path):
    p,run_dir,run,source_audit=fixture(tmp_path)
    verified=verify_control_replay(p,run_dir,run,source_audit)
    assert verified['control_steps_checked']==139
    assert verified['maximum_action_error']<2e-6
    report=json.loads(p.read_text());report['plan_sha256']='0'*64;p.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='does not certify'):
        verify_control_replay(p,run_dir,run,source_audit)


def test_replay_arrays_cannot_hide_an_action_change_behind_a_low_reported_error(tmp_path):
    import numpy as np
    p,run_dir,run,source_audit=fixture(tmp_path)
    arrays=load(tmp_path/'replay.npz');arrays['residual'][31,2]+=.02
    np.savez_compressed(tmp_path/'replay.npz',**arrays)
    report=json.loads(p.read_text());report['replay_sha256']=sha(tmp_path/'replay.npz')
    report['maximum_action_error']=0.;p.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='disagree'):
        verify_control_replay(p,run_dir,run,source_audit)
