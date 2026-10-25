from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_ppo_k_leg_ramp as analysis
import collect_ppo_k_leg_ramp as collection
from controller_trace import native_trace

EVIDENCE=ROOT/'results/ppo_k_leg_ramp_training_20261025_v1/interim_evidence_v1'


def test_running_training_and_smoke_evidence_never_admitted_as_completed_nominal():
    with pytest.raises(AssertionError):collection.verify_completed(EVIDENCE)


def test_real_cpu_smoke_targets_and_DC_motor_records_match_and_detect_tampering():
    f=EVIDENCE/'cpu_smoke_execution_v1';raw=analysis.npz(f/'rollout.npz')
    command=analysis.npz(f/'leg_command_trajectory.npz');trace=native_trace(raw,'policy_mujoco')
    ref=analysis.npz(ROOT/'results/ppo_k_nonlinear_tire_20261025_v2/reference.npz')
    args=(raw,trace,'mujoco',ref['joint_pos'][0],ref['joint_names'].tolist())
    result=analysis.command_audit(command,*args)
    assert result['maximum_leg_torque_reconstruction_error_nm']==0.
    command['actual'][100,0]+=.01
    with pytest.raises(AssertionError):analysis.command_audit(command,*args)
