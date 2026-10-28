"""Counterexamples that an average or any-wheel contact gate would accept."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_hockey_contact_priority import bilateral_gate


def evaluate(touch, unknown=None):
    start=np.arange(5)*.0025
    return bilateral_gate(start,start+.0025,np.array(touch),np.zeros((5,2),bool) if unknown is None else unknown,(0.,.01))


def test_alternating_single_wheel_contact_fails_even_with_100_percent_any_wheel_contact():
    r=evaluate([[1,1],[1,0],[0,1],[1,1],[1,1]])
    assert r["gate_status"] == "fail"
    assert not r["eligible_for_task_comparison"]
    assert r["bilateral_interruption"]["total_s"] == .005
    assert r["both_touch_fraction"] == .5


def test_bilateral_contact_passes_without_any_force_or_load_ratio_input():
    r=evaluate(np.ones((5,2),bool))
    assert r["gate_status"] == "pass"
    assert r["eligible_for_task_comparison"]
    assert not r["full_task_success_claimed"]


def test_one_unknown_wheel_blocks_bilateral_qualification():
    touch=np.ones((5,2),bool);touch[2,1]=False
    unknown=np.zeros_like(touch);unknown[2,1]=True
    r=evaluate(touch,unknown)
    assert r["gate_status"] == "unverified"
    assert r["unknown_only"]["total_s"] == .0025
    assert not r["eligible_for_task_comparison"]


def test_start_and_terminal_contact_are_not_exempted():
    for i in [0,4]:
        touch=np.ones((5,2),bool);touch[i,0]=False
        r=evaluate(touch)
        assert r["gate_status"] == "fail"
        assert r["first_nonbilateral_contact_s"] == i*.0025


def test_missing_sample_is_unverified_instead_of_vacuously_continuous():
    start=np.array([0.,.0025,.0075,.01])
    r=bilateral_gate(start,start+.0025,np.ones((4,2),bool),np.zeros((4,2),bool),(0.,.01))
    assert r["gate_status"] == "unverified"
    assert not r["record_complete"]
