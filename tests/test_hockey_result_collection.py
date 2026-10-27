"""Readiness gates prevent partial/failed/shared-GPU jobs becoming comparisons."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_hockey_conditioning_results import validate_completed_status


def complete():
    return [dict(id=k,status='completed',returncode=0,gpu_before_launch=dict(idle=True)) for k in ('reference_dq','nominal_command')]


def test_requires_both_named_completed_cases():
    validate_completed_status(complete())
    with pytest.raises(ValueError):validate_completed_status(complete()[:1])
    with pytest.raises(ValueError):validate_completed_status(list(reversed(complete())))


@pytest.mark.parametrize('field,value',[('status','running'),('returncode',1),('gpu_before_launch',{'idle':False})])
def test_incomplete_or_unapproved_gpu_case_is_rejected(field,value):
    rows=complete();rows[1][field]=value
    with pytest.raises(ValueError):validate_completed_status(rows)
