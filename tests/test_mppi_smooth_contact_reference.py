"""Causal decoder tests; real native physics is separately parity tested."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_mppi_smooth_contact_reference as s


def test_horizon_filter_equals_step_by_step_executed_filter():
    command=np.random.default_rng(5).normal(size=(1,17,8));initial=np.linspace(-.3,.3,8);alpha=np.r_[np.full(6,.25),[.5,.5]]
    planned=s.filtered_controls(command,initial,alpha);previous=initial.copy();executed=[]
    for step in range(17):
        previous=s.filtered_controls(command[:,step:step+1],previous,alpha)[0,0];executed.append(previous.copy())
    np.testing.assert_array_equal(planned[0],executed)
    assert np.max(abs(planned))<=1


def test_filter_preserves_independent_left_right_commands():
    command=np.zeros((1,2,8));command[:,:,0]=.4;command[:,:,1]=-.2
    out=s.filtered_controls(command,np.zeros(8),np.ones(8)*.25)
    np.testing.assert_allclose(out[0,0,:2],[.1,-.05]);np.testing.assert_allclose(out[0,1,:2],[.175,-.0875])


def test_materialized_parent_is_pinned_and_both_prediction_and_plant_are_filtered():
    source=Path(s.base.__file__).read_text();modified=s.materialize(source)
    assert modified.count('filtered_controls(')==2
    assert "base_generator_source_sha256='"+s.PARENT_SHA in modified
    compile(modified,'<test>','exec')
    with pytest.raises(ValueError):s.materialize(source+'\n')
