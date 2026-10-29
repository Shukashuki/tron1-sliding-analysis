from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_hockey_contact_liftoff import loss_onsets, vertical_windows, control_update_context


def test_substep_oscillation_can_cancel_at_control_rate():
    physical, control = vertical_windows(np.array([0., 1., 0., -1., 0., 1., 0., -1., 0.]), .0025)
    np.testing.assert_array_equal(control, [0.])
    assert np.sqrt(np.mean(physical**2)) == 400.


def test_contact_loss_is_first_missing_state_including_terminal():
    np.testing.assert_array_equal(loss_onsets([True, False, False, True, False]), [1, 4])


def test_incomplete_window_is_rejected():
    with pytest.raises(ValueError, match='Incomplete'):
        vertical_windows(np.zeros(8), .0025)


def test_action_held_through_last_substep_and_new_action_on_next_interval():
    actions=np.repeat(np.array([[0.],[.1],[.3]]),8,axis=1)
    assert not control_update_context(actions,8)['preceding_action_change_available']
    first=control_update_context(actions,np.int64(9))
    json.dumps(first,allow_nan=False)
    last=control_update_context(actions,16)
    next_first=control_update_context(actions,17)
    assert first['control_index']==last['control_index']==1
    assert first['physical_substep_after_control_update']==1
    assert last['physical_substep_after_control_update']==8
    assert next_first['control_index']==2
    np.testing.assert_allclose(first['action_change_RMS'],.1)
    np.testing.assert_allclose(next_first['action_change_RMS'],.2)
    with pytest.raises(ValueError):control_update_context(actions,25)
