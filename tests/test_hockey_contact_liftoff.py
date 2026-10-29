from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_hockey_contact_liftoff import loss_onsets, vertical_windows


def test_substep_oscillation_can_cancel_at_control_rate():
    physical, control = vertical_windows(np.array([0., 1., 0., -1., 0., 1., 0., -1., 0.]), .0025)
    np.testing.assert_array_equal(control, [0.])
    assert np.sqrt(np.mean(physical**2)) == 400.


def test_contact_loss_is_first_missing_state_including_terminal():
    np.testing.assert_array_equal(loss_onsets([True, False, False, True, False]), [1, 4])


def test_incomplete_window_is_rejected():
    with pytest.raises(ValueError, match='Incomplete'):
        vertical_windows(np.zeros(8), .0025)
