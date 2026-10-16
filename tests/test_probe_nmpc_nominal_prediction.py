"""Pure bookkeeping tests; no additional transitions consume the probe budget."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_nmpc_nominal_prediction import SELECTION, contact_history_at, selected_indices


def test_declared_sampling_is_exactly_sixteen_distinct_archived_action_frames():
    indices = selected_indices(349)
    assert indices.tolist() == [0, 19, 43, 56, 72, 78, 87, 100, 125, 145, 150, 154, 155, 156, 175, 225]
    assert len(indices) == len(set(indices)) == len(SELECTION) == 16
    for invalid in (True, 10, 225, 349.):
        with pytest.raises(ValueError):
            selected_indices(invalid)


def test_recorded_history_is_three_physical_post_forward_samples_with_zero_reset_memory():
    forces = np.arange(10*9*3, dtype=float).reshape(10, 9, 3)
    before = forces.copy()
    np.testing.assert_array_equal(contact_history_at(forces, 0), np.zeros((3, 9, 3)))
    np.testing.assert_array_equal(contact_history_at(forces, 1), np.array([np.zeros((9, 3)), np.zeros((9, 3)), forces[1]]))
    np.testing.assert_array_equal(contact_history_at(forces, 2), np.array([np.zeros((9, 3)), forces[1], forces[2]]))
    np.testing.assert_array_equal(contact_history_at(forces, 4), forces[2:5])
    np.testing.assert_array_equal(contact_history_at(forces, 9), forces[7:10])
    np.testing.assert_array_equal(forces, before)


@pytest.mark.parametrize("index", [-1, 10, True, 1.5])
def test_unobserved_or_nonphysical_history_indices_are_rejected(index):
    with pytest.raises(ValueError, match="history/sample"):
        contact_history_at(np.zeros((10, 9, 3)), index)
