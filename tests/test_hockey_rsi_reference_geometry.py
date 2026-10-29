import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('rsi_geometry',
    Path(__file__).resolve().parents[1] / 'scripts/audit_hockey_rsi_reference_geometry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_terminal_frame_is_not_sampled():
    np.testing.assert_array_equal(module.bin_geometry_probabilities([False, False, True], 3), [0, 0, 0])


def test_fractional_bin_boundary():
    # Three selectable frames, two bins. Only the middle frame is valid;
    # each bin contains half of it over a total width of 1.5 frames.
    np.testing.assert_allclose(module.bin_geometry_probabilities([False, True, False, True], 2), [1/3, 1/3])


def test_uniform_bins_recover_selectable_frame_fraction():
    valid = np.zeros(139, dtype=bool)
    valid[[0, 1, 2, 35]] = True
    np.testing.assert_allclose(module.bin_geometry_probabilities(valid, 14).mean(), 4/138)
