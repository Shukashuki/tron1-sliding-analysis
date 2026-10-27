"""Check physically known axle/rolling directions and left/right separation."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retarget_hockey_gvhmr_gmr import wheel_headings_from_axles


def test_independent_left_right_headings_and_time_axis():
    headings = np.array([[0, 90], [30, -45], [60, -30], [80, -10]])
    radians = np.radians(headings)
    axles = np.stack((-np.sin(radians), np.cos(radians), np.zeros_like(radians)), axis=-1)
    np.testing.assert_allclose(wheel_headings_from_axles(axles), headings, atol=1e-12)


def test_wheel_spin_does_not_change_rolling_plane_heading():
    yaw = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    spins = np.radians([0, 50, 160, 240])[:, None] * np.array([0, 1, 0])
    orientations = yaw @ Rotation.from_rotvec(spins).as_matrix()
    axles = orientations[..., :, 1]
    axles = np.repeat(axles[:, None], 2, axis=1)
    np.testing.assert_allclose(wheel_headings_from_axles(axles), 35., atol=1e-12)


def test_vertical_axle_does_not_fake_a_valid_heading():
    with pytest.raises(ValueError, match="vertical"):
        wheel_headings_from_axles(np.tile([0., 0., 1.], (3, 2, 1)))
