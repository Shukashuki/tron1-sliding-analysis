"""Physical counterexamples for force-independent contact eligibility."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_hockey_ground_contact import contact_kinematics


def fixture():
    trace = dict(body_names=np.array(["wheel_L_Link", "wheel_R_Link"]),
                 body_com_pos=np.zeros((2, 2, 3)), body_com_vel=np.zeros((2, 2, 3)),
                 body_omega=np.zeros((2, 2, 3)), body_quat=np.tile([1., 0, 0, 0], (2, 2, 1)),
                 environment_origin_w=np.zeros(3))
    raw = dict(physical_sample_indices=np.array([1]))
    for side in (0, 1):
        for index, value in enumerate((np.array([[[0.]]]), np.array([[[0., 0., -.1]]]),
                                      np.array([[[0., 0., 1.]]]), np.array([[[0.]]]),
                                      np.array([[[1]]]), np.array([[[0]]]))):
            raw[f"w{side}_normal_buffer_{index}"] = value
    return trace, raw


class GroundContactTests(unittest.TestCase):
    def test_zero_force_touch_is_eligible_and_pure_rolling_does_not_slide(self):
        trace, raw = fixture()
        trace["body_com_vel"][1, 0] = [1., 0., 0.]
        trace["body_omega"][1, 0] = [0., 10., 0.]
        result = contact_kinematics(trace, raw, .0001)
        self.assertTrue(result["touching"][1, 0])
        self.assertAlmostEqual(result["tangent_rms_m_s"][1, 0], 0.)
        trace["body_com_vel"][1, 0, 1] = .2
        result = contact_kinematics(trace, raw, .0001)
        self.assertAlmostEqual(result["tangent_rms_m_s"][1, 0], .2)
        self.assertAlmostEqual(result["lateral_rms_m_s"][1, 0], .2)

    def test_positive_distance_manifold_and_force_do_not_prove_touching(self):
        trace, raw = fixture()
        raw["w0_normal_buffer_3"][:] = .003
        raw["w0_normal_buffer_0"][:] = 100.
        result = contact_kinematics(trace, raw, .0001)
        self.assertEqual(result["contact_point_count"][1, 0], 1)
        self.assertFalse(result["touching"][1, 0])
        self.assertTrue(result["touching"][1, 1])
        self.assertTrue(np.isnan(result["tangent_rms_m_s"][1, 0]))

    def test_absent_contact_is_not_zero_slip_or_zero_distance(self):
        trace, raw = fixture()
        raw["w1_normal_buffer_4"][:] = 0
        result = contact_kinematics(trace, raw, .0001)
        self.assertFalse(result["touching"][1, 1])
        self.assertTrue(np.isnan(result["min_separation_m"][1, 1]))
        self.assertTrue(np.isnan(result["lateral_rms_m_s"][1, 1]))

    def test_invalid_separation_cannot_silently_become_no_contact(self):
        trace, raw = fixture()
        raw["w0_normal_buffer_3"][:] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite native separation"):
            contact_kinematics(trace, raw, .0001)

    def test_missing_contact_normal_is_unknown_not_a_zero_distance_touch(self):
        trace, raw = fixture()
        raw["w0_normal_buffer_2"][:] = 0
        result = contact_kinematics(trace, raw, .0001)
        self.assertFalse(result["touching"][1, 0])
        self.assertTrue(result["contact_unknown"][1, 0])
        self.assertTrue(np.isnan(result["min_separation_m"][1, 0]))
        self.assertEqual(result["velocity_point_count"][1, 0], 0)
        self.assertTrue(np.isnan(result["tangent_rms_m_s"][1, 0]))


if __name__ == "__main__":
    unittest.main()
