"""Post-fall motion and isolated contact must not become controlled sliding."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_hockey_ppo import contact_event_summary


def fixture():
    n = 401; t = np.arange(n)*.0025
    signal = dict(time_s=t, root_tilt_deg=np.zeros(n), native_sample_valid=t>0,
        normal_fz=np.full((n, 2), 100.), wheel_beta_valid=np.ones((n, 2), dtype=bool),
        wheel_axial_beta_deg=np.full((n, 2), 30.), normal_point_lateral_rms=np.full((n, 2), .2),
        aggregate_utilization=np.ones((n, 2)), anchor_power=np.full((n, 2), -1.),
        dissipative_slipping_anchor_count=np.ones((n, 2)))
    return signal, dict(brake_start_s=.4, brake_end_s=.9)


class EventTests(unittest.TestCase):
    def test_no_post_fall_recovery_credit(self):
        s, c = fixture()
        s["root_tilt_deg"][200] = 46  # Briefly falls at .5, then looks upright again.
        first, rows = contact_event_summary(s, c)
        self.assertEqual(first, .5)
        for row in rows:
            event = row["pre_fall_braking_high_beta_saturated_dissipative_slip"]
            self.assertAlmostEqual(event["longest_duration_s"], .0975)
            self.assertTrue(all(v["end_s"] < first for v in event["events"]))

    def test_low_beta_slip_is_still_physical_slip(self):
        s, c = fixture(); s["wheel_axial_beta_deg"][:] = 10
        _, rows = contact_event_summary(s, c)
        for row in rows:
            self.assertAlmostEqual(row["pre_fall_braking_saturated_dissipative_slip"]["longest_duration_s"], .5)
            self.assertEqual(row["pre_fall_braking_high_beta_saturated_dissipative_slip"]["longest_duration_s"], 0.)

    def test_gap_breaks_event_and_force_threshold_is_strict(self):
        s, c = fixture(); s["normal_fz"][240, 0] = 5
        _, rows = contact_event_summary(s, c)
        event = rows[0]["pre_fall_braking_saturated_dissipative_slip"]
        self.assertEqual(len(event["events"]), 2)
        self.assertAlmostEqual(event["total_duration_s"], .4975)
        self.assertAlmostEqual(event["longest_duration_s"], .3)


if __name__ == "__main__":
    unittest.main()
