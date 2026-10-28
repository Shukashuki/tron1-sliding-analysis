"""Independent command-coordinate and phase-boundary checks for new actors."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from analyze_hockey_ppo import audit_observations


def fixture():
    n, samples = 139, 1113
    protocol = json.loads((ROOT/"config/hockey_ppo_learning_v1.json").read_text())
    protocol["phase_clock"] = "integer_reference_frames"
    # Facing world +Y while both actual and requested velocities point world +X.
    # The desired velocity in yaw coordinates must be [0,-1].
    q = np.tile([2**-.5, 0, 0, 2**-.5], (samples, 1))
    v = np.tile([1., 0, 0], (samples, 1))
    names = np.array(["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
                      "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"])
    trace = dict(root_quat=q, root_pos=np.zeros((samples, 3)), root_com_vel=v,
        root_omega=np.zeros((samples, 3)), joint_names=names,
        joint_pos=np.zeros((samples, 8)), joint_vel=np.zeros((samples, 8)), actions=np.zeros((n, 8)))
    reference = dict(joint_names=names, joint_pos=np.zeros((n, 8)), joint_vel=np.zeros((n, 8)),
        body_quat_w=q[:n, None], body_pos_w=np.zeros((n, 1, 3)),
        body_lin_vel_w=v[:n, None], body_ang_vel_w=np.zeros((n, 1, 3)))
    obs = np.zeros((n, 56))
    phase = np.arange(n)*2*np.pi/(n-1)
    obs[:, 12], obs[:, 13] = np.sin(phase), np.cos(phase)
    obs[:, 17:23] = [1, 0, 0, 1, 0, 0]  # First two columns of identity, row-major.
    obs[:, 24] = -1  # Body-frame root velocity.
    obs[:, 52] = -1  # Appended desired velocity.
    obs[20:90, 54] = 1
    obs[90:, 55] = 1
    trace["observations"] = obs
    return trace, reference, protocol


class ObservationAuditTests(unittest.TestCase):
    def test_archived_float32_phase_boundary_is_replayed_honestly(self):
        t, r, p = fixture(); p.pop("phase_clock")
        t["observations"][20, 54] = 0
        self.assertTrue(audit_observations(t, r, p)["all_56_fields_reconstructed"])

    def test_yaw_coordinates_and_phase_boundaries(self):
        t, r, p = fixture()
        result = audit_observations(t, r, p)
        self.assertTrue(result["all_56_fields_reconstructed"])
        self.assertLess(result["extra_field_maximum_error"], 1e-12)

    def test_world_velocity_cannot_masquerade_as_yaw_velocity(self):
        t, r, p = fixture(); t["observations"][:, 51:53] = [1, 0]
        with self.assertRaises(ValueError):
            audit_observations(t, r, p)

    def test_one_frame_late_stop_command_rejected(self):
        t, r, p = fixture(); t["observations"][90, 54:56] = [1, 0]
        with self.assertRaises(ValueError):
            audit_observations(t, r, p)


if __name__ == "__main__":
    unittest.main()
