"""Temporal and material-point counterexamples, independent of the simulator."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_hockey_contact_timing import material_step_velocity, previous_state_trace
from analyze_hockey_ground_contact import contact_kinematics
from test_hockey_ground_contact import fixture


class TimingTests(unittest.TestCase):
    def test_pre_step_slip_uses_pre_step_twist_not_post_impact_velocity(self):
        trace, raw = fixture()
        trace["body_com_vel"][0,0] = [1.,.2,0.]
        trace["body_omega"][0,0] = [0.,10.,0.]
        trace["body_com_vel"][1,0] = [0.,0.,0.]
        trace["body_omega"][1,0] = [0.,0.,0.]
        old=contact_kinematics(trace,raw,.0001)
        aligned=contact_kinematics(previous_state_trace(trace),raw,.0001)
        self.assertAlmostEqual(old["tangent_rms_m_s"][1,0],0.)
        self.assertAlmostEqual(aligned["tangent_rms_m_s"][1,0],.2)
        np.testing.assert_array_equal(old["touching"],aligned["touching"])

    def test_material_transport_is_not_motion_of_the_geometric_lowest_point(self):
        dt=.0025;theta=10*dt
        point=np.array([[0.,0.,-.1]])
        velocity=material_step_velocity(point,np.zeros(3),np.array([dt,0.,0.]),
            np.eye(3),Rotation.from_rotvec([0.,theta,0.]).as_matrix(),dt)[0]
        self.assertAlmostEqual(velocity[0],1-.1*np.sin(theta)/dt)
        self.assertAlmostEqual(velocity[2],.1*(1-np.cos(theta))/dt)
        self.assertLess(abs(velocity[0]),.001)
        # The cylinder's geometric lowest point translates at 1 m/s, unlike
        # this nearly stationary rolling material point at the beginning.
        self.assertGreater(abs(velocity[0]-1.),.99)


if __name__=="__main__":unittest.main()
