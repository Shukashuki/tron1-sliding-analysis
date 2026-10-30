import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_hockey_braking_effectiveness import planar_effects


class PlanarBrakingTests(unittest.TestCase):
    def test_equal_opposite_forces_cancel_without_COM_work(self):
        f=np.array([[[10,0],[-10,0]]],float);v=np.array([[1,0],[1,0]],float)
        e=planar_effects(f,v,2,.01)
        self.assertEqual(e['cancellation_fraction'][0],1)
        self.assertAlmostEqual(e['COM_xy_work_per_wheel_J'].sum(),0)

    def test_force_work_matches_kinetic_energy_and_rotation_is_invariant(self):
        f=np.array([[[-4,1],[-2,-1]]],float);v=np.array([[2,1],[1.97,1]],float)
        e=planar_effects(f,v,2,.01)
        np.testing.assert_allclose(e['COM_work_balance_residual_J'],0,atol=1e-14)
        R=np.array([[0,-1],[1,0]],float)
        r=planar_effects(f@R.T,v@R.T,2,.01)
        for key in ('cancellation_fraction','braking_acceleration_per_wheel_m_s2','COM_xy_work_per_wheel_J'):
            np.testing.assert_allclose(e[key],r[key],atol=1e-14)

    def test_zero_force_ratio_and_stationary_direction_are_undefined(self):
        e=planar_effects(np.zeros((1,2,2)),np.zeros((2,2)),2,.01)
        self.assertTrue(np.isnan(e['cancellation_fraction']).all())
        self.assertTrue(np.isnan(e['braking_acceleration_per_wheel_m_s2']).all())


if __name__=='__main__':unittest.main()
