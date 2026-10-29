"""Rigid wheel geometry checks; these never claim contact-force feasibility."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_reference_contact_gap import cylinder_gap, root_roll_feasibility, rotate_axis, wheel_geometry


class ReferenceGapTests(unittest.TestCase):
    def setUp(self):
        self.g=dict(radius_m=.127,half_width_m=.025,floor_z_m=0.,center_y_in_link_m=[.00707,-.00707])
        self.root=np.array([0.,0.,.75])
        self.center=np.array([[0.,.2,.137],[0.,-.2,.127]])
        self.axes=np.array([[0.,1.,0.],[0.,1.,0.]])
        self.forward=np.array([1.,0.,0.])

    def test_two_horizontal_wheels_height_difference_has_known_roll_solution(self):
        r=root_roll_feasibility(self.root,self.forward,self.center,self.axes,self.g)
        self.assertTrue(r['geometrically_feasible'])
        self.assertAlmostEqual(r['root_roll_delta_deg'],-np.degrees(np.arctan2(.01,.4)),places=8)
        self.assertLess(max(abs(v) for v in r['gap_residual_m']),1e-10)
        np.testing.assert_allclose(rotate_axis(self.forward,self.forward,np.radians(r['root_roll_delta_deg'])),self.forward,atol=1e-12)

    def test_geometry_solution_is_invariant_to_world_heading(self):
        a=.8;z=np.array([0.,0.,1.])
        original=root_roll_feasibility(self.root,self.forward,self.center,self.axes,self.g)
        moved=root_roll_feasibility(rotate_axis(self.root,z,a),rotate_axis(self.forward,z,a),
            rotate_axis(self.center,z,a),rotate_axis(self.axes,z,a),self.g)
        self.assertAlmostEqual(original['root_roll_delta_deg'],moved['root_roll_delta_deg'],places=9)
        self.assertAlmostEqual(original['root_z_delta_m'],moved['root_z_delta_m'],places=9)

    def test_no_root_within_bound_is_not_reported_as_feasible(self):
        center=np.array([[.2,0.,.147],[-.2,0.,.127]])
        r=root_roll_feasibility(self.root,self.forward,center,self.axes,self.g)
        self.assertFalse(r['geometrically_feasible'])

    def test_finite_width_affects_tilted_wheel_clearance(self):
        a=.25
        axis=rotate_axis(self.axes,self.forward,a)
        expected=self.center[:,2]-.127*np.cos(a)-.025*np.sin(a)
        np.testing.assert_allclose(cylinder_gap(self.center,axis,self.g),expected,atol=1e-12)

    def test_wheel_axis_selection_preserves_time_dimension(self):
        p=np.zeros((5,3,3));p[:,1:,2]=.127
        q=np.zeros((5,3,4));q[...,0]=1
        d=dict(body_names=np.array(['base_Link','wheel_L_Link','wheel_R_Link']),pos=p,quat=q)
        center,axis,gap=wheel_geometry(d,'pos','quat',self.g)
        self.assertEqual(center.shape,(5,2,3));self.assertEqual(axis.shape,(5,2,3))
        np.testing.assert_allclose(gap,0.,atol=1e-12)


if __name__=='__main__':unittest.main()
