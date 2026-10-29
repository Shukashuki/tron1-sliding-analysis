"""Counterexamples for the observed endpoint and saturated-penalty failures."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hockey_contact_refinement_math import bilateral_substep_stop,angular_pseudo_huber,bilateral_contact_loss


class ContactRefinementTests(unittest.TestCase):
    def test_contact_loss_penalizes_brief_native_or_post_gap_loss_with_same_endpoints(self):
        touching=torch.ones((9,8,2),dtype=torch.bool)
        for missed in range(9):touching[missed,:missed,1]=False
        gap=torch.zeros_like(touching,dtype=torch.float32)
        cost=bilateral_contact_loss(touching,gap)
        self.assertEqual(float(cost[0]),0.);self.assertEqual(float(cost[-1]),1.)
        self.assertEqual(float(cost[1]),.5625)
        self.assertTrue(bool((cost[1:]>cost[:-1]).all()))
        post_gap=gap.clone();post_gap[~touching]=.0002
        torch.testing.assert_close(bilateral_contact_loss(torch.ones_like(touching),post_gap),cost,rtol=0,atol=0)
        with self.assertRaises(ValueError):bilateral_contact_loss(touching[:,:7],gap[:,:7])

    def inputs(self):
        return [torch.ones((1,8,2),dtype=torch.bool),torch.zeros((1,8,2)),
                torch.zeros((1,8,2)),torch.zeros((1,8,2)),torch.ones((1,8)),
                torch.ones((1,8)),torch.ones(1)]

    def test_continuous_stop_and_inactive_phase(self):
        values=self.inputs();self.assertEqual(float(bilateral_substep_stop(*values)),1.)
        values[-1].zero_();self.assertEqual(float(bilateral_substep_stop(*values)),0.)

    def test_single_wheel_contact_cannot_earn_a_bilateral_stop(self):
        values=self.inputs();values[0][:,:,1]=False
        self.assertEqual(float(bilateral_substep_stop(*values)),0.)

    def test_one_interior_contact_loss_keeps_partial_feedback_but_loses_worst_step_credit(self):
        values=self.inputs();values[0][:,3,1]=False
        self.assertAlmostEqual(float(bilateral_substep_stop(*values)),.4375)
        values=self.inputs();values[1][:,3,1]=.0002
        self.assertAlmostEqual(float(bilateral_substep_stop(*values)),.4375)

    def test_interior_speed_spike_is_not_hidden_by_stopped_endpoints(self):
        values=self.inputs();values[2][:,3,0]=.2;values[3][:,2,0]=.2
        self.assertLess(float(bilateral_substep_stop(*values)),.4)
        self.assertGreater(float(bilateral_substep_stop(*values)),0.)

    def test_acceleration_above_old_cap_remains_distinguishable(self):
        acceleration=torch.tensor([0.,5.,50.,100.,750.])
        result=angular_pseudo_huber(acceleration[:,None].square().expand(-1,8))
        self.assertTrue(bool((result[1:]>result[:-1]).all()))
        self.assertAlmostEqual(float(result[1]),(5/50)**2,delta=3e-5)
        self.assertGreater(float(result[-1]),20.)

    def test_partial_windows_are_rejected(self):
        values=self.inputs();values[0]=values[0][:,:7]
        with self.assertRaises(ValueError):bilateral_substep_stop(*values)
        with self.assertRaises(ValueError):angular_pseudo_huber(torch.ones((1,7)))

    def test_xyz_stop_detects_vertical_motion_hidden_from_planar_stop(self):
        planar=self.inputs();self.assertEqual(float(bilateral_substep_stop(*planar)),1.)
        values=self.inputs()
        for k in (2,3):values[k]=torch.zeros((1,8,3))
        values[2][:,3,2]=.2;values[3][:,2,2]=.2
        self.assertLess(float(bilateral_substep_stop(*values,velocity_dimensions=3)),.4)
        values[-1].zero_()
        self.assertEqual(float(bilateral_substep_stop(*values,velocity_dimensions=3)),0.)
        with self.assertRaises(ValueError):bilateral_substep_stop(*values)

    def test_xyz_with_zero_vertical_speed_matches_planar_score(self):
        values=self.inputs();values[2][:,:,0]=.03;values[3][:,:,1]=.07
        original=bilateral_substep_stop(*values)
        for k in (2,3):values[k]=torch.cat((values[k],torch.zeros((1,8,1))),-1)
        torch.testing.assert_close(bilateral_substep_stop(*values,velocity_dimensions=3),original,rtol=0,atol=0)

    def test_dense_xyz_stop_distinguishes_fast_motion_and_does_not_reward_liftoff(self):
        values=self.inputs()
        for k in (2,3):values[k]=torch.zeros((1,8,3))
        options=dict(velocity_dimensions=3,speed_form='separate_pseudo_huber')
        self.assertEqual(float(bilateral_substep_stop(*values,**options)),1.)
        scores=[]
        for speed in (.2,.5,1.3):
            for k in (2,3):values[k][:,:,2]=speed
            scores.append(float(bilateral_substep_stop(*values,**options)))
        self.assertTrue(all(a>b for a,b in zip(scores,scores[1:])))
        grounded=scores[-1];values[0].zero_()
        self.assertAlmostEqual(float(bilateral_substep_stop(*values,**options)),grounded-1.,places=5)
        values[-1].zero_()
        self.assertEqual(float(bilateral_substep_stop(*values,**options)),0.)

    def test_dense_xyz_stop_detects_one_interior_velocity_spike(self):
        values=self.inputs()
        for k in (2,3):values[k]=torch.zeros((1,8,3))
        options=dict(velocity_dimensions=3,speed_form='separate_pseudo_huber')
        values[2][:,3,2]=1.3
        self.assertLess(float(bilateral_substep_stop(*values,**options)),-10.)
        with self.assertRaises(ValueError):bilateral_substep_stop(*values,velocity_dimensions=3,speed_form='unknown')
        with self.assertRaises(ValueError):bilateral_substep_stop(*self.inputs(),speed_form='separate_pseudo_huber')


if __name__=='__main__':unittest.main()
