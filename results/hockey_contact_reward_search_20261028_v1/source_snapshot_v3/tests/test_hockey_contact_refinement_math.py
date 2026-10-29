"""Counterexamples for the observed endpoint and saturated-penalty failures."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hockey_contact_refinement_math import bilateral_substep_stop,angular_pseudo_huber


class ContactRefinementTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
