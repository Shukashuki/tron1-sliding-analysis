"""Protect the distinction between one wheel, neither wheel and unknown data."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_hockey_contact_screen_failures import contact_states, summarize_states, terminal_stop_breakdown


class ContactFailureStatesTests(unittest.TestCase):
    def test_unknown_normal_cannot_be_counted_as_confirmed_airborne(self):
        touching=np.array([[1,1],[1,0],[0,1],[0,0],[0,0]],bool)
        unknown=np.zeros_like(touching);unknown[-1,1]=True
        np.testing.assert_array_equal(contact_states(touching,unknown),[0,1,2,3,4])
        with self.assertRaisesRegex(ValueError,'unknown'):
            contact_states(np.ones((1,2),bool),np.array([[1,0]],bool))

    def test_intervals_are_partitioned_once_and_terminal_is_checked_separately(self):
        dt=.0025;start=np.arange(5)*dt;end=start+dt
        touch=np.array([[1,1],[1,0],[0,1],[0,0],[1,0]],bool)
        r=summarize_states(start,end,touch,np.zeros_like(touch),.01)
        for name in ('both','left_only','right_only','neither'):
            self.assertAlmostEqual(r['states'][name]['duration_s'],dt)
        self.assertEqual(r['states']['unknown']['duration_s'],0)
        self.assertEqual(r['terminal_state'],'left_only')
        self.assertEqual(r['terminal_touching'],[True,False])
        self.assertAlmostEqual(r['states']['neither']['events'][0]['duration_s'],dt)

    def test_missing_interval_or_terminal_cannot_produce_a_complete_summary(self):
        start=np.arange(5)*.0025;end=start+.0025;touch=np.ones((5,2),bool);unknown=np.zeros_like(touch)
        for ids in (np.array([0,1,3,4]),np.arange(4)):
            with self.assertRaises(ValueError):
                summarize_states(start[ids],end[ids],touch[ids],unknown[ids],.01)

    def test_terminal_endpoint_interruption_breaks_hold_without_double_counting_causes(self):
        time=np.arange(201)*.0025;touch=np.ones((201,2),bool);unknown=np.zeros_like(touch)
        speed=np.zeros(201);tilt=np.zeros(201)
        touch[-1,0]=False;speed[-1]=.2
        r=terminal_stop_breakdown(time,touch,unknown,speed,tilt,.5)
        self.assertAlmostEqual(r['hold_s'],.4975)
        self.assertEqual(r['state_samples'],201)
        self.assertEqual(r['violations']['bilateral_contact_missing']['samples'],1)
        self.assertEqual(r['violations']['speed_above_0p1_m_s']['samples'],1)
        self.assertFalse(r['terminal']['stop_state'])
        self.assertTrue(r['cause_sample_counts_are_not_additive'])

    def test_slowdown_can_limit_hold_while_both_wheels_remain_touching(self):
        time=np.arange(201)*.0025;touch=np.ones((201,2),bool)
        speed=np.zeros(201);speed[:22]=.11
        r=terminal_stop_breakdown(time,touch,np.zeros_like(touch),speed,np.zeros(201),.5)
        self.assertAlmostEqual(r['hold_s'],.445)
        self.assertEqual(r['violations']['bilateral_contact_missing']['samples'],0)
        self.assertEqual(r['violations']['speed_above_0p1_m_s']['samples'],22)
        self.assertTrue(r['terminal']['stop_state'])


if __name__=='__main__':unittest.main()
