import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_hockey_braking_contacts import persistent_onsets


class SlipTransitionTests(unittest.TestCase):
    def test_sustained_transition_allows_short_uncertain_band(self):
        low=np.zeros(50,bool);low[:12]=True
        slip=np.zeros(50,bool);slip[16:35]=True
        result=persistent_onsets(low,slip,np.ones(50,bool),.0025)
        self.assertEqual(result,[dict(onset_index=16,low_slip_start_index=4,low_slip_end_index=12)])

    def test_contact_loss_invalidates_transition(self):
        low=np.zeros(50,bool);low[:12]=True
        slip=np.zeros(50,bool);slip[16:35]=True
        contact=np.ones(50,bool);contact[14]=False
        self.assertEqual(persistent_onsets(low,slip,contact,.0025),[])

    def test_initial_sliding_and_short_spikes_are_not_transitions(self):
        self.assertEqual(persistent_onsets(np.zeros(30,bool),np.ones(30,bool),np.ones(30,bool),.0025),[])
        low=np.zeros(50,bool);low[:12]=True
        slip=np.zeros(50,bool);slip[16:20]=True
        self.assertEqual(persistent_onsets(low,slip,np.ones(50,bool),.0025),[])

    def test_overlapping_labels_rejected(self):
        with self.assertRaises(ValueError):
            persistent_onsets(np.ones(30,bool),np.ones(30,bool),np.ones(30,bool),.0025)

    def test_same_low_slip_period_cannot_reuse_after_intervening_sliding(self):
        low=np.zeros(50,bool);low[:12]=True
        slip=np.zeros(50,bool);slip[14:23]=True;slip[24:40]=True
        result=persistent_onsets(low,slip,np.ones(50,bool),.0025)
        self.assertEqual([x['onset_index'] for x in result],[14])


if __name__=='__main__':unittest.main()
