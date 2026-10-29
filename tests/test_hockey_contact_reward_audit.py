"""Audit labels must describe the configured reward, not the old baseline."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_contact_reward_signals import declared_scores


class RewardAuditTests(unittest.TestCase):
    def inputs(self):
        cfg=dict(bilateral_sliding=dict(angular_acceleration_scale_rad_s2=50.,contact_tolerance_m=.0001),
            reward_scales=dict(stop_speed_m_s=.1,stop_tilt_rad=.35),reward_weights=dict(stop_hold=1.))
        args=(np.ones((1,8,2),bool),np.zeros((1,8,2)),np.zeros((9,2)),
              np.ones(9),np.ones(1,bool),np.full((1,8),10000.,dtype=np.float32),np.array([.019]))
        return cfg,args

    def test_exceeding_old_scale_is_saturation_only_for_the_bounded_reward(self):
        cfg,args=self.inputs();old=declared_scores(cfg,*args)
        cfg['reward_refinement']=dict(kind='hockey_contact_refinement_v1',stop_term='unchanged',angular_term='pseudo_huber')
        new=declared_scores(cfg,*args)
        self.assertTrue(old['saturation'].all());self.assertFalse(new['saturation'].any())
        self.assertTrue(new['legacy_scale_exceedance'].all())
        self.assertEqual(old['angular_form'],'bounded_quadratic');self.assertEqual(new['angular_form'],'pseudo_huber')
        self.assertGreater(float(new['angular'][0]),float(old['angular'][0]))
        np.testing.assert_array_equal(new['stop'],args[-1]);self.assertTrue(new['stop_uses_force_proxy'])

    def test_geometric_stop_does_not_reuse_the_native_force_proxy(self):
        cfg,args=self.inputs();args=list(args);args[-1]=np.zeros(1)
        cfg['reward_refinement']=dict(kind='hockey_contact_refinement_v1',stop_term='bilateral_substeps',angular_term='pseudo_huber')
        result=declared_scores(cfg,*args)
        self.assertFalse(result['stop_uses_force_proxy']);self.assertAlmostEqual(float(result['stop'][0]),.02)
        args[0][:,:,1]=False
        self.assertEqual(float(declared_scores(cfg,*args)['stop'][0]),0.)

    def test_unknown_formula_is_rejected_instead_of_silently_treated_as_legacy(self):
        cfg,args=self.inputs()
        cfg['reward_refinement']=dict(kind='hockey_contact_refinement_v1',stop_term='unchanged',angular_term='unknown')
        with self.assertRaises(ValueError):declared_scores(cfg,*args)


if __name__=='__main__':unittest.main()
