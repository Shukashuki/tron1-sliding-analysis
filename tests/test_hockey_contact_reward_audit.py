"""Audit labels must describe the configured reward, not the old baseline."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_contact_reward_signals import declared_scores,frozen_refinement_code


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

    def test_xyz_requires_vertical_velocity_and_scores_it(self):
        cfg,args=self.inputs();args=list(args)
        cfg['reward_refinement']=dict(kind='hockey_contact_refinement_v1',stop_term='bilateral_substeps_xyz',angular_term='pseudo_huber')
        with self.assertRaises(ValueError):declared_scores(cfg,*args)
        args[2]=np.column_stack((args[2],np.full(9,.2)))
        result=declared_scores(cfg,*args)
        self.assertFalse(result['stop_uses_force_proxy'])
        self.assertLess(float(result['stop'][0]),.001)

    def test_frozen_v3_replay_remains_isolated_from_new_xyz_extension(self):
        import hockey_contact_refinement_math as current
        root=Path(__file__).resolve().parents[1]
        code=frozen_refinement_code(root/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v3')
        self.assertIs(sys.modules['hockey_contact_refinement_math'],current)
        cfg,args=self.inputs();args=list(args)
        cfg['reward_refinement']=dict(kind='hockey_contact_refinement_v1',stop_term='bilateral_substeps',angular_term='pseudo_huber')
        args[2]=np.column_stack((args[2],np.full(9,1.5)))
        result=declared_scores(cfg,*args,reward_code=code)
        self.assertAlmostEqual(float(result['stop'][0]),.02)
        cfg['reward_refinement']['stop_term']='bilateral_substeps_xyz'
        with self.assertRaises(ValueError):declared_scores(cfg,*args,reward_code=code)

    def test_loading_frozen_code_does_not_write_bytecode_into_snapshot(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v3'
        with tempfile.TemporaryDirectory() as temporary:
            bundle=Path(temporary);rows=[]
            for name in ('hockey_contact_refinement.py','hockey_contact_refinement_math.py'):
                data=(source/name).read_bytes();(bundle/name).write_bytes(data)
                rows.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
            (bundle/'manifest.json').write_text(json.dumps(dict(files=rows)))
            with mock.patch.object(sys,'dont_write_bytecode',False):
                frozen_refinement_code(bundle)
                self.assertFalse(sys.dont_write_bytecode)
            self.assertFalse((bundle/'__pycache__').exists())


if __name__=='__main__':unittest.main()
