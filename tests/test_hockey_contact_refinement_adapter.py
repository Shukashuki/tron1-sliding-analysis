"""Guard opt-in reward changes and post-step collection against reset leakage."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hockey_contact_search import DenseContactWindow,objective
from hockey_contact_refinement import RefinedContactWindow,configure,refined_components,validate_settings
from test_hockey_bilateral_training import fixture


def protocol(stop='bilateral_substeps',angular='pseudo_huber'):
    return dict(reward_refinement=dict(kind='hockey_contact_refinement_v1',stop_term=stop,angular_term=angular),
                reward_scales=dict(stop_speed_m_s=.1,stop_tilt_rad=.35),phase_clock='integer_reference_frames')


class ContactRefinementAdapterTests(unittest.TestCase):
    def test_other_contact_and_sliding_terms_remain_unchanged(self):
        samples,mg,cfg=fixture();samples['root_velocity_xy_post']=samples['root_velocity_xy'].clone()
        samples['up_post']=samples['up'].clone();samples['angular_acceleration_sq'].fill_(10000.)
        original=objective(samples,mg,cfg)
        refined=refined_components(samples,mg,cfg,protocol(),torch.ones(len(mg),dtype=torch.bool))
        for key in ('contact_fraction','contact_loss','clearance','sliding'):
            torch.testing.assert_close(refined[key],original[key])
        self.assertGreater(float(refined['angular_acceleration']),float(original['angular_acceleration']))

    def test_disabled_math_is_identical_to_original_components(self):
        samples,mg,cfg=fixture()
        result=refined_components(samples,mg,cfg,protocol('unchanged','bounded_quadratic'),torch.ones(len(mg)))
        original=objective(samples,mg,cfg);self.assertEqual(set(result),set(original))
        for key in result:torch.testing.assert_close(result[key],original[key],rtol=0,atol=0)

    def test_configure_preserves_weights_and_unrelated_reward(self):
        original=lambda env:None
        stop=SimpleNamespace(func=original,params={'key':'stop_hold'},weight=1.5)
        xy=SimpleNamespace(func=original,params={'key':'xy'},weight=2.)
        cfg=SimpleNamespace(rewards=SimpleNamespace(hockey_stop_hold=stop,hockey_xy=xy))
        configure(cfg,protocol('unchanged','pseudo_huber'))
        self.assertIs(stop.func,original)
        configure(cfg,protocol())
        self.assertIs(xy.func,original);self.assertEqual(xy.params,{'key':'xy'});self.assertEqual(xy.weight,2.)
        self.assertEqual(stop.weight,1.5);self.assertEqual(stop.params,{})

    def window(self):
        w=object.__new__(RefinedContactWindow);w.env=SimpleNamespace(num_envs=1);w.samples=2
        w.values={'root_velocity_xy_post':torch.zeros((1,8,2)),'up_post':torch.zeros((1,8))}
        w.fault=torch.zeros(1,dtype=torch.bool);w.fault_reasons={'nonfinite':torch.zeros(1,dtype=torch.bool)}
        w.before={'root_com_lin_vel_w':torch.tensor([[.1,.2,.3]]),'root_quat_w':torch.tensor([[1.,0.,0.,0.]])}
        return w

    def test_reset_refresh_does_not_count_as_physics_and_new_step_uses_post_state(self):
        w=self.window()
        with mock.patch.object(DenseContactWindow,'update',lambda self:None):w.update()
        self.assertEqual(float(w.values['up_post'].sum()),0.)
        def advance(owner):owner.samples+=1
        with mock.patch.object(DenseContactWindow,'update',advance):w.update()
        torch.testing.assert_close(w.values['root_velocity_xy_post'][0,2],torch.tensor([.1,.2]))
        self.assertEqual(float(w.values['up_post'][0,2]),1.)
        self.assertEqual(float(w.values['up_post'][:,:2].sum()),0.)
        w.before['root_com_lin_vel_w'][0,0]=float('nan')
        with mock.patch.object(DenseContactWindow,'update',advance):w.update()
        self.assertTrue(bool(w.fault[0]));self.assertTrue(bool(w.fault_reasons['nonfinite'][0]))

    def test_incomplete_physical_window_cannot_be_scored(self):
        w=self.window();w.active=True;w.samples=7;w.cache=None
        with self.assertRaisesRegex(RuntimeError,'eight'):w.components()

    def test_settings_reject_implicit_or_misspelled_refinements(self):
        with self.assertRaises(ValueError):validate_settings({'stop_term':'bilateral_substeps'})
        with self.assertRaises(ValueError):validate_settings(protocol(angular='pseudohuber')['reward_refinement'])

    def test_xyz_changes_stop_only_and_collects_distinct_pre_post_vertical_velocity(self):
        samples,mg,cfg=fixture();samples['root_velocity_xy'].zero_()
        samples['root_velocity_xy_post']=samples['root_velocity_xy'].clone()
        samples['up_post']=samples['up'].clone()
        for key in ('root_velocity_xyz_pre','root_velocity_xyz_post'):
            samples[key]=torch.cat((samples['root_velocity_xy'],torch.full((1,8,1),.2)),-1)
        phase=torch.ones(len(mg),dtype=torch.bool)
        planar=refined_components(samples,mg,cfg,protocol(),phase)
        xyz=refined_components(samples,mg,cfg,protocol('bilateral_substeps_xyz'),phase)
        self.assertLess(float(xyz['stop_hold_refined']),float(planar['stop_hold_refined']))
        for key in planar:
            if key!='stop_hold_refined':torch.testing.assert_close(xyz[key],planar[key],rtol=0,atol=0)
        w=self.window()
        for key in ('root_velocity_xyz_pre','root_velocity_xyz_post'):w.values[key]=torch.zeros((1,8,3))
        with mock.patch.object(DenseContactWindow,'update',lambda owner:None):w.update()
        self.assertEqual(float(w.values['root_velocity_xyz_pre'].sum()),0.)
        def advance(owner):
            owner.samples+=1
            owner.before=dict(owner.before,root_com_lin_vel_w=torch.tensor([[.4,.5,.6]]))
        with mock.patch.object(DenseContactWindow,'update',advance):w.update()
        torch.testing.assert_close(w.values['root_velocity_xyz_pre'][0,2],torch.tensor([.1,.2,.3]))
        torch.testing.assert_close(w.values['root_velocity_xyz_post'][0,2],torch.tensor([.4,.5,.6]))
        w.before['root_com_lin_vel_w'][0,2]=float('nan')
        with mock.patch.object(DenseContactWindow,'update',advance):w.update()
        self.assertTrue(bool(w.fault[0]))

    def test_xyz_reset_before_first_physics_step_has_no_prior_state(self):
        w=self.window();del w.before
        for key in ('root_velocity_xyz_pre','root_velocity_xyz_post'):w.values[key]=torch.zeros((1,8,3))
        with mock.patch.object(DenseContactWindow,'update',lambda owner:None):w.update()
        self.assertEqual(float(w.values['root_velocity_xyz_pre'].sum()),0.)


if __name__=='__main__':unittest.main()
