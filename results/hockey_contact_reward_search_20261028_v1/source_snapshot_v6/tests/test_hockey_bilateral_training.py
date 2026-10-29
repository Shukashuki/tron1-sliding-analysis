"""Bilateral contact, free load transfer, timing and anti-reward-hacking cases."""
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from hockey_bilateral_task import objective, geometric_sample, cylinder_gap, BilateralWindow


def fixture():
    cfg=json.loads((ROOT/'config/hockey_bilateral_training_v1.json').read_text())
    one=torch.ones((1,8,2));single=torch.ones((1,8))
    s=dict(touching=one,post_gap=one*0,normal_fz=one*100,utilization_proxy=one,
           lateral_slip_rms=one*.2,friction_anchor_power=one*-14,opposing_sliding_anchors=one,
           friction_force_w=torch.zeros((1,8,2,3)),root_velocity_xy=torch.ones((1,8,2)),
           xy_error_sq=single*0,velocity_error_sq=single*0,up=single,braking_phase=single,
           angular_acceleration_sq=single*0)
    s['friction_force_w'][...,1]=-70
    return s,torch.tensor([200.]),cfg


class BilateralTests(unittest.TestCase):
    def test_one_missing_wheel_for_one_substep_cancels_sliding_and_tracking_eligibility(self):
        s,mg,c=fixture();s['touching'][:,3,1]=0
        r=objective(s,mg,c)
        self.assertEqual(float(r['continuous']),0.)
        self.assertEqual(float(r['sliding']),0.)
        self.assertGreater(float(r['contact_loss']),.5)

    def test_unequal_load_and_zero_load_do_not_redefine_contact(self):
        s,mg,c=fixture();s['normal_fz'][...,0]=200;s['normal_fz'][...,1]=0
        r=objective(s,mg,c)
        self.assertEqual(float(r['continuous']),1.)
        self.assertEqual(float(r['contact_loss']),0.)
        self.assertGreater(float(r['sliding']),.9)
        s['normal_fz'].zero_()
        r=objective(s,mg,c)
        self.assertEqual(float(r['continuous']),1.)
        self.assertEqual(float(r['sliding']),0.)

    def test_post_step_liftoff_blocks_reward_before_next_native_sample(self):
        s,mg,c=fixture();s['post_gap'][:,-1,0]=.002
        r=objective(s,mg,c)
        self.assertEqual(float(r['continuous']),0.)
        self.assertGreater(float(r['clearance']),0.)

    def test_bounded_sliding_requires_motion_opposition_braking_and_task(self):
        s,mg,c=fixture();original=float(objective(s,mg,c)['sliding'])
        for key,value in [('lateral_slip_rms',0),('friction_anchor_power',1),('opposing_sliding_anchors',0),('braking_phase',0),('up',.7),('root_velocity_xy',0)]:
            s,mg,c=fixture();s[key].fill_(value)
            self.assertEqual(float(objective(s,mg,c)['sliding']),0.)
        s,mg,c=fixture();s['lateral_slip_rms']*=1000;s['friction_anchor_power']*=1000
        self.assertEqual(float(objective(s,mg,c)['sliding']),original)
        s['xy_error_sq'].fill_(1.)
        self.assertLess(float(objective(s,mg,c)['sliding']),2e-5)

    def test_geometry_uses_material_velocity_even_at_zero_force(self):
        n=(torch.zeros((1,1)),torch.tensor([[0.,0.,-.1]]),torch.tensor([[0.,0.,1.]]),
           torch.zeros((1,1)),torch.ones((1,1),dtype=torch.int32),torch.zeros((1,1),dtype=torch.int32))
        def run(v):return geometric_sample(n,torch.zeros((1,3)),torch.tensor([v]),torch.tensor([[0.,10.,0.]]),torch.tensor([[1.,0.,0.,0.]]),.0001)
        rolling=run([1.,0.,0.]);sliding=run([1.,.2,0.])
        self.assertTrue(bool(rolling['touching']))
        self.assertEqual(float(rolling['lateral_slip_rms']),0.)
        self.assertAlmostEqual(float(sliding['lateral_slip_rms']),.2,places=6)
        n[3].fill_(.002)
        self.assertFalse(bool(run([1.,.2,0.])['touching']))
        n[3].zero_();n[2].zero_()
        self.assertTrue(bool(run([1.,.2,0.])['unknown']))

    def test_finite_width_and_camber_affect_wheel_clearance(self):
        _,_,c=fixture();g=c['geometry']
        pos=torch.tensor([[0.,0.,g['radius_m']]],dtype=torch.float64)
        q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
        self.assertAlmostEqual(float(cylinder_gap(pos,q,g,0)),0.)
        angle=torch.tensor(.2,dtype=torch.float64)
        q[0]=torch.stack((torch.cos(angle/2),torch.sin(angle/2),angle*0,angle*0))
        left=cylinder_gap(pos,q,g,0);right=cylinder_gap(pos,q,g,1)
        self.assertGreater(float(left-right),0.)

    def test_collector_pairs_previous_state_and_ignores_reset_refresh(self):
        from types import SimpleNamespace as NS
        from hockey_ppo_task import SENSORS, WHEELS
        _,_,cfg=fixture();radius=cfg['geometry']['radius_m']
        pos=torch.tensor([[[0.,0.,radius],[0.,0.,radius]]])
        q=torch.tensor([[[1.,0.,0.,0.],[1.,0.,0.,0.]]])
        vel=torch.zeros((1,2,3));vel[...,1]=.2
        d=NS(body_pos_w=pos,body_com_pos_w=pos.clone(),body_com_lin_vel_w=vel,
             body_com_ang_vel_w=torch.zeros_like(vel),body_quat_w=q,
             root_pos_w=torch.tensor([[0.,0.,.5]]),root_quat_w=q[:,0].clone(),
             root_com_lin_vel_w=torch.tensor([[0.,.2,0.]]),root_com_ang_vel_w=torch.zeros((1,3)))
        robot=NS(data=d,body_names=list(WHEELS),root_physx_view=NS(get_masses=lambda:torch.tensor([[20.]])))
        normal=(torch.tensor([[100.]]),torch.zeros((1,3)),torch.tensor([[0.,0.,1.]]),
                torch.zeros((1,1)),torch.ones((1,1),dtype=torch.int32),torch.zeros((1,1),dtype=torch.int32))
        friction=(torch.tensor([[0.,-70.,0.]]),normal[1],normal[4],normal[5])
        view=NS(get_contact_data=lambda dt:normal,get_friction_data=lambda dt:friction)
        force=torch.zeros((1,1,1,3));force[...,2]=100
        class Scene:
            def __getitem__(self,name):return robot
            def update(self,dt):pass
        scene=Scene();scene.env_origins=torch.zeros((1,3))
        scene.sensors={n:NS(data=NS(force_matrix_w=force),contact_physx_view=view) for n in SENSORS}
        motion=NS(time_step_total=139,body_pos_w=torch.zeros((139,1,3)),body_lin_vel_w=torch.zeros((139,1,3)))
        command=NS(time_steps=torch.tensor([20]),motion=motion)
        e=NS(num_envs=1,device='cpu',scene=scene,mjlab_recipe_mu=torch.tensor([.7]),
             cfg=NS(hockey_protocol=dict(bilateral_sliding=cfg,brake_start_s=.4,brake_end_s=1.8)),
             command_manager=NS(get_term=lambda name:command),_sim_step_counter=0)
        def step(action):
            for k in range(8):
                d.body_com_lin_vel_w[...,1]=(k+1)/10
                e._sim_step_counter+=1;scene.update(.0025)
            scene.update(.0025)  # reset refresh is not a physics sample
        e.step=step
        window=BilateralWindow(e);e.step(None)
        self.assertEqual(window.samples,8)
        self.assertEqual(window.physical_samples,8)
        self.assertAlmostEqual(float(window.values['lateral_slip_rms'][0,0,0]),.2,places=6)
        self.assertAlmostEqual(float(window.values['lateral_slip_rms'][0,1,0]),.1,places=6)
        self.assertAlmostEqual(float(window.values['utilization_proxy'][0,0,0]),1.,places=6)
        self.assertEqual(float(window.components()['continuous']),1.)
        command.time_steps.fill_(100);e.step(None)
        self.assertEqual(window.physical_samples,16)
        self.assertEqual(float(window.values['braking_phase'].sum()),0.)
        # The next native API call may reuse an earlier returned buffer. The
        # collector must own its normal-contact copy before reading friction.
        def normal_read(dt):
            normal[3].zero_()
            return normal
        def friction_read(dt):
            normal[3].fill_(.01)
            return friction
        view.get_contact_data=normal_read;view.get_friction_data=friction_read
        e.step(None)
        self.assertEqual(float(window.components()['continuous']),1.)
        self.assertFalse(bool(window.fault.any()))


if __name__=='__main__':unittest.main()
