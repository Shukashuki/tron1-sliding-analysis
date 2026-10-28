"""Physical counterexamples for the boundary objective, independent of Isaac."""
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from hockey_friction_boundary import boundary_objective, boundary_sample, FrictionBoundaryWindow, reward_term


def fixture():
    cfg = json.loads((ROOT/'config/hockey_friction_boundary_v1.json').read_text())['friction_boundary']
    a = torch.ones((1, 8, 2))
    s = dict(normal_fz=a*100, utilization_proxy=a, lateral_slip_rms=a*.2,
        friction_anchor_power=a*-14, opposing_sliding_anchors=a,
        friction_force_w=torch.zeros((1, 8, 2, 3)), root_velocity_xy=torch.ones((1, 8, 2)),
        xy_error_sq=torch.zeros((1, 8)), velocity_error_sq=torch.zeros((1, 8)),
        up=torch.ones((1, 8)), braking_phase=torch.ones((1, 8)))
    s['friction_force_w'][..., 1] = -70
    return s, torch.tensor([200.]), cfg


def score(s, mg, cfg):
    return float(boundary_objective(s, mg, cfg)['reward'][0])


class BoundaryTests(unittest.TestCase):
    def test_saturation_requires_motion_dissipation_and_braking(self):
        for field, value in [('lateral_slip_rms', 0.), ('friction_anchor_power', 1.),
                             ('opposing_sliding_anchors', 0.), ('root_velocity_xy', 0.)]:
            with self.subTest(field=field):
                s, mg, cfg = fixture(); s[field].fill_(value)
                self.assertEqual(score(s, mg, cfg), 0.)

    def test_single_wheel_sufficient_but_one_unloaded_substep_blocks_bonus(self):
        s, mg, cfg = fixture(); s['normal_fz'][..., 1] = 0
        self.assertAlmostEqual(score(s, mg, cfg), 1.)
        s['normal_fz'][:, 3] = 0
        self.assertEqual(score(s, mg, cfg), 0.)

    def test_load_collapse_does_not_keep_full_reward_at_same_rho(self):
        s, mg, cfg = fixture(); original = score(s, mg, cfg)
        s['normal_fz'].fill_(6)
        self.assertLess(score(s, mg, cfg), original*.13)
        s['normal_fz'].zero_()
        self.assertEqual(score(s, mg, cfg), 0.)

    def test_stopped_out_of_phase_fallen_or_off_path_get_no_or_small_bonus(self):
        for field, value in [('braking_phase', 0.), ('up', .7)]:
            s, mg, cfg = fixture(); s[field].fill_(value)
            self.assertEqual(score(s, mg, cfg), 0.)
        s, mg, cfg = fixture(); s['xy_error_sq'].fill_(1.)
        self.assertLess(score(s, mg, cfg), 2e-5)

    def test_reward_bounded_and_no_incentive_for_unbounded_slip_power(self):
        s, mg, cfg = fixture(); original = score(s, mg, cfg)
        s['lateral_slip_rms'] *= 1000; s['friction_anchor_power'] *= 1000
        self.assertEqual(score(s, mg, cfg), original)
        s['utilization_proxy'].fill_(1.06)
        self.assertEqual(score(s, mg, cfg), 0.)

    def test_native_material_point_velocity_and_actual_mu(self):
        # One contact at z=-0.1, wheel rolling in +X: omega_y=+10 cancels v_x=1.
        # Adding +Y translation creates lateral slip, with a dissipative -Y force.
        n = (torch.tensor([[100.]]), torch.tensor([[0., 0., -.1]]),
             torch.tensor([[0., 0., 1.]]), torch.zeros(1), torch.ones((1, 1), dtype=torch.int32),
             torch.zeros((1, 1), dtype=torch.int32))
        f = (torch.tensor([[0., -70., 0.]]), n[1], n[4], n[5])
        def run(vy, mu):
            return boundary_sample(n, f, torch.zeros((1, 3)), torch.tensor([[1., vy, 0.]]),
                torch.tensor([[0., 10., 0.]]), torch.tensor([[1., 0., 0., 0.]]), torch.tensor([mu]))
        rolling = run(0., .7)
        self.assertEqual(float(rolling['lateral_slip_rms']), 0.)
        self.assertEqual(int(rolling['opposing_sliding_anchors']), 0)
        sliding = run(.2, .7)
        self.assertAlmostEqual(float(sliding['utilization_proxy']), 1.)
        self.assertAlmostEqual(float(sliding['lateral_slip_rms']), .2)
        self.assertAlmostEqual(float(sliding['friction_anchor_power']), -14.)
        self.assertEqual(int(sliding['opposing_sliding_anchors']), 1)
        self.assertAlmostEqual(float(run(.2, 1.4)['utilization_proxy']), .5)
        self.assertFalse(bool(sliding['fault'].any()))

    def test_collector_uses_world_dr_and_clears_phase_after_reset_refresh(self):
        from types import SimpleNamespace as NS
        from test_hockey_ppo_task import fixture as env_fixture
        from hockey_ppo_task import SENSORS
        e = env_fixture(frame=20)
        del e.hockey_support_window
        e.cfg.hockey_protocol['friction_boundary'] = fixture()[2]
        e.mjlab_recipe_mu = torch.tensor([.7, 1.2])
        c = e.command_manager.get_term('motion'); robot = e.scene['robot']; d = robot.data
        robot.body_names = c.cfg.body_names
        d.body_com_pos_w = c.motion.body_pos_w[0].repeat(2, 1, 1)
        d.body_com_lin_vel_w = c.robot_body_lin_vel_w.clone(); d.body_com_lin_vel_w[..., 1] = .2
        d.body_com_ang_vel_w = torch.zeros((2, 3, 3)); d.body_quat_w = c.robot_body_quat_w
        d.root_com_lin_vel_w[:, 1] = .2
        robot.root_physx_view = NS(get_masses=lambda: torch.full((2, 1), 20.))
        counts=torch.ones((2, 1), dtype=torch.int32); starts=torch.tensor([[0], [1]],dtype=torch.int32)
        points=torch.tensor([[0., 0., .5], [0., 0., .5]])
        normal=(torch.full((2, 1),100.),points,torch.tensor([[0.,0.,1.],[0.,0.,1.]]),torch.zeros((2,1)),counts,starts)
        friction=(torch.tensor([[0.,-70.,0.],[0.,-70.,0.]]),points,counts,starts)
        view=NS(get_contact_data=lambda dt:normal,get_friction_data=lambda dt:friction)
        force=torch.zeros((2,1,1,3));force[...,2]=100
        e.scene.sensors={n:NS(data=NS(force_matrix_w=force),contact_physx_view=view) for n in SENSORS}
        e.scene.update=lambda dt:None
        rewards=[]
        def step(action):
            e.common_step_counter+=1
            for _ in range(8):
                e._sim_step_counter+=1;e.scene.update(.0025)
            rewards.append(reward_term(e).clone())
            e.scene.update(.0025)  # Nonphysical reset refresh must be ignored.
        e.step=step
        window=FrictionBoundaryWindow(e)
        e.step(None)
        self.assertEqual(window.samples,8)
        self.assertEqual(window.sample_count,8)
        self.assertTrue(torch.allclose(window.values['utilization_proxy'][0],torch.ones((8,2))))
        self.assertTrue(torch.allclose(window.values['utilization_proxy'][1],torch.full((8,2),70/120)))
        self.assertGreater(float(rewards[0][0]),0.)
        c.time_steps.fill_(100);e.step(None)
        self.assertTrue(torch.equal(rewards[1],torch.zeros(2)))
        self.assertEqual(window.sample_count,16)
        self.assertEqual(window.reward_count,4)


if __name__ == '__main__':
    unittest.main()
