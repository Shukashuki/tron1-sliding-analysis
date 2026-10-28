"""CPU Torch checks of task semantics and physical-window lifecycle."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import hockey_ppo_task as task
from train_hockey_ppo import runner_config


def fixture(frame=20):
    config = json.loads((ROOT/"config/hockey_ppo_learning_v1.json").read_text())
    n, frames, bodies = 2, 139, 3
    q = torch.zeros((frames, bodies, 4)); q[..., 0] = 1
    pos = torch.zeros((frames, bodies, 3)); pos[..., 2] = .6
    vel = torch.zeros_like(pos); vel[..., 0] = 1
    motion = NS(body_pos_w=pos, body_quat_w=q, body_lin_vel_w=vel,
                body_ang_vel_w=torch.zeros_like(pos), joint_pos=torch.zeros((frames, 8)), time_step_total=frames)
    data = NS(root_quat_w=q[0, 0].repeat(n, 1), root_com_lin_vel_w=vel[0, 0].repeat(n, 1),
              root_com_ang_vel_w=torch.zeros((n, 3)), root_pos_w=pos[0, 0].repeat(n, 1),
              joint_pos=torch.zeros((n, 8)))
    robot = NS(data=data)
    command = NS(time_steps=torch.full((n,), frame), motion=motion,
        cfg=NS(body_names=["base_Link", *task.WHEELS]),
        robot_body_quat_w=q[0].repeat(n, 1, 1), robot_body_lin_vel_w=vel[0].repeat(n, 1, 1),
        leg_joint_indices=torch.arange(6), anchor_lin_vel_w=vel[0, 0].repeat(n, 1),
        anchor_ang_vel_w=torch.zeros((n, 3)), robot_anchor_quat_w=data.root_quat_w)
    class Scene(dict):
        pass
    scene = Scene(robot=robot); scene.env_origins = torch.zeros((n, 3))
    return NS(num_envs=n, device="cpu", step_dt=.02, cfg=NS(hockey_protocol=config),
        command_manager=NS(get_term=lambda name: command), scene=scene,
        common_step_counter=1, _sim_step_counter=8,
        hockey_support_window=NS(active=True, samples=8, mg=torch.full((n,), 200.),
                                force=torch.full((n, 8, 2), 100.)))


class RewardTests(unittest.TestCase):
    def test_single_wheel_and_alternating_support(self):
        f = torch.zeros((3, 8, 2)); f[0, :, 0] = 100
        f[1, ::2, 0] = 100; f[1, 1::2, 1] = 100
        f[2] = 1000
        self.assertTrue(torch.equal(task.support_deficit(f, torch.full((3,), 200.), .5), torch.zeros(3)))

    def test_one_missing_substep_is_not_hidden_by_last_contact(self):
        f = torch.full((1, 8, 2), 100.); f[:, 3] = 0
        self.assertAlmostEqual(float(task.support_deficit(f, torch.tensor([200.]), .5)), .5625)
        f[:] = 0
        self.assertEqual(float(task.support_deficit(f, torch.tensor([200.]), .5)), 1.)

    def test_height_and_wheel_spin_do_not_change_imitation(self):
        e = fixture(); before = {k:v.clone() for k,v in task.rewards(e).items()}
        e.scene["robot"].data.root_pos_w[:, 2] += 7
        c = e.command_manager.get_term("motion")
        c.motion.body_pos_w[..., 2] -= 9
        c.robot_body_quat_w[:, 1:] = torch.tensor([.6, 0., .8, 0.])
        e.common_step_counter += 1
        after = task.rewards(e)
        for key in before:
            self.assertTrue(torch.allclose(before[key], after[key], atol=1e-6), key)

    def test_stop_reward_and_beta_at_zero_speed(self):
        e = fixture(frame=100)
        c = e.command_manager.get_term("motion")
        c.motion.body_lin_vel_w[:] = 0
        c.robot_body_lin_vel_w[:] = 0
        e.scene["robot"].data.root_com_lin_vel_w[:] = 0
        values = task.rewards(e)
        self.assertTrue(torch.equal(values["beta"], torch.zeros(2)))
        self.assertTrue(torch.equal(values["stop_hold"], torch.ones(2)))
        self.assertTrue(all(torch.isfinite(v).all() for v in values.values()))

    def test_reward_targets_endpoint(self):
        e = fixture(frame=20)
        c = e.command_manager.get_term("motion")
        c.motion.body_pos_w[20, 0, 0] = 20
        self.assertTrue(torch.equal(task.rewards(e)["xy"], torch.ones(2)))

    def test_loaded_beta_and_free_lift(self):
        e = fixture(); c = e.command_manager.get_term("motion")
        c.robot_body_lin_vel_w[..., 1] = 1
        c.motion.body_lin_vel_w[..., 1] = 1
        e.hockey_support_window.force[..., 1] = 0
        values = task.rewards(e)
        self.assertTrue(torch.equal(values["beta"], torch.full((2,), .5)))
        self.assertTrue(torch.equal(values["support"], torch.zeros(2)))

    def test_explicit_stop_command(self):
        e = fixture(frame=90)
        result = task.desired_command(e)
        self.assertEqual(tuple(result.shape), (2, 5))
        self.assertTrue(torch.equal(result[:, -2:], torch.tensor([[0., 1.], [0., 1.]])))

    def test_integer_braking_boundary_is_not_delayed_by_float32(self):
        e = fixture(frame=20)
        e.cfg.hockey_protocol["phase_clock"] = "integer_reference_frames"
        self.assertTrue(torch.equal(task.desired_command(e)[:, -2:], torch.tensor([[1., 0.], [1., 0.]])))

    def test_pair_uses_identical_hyperparameters(self):
        cfg = fixture().cfg.hockey_protocol
        self.assertEqual(runner_config(cfg)["algorithm"]["learning_rate"], .0003)
        self.assertEqual(cfg["variants"], {"tracking": 0., "support": .4})

    def test_incomplete_physical_window_rejected(self):
        e = fixture(); e.hockey_support_window.samples = 7
        with self.assertRaises(RuntimeError):
            task.rewards(e)


class CollectorTests(unittest.TestCase):
    def test_eight_samples_and_no_force_leak_across_resets(self):
        e = fixture()
        del e.hockey_support_window
        e.scene["robot"].root_physx_view = NS(get_masses=lambda: torch.full((2, 1), 20.))
        e.scene.sensors = {name: NS(data=NS(force_matrix_w=torch.zeros((2, 1, 1, 3)))) for name in task.SENSORS}
        e.scene.update = lambda dt: None
        observed = []
        def step(action):
            for i in range(8):
                e._sim_step_counter += 1
                for sensor in e.scene.sensors.values():
                    sensor.data.force_matrix_w[..., 2] = action+i
                e.scene.update(.0025)
            observed.append(e.hockey_support_window.force.clone())
            e.scene.update(.0025)  # Reset refresh without simulation must be ignored.
            return 123
        e.step = step
        collector = task.SupportWindow(e)
        self.assertEqual(e.step(100), 123)
        e.step(0)
        self.assertEqual(collector.samples, 8)
        self.assertEqual(observed[0][0, 0, 0], 100)
        self.assertEqual(observed[1][0, 0, 0], 0)
        self.assertTrue(torch.equal(observed[1][0, :, 0], torch.arange(8).float()))


if __name__ == "__main__":
    unittest.main()
