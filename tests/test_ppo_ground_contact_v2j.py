"""Real Torch CPU fixtures only, never evidence of executed PhysX/training."""
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ppo_ground_contact_v2j as contact
import test_ppo_ground_contact_v2i as fixture


class RootData:
    def __init__(self, count):
        self._sim_timestamp = 0.
        self._root_com_vel_w = SimpleNamespace(timestamp=-1.)
        self.value = torch.zeros((count, 6), dtype=torch.float32)
        self.refresh_cache = True

    @property
    def root_com_vel_w(self):
        if self.refresh_cache: self._root_com_vel_w.timestamp = self._sim_timestamp
        return self.value

    @property
    def root_com_ang_vel_w(self):
        return self.root_com_vel_w[:, 3:6]


def add_angular_channel(env):
    data = RootData(env.num_envs); env.scene.robot.data = data
    env.angular_sequence = torch.zeros((env.num_envs, 8, 2))
    env.cache_increment = .0025
    original = env.scene.update
    def update(scene, dt, **kwargs):
        original(dt, **kwargs)
        data._sim_timestamp += env.cache_increment
        if env._sim_step_counter:
            data.value[:, 3:5] = env.angular_sequence[:, (env._sim_step_counter-1) % 8]
    env.scene.update = MethodType(update, env.scene)
    return data


class GroundContactV2JTest(unittest.TestCase):
    setUp = fixture.GroundContactV2ITest.setUp
    close_owners = fixture.GroundContactV2ITest.close_owners

    def install(self):
        if not hasattr(self.env.scene.robot, "data"): add_angular_channel(self.env)
        owner = contact.GroundContactV2J(self.env, allow_cpu_fixture=True)
        self.owners.append(owner)
        return owner

    def expected(self, owner):
        return contact.definition.reward_torch(torch.stack(self.env.force_sequence, dim=1),
            owner.mg, self.anchor, self.speed, self.beta,
            torch.linalg.vector_norm(self.env.command.body_lin_vel_w[:, 0, :2], dim=-1),
            owner.angular)["raw_reward"]

    def test_actual_start_and_all_eight_samples_not_repeated_end_state(self):
        data = add_angular_channel(self.env); data.value[:, 3:5] = .7
        owner = self.install()
        self.env.angular_sequence[:] = torch.tensor([[[.4, 0.], [0., .2], [-.2, 0.], [.1, 0.],
            [0., -.2], [.2, 0.], [0., 0.], [0., 0.]]]*2)
        actual = self.env.step()
        torch.testing.assert_close(actual, self.expected(owner))
        torch.testing.assert_close(owner.angular[:, 0], torch.full((2, 2), .7))
        torch.testing.assert_close(owner.angular[:, 1:], self.env.angular_sequence)
        self.assertTrue(torch.all(actual < 1.))
        audit = owner.audit(health=True)
        self.assertEqual((audit["angular_boundary_samples"], audit["angular_physics_samples"]), (1, 8))
        self.assertEqual((audit["angular_environment_boundary_samples"], audit["angular_environment_physics_samples"]), (2, 16))
        self.assertTrue(audit["angular_cache_refresh_health"])

    def test_next_window_uses_real_current_state_and_reset_does_not_reuse_previous(self):
        owner = self.install(); self.env.angular_sequence.fill_(.7)
        self.env.step()
        torch.testing.assert_close(self.env.step(), torch.ones(2))
        torch.testing.assert_close(owner.angular[:, 0], torch.full((2, 2), .7))
        owner.reset([0]); owner.angular_data.value[:, 3:5] = 3.
        self.env.angular_sequence.fill_(3.)
        torch.testing.assert_close(self.env.step(), torch.ones(2))
        self.assertEqual(owner.angular_starts, 3)

    def test_single_or_alternating_support_allowed_with_zero_increment(self):
        owner = self.install()
        for alternate in (False, True):
            self.env.force_sequence = [torch.tensor([[100., 0.], [200., 0.]]) if not alternate or i%2 == 0
                else torch.tensor([[0., 100.], [0., 200.]]) for i in range(8)]
            torch.testing.assert_close(self.env.step(), torch.ones(2))
        self.assertEqual(owner.angular_samples, 16)

    def test_beta_does_not_gate_cost_and_unloaded_cost_cannot_hide_gap(self):
        owner = self.install(); self.beta.zero_()
        self.env.angular_sequence[:] = torch.arange(1, 9)[None, :, None]*100.
        torch.testing.assert_close(self.env.step(), torch.full((2,), -.5))
        self.env.force_sequence = [torch.zeros((2, 2)) for _ in range(8)]
        torch.testing.assert_close(self.env.step(), torch.full((2,), -1.))

    def test_duplicate_scene_refresh_and_reward_do_not_duplicate_samples(self):
        owner = self.install(); self.env.extra_refresh = self.env.read_twice = True
        self.env.angular_sequence[:] = torch.arange(8)[None, :, None]*.2
        actual = self.env.step(); torch.testing.assert_close(actual, self.expected(owner))
        self.env.scene.update(.0025)
        audit = owner.audit(health=True)
        self.assertEqual((owner.angular_starts, owner.angular_samples, owner.angular_unique_reads), (1, 8, 1))
        self.assertNotEqual(owner.angular_data._sim_timestamp, self.env.sim.current_time)
        self.assertTrue(audit["angular_cache_refresh_health"])

    def test_stale_cache_or_missing_physics_epoch_rejected(self):
        for mode in ("stale", "no_epoch", "wrong_epoch"):
            with self.subTest(mode=mode):
                self.env = fixture.Env(); owner = self.install()
                if mode == "stale": owner.angular_data.refresh_cache = False
                else: self.env.cache_increment = 0. if mode == "no_epoch" else .005
                with self.assertRaisesRegex(ValueError, "cache stale|epoch did not advance"): self.env.step()
                self.assertEqual(owner.completed_steps, 0); owner.close()

    def test_constructor_bad_channel_before_lifecycle_attachment(self):
        for mode in ("shape", "dtype", "cache", "epoch"):
            with self.subTest(mode=mode):
                self.env = fixture.Env(); data = add_angular_channel(self.env)
                if mode == "shape": data.value = torch.zeros((2, 2))
                if mode == "dtype": data.value = data.value.double()
                if mode == "cache": data.refresh_cache = False
                if mode == "epoch": data._sim_timestamp = float("nan")
                before = self.env.step, self.env.scene.update
                with self.assertRaises(ValueError): self.install()
                self.assertEqual(before, (self.env.step, self.env.scene.update))
                self.assertFalse(hasattr(self.env, contact.MARKER))

    def test_getter_identity_and_shape_cannot_change_at_runtime(self):
        for mode in ("identity", "shape", "dtype", "getter"):
            with self.subTest(mode=mode):
                self.env = fixture.Env(); owner = self.install(); data = owner.angular_data
                if mode == "identity": self.env.scene.robot.data = RootData(2)
                if mode == "shape": data.value = torch.zeros((2, 2))
                if mode == "dtype": data.value = data.value.double()
                if mode == "getter":
                    with patch.object(RootData, "root_com_ang_vel_w", property(lambda data: data.value[:, 3:6])):
                        with self.assertRaisesRegex(ValueError, "getter changed"): self.env.step()
                else:
                    with self.assertRaises(ValueError): self.env.step()
                self.assertEqual(owner.completed_steps, 0); owner.close()

    def test_nonfinite_start_or_substep_poison_even_without_support_and_fault_survives(self):
        for position in ("start", "substep"):
            with self.subTest(position=position):
                self.env = fixture.Env(); owner = self.install()
                if position == "start": owner.angular_data.value[0, 3] = float("nan")
                else: self.env.angular_sequence[0, 2, 0] = float("nan")
                self.env.force_sequence = [torch.zeros((2, 2)) for _ in range(8)]
                value = self.env.step(); self.assertTrue(torch.isnan(value[0])); self.assertTrue(torch.isfinite(value[1]))
                owner.reset([0]); self.env.angular_sequence.zero_(); owner.angular_data.value.zero_(); self.env.step()
                self.assertFalse(owner.audit(health=True)["device_finite_angular_health"])
                owner.close()

    def test_parent_force_reset_close_methods_are_not_rewritten(self):
        for name in ("_matrix", "_complete", "_phase", "_attached", "reset", "close"):
            self.assertIs(getattr(contact.GroundContactV2J, name), getattr(contact.parent.GroundContactV2I, name))

    def test_no_host_reads_or_filesystem_in_reward_hotpath(self):
        owner = self.install(); self.env.angular_sequence.fill_(.4)
        def forbidden(*a, **k): raise AssertionError("No host/file read in J hotpath")
        with patch.object(torch.Tensor, "cpu", forbidden), patch.object(torch.Tensor, "numpy", forbidden), \
             patch.object(torch.Tensor, "item", forbidden), patch.object(torch.Tensor, "__bool__", forbidden), \
             patch.object(Path, "read_bytes", forbidden):
            result = self.env.step()
        torch.testing.assert_close(result, self.expected(owner))
