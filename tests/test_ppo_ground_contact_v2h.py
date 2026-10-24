"""Real Torch CPU lifecycle fixtures; not PhysX/GPU or policy evidence."""
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ppo_ground_contact_v2h as contact
import test_ppo_ground_contact_v2g as fixture


class RootData:
    def __init__(self, count):
        self._sim_timestamp = 0.
        self._root_com_vel_w = SimpleNamespace(timestamp=-1.)
        self.value = torch.zeros((count, 6), dtype=torch.float32)
        self.refresh_cache = True

    @property
    def root_com_vel_w(self):
        if self.refresh_cache:
            self._root_com_vel_w.timestamp = self._sim_timestamp
        return self.value

    @property
    def root_com_lin_vel_w(self):
        return self.root_com_vel_w[:, :3]


def add_velocity_channel(env):
    data = RootData(env.num_envs)
    env.scene.robot.data = data
    env.vz_sequence = torch.zeros((env.num_envs, 8))
    env.cache_increment = .0025
    original_update = env.scene.update
    def update(scene, dt, **kwargs):
        original_update(dt, **kwargs)
        data._sim_timestamp += env.cache_increment
        if env._sim_step_counter:
            data.value[:, 2] = env.vz_sequence[:, (env._sim_step_counter-1) % 8]
    env.scene.update = MethodType(update, env.scene)
    return data


class GroundContactV2HTest(unittest.TestCase):
    setUp = fixture.GroundContactV2GTest.setUp
    close_owners = fixture.GroundContactV2GTest.close_owners

    def install(self):
        if not hasattr(self.env.scene.robot, "data"):
            add_velocity_channel(self.env)
        owner = contact.GroundContactV2H(self.env, allow_cpu_fixture=True)
        self.owners.append(owner)
        return owner

    def expected(self, owner):
        return contact.definition.reward_torch(torch.stack(self.env.force_sequence, dim=1),
            owner.mg, self.anchor, self.speed, self.beta,
            torch.linalg.vector_norm(self.env.command.body_lin_vel_w[:, 0, :2], dim=-1),
            self.env.vz_sequence)["raw_reward"]

    def test_uses_all_eight_actual_samples_not_repeated_endstate(self):
        owner = self.install()
        self.env.vz_sequence[:] = torch.tensor([[.6, 0., -.6, .3, .45, 0., -.45, 0.],
                                               [0., .6, 0., 0., 0., 0., 0., 0.]])
        actual = self.env.step()
        torch.testing.assert_close(actual, self.expected(owner))
        torch.testing.assert_close(owner.com_vz, self.env.vz_sequence)
        self.assertTrue(torch.all(actual < 1.))
        self.assertEqual((owner.total_samples, owner.velocity_samples), (8, 8))
        audit = owner.audit(health=True)
        self.assertEqual(audit["velocity_environment_samples"], 16)
        self.assertEqual(audit["velocity_unique_reward_windows"], 1)
        self.assertTrue(audit["velocity_cache_refresh_health"])
        self.assertTrue(audit["device_finite_com_vz_health"])
        self.assertAlmostEqual(audit["mean_added_velocity_penalty"], float((1.-actual).mean()))

    def test_zero_velocity_matches_immutable_g_for_single_or_alternating_support(self):
        owner = self.install()
        for alternate in (False, True):
            self.env.force_sequence = [torch.tensor([[100., 0.], [200., 0.]])
                if not alternate or i % 2 == 0 else torch.tensor([[0., 100.], [0., 200.]])
                for i in range(8)]
            torch.testing.assert_close(self.env.step(), torch.ones(2))
        self.assertEqual(owner.velocity_samples, 16)

    def test_cost_is_ungated_by_beta_and_only_applies_while_supported(self):
        owner = self.install()
        self.beta.zero_(); self.env.vz_sequence.fill_(.6)
        self.env.force_sequence[2].zero_()
        torch.testing.assert_close(self.env.step(), self.expected(owner))
        torch.testing.assert_close(owner.reward_fault, torch.zeros(2, dtype=torch.bool))
        self.assertAlmostEqual(owner.audit(health=True)["mean_added_velocity_penalty"], .5*7/8)

    def test_duplicate_refreshes_and_repeated_rewards_do_not_duplicate_samples(self):
        owner = self.install()
        self.env.extra_refresh = self.env.read_twice = True
        self.env.vz_sequence.fill_(.45)
        torch.testing.assert_close(self.env.step(), self.expected(owner))
        self.env.scene.update(.0025)
        audit = owner.audit(health=True)
        self.assertEqual((self.env.scene.calls, owner.velocity_samples, owner.total_samples), (17, 8, 8))
        self.assertEqual((owner.unique_reads, owner.beta_unique_reads, owner.velocity_unique_reads), (1, 1, 1))
        # Cache epoch includes extra refreshes: it need not equal native physics time.
        self.assertNotEqual(owner.velocity_data._sim_timestamp, self.env.sim.current_time)
        self.assertTrue(audit["velocity_cache_refresh_health"])
        torch.testing.assert_close(self.env.step(), self.expected(owner))

    def test_stale_cache_and_missing_scene_epoch_advance_are_rejected(self):
        for mode in ("stale", "no_epoch", "wrong_epoch"):
            with self.subTest(mode=mode):
                self.env = fixture.Env(); owner = self.install()
                if mode == "stale": owner.velocity_data.refresh_cache = False
                else: self.env.cache_increment = 0. if mode == "no_epoch" else .005
                with self.assertRaisesRegex(ValueError, "cache is stale|epoch did not refresh"):
                    self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                self.assertFalse(owner.inside_step)
                owner.close()

    def test_constructor_rejects_bad_channel_before_attaching_hooks(self):
        for mode in ("shape", "dtype", "cache", "epoch"):
            with self.subTest(mode=mode):
                self.env = fixture.Env(); data = add_velocity_channel(self.env)
                if mode == "shape": data.value = torch.zeros((2, 2))
                if mode == "dtype": data.value = data.value.double()
                if mode == "cache": data.refresh_cache = False
                if mode == "epoch": data._sim_timestamp = float("nan")
                before = (self.env.step, self.env.scene.update)
                with self.assertRaises(ValueError): self.install()
                self.assertEqual((self.env.step, self.env.scene.update), before)
                self.assertFalse(hasattr(self.env, contact.MARKER))

    def test_runtime_channel_identity_dtype_shape_and_getter_are_guarded(self):
        for mode in ("identity", "shape", "dtype", "getter"):
            with self.subTest(mode=mode):
                self.env = fixture.Env(); owner = self.install()
                data = owner.velocity_data
                if mode == "identity": self.env.scene.robot.data = RootData(2)
                if mode == "shape": data.value = torch.zeros((2, 2))
                if mode == "dtype": data.value = data.value.double()
                if mode == "getter":
                    with patch.object(RootData, "root_com_lin_vel_w", property(lambda data: data.value[:, :3])):
                        with self.assertRaisesRegex(ValueError, "getter changed"): self.env.step()
                else:
                    with self.assertRaises((ValueError, IndexError)): self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                owner.close()

    def test_nonfinite_velocity_poison_and_lifetime_fault_survive_reset(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                self.env = fixture.Env(); owner = self.install()
                self.env.vz_sequence[0, 1] = bad
                # Invalid COM data cannot become healthy just because no wheel is loaded.
                self.env.force_sequence[1].zero_()
                value = self.env.step()
                self.assertTrue(torch.isnan(value[0])); self.assertTrue(torch.isfinite(value[1]))
                owner.reset([0]); self.env.vz_sequence.zero_()
                self.env.step()
                audit = owner.audit(health=True)
                self.assertFalse(audit["device_finite_com_vz_health"])
                self.assertFalse(audit["device_finite_reward_health"])
                owner.close()

    def test_reset_after_reward_normal_reference_advance_and_hook_restore(self):
        owner = self.install()
        original_step, original_update = owner.original_step, owner.original_update
        self.env.reset_after_reward = self.env.advance_after_reward = True
        for _ in range(3): self.env.step()
        self.assertEqual((owner.completed_steps, owner.reset_calls, owner.velocity_samples), (3, 3, 24))
        owner.close()
        self.assertEqual((self.env.step, self.env.scene.update), (original_step, original_update))

    def test_parent_force_step_reset_and_close_methods_are_unmodified(self):
        for name in ("step", "_matrix", "_complete", "_phase", "_attached", "reset", "close"):
            self.assertIs(getattr(contact.GroundContactV2H, name), getattr(contact.parent.GroundContactV2G, name))

    def test_hot_path_has_no_host_tensor_reads_or_file_io(self):
        owner = self.install(); self.env.vz_sequence.fill_(.45)
        def forbidden(*args, **kwargs): raise AssertionError("Host read in H hot path")
        with patch.object(torch.Tensor, "cpu", forbidden), patch.object(torch.Tensor, "numpy", forbidden), \
             patch.object(torch.Tensor, "item", forbidden), patch.object(torch.Tensor, "__bool__", forbidden), \
             patch.object(Path, "read_bytes", forbidden):
            actual = self.env.step()
        torch.testing.assert_close(actual, self.expected(owner))

    def test_audit_identity_preserves_g_and_explicitly_labels_cpu_fixture(self):
        owner = self.install(); self.env.step()
        audit = owner.audit(health=True)
        self.assertEqual(audit["reward_definition_version"], "ground_contact_v2h_any_wheel")
        self.assertEqual(audit["settings"], contact.definition.validate_settings())
        self.assertEqual(audit["immutable_v2g_definition_sha256"], contact.definition.BASE_G_DEFINITION_SHA256)
        self.assertEqual(audit["immutable_v2g_collector_sha256"], contact.PARENT_COLLECTOR_SHA256)
        self.assertTrue(audit["cpu_fixture_only"])
        self.assertFalse(audit["velocity_source_verified"])
        self.assertIsNone(audit["velocity_source_sha256"])
        self.assertFalse(audit["velocity_is_finite_difference"])
        self.assertFalse(audit["velocity_end_state_repeated"])


if __name__ == "__main__": unittest.main()
