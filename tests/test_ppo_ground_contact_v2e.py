"""CPU-only lifecycle fixtures; no Isaac, renderer, GPU, or training is started.

Runnable with unittest in the local Torch environment even without pytest.
These tests cannot verify actual PhysX force units, direction, or sensor filters.
"""
from pathlib import Path
import struct
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("Torch is unavailable; collector CPU fixtures skipped") from error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ppo_ground_contact_v2e as contact


def make_sensor(body, count):
    return SimpleNamespace(
        body_names=[body],
        cfg=SimpleNamespace(filter_prim_paths_expr=[contact.GROUND], update_period=0., history_length=0),
        _sim_physics_dt=.0025,
        data=SimpleNamespace(force_matrix_w=torch.zeros((count, 1, 1, 3), dtype=torch.float32)))


class Scene:
    def __init__(self, robot, count):
        self.robot = robot
        self.sensors = {name: make_sensor(body, count) for name, body in zip(contact.SENSORS, contact.WHEELS)}
        self.calls = 0

    def __getitem__(self, key):
        if key != "robot": raise KeyError(key)
        return self.robot

    def update(self, dt, **kwargs):
        self.calls += 1


class Env:
    def __init__(self):
        self.device, self.num_envs = "cpu", 2
        self.physics_dt, self.step_dt = .0025, .02
        self.cfg = SimpleNamespace(decimation=8,
            sim=SimpleNamespace(dt=.0025, gravity=(0., 0., -10.)),
            scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type="plane")))
        self.sim = SimpleNamespace(current_time_step_index=0, current_time=0.,
                                   get_physics_dt=lambda: self.physics_dt)
        self._sim_step_counter = 0
        self.mass_reads = 0
        self.masses = torch.tensor([[8., 1., 1.], [16., 2., 2.]], dtype=torch.float64)
        def get_masses():
            self.mass_reads += 1
            return self.masses
        robot = SimpleNamespace(body_names=["base_Link", *contact.WHEELS],
                                root_physx_view=SimpleNamespace(get_masses=get_masses))
        self.scene = Scene(robot, self.num_envs)
        self.normal = torch.tensor([[50., 50.], [100., 100.]], dtype=torch.float32)
        self.force_sequence = [self.normal.clone() for _ in range(8)]
        self.skip_reward = self.read_twice = self.reset_before_reward = False
        self.reset_after_reward = self.read_after_reset = self.extra_refresh = False
        self.scene_dt = .0025
        self.clock_increment = .0025
        self.counter_jump = self.sim_counter_jump = 1
        self.duplicate_dt = False

    def step(self):
        owner = getattr(self, contact.MARKER)
        for values in self.force_sequence:
            for side, name in enumerate(contact.SENSORS):
                matrix = self.scene.sensors[name].data.force_matrix_w
                matrix.zero_()
                matrix[:, 0, 0, 2] = values[:, side]
            self._sim_step_counter += self.counter_jump
            self.sim.current_time_step_index += self.sim_counter_jump
            self.sim.current_time += self.clock_increment
            if self.duplicate_dt:
                self.scene.update(self.scene_dt, dt=self.scene_dt)
            else:
                self.scene.update(self.scene_dt)
            if self.extra_refresh:
                self.scene.update(self.scene_dt)
        if self.reset_before_reward: owner.reset([0])
        result = None if self.skip_reward else owner.reward().clone()
        if self.read_twice:
            torch.testing.assert_close(result, owner.reward(), equal_nan=True)
        if self.reset_after_reward: owner.reset([0])
        if self.read_after_reset: owner.reward()
        return result


class GroundContactV2ETest(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.anchor = torch.ones(2)
        self.speed = torch.ones(2)
        height = ModuleType("height_smoothness")
        height.anchor_xy_tracking = lambda env, name, std: self.anchor
        training = ModuleType("training"); training.__path__ = []
        drift = ModuleType("training.tron1_drift")
        drift.speed_tracking = lambda env, std: self.speed
        self.modules = patch.dict(sys.modules, {"height_smoothness": height,
            "training": training, "training.tron1_drift": drift})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.owners = []
        self.addCleanup(self.close_owners)

    def close_owners(self):
        for owner in reversed(self.owners):
            if not owner.closed:
                owner.close()

    def install(self):
        owner = contact.GroundContactV2E(self.env, allow_cpu_fixture=True)
        self.owners.append(owner)
        return owner

    def test_balanced_single_wheel_and_alternating_support_are_equivalent(self):
        owner = self.install()
        balanced = self.env.step()
        self.env.force_sequence = [torch.tensor([[100., 0.], [200., 0.]])] * 8
        single = self.env.step()
        self.env.force_sequence = [torch.tensor([[100., 0.], [200., 0.]]) if i % 2 == 0
            else torch.tensor([[0., 100.], [0., 200.]]) for i in range(8)]
        alternate = self.env.step()
        for value in (balanced, single, alternate):
            torch.testing.assert_close(value, torch.ones(2))
        self.assertEqual((owner.completed_steps, owner.unique_reads, owner.total_samples), (3, 3, 24))
        self.assertEqual(self.env.mass_reads, 1)
        self.assertEqual(self.env._sim_step_counter, 24)
        self.assertTrue(owner.audit(health=True)["device_finite_normal_force_health"])

    def test_zero_total_support_gap_reduces_reward_without_height_or_bilateral_target(self):
        owner = self.install()
        self.env.force_sequence[3] = torch.zeros((2, 2))
        torch.testing.assert_close(self.env.step(), torch.full((2,), .875))
        self.assertEqual(owner.samples, 8)

    def test_cannot_stack_over_an_a_collector_owner(self):
        for target in (self.env, self.env.scene):
            setattr(target, "_tron1_ground_contact_v2", object())
            with self.assertRaisesRegex(ValueError, "fresh collector"):
                contact.GroundContactV2E(self.env, allow_cpu_fixture=True)
            delattr(target, "_tron1_ground_contact_v2")

    def test_cannot_stack_over_a_b_collector_owner(self):
        for target in (self.env, self.env.scene):
            setattr(target, "_tron1_ground_contact_v2b", object())
            with self.assertRaisesRegex(ValueError, "fresh collector"):
                contact.GroundContactV2E(self.env, allow_cpu_fixture=True)
            delattr(target, "_tron1_ground_contact_v2b")

    def test_cannot_stack_over_a_c_collector_owner(self):
        for target in (self.env, self.env.scene):
            setattr(target, "_tron1_ground_contact_v2c", object())
            with self.assertRaisesRegex(ValueError, "fresh collector"):
                contact.GroundContactV2E(self.env, allow_cpu_fixture=True)
            delattr(target, "_tron1_ground_contact_v2c")

    def test_cannot_stack_over_a_d_collector_owner(self):
        for target in (self.env, self.env.scene):
            setattr(target, "_tron1_ground_contact_v2d", object())
            with self.assertRaisesRegex(ValueError, "fresh collector"):
                contact.GroundContactV2E(self.env, allow_cpu_fixture=True)
            delattr(target, "_tron1_ground_contact_v2d")

    def test_E_audit_binds_all_immutable_definitions(self):
        owner = self.install(); self.env.step()
        report = owner.audit(health=True)
        self.assertEqual(report["immutable_v2a_definition_sha256"], contact.definition.BASE_DEFINITION_SHA256)
        self.assertEqual(report["immutable_v2b_definition_sha256"], contact.definition.BASE_B_DEFINITION_SHA256)
        self.assertEqual(report["immutable_v2c_definition_sha256"], contact.definition.BASE_C_DEFINITION_SHA256)
        self.assertEqual(report["immutable_v2d_definition_sha256"], contact.definition.BASE_D_DEFINITION_SHA256)
        self.assertEqual(report["reward_definition_version"], "ground_contact_v2e_any_wheel")

    def test_repeated_reward_reads_and_duplicate_sensor_refresh_are_not_new_samples(self):
        owner = self.install()
        self.env.read_twice = self.env.extra_refresh = True
        self.env.step()
        self.assertEqual(owner.total_samples, 8)
        self.assertEqual(owner.unique_reads, 1)
        self.assertEqual(self.env.scene.calls, 16)
        self.env.scene.update(.0025)
        self.assertEqual(owner.total_samples, 8)

    def test_autoreset_after_reward_is_allowed_but_postreset_read_is_not(self):
        owner = self.install()
        self.env.reset_after_reward = True
        self.env.step()
        self.assertEqual(owner.reset_calls, 1)
        self.assertEqual(owner.completed_steps, 1)
        self.env.read_after_reset = True
        with self.assertRaisesRegex(ValueError, "after autoreset"):
            self.env.step()

    def test_reset_before_reward_and_missing_reward_are_fatal(self):
        for mode in ("reset_before_reward", "skip_reward"):
            with self.subTest(mode=mode):
                self.env = Env(); owner = self.install(); setattr(self.env, mode, True)
                with self.assertRaises(ValueError): self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                self.assertFalse(owner.inside_step)
                owner.close()

    def test_exactly_eight_not_four_or_nine_substeps(self):
        for count in (4, 9):
            with self.subTest(count=count):
                self.env = Env(); owner = self.install()
                self.env.force_sequence = [self.env.normal.clone()] * count
                with self.assertRaises(ValueError): self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                self.assertFalse(owner.inside_step)
                self.assertFalse(owner.inside_update)
                owner.close()

    def test_bad_clock_and_scene_dt_are_rejected(self):
        variants = [("counter_jump", 2), ("counter_jump", 0), ("sim_counter_jump", 2),
            ("clock_increment", .005), ("clock_increment", .002500001),
            ("scene_dt", .02), ("duplicate_dt", True)]
        for field, value in variants:
            with self.subTest(field=field, value=value):
                self.env = Env(); owner = self.install(); setattr(self.env, field, value)
                with self.assertRaises((ValueError, TypeError)): self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                owner.close()

    def test_realistic_float32_dt_rounding_is_tolerated_on_actual_clock(self):
        owner = self.install()
        self.env.clock_increment = struct.unpack("f", struct.pack("f", .0025))[0]
        torch.testing.assert_close(self.env.step(), torch.ones(2))
        self.assertEqual(owner.completed_steps, 1)

    def test_bad_configuration_fails_before_hook_attachment(self):
        variants = [("physics_dt", .005), ("step_dt", .01)]
        for field, value in variants:
            with self.subTest(field=field):
                self.env = Env(); setattr(self.env, field, value)
                with self.assertRaises(ValueError): self.install()
                self.assertNotIn("step", vars(self.env))
                self.assertNotIn("update", vars(self.env.scene))
                self.assertFalse(hasattr(self.env, contact.MARKER))
        self.env = Env(); self.env.cfg.decimation = 4
        with self.assertRaises(ValueError): self.install()
        self.assertFalse(hasattr(self.env, contact.MARKER))

    def test_constructor_bad_sensor_has_no_persistent_side_effects(self):
        variants = ("shape", "dtype", "filter", "physics_dt", "wheel_name")
        for variant in variants:
            with self.subTest(variant=variant):
                self.env = Env(); sensor = self.env.scene.sensors[contact.SENSORS[1]]
                if variant == "shape": sensor.data.force_matrix_w = torch.zeros((2, 1, 3))
                if variant == "dtype": sensor.data.force_matrix_w = sensor.data.force_matrix_w.double()
                if variant == "filter": sensor.cfg.filter_prim_paths_expr = ["/wrong/collider"]
                if variant == "physics_dt": sensor._sim_physics_dt = .02
                if variant == "wheel_name": sensor.body_names = ["base_Link"]
                with self.assertRaises(ValueError): self.install()
                self.assertFalse(hasattr(self.env, contact.MARKER))
                self.assertFalse(hasattr(self.env.scene, contact.MARKER))
                self.assertNotIn("step", vars(self.env))
                self.assertNotIn("update", vars(self.env.scene))

    def test_constructor_missing_sensor_and_invalid_mass_leave_no_hooks(self):
        for variant in ("missing_sensor", "negative_mass", "zero_mass", "nan_mass", "mass_shape"):
            with self.subTest(variant=variant):
                self.env = Env()
                if variant == "missing_sensor": del self.env.scene.sensors[contact.SENSORS[0]]
                if variant == "negative_mass": self.env.masses[0, 0] = -1.
                if variant == "zero_mass": self.env.masses[0] = 0.
                if variant == "nan_mass": self.env.masses[0, 0] = float("nan")
                if variant == "mass_shape": self.env.masses = self.env.masses[:, :2]
                with self.assertRaises(ValueError): self.install()
                self.assertFalse(hasattr(self.env, contact.MARKER))
                self.assertFalse(hasattr(self.env.scene, contact.MARKER))
                self.assertNotIn("step", vars(self.env))
                self.assertNotIn("update", vars(self.env.scene))

    def test_sensor_replacement_and_runtime_channel_shape_dtype_are_fatal(self):
        for variant in ("identity", "shape", "dtype"):
            with self.subTest(variant=variant):
                self.env = Env(); owner = self.install()
                sensor = self.env.scene.sensors[contact.SENSORS[0]]
                if variant == "identity":
                    self.env.scene.sensors[contact.SENSORS[0]] = make_sensor(contact.WHEELS[0], 2)
                if variant == "shape": sensor.data.force_matrix_w = torch.zeros((2, 1, 2, 3))
                if variant == "dtype": sensor.data.force_matrix_w = sensor.data.force_matrix_w.double()
                with self.assertRaises(ValueError): self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                owner.close()

    def test_invalid_force_poison_and_lifetime_health_cannot_be_reset_away(self):
        for bad in (float("nan"), float("inf"), -.001):
            with self.subTest(bad=bad):
                self.env = Env(); owner = self.install()
                self.env.force_sequence[1][0, 0] = bad
                result = self.env.step()
                self.assertTrue(torch.isnan(result[0]))
                self.assertTrue(torch.isfinite(result[1]))
                self.assertFalse(owner.audit(health=True)["device_finite_normal_force_health"])
                self.assertFalse(owner.audit(health=True)["device_finite_reward_health"])
                owner.reset([0])
                self.env.force_sequence = [self.env.normal.clone() for _ in range(8)]
                torch.testing.assert_close(self.env.step(), torch.ones(2))
                self.assertFalse(owner.audit(health=True)["device_finite_normal_force_health"])
                owner.close()

    def test_tiny_tolerated_negative_force_is_not_fake_load(self):
        owner = self.install()
        self.env.force_sequence = [torch.full((2, 2), -5.e-5)] * 8
        torch.testing.assert_close(self.env.step(), torch.zeros(2))
        self.assertTrue(owner.audit(health=True)["device_finite_normal_force_health"])

    def test_invalid_tracking_quality_poison_is_not_a_healthy_reward(self):
        owner = self.install()
        self.anchor = torch.tensor([float("nan"), 1.])
        result = self.env.step()
        self.assertTrue(torch.isnan(result[0]))
        self.assertTrue(torch.isfinite(result[1]))
        health = owner.audit(health=True)
        self.assertTrue(health["device_finite_normal_force_health"])
        self.assertFalse(health["device_finite_reward_health"])
        owner.reset([0])
        self.anchor = torch.ones(2)
        torch.testing.assert_close(self.env.step(), torch.ones(2))
        self.assertFalse(owner.audit(health=True)["device_finite_reward_health"])

    def test_wrong_tracking_batch_shape_and_bool_dtype_are_rejected(self):
        for values in (torch.ones((2, 3)), torch.ones(2, dtype=torch.bool)):
            with self.subTest(shape=tuple(values.shape), dtype=values.dtype):
                self.env = Env(); owner = self.install(); self.anchor = values
                with self.assertRaises(ValueError): self.env.step()
                self.assertEqual(owner.completed_steps, 0)
                owner.close()

    def test_reward_read_outside_control_window_is_rejected(self):
        owner = self.install()
        with self.assertRaises(ValueError): owner.reward()
        self.env.step()
        with self.assertRaises(ValueError): owner.reward()

    def test_hooks_restore_class_methods_and_enforce_lifo_ownership(self):
        owner = self.install()
        self.assertIn("step", vars(self.env))
        self.assertIn("update", vars(self.env.scene))
        wrapper = self.env.step
        self.env.step = lambda: None
        with self.assertRaises(ValueError): owner.close()
        self.env.step = wrapper
        owner.close()
        self.assertNotIn("step", vars(self.env))
        self.assertNotIn("update", vars(self.env.scene))
        self.assertIs(self.env.step.__func__, Env.step)
        self.assertIs(self.env.scene.update.__func__, Scene.update)
        self.assertFalse(hasattr(self.env, contact.MARKER))
        self.assertFalse(hasattr(self.env.scene, contact.MARKER))
        self.assertTrue(owner.close()["closed"])

    def test_double_owner_is_rejected_without_replacing_first_owner(self):
        owner = self.install()
        before = (self.env.step, self.env.scene.update)
        with self.assertRaises(ValueError): self.install()
        self.assertIs(getattr(self.env, contact.MARKER), owner)
        self.assertEqual((self.env.step, self.env.scene.update), before)
        self.assertEqual(self.env.mass_reads, 1)

    def test_original_instance_method_overrides_are_restored_exactly(self):
        self.env.step = MethodType(lambda env: Env.step(env), self.env)
        self.env.scene.update = MethodType(lambda scene, dt: Scene.update(scene, dt), self.env.scene)
        before_step, before_update = self.env.step, self.env.scene.update
        owner = self.install()
        self.env.step()
        owner.close()
        self.assertIs(self.env.step, before_step)
        self.assertIs(self.env.scene.update, before_update)

    def test_hot_step_avoids_explicit_cpu_numpy_item_and_tensor_bool_reads(self):
        owner = self.install()
        def forbidden(*args, **kwargs):
            raise AssertionError("GPU hot path attempted a host scalar or array read")
        with patch.object(torch.Tensor, "cpu", forbidden), patch.object(torch.Tensor, "numpy", forbidden), \
                patch.object(torch.Tensor, "item", forbidden), patch.object(torch.Tensor, "__bool__", forbidden):
            value = self.env.step()
        torch.testing.assert_close(value, torch.ones(2))
        self.assertEqual(owner.total_samples, 8)


if __name__ == "__main__":
    unittest.main()



