"""Offline B/C objective accounting; these tests do not execute a controller."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ground_contact_reward_v2b as reward_a
import ground_contact_reward_v2c as reward_b
import nmpc_batch_ground_contact_v2b_400hz as objective_a
import nmpc_batch_ground_contact_v2c_400hz as objective_b
from nmpc_tracking_objective import BODY_NAMES


def fixture():
    """Diverse finite recorded-state inputs, not a simulated trajectory."""
    rng = np.random.default_rng(3819)
    shape, bodies = (3, 5), len(BODY_NAMES)
    native = {}
    for name, width in (("body_pos_w", 3), ("body_quat_w", 4),
                        ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        pre = rng.normal(0., .08, (*shape, bodies, width))
        post = rng.normal(0., .08, (*shape, 8, bodies, width))
        if width == 4:
            pre[..., 0] += 1.; post[..., 0] += 1.
            pre /= np.linalg.norm(pre, axis=-1, keepdims=True)
            post /= np.linalg.norm(post, axis=-1, keepdims=True)
        if name == "body_lin_vel_w":
            pre[..., 0] += 1.; post[..., 0] += 1.
        native["pre_" + name], native[name] = pre, post
    net = rng.normal(0., 2., (*shape, 8, bodies, 3))
    native.update(qpos=rng.uniform(-.3, .3, (*shape, 8, 15)),
        body_net_forces_w=net, contact_force_history_w=net[..., -3:, :, :].copy(),
        interval_time_s=np.broadcast_to(np.arange(1, 9)*.0025, (*shape, 8)).copy(),
        ground_contact_count=np.ones((*shape, 8, 2), dtype=np.int32),
        ground_normal_force_w=np.zeros((*shape, 8, 2, 3)))
    fz = native["ground_normal_force_w"][..., 2]
    fz[..., 0] = 100.
    fz[0, 0] = 50.                         # Balanced, total mg.
    fz[0, 1, ::2, 0] = 0.
    fz[0, 1, ::2, 1] = 100.               # Alternating total mg.
    fz[0, 2, :, 0] = 25.                  # Continuous weak support.
    fz[0, 3, 3] = 0.                      # Exactly one missing sample.
    fz[0, 4, 1:7] = 0.                    # Several missing samples.
    fz[1, 0, 2] = (700., 0.)             # Same impulse gate in A/B.
    fz[1, 1, 1:4] = 0.
    fz[1, 1, 4] = (1000., 0.)
    fz[1, 2] = 0.
    fz[1, 3, :, 0] = np.linspace(10., 60., 8)
    fz[2, :, 4] = 0.
    reference = {"body_names": BODY_NAMES, **{
        name: native["pre_" + name].copy() for name in
        ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}}
    reference["body_pos_w"][..., 0] += .015
    reference["body_lin_vel_w"][..., 1] += .08
    profile = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    profile["action_rate_weight"] = -.5
    return native, reference, profile, rng.uniform(-.7, .7, (*shape, 8))


def evaluate(objective, native, reference, actions):
    return objective.evaluate_native(native, reference, actions, np.full((3, 8), .1),
        np.array([.01, -.02, .03]), joint_qpos_indices=np.arange(7, 15),
        soft_joint_pos_limits=np.tile([-2., 2.], (8, 1)),
        soft_joint_limits_source="isaac_runtime", initial_reset_samples=True)


class ObjectiveV2cTests(unittest.TestCase):
    def assert_bitwise(self, left, right):
        left, right = np.asarray(left), np.asarray(right)
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.tobytes(), right.tobytes())

    def test_only_grounded_changes_and_exact_stage_horizon_delta(self):
        native, reference, profile, actions = fixture()
        before = copy.deepcopy(native)
        for gamma in (.99, 1., .8):
            with self.subTest(gamma=gamma):
                a = evaluate(objective_a.BatchGroundContactTrackingObjective400Hz(
                    profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
                b = evaluate(objective_b.BatchGroundContactTrackingObjective400Hz(
                    profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
                self.assertEqual(len(a["raw_terms"]), 15)
                self.assertEqual(tuple(a["raw_terms"]), tuple(b["raw_terms"]))
                for channel in ("raw_terms", "weighted_terms", "term_costs"):
                    for term in a[channel]:
                        if term != "drift_grounded":
                            self.assert_bitwise(a[channel][term], b[channel][term])
                ca, cb = a["ground_contact_v2"], b["ground_contact_v2"]
                for name in ca:
                    if name != "raw_reward":
                        self.assert_bitwise(ca[name], cb[name])
                expected = .02*.4*(1.-ca["support_mean"])
                np.testing.assert_allclose(b["stage_cost"]-a["stage_cost"], expected, rtol=1e-12, atol=1e-16)
                np.testing.assert_allclose(b["term_costs"]["drift_grounded"]-a["term_costs"]["drift_grounded"], expected, rtol=1e-12, atol=1e-16)
                np.testing.assert_allclose(b["candidate_costs"]-a["candidate_costs"],
                    np.sum(expected*gamma**np.arange(5), axis=1), rtol=1e-12, atol=2e-16)
                np.testing.assert_allclose(b["undiscounted_candidate_costs"]-a["undiscounted_candidate_costs"],
                    expected.sum(-1), rtol=1e-12, atol=2e-16)
                self.assertTrue(np.all(expected >= 0.))
                for result in (a, b):
                    np.testing.assert_allclose(result["stage_cost"], sum(result["term_costs"].values()), rtol=0, atol=1e-15)
        for name in native:
            self.assert_bitwise(native[name], before[name])

    def test_single_balanced_and_alternating_full_load_identical(self):
        force = np.zeros((4, 8, 2)); force[0, :, 0] = 100.
        force[1] = 50.; force[2, ::2, 0] = 100.; force[2, 1::2, 1] = 100.
        force[3, :, 1] = 50.  # Frozen saturation remains .5mg, not 1mg.
        a, b = [module.reward_numpy(force, 100., 1., 1.) for module in (reward_a, reward_b)]
        np.testing.assert_array_equal(a["raw_reward"], np.ones(4))
        np.testing.assert_array_equal(b["raw_reward"], np.ones(4))

    def test_weak_load_gets_dense_deficit_and_gaps_cannot_be_compensated(self):
        force = np.zeros((4, 8, 2)); force[:, :, 0] = 25.
        force[1, 3] = 0.; force[2, 1:7] = 0.; force[3, 3] = (1500., 0.)
        a, b = [module.reward_numpy(force, 100., .7, .8) for module in (reward_a, reward_b)]
        self.assertEqual(b["raw_reward"][0], a["raw_reward"][0]-.5)
        np.testing.assert_array_equal(b["raw_reward"][[1, 2]], -b["support_deficit"][[1, 2]])
        self.assertEqual(a["raw_reward"][1], 0.)
        self.assertLess(b["impact_gate"][3], 1.)
        self.assert_bitwise(a["impact_gate"], b["impact_gate"])

    def test_dense_deficit_is_ungated_and_clipped_load_cannot_erase_gaps(self):
        force = np.zeros((3, 8, 2)); force[..., 0] = 100.
        force[0, 0] = 0.; force[1] = 0.; force[2, 0] = 0.; force[2, 1] = (2000., 0.)
        value = reward_b.reward_numpy(force, 100., 0., 1.)
        np.testing.assert_array_equal(value["raw_reward"], [-.125, -1., -.125])
        np.testing.assert_array_equal(value["raw_reward"], -value["support_deficit_penalty"])
        self.assertTrue(np.all(value["raw_reward"] >= -1.))
        self.assertTrue(np.all(value["raw_reward"] <= 1.))

    def test_counts_are_validated_but_never_a_reward_gate(self):
        native, reference, profile, actions = fixture()
        objective = objective_b.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.)
        original = evaluate(objective, native, reference, actions)
        for value in (0, 32):
            native["ground_contact_count"].fill(value)
            changed = evaluate(objective, native, reference, actions)
            for term in original["raw_terms"]:
                self.assert_bitwise(original["raw_terms"][term], changed["raw_terms"][term])
        native["ground_contact_count"][0, 0, 0, 0] = -1
        with self.assertRaises(ValueError): evaluate(objective, native, reference, actions)

    def test_other_settings_weights_and_clock_unchanged(self):
        a, b = reward_a.validate_settings(), reward_b.validate_settings()
        self.assertEqual({key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}, {"candidate", "support_deficit_weight"})
        self.assertEqual((b["mean_weight"], b["minimum_weight"]), (0., 1.))
        _, _, profile, _ = fixture()
        oa = objective_a.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.)
        ob = objective_b.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.)
        self.assertEqual(oa.weights, ob.weights)
        self.assertEqual(oa.dt, ob.dt)
        self.assertEqual(oa.reward_upper_bound, ob.reward_upper_bound)
        self.assertEqual(oa.contract()["physical_substeps"], ob.contract()["physical_substeps"])
        self.assertFalse(ob.contract()["absolute_height_constraint"])
        self.assertFalse(ob.contract()["zero_sideslip_target"])


if __name__ == "__main__":
    unittest.main()
