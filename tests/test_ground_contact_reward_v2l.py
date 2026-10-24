"""Physical-force arithmetic and real Torch parity; not controller success tests."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2l as mod


def evaluate(force, mg=200.):
    return mod.reward_numpy(np.asarray(force), mg, 1., 1., 1., 1.)


class NormalForceSlewTests(unittest.TestCase):
    def test_constant_single_double_or_alternating_support_has_no_added_cost(self):
        single = np.tile([200., 0.], (8, 1))
        double = np.full((8, 2), 100.)
        alternating = single.copy(); alternating[1::2] = [0., 200.]
        for force in (single, double, alternating):
            with self.subTest(force=force.tolist()):
                result = evaluate(force)
                self.assertEqual(result["normal_force_slew_cost"], 0.)
                self.assertEqual(result["raw_reward"], 1.)
                self.assertEqual(result["K_raw_reward_before_normal_force_slew"], 1.)

    def test_zero_force_keeps_full_K_gap_cost_not_a_positive_flight_bonus(self):
        result = evaluate(np.zeros((8, 2)))
        self.assertEqual(result["normal_force_slew_cost"], 0.)
        self.assertEqual(result["support_gap"], 1.)
        self.assertEqual(result["raw_reward"], -1.)

    def test_internal_impulse_exact_two_of_seven_changes_and_bounded_cap(self):
        force = np.tile([200., 0.], (8, 1)); force[4, 0] = 1400.
        result = evaluate(force)
        self.assertEqual(result["normal_force_slew_samples"].tolist(), [0., 0., 0., 1., 1., 0., 0.])
        self.assertEqual(result["normal_force_slew_cost"], 2./7.)
        self.assertEqual(result["raw_reward"],
                         result["K_raw_reward_before_normal_force_slew"]-2./7.)
        self.assertEqual(result["net_ground_normal_force_increment_n"].shape, (7,))

    def test_first_last_sample_change_has_one_difference_no_invented_boundary(self):
        for index in (0, 7):
            with self.subTest(index=index):
                force = np.tile([200., 0.], (8, 1)); force[index, 0] = 400.
                result = evaluate(force)
                self.assertEqual(result["normal_force_slew_cost"], 1./7.)
                self.assertEqual(result["normal_force_slew_samples"].shape, (7,))

    def test_order_and_live_mg_scaling_matter(self):
        alternating = np.zeros((8, 2)); alternating[::2, 0] = 200.
        clustered = np.zeros((8, 2)); clustered[:4, 0] = 200.
        self.assertEqual(evaluate(alternating)["normal_force_slew_cost"], 1.)
        self.assertEqual(evaluate(clustered)["normal_force_slew_cost"], 1./7.)
        np.testing.assert_allclose(evaluate(2.*alternating, 400.)["raw_reward"],
                                   evaluate(alternating)["raw_reward"])

    def test_parent_all_components_and_intended_beta_bonus_retained(self):
        rng = np.random.default_rng(42)
        force = rng.uniform(0., 500., (4, 8, 2))
        old = mod.previous.reward_numpy(force, 200., .8, .9, .7, 1.)
        new = mod.reward_numpy(force, 200., .8, .9, .7, 1.)
        for key, value in old.items():
            if key != "raw_reward":
                with self.subTest(key=key): np.testing.assert_array_equal(new[key], value)
        np.testing.assert_array_equal(new["K_raw_reward_before_normal_force_slew"], old["raw_reward"])
        self.assertTrue(np.all(new["raw_reward"] <= old["raw_reward"]))

    def test_batched_numpy_real_torch_float32_float64_and_autograd(self):
        import torch
        rng = np.random.default_rng(1)
        for dtype, torch_dtype in ((np.float32, torch.float32), (np.float64, torch.float64)):
            with self.subTest(dtype=str(dtype)):
                force = rng.uniform(10., 300., (2, 3, 8, 2)).astype(dtype)
                mg = np.full((2, 3), 220., dtype=dtype)
                expected = mod.reward_numpy(force, mg, .8, .9, .7, 1.)
                value = torch.tensor(force, dtype=torch_dtype, requires_grad=True)
                actual = mod.reward_torch(value, mg, .8, .9, .7, 1., validate=False)
                for key in expected:
                    np.testing.assert_allclose(actual[key].detach().numpy(), expected[key],
                                               rtol=3.e-6, atol=3.e-6, err_msg=key)
                actual["raw_reward"].sum().backward()
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_invalid_inputs_rejected_or_hotpath_poisoned(self):
        import torch
        for value in (float("nan"), float("inf"), -1.):
            force = np.tile([200., 0.], (8, 1)); force[3, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError): evaluate(force)
                actual = mod.reward_torch(torch.tensor(force), 200., 1., 1., 1., 1., validate=False)
                self.assertTrue(torch.isnan(actual["raw_reward"]))
                self.assertTrue(torch.isnan(actual["normal_force_slew_cost"]))

    def test_hotpath_has_no_file_reads_or_tensor_host_scalar(self):
        import torch
        force = torch.full((4, 8, 2), 100.)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host bool")):
            result = mod.reward_torch(force, 200., 1., 1., 1., 1., validate=False)
        self.assertEqual(tuple(result["raw_reward"].shape), (4,))

    def test_frozen_scope_settings_and_formula_do_not_redefine_acceptance(self):
        self.assertEqual(mod.formula_metadata()["raw_reward_bounds"], [-2., 1.])
        self.assertEqual(mod.DEFAULT_SETTINGS["support_force_threshold_n"], 5.)
        self.assertEqual(mod.DEFAULT_SETTINGS["external_reward_weight"], .4)
        for key in mod.DEFAULT_SETTINGS:
            bad = copy.deepcopy(mod.DEFAULT_SETTINGS)
            bad[key] = "changed" if isinstance(bad[key], str) else bad[key]+.1
            with self.subTest(key=key):
                with self.assertRaises(ValueError): mod.validate_settings(bad)
        mod.verify_base()


if __name__ == "__main__":
    unittest.main()

