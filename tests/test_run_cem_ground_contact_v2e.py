"""CPU-only actual-stage D/E accounting and fixed-runner safety contracts."""
from __future__ import annotations

import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cem_ground_contact_v2d as runner_a
import run_cem_ground_contact_v2e as runner_b


def objective():
    names = ("motion_global_anchor_pos", "drift_speed", "drift_grounded") + tuple("unchanged"+str(i) for i in range(12))
    raw = {name: .6+i*.01 for i, name in enumerate(names)}
    weighted = {name: .2*value for name, value in raw.items()}
    costs = {name: .02*(.2-value) for name, value in weighted.items()}
    return {"raw_terms": raw, "weighted_terms": weighted, "term_costs": costs,
        "reward_upper_bound": 12., "total_reward": sum(weighted.values()),
        "stage_cost": 0., "base_com_vz": -.23, "surrogate_flags": ["legacy"]}


class ActualStageV2eTests(unittest.TestCase):
    def test_other_fourteen_terms_bitwise_and_expected_stage_cost_delta(self):
        legacy = objective(); frozen = copy.deepcopy(legacy)
        for missing in (0, 1, 3, 8):
            with self.subTest(missing=missing):
                force = np.zeros((8, 2, 3)); force[:, 0, 2] = 100.
                force[:missing] = 0.
                if missing == 3: force[7, 0, 2] = 900.  # Same impact gate, not a compensation credit.
                a, b = [module.replace_actual_grounded(legacy, force, 100.) for module in (runner_a, runner_b)]
                self.assertEqual(legacy, frozen)
                self.assertEqual(len(b["raw_terms"]), 15)
                for channel in ("raw_terms", "weighted_terms", "term_costs"):
                    for name in legacy[channel]:
                        if name != "drift_grounded":
                            self.assertEqual(np.asarray(a[channel][name]).tobytes(), np.asarray(b[channel][name]).tobytes())
                ca, cb = a["ground_contact_v2"], b["ground_contact_v2"]
                expected = -.02*.4*ca["support_deficit"]
                self.assertAlmostEqual(b["stage_cost"]-a["stage_cost"], expected, places=15)
                self.assertAlmostEqual(b["term_costs"]["drift_grounded"]-a["term_costs"]["drift_grounded"], expected, places=15)
                self.assertEqual(cb["support"], cb["support_mean"])
                for name in ca:
                    if name not in {"raw_reward", "support_deficit_penalty"}:
                        self.assertEqual(np.asarray(ca[name]).tobytes(), np.asarray(cb[name]).tobytes())
                self.assertEqual(b["total_reward"], sum(b["weighted_terms"].values()))
                self.assertEqual(b["stage_cost"], .02*(b["reward_upper_bound"]-b["total_reward"]))

    def test_single_alternating_balanced_and_weak_load_not_bilateral(self):
        for load in (25., 50., 100.):
            with self.subTest(load=load):
                force = np.zeros((8, 2, 3)); force[:, 0, 2] = load
                original = runner_b.replace_actual_grounded(objective(), force, 100.)
                force[::2, 0, 2] = 0.; force[::2, 1, 2] = load
                alternate = runner_b.replace_actual_grounded(objective(), force, 100.)
                force[..., 2] = load/2.
                balanced = runner_b.replace_actual_grounded(objective(), force, 100.)
                self.assertEqual(original["raw_terms"]["drift_grounded"], alternate["raw_terms"]["drift_grounded"])
                self.assertEqual(original["raw_terms"]["drift_grounded"], balanced["raw_terms"]["drift_grounded"])

    def test_loss_is_dense_even_when_tracking_bonus_is_zero(self):
        base = objective(); base["raw_terms"]["motion_global_anchor_pos"] = 0.
        force = np.zeros((8, 2, 3)); force[:, 0, 2] = 100.; force[0] = 0.
        a, b = [module.replace_actual_grounded(base, force, 100.) for module in (runner_a, runner_b)]
        self.assertEqual(a["raw_terms"]["drift_grounded"], -.125)
        self.assertEqual(b["raw_terms"]["drift_grounded"], 0.)
        self.assertAlmostEqual(b["stage_cost"]-a["stage_cost"], -.02*.4/8., places=15)

    def test_invalid_force_channels_refused_not_replaced_by_net_force(self):
        for kind in ("missing", "shape", "tangent", "nan", "negative"):
            with self.subTest(kind=kind):
                force = np.zeros((8, 2, 3)); force[:, 0, 2] = 100.
                if kind == "missing": force = force[:7]
                if kind == "shape": force = force[..., 2]
                if kind == "tangent": force[0, 0, 0] = 1.
                if kind == "nan": force[0, 0, 2] = np.nan
                if kind == "negative": force[0, 0, 2] = -1.
                with self.assertRaises(ValueError): runner_b.replace_actual_grounded(objective(), force, 100.)

    def test_cli_forbids_unfrozen_optimizer_friction_reward_or_gate_options(self):
        parser = runner_b.parser()
        fixed = ["--native-library", "frozen.so", "--output-dir", "fresh"]
        self.assertEqual(parser.parse_args(fixed+["--max-policy-steps", "1"]).max_policy_steps, 1)
        for extra in (["--mu", ".1"], ["--grounded-weight", "1"], ["--horizon-steps", "3"],
                      ["--support-load-fraction", ".1"], ["--force-threshold", "0"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(fixed+extra)

    def test_existing_output_and_invalid_length_guard_before_native_or_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner_b.nominal, "inputs", side_effect=AssertionError("No input or simulator work")):
                with self.assertRaisesRegex(ValueError, "fresh output"):
                    runner_b.run(SimpleNamespace(output_dir=Path(directory)))
                for value in (0, 350, True, -1):
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, "smoke length"):
                        runner_b.run(SimpleNamespace(output_dir=Path(directory)/"new", max_policy_steps=value))

    def test_no_training_inference_rebuild_or_counterfactual_performance_claim(self):
        source = Path(runner_b.__file__).read_text()
        self.assertNotIn("torch.load(", source)
        self.assertNotIn("torch.jit.load(", source)
        self.assertNotIn("build_native_library(", source)
        self.assertIn('raw[name].copy() for name in ("initial_qpos", "initial_qvel")', source)
        self.assertIn('"PPO_actions_or_future_states_used_for_initial_plan": False', source)
        self.assertIn('"termination_or_reset_after_initialization": False', source)
        self.assertIn('"height_contact_tilt_failure_gate": False', source)
        self.assertIn("objective = replace_actual_grounded(legacy_objective, ground_window, mg)", source)


if __name__ == "__main__":
    unittest.main()
