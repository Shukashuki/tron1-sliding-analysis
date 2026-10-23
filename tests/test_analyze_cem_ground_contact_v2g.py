"""Strict offline F identity, reward-recompute and unchanged-screen tests.

An A trace is used ONLY as a numerical reward-audit fixture, never relabeled
as an actual F controller rollout or passed through the F provenance gate.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_cem_ground_contact_v2g as mod


def metadata():
    report = {"status": "completed", "smoke_only": False, "truncated_smoke": False,
        "completed_policy_steps": 349, "native_samples": 2793, "completed_full_reference": True,
        "kind": "400hz_CEM_ground_contact_reward_v2g_anywheel",
        "reward_definition_version": "ground_contact_v2g_any_wheel", "protocol_sha256": mod.PROTOCOL_SHA256,
        "source_sha256": copy.deepcopy(mod.SOURCE_PINS), "condition": "beta_on", "plant_mu": .7,
        "prediction_mu_scenarios": [.7], "prediction_friction_known": True,
        "floor_filter_verified": True, "termination_or_reset_after_initialization": False,
        "height_contact_tilt_failure_gate": False, "physics_dt_s": .0025, "policy_dt_s": .02, "decimation": 8,
        "ground_contact_reward_v2": mod.definition.formula_metadata(), "ground_reward_external_weight": .4,
        "reward_term_count": 15, "other_reward_terms_unchanged": 14, "actual_reward_uses_ground_contact_v2": True,
        "native_library_sha256": mod.LIBRARY_SHA256, "native_backend": {"native_abi": 3},
        "CEM_config": mod.read_json(mod.runner.nominal.DEFAULT_SETTINGS)["nmpc"], "discount_factor": .99}
    experiment = {"status": "completed", "kind": "ground_contact_reward_v2g_CEM_nominal_only",
        "source_sha256": copy.deepcopy(mod.SOURCE_PINS), "native_abi": 3, "sample_size": 1,
        "new_policy_training_or_inference": False, "automatic_retry": False}
    return report, experiment


def screen_fixture():
    windows = {name: {"tracking": {"xy_rmse_m": .1, "xy_peak_m": .2, "speed_retention": 1., "beta_rmse_deg": 5.},
        "world_angular_xy_common10_100": {"common_band_rms": .5}} for name in ("core", "whole_reference_overlap")}
    value = {"windows": windows, "any_wheel_ground_support": {"full_contact": {
        "every_physical_sample_supported": True, "any_wheel_support_sample_fraction": 1., "longest_no_support_observed_s": 0.}}}
    old = copy.deepcopy(value); old["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"] = 1.
    return value, old


class AnalyzerV2gTests(unittest.TestCase):
    def test_exact_frozen_protocol_and_source_metadata(self):
        report, experiment = metadata()
        mod.validate_metadata(report, experiment)
        self.assertEqual(len(mod.SOURCE_PINS), 22)
        self.assertTrue({"scripts/ground_contact_reward_v2.py", "scripts/ground_contact_reward_v2b.py",
            "scripts/ground_contact_reward_v2c.py", "scripts/ground_contact_reward_v2d.py", "scripts/ground_contact_reward_v2f.py", "scripts/ground_contact_reward_v2g.py"} <= mod.SOURCE_PINS.keys())
        self.assertEqual(mod.load_protocol()["nominal_acceptance_before_training"],
            mod.read_json(mod.base.DEFAULT_PROTOCOL)["nominal_acceptance_before_training"])

    def test_old_smoke_running_prefix_and_wrong_native_contract_rejected(self):
        mutations = {"status": "running", "smoke_only": True, "truncated_smoke": True,
            "completed_policy_steps": 1, "native_samples": 9, "completed_full_reference": False,
            "kind": "400hz_CEM_ground_contact_reward_v2a_anywheel",
            "reward_definition_version": "ground_contact_v2_any_wheel", "protocol_sha256": "0"*64,
            "condition": "beta_off", "plant_mu": .15, "prediction_mu_scenarios": [.15],
            "prediction_friction_known": False, "floor_filter_verified": False,
            "termination_or_reset_after_initialization": True, "height_contact_tilt_failure_gate": True,
            "physics_dt_s": .005, "policy_dt_s": .04, "decimation": 4,
            "ground_reward_external_weight": 1., "reward_term_count": 14,
            "other_reward_terms_unchanged": 13, "actual_reward_uses_ground_contact_v2": False,
            "native_library_sha256": "0"*64}
        for key, value in mutations.items():
            with self.subTest(key=key):
                report, experiment = metadata(); report[key] = value
                with self.assertRaises(ValueError): mod.validate_metadata(report, experiment)

    def test_missing_extra_or_changed_source_ledger_refused(self):
        for change in ("missing", "extra", "hash", "experiment"):
            with self.subTest(change=change):
                report, experiment = metadata()
                if change == "missing": report["source_sha256"].pop(next(iter(mod.SOURCE_PINS)))
                if change == "extra": report["source_sha256"]["unapproved.py"] = "0"*64
                if change == "hash": report["source_sha256"][next(iter(mod.SOURCE_PINS))] = "0"*64
                if change == "experiment": experiment["source_sha256"] = {}
                with self.assertRaises(ValueError): mod.validate_metadata(report, experiment)

    def test_contact_formula_and_native_abi_cannot_silently_fall_back(self):
        for kind in ("Aformula", "weakthreshold", "ABI", "batchABI", "scope", "retry"):
            with self.subTest(kind=kind):
                report, experiment = metadata()
                if kind == "Aformula": report["ground_contact_reward_v2"] = mod.definition.base.formula_metadata()
                if kind == "weakthreshold": report["ground_contact_reward_v2"]["settings"]["support_load_fraction"] = .1
                if kind == "ABI": experiment["native_abi"] = 2
                if kind == "batchABI": report["native_backend"]["native_abi"] = 2
                if kind == "scope": experiment["sample_size"] = 2
                if kind == "retry": experiment["automatic_retry"] = True
                with self.assertRaises(ValueError): mod.validate_metadata(report, experiment)

    def test_protocol_hash_cannot_accept_relaxed_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"altered.json"
            value = mod.read_json(mod.DEFAULT_PROTOCOL)
            value["nominal_acceptance_before_training"]["any_wheel_support_fraction_min"] = .8
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "protocol SHA"):
                mod.load_protocol(path)

    def test_single_sample_gap_still_fails_same_locked_screen(self):
        candidate, old = screen_fixture(); protocol = mod.load_protocol()
        screen = mod.base.nominal_screen(candidate, old, protocol)
        self.assertTrue(screen["numeric_nominal_episode_pass"])
        self.assertFalse(screen["overall_task_goal_complete_claimed"])
        candidate["any_wheel_ground_support"]["full_contact"].update(
            every_physical_sample_supported=False, any_wheel_support_sample_fraction=.999,
            longest_no_support_observed_s=.0025)
        failed = mod.base.nominal_screen(candidate, old, protocol)
        self.assertEqual(failed["failed_checks"], ["any_wheel_every_physical_sample", "longest_no_support_gap"])

    def test_existing_output_guard_and_partial_artifacts_do_not_score(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(mod, "load_completed", side_effect=AssertionError("No source read")):
                with self.assertRaisesRegex(ValueError, "fresh output"):
                    mod.run(SimpleNamespace(output_dir=Path(directory)))
            with self.assertRaisesRegex(ValueError, "Missing direct completed"):
                mod.load_completed(directory)

    def test_completed_actual_A_episode_is_not_C(self):
        path = ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7"
        if not path.is_dir(): self.skipTest("Optional original local A fixture absent")
        with self.assertRaisesRegex(ValueError, "Wrong G identity"):
            mod.load_completed(path)

    def test_completed_actual_B_episode_is_not_C(self):
        path = ROOT/"outputs/ground_contact_v2b_20261023_v1/cem_mu0p7"
        if not (path/"experiment.json").is_file(): self.skipTest("Optional completed prior B fixture absent")
        with self.assertRaisesRegex(ValueError, "Wrong G identity"):
            mod.load_completed(path)

    def test_optional_B_comparison_remains_actual_B_not_counterfactual_E(self):
        path = ROOT/"outputs/ground_contact_v2b_20261023_v1/cem_mu0p7"
        if not (path/"experiment.json").is_file(): self.skipTest("Optional completed prior B fixture absent")
        reference = mod.base.metrics.reference_data(mod.runner.paired.DEFAULT_REFERENCE)
        raw, report, _, _ = mod.prior_b.load_completed(path)
        result = mod.load_v2b(path, reference, raw, report)
        self.assertEqual(result["input_sha256"]["rollout.npz"], mod.V2B_PINS["rollout.npz"])
        self.assertFalse(result["episode_returns_comparable_claimed"])
        self.assertFalse(result["counterfactual_G_result_claimed"])
        changed = copy.deepcopy(report); changed["CEM_config"]["horizon_steps"] += 1
        with self.assertRaisesRegex(ValueError, "B/G fixed setting"):
            mod.load_v2b(path, reference, raw, changed)

    def test_completed_actual_C_episode_is_not_D_and_optional_C_comparison_is_honest(self):
        path = ROOT/"outputs/ground_contact_v2c_20261023_v1/cem_mu0p7"
        if not (path/"experiment.json").is_file(): self.skipTest("Optional prior actual C absent")
        with self.assertRaisesRegex(ValueError, "Wrong G identity"): mod.load_completed(path)
        reference = mod.base.metrics.reference_data(mod.runner.paired.DEFAULT_REFERENCE)
        raw, report, _, _ = mod.prior_c.load_completed(path)
        result = mod.load_v2c(path, reference, raw, report)
        self.assertEqual(result["input_sha256"]["rollout.npz"], mod.V2C_PINS["rollout.npz"])
        self.assertFalse(result["episode_returns_comparable_claimed"])
        self.assertFalse(result["counterfactual_G_result_claimed"])

    def test_completed_actual_D_episode_is_not_E_and_prior_D_stays_source_bound(self):
        path = ROOT/"outputs/ground_contact_v2d_20261023_v1/cem_mu0p7"
        if not (path/"experiment.json").is_file(): self.skipTest("Optional prior actual D absent")
        with self.assertRaisesRegex(ValueError, "Wrong G identity"): mod.load_completed(path)
        reference = mod.base.metrics.reference_data(mod.runner.paired.DEFAULT_REFERENCE)
        raw, report, _, _ = mod.prior_d.load_completed(path)
        self.assertEqual(len(report["source_sha256"]), 20)
        result = mod.load_v2d(path, reference, raw, report)
        self.assertEqual(result["input_sha256"]["rollout.npz"], mod.V2D_PINS["rollout.npz"])
        self.assertFalse(result["episode_returns_comparable_claimed"])
        self.assertFalse(result["counterfactual_G_result_claimed"])

    def test_fixed_optimizer_or_discount_cannot_change(self):
        for name in ("discount_factor", "CEM_config"):
            with self.subTest(name=name):
                report, experiment = metadata()
                if name == "discount_factor": report[name] = .95
                else: report[name]["horizon_steps"] = 31
                with self.assertRaisesRegex(ValueError, "optimizer/discount"):
                    mod.validate_metadata(report, experiment)

    def test_optional_completed_ACTUAL_E_full_reward_and_file_proof(self):
        path = ROOT/"outputs/ground_contact_v2g_20261023_v1/cem_mu0p7"
        if not (path/"experiment.json").is_file():
            self.skipTest("Formal ACTUAL F terminal files are not available; never score a prefix")
        raw, report, hashes, audit = mod.load_completed(path)
        self.assertEqual((len(raw["time_s"]), len(raw["actions"])), (2793, 349))
        self.assertEqual(report["source_sha256"], mod.SOURCE_PINS)
        self.assertEqual(audit["status"], "verified_exact")
        self.assertEqual(audit["stages"], 349)
        self.assertEqual(hashes["rollout.npz"], report["rollout_sha256"])
        self.assertFalse(audit["PPO_return_recomputed"])

    def test_published_original_baseline_exact_bindings_and_metrics(self):
        if not mod.base.BASELINE_SUMMARY.is_file(): self.skipTest("Optional published original baseline absent")
        reference = mod.base.metrics.reference_data(mod.runner.paired.DEFAULT_REFERENCE)
        actual, hashes = mod.load_original(reference)
        self.assertEqual(hashes["rollout.npz"], mod.BASELINE_PINS["rollout.npz"])
        self.assertIn("core", actual["windows"])

    def test_E_reward_audit_arithmetic_fixture_exact_and_tamper_rejected(self):
        """A stored states/actions only test arithmetic; NO actual F performance claim."""
        path = ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7"
        if not path.is_dir(): self.skipTest("Optional original local A arithmetic fixture absent")
        with np.load(path/"rollout.npz", allow_pickle=False) as z:
            raw = {key: z[key].copy() for key in z.files}
        report = mod.read_json(path/"report.json")
        report["ground_contact_reward_v2"] = mod.definition.formula_metadata()
        decisions = json.loads((path/"solver_log.json").read_text())
        with np.load(mod.runner.paired.DEFAULT_REFERENCE, allow_pickle=False) as z:
            root_id = z["body_names"].tolist().index("base_Link")
            reference_speed = np.linalg.norm(z["body_lin_vel_w"][:, root_id, :2].astype(float), axis=-1)
        for row in decisions:
            i = row["frame"]
            row["objective"] = mod.runner.replace_actual_grounded(row["objective"],
                raw["ground_normal_force_window_w"][i], report["model_weight_n"],
                reference_speed=float(reference_speed[i]))
        raw["stage_cost"] = np.array([row["objective"]["stage_cost"] for row in decisions])
        raw["stage_reward"] = np.array([.02*row["objective"]["total_reward"] for row in decisions])
        audit = mod.runner.recompute_actual_rewards(raw, report, decisions)
        self.assertEqual((audit["status"], audit["stages"], audit["reward_terms"]), ("verified_exact", 349, 15))
        self.assertFalse(audit["PPO_return_recomputed"])
        decisions[60]["objective"]["raw_terms"]["drift_grounded"] += .01
        with self.assertRaisesRegex(ValueError, "exactly reproduce"):
            mod.runner.recompute_actual_rewards(raw, report, decisions)


    def test_actual_F_episode_is_not_G_and_optional_F_comparison_preserves_F_proof(self):
        path=ROOT/"outputs/ground_contact_v2f_20261023_v1/cem_mu0p7"
        self.assertTrue((path/"experiment.json").is_file())
        with self.assertRaisesRegex(ValueError, "Wrong G identity"): mod.load_completed(path)
        reference=mod.base.metrics.reference_data(mod.runner.paired.DEFAULT_REFERENCE)
        raw,report,_,_=mod.prior_f.load_completed(path)
        result=mod.load_v2f(path,reference,raw,report)
        self.assertEqual(result["input_sha256"]["rollout.npz"],mod.V2F_PINS["rollout.npz"])
        self.assertFalse(result["episode_returns_comparable_claimed"])
        self.assertFalse(result["counterfactual_G_result_claimed"])


if __name__ == "__main__":
    unittest.main()

