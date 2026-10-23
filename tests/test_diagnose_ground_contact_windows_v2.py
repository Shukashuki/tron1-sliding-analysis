"""Recorded-contact arithmetic fixtures only; no counterfactual dynamics/reward."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import diagnose_ground_contact_windows_v2 as diagnostic
from controller_trace import native_trace
from test_analyze_ppo_ground_contact_v2 import recording, reference, new_fixture


def inputs():
    raw = recording()
    trace = native_trace(raw, "policy_mujoco")
    return trace, raw["wheel_ground_contact_force_w_n"][..., 2].copy(), reference()


def analyze(trace, force, ref, **kwargs):
    return diagnostic.analyze_contact(trace["time_s"], trace["position"], trace["com_velocity"],
        trace["quaternion"], force, 100., ref, **kwargs)


class ContactWindowDiagnosticTest(unittest.TestCase):
    def test_at_least_one_not_bilateral_and_total_threshold_is_distinct(self):
        trace, force, ref = inputs()
        result = analyze(trace, force, ref)
        load = result["windows"]["contact"]["threshold_load_fractions"]
        self.assertEqual(load["strict_gt5N"]["any_wheel"]["sample_fraction"], 1.)
        self.assertEqual(load["strict_gt5N"]["both_wheels_diagnostic_not_required"]["sample_fraction"], 0.)
        force[:] = 30.
        result = analyze(trace, force, ref)
        level = result["windows"]["contact"]["threshold_load_fractions"]["strict_gt0p5mg"]
        self.assertEqual(level["any_wheel"]["sample_fraction"], 0.)
        self.assertEqual(level["total_wheel_force_strict_gt_threshold"]["sample_fraction"], 1.)
        self.assertFalse(result["bilateral_requirement_added"])

    def test_strict_zero_and_five_newton_thresholds_preserved(self):
        trace, force, ref = inputs()
        force[:] = [5., 0.]
        result = analyze(trace, force, ref)
        thresholds = result["windows"]["core"]["threshold_load_fractions"]
        self.assertEqual(thresholds["strict_gt0N"]["any_wheel"]["sample_fraction"], 1.)
        self.assertEqual(thresholds["strict_gt5N"]["any_wheel"]["sample_fraction"], 0.)
        self.assertEqual(result["windows"]["core"]["total_upward_Fz_over_mg_quantiles_native_samples"]["maximum"], .05)

    def test_isolated_subcontrol_and_persistent_gaps_have_native_context(self):
        trace, force, ref = inputs()
        force[100] = 0.
        force[120:124] = 0.
        force[200:240] = 0.
        result = analyze(trace, force, ref)
        window = result["windows"]["contact"]
        self.assertEqual(window["isolated_uncensored_2p5ms_gap_count"], 1)
        self.assertEqual(window["gap_duration_histogram"]["multi_sample_lt20ms"]["events"], 1)
        self.assertEqual(window["observed_ge100ms_gap_count_including_censored_lower_bounds"], 1)
        longest = window["longest_gap_events_with_native_context"][0]
        self.assertAlmostEqual(longest["observed_duration_s"], .1)
        self.assertEqual(longest["native_samples"], 40)
        context = longest["context"]
        self.assertLess(context["xy"]["rmse_on_available_overlap_m"], 1e-14)
        self.assertAlmostEqual(context["COM_speed_xy_m_s"]["mean"], np.sqrt(1.0625))
        self.assertEqual(context["COM_vertical_velocity_world_m_s"]["maximum"], 0.)
        self.assertFalse(context["height_or_flight_classification"])

    def test_boundary_censoring_does_not_invent_future_gap_or_tail_XY(self):
        trace, force, ref = inputs()
        force[2784] = 0.
        result = analyze(trace, force, ref)
        contact = result["windows"]["contact"]
        self.assertEqual(contact["any_wheel_5N"]["unsupported_physical_samples"], 1)
        self.assertEqual(contact["any_wheel_5N"]["longest_no_support_observed_s"], 0.)
        self.assertEqual(contact["gap_duration_histogram"]["boundary_censored_lt100ms_observed"]["events"], 1)
        force[2784] = 100.; force[2785:] = 0.
        result = analyze(trace, force, ref)
        tail = result["windows"]["full_native"]["longest_gap_events_with_native_context"][0]
        self.assertTrue(tail["right_window_censored"])
        self.assertFalse(tail["context"]["xy"]["reference_coverage_complete"])
        self.assertEqual(tail["context"]["xy"]["reference_overlap_samples"], 0)
        self.assertIsNone(tail["context"]["xy"]["rmse_on_available_overlap_m"])
        self.assertTrue(result["windows"]["contact"]["any_wheel_5N"]["every_physical_sample_supported"])

    def test_3mg_impacts_use_post_action_eight_substeps_not_initial_snapshot(self):
        trace, force, ref = inputs()
        force[0, 0] = 4000.
        force[41, 0] = 600.
        result = analyze(trace, force, ref)
        impacts = result["eight_substep_windows"]
        self.assertEqual(impacts["impact_window_count"], 1)
        event = impacts["impact_windows"][0]
        self.assertEqual(event["decision_index"], 5)
        self.assertEqual(event["native_physical_indices_inclusive"], [41, 48])
        self.assertEqual(event["per_wheel_peak_Fz_over_mg"], [6., 0.])
        self.assertAlmostEqual(event["impact_gate_force_factor_only"], np.exp(-1/16))
        force[41, 0] = 300.
        self.assertEqual(analyze(trace, force, ref)["eight_substep_windows"]["impact_window_count"], 0)

    def test_optional_temporal_factor_sensitivity_is_not_actual_policy_reward_or_performance(self):
        trace, force, ref = inputs()
        force[16] = 0.
        result = analyze(trace, force, ref)
        self.assertFalse(result["offline_contact_factor_sensitivity"]["included"])
        result = analyze(trace, force, ref, include_sensitivity=True)
        comparison = result["offline_contact_factor_sensitivity"]
        self.assertTrue(comparison["tracking_gate_not_reconstructed"])
        self.assertTrue(comparison["not_actual_PPO_reward_or_return"])
        self.assertTrue(comparison["not_V2b_controller_performance_or_counterfactual_trajectory"])
        self.assertEqual(comparison["windows_with_positive_mean_but_zero_minimum"], 1)
        self.assertEqual(comparison["V2a_ungated_support_factor"]["minimum"], .75*7/8)
        self.assertEqual(comparison["V2b_ungated_support_factor"]["minimum"], 0.)
        self.assertFalse(result["new_controller_performance_claimed"])
        self.assertFalse(result["overall_task_success_claimed"])

    def test_nonwheel_contact_is_only_diagnostic_when_actually_measured(self):
        trace, force, ref = inputs()
        result = analyze(trace, force, ref)
        self.assertFalse(result["windows"]["full_native"]["nonwheel_ground_contact_diagnostic"]["available"])
        nonwheel = np.zeros(2793); nonwheel[200:240] = 20.
        result = analyze(trace, force, ref, nonwheel=nonwheel)
        bearing = result["windows"]["contact"]["nonwheel_ground_contact_diagnostic"]
        self.assertTrue(bearing["available"])
        self.assertFalse(bearing["used_as_automatic_failure_gate"])
        self.assertEqual(bearing["persistent_events"]["persistent_event_count"], 1)

    def test_invalid_physical_values_and_clock_fail_closed(self):
        trace, force, ref = inputs()
        for kind in ("mg", "nan", "negative", "clock", "quaternion"):
            with self.subTest(kind=kind):
                t, f = copy.deepcopy(trace), force.copy()
                if kind == "nan": f[10, 0] = np.nan
                if kind == "negative": f[10, 0] = -.001
                if kind == "clock": t["time_s"][10] += .0001
                if kind == "quaternion": t["quaternion"][10] *= 2.
                with self.assertRaises(ValueError):
                    diagnostic.analyze_contact(t["time_s"], t["position"], t["com_velocity"], t["quaternion"],
                        f, 0. if kind == "mg" else 100., ref)

    def test_compact_reproduces_all_diagnostics_without_files_or_models(self):
        trace, force, ref = inputs()
        arrays = {"time_s": trace["time_s"], "position": trace["position"], "velocity": trace["com_velocity"],
            "quaternion": trace["quaternion"], "ground_normal_Fz_n": force, "evaluation_weight_n": np.array(100.)}
        arrays.update({"reference_"+name: ref[name] for name in ("time_s", "position", "velocity")})
        expected = analyze(trace, force, ref, include_sensitivity=True)
        with patch("builtins.open", side_effect=AssertionError("No external files allowed")), \
                patch.object(Path, "read_text", side_effect=AssertionError("No assets allowed")):
            actual = diagnostic.recompute_compact(arrays, include_sensitivity=True)
        self.assertEqual(actual, expected)

    def test_future_PPO_requires_parent_proof_and_documented_evaluation_not_training_weight(self):
        with tempfile.TemporaryDirectory(prefix="tron1-contact-diagnostic-test-") as temp:
            folder = Path(temp)/"new-synthetic-PPO"
            new_fixture(folder, "mujoco")
            args = SimpleNamespace(kind="policy_mujoco", raw=folder/"rollout.npz", report=folder/"report.json",
                raw_sha256=diagnostic.sha256(folder/"rollout.npz"), report_sha256=diagnostic.sha256(folder/"report.json"),
                protocol=diagnostic.ppo.DEFAULT_PROTOCOL, mg_n=None, mg_source=None)
            with self.assertRaisesRegex(ValueError, "Evaluation mg is unavailable"):
                diagnostic._load_inputs(args)
            args.mg_n = 100.
            with self.assertRaisesRegex(ValueError, "documented evaluation-weight source"):
                diagnostic._load_inputs(args)
            args.mg_source = "Synthetic fixture evaluation weight, not a real robot measurement"
            trace, force, _, mg, source, proof, _ = diagnostic._load_inputs(args)
            self.assertEqual(mg, 100.)
            self.assertEqual(source, args.mg_source)
            self.assertEqual(proof["status"], "verified")
            args.raw_sha256 = "0"*64
            with self.assertRaisesRegex(ValueError, "input SHA differs"):
                diagnostic._load_inputs(args)


CEM = ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7"


@unittest.skipUnless((CEM/"rollout.npz").is_file() and (CEM/"report.json").is_file()
                    and diagnostic.ppo.DEFAULT_REFERENCE.is_file(), "Published CEM/reference fixture not installed")
class PublishedCEMContactDiagnosticTest(unittest.TestCase):
    def test_SHA_bound_recorded_CEM_load_and_offline_pipeline_do_not_claim_V2b_performance(self):
        with tempfile.TemporaryDirectory(prefix="tron1-contact-diagnostic-cem-test-") as temp:
            args = SimpleNamespace(kind="nmpc", raw=CEM/"rollout.npz", report=CEM/"report.json",
                raw_sha256="b4a6dc3ee5e22977140c669f4a9041fe207abaa1ecd129814bc6d3343e01a8ae",
                report_sha256="a1573ea6a2c97d9c1a6f132f14d1b0a3aab9abdd650606b5f73602c980061081",
                protocol=diagnostic.ppo.DEFAULT_PROTOCOL, reference=diagnostic.ppo.DEFAULT_REFERENCE,
                mg_n=None, mg_source=None, include_v2b_sensitivity=True, output_dir=Path(temp)/"new-analysis")
            result = diagnostic.run(args)
            self.assertAlmostEqual(result["normalization_weight_n"], 218.49813011007384)
            self.assertFalse(result["new_controller_performance_claimed"])
            self.assertTrue(result["offline_contact_factor_sensitivity"]["not_actual_PPO_reward_or_return"])
            with np.load(args.output_dir/"native_contact_context.npz", allow_pickle=False) as archive:
                actual = diagnostic.recompute_compact({key: archive[key] for key in archive.files}, include_sensitivity=True)
            for key, value in actual.items(): self.assertEqual(value, result[key])
            self.assertNotIn(b"\r\n", (args.output_dir/"gap_events.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "Preserve diagnostics"): diagnostic.run(args)
            args.mg_n, args.mg_source = 100., "wrong synthetic scale"
            with self.assertRaisesRegex(ValueError, "disagrees with validation report"):
                diagnostic._load_inputs(args)


if __name__ == "__main__": unittest.main()
