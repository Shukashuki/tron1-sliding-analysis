"""Synthetic CPU unit fixtures only; no new policy/physics or actual-result claims."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_archived_wheel_friction as mod


def arrays_fixture():
    normal = np.zeros((2793, 2, 3)); normal[..., 2] = 100.
    total = normal.copy(); total[:, 0, 0] = 70.; total[:, 1, 1] = 35.
    return {"time_s": np.arange(2793) * .0025,
        "physics_dt_s": np.array(.0025), "policy_dt_s": np.array(.02),
        "wheel_ground_total_contact_force_w_n": total,
        "wheel_ground_contact_force_w_n": normal,
        "wheel_ground_contact_count": np.ones((2793, 2), dtype=np.int32),
        "wheel_lateral_contact_slip_m_s": np.tile([.1, .01], (2793, 1))}


def report_fixture(raw_sha256="a" * 64):
    wrapper = ROOT / "scripts/eval_ppo_ground_contact_v2d.py"
    proof = {"status": "verified", "smoke_only": False, "trained_policy_claimed": True,
        "validation_backend": "native CPU MuJoCo", "additional_training_updates": 0,
        "legacy_evaluator_sha256": mod.LEGACY_EVALUATOR_SHA256, "recording_sha256": raw_sha256,
        "native_states": 2793, "policy_decisions": 349, "physics_dt_s": .0025,
        "policy_dt_s": .02, "decimation": 8, "duration_s": 6.98,
        "normalized_actor_sha256": "b" * 64, "wrapper_sha256": mod.sha256(wrapper),
        "cohort_id": "SYNTHETIC_UNIT_FIXTURE_NOT_ACTUAL_POLICY"}
    return {"engine": "MuJoCo", "status": "completed", "full_horizon_diagnostic": True,
        "completed_full_reference": True, "termination": "motion_end", "hidden_resets": 0,
        "root_state_writes": 1, "start_reference_frame": 0, "final_reference_frame": 348,
        "reference_frames": 349, "policy_steps": 349, "physics_steps": 2792,
        "expected_duration_s": 6.98, "recorded_duration_s": 6.98,
        "motion_sha256": mod.MOTION_SHA256, "model_sha256": mod.MODEL_SHA256,
        "terrain": None, "terrain_runtime_audit": None,
        "policy_kind": "normalized_torchscript_actor", "policy_sha256": "b" * 64,
        "contract": {"policy_algorithm": "PPO", "physics_dt": .0025, "decimation": 8},
        "evaluator_source_sha256": mod.sha256(wrapper), "ground_contact_v2d_evaluation": proof,
        "contact_slip_helper_sha256": mod.NORMAL_SLIP_HELPER_SHA256,
        "wheel_lateral_contact_slip_evidence_kind": "actual_ground_contact_points",
        "contact_slip_measurement": {"active_point_normal_force_threshold_n": 5.,
            "active_point_comparison": ">", "ground_is_stationary": True, "sample_dt_s": .0025,
            "aggregation": "normal-force-weighted RMS over active contact points per wheel",
            "ground_force_kind": "sum of normal contact forces, matching the filtered PhysX contact API"},
        "contact_material_audit": {"native_mapping_verified": True, "explicit_contact_pair_count": 0,
            "compiled_native_timestep_s": .0025, "colliding_geom_names": list(mod.GEOM_NAMES),
            "colliding_geom_ids": list(range(10)), "wheel_geom_ids": [5, 9],
            "geom_friction": [[.7, 0., 0.]] * 10, "pair_coefficient": .7,
            "geom_priority": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1], "geom_condim": [3] * 10,
            "native_plant_readback": {"solver_configuration": {"timestep": .0025, "cone": 0.}}}}


def provenance():
    return mod.validate_report(report_fixture(), "a" * 64)


class ArchivedFrictionTests(unittest.TestCase):
    def test_pinned_sources_and_report_material_are_not_guessed(self):
        value = provenance()
        self.assertEqual(value["effective_pair_mu"], .7)
        self.assertEqual(value["native_cone_name"], "pyramidal")
        self.assertFalse(value["actual_engine_boundary_computable"])
        self.assertFalse(value["legacy_whole_file_locally_rehashed"])
        self.assertEqual(mod.sha256(mod.cone.__file__), mod.DIAGNOSTIC_HELPER_SHA256)
        self.assertEqual(value["collector_source_sha256"][str(mod.NORMAL_HELPER.relative_to(ROOT))], mod.NORMAL_SLIP_HELPER_SHA256)

    def test_known_l_r_window_quantiles_and_denominators(self):
        rows, arrays, result = mod.analyze(arrays_fixture(), provenance())
        self.assertEqual([(r["window"], r["wheel"]) for r in rows],
                         [("core", "L"), ("core", "R"), ("full_contact", "L"), ("full_contact", "R")])
        self.assertEqual([r["native_samples"] for r in rows], [1025, 1025, 2777, 2777])
        for row in rows:
            expected = 1. if row["wheel"] == "L" else .5
            self.assertAlmostEqual(row["rho_l2_reference_p50"], expected)
            self.assertAlmostEqual(row["rho_l2_reference_p95"], expected)
            self.assertEqual(row["near_reference_0p9_fraction_of_loaded"], float(row["wheel"] == "L"))
            self.assertEqual(row["loaded_valid_resultant_samples"], row["native_samples"])
            self.assertFalse(row["actual_per_contact_engine_boundary_available"])
        self.assertTrue(np.isnan(arrays["engine_rho"]).all())
        self.assertTrue(np.isnan(arrays["slip_total_speed_m_s"]).all())
        self.assertFalse(result["static_friction_state_inferred"])
        self.assertTrue(result["resultant_reference_lower_bound_only"])

    def test_weak_point_count_zero_never_becomes_nonslip(self):
        raw = arrays_fixture(); i = 600
        raw["wheel_ground_contact_force_w_n"][i, 0, 2] = 6.
        raw["wheel_ground_total_contact_force_w_n"][i, 0] = (4.2, 0., 6.)
        raw["wheel_ground_contact_count"][i, 0] = 0
        raw["wheel_lateral_contact_slip_m_s"][i, 0] = 0.
        rows, arrays, _ = mod.analyze(raw, provenance())
        row = rows[0]
        self.assertEqual(row["loaded_valid_resultant_samples"], 1025)
        self.assertEqual(row["patch_lateral_RMS_eligible_samples"], 1024)
        self.assertEqual(row["patch_lateral_RMS_unavailable_loaded_samples"], 1)
        self.assertEqual(row["observed_near_reference_AND_patch_lateral_RMS_gt0p05_fraction_of_loaded_lower_bound"], 1024/1025)
        self.assertEqual(row["patch_RMS_gt0p05_fraction_given_near_reference_AND_patch_RMS_eligible"], 1.)
        self.assertTrue(np.isnan(arrays["patch_lateral_RMS_with_count_eligibility_m_s"][i, 0]))
        self.assertFalse(arrays["wheel_lateral_slip_valid"][i, 0])

    def test_unloaded_and_no_valid_window_are_missing_not_zero_rho(self):
        raw = arrays_fixture(); raw["wheel_ground_contact_force_w_n"][:] = 0.
        raw["wheel_ground_total_contact_force_w_n"][:] = 0.
        raw["wheel_ground_contact_count"][:] = 0; raw["wheel_lateral_contact_slip_m_s"][:] = 0.
        rows, arrays, _ = mod.analyze(raw, provenance())
        self.assertTrue(np.isnan(arrays["coulomb_reference_rho_l2"]).all())
        for row in rows:
            self.assertEqual(row["loaded_valid_resultant_samples"], 0)
            self.assertIsNone(row["rho_l2_reference_p50"])
            self.assertIsNone(row["near_reference_0p9_fraction_of_loaded"])

    def test_raw_ratio_not_clipped_and_roundoff_is_distinguished(self):
        raw = arrays_fixture(); raw["wheel_ground_total_contact_force_w_n"][:, 0, 0] = np.nextafter(70., np.inf)
        rows, arrays, _ = mod.analyze(raw, provenance())
        self.assertGreater(arrays["coulomb_reference_rho_l2"][600, 0], 1.)
        self.assertEqual(rows[0]["raw_strict_float_rho_gt1_fraction_of_loaded_NOT_physical_violation"], 1.)
        self.assertEqual(rows[0]["rho_gt1plus1e_8_fraction_of_loaded_exploratory"], 0.)
        self.assertLess(rows[0]["rho_above1_max_excess_unclipped"], 1e-8)

    def test_compact_replays_every_row_and_signal_without_original_raw(self):
        p = provenance(); rows, arrays, _ = mod.analyze(arrays_fixture(), p)
        replay = mod.recompute_compact(arrays, p)
        self.assertEqual(replay["rows"], rows)
        self.assertFalse(replay["original_raw_assets_weights_or_physics_required"])
        wrong = copy.deepcopy(arrays); wrong["coulomb_reference_rho_l2"][10, 0] += .01
        with self.assertRaisesRegex(ValueError, "derived signal"):
            mod.recompute_compact(wrong, p)
        wrong = copy.deepcopy(arrays); wrong["effective_mu"][0, 0, 0] = .5
        with self.assertRaisesRegex(ValueError, "effective-mu"):
            mod.recompute_compact(wrong, p)

    def test_observed_persistence_has_no_interval_fill_and_missing_breaks_runs(self):
        time = np.arange(7) * .0025
        mask = np.array([False, True, False, True, True, False, True])
        value = mod.observed_positive_spans(time, mask)
        self.assertEqual(value["event_count"], 3)
        self.assertAlmostEqual(value["longest_observed_sample_span_lower_bound_s"], .0025)
        self.assertEqual(value["longest_positive_samples"], 2)
        self.assertEqual(value["single_positive_sample_span_s"], 0.)
        singleton = mod.observed_positive_spans(time, np.array([False, True, False, False, False, False, False]))
        self.assertEqual(singleton["longest_observed_sample_span_lower_bound_s"], 0.)
        empty = mod.observed_positive_spans(time, np.zeros(7, dtype=bool))
        self.assertEqual(empty["event_count"], 0)
        self.assertIsNone(empty["longest_observed_sample_span_lower_bound_s"])
        raw = arrays_fixture(); raw["wheel_ground_contact_count"][600, 0] = 0
        raw["wheel_lateral_contact_slip_m_s"][600, 0] = 0.
        rows, compact, _ = mod.analyze(raw, provenance())
        self.assertEqual(rows[0]["near_reference_observed_event_count"], 1)
        self.assertEqual(rows[0]["near_reference_AND_eligible_patch_RMS_slip_observed_event_count"], 2)
        self.assertEqual(mod.recompute_compact(compact, provenance())["rows"], rows)
    def test_reject_physx_cem_partial_reset_or_bad_report_identity(self):
        changes = {"engine": "PhysX", "status": "running", "completed_full_reference": False,
            "hidden_resets": 1, "root_state_writes": 2, "termination": "fall", "physics_steps": 9,
            "policy_steps": 1, "expected_duration_s": 7., "model_sha256": "0" * 64,
            "motion_sha256": "0" * 64, "policy_kind": "full_state_CEM", "terrain": {},
            "contact_slip_helper_sha256": "0" * 64, "evaluator_source_sha256": "0" * 64}
        for name, value in changes.items():
            with self.subTest(name=name):
                report = report_fixture(); report[name] = value
                with self.assertRaises(ValueError): mod.validate_report(report, "a" * 64)

    def test_reject_unknown_mu_pair_condim_clock_or_patch_provenance(self):
        for kind in ("pair", "mu", "condim", "cone", "priority", "slipthreshold", "slipcomparison", "legacy", "rawbind"):
            with self.subTest(kind=kind):
                report = report_fixture(); material = report["contact_material_audit"]
                if kind == "pair": material["explicit_contact_pair_count"] = 1
                if kind == "mu": material["pair_coefficient"] = .5
                if kind == "condim": material["geom_condim"][5] = 6
                if kind == "cone": material["native_plant_readback"]["solver_configuration"]["cone"] = 2.
                if kind == "priority": material["geom_priority"][5] = 0
                if kind == "slipthreshold": report["contact_slip_measurement"]["active_point_normal_force_threshold_n"] = 0.
                if kind == "slipcomparison": report["contact_slip_measurement"]["active_point_comparison"] = ">="
                if kind == "legacy": report["ground_contact_v2d_evaluation"]["legacy_evaluator_sha256"] = "0" * 64
                if kind == "rawbind": report["ground_contact_v2d_evaluation"]["recording_sha256"] = "0" * 64
                with self.assertRaises(ValueError): mod.validate_report(report, "a" * 64)

    def test_array_force_clock_count_and_slip_contracts(self):
        for kind in ("missingtotal", "tangentnormal", "negative", "normalz", "shape", "nan", "clock", "count", "slip", "falsezero"):
            with self.subTest(kind=kind):
                raw = arrays_fixture()
                if kind == "missingtotal": raw.pop("wheel_ground_total_contact_force_w_n")
                if kind == "tangentnormal": raw["wheel_ground_contact_force_w_n"][0, 0, 0] = .1
                if kind == "negative": raw["wheel_ground_contact_force_w_n"][0, 0, 2] = -1.
                if kind == "normalz": raw["wheel_ground_total_contact_force_w_n"][0, 0, 2] += 1.
                if kind == "shape": raw["wheel_ground_total_contact_force_w_n"] = raw["wheel_ground_total_contact_force_w_n"][:-1]
                if kind == "nan": raw["wheel_ground_contact_force_w_n"][0, 0, 2] = np.nan
                if kind == "clock": raw["time_s"][100] = 0.
                if kind == "count": raw["wheel_ground_contact_count"][0, 0] = -1
                if kind == "slip": raw["wheel_lateral_contact_slip_m_s"][0, 0] = -1.
                if kind == "falsezero": raw["wheel_ground_contact_count"][0, 0] = 0
                with self.assertRaises(ValueError): mod.validate_arrays(raw)

    def test_fresh_output_guard_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fresh output"):
                mod.run(SimpleNamespace(output_dir=Path(directory)))

    def test_synthetic_end_to_end_exclusive_outputs_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw_path = root / "rollout.npz"; report_path = root / "report.json"
            np.savez_compressed(raw_path, **arrays_fixture())
            mod.write_json(report_path, report_fixture(mod.sha256(raw_path)))
            args = SimpleNamespace(raw=raw_path, report=report_path, raw_sha256=mod.sha256(raw_path),
                report_sha256=mod.sha256(report_path), output_dir=root / "fresh", legacy_source=None)
            result = mod.run(args)
            self.assertEqual(set(path.name for path in args.output_dir.iterdir()),
                             {"friction_signals.npz", "metric_table.csv", "summary.json"})
            self.assertFalse(result["actual_per_contact_engine_boundary_computed"])
            self.assertFalse(result["reward_or_acceptance_changed"])
            self.assertFalse(result["overall_goal_complete_claimed"])
            with np.load(args.output_dir / "friction_signals.npz", allow_pickle=False) as z:
                compact = {key: z[key].copy() for key in z.files}
            self.assertEqual(mod.recompute_compact(compact, result["provenance"])["rows"], result["rows"])
            with self.assertRaisesRegex(ValueError, "fresh output"): mod.run(args)

    def test_cli_no_guessed_mu_engine_boundary_or_acceptance_override(self):
        args = ["--raw", "r.npz", "--report", "r.json", "--raw-sha256", "a" * 64,
                "--report-sha256", "b" * 64, "--output-dir", "fresh"]
        self.assertEqual(mod.parser().parse_args(args).legacy_source, None)
        with self.assertRaises(SystemExit): mod.parser().parse_args(args + ["--mu", ".7"])

    def test_optional_exact_frozen_legacy_source(self):
        path = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
        if not path.is_file(): self.skipTest("Optional frozen legacy content fixture absent")
        value = mod.validate_report(report_fixture(), "a" * 64, path)
        self.assertTrue(value["legacy_whole_file_locally_rehashed"])
        self.assertEqual(value["optional_legacy_source_evidence"]["sha256"], mod.LEGACY_EVALUATOR_SHA256)


if __name__ == "__main__":
    unittest.main()
