"""CPU-only packaging/replay checks; historical D data is never relabelled F."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import package_cem_ground_contact_v2g_nominal as package

D_PUBLIC = ROOT/"results/ground_contact_v2d_cem_nominal_20261023_v1"


class SourceAndScopeTests(unittest.TestCase):
    def test_readonly_replay_status_names_actual_G_not_parent_F(self):
        source=Path(package.__file__).read_text()
        self.assertIn("ALL_EIGHT_CASE_METRICS_AND_ACTUAL_G_CONTACT_DIAGNOSTICS_EXACT",source)
        self.assertNotIn("ALL_EIGHT_CASE_METRICS_AND_ACTUAL_F_CONTACT_DIAGNOSTICS_EXACT",source)

    def test_frozen_E_sources_protocol_and_21_member_ledger(self):
        protocol = package.verify_sources()
        self.assertEqual(len(package.audit.SOURCE_PINS), 22)
        self.assertEqual(protocol["ground_contact_reward_v2"]["support_gap_weight"], 1.)
        self.assertEqual(protocol["ground_contact_reward_v2"]["mean_weight"], .75)
        self.assertEqual(protocol["ground_contact_reward_v2"]["minimum_weight"], .25)

    def test_source_archive_ledger_only_code_configs_no_binary_or_assets(self):
        ledger = package.source_ledger()
        self.assertGreaterEqual(len(ledger), 21)
        self.assertEqual({name: ledger[name] for name in package.audit.SOURCE_PINS}, package.audit.SOURCE_PINS)
        for name, digest in ledger.items():
            with self.subTest(name=name):
                self.assertIn(Path(name).suffix, (".py", ".json", ".cpp"))
                self.assertEqual(package.sha256(ROOT/name), digest)
        self.assertIn("scripts/package_cem_ground_contact_v2g_nominal.py", ledger)
        self.assertIn("scripts/diagnose_ground_contact_windows_v2.py", ledger)

    def test_replay_template_has_no_extract_or_inference(self):
        source = package._replay_source()
        compile(source, "reproduce.py", "exec")
        self.assertNotIn("extractall", source)
        self.assertIn("verify_public", source)
        self.assertIn("dont_write_bytecode", source)
        self.assertIn("artifact_manifest.json", source)

    def test_CEM_bundle_never_infers_PPO_results_or_goal(self):
        summary = json.loads((D_PUBLIC/"cem_analysis.json").read_text())
        # Only exercise the pure status helper using an explicitly historical
        # fixture; this does not produce an F result or public artifact.
        fixture = copy.deepcopy(summary)
        fixture["actual_V2g"] = fixture.pop("actual_V2d")
        status = package.status_from_summary(fixture)
        self.assertFalse(status["goal_complete"])
        self.assertFalse(status["PPO_G"]["new_formal_policy_results_claimed"])
        self.assertNotIn("RUNNING", status["PPO_G"]["status"])
        self.assertTrue(status["CEM_G"]["offline_solver_not_real_time"])
        self.assertEqual(status["CEM_G"]["failed_checks"], fixture["nominal_screen"]["failed_checks"])
        self.assertFalse(status["episode_reward_ranking_across_definitions_claimed"])

    def test_existing_D_summary_cannot_be_published_as_E(self):
        summary = json.loads((D_PUBLIC/"cem_analysis.json").read_text())
        report = json.loads((D_PUBLIC/"cem_report.json").read_text())
        experiment = json.loads((D_PUBLIC/"cem_experiment.json").read_text())
        with self.assertRaises(ValueError):
            package.validate_summary(summary, report, experiment)

    def test_pending_smoke_and_incomplete_are_rejected_before_packaging(self):
        report = json.loads((D_PUBLIC/"cem_report.json").read_text())
        experiment = json.loads((D_PUBLIC/"cem_experiment.json").read_text())
        for change in ({"status": "running"}, {"smoke_only": True}, {"completed_policy_steps": 1},
                       {"native_samples": 9}, {"completed_full_reference": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bad = copy.deepcopy(report); bad.update(change)
                package.audit.validate_metadata(bad, experiment)

    def test_missing_prior_case_rejected(self):
        with self.assertRaises(ValueError):
            package.expected_cases({"original_grounded_baseline": {}, "actual_V2g": {}})


class NativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((D_PUBLIC/"cem_analysis.json").read_text())
        cls.arrays = package.read_arrays(D_PUBLIC/"native_comparison_signals.npz")

    def test_historical_five_case_metrics_exact_without_raw_assets(self):
        reference = {name.removeprefix("reference__"): value for name, value in self.arrays.items()
                     if name.startswith("reference__")}
        expected = {"original_grounded_baseline": self.summary["original_grounded_baseline"],
            **{"prior_actual_V2"+s: self.summary["optional_prior_actual_V2"+s]["metrics"] for s in "abc"},
            "actual_V2d": self.summary["actual_V2d"]}
        for case, metrics in expected.items():
            with self.subTest(case=case):
                trace = {name.removeprefix(case+"__"): value for name, value in self.arrays.items()
                         if name.startswith(case+"__")}
                self.assertEqual(package.recorded_metrics(trace, reference), metrics)

    def test_seven_case_compact_schema_helper_with_explicit_duplicate_fixture(self):
        # A seventh duplicate fixture tests generic array assembly only. It is not
        # accepted by validate_summary and is never written as an F result.
        arrays = dict(self.arrays)
        for name, value in list(arrays.items()):
            if name.startswith("actual_V2d__"):
                suffix = name.removeprefix("actual_V2d__")
                arrays["prior_actual_V2d__"+suffix] = value
                arrays["prior_actual_V2e__"+suffix] = value
                arrays["prior_actual_V2f__"+suffix] = value
                arrays["actual_V2g__"+suffix] = value
                del arrays[name]
        result = package.recompute_comparison(arrays)
        self.assertEqual(tuple(result), package.CASE_ORDER)
        self.assertEqual(result["actual_V2g"], self.summary["actual_V2d"])
        self.assertEqual(result["prior_actual_V2d"], self.summary["actual_V2d"])

    def test_prefix_missing_case_extra_array_and_wrong_clock_rejected(self):
        reference = {name.removeprefix("reference__"): value for name, value in self.arrays.items()
                     if name.startswith("reference__")}
        trace = {name.removeprefix("actual_V2d__"): value for name, value in self.arrays.items()
                 if name.startswith("actual_V2d__")}
        for clock in (trace["time_s"][:-1], trace["time_s"]+.0025):
            with self.subTest(shape=clock.shape), self.assertRaises(ValueError):
                bad = dict(trace); bad["time_s"] = clock
                package.recorded_metrics(bad, reference)
        with self.assertRaises(ValueError):
            package.recompute_comparison(self.arrays)

    def test_historical_all_contact_keys_exact_helper_without_E_claim(self):
        diagnosis = json.loads((D_PUBLIC/"contact_diagnostics.json").read_text())
        arrays = package.read_arrays(D_PUBLIC/"native_contact_context.npz")
        actual = package.audit.contact.recompute_compact(arrays, include_sensitivity=False)
        self.assertTrue(actual)
        for name, value in actual.items():
            with self.subTest(contact_key=name):
                self.assertEqual(value, diagnosis[name])
        with self.assertRaises(ValueError):
            package.verify_contact(arrays, diagnosis, self.summary)


class ArchiveSafetyTests(unittest.TestCase):
    def fixture(self, folder, name="scripts/example.py", *, payload=b"value=1\n", symlink=False):
        target = folder/"executed_source.tar.gz"
        with tarfile.open(target, "x:gz") as archive:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE; member.linkname = "example.py"
                archive.addfile(member)
            else:
                member.size = len(payload); archive.addfile(member, io.BytesIO(payload))
        entry = {"path": name, "bytes": len(payload) if not symlink else 0,
                 "sha256": hashlib.sha256(payload).hexdigest()}
        source = {"archive": {"sha256": package.sha256(target)},
            "executed_controller_source_sha256": package.audit.SOURCE_PINS,
            "models_weights_credentials_native_libraries_included": False, "files": [entry]}
        (folder/"source_snapshot.manifest.json").write_text(json.dumps(source))
        return {"source_archive_sha256": source["archive"]["sha256"],
                "replay_helper_sha256": {name: entry["sha256"]}}

    def test_source_archive_read_in_memory_without_extract_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); manifest = self.fixture(folder)
            before = sorted(p.name for p in folder.iterdir())
            with patch.object(tarfile.TarFile, "extractall", side_effect=AssertionError("No extraction")), \
                 patch.object(tarfile.TarFile, "extract", side_effect=AssertionError("No extraction")):
                package.verify_archive(folder, manifest)
            self.assertEqual(before, sorted(p.name for p in folder.iterdir()))

    def test_binary_traversal_absolute_and_symlink_rejected(self):
        for name, link in (("library.so", False), ("../bad.py", False), ("/bad.py", False),
                           ("scripts/link.py", True)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp); manifest = self.fixture(folder, name, symlink=link)
                with self.assertRaises(ValueError):
                    package.verify_archive(folder, manifest)

    def test_archive_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); manifest = self.fixture(folder)
            manifest["source_archive_sha256"] = "0"*64
            with self.assertRaises(ValueError):
                package.verify_archive(folder, manifest)

    def test_wrong_public_identity_rejected(self):
        with self.assertRaises(ValueError):
            package.verify_public(D_PUBLIC)


if __name__ == "__main__":
    unittest.main()

