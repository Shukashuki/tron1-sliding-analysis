"""Offline C publication guards; synthetic unit fixtures are NOT C results."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import package_ppo_ground_contact_v2c_nominal as mod


def synthetic_rows(passes=(True, True), contact=True):
    """Only status-format unit data: not a source-bound model/episode fixture."""
    return {"rows": [{"id": "physx_trained_to_"+engine, "validation_backend": label,
        "new_V2": {"any_wheel_ground_support": {"full_contact": {
            "every_physical_sample_supported": contact, "any_wheel_support_sample_fraction": 1. if contact else .8,
            "longest_no_support_observed_s": 0. if contact else .04}}},
        "nominal_screen": {"numeric_nominal_episode_pass": passed,
            "failed_checks": [] if passed else ["core_HF_reduction"]}}
        for (engine, label), passed in zip(mod.audit.BACKENDS.items(), passes, strict=True)]}


def evidence():
    return {engine: {"actor_sha256": "a"*64, "checkpoint_sha256": "b"*64} for engine in mod.audit.BACKENDS}


class PackageCTests(unittest.TestCase):
    def test_existing_output_guard_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mod, "verify_sources", side_effect=AssertionError("No input work")):
            with self.assertRaisesRegex(ValueError, "fresh output"):
                mod.run(SimpleNamespace(output_dir=Path(directory), input_root=Path(directory)/"missing"))

    def test_real_analyzer_renderer_and_C_protocol_pins(self):
        protocol = mod.verify_sources()
        self.assertEqual(protocol["cohort_id"], "tron1_ground_contact_v2c_any_wheel_400hz_v1")
        self.assertEqual(protocol["ground_contact_reward_v2"]["mean_weight"], 0.)
        self.assertEqual(protocol["ground_contact_reward_v2"]["minimum_weight"], 1.)
        self.assertEqual(protocol["ground_contact_reward_v2"]["support_deficit_weight"], 1.)
        self.assertEqual({"ground_contact_reward_v2.py", "ground_contact_reward_v2b.py", "ground_contact_reward_v2c.py"}
            & mod.audit.TRAINING_SOURCE_PINS.keys(), {"ground_contact_reward_v2.py", "ground_contact_reward_v2b.py", "ground_contact_reward_v2c.py"})
        self.assertEqual(len(mod.PROOF_FILES), 14)
        self.assertTrue(set(mod.BOUND_PROOFS) <= set(mod.PROOF_FILES))
        self.assertTrue(all(name.endswith(".json") for name in mod.PROOF_FILES))
        self.assertFalse({"policy.pt", "model_final.pt"} & set(mod.PROOF_FILES))
        self.assertEqual(set(mod.SNAPSHOT_PINS), {"source_snapshot_v1", "continuation_snapshot_v1"})
        self.assertNotIn("evaluation_dependencies_v1", mod.SNAPSHOT_PINS)

    def test_status_uses_actual_screen_verdict_not_lower_HF_or_fixed_failure(self):
        for passes, contact in (((True, True), True), ((True, False), True), ((False, False), False)):
            with self.subTest(passes=passes, contact=contact):
                status = mod.status_from_summary(synthetic_rows(passes, contact), evidence(), "synthetic-unit-fixture")
                self.assertEqual(status["both_numerical_nominal_screens_passed"], all(passes))
                self.assertEqual(status["strict_continuous_anywheel_contact_achieved"], contact)
                self.assertEqual([row["numeric_nominal_episode_pass"] for row in status["cases"]], list(passes))
                self.assertFalse(status["goal_complete"])
                self.assertTrue(status["formal_training_and_validation_complete"])
                self.assertFalse(status["weights_included_or_executed_locally"])
                self.assertEqual(status["conditions"]["single_wheel_support_allowed"], True)
                self.assertIn("legacy evaluator reward logging is NOT numerically compared with CEM returns", status["fairness"]["evaluation"])
        self.assertIn("REVIEW_PENDING", mod.status_from_summary(synthetic_rows(), evidence(), "fixture")["status"])

    def test_actual_A_analysis_never_accepted_as_C_publication(self):
        path = ROOT/"outputs/ground_contact_v2_20261023_v1/ppo_dual_engine_analysis_v1/summary.json"
        if not path.is_file(): self.skipTest("Optional actual A analysis absent")
        with self.assertRaisesRegex(ValueError, "actual frozen C analysis"):
            mod.validate_summary(mod.read_json(path))

    def test_actual_B_analysis_and_renderer_never_substitute_for_C(self):
        root = ROOT/"outputs/ground_contact_v2b_20261023_v1"
        analysis = root/"ppo_dual_engine_analysis_v1/summary.json"
        render = root/"ppo_render_v1/provenance.json"
        if not (analysis.is_file() and render.is_file()): self.skipTest("Optional actual B completed fixtures absent")
        summary = mod.read_json(analysis)
        with self.assertRaisesRegex(ValueError, "actual frozen C analysis"):
            mod.validate_summary(summary)
        with self.assertRaisesRegex(ValueError, "actual frozen C renderer"):
            mod.verify_render(mod.read_json(render), summary, render.parent)

    def test_metric_arithmetic_replay_using_explicit_A_fixture_only(self):
        """Reuse frozen identical arithmetic; bypass identity ONLY inside this unit test."""
        folder = ROOT/"outputs/ground_contact_v2_20261023_v1/ppo_dual_engine_analysis_v1"
        if not (folder/"summary.json").is_file(): self.skipTest("Optional actual A arithmetic fixture absent")
        summary = mod.read_json(folder/"summary.json")
        with patch.object(mod, "validate_summary", return_value={}):
            result = mod.recompute_compact(folder/"native_states.npz", summary)
            self.assertEqual(set(result), set(mod.COMPACT_CASE_ORDER))
            wrong = copy.deepcopy(summary)
            wrong["rows"][0]["new_V2"]["windows"]["core"]["tracking"]["xy_rmse_m"] += .001
            with self.assertRaisesRegex(ValueError, "recomputation"):
                mod.recompute_compact(folder/"native_states.npz", wrong)

    def test_actual_A_renderer_never_accepted_as_C(self):
        root = ROOT/"outputs/ground_contact_v2_20261023_v1"
        if not (root/"ppo_render_v1/provenance.json").is_file(): self.skipTest("Optional actual A render absent")
        with self.assertRaisesRegex(ValueError, "actual frozen C renderer"):
            mod.verify_render(mod.read_json(root/"ppo_render_v1/provenance.json"),
                mod.read_json(root/"ppo_dual_engine_analysis_v1/summary.json"), root/"ppo_render_v1")

    def test_render_schema_maps_C_v2c_cases_to_new_analysis_cases(self):
        """Schema-only synthetic fixture; dummy bytes are never published/rendered."""
        summary = synthetic_rows(); summary["reference_sha256"] = mod.audit.INPUT_PINS["motion"]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); artifacts = {}
            for name in ("comparison.png", "fullbody_comparison.mp4", "replay_states.npz"):
                (folder/name).write_bytes(b"synthetic unit fixture, NOT an actual render")
                artifacts[name] = mod.sha256(folder/name)
            render = {"kind": "PPO_contact_V2c_min8_ungated_deficit_actual_case_caption_recorded_replay",
                "cohort_id": mod.audit.evaluator.COHORT_ID, "reward_definition_version": mod.audit.evaluator.REWARD_VERSION,
                "renderer_sha256": mod.RENDERER_SHA256, "protocol_sha256": mod.audit.PROTOCOL_SHA256,
                "support_deficit_weight": 1., "external_grounded_weight": .4,
                "reference_sha256": summary["reference_sha256"], "model_sha256": mod.audit.INPUT_PINS["model"],
                "case_order": mod.CASE_ORDER.copy(), "full_four_panel_comparison": True, "explicit_subset": None,
                "video_generated": True, "native_states_per_case": 2793, "video_frames": 350, "video_fps": 50,
                "video_size_pixels": [2560, 850], "camera_same_distance_all_cases_frames": True,
                "camera_yaw_follow": False, "camera_zoom": False, "robot_mesh_alpha": 1.,
                "new_physics_policy_inference_training": False, "smoothing_interpolation_alignment_height_correction": False,
                "goal_completion_claimed": False, "training_audit_required_and_bound": True,
                "A_B_formal_results_changed_or_substituted": False, "cases": {}, "input_evidence": {},
                "replay_FK_audit": {}, "artifacts": artifacts}
            for engine, row in zip(mod.audit.BACKENDS, summary["rows"], strict=True):
                row["input_evidence"] = {"unit_fixture_engine": engine}
                row["baseline_evidence"] = {"unit_fixture_baseline": engine}
                row["original_grounded_baseline"] = {"unit_fixture": "NOT_ACTUAL_BASELINE"}
                for variant, field, evidence_key in (("baseline", "original_grounded_baseline", "baseline_evidence"),
                                                     ("v2c", "new_V2", "input_evidence")):
                    key = variant+"_"+engine
                    render["cases"][key] = {"analysis": row[field], "A_reward_policy_substituted": False,
                        "new_nominal_screen": row["nominal_screen"] if variant == "v2c" else None,
                        "C_cohort_id": mod.audit.evaluator.COHORT_ID if variant == "v2c" else None}
                    render["input_evidence"][key] = row[evidence_key]
                    render["replay_FK_audit"][key] = {"samples_checked": 2793, "root_pose_copied_exactly": True,
                        "max_wheel_origin_error_m": 0., "tolerance_m": 1e-4}
            mod.verify_render(render, summary, folder)
            for field, wrong in (("case_order", ["baseline_physx", "new_physx", "baseline_mujoco", "new_mujoco"]),
                                 ("camera_zoom", True), ("video_frames", 349), ("goal_completion_claimed", True),
                                 ("renderer_sha256", "0"*64), ("training_audit_required_and_bound", False),
                                 ("support_deficit_weight", 0.), ("external_grounded_weight", .8)):
                with self.subTest(field=field):
                    bad = copy.deepcopy(render); bad[field] = wrong
                    with self.assertRaises(ValueError): mod.verify_render(bad, summary, folder)

    def test_both_real_C_source_snapshot_manifests_are_valid(self):
        for stem, digest in mod.SNAPSHOT_PINS.items():
            with self.subTest(stem=stem):
                result = mod.verify_source_snapshot(mod.DEFAULT_ROOT/(stem+".tar.gz"),
                    mod.DEFAULT_ROOT/(stem+".manifest.json"), digest)
                self.assertEqual(result["archive_sha256"], digest)

    def test_archive_rejects_binary_link_traversal_and_duplicate_member_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("binary", "symlink", "escape", "duplicate"):
                with self.subTest(kind=kind):
                    name = "scripts/source.py" if kind in ("symlink", "duplicate") else "../escape.py" if kind == "escape" else "actor.pt"
                    target = root/(kind+".tar.gz")
                    item = tarfile.TarInfo(name); payload = b"pass\n"; item.size = len(payload)
                    if kind == "symlink": item.type = tarfile.SYMTYPE; item.linkname = "elsewhere.py"
                    with tarfile.open(target, "w:gz") as archive: archive.addfile(item, None if kind == "symlink" else io.BytesIO(payload))
                    entry = {"path": name, "size_bytes": len(payload), "sha256": mod.hashlib.sha256(payload).hexdigest()}
                    manifest = root/(kind+".json")
                    mod.write_json(manifest, {"archive_sha256": mod.sha256(target), "no_models_weights_credentials": True,
                        "files": [entry, entry] if kind == "duplicate" else [entry]})
                    with self.assertRaises(ValueError): mod.verify_source_snapshot(target, manifest)

    def test_frozen_source_archive_SHA_cannot_be_replaced_with_other_valid_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root/"source.tar.gz"; payload = b"pass\n"
            item = tarfile.TarInfo("scripts/source.py"); item.size = len(payload)
            with tarfile.open(target, "w:gz") as archive: archive.addfile(item, io.BytesIO(payload))
            manifest = root/"source.json"
            mod.write_json(manifest, {"archive_sha256": mod.sha256(target), "no_models_weights_credentials": True,
                "files": [{"path": item.name, "size_bytes": len(payload), "sha256": mod.hashlib.sha256(payload).hexdigest()}]})
            with self.assertRaisesRegex(ValueError, "snapshot SHA"):
                mod.verify_source_snapshot(target, manifest, "0"*64)

    def test_no_missing_formal_proofs_are_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing direct allowlisted C"):
                mod.verify_training_proofs(directory, {}, {})

    def test_scope_contains_no_training_inference_remote_or_physics_calls(self):
        source = Path(mod.__file__).read_text()
        for forbidden in ("torch.load(", "torch.jit.load(", "mj_step(", "subprocess.", "ssh ", "build_native_library("):
            self.assertNotIn(forbidden, source)
        self.assertIn('"goal_complete": False', source)
        self.assertIn('"no_separate_evaluation_dependencies_snapshot_fabricated": True', source)

    def test_local_replay_ledger_contains_C_not_an_A_or_B_relabel(self):
        ledger = mod.local_source_ledger()
        self.assertIn("scripts/render_ppo_ground_contact_v2c.py", ledger)
        self.assertIn("scripts/analyze_ppo_ground_contact_v2c.py", ledger)
        self.assertIn("scripts/package_ppo_ground_contact_v2c_nominal.py", ledger)
        self.assertIn("config/ppo_ground_contact_v2c_400hz_v1.json", ledger)
        self.assertTrue(all(Path(name).suffix in (".py", ".json") for name in ledger))

    def test_public_artifact_path_escape_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mod.write_json(root/"artifact_manifest.json", {"kind": "verified_FORMAL_C_min8_deficit_PPO_completed_nominal_bundle",
                "models_weights_credentials_included": False, "artifacts": {"../actor.pt": {"sha256": "0"*64, "bytes": 2}}})
            with patch.object(mod, "verify_sources", return_value={}):
                with self.assertRaisesRegex(ValueError, "Unsafe/missing"):
                    mod.verify_published(root)

    def test_optional_ACTUAL_formal_C_analysis_and_render_when_completed(self):
        root = mod.DEFAULT_ROOT
        analysis, render = root/"ppo_dual_engine_analysis_v1/summary.json", root/"ppo_render_v1/provenance.json"
        if not (analysis.is_file() and render.is_file()):
            self.skipTest("Actual formal C dual-engine analysis/render not complete; no C results invented")
        summary = mod.read_json(analysis)
        mod.recompute_compact(analysis.parent/"native_states.npz", summary)
        mod.verify_render(mod.read_json(render), summary, render.parent)


if __name__ == "__main__":
    unittest.main()
