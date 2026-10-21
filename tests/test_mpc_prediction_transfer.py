"""Audit-only unit fixtures and exact replay of published measured recordings.

Fabricated metadata below is restricted to in-memory unit tests. These tests
never write or publish a synthetic physical episode or experiment result.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_mpc_prediction_transfer as analysis
from controller_references import get_reference


UNIT_HASH = "1" * 64


def unit_reports():
    """Build report-schema fixtures only, with no numerical rollout arrays."""
    spec = get_reference()
    objective = {
        "absolute_height_target": False, "reference_vertical_velocity_target": False,
        "height_regularization": {"mode": "vertical_acceleration_only"},
        "weights": {"speed": 1.}, "stds": {"speed": .35}, "dt_s": .02,
        "reference_order": "held_pre_action",
    }
    shared = {
        "status": "completed", "motion_sha256": spec["motion_npz_sha256"],
        "reference_id": spec["reference_id"], "plant_mu": .7,
        "prediction_mu_scenarios": [.7], "prediction_receives_plant_mu": False,
        "truncated_smoke": False, "completed_policy_steps": 349,
        "pose_height_failure_gate": False, "learned_policy_used": False,
        "tvlqr_used": False, "root_state_prescribed_after_reset": False,
        "external_root_forces_used": False,
        "effective_cem_config": {
            "horizon_steps": 30, "control_blocks": 10, "candidates": 64,
            "iterations": 3, "elite_count": 8, "seed": 42,
            "parameterization": "warm_start_residual_blocks",
        },
        "action_parameterization": {"mode": "warm_start_residual_blocks"},
        "objective_contract": objective, "settings": {"unit_fixture_only": True},
        "contact_slip_definition": "Measured ground-contact-point lateral-speed RMS",
        "rollout_sha256": UNIT_HASH, "_capture_sha256": UNIT_HASH,
    }
    isaac, mujoco = copy.deepcopy(shared), copy.deepcopy(shared)
    isaac.update(engine="Isaac/PhysX", mujoco_model_sha256=UNIT_HASH,
                 fixed_evaluation={"all_termination_terms_removed": True},
                 material_audit={"expected_mu": .7, "robot_static_dynamic_range": [.7, .7],
                                 "ground_bound_materials": [{"coefficients": [.7, .7, 0.]}]})
    mujoco.update(engine="MuJoCo", model_sha256=UNIT_HASH,
                  plant_material_readback={"compiled_sliding_friction_range": [.7, .7]})
    old = {"isaac": isaac, "mujoco": mujoco}
    for report in old.values():
        report["prediction_engine"] = "MuJoCo"
    new = copy.deepcopy(old)
    for report in new.values():
        report.update(prediction_engine="Isaac/PhysX", prediction_backend="physx")
    summary = {
        "reference_spec": spec,
        "input_sha256": {"model": UNIT_HASH, **{plant + suffix: UNIT_HASH
                         for plant in analysis.PLANTS for suffix in ("_report", "_trace")}},
    }
    return summary, old, new


def unit_worker(expected):
    """Create a complete in-memory worker metadata fixture, not runtime proof."""
    sources = {name: UNIT_HASH for name in analysis.REQUIRED_RUNTIME_SOURCES}
    actuators = {}
    for name, kp, kd, effort, speed in (("legs", 500., 10., 80., 15.), ("wheels", 0., 0., 12., 100.)):
        actuators[name] = {key: [value, value] for key, value in {
            "stiffness": kp, "damping": kd, "effort_limit_nm": effort,
            "velocity_limit_rad_s": speed, "saturation_effort_nm": effort,
            "velocity_at_effort_limit_rad_s": 2. * speed}.items()}
    runtime = {
        "engine": "Isaac/PhysX", "num_candidate_environments": 64,
        "coordinate_identical_candidates": True, "cross_environment_collision_filtering": True,
        "environment_spacing_m": 0., "physics_dt_s": .005, "control_dt_s": .02,
        "body_order": list(analysis.BODY_NAMES), "action_order": list(analysis.JOINT_NAMES),
        "objective_contract": expected["objective_contract"],
        "motion_sha256": expected["motion_sha256"], "asset_sha256": UNIT_HASH,
        "device_audit": {
            "status": "verified", "native_and_tensor_device_consistency_verified": True,
            "requested_device": "cuda:0", "environment_device": "cuda:0",
            "physics_sim_view_frontend_device": "cuda:0", "robot_native_transform_tensor_device": "cuda:0",
            "native_physics_backend_device_ordinal": 0, "torch_current_device": 0,
            "physics_cudaDevice_setting": 0, "physics_sceneMultiGPUMode_setting": 0,
            "cuda_visible_devices": None,
        },
        "plant_access": False, "trained_policy_used": False, "root_forces_used": False,
        "fixed_evaluation": {"all_termination_terms_removed": True},
        "legacy_joint_friction_audit": {"all_environments_verified_zero": True},
        "material_audit": {"expected_mu": .7, "robot_static_dynamic_range": [.7, .7],
                           "robot_restitution_range": [0., 0.],
                           "ground_bound_materials": [{"coefficients": [.7, .7, 0.]}]},
        "actuator_audit": actuators, "source_sha256": sources,
    }
    return {
        "kind": "isolated_isaac_physx_batch_tracking_predictor",
        "prediction_engine": "Isaac/PhysX", "population_batching": True,
        "production_cem_wired": True, "learned_initialization": False,
        "runtime_cem_population_path": "predictor.rollout_batch", "completed_rollout_batches": 1,
        "last_reset_audit": {"physics_steps_during_reset": 0},
        "objective": expected["objective_contract"], "prediction_mu_scenarios": [.7],
        "motion_sha256": expected["motion_sha256"],
        "integration_state_semantics": "Integrate continuously over the full horizon",
        "nominal_seed_source": "unchanged physics-only bias-force reference plan",
        "physics_dt_s": .005, "control_dt_s": .02, "worker_runtime": runtime,
        "worker_source_sha256": UNIT_HASH, "adapter_source_sha256": UNIT_HASH,
    }


def complete_unit_reports():
    summary, old, new = unit_reports()
    for plant, report in new.items():
        worker = unit_worker(old["isaac"])
        key = "prediction_backend_metadata" if plant == "isaac" else "physx_prediction_backend"
        report[key] = worker
        report["source_sha256"] = worker["worker_runtime"]["source_sha256"]
    return summary, old, new


class PredictionTransferAuditTests(unittest.TestCase):
    def test_published_actual_pair_replays_exactly(self):
        summary, arrays = analysis.load_pair(analysis.OLD_PAIR)
        traces, signals = analysis.replay_pair(summary, arrays)
        self.assertEqual(set(traces), {"reference", "isaac", "mujoco"})
        self.assertEqual([len(traces[plant]["time_s"]) for plant in analysis.PLANTS], [1397, 1397])
        self.assertEqual(set(signals), set(traces))

    @unittest.skipUnless(all(path.is_file() for path in analysis.OLD_RAW.values()),
                         "Private archived raw reports are optional; public compact replay is tested separately")
    def test_actual_archived_raw_pair_passes_contract_reaudit(self):
        summary, _ = analysis.load_pair(analysis.OLD_PAIR)
        reports = {plant: analysis.read_report(path) for plant, path in analysis.OLD_RAW.items()}
        result = analysis.pair_analysis.audit_reports(
            "NMPC", summary["reference_spec"], reports["isaac"], reports["mujoco"],
            isaac_trace_sha256=summary["input_sha256"]["isaac_trace"],
            mujoco_trace_sha256=summary["input_sha256"]["mujoco_trace"],
            model_sha256=summary["input_sha256"]["model"])
        self.assertTrue(result["height_free_objective_and_evaluation"])

    def test_unit_report_schema_accepts_both_real_runner_metadata_keys(self):
        summary, old, new = complete_unit_reports()
        with patch.object(analysis.pair_analysis, "audit_reports", wraps=analysis.pair_analysis.audit_reports) as audited:
            result = analysis.audit_raw_reports(summary, copy.deepcopy(summary), old, new)
        self.assertEqual(audited.call_count, 2)
        self.assertEqual(set(result["pair_report_contract_audits"]), {"mujoco", "physx"})
        for worker in result["physx_predictor_worker_audits"].values():
            self.assertEqual(worker["completed_rollout_batches"], 1)
            self.assertEqual(worker["last_reset_audit"]["physics_steps_during_reset"], 0)

    def test_unit_worker_requires_actual_batches_population_path_and_zero_step_reset(self):
        _, old, reports = complete_unit_reports()
        changes = (
            {"runtime_cem_population_path": "scalar_fallback"},
            {"completed_rollout_batches": 0},
            {"completed_rollout_batches": True},
            {"last_reset_audit": {"physics_steps_during_reset": 1}},
            {"last_reset_audit": {}},
        )
        for values in changes:
            with self.subTest(values=values):
                report = copy.deepcopy(reports["isaac"])
                report["prediction_backend_metadata"].update(values)
                with self.assertRaises(ValueError):
                    analysis.audit_worker(report, old["isaac"])

    def test_unit_supplied_pair_cannot_skip_native_plant_contracts(self):
        changes = (
            ("mujoco", "root_state_prescribed_after_reset", True),
            ("isaac", "external_root_forces_used", True),
            ("isaac", "engine", "MuJoCo"),
            ("mujoco", "learned_policy_used", True),
            ("isaac", "material_audit", {"expected_mu": .7, "robot_static_dynamic_range": [.2, .2]}),
        )
        for plant, key, value in changes:
            with self.subTest(plant=plant, field=key):
                summary, old, new = complete_unit_reports()
                new[plant][key] = value
                with self.assertRaises(ValueError):
                    analysis.audit_raw_reports(summary, copy.deepcopy(summary), old, new)

    @unittest.skipUnless((analysis.DEFAULT_PROBE / "report.json").is_file()
                         and (analysis.PUBLIC_INPUTS / "physx_mujoco/report.json").is_file(),
                         "Actual published probe and completed production report required")
    def test_actual_probe_source_binding_and_evidence_scope(self):
        expected = analysis.read_report(analysis.prefer_capture_report(analysis.OLD_RAW["isaac"], "mujoco_isaac"))
        report = analysis.read_report(analysis.PUBLIC_INPUTS / "physx_mujoco/report.json")
        worker = analysis.audit_worker(report, expected)
        result = analysis.audit_physx_probe(analysis.DEFAULT_PROBE, {"mujoco": worker}, expected)
        self.assertTrue(result["production_source_and_asset_binding_verified"])
        self.assertEqual(result["reference_frames"], [0, 100, 150])
        self.assertEqual(len(result["stored_array_checks_recomputed_here"]), 3)
        self.assertEqual(set(result["probe_sha256"]), {"report.json", "frame_000.npz", "frame_100.npz", "frame_150.npz"})
        self.assertFalse(result["repeat_trajectories_recomputed_independently"])
        self.assertFalse(result["torque_and_scalar_batch_parity_recomputed_independently"])
        # A different implementation cannot reuse this calibration certificate.
        changed = copy.deepcopy(worker)
        changed["source_sha256"]["physx_batch_predictor.py"] = UNIT_HASH
        with self.assertRaisesRegex(ValueError, "source/asset identity"):
            analysis.audit_physx_probe(analysis.DEFAULT_PROBE, {"mujoco": changed}, expected)

    @unittest.skipUnless((analysis.DEFAULT_PROBE / "frame_000.npz").is_file(),
                         "Actual published single-rollout probe arrays required")
    def test_actual_stored_probe_arrays_reject_unit_tampering(self):
        report = json.loads((analysis.DEFAULT_PROBE / "report.json").read_text())
        case = report["cases"][0]
        saved = analysis.pair_analysis.load_arrays(analysis.DEFAULT_PROBE / "frame_000.npz")
        result = analysis.audit_probe_arrays(saved, case)
        self.assertTrue(result["json_costs_match_exactly"])
        for field, kind in (("controls", "bounds"), ("costs", "json"),
                            ("stage_cost", "sum"), ("controls", "duplicate"),
                            ("trajectory_body_pos_w", "shape")):
            with self.subTest(field=field, tamper=kind):
                arrays = dict(saved)
                arrays[field] = saved[field].copy()
                if kind == "bounds":
                    arrays[field][1, 0, 0] = 2.
                elif kind == "json":
                    arrays[field][1] += .01
                elif kind == "sum":
                    arrays[field][0, 0] += .01
                elif kind == "duplicate":
                    arrays[field][-1, 0, 0] += .01
                else:
                    arrays[field] = arrays[field][:, :-1]
                with self.assertRaises(ValueError):
                    analysis.audit_probe_arrays(arrays, case)

    def test_public_report_defaults_fallback_without_writing_files(self):
        original = ROOT / "outputs/missing-unit-capture/report.json"
        with patch.object(Path, "is_file", return_value=False):
            self.assertEqual(analysis.prefer_capture_report(original, "mujoco_isaac"),
                             analysis.PUBLIC_INPUTS / "mujoco_isaac/report.json")
            self.assertEqual(analysis.prefer_capture_directory(original.parent, "physx_isaac"),
                             analysis.PUBLIC_INPUTS / "physx_isaac")
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(analysis.prefer_capture_report(original, "mujoco_isaac"), original)
            self.assertEqual(analysis.prefer_capture_directory(original.parent, "physx_isaac"), original.parent)

    @unittest.skipUnless((analysis.PUBLIC_INPUTS / "mujoco_mujoco/report.json").is_file(),
                         "Actual public legacy report required")
    def test_actual_missing_legacy_engine_label_recovered_without_modifying_report(self):
        path = analysis.PUBLIC_INPUTS / "mujoco_mujoco/report.json"
        captured = analysis.pair_analysis.sha256(path)
        report = analysis.read_report(path)
        self.assertNotIn("prediction_engine", report)
        before = copy.deepcopy(report)
        identity = analysis.archived_prediction_identity(report)
        self.assertEqual(identity["engine"], "MuJoCo")
        self.assertEqual(identity["identity_source"], "legacy_native_worker_source_and_build_metadata")
        self.assertFalse(identity["original_report_modified"])
        self.assertEqual(report, before)
        self.assertEqual(analysis.pair_analysis.sha256(path), captured)
        summary, _ = analysis.load_pair(analysis.OLD_PAIR)
        self.assertEqual(captured, summary["input_sha256"]["mujoco_report"])

    @unittest.skipUnless((analysis.PUBLIC_INPUTS / "mujoco_mujoco/report.json").is_file(),
                         "Actual public legacy report required")
    def test_actual_legacy_identity_rejects_unit_tampered_metadata_and_declared_physx(self):
        original = analysis.read_report(analysis.PUBLIC_INPUTS / "mujoco_mujoco/report.json")
        for change in ("kind", "source_hash", "build_hash", "declared_physx"):
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                metadata = report["native_prediction_backend"]
                if change == "kind":
                    metadata["kind"] = "unverified_predictor"
                elif change == "source_hash":
                    metadata["workers"]["0.7"]["source_sha256"] = UNIT_HASH
                elif change == "build_hash":
                    metadata["workers"]["0.7"]["native_library_sha256"] = "invalid"
                else:
                    report["prediction_engine"] = "Isaac/PhysX"
                with self.assertRaises(ValueError):
                    analysis.archived_prediction_identity(report)

    def test_unit_optional_raw_rollout_rehash_cannot_disagree_with_report(self):
        summary, old, new = complete_unit_reports()
        for report in (*old.values(), *new.values()):
            report["_rollout_capture_sha256"] = UNIT_HASH
        result = analysis.audit_raw_reports(summary, copy.deepcopy(summary), old, new)
        self.assertEqual(result["raw_rollout_files_rehashed"], 4)
        new["isaac"]["_rollout_capture_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "raw rollout differs"):
            analysis.audit_raw_reports(summary, copy.deepcopy(summary), old, new)


if __name__ == "__main__":
    unittest.main()
