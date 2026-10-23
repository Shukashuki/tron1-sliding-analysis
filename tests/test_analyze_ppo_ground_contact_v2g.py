"""CPU arithmetic/provenance fixtures only; synthetic actors are never loaded."""
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
import analyze_ppo_ground_contact_v2g as analyzer
from nmpc_tracking_objective import JOINT_NAMES


def reference():
    t = np.arange(349)*.02
    return {"time_s": t, "position": np.column_stack((t, .25*t, np.full_like(t, .75))),
        "velocity": np.tile([1., .25, 0.], (349, 1)),
        "quaternion": np.tile([1., 0., 0., 0.], (349, 1))}


def recording(engine="mujoco", amplitude=.1):
    t = np.arange(2793)*.0025
    angular = np.column_stack((amplitude*np.sin(2*np.pi*20*t), np.zeros((2793, 2))))
    force = np.zeros((2793, 2, 3)); force[:, 0, 2] = 100.
    actions = np.zeros((349, 8))
    raw = {"time_s": t, "physics_dt_s": np.array(.0025), "policy_dt_s": np.array(.02),
        "policy_time_s": np.arange(349)*.02, "actions_raw": actions.copy(),
        "actions_clipped": actions.copy(), "previous_clipped_actions": actions.copy(),
        "action_observation": np.zeros((349, 51)), "root_pos": np.column_stack((t, .25*t, np.full_like(t, .75))),
        "root_quat_wxyz": np.tile([1., 0., 0., 0.], (2793, 1)),
        "root_com_lin_vel_w": np.tile([1., .25, 0.], (2793, 1)), "root_ang_vel_w": angular,
        "joint_pos": np.zeros((2793, 8)), "joint_vel": np.zeros((2793, 8)),
        "joint_torque_nm": np.zeros((2793, 8)), "joint_names": np.array(JOINT_NAMES),
        "wheel_pos": np.zeros((2793, 2, 3)), "wheel_ground_contact_count": np.zeros((2793, 2), dtype=int),
        "wheel_ground_contact_force_w_n": force, "wheel_lateral_contact_slip_m_s": np.zeros((2793, 2)),
        "nonwheel_ground_contact_force_norm_n": np.zeros(2793)}
    if engine == "physx":
        fields = {"root_position_m": "root_pos", "root_quaternion_wxyz": "root_quat_wxyz",
            "root_linear_velocity_w_m_s": "root_com_lin_vel_w", "root_angular_velocity_w_rad_s": "root_ang_vel_w",
            "joint_position_rad": "joint_pos", "joint_velocity_rad_s": "joint_vel", "wheel_position_m": "wheel_pos",
            "applied_joint_torque_nm": "joint_torque_nm"}
        for target, source in fields.items(): raw[target] = raw[source][:, None].copy()
        for field in ("wheel_ground_contact_count", "wheel_ground_contact_force_w_n",
                      "wheel_lateral_contact_slip_m_s", "nonwheel_ground_contact_force_norm_n"):
            raw[field] = raw[field][:, None]
        for key, width in (("actions_raw", 8), ("actions_clipped", 8), ("action_observation", 51)):
            raw[key] = np.zeros((2793, 1, width))
        raw["valid_mask"] = np.ones((2793, 1), dtype=bool)
        raw["simulation_step_counter"] = np.arange(2793)
        raw["action_was_applied"] = np.ones((2793, 1), dtype=bool)
        raw["action_was_applied"][0] = False
    return raw


def new_fixture(folder, engine):
    """SHA-linked synthetic metadata, NOT evidence of executed physics/training."""
    folder.mkdir()
    raw = recording(engine)
    raw_name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
    np.savez_compressed(folder/raw_name, **raw)
    protocol = analyzer.load_protocol()
    profile = json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text())
    profile.update(action_rate_weight=-.5, ground_contact_reward_v2=analyzer.definition.validate_settings())
    trained = copy.deepcopy(profile); profile["contact_friction"] = .7
    contract = json.loads((ROOT/"config/nominal_actuator_contract.json").read_text())
    contract.update(physics_dt=.0025, decimation=8, policy_fps=50., full_horizon_diagnostic=True,
        actor_includes_observation_normalizer=True, training_environment_transitions=12288000,
        reward_definition_version=analyzer.evaluator.REWARD_VERSION,
        task_profile=profile, trained_task_profile=trained,
        training_domain_randomization=protocol["training_randomization"],
        evaluation_contact_friction_override={"evaluation_contact_friction": .7})
    ledger = {name: "c"*64 for name in ("ground_contact_v2g_training_audit.json", "cohort_completion_audit.json",
        "manifest.json", "policy_contract.json", "final_policy_contract.json", "run_status.json")}
    ledger.update({"model_final.pt": "d"*64, "policy.pt": "e"*64})
    runtime = {"reward_definition_version": analyzer.evaluator.REWARD_VERSION,
        "settings": analyzer.definition.validate_settings(), "physics_dt_s": .0025,
        "policy_dt_s": .02, "substeps": 8, "steps_started": 24008, "steps_completed": 24008,
        "physics_samples": 192064, "unique_reward_windows": 24008,
        "device_finite_normal_force_health": True, "device_finite_reward_health": True,
        "cpu_fixture_only": False, "single_wheel_support_allowed": True, "bilateral_constraint": False,
        "height_or_zero_slip_target": False, "observation_action_or_physics_change": False,
        "module_sha256": analyzer.TRAINING_SOURCE_PINS["ppo_ground_contact_v2g.py"],
        "definition_sha256": analyzer.TRAINING_SOURCE_PINS["ground_contact_reward_v2g.py"],
        "immutable_v2a_definition_sha256": analyzer.TRAINING_SOURCE_PINS["ground_contact_reward_v2.py"],
        "immutable_v2b_definition_sha256": analyzer.TRAINING_SOURCE_PINS["ground_contact_reward_v2b.py"],
        "immutable_v2c_definition_sha256": analyzer.TRAINING_SOURCE_PINS["ground_contact_reward_v2c.py"],
        "immutable_v2d_definition_sha256": analyzer.TRAINING_SOURCE_PINS["ground_contact_reward_v2d.py"],
        "force_bearing_samples_L_R_any_both": [30, 20, 40, 10]}
    training = {"status": "completed_verified", "cohort_id": analyzer.evaluator.COHORT_ID,
        "reward_definition_version": analyzer.evaluator.REWARD_VERSION,
        "protocol_sha256": analyzer.PROTOCOL_SHA256, "training_environment_steps": 12288000,
        "normalizer_in_actor": True, "from_scratch": True, "smoke_only": False,
        "trained_policy_claimed": True, "other14_reward_terms_changed": False, "new_reward_count": 15,
        "settings": analyzer.definition.validate_settings(), "checkpoint_sha256": "d"*64,
        "exported_policy_sha256": "e"*64, "manifest_sha256": "c"*64,
        "policy_contract_sha256": "c"*64, "initial_policy_state_sha256": analyzer.evaluator.EXPECTED_INITIAL_POLICY_SHA256,
        "source_sha256": {"/remote/frozen/scripts/"+name: digest for name,digest in analyzer.TRAINING_SOURCE_PINS.items()},
        "contact_runtime_audit": runtime}
    from ground_contact_beta_contract import BETA_SOURCE_SHA256, BETA_QUALITY_SOURCE
    beta_path = "/remote/frozen/upstream/tron1-learning-from-human/training/tron1_drift.py"
    training["source_sha256"][beta_path] = BETA_SOURCE_SHA256
    runtime.update(immutable_v2f_definition_sha256=analyzer.TRAINING_SOURCE_PINS["ground_contact_reward_v2f.py"],
        beta_quality_source=BETA_QUALITY_SOURCE, beta_quality_source_sha256=BETA_SOURCE_SHA256,
        beta_quality_source_path=beta_path, beta_quality_source_verified=True,
        beta_quality_std_rad=.12, beta_low_reference_speed_m_s=.3, beta_num_envs=512,
        beta_unique_reward_reads=24008, beta_moving_reference_samples=24008*512,
        beta_low_reference_samples=0, beta_environment_reward_samples=24008*512,
        device_held_reference_phase_health=True, device_low_reference_speed_gate_health=True,
        beta_quality_is_unweighted_existing_term=True, reference_beta_not_zero_beta_is_intended=True)
    training_path = folder/"ground_contact_v2g_training_audit.json"
    training_path.write_text(json.dumps(training))
    ledger[training_path.name] = analyzer.sha256(training_path)
    audit = {"kind": analyzer.evaluator.KIND, "cohort_id": analyzer.evaluator.COHORT_ID,
        "reward_definition_version": analyzer.evaluator.REWARD_VERSION, "training_backend": "physx",
        "smoke_only": False, "trained_policy_claimed": True, "additional_training_updates": 0,
        "status": "verified", "validation_backend": analyzer.BACKENDS[engine],
        "protocol_sha256": analyzer.PROTOCOL_SHA256, "wrapper_sha256": analyzer.sha256(analyzer.evaluator.__file__),
        "normalizer_included_in_actor": True, "normalization_applications_per_inference": 1,
        "extra_normalization": False, "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "initial_policy_state_sha256": analyzer.evaluator.EXPECTED_INITIAL_POLICY_SHA256,
        "recording_sha256": analyzer.sha256(folder/raw_name), "single_wheel_support_allowed": True,
        "bilateral_support_required": False, "absolute_height_gate": False, "airborne_failure": False,
        "normalized_actor_sha256": "e"*64, "checkpoint_sha256": "d"*64,
        "legacy_evaluator_sha256": "f"*64, "generated_source_sha256": "f"*64, "cohort_input_sha256": ledger}
    report = {"ground_contact_v2g_evaluation": audit,
        "policy_export" if engine == "physx" else "contract": contract}
    if engine == "physx":
        audit.update(physical_gpu_index_requested=3, deployment_actor_sha256="a"*64)
        report.update(status="evaluated", deterministic_start_frame=0, domain_randomization_enabled=False,
            observation_noise_enabled=False, terminal_states_captured_before_auto_reset=True,
            runtime_device_audit={"status": "verified", "requested_device": "cuda:3"},
            motion_file_sha256=analyzer.INPUT_PINS["motion"],
            contact_material_audit={"live_physx_materials_verified": True, "ground_material_usd_verified": True,
                "recipe": {"training": False, "wheel_contact_friction": {"range": [.7, .7], "physx_ground_friction": 1.}},
                "encoder_bias_range_rad": [0., 0.]})
    else:
        report.update(status="completed", completed_full_reference=True, policy_steps=349, physics_steps=2792,
            hidden_resets=0, root_state_writes=1, start_reference_frame=0,
            domain_parameters_resampled_during_episode=False, policy_sha256="e"*64,
            motion_sha256=analyzer.INPUT_PINS["motion"], model_sha256=analyzer.INPUT_PINS["model"],
            contact_material_audit={"native_mapping_verified": True, "explicit_contact_pair_count": 0,
                "pair_coefficient": .7, "compiled_native_timestep_s": .0025,
                "geom_friction": [[.7, .005, .0001]]*3, "colliding_geom_ids": [0, 1, 2],
                "wheel_geom_ids": [1, 2], "geom_priority": [0, 1, 1]})
    (folder/"report.json").write_text(json.dumps(report))
    proof = {"status": "verified", "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "physics_dt_s": .0025, "policy_dt_s": .02, "recording_sha256": analyzer.sha256(folder/raw_name),
        "validation_backend": analyzer.BACKENDS[engine], "checkpoint_sha256": "d"*64,
        "report_sha256": analyzer.sha256(folder/"report.json"), "evaluation_mu": .7}
    (folder/"remote_verification.json").write_text(json.dumps(proof))
    return raw, report, proof


class AnalyzeContactV2Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tron1-contact-analysis-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_frozen_protocol_and_acceptance_unchanged(self):
        protocol = analyzer.load_protocol()
        criteria = protocol["nominal_acceptance_before_training"]
        self.assertEqual(criteria["any_wheel_support_fraction_min"], 1.)
        self.assertEqual(criteria["longest_no_support_interval_s_max"], 0.)
        self.assertEqual(criteria["core_and_whole_xy_rmse_m_max"], .15)
        self.assertEqual(criteria["core_10_to_100hz_angular_xy_rms_ratio_to_original_baseline_max"], .7)
        altered = copy.deepcopy(protocol)
        altered["nominal_acceptance_before_training"]["any_wheel_support_fraction_min"] = .9
        path = self.directory/"easier.json"; path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "acceptance bytes"):
            analyzer.load_protocol(path)

    def test_perfect_nonzero_beta_one_wheel_support_works_without_contact_count_gate(self):
        for engine in analyzer.BACKENDS:
            with self.subTest(engine=engine):
                result, trace = analyzer.recorded_metrics(recording(engine), engine, reference())
                core = result["windows"]["core"]["tracking"]
                self.assertLess(core["xy_rmse_m"], 1e-14)
                self.assertAlmostEqual(core["speed_retention"], 1.)
                self.assertLess(core["beta_rmse_deg"], 1e-12)
                self.assertEqual(core["beta_valid_samples"], 1025)
                support = result["any_wheel_ground_support"]["full_contact"]
                self.assertTrue(support["every_physical_sample_supported"])
                self.assertTrue((trace["contact_count"] == 0).all())
                self.assertFalse(result["bilateral_support_required"])
                self.assertFalse(result["windows"]["full_native"]["tracking"]["available"])

    def test_contact_threshold_strict_and_no_support_gaps_not_inferred_as_flight(self):
        raw = recording()
        raw["wheel_ground_contact_force_w_n"][100, 0, 2] = 5.
        raw["wheel_ground_contact_force_w_n"][101, 0, 2] = 0.
        result, _ = analyzer.recorded_metrics(raw, "mujoco", reference())
        support = result["any_wheel_ground_support"]["full_contact"]
        self.assertEqual(support["unsupported_physical_samples"], 2)
        self.assertAlmostEqual(support["longest_no_support_observed_s"], .005)
        self.assertFalse(support["flight_or_height_inferred"])

    def test_terminal_unsupported_sample_fails_every_sample_even_without_future_duration(self):
        raw = recording()
        index = round(6.96/.0025)
        raw["wheel_ground_contact_force_w_n"][index, 0, 2] = 0.
        candidate, _ = analyzer.recorded_metrics(raw, "mujoco", reference())
        baseline, _ = analyzer.recorded_metrics(recording(amplitude=.2), "mujoco", reference())
        contact = candidate["any_wheel_ground_support"]["full_contact"]
        self.assertEqual(contact["longest_no_support_observed_s"], 0.)
        self.assertFalse(contact["every_physical_sample_supported"])
        screen = analyzer.nominal_screen(candidate, baseline, analyzer.load_protocol(), "mujoco")
        self.assertTrue(screen["checks"]["longest_no_support_gap"])
        self.assertFalse(screen["checks"]["any_wheel_every_physical_sample"])
        self.assertFalse(screen["numeric_nominal_episode_pass"])

    def test_only_declared_contact_window_is_scored_with_native_tail_still_preserved(self):
        raw = recording()
        raw["wheel_ground_contact_force_w_n"][:8] = 0.
        raw["wheel_ground_contact_force_w_n"][2785:] = 0.
        result, _ = analyzer.recorded_metrics(raw, "mujoco", reference())
        self.assertTrue(result["any_wheel_ground_support"]["full_contact"]["every_physical_sample_supported"])
        self.assertEqual(result["windows"]["full_native"]["native_samples"], 2793)
        self.assertFalse(result["windows"]["full_native"]["tracking"]["coverage_complete"])

    def test_compact_metrics_recompute_exactly_without_files_models_or_fk(self):
        expected, trace = analyzer.recorded_metrics(recording(), "mujoco", reference())
        compact = analyzer.compact_arrays({"new_mujoco": trace}, reference())
        with patch("builtins.open", side_effect=AssertionError("No file access allowed")), \
                patch.object(Path, "read_text", side_effect=AssertionError("No path reads allowed")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("No path reads allowed")):
            actual = analyzer.recompute_compact(compact)["new_mujoco"]
        expected.pop("support_force_source")
        self.assertEqual(actual, expected)

    def test_identical_external_criteria_never_claim_overall_goal_or_qualitative_review(self):
        candidate, _ = analyzer.recorded_metrics(recording(amplitude=.1), "mujoco", reference())
        baseline, _ = analyzer.recorded_metrics(recording(amplitude=.2), "mujoco", reference())
        screen = analyzer.nominal_screen(candidate, baseline, analyzer.load_protocol(), "mujoco")
        self.assertTrue(screen["numeric_nominal_episode_pass"])
        self.assertAlmostEqual(screen["core_HF_ratio_to_original_same_engine_PPO"], .5)
        self.assertFalse(screen["overall_task_goal_complete_claimed"])
        self.assertFalse(screen["qualitative_whole_body_review_complete"])

    def test_smaller_HF_does_not_override_bad_XY_or_lost_support(self):
        raw = recording(amplitude=.01)
        raw["root_pos"][:, 0] += 1.
        raw["wheel_ground_contact_force_w_n"][500, 0, 2] = 0.
        candidate, _ = analyzer.recorded_metrics(raw, "mujoco", reference())
        baseline, _ = analyzer.recorded_metrics(recording(amplitude=.2), "mujoco", reference())
        screen = analyzer.nominal_screen(candidate, baseline, analyzer.load_protocol(), "mujoco")
        self.assertTrue(screen["checks"]["core_HF_reduction"])
        self.assertFalse(screen["checks"]["core_xy_rmse"])
        self.assertFalse(screen["checks"]["any_wheel_every_physical_sample"])
        self.assertFalse(screen["numeric_nominal_episode_pass"])

    def test_partial_bad_clock_or_negative_count_force_are_rejected(self):
        for field in ("time", "count", "force", "nan"):
            with self.subTest(field=field):
                raw = recording()
                if field == "time": raw["time_s"][10] += .00001
                if field == "count": raw["wheel_ground_contact_count"][20, 0] = -1
                if field == "force": raw["wheel_ground_contact_force_w_n"][20, 0, 2] = -.01
                if field == "nan": raw["root_com_lin_vel_w"][20, 0] = np.nan
                with self.assertRaises(ValueError): analyzer.recorded_metrics(raw, "mujoco", reference())

    def test_strict_formal_loader_accepts_both_native_schemas_without_loading_any_weight(self):
        for engine in analyzer.BACKENDS:
            with self.subTest(engine=engine):
                folder = self.directory/engine
                new_fixture(folder, engine)
                raw, evidence = analyzer.load_completed(folder, engine)
                self.assertEqual(len(raw["time_s"]), 2793)
                self.assertEqual(evidence["checkpoint_sha256"], "d"*64)
                self.assertTrue(evidence["training_artifacts_verified_on_execution_host_not_locally_rehashed"])
                self.assertFalse((folder/"model_final.pt").exists())
                self.assertFalse((folder/"policy.pt").exists())

    def test_missing_parent_proof_or_changed_bytes_are_rejected(self):
        folder = self.directory/"missing-proof"; new_fixture(folder, "mujoco")
        (folder/"remote_verification.json").rename(folder/"retained-proof.json")
        with self.assertRaisesRegex(ValueError, "artifact"): analyzer.load_completed(folder, "mujoco")
        folder = self.directory/"changed-report"; new_fixture(folder, "mujoco")
        with (folder/"report.json").open("a") as stream: stream.write(" ")
        with self.assertRaisesRegex(ValueError, "proof differs"): analyzer.load_completed(folder, "mujoco")

    def test_rehashed_bad_semantics_old_actor_and_smoke_fail_closed(self):
        variants = ("smoke", "old_actor", "beta_zero", "wrong_clock", "reset", "wrong_mu", "second_normalizer")
        for variant in variants:
            with self.subTest(variant=variant):
                folder = self.directory/variant
                _, report, proof = new_fixture(folder, "mujoco")
                audit = report["ground_contact_v2g_evaluation"]
                if variant == "smoke": audit["smoke_only"] = True
                if variant == "old_actor": audit["normalized_actor_sha256"] = analyzer.BASELINE_ACTOR
                if variant == "beta_zero": report["contract"]["task_profile"]["reward_weights"]["sideslip"] = 0.
                if variant == "wrong_clock": audit["decimation"] = 4
                if variant == "reset": report["hidden_resets"] = 1
                if variant == "wrong_mu": report["contract"]["task_profile"]["contact_friction"] = .15
                if variant == "second_normalizer": audit["extra_normalization"] = True
                (folder/"report.json").write_text(json.dumps(report))
                proof["report_sha256"] = analyzer.sha256(folder/"report.json")
                (folder/"remote_verification.json").write_text(json.dumps(proof))
                with self.assertRaises(ValueError): analyzer.load_completed(folder, "mujoco")

    def test_profile_alignment_rejects_changes_beyond_grounded_definition(self):
        _, report, _ = new_fixture(self.directory/"profile", "mujoco")
        new = report["contract"]
        baseline = copy.deepcopy(new)
        baseline["trained_task_profile"].pop("ground_contact_reward_v2")
        analyzer._compare_profiles(new, baseline)
        altered = copy.deepcopy(new)
        altered["trained_task_profile"]["reward_weights"]["speed"] = 2.
        with self.assertRaisesRegex(ValueError, "Other existing"):
            analyzer._compare_profiles(altered, baseline)

    def test_d_metrics_and_acceptance_are_exact_frozen_a_functions(self):
        self.assertIs(analyzer.recorded_metrics, analyzer.original.recorded_metrics)
        self.assertIs(analyzer.nominal_screen, analyzer.original.nominal_screen)
        self.assertIs(analyzer.recompute_compact, analyzer.original.recompute_compact)
        self.assertEqual(analyzer.load_protocol()["nominal_acceptance_before_training"],
                         analyzer.original.load_protocol()["nominal_acceptance_before_training"])

    def test_d_formal_audit_sources_are_required_and_ancestor_cores_cannot_be_substituted(self):
        for name in ("ground_contact_reward_v2.py", "ground_contact_reward_v2b.py", "ground_contact_reward_v2c.py", "ground_contact_reward_v2d.py", "ground_contact_reward_v2g.py",
                     "ppo_ground_contact_v2g.py", "train_ppo_ground_contact_v2g.py"):
            folder = self.directory/name
            new_fixture(folder, "mujoco")
            path = folder/"ground_contact_v2g_training_audit.json"
            audit = json.loads(path.read_text())
            audit["source_sha256"]["/remote/frozen/scripts/"+name] = "0"*64
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "source ledger"):
                analyzer.load_completed(folder, "mujoco")

    def test_copied_training_audit_bytes_and_actor_cannot_be_rebound(self):
        for field,value in (("checkpoint_sha256", "9"*64), ("initial_policy_state_sha256", "2"*64)):
            folder = self.directory/field; new_fixture(folder, "mujoco")
            path = folder/"ground_contact_v2g_training_audit.json"
            audit = json.loads(path.read_text()); audit[field] = value
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "initial_policy_state_sha256" if field == "initial_policy_state_sha256" else "not bound"):
                analyzer.load_completed(folder, "mujoco")

    def test_smoke_budget_or_wrong_substep_runtime_is_not_formal_evidence(self):
        for field,value in (("training_environment_steps",1536), ("runtime", 448)):
            folder = self.directory/field; new_fixture(folder, "mujoco")
            path = folder/"ground_contact_v2g_training_audit.json"
            audit = json.loads(path.read_text())
            if field == "runtime": audit["contact_runtime_audit"]["physics_samples"] = value
            else: audit[field] = value
            path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError): analyzer.load_completed(folder,"mujoco")

    def test_old_a_report_tag_and_metadata_are_not_accepted_as_b(self):
        folder = self.directory/"old-a"; _,report,proof = new_fixture(folder,"mujoco")
        report["ground_contact_v2_evaluation"] = report.pop("ground_contact_v2g_evaluation")
        (folder/"report.json").write_text(json.dumps(report))
        proof["report_sha256"] = analyzer.sha256(folder/"report.json")
        (folder/"remote_verification.json").write_text(json.dumps(proof))
        with self.assertRaises(ValueError): analyzer.load_completed(folder,"mujoco")

    def test_offline_import_source_pin_fails_closed_without_changing_any_source(self):
        with patch.dict(analyzer.IMPORT_SOURCE_PINS, {"eval_ppo_ground_contact_v2g.py": "0"*64}):
            with self.assertRaisesRegex(ValueError, "imported A/B/C/D/F/G source"):
                analyzer.load_protocol()

    def test_missing_formal_training_audit_is_not_replaced_by_a_smoke_or_report_claim(self):
        folder = self.directory/"missing-training-audit"; new_fixture(folder,"mujoco")
        (folder/"ground_contact_v2g_training_audit.json").unlink()
        with self.assertRaisesRegex(ValueError,"artifact"):
            analyzer.load_completed(folder,"mujoco")

    def test_e_training_proof_requires_d_ancestor_and_allows_zero_raw(self):
        folder=self.directory/"zero-e-raw";new_fixture(folder,"mujoco")
        path=folder/"ground_contact_v2g_training_audit.json"
        audit=json.loads(path.read_text())
        audit["contact_runtime_audit"]["mean_raw_reward"]=0.
        path.write_text(json.dumps(audit))
        actual,_=analyzer.load_training_audit(path)
        self.assertEqual(actual["contact_runtime_audit"]["mean_raw_reward"],0.)
        audit["contact_runtime_audit"]["immutable_v2d_definition_sha256"]="0"*64
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,"immutable_v2d"):
            analyzer.load_training_audit(path)

    def test_d_deficit_settings_cannot_be_relabelled_as_e(self):
        import ground_contact_reward_v2d as base_d
        folder=self.directory/"relabeled-b";new_fixture(folder,"mujoco")
        path=folder/"ground_contact_v2g_training_audit.json"
        audit=json.loads(path.read_text());audit["settings"]=base_d.validate_settings()
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,"settings"):
            analyzer.load_training_audit(path)

    def test_cli_exposes_d_download_root_and_explicit_training_metadata(self):
        args = analyzer.parser().parse_args(["--input-root",str(self.directory),
            "--training-audit",str(self.directory/"proof.json"),"--output-dir",str(self.directory/"analysis")])
        self.assertEqual(args.input_root,self.directory)
        self.assertEqual(args.training_audit,self.directory/"proof.json")
        self.assertEqual(args.protocol,analyzer.DEFAULT_PROTOCOL)


    def test_g_beta_phase_source_and_complete_window_proof_cannot_be_dropped(self):
        for key,value in [("device_held_reference_phase_health", False),
            ("beta_quality_source_verified", False), ("beta_quality_source_sha256", "0"*64),
            ("beta_unique_reward_reads", 24000), ("beta_environment_reward_samples", 1),
            ("immutable_v2f_definition_sha256", "0"*64), ("beta_quality_source_path", "/unexecuted")]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                folder=Path(temp)/"record";new_fixture(folder,"mujoco")
                path=folder/"ground_contact_v2g_training_audit.json";audit=json.loads(path.read_text())
                audit["contact_runtime_audit"][key]=value;path.write_text(json.dumps(audit))
                with self.assertRaises(ValueError): analyzer.load_training_audit(path)

@unittest.skipUnless(analyzer.BASELINE_SUMMARY.is_file() and analyzer.DEFAULT_REFERENCE.is_file()
    and (analyzer.BASELINE_ROOT/"cases/physx/mujoco/rollout.npz").is_file(),
    "Published local baseline/reference inputs are not installed")
class PublishedBaselineTest(unittest.TestCase):
    def test_each_original_same_engine_baseline_is_sha_bound_and_recomputes_exactly(self):
        ref = analyzer.metrics.reference_data(analyzer.DEFAULT_REFERENCE)
        for engine in analyzer.BACKENDS:
            raw, evidence = analyzer.load_baseline(engine)
            result, _ = analyzer.recorded_metrics(raw, engine, ref)
            for name, original in (("core", "core"), ("whole_reference_overlap", "reference_overlap"), ("full_native", "full")):
                self.assertEqual(result["windows"][name], evidence["published_row"]["windows"][original])
            with self.assertRaises(ValueError):
                analyzer.load_completed(analyzer.BASELINE_ROOT/"cases/physx"/engine, engine)

    def test_partial_pipeline_emits_no_completed_goal_claim_and_assets_free_compact_data(self):
        with tempfile.TemporaryDirectory(prefix="tron1-contact-analysis-pipeline-") as temp:
            folder = Path(temp)/"synthetic-input"; new_fixture(folder, "mujoco")
            output = Path(temp)/"analysis"
            args = SimpleNamespace(physx_dir=None, mujoco_dir=folder, protocol=analyzer.DEFAULT_PROTOCOL,
                reference=analyzer.DEFAULT_REFERENCE, baseline_root=analyzer.BASELINE_ROOT,
                baseline_summary=analyzer.BASELINE_SUMMARY, output_dir=output)
            summary = analyzer.run(args)
            self.assertEqual(len(summary["rows"]), 1)
            self.assertFalse(summary["both_PPO_validation_backends_complete"])
            self.assertFalse(summary["overall_task_goal_complete_claimed"])
            self.assertFalse(summary["qualitative_whole_body_review_complete"])
            with np.load(output/"native_states.npz", allow_pickle=False) as archive:
                recomputed = analyzer.recompute_compact({name: archive[name].copy() for name in archive.files})
            expected = copy.deepcopy(summary["rows"][0]["new_V2"]); expected.pop("support_force_source")
            self.assertEqual(recomputed["new_mujoco"], expected)
            self.assertNotIn(b"\r\n", (output/"metric_table.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "Preserve existing"): analyzer.run(args)

    def test_dual_pipeline_from_input_root_preserves_d_identity_and_all_native_samples(self):
        with tempfile.TemporaryDirectory(prefix="tron1-contact-b-dual-pipeline-") as temp:
            folder=Path(temp)/"inputs"; folder.mkdir()
            new_fixture(folder/"physx","physx"); new_fixture(folder/"mujoco","mujoco")
            output=Path(temp)/"analysis"
            args=SimpleNamespace(input_root=folder,physx_dir=None,mujoco_dir=None,
                protocol=analyzer.DEFAULT_PROTOCOL,reference=analyzer.DEFAULT_REFERENCE,
                baseline_root=analyzer.BASELINE_ROOT,baseline_summary=analyzer.BASELINE_SUMMARY,
                training_audit=folder/"physx/ground_contact_v2g_training_audit.json",output_dir=output)
            summary=analyzer.run(args)
            self.assertTrue(summary["both_PPO_validation_backends_complete"])
            self.assertEqual(summary["reward_definition_version"],"ground_contact_v2g_any_wheel")
            self.assertEqual(summary["cohort_id"],analyzer.evaluator.COHORT_ID)
            self.assertFalse(summary["overall_task_goal_complete_claimed"])
            for row in summary["rows"]:
                self.assertEqual(row["input_evidence"]["formal_training_source_evidence"]["sha256"],
                    analyzer.sha256(args.training_audit))
            with np.load(output/"native_states.npz",allow_pickle=False) as native:
                self.assertEqual(len(native["case_ids"]),4)
                for case in native["case_ids"]:
                    self.assertEqual(native[str(case)+"__time_s"].shape,(2793,))
            self.assertEqual(len((output/"metric_table.csv").read_text().splitlines()),3)


if __name__ == "__main__":
    unittest.main()

