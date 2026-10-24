"""CPU-only source/cohort-hook tests; no Isaac, policy, GPU or training launch.

The executable source in mock-run tests is an explicit test probe, not a trainer.
Run with unittest even when pytest is absent from the local Torch environment.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_ppo_ground_contact_v2h as trainer
from ground_contact_beta_contract import BETA_SOURCE_SHA256, BETA_QUALITY_SOURCE
from test_ground_contact_velocity_contract import proof as velocity_proof


class TrainerV2Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tron1-contact-v2-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.snapshot = self.folder / "baseline"
        (self.snapshot / "scripts").mkdir(parents=True)
        self.protocol_path = self.folder / "protocol.json"
        self.protocol = {"cohort_id": "tron1_ground_contact_v2h_any_wheel_400hz_v1",
            "reward_definition_version": trainer.VERSION,
            "ground_contact_reward_v2": trainer.definition.validate_settings()}
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.output = self.folder / "output"
        self.output.mkdir()
        self.contract_file = self.output / "policy_contract.json"
        self.contract_file.write_text("{}")
        self.source_file = self.folder / "frozen_source.txt"
        self.source_file.write_text("fixture-source-not-a-model")
        self.ledger = {str(self.source_file): trainer.sha256(self.source_file)}
        self.health = {"device_finite_normal_force_health": True, "device_finite_reward_health": True,
            "steps_started": 8, "steps_completed": 8, "physics_samples": 64,
            "unique_reward_windows": 8, "force_bearing_samples_L_R_any_both": [8, 8, 8, 8]}
        self.health.update(velocity_proof())
        self.owner = SimpleNamespace(audit=lambda health=False: copy.deepcopy(self.health))
        self.beta_root = ROOT/"upstream/tron1-learning-from-human"
        self.beta_path = (self.beta_root/"training/tron1_drift.py").resolve()
        self.ledger[str(self.beta_path)] = BETA_SOURCE_SHA256
        self.health.update(beta_quality_source=BETA_QUALITY_SOURCE, beta_quality_source_sha256=BETA_SOURCE_SHA256,
            beta_quality_source_path=str(self.beta_path), beta_quality_source_verified=True, cpu_fixture_only=False,
            beta_quality_std_rad=.12, beta_low_reference_speed_m_s=.3, beta_num_envs=2,
            beta_unique_reward_reads=8, beta_moving_reference_samples=16, beta_low_reference_samples=0,
            beta_environment_reward_samples=16, device_held_reference_phase_health=True,
            device_low_reference_speed_gate_health=True, beta_quality_is_unweighted_existing_term=True,
            reference_beta_not_zero_beta_is_intended=True)
        self.args = SimpleNamespace(backend="physx", output_dir=self.output, num_envs=2)
        self.original_profile = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
        self.original_profile["action_rate_weight"] = -.5
        self.prepared_directory = self.folder / "declared"
        self.cohort = SimpleNamespace(ROOT=self.snapshot)
        self.cohort.prepare_declared_inputs = MagicMock(side_effect=self.prepare_fixture_inputs)
        self.plan = {"base": self.snapshot / "scripts/train_ppo_backend.py", "inherited_argv": [],
            "source_hashes": {}, "run_kind": "smoke"}
        self.cohort.prepare_plan = MagicMock(return_value=self.plan)
        for name in ("before_start", "configure_physx", "complete", "finalize_contract", "launch_app", "set_torch_device"):
            setattr(self.cohort, name, MagicMock())
        self.cohort.before_learn = MagicMock(side_effect=lambda env, args, manifest, runner, plan:
                                            manifest.setdefault("cohort", {}))
        self.contact = ModuleType("ppo_ground_contact_v2h")
        self.contact.configure_sensors = MagicMock()
        self.contact.reward_term = MagicMock(return_value=SimpleNamespace(weight=.4))
        self.contact.install = MagicMock(return_value=self.owner)

    def prepare_fixture_inputs(self, *args, **kwargs):
        self.prepared_directory.mkdir()
        task = self.prepared_directory / "task_profile.json"
        contract = self.prepared_directory / "contract.json"
        task.write_text(json.dumps(self.original_profile))
        contract.write_text(json.dumps({"task_profile": self.original_profile,
                                        "trained_task_profile": self.original_profile, "actor_dim": 51}))
        (self.prepared_directory / "derivation.json").write_text("{}")
        return {"task_profile": task, "contract": contract,
                "derived_input_sha256": {"task_profile": "old", "contract": "old"}}

    def invoke(self, probe):
        test_probe = ModuleType("_ground_contact_v2h_test_probe")
        test_probe.run = probe
        source = ("def main():\n"
                  "    import _ground_contact_v2h_test_probe as probe\n"
                  "    probe.run(globals())\n")
        with patch.object(trainer, "load_base", return_value=self.cohort), \
                patch.object(trainer, "source_ledger", return_value=copy.deepcopy(self.ledger)), \
                patch.object(trainer, "materialize_source", return_value=source), \
                patch.dict(sys.modules, {"_ground_contact_v2h_test_probe": test_probe,
                                         "ppo_ground_contact_v2h": self.contact}):
            trainer.run(["--base-snapshot", str(self.snapshot), "--cohort-protocol", str(self.protocol_path),
                         "--tron1-source", str(self.beta_root)])

    def test_real_frozen_source_builds_and_installs_before_any_setup_smoke(self):
        cohort = trainer.load_base(ROOT)
        source = trainer.materialize_source(cohort)
        compile(source, "<test-contact-v2-source-only>", "exec")
        self.assertEqual(source.count("accumulator = _contact_v2_install(env, _cohort_plan)"), 1)
        self.assertLess(source.index("_contact_v2_install(env, _cohort_plan)"), source.index("for _ in range(args.smoke_steps)"))
        self.assertNotIn("runner.load(", source)
        self.assertIn("runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)", source)

    def test_source_anchor_must_be_unique(self):
        (self.snapshot / "scripts/train_ppo_backend.py").write_text("fixture")
        anchor = "            accumulator = install_drift_reward_sampling(env, task)\n"
        for text in ("def main(): pass\n", "def main():\n" + anchor * 2):
            with self.subTest(text=text):
                cohort = SimpleNamespace(ROOT=self.snapshot, materialize_source=lambda value: text)
                with self.assertRaisesRegex(ValueError, "one counted"):
                    trainer.materialize_source(cohort)

    def test_frozen_base_wrapper_missing_or_edited_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"): trainer.load_base(self.snapshot)
        (self.snapshot / "scripts/train_ppo_backend_400hz.py").write_text("# not frozen\n")
        with self.assertRaisesRegex(ValueError, "differs"): trainer.load_base(self.snapshot)

    def test_source_ledger_binds_new_and_existing_source_files(self):
        ledger = trainer.source_ledger(ROOT)
        for name in ("train_ppo_ground_contact_v2h.py", "ground_contact_reward_v2.py", "ground_contact_reward_v2b.py", "ground_contact_reward_v2c.py", "ground_contact_reward_v2d.py", "ground_contact_reward_v2f.py", "ground_contact_reward_v2g.py", "ground_contact_reward_v2h.py", "ppo_ground_contact_v2g.py", "ground_contact_velocity_contract.py", "ground_contact_beta_contract.py", "ppo_ground_contact_v2h.py",
                     "train_ppo_backend.py", "train_ppo_backend_400hz.py", "height_smoothness.py"):
            path = str((ROOT / "scripts" / name).resolve())
            self.assertEqual(ledger[path], trainer.sha256(path))

    def test_only_new_reward_definition_metadata_is_added_to_declared_inputs(self):
        before = copy.deepcopy(self.original_profile)
        def probe(namespace):
            result = self.cohort.prepare_declared_inputs()
            task = json.loads(result["task_profile"].read_text())
            contract = json.loads(result["contract"].read_text())
            compare = copy.deepcopy(task)
            compare.pop("ground_contact_reward_v2")
            compare["description"] = before["description"]
            self.assertEqual(compare, before)
            self.assertEqual(task["ground_contact_reward_v2"], self.protocol["ground_contact_reward_v2"])
            self.assertEqual(contract["task_profile"], contract["trained_task_profile"])
            self.assertEqual(contract["task_profile"], task)
            self.assertEqual(contract["actor_dim"], 51)
            self.assertEqual(contract["reward_definition_version"], trainer.VERSION)
            self.assertEqual(result["derived_input_sha256"]["task_profile"], trainer.sha256(result["task_profile"]))
            self.assertEqual(result["derived_input_sha256"]["contract"], trainer.sha256(result["contract"]))
            derivation = json.loads((self.prepared_directory / "derivation.json").read_text())
            self.assertEqual(derivation["derived_inputs_sha256"], result["derived_input_sha256"])
        self.invoke(probe)

    def test_physics_config_keeps_base_recipe_and_replaces_only_existing_term(self):
        def probe(namespace):
            cfg = SimpleNamespace(scene=object(), rewards=SimpleNamespace(drift_grounded="old"))
            recipe = object()
            namespace["_cohort_configure_physx"](cfg, self.args, self.plan, recipe)
            self.cohort.configure_physx.assert_called_once_with(cfg, self.args, self.plan, recipe)
            self.contact.configure_sensors.assert_called_once_with(cfg.scene)
            self.contact.reward_term.assert_called_once_with(.4)
            self.assertEqual(cfg.rewards.drift_grounded.weight, .4)
        self.invoke(probe)

    def test_mjwarp_branch_is_rejected_before_baseline_launch_hook(self):
        def probe(namespace):
            args = SimpleNamespace(backend="mjlab")
            with self.assertRaisesRegex(ValueError, "PhysX-trained"):
                namespace["_cohort_before_start"](args, self.plan)
            self.cohort.before_start.assert_not_called()
        self.invoke(probe)

    def startup(self, namespace):
        namespace["_contact_v2_install"](object(), self.plan)
        manifest = {"initial_policy_state_sha256": trainer.EXPECTED_INITIAL_POLICY_SHA256, "cohort": {}}
        namespace["_cohort_before_learn"](object(), self.args, manifest, object(), self.plan)
        return manifest

    def test_loaded_startup_health_is_recorded_before_learning(self):
        def probe(namespace):
            manifest = self.startup(namespace)
            self.assertEqual(manifest["ground_contact_v2h"]["runtime"], self.health)
            self.assertTrue(manifest["ground_contact_v2h"]["from_scratch"])
            self.assertFalse(manifest["ground_contact_v2h"]["other14_reward_terms_changed"])
            self.assertEqual(manifest["cohort"]["reward_definition_version"], trainer.VERSION)
            self.assertTrue((self.output / "ground_contact_v2h_startup_audit.json").is_file())
        self.invoke(probe)

    def test_zero_raw_contact_reward_is_valid_without_a_deficit_penalty(self):
        self.health["mean_raw_reward"] = 0.
        def probe(namespace):
            manifest = self.startup(namespace)
            self.assertEqual(manifest["ground_contact_v2h"]["runtime"]["mean_raw_reward"], 0.)
            report = {"training_environment_steps":1536,"checkpoint_sha256":"checkpoint","exported_policy_sha256":"actor"}
            namespace["_cohort_complete"](self.args,manifest,report,self.plan)
            audit = json.loads((self.output/"ground_contact_v2h_training_audit.json").read_text())
            self.assertEqual(audit["contact_runtime_audit"]["mean_raw_reward"],0.)
        self.invoke(probe)

    def test_e_rejects_a_different_from_scratch_initial_network_before_learning(self):
        def probe(namespace):
            namespace["_contact_v2_install"](object(),self.plan)
            manifest={"initial_policy_state_sha256":"0"*64,"cohort":{}}
            with self.assertRaisesRegex(ValueError,"identical from-scratch"):
                namespace["_cohort_before_learn"](object(),self.args,manifest,object(),self.plan)
        self.invoke(probe)

    def test_startup_rejects_unhealthy_forces_reward_clock_or_no_loaded_contacts(self):
        variants = [("device_finite_normal_force_health", False), ("device_finite_reward_health", False),
            ("physics_samples", 63), ("unique_reward_windows", 7), ("steps_completed", 7),
            ("force_bearing_samples_L_R_any_both", [0, 0, 0, 0])]
        def probe(namespace):
            namespace["_contact_v2_install"](object(), self.plan)
            original = copy.deepcopy(self.health)
            for name, value in variants:
                with self.subTest(name=name):
                    self.health = copy.deepcopy(original); self.health[name] = value
                    with self.assertRaises(ValueError):
                        namespace["_cohort_before_learn"](object(), self.args, {"cohort": {}}, object(), self.plan)
            self.health = original
        self.invoke(probe)

    def test_completed_smoke_has_distinct_identity_and_no_formal_policy_claim(self):
        def probe(namespace):
            manifest = self.startup(namespace)
            report = {"training_environment_steps": 1536, "checkpoint_sha256": "checkpoint", "exported_policy_sha256": "actor"}
            namespace["_cohort_complete"](self.args, manifest, report, self.plan)
            audit = json.loads((self.output / "ground_contact_v2h_training_audit.json").read_text())
            self.assertEqual(audit["reward_definition_version"], trainer.VERSION)
            self.assertTrue(audit["smoke_only"])
            self.assertFalse(audit["trained_policy_claimed"])
            self.assertTrue(audit["from_scratch"])
            self.assertEqual(audit["new_reward_count"], 15)
            self.assertFalse(audit["other14_reward_terms_changed"])
            self.cohort.complete.assert_called_once()
        self.invoke(probe)

    def test_completion_rejects_lifetime_reward_fault_or_changed_sources(self):
        def probe(namespace):
            manifest = self.startup(namespace)
            report = {"training_environment_steps": 1536, "checkpoint_sha256": "checkpoint", "exported_policy_sha256": "actor"}
            self.health["device_finite_reward_health"] = False
            with self.assertRaisesRegex(ValueError, "collector audit"):
                namespace["_cohort_complete"](self.args, manifest, report, self.plan)
            self.health["device_finite_reward_health"] = True
            self.source_file.write_text("changed while running")
            with self.assertRaisesRegex(ValueError, "sources changed"):
                namespace["_cohort_complete"](self.args, manifest, report, self.plan)
            self.cohort.complete.assert_not_called()
        self.invoke(probe)

    def test_python_process_state_restored_after_main_raises(self):
        previous_path, previous_argv = list(sys.path), sys.argv
        def probe(namespace): raise RuntimeError("explicit test-probe failure")
        with self.assertRaisesRegex(RuntimeError, "test-probe"):
            self.invoke(probe)
        self.assertEqual(sys.path, previous_path)
        self.assertIs(sys.argv, previous_argv)

    def test_invalid_protocol_before_generated_exec_leaves_python_path_unchanged(self):
        previous_path, previous_argv = list(sys.path), sys.argv
        self.protocol["reward_definition_version"] = "not-v2"
        self.protocol_path.write_text(json.dumps(self.protocol))
        try:
            with self.assertRaisesRegex(ValueError, "new contact-reward protocol"):
                self.invoke(lambda namespace: self.fail("Must not execute generated probe"))
            self.assertEqual(sys.path, previous_path)
            self.assertIs(sys.argv, previous_argv)
        finally:
            # Protect other tests even while reporting an early-error cleanup bug.
            sys.path[:] = previous_path
            sys.argv = previous_argv


    def test_beta_phase_counts_and_provenance_are_required_before_learning(self):
        original = copy.deepcopy(self.health)
        variants = [("device_held_reference_phase_health", False), ("device_low_reference_speed_gate_health", False),
            ("beta_quality_source_verified", False), ("cpu_fixture_only", True),
            ("beta_quality_source_sha256", "0"*64), ("beta_unique_reward_reads", 7),
            ("beta_moving_reference_samples", 15), ("beta_environment_reward_samples", 15),
            ("beta_quality_std_rad", .2), ("reference_beta_not_zero_beta_is_intended", False)]
        for key, value in variants:
            with self.subTest(key=key):
                self.health = copy.deepcopy(original); self.health[key] = value
                def probe(namespace):
                    with self.assertRaises(ValueError): self.startup(namespace)
                self.invoke(probe)
        self.health = original

    def test_missing_or_changed_existing_beta_source_rejected_before_generated_main(self):
        self.beta_root = self.folder/"wrong_source"
        with self.assertRaises(ValueError):
            self.invoke(lambda namespace: self.fail("Do not execute main with missing beta source"))
        (self.beta_root/"training").mkdir(parents=True)
        (self.beta_root/"training/tron1_drift.py").write_text("# changed beta\n")
        with self.assertRaisesRegex(ValueError, "beta source SHA"):
            self.invoke(lambda namespace: self.fail("Do not execute main with edited beta source"))


if __name__ == "__main__":
    unittest.main()



def test_h_rejects_velocity_proof_before_learning():
    test = TrainerV2Test(); test.setUp()
    try:
        for key, value in (("velocity_physics_samples", 63), ("velocity_source_verified", False),
                           ("velocity_end_state_repeated", True), ("device_finite_com_vz_health", False)):
            saved = copy.deepcopy(test.health); test.health[key] = value
            def probe(namespace):
                namespace["_contact_v2_install"](object(), test.plan)
                with test.assertRaises(ValueError):
                    namespace["_cohort_before_learn"](object(), test.args,
                        {"initial_policy_state_sha256": trainer.EXPECTED_INITIAL_POLICY_SHA256, "cohort": {}}, object(), test.plan)
            test.invoke(probe); test.health = saved
    finally:
        test.doCleanups()

