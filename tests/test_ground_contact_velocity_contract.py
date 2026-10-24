"""Metadata rejection tests; passing fixtures do not establish GPU evidence."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_velocity_contract as contract


def proof():
    return dict(velocity_source=contract.VELOCITY_SOURCE,
        velocity_source_sha256=contract.ARTICULATION_DATA_SHA256,
        velocity_source_path="/fixture/isaaclab/articulation_data.py",
        velocity_source_verified=True, cpu_fixture_only=False,
        velocity_signal="base_COM_worldZ_m_s", velocity_all8_actual_substeps=True,
        velocity_is_finite_difference=False, velocity_end_state_repeated=False,
        velocity_cache_epoch_same_refresh_not_absolute_sim_time=True,
        velocity_cache_refresh_health=True, device_finite_com_vz_health=True,
        velocity_num_envs=2, steps_started=8, steps_completed=8,
        unique_reward_windows=8, velocity_unique_reward_windows=8,
        velocity_physics_samples=64, physics_samples=64,
        velocity_environment_samples=128, mean_added_velocity_penalty=.1)


class VelocityContractTest(unittest.TestCase):
    def test_complete_runtime_metadata_and_window_count(self):
        audit = proof()
        self.assertIs(contract.verify_velocity_runtime(audit, expected_envs=2, expected_windows=8), audit)

    def test_missing_or_mutated_provenance_and_sample_fields_are_rejected(self):
        original = proof()
        for key in original:
            for missing in (True, False):
                with self.subTest(key=key, missing=missing):
                    audit = copy.deepcopy(original)
                    if missing: del audit[key]
                    else:
                        value = original[key]
                        audit[key] = (not value if isinstance(value, bool) else
                            value+1 if isinstance(value, int) else float("nan")
                            if isinstance(value, float) else "wrong")
                    with self.assertRaises(ValueError): contract.verify_velocity_runtime(audit, expected_envs=2)

    def test_explicit_cpu_fixture_cannot_pass_production_proof(self):
        audit = proof(); audit["cpu_fixture_only"] = True
        with self.assertRaises(ValueError): contract.verify_velocity_runtime(audit, expected_envs=2)

    def test_invalid_penalty_or_expected_count_is_rejected(self):
        for value in (-.1, .501, True, float("inf"), "0.1"):
            audit = proof(); audit["mean_added_velocity_penalty"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.verify_velocity_runtime(audit, expected_envs=2)
        for count in (0, -1, True, 2., 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                contract.verify_velocity_runtime(proof(), expected_envs=count)
        with self.assertRaises(ValueError):
            contract.verify_velocity_runtime(proof(), expected_envs=2, expected_windows=7)

    def test_fake_data_and_unpinned_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual native"):
            contract.verify_velocity_data_source(SimpleNamespace())
        with tempfile.TemporaryDirectory(prefix="tron1-vz-contract-") as folder:
            source = Path(folder)/"fixture.py"; source.write_text("# fake API\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                contract.verify_velocity_source_file(source)
            link = Path(folder)/"link.py"; link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "direct existing"):
                contract.verify_velocity_source_file(link)


if __name__ == "__main__": unittest.main()
