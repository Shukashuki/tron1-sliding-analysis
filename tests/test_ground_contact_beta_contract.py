"""Pure provenance fixtures, never simulator or GPU evidence."""
import copy
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_beta_contract as contract

def audit():
    return dict(steps_started=8, steps_completed=8, unique_reward_windows=8, beta_unique_reward_reads=8,
        beta_quality_source=contract.BETA_QUALITY_SOURCE, beta_quality_source_sha256=contract.BETA_SOURCE_SHA256,
        beta_quality_std_rad=.12, beta_low_reference_speed_m_s=.3, beta_num_envs=2,
        device_held_reference_phase_health=True, device_low_reference_speed_gate_health=True,
        beta_quality_source_verified=True, cpu_fixture_only=False,
        beta_quality_is_unweighted_existing_term=True, reference_beta_not_zero_beta_is_intended=True,
        beta_moving_reference_samples=12, beta_low_reference_samples=4, beta_environment_reward_samples=16)

def test_actual_existing_beta_source_is_pinned():
    path = ROOT/"upstream/tron1-learning-from-human/training/tron1_drift.py"
    result = contract.verify_beta_source(path)
    assert result["path"] == str(path.resolve()) and result["sha256"] == contract.BETA_SOURCE_SHA256

def test_missing_edited_and_symlink_beta_sources_fail(tmp_path):
    path = tmp_path/"drift.py"
    with pytest.raises(ValueError): contract.verify_beta_source(path)
    path.write_text("# wrong beta source\n")
    with pytest.raises(ValueError): contract.verify_beta_source(path)
    link = tmp_path/"alias.py"
    link.symlink_to(ROOT/"upstream/tron1-learning-from-human/training/tron1_drift.py")
    with pytest.raises(ValueError): contract.verify_beta_source(link)

def test_runtime_requires_same_unique_windows_and_all_environment_samples():
    value = audit()
    assert contract.verify_beta_runtime(value, expected_envs=2, expected_windows=8) is value

@pytest.mark.parametrize("key,value", [
    ("steps_started", 9), ("steps_completed", 0), ("beta_unique_reward_reads", 16),
    ("beta_moving_reference_samples", -1), ("beta_moving_reference_samples", True),
    ("beta_low_reference_samples", 5), ("beta_environment_reward_samples", 15),
    ("device_held_reference_phase_health", False), ("device_low_reference_speed_gate_health", False),
    ("beta_quality_source_verified", False), ("cpu_fixture_only", True),
    ("reference_beta_not_zero_beta_is_intended", False)])
def test_incomplete_or_inconsistent_runtime_is_rejected(key, value):
    result = audit(); result[key] = value
    with pytest.raises(ValueError): contract.verify_beta_runtime(result, expected_envs=2, expected_windows=8)

@pytest.mark.parametrize("envs,windows", [(True, 8), (0, 8), (2, True), (2, 9)])
def test_expected_counts_are_positive_explicit_integer_values(envs, windows):
    with pytest.raises(ValueError): contract.verify_beta_runtime(audit(), expected_envs=envs, expected_windows=windows)

