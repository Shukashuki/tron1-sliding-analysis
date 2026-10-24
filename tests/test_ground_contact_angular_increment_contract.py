"""Synthetic metadata fixtures only; no native API/runtime success is implied."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import ground_contact_angular_increment_contract as m


def fixture_proof(audit, count, windows):
    audit.update(angular_source=m.ANGULAR_SOURCE, angular_source_sha256=m.ARTICULATION_DATA_SHA256,
        angular_source_path="/synthetic/isaaclab/articulation_data.py", angular_source_verified=True,
        angular_signal="root_COM_worldXY_rad_s", angular_num_envs=count,
        angular_all8_actual_substeps=True, angular_start_is_current_physical_state=True,
        angular_velocity_is_pose_finite_difference=False, angular_end_state_repeated=False,
        angular_cache_same_refresh_not_absolute_time=True, angular_cache_refresh_health=True,
        device_finite_angular_health=True, angular_physics_samples=8*windows,
        angular_boundary_samples=windows, angular_environment_physics_samples=count*8*windows,
        angular_environment_boundary_samples=count*windows, angular_unique_reward_windows=windows,
        mean_added_angular_penalty=.1)
    return audit


def audit_fixture():
    return fixture_proof({"cpu_fixture_only":False,"steps_started":56,"steps_completed":56,
        "unique_reward_windows":56,"physics_samples":448},32,56)


def test_fixture_metadata_consistent_but_not_native_execution_evidence():
    a=audit_fixture();assert m.verify_angular_runtime(a,expected_envs=32,expected_windows=56) is a


@pytest.mark.parametrize("field,bad", [
    ("angular_source_verified",False),("angular_source_sha256","0"*64),
    ("angular_source_path","relative.py"),("cpu_fixture_only",True),
    ("angular_boundary_samples",0),("angular_physics_samples",56),
    ("angular_environment_boundary_samples",1),("angular_environment_physics_samples",448),
    ("angular_unique_reward_windows",57),("angular_start_is_current_physical_state",False),
    ("angular_velocity_is_pose_finite_difference",True),("angular_end_state_repeated",True),
    ("angular_cache_refresh_health",False),("device_finite_angular_health",False),
    ("mean_added_angular_penalty",float("nan")),("mean_added_angular_penalty",.6)])
def test_reject_relabelled_or_incomplete_proof(field,bad):
    a=copy.deepcopy(audit_fixture());a[field]=bad
    with pytest.raises(ValueError):m.verify_angular_runtime(a,expected_envs=32,expected_windows=56)


def test_source_requires_actual_class_and_not_missing_file(tmp_path):
    with pytest.raises(ValueError):m.verify_angular_data_source(object())
    with pytest.raises(ValueError):m.verify_angular_source_file(tmp_path/"absent.py")
