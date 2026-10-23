"""Offline-only array integrity and explicit incomplete-finalization scope."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import recover_ground_force_api_probe_v1 as mod


def fixture():
    n = 2792
    f = np.zeros((n, 2, 3)); f[..., 2] = 9.
    arrays = {name: f.copy() for name in ("point_sum_w_n", "legacy32_direct_matrix_w_n", "filtered64_matrix_w_n", "legacy_recorder_force_w_n")}
    arrays.update(counter=np.arange(1, n+1), sim_index=np.arange(11, n+11), sim_time=np.arange(1, n+1)*.0025,
        time_s=np.arange(1, n+1)*.0025, point_counts=np.full((n, 2), 3), point_starts=np.full((n, 2), 2),
        point_buffer_lengths=np.full((n, 2), 32), legacy32_capacity=np.full((n, 2), 32),
        point_counts_gt5N=np.zeros((n, 2)), legacy_recorder_count_gt5N=np.zeros((n, 2)))
    force = np.zeros((n+1, 1, 2, 3)); force[1:, 0] = f
    raw = {"simulation_step_counter": np.arange(n+1), "time_s": np.arange(n+1)*.0025,
        "wheel_ground_contact_force_w_n": force, "wheel_ground_contact_count": np.zeros((n+1, 1, 2))}
    return arrays, raw


def test_exact_full_counter_and_force_recovery_without_live_metadata():
    arrays, raw = fixture(); result = mod.validate_arrays(arrays, raw)
    assert result["samples"] == 2792 and result["legacy32_possible_saturation"] is False
    assert result["comparisons"]["point_sum_vs_filtered64"]["max_abs_vector_difference_n"] == 0.
    assert "mass" not in str(result)


@pytest.mark.parametrize("fault", ("clock", "counter", "missing", "nan", "counts", "recorded_force"))
def test_invalid_or_misaligned_saved_arrays_rejected(fault):
    arrays, raw = fixture()
    if fault == "clock": arrays["time_s"][4] += .001
    if fault == "counter": arrays["counter"][4] += 1
    if fault == "missing": arrays["sim_time"] = arrays["sim_time"][:-1]
    if fault == "nan": arrays["point_sum_w_n"][0, 0, 0] = np.nan
    if fault == "counts": arrays["point_counts"][0, 0] = 33
    if fault == "recorded_force": raw["wheel_ground_contact_force_w_n"][1, 0, 0, 2] += 1
    with pytest.raises(ValueError): mod.validate_arrays(arrays, raw)


def test_existing_recovery_directory_rejected_before_input_reads(tmp_path):
    with pytest.raises(ValueError, match="Fresh recovery"):
        mod.run(SimpleNamespace(output_dir=tmp_path))


def test_never_synthesizes_success_report_or_mass_usd():
    source = Path(mod.__file__).read_text()
    assert mod.STATUS == "FAILED_FINALIZATION_ARRAYS_RECOVERED"
    assert '"finalization_completed": False' in source
    assert '"original_A_formal_result": False' in source
    assert 'write_json(output/"recovery_audit.json"' in source
    assert 'write_json(output/"report.json"' not in source
    assert mod.sha256(mod.probe.__file__) == mod.V1_SHA


def test_full_offline_recovery_writes_only_honest_independent_audit(tmp_path):
    arrays, raw = fixture()
    n = 2793
    raw.update(physics_dt_s=.0025, policy_dt_s=.02, valid_mask=np.ones((n, 1), dtype=bool),
        action_was_applied=np.r_[False, np.ones(n-1, dtype=bool)][:, None])
    for name, width in (("action_observation", 51), ("actions_raw", 8), ("actions_clipped", 8),
        ("root_position_m", 3), ("root_quaternion_wxyz", 4), ("joint_position_rad", 8),
        ("joint_velocity_rad_s", 8), ("applied_joint_torque_nm", 8)):
        raw[name] = np.zeros((n, 1, width))
    raw["root_quaternion_wxyz"][..., 0] = 1.
    np.savez_compressed(tmp_path/"physics_trajectory.npz", **raw)
    np.savez_compressed(tmp_path/"probe_arrays.npz", **arrays)
    source = {"wrapper_sha256": mod.V1_SHA, "kind": mod.probe.base.KIND,
        "normalized_actor_sha256": mod.probe.ACTOR_SHA, "checkpoint_sha256": mod.probe.CHECKPOINT_SHA,
        "recording_sha256": mod.sha256(tmp_path/"physics_trajectory.npz"), "native_states": 2793,
        "policy_decisions": 349, "validation_backend": "PhysX"}
    mod.write_json(tmp_path/"evaluation_source_audit.json", source)
    (tmp_path/"driver.log").write_text("FileExistsError: evaluation_source_audit.json already exists\n")
    args = SimpleNamespace(raw=tmp_path/"physics_trajectory.npz", probe_arrays=tmp_path/"probe_arrays.npz",
        source_audit=tmp_path/"evaluation_source_audit.json", failure_log=tmp_path/"driver.log", output_dir=tmp_path/"recovered")
    result = mod.run(args)
    assert result["status"] == mod.STATUS and result["finalization_completed"] is False
    assert {p.name for p in args.output_dir.iterdir()} == {"recovery_audit.json"}
    assert mod.read_json(args.source_audit) == source
    assert "mass_and_gravity" not in result["diagnostics"]
