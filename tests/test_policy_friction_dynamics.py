"""Small asset-free checks for raw policy archive/clock/contact handling."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_policy_friction_dynamics import (
    append_nonwheel, audit_policy_clock, audit_xy_replay, audited_raw_case,
    reference_numeric_representation_audit,
)


def test_clock_keeps_every_original_physical_sample():
    audit_policy_clock({"time_s": np.arange(1397)*.005, "physics_dt_s": .005})


@pytest.mark.parametrize("kind", ["short", "reset", "missing", "dt"])
def test_clock_rejects_incomplete_reset_or_changed_sampling(kind):
    time = np.arange(1397)*.005; dt = .005
    if kind == "short": time = time[:-1]
    if kind == "reset": time[200:] -= 1.
    if kind == "missing": time[200] += .005
    if kind == "dt": dt = .02
    with pytest.raises(ValueError): audit_policy_clock({"time_s": time, "physics_dt_s": dt})


def test_missing_nonwheel_evidence_is_not_invented_as_zero():
    trace = {"sentinel": 3}
    assert append_nonwheel({}, trace, "isaac") is trace and trace == {"sentinel": 3}


@pytest.mark.parametrize("field", [None, "rmse", "peak", "verdict", "coverage"])
def test_original_published_grid_xy_names_and_both_error_metrics(field):
    canonical = {"coverage_complete": True, "reference_xy_rmse_m": .12,
                 "reference_xy_peak_m": .2, "large_xy_exploratory_screen": False}
    original = {"core_complete": True, "full_core_reference_xy_rmse_m": .12,
                "full_core_reference_xy_peak_m": .2, "large_reference_xy_screen": False}
    if field == "rmse": original["full_core_reference_xy_rmse_m"] = .13
    if field == "peak": original["full_core_reference_xy_peak_m"] = .3
    if field == "verdict": original["large_reference_xy_screen"] = True
    if field == "coverage": original["core_complete"] = False
    if field is None: audit_xy_replay(canonical, original)
    else:
        with pytest.raises(ValueError): audit_xy_replay(canonical, original)


@pytest.mark.parametrize("engine", ["isaac", "mujoco"])
def test_present_nonwheel_evidence_is_copied_with_native_axes(engine):
    shape = (1397, 1) if engine == "isaac" else (1397,)
    raw = {"nonwheel_ground_contact_count": np.ones(shape),
           "nonwheel_ground_contact_force_norm_n": np.full(shape, 10.)}
    result = append_nonwheel(raw, {}, engine)
    assert result["nonwheel_ground_contact_count"].shape == (1397,)
    assert np.all(result["nonwheel_ground_contact_force_norm_n"] == 10.)
    assert not np.shares_memory(raw["nonwheel_ground_contact_count"], result["nonwheel_ground_contact_count"])


@pytest.mark.parametrize("kind", ["partial", "shape", "negative", "fractional", "nan"])
def test_invalid_recorded_contact_evidence_is_rejected(kind):
    raw = {"nonwheel_ground_contact_count": np.zeros(1397),
           "nonwheel_ground_contact_force_norm_n": np.zeros(1397)}
    if kind == "partial": raw.pop("nonwheel_ground_contact_force_norm_n")
    if kind == "shape": raw["nonwheel_ground_contact_count"] = np.zeros((1397, 1))
    if kind == "negative": raw["nonwheel_ground_contact_force_norm_n"][0] = -1
    if kind == "fractional": raw["nonwheel_ground_contact_count"][0] = .5
    if kind == "nan": raw["nonwheel_ground_contact_count"][0] = np.nan
    with pytest.raises(ValueError): append_nonwheel(raw, {}, "mujoco")


def create_case(folder):
    result = {}
    for engine, name in (("isaac", "physics_trajectory.npz"), ("mujoco", "rollout.npz")):
        for role, filename in ((engine+"_trace", name), (engine+"_report", "report.json")):
            path = folder/engine/filename; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(role.encode())
            result[role+"_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_all_four_raw_files_bound_to_published_capture(tmp_path):
    expected = create_case(tmp_path/"PPO/mu0p100")
    files, hashes = audited_raw_case(tmp_path, "PPO", .1, expected)
    assert len(files) == len(hashes) == 4
    assert all(hashes[name] == expected[name+"_sha256"] for name in files)


@pytest.mark.parametrize("role", ["isaac_trace", "isaac_report", "mujoco_trace", "mujoco_report"])
def test_each_changed_raw_archive_is_rejected(tmp_path, role):
    expected = create_case(tmp_path/"SAC/mu0p100")
    expected[role+"_sha256"] = "0"*64
    with pytest.raises(ValueError): audited_raw_case(tmp_path, "SAC", .1, expected)


def numeric_reference():
    times = np.array([0., 1., 2.])
    positions = np.array([[.123456789, .321654987, .75], [1.23456789, .765432109, .75],
                          [2.34567891, -.543210987, .75]], dtype=np.float64)
    motion = {"body_names": np.array(["base_Link"]), "body_pos_w": positions.astype(np.float32)[:, None],
              "time_s": times.copy()}
    robot = {"root_target_position_m": positions.copy(), "time_s": times.copy()}
    return motion, robot


def test_reference_representation_is_exact_verified_cast_on_original_timestamps():
    motion, robot = numeric_reference()
    proof = reference_numeric_representation_audit(motion, robot)
    expected = np.linalg.norm(motion["body_pos_w"][:, 0, :2].astype(float)-robot["root_target_position_m"][:, :2], axis=1).max()
    assert proof["analytic_metric_change_bound_m"] == expected > 0
    assert proof["float32_cast_bitwise_equal"] is True and proof["identical_reference_timestamps"] is True
    assert proof["translation_rotation_or_time_warp_applied"] is False
    assert proof["historical_reference_or_metrics_modified"] is False


@pytest.mark.parametrize("change", ["translation", "rotation", "time_shift", "time_warp", "float64_motion", "missing_base", "nan"])
def test_representation_proof_rejects_alignment_cast_or_clock_substitutions(change):
    motion, robot = numeric_reference()
    if change == "translation": motion["body_pos_w"][:, 0, 0] += .1
    elif change == "rotation": motion["body_pos_w"][:, 0, :2] = motion["body_pos_w"][:, 0, :2][:, ::-1]
    elif change == "time_shift": motion["time_s"] += .01
    elif change == "time_warp": motion["time_s"][1] += .01
    elif change == "float64_motion": motion["body_pos_w"] = motion["body_pos_w"].astype(np.float64)
    elif change == "missing_base": motion["body_names"] = np.array(["other"])
    else: robot["root_target_position_m"][1, 0] = np.nan
    with pytest.raises(ValueError): reference_numeric_representation_audit(motion, robot)


def numeric_metrics():
    motion, robot = numeric_reference(); proof = reference_numeric_representation_audit(motion, robot)
    query = np.linspace(0., 2., 17)
    original = np.column_stack([np.interp(query, robot["time_s"], robot["root_target_position_m"][:, j]) for j in (0, 1)])
    exported = np.column_stack([np.interp(query, motion["time_s"], motion["body_pos_w"][:, 0, j]) for j in (0, 1)])
    observed = original+[.05, -.03]
    def metrics(reference):
        error = np.linalg.norm(observed-reference, axis=1)
        return float(np.sqrt(np.mean(error**2))), float(error.max())
    rmse, peak = metrics(exported); old_rmse, old_peak = metrics(original)
    canonical = {"coverage_complete": True, "samples_in_window": 17,
        "reference_xy_rmse_m": rmse, "reference_xy_peak_m": peak, "large_xy_exploratory_screen": False}
    legacy = {"core_complete": True, "observed_core_samples": 17,
        "full_core_reference_xy_rmse_m": old_rmse, "full_core_reference_xy_peak_m": old_peak,
        "large_reference_xy_screen": False}
    return canonical, legacy, proof, observed


def test_triangle_bound_retains_both_metric_versions_without_overwriting_history():
    canonical, legacy, proof, observed = numeric_metrics()
    before = legacy.copy()
    result = audit_xy_replay(canonical, legacy, reference_audit=proof, observed_xy=observed)
    assert legacy == before
    assert result["legacy_robot_reference_float64_metrics"]["reference_xy_rmse_m"] == legacy["full_core_reference_xy_rmse_m"]
    assert result["canonical_motion_float32_metrics"]["reference_xy_rmse_m"] == canonical["reference_xy_rmse_m"]
    assert all(value <= result["combined_metric_change_bound_m"] for value in result["absolute_metric_changes_m"].values())
    assert result["xy_screen_unchanged"] is True and result["historical_metrics_overwritten"] is False


@pytest.mark.parametrize("change", ["rmse_over_bound", "peak_over_bound", "different_samples", "arbitrary_tolerance", "screen_flip", "missing_observed"])
def test_numeric_replay_cannot_hide_out_of_bound_changes_or_screen_disagreement(change):
    canonical, legacy, proof, observed = numeric_metrics()
    if change == "rmse_over_bound": canonical["reference_xy_rmse_m"] += .001
    elif change == "peak_over_bound": canonical["reference_xy_peak_m"] += .001
    elif change == "different_samples": legacy["observed_core_samples"] = 16
    elif change == "arbitrary_tolerance": proof = {"analytic_metric_change_bound_m": 100.}
    elif change == "screen_flip": canonical["large_xy_exploratory_screen"] = True
    else: observed = None
    with pytest.raises(ValueError): audit_xy_replay(canonical, legacy, reference_audit=proof, observed_xy=observed)


def test_absent_reference_proof_does_not_silently_relax_previous_xy_check():
    canonical, legacy, _, _ = numeric_metrics()
    legacy["full_core_reference_xy_rmse_m"] = canonical["reference_xy_rmse_m"]+1e-12
    legacy["full_core_reference_xy_peak_m"] = canonical["reference_xy_peak_m"]
    with pytest.raises(ValueError, match="proven reference-representation bound"):
        audit_xy_replay(canonical, legacy)
