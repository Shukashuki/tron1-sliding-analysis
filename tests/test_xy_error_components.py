"""Same-time geometry from synthetic/public vectors, without FK or simulation."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_xy_error_components as xy


def quaternion_yaw(angle):
    angle = np.asarray(angle)
    return np.column_stack((np.cos(angle/2), np.zeros((len(angle), 2)), np.sin(angle/2)))


def fixture(*, stationary=False, curve=False, error=(-.2, .1)):
    rt, t = np.arange(101)*.02, np.arange(401)*.005
    rp = np.zeros((len(rt), 3))
    rp[:, 0] = 0 if stationary else rt
    if curve:
        rp[:, 1] = .5*rt**2
    reference = xy.prepare_reference(rt, rp, quaternion_yaw(np.zeros(len(rt))), [0., 2.])
    position = np.column_stack((np.interp(t, rt, rp[:, 0]), np.interp(t, rt, rp[:, 1]), np.zeros(len(t))))
    position[:, :2] += error
    velocity = np.column_stack((np.full(len(t), 0. if stationary else 1.), t if curve else np.zeros(len(t)), np.zeros(len(t))))
    trace = dict(time_s=t, physics_dt_s=.005, position=position,
                 quaternion=quaternion_yaw(np.zeros(len(t))), com_velocity=velocity,
                 angular_velocity=np.zeros((len(t), 3)))
    return reference, trace


def test_straight_signs_energy_same_validity_and_no_input_mutation():
    reference, trace = fixture()
    before = copy.deepcopy((reference, trace))
    data = xy.decompose(reference, trace)
    np.testing.assert_allclose(data["along_m"], -.2, atol=1e-13)
    np.testing.assert_allclose(data["normal_m"], .1, atol=1e-13)
    result = xy.summarize(data, [.4, 1.6])
    metrics = result["decomposition"]
    assert metrics["along_energy_fraction"] == pytest.approx(.8)
    assert metrics["normal_energy_fraction"] == pytest.approx(.2)
    assert metrics["behind_fraction"] == 1.
    assert metrics["maximum_squared_decomposition_residual_m2"] < 1e-15
    onset = result["onsets"]["behind_beyond_0.15m"]
    assert onset["source_interval"]["time_s"] == 0.
    assert onset["core"]["time_s"] == .4
    assert onset["core"]["already_active_at_window_start"]
    for old, current in zip(before, (reference, trace)):
        for name in old:
            np.testing.assert_array_equal(old[name], current[name])


def test_demonstrated_endpoint_derivative_excludes_recovery_and_tail():
    t = np.arange(151)*.02
    position = np.column_stack((t, .5*t**2, np.zeros(len(t))))
    position[t > 2.] = 1e8
    reference = xy.prepare_reference(t, position, quaternion_yaw(np.zeros(len(t))), [0., 2.])
    np.testing.assert_allclose(reference["velocity_xy_m_s"], np.column_stack((np.ones(101), t[:101])), atol=2e-13)
    np.testing.assert_array_equal(reference["original_sample_indices"], np.arange(101))
    assert reference["time_s"][-1] == 2.


def test_curved_tangent_interpolates_components_then_normalizes():
    reference, trace = fixture(curve=True)
    data = xy.decompose(reference, trace)
    t = data["time_s"]
    expected_tangent = np.column_stack((np.ones(len(t)), t))/np.sqrt(1+t*t)[:, None]
    np.testing.assert_allclose(data["along_m"], expected_tangent @ np.array([-.2, .1]), atol=1e-13)
    np.testing.assert_allclose(data["normal_m"], (expected_tangent*np.array([.1, .2])).sum(axis=1), atol=1e-13)
    np.testing.assert_allclose(data["along_m"]**2+data["normal_m"]**2, data["total_error_m"]**2, atol=1e-15)


def test_rigid_transport_uses_origin_not_com_and_does_not_modify_inputs():
    quaternion = quaternion_yaw(np.array([.2, .7, 1.2]))
    omega = np.tile([0., 0., 3.], (3, 1))
    origin = np.tile([1., .4, .2], (3, 1))
    com = origin + np.cross(omega, xy.quaternion_rotate(quaternion, xy.COM_OFFSET_M))
    old = com.copy()
    np.testing.assert_allclose(xy.root_origin_velocity(com, omega, quaternion), origin, atol=1e-15)
    np.testing.assert_array_equal(com, old)
    assert not np.allclose(com[:, :2], origin[:, :2])


def test_stationary_reference_never_fabricates_projection_or_course_event():
    reference, trace = fixture(stationary=True)
    data = xy.decompose(reference, trace)
    assert not data["projection_valid"].any()
    assert np.isnan(data["along_m"]).all() and np.isnan(data["reference_course_rad"]).all()
    result = xy.summarize(data, [.4, 1.6])
    assert result["decomposition"]["along_m"] is None
    assert result["decomposition"]["total_error_m_all_samples"]["rmse"] > 0.
    assert result["angular_timing"]["course"]["reference_core_start_deg"] is None
    assert result["angular_timing"]["course"]["first_sustained_upward_crossings"] == []
    json.dumps(result, allow_nan=False)


def test_low_actual_speed_and_vertical_forward_axis_have_unavailable_angles():
    reference, trace = fixture()
    trace["com_velocity"][:] = 0.
    trace["quaternion"][:] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    data = xy.decompose(reference, trace)
    assert data["projection_valid"].all()
    assert not data["actual_course_valid"].any() and np.isnan(data["actual_course_rad"]).all()
    assert not data["actual_yaw_valid"].any() and np.isnan(data["actual_yaw_rad"]).all()


def test_partial_tangent_validity_uses_same_samples_for_component_energy():
    reference, trace = fixture()
    reference["velocity_xy_m_s"][:50, 0] = .1
    trace["position"][trace["time_s"] < .9, :2] += 10.
    data = xy.decompose(reference, trace)
    summary = xy.summarize(data, [.4, 1.6])
    stats = summary["decomposition"]
    assert 0 < stats["projection_valid_samples"] < summary["native_samples"]
    assert stats["total_error_m_all_samples"]["rmse"] > stats["total_error_m_projection_valid"]["rmse"]
    assert stats["along_energy_fraction"]+stats["normal_energy_fraction"] == pytest.approx(1.)
    np.testing.assert_allclose(data["along_m"][data["projection_valid"]], -.2, atol=1e-13)


def test_heading_wrap_is_not_artificial_full_turn():
    reference, trace = fixture()
    reference["yaw_rad"] = np.radians(np.linspace(170., 190., len(reference["time_s"])))
    angle = np.interp(trace["time_s"], reference["time_s"], reference["yaw_rad"])+np.radians(5.)
    trace["quaternion"] = quaternion_yaw(angle)
    data = xy.decompose(reference, trace)
    np.testing.assert_allclose(np.degrees(data["yaw_error_rad"]), 5., atol=1e-12)
    np.testing.assert_allclose(np.degrees(data["actual_yaw_rad"])[[0, -1]], [175., 195.], atol=1e-12)


def test_persistent_onset_is_elapsed_time_not_sample_count_and_resets():
    time = np.arange(41)*.005
    condition = np.zeros(41, dtype=bool)
    condition[:20] = True  # .095s span, insufficient
    assert xy.persistent_onset(time, condition, [0., .2]) is None
    condition[20] = True
    event = xy.persistent_onset(time, condition, [0., .2])
    assert event["time_s"] == 0. and event["confirmation_time_s"] == .1
    condition[10] = False
    assert xy.persistent_onset(time, condition, [0., .2]) is None


@pytest.mark.parametrize("problem", ["dropped_timestamp", "bad_quaternion", "negative_threshold"])
def test_invalid_native_inputs_fail_closed(problem):
    reference, trace = fixture()
    kwargs = {}
    if problem == "dropped_timestamp":
        trace["time_s"][200:] += .005
    elif problem == "bad_quaternion":
        trace["quaternion"][100] = [0., 0., 0., 0.]
    else:
        kwargs["speed_threshold_m_s"] = -1.
    with pytest.raises(ValueError):
        xy.decompose(reference, trace, **kwargs)


def test_no_overwrite_precedes_loading_inputs(tmp_path):
    with pytest.raises(FileExistsError, match="Preserving"):
        xy.run("not-present-baseline", "not-present-residual", tmp_path)


def test_published_pair_replays_exact_total_rms_and_demonstration_mask():
    for directory in ("controller_primary_nmpc_nominal_pair_v1", "controller_primary_nmpc_residual_nominal_pair_v1"):
        summary, compact, cases, _ = xy.load_pair(ROOT/"results"/directory)
        for engine, data in cases.items():
            stats = xy.summarize(data, summary["reference_spec"]["evaluation_core_interval_s"])["decomposition"]
            assert len(data["time_s"]) == 993
            assert len(data["native_source_interval_mask"]) == 1397
            assert data["projection_valid"].all()
            assert stats["projection_valid_samples"] == 513
            assert stats["total_error_m_all_samples"]["rmse"] == pytest.approx(
                summary["windows"]["core"][engine]["reference_xy_rmse_m"], abs=1e-14)
            assert stats["along_energy_fraction"]+stats["normal_energy_fraction"] == pytest.approx(1.)
            assert stats["maximum_squared_decomposition_residual_m2"] < 1e-15
            assert data["time_s"][-1] <= 4.96+1e-9
            np.testing.assert_array_equal(data["actual_xy_m"], compact[f"{engine}_position"][:993, :2])


def test_compact_hash_tamper_rejected(tmp_path):
    source = ROOT/"results/controller_primary_nmpc_nominal_pair_v1"
    shutil.copyfile(source/"summary.json", tmp_path/"summary.json")
    (tmp_path/"pair_trace.npz").write_bytes((source/"pair_trace.npz").read_bytes()+b"tamper")
    with pytest.raises(ValueError, match="matching byte hash"):
        xy.load_pair(tmp_path)
