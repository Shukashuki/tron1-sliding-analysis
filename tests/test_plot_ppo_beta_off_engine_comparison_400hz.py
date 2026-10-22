from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import plot_ppo_beta_off_engine_comparison_400hz as mod


def fixture():
    time = np.arange(2793)*.0025
    reference_time = np.arange(349)*.02
    reference = {"time_s": reference_time, "position": np.column_stack((reference_time, reference_time*0., reference_time*0.+.75))}
    traces = {}
    for key, displacement in (("physx", .1), ("mujoco", .2)):
        traces[key] = {"time_s": time.copy(), "physics_dt_s": .0025,
            "position": np.column_stack((time+displacement, time*0., time*0.+.75)),
            "quaternion": np.tile([1., 0., 0., 0.], (len(time), 1)),
            "angular_velocity": np.column_stack((np.sin(2*np.pi*25*time), time*0., time*0.))}
    return traces, reference


def test_native_times_overlap_no_endpoint_hold_or_clock_replacement():
    traces, reference = fixture()
    traces["mujoco"]["time_s"] = np.cumsum(np.r_[0., np.full(2792, .0025)])
    native = traces["mujoco"]["time_s"].copy()
    value = mod.diagnostic_data(traces, reference)
    for key, expected in (("physx", .1), ("mujoco", .2)):
        assert len(value[key]["native_time_s"]) == 2793
        assert len(value[key]["reference_overlap_time_s"]) == 2785
        assert value[key]["reference_overlap_time_s"][-1] == pytest.approx(6.96)
        assert value[key]["xy_error_m"] == pytest.approx(np.full(2785, expected))
        assert value[key]["common10_100_rms_rad_s"]["core"] == pytest.approx(.5, abs=.002)
        assert value[key]["first_persistent_tilt30_diagnostic"] is None
    np.testing.assert_array_equal(native, value["mujoco"]["native_time_s"])


def test_each_native_fall_preserved_independently():
    traces, reference = fixture()
    before = {}
    for key, onset in (("physx", 2.92), ("mujoco", 4.01)):
        mask = traces[key]["time_s"] >= onset
        traces[key]["quaternion"][mask] = [np.cos(np.pi/4), np.sin(np.pi/4), 0., 0.]
        before[key] = traces[key]["quaternion"].copy()
    values = mod.diagnostic_data(traces, reference)
    for key, onset in (("physx", 2.92), ("mujoco", 4.01)):
        event = values[key]["first_persistent_tilt30_diagnostic"]
        assert event["onset_s"] == pytest.approx(onset)
        assert event["confirmation_s"] == pytest.approx(onset+.1)
        np.testing.assert_array_equal(before[key], traces[key]["quaternion"])
        assert values[key]["tilt_deg"][-1] == pytest.approx(90.)


def test_rejects_old200hz_or_missing_engine():
    traces, reference = fixture()
    traces["mujoco"]["physics_dt_s"] = .005
    with pytest.raises(ValueError, match="old200Hz"):
        mod.diagnostic_data(traces, reference)
    with pytest.raises(ValueError, match="exactly"):
        mod.diagnostic_data({"physx": traces["physx"]}, reference)


def test_four_horizontal_panels_do_not_claim_four_training_cells(tmp_path):
    from PIL import Image
    traces, reference = fixture()
    path = tmp_path/"diagnostics.png"
    info = mod.draw_diagnostics(traces, reference, path)
    assert info["layout"] == [1, 4]
    assert info["new_simulation"] is False
    assert info["tilt30_used_as_failure_gate"] is False
    assert info["same_actor_different_native_plants"] is True
    assert info["identical_action_sequence_claimed"] is False
    assert info["MJWarp_trained_actor_included"] is False
    assert set(info["persistent_tilt30_diagnostic_by_engine"]) == {"physx", "mujoco"}
    assert Image.open(path).size == (3200, 832)
