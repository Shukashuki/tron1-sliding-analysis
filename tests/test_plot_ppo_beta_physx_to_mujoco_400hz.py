from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import plot_ppo_beta_physx_to_mujoco_400hz as mod


def fixture():
    time = np.arange(2793)*.0025
    reference_time = np.arange(349)*.02
    reference = {"time_s": reference_time, "position": np.column_stack((reference_time, reference_time*0., reference_time*0.+.75))}
    traces = {}
    for key, displacement in (("beta_on", .1), ("beta_off", .2)):
        traces[key] = {"time_s": time.copy(), "physics_dt_s": .0025,
            "position": np.column_stack((time+displacement, time*0., time*0.+.75)),
            "quaternion": np.tile([1., 0., 0., 0.], (len(time), 1)),
            "angular_velocity": np.column_stack((np.sin(2*np.pi*25*time), time*0., time*0.))}
    return traces, reference


def test_native_data_and_overlap_do_not_hold_reference_endpoint():
    traces, reference = fixture()
    value = mod.diagnostic_data(traces, reference)
    for key, expected in (("beta_on", .1), ("beta_off", .2)):
        assert len(value[key]["native_time_s"]) == 2793
        assert len(value[key]["reference_overlap_time_s"]) == 2785
        assert value[key]["reference_overlap_time_s"][-1] == pytest.approx(6.96)
        assert value[key]["xy_error_m"] == pytest.approx(np.full(2785, expected))
        assert value[key]["common10_100_rms_rad_s"]["core"] == pytest.approx(.5, abs=.002)
        assert value[key]["first_persistent_tilt30_diagnostic"] is None


def test_postcore_persistent_tilt_is_diagnostic_and_uses_unmodified_poses():
    traces, reference = fixture()
    key = "beta_off"
    mask = traces[key]["time_s"] >= 4.01
    traces[key]["quaternion"][mask] = [np.cos(np.pi/4), np.sin(np.pi/4), 0., 0.]
    previous = traces[key]["quaternion"].copy()
    value = mod.diagnostic_data(traces, reference)[key]
    assert value["first_persistent_tilt30_diagnostic"]["onset_s"] == pytest.approx(4.01)
    assert value["first_persistent_tilt30_diagnostic"]["confirmation_s"] == pytest.approx(4.11)
    assert np.array_equal(previous, traces[key]["quaternion"])
    assert value["tilt_deg"][mask] == pytest.approx(np.full(mask.sum(), 90.))


def test_rejects_old200hz_or_missing_condition():
    traces, reference = fixture()
    traces["beta_off"]["physics_dt_s"] = .005
    with pytest.raises(ValueError, match="old200Hz"):
        mod.diagnostic_data(traces, reference)
    with pytest.raises(ValueError, match="exactly"):
        mod.diagnostic_data({"beta_on": traces["beta_on"]}, reference)


def test_draws_four_horizontal_diagnostics(tmp_path):
    from PIL import Image
    traces, reference = fixture()
    path = tmp_path/"diagnostics.png"
    info = mod.draw_diagnostics(traces, reference, path)
    assert info["layout"] == [1, 4]
    assert info["new_simulation"] is False
    assert info["tilt30_used_as_failure_gate"] is False
    assert Image.open(path).size == (3200, 832)
