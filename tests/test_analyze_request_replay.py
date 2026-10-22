import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_physx_mujoco_request_replay import summarize


def sample():
    n = 81
    data = {"time_s": np.arange(n)*.0025}
    for engine in ("physx", "mujoco"):
        data[engine+"_root_position_m"] = np.zeros((n, 3))
        data[engine+"_applied_joint_torque_nm"] = np.zeros((n, 8))
        data[engine+"_wheel_ground_contact_force_w_n"] = np.zeros((n, 2, 3))
        data[engine+"_wheel_ground_contact_count"] = np.zeros((n, 2), dtype=int)
    return data


def test_native_window_and_no_flight_claim():
    result = summarize(sample())
    assert set(result["windows"]) == {"0.2"}
    assert result["windows"]["0.2"]["samples"] == 81
    assert result["windows"]["0.2"]["post_action_force_torque_samples"] == 80
    assert result["windows"]["0.2"]["physx"]["neither_wheel_load_eligible_fraction"] == 1
    assert result["same_time_xy_gap_first_observed_above_m"]["0.01"] is None


def test_first_strict_threshold_and_torque_difference():
    data = sample()
    data["mujoco_root_position_m"][4:, 0] = .02
    data["mujoco_applied_joint_torque_nm"][:, :6] = 2
    result = summarize(data)
    assert result["same_time_xy_gap_first_observed_above_m"]["0.01"] == .01
    assert result["windows"]["0.2"]["leg_actual_torque_interengine_rms_nm"] == 2


def test_reject_changed_native_clock():
    data = sample(); data["time_s"][8] += .001
    with pytest.raises(ValueError, match="times"):
        summarize(data)


def test_pre_action_torque_is_not_an_applied_action_difference():
    data = sample(); data["physx_applied_joint_torque_nm"][0] = 80
    assert summarize(data)["windows"]["0.2"]["leg_actual_torque_interengine_rms_nm"] == 0


@pytest.mark.parametrize("field", ["wheel_ground_contact_force_w_n", "wheel_ground_contact_count"])
def test_reject_nonfinite_contact_records(field):
    data = sample()
    data["mujoco_"+field] = data["mujoco_"+field].astype(float)
    data["mujoco_"+field][2] = np.nan
    with pytest.raises(ValueError, match="contact"):
        summarize(data)
