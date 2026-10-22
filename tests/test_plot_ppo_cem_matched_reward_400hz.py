from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import plot_ppo_cem_matched_reward_400hz as mod


def fixture():
    t = np.arange(2793)*.0025; rt = np.arange(349)*.02
    ref = {"time_s": rt, "position": np.column_stack((rt, rt*0., rt*0.+.75))}
    traces = {key: {"time_s": t.copy(), "physics_dt_s": .0025,
        "position": np.column_stack((t+.1*slot, t*0., t*0.+.75)),
        "quaternion": np.tile([1., 0., 0., 0.], (2793, 1)),
        "angular_velocity": np.column_stack((np.sin(2*np.pi*25*t), t*0., t*0.))}
        for slot, key in enumerate(mod.ORDER)}
    return traces, ref


def test_native_overlap_full_record_and_HF_with_independent_four_cases():
    traces, ref = fixture()
    traces["cem_beta_off"]["time_s"] = np.cumsum(np.r_[0., np.full(2792, .0025)])
    before = traces["cem_beta_off"]["time_s"].copy()
    values = mod.diagnostic_data(traces, ref)
    for slot, key in enumerate(mod.ORDER):
        assert len(values[key]["native_time_s"]) == 2793
        assert len(values[key]["reference_overlap_time_s"]) == 2785
        assert values[key]["xy_error_m"] == pytest.approx(np.full(2785, slot*.1))
        assert values[key]["common10to100_omegaXY_RMS_rad_s"]["core"] == pytest.approx(.5, abs=.002)
    np.testing.assert_array_equal(before, values["cem_beta_off"]["native_time_s"])


def test_each_fall_diagnostic_preserved_no_state_filtering():
    traces, ref = fixture()
    for key, onset in zip(mod.ORDER, (3., 4., 4.01, 5.)):
        traces[key]["quaternion"][traces[key]["time_s"] >= onset] = [2**-.5, 2**-.5, 0., 0.]
    values = mod.diagnostic_data(traces, ref)
    for key, onset in zip(mod.ORDER, (3., 4., 4.01, 5.)):
        assert values[key]["persistent_tilt30_diagnostic"]["onset_s"] == pytest.approx(onset)
        assert values[key]["tilt_deg"][-1] == pytest.approx(90.)


def test_no_old200hz_or_incomplete_substitutes():
    traces, ref = fixture()
    with pytest.raises(ValueError, match="all four"):
        mod.diagnostic_data({key: value for key, value in traces.items() if key != "cem_beta_off"}, ref)
    traces["cem_beta_on"]["physics_dt_s"] = .005
    with pytest.raises(ValueError, match="original400Hz"):
        mod.diagnostic_data(traces, ref)


def test_four_horizontal_diagnostics_and_return_caveat(tmp_path):
    from PIL import Image
    traces, ref = fixture(); path = tmp_path/"turn.png"
    info = mod.draw_diagnostics(traces, ref, path)
    assert Image.open(path).size == (3200, 832)
    assert info["layout"] == [1, 4] and info["stage_reward_matched"] is True
    assert info["same_complete_PPO_reward_returns_claimed"] is False
    assert info["tilt_or_contact_failure_gate"] is False
