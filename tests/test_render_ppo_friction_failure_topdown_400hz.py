"""Recorded-only, partial-safe low-mu topdown rendering checks."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_ppo_friction_failure_topdown_400hz as mod


def fixture():
    reference_time = np.arange(349) * .02
    arrays = {"physics_dt_s": np.asarray(.0025),
        "reference_time_s": reference_time,
        "reference_position": np.column_stack((reference_time, reference_time*0., reference_time*0.+.7)),
        "reference_velocity": np.tile([1., 0., 0.], (349, 1)),
        "reference_quaternion": np.tile([1., 0., 0., 0.], (349, 1))}
    rows, ids = [], []
    for index, mu in enumerate(mod.SELECTED_MU):
        key = mod.frozen.case_key(mod.GROUP, mu); ids.append(key)
        row = {"id": key, "mu": mu, "group": mod.GROUP, "controller": "PPO", "condition": "beta_on",
               "status": "completed_verified", "evaluated": True,
               "windows": {name: {"tracking": {"xy_rmse_m": .2, "speed_retention": .98}}
                           for name in ("core", "reference_overlap")}}
        row["windows"]["full"] = {"maximum_tilt_deg": 42.}
        rows.append(row)
        prefix = key.replace("/", "__") + "__"
        for name, shape in mod.SHAPES.items():
            arrays[prefix + name] = np.zeros(shape)
        time = np.arange(2793)*.0025
        arrays[prefix+"time_s"] = time
        arrays[prefix+"position"] = np.column_stack((time, time*0.+index, time*0.+.7))
        arrays[prefix+"wheel_pos"] = np.stack((arrays[prefix+"position"]+[0., .2, -.5],
                                              arrays[prefix+"position"]+[0., -.2, -.5]), axis=1)
        arrays[prefix+"quaternion"][:, 0] = 1.
        arrays[prefix+"wheel_quaternion_wxyz"][:, :, 0] = 1.
        arrays[prefix+"com_velocity"][:, 0] = 1.
        arrays[prefix+"wheel_center_velocity_w"][:, :, 0] = 1.
    arrays["case_ids"] = np.asarray(ids)
    return {"rows": rows}, arrays


def test_exact_four_mu_selection_keeps_native_samples_unmodified():
    summary, arrays = fixture()
    panels, reference = mod.select_panels(summary, arrays)
    assert [p["mu"] for p in panels] == [.01, .05, .1, .15]
    for p in panels:
        prefix = p["id"].replace("/", "__") + "__"
        assert np.array_equal(p["trace"]["position"], arrays[prefix+"position"])
        assert len(p["trace"]["time_s"]) == 2793
        assert p["trace"]["time_s"][-1] == 6.98
    assert reference["time_s"][-1] == 6.96


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "smoke_completed"),
                                         ("evaluated", False), ("condition", "beta_off")])
def test_pending_or_wrong_condition_cannot_be_displayed(field, value):
    summary, arrays = fixture(); summary["rows"][0][field] = value
    with pytest.raises(ValueError, match="pending or a different"):
        mod.select_panels(summary, arrays)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "shape", "clock", "nan", "negative_count", "fractional_count"])
def test_native_trace_shape_clock_finiteness_and_identity(mutation):
    summary, arrays = fixture(); prefix = str(arrays["case_ids"][0]).replace("/", "__")+"__"
    if mutation == "missing": arrays["case_ids"] = arrays["case_ids"][1:]
    if mutation == "duplicate": arrays["case_ids"] = np.append(arrays["case_ids"], arrays["case_ids"][0])
    if mutation == "shape": arrays[prefix+"position"] = arrays[prefix+"position"][:-1]
    if mutation == "clock": arrays[prefix+"time_s"][1] = .005
    if mutation == "nan": arrays[prefix+"position"][4, 1] = np.nan
    if mutation == "negative_count": arrays[prefix+"contact_count"][4, 0] = -1
    if mutation == "fractional_count": arrays[prefix+"contact_count"][4, 0] = .5
    with pytest.raises(ValueError): mod.select_panels(summary, arrays)


def test_contact_opacity_requires_both_count_and_strict_upward_force():
    trace = {"contact_count": np.array([[1, 0], [1, 1], [0, 1]]),
             "contact_force": np.array([[[0, 0, 5], [0, 0, 8]],
                                        [[0, 0, 5.0001], [0, 0, -10]],
                                        [[0, 0, 10], [0, 0, 10]]])}
    assert np.array_equal(mod.bearing_mask(trace), [[False, False], [True, False], [False, True]])


def test_velocity_arrows_are_independent_of_body_geometry_and_min_speed_inclusive():
    trace = {"body_heading_geometry_valid": np.array([False, True, True]),
        "wheel_heading_geometry_valid": np.array([[False, True], [True, False], [True, True]]),
        "signals": {"speed_m_s": np.array([.25, .249, 1.])},
        "contact_count": np.zeros((3, 2)), "contact_force": np.zeros((3, 2, 3))}
    masks = mod.glyph_masks(trace)
    assert np.array_equal(masks["velocity"], [True, False, True])
    assert np.array_equal(masks["body"], [False, True, True])
    assert not masks["wheel"][0, 0] and not masks["bearing"].any()


def test_limits_include_late_full_paths_and_wheels_without_recentring_each_panel():
    summary, arrays = fixture(); panels, reference = mod.select_panels(summary, arrays)
    panels[-1]["trace"]["position"][-1, :2] = [20., -8.]
    panels[0]["trace"]["wheel_pos"][-1, 0, :2] = [-5., 7.]
    xlim, ylim = mod.shared_limits(panels, reference)
    assert xlim[0] < -5 and xlim[1] > 20 and ylim[0] < -8 and ylim[1] > 7
    assert xlim[1]-xlim[0] == pytest.approx(ylim[1]-ylim[0])


def test_rendering_smoke_has_identical_limits_ticks_and_no_truncation(tmp_path):
    summary, arrays = fixture(); panels, reference = mod.select_panels(summary, arrays)
    result = mod.render(panels, reference, tmp_path)
    assert (tmp_path/"ppo_beta_on_low_mu_topdown.png").stat().st_size > 10000
    evidence = result["panels"]; assert len(evidence) == 4
    for field in ("world_xlim_m", "world_ylim_m", "world_xticks_m", "world_yticks_m", "aspect"):
        assert all(p[field] == evidence[0][field] for p in evidence)
    assert all(p["native_samples_plotted"] == 2793 and p["last_native_time_s"] == 6.98 for p in evidence)
    assert result["glyph_native_indices"][-1] == 2792
    assert result["height_airborne_tilt_or_beta_failure_gate"] is False


def test_existing_output_refused_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(SimpleNamespace(summary="absent.json", compact="absent.npz", output_dir=tmp_path))
