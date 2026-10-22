"""Partial-safe best-core trajectories are not mislabeled task successes."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_ppo_cem_success_paths_400hz as mod


def fixture():
    time = np.arange(2793)*.0025; rt = np.arange(349)*.02
    arrays = {"physics_dt_s": np.asarray(.0025), "reference_time_s": rt,
        "reference_position": np.column_stack((rt, rt*0., rt*0.+.7)),
        "reference_velocity": np.tile([1., 0., 0.], (349, 1)),
        "reference_quaternion": np.tile([1., 0., 0., 0.], (349, 1))}
    rows, ids = [], []
    for i, (group, mu) in enumerate(mod.SELECTED):
        key = mod.frozen.case_key(group, mu); ids.append(key)
        core, whole = {"xy_rmse_m": .1, "xy_peak_m": .2, "speed_retention": 1.}, {"xy_rmse_m": .1, "xy_peak_m": .4}
        rows.append({"id": key, "group": group, "mu": mu, "evaluated": True,
            "controller": "PPO" if group.startswith("ppo") else "MPC-CEM", "condition": "beta_on",
            "status": "completed_verified", "core_large_xy_screen": False, "overlap_large_xy_screen": True,
            "windows": {"core": {"tracking": core}, "reference_overlap": {"tracking": whole}}})
        prefix = key.replace("/", "__")+"__"
        for name, shape in mod.lowmu.SHAPES.items(): arrays[prefix+name] = np.zeros(shape)
        arrays[prefix+"time_s"] = time.copy()
        arrays[prefix+"position"] = np.column_stack((time, time*0.+i, time*0.+.7))
        arrays[prefix+"quaternion"][:, 0] = 1
        arrays[prefix+"wheel_quaternion_wxyz"][:, :, 0] = 1
        arrays[prefix+"com_velocity"][:, 0] = 1
        arrays[prefix+"wheel_center_velocity_w"][:, :, 0] = 1
    arrays["case_ids"] = np.asarray(ids)
    return {"rows": rows}, arrays


@pytest.mark.parametrize("rmse,peak,expected",[(.15,.3,"BELOW"),(.15001,.2,"LARGE"),(.1,.30001,"LARGE"),(.1,.456,"LARGE")])
def test_screen_uses_rmse_or_peak_strict_threshold(rmse, peak, expected):
    assert mod.screen({"xy_rmse_m": rmse, "xy_peak_m": peak}) == expected


def test_exact_order_keeps_late_native_samples_and_does_not_call_core_pass_success():
    summary, arrays = fixture(); panels, reference = mod.select_panels(summary, arrays)
    assert [(p["group"], p["mu"]) for p in panels] == list(mod.SELECTED)
    assert all(p["core_screen"] == "BELOW" and p["whole_screen"] == "LARGE" for p in panels)
    assert all(len(p["trace"]["time_s"]) == 2793 and p["trace"]["time_s"][-1] == 6.98 for p in panels)
    selected = mod.selected_arrays(panels, arrays)
    assert selected["case_ids"].tolist() == arrays["case_ids"].tolist()
    assert all(np.array_equal(value, selected[name]) for name, value in arrays.items())


@pytest.mark.parametrize("field,value",[("status","running"),("status","smoke_completed"),("evaluated",False),("condition","beta_off")])
def test_pending_or_other_condition_refused(field, value):
    summary, arrays = fixture(); summary["rows"][0][field] = value
    with pytest.raises(ValueError, match="pending or a different"): mod.select_panels(summary, arrays)


@pytest.mark.parametrize("mutation",["shape","clock","nan","duplicate","screen"])
def test_shape_clock_finite_identity_and_metric_screen_verified(mutation):
    summary, arrays = fixture(); prefix = arrays["case_ids"][0].replace("/","__")+"__"
    if mutation == "shape": arrays[prefix+"position"] = arrays[prefix+"position"][:-1]
    if mutation == "clock": arrays[prefix+"time_s"][1] = .01
    if mutation == "nan": arrays[prefix+"position"][3,0] = np.nan
    if mutation == "duplicate": arrays["case_ids"] = np.append(arrays["case_ids"], arrays["case_ids"][0])
    if mutation == "screen": summary["rows"][0]["overlap_large_xy_screen"] = False
    with pytest.raises(ValueError): mod.select_panels(summary, arrays)


def test_both_figures_share_exact_axes_and_preserve_after4s(tmp_path):
    summary, arrays = fixture(); panels, reference = mod.select_panels(summary, arrays)
    panels[-1]["trace"]["position"][-1, :2] = [10., -5.]
    limits = mod.lowmu.shared_limits(panels, reference)
    all_four = mod.render(panels, reference, limits, tmp_path, "four.png")
    two = mod.render([panels[1],panels[3]], reference, limits, tmp_path, "two.png")
    evidence = all_four["panels"]+two["panels"]
    for name in ("world_xlim_m","world_ylim_m","world_xticks_m","world_yticks_m","aspect"):
        assert all(p[name] == evidence[0][name] for p in evidence)
    assert all(p["native_samples_plotted"] == 2793 and p["native_final_time_s"] == 6.98 for p in evidence)
    assert all(p["core_samples_bold"] == 1025 and p["plot_cropped_at_core_end"] is False for p in evidence)
    assert all((tmp_path/name).stat().st_size > 10000 for name in ("four.png","two.png"))


def test_existing_output_refused(tmp_path):
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(SimpleNamespace(summary="absent",compact="absent",output_dir=tmp_path))


def test_selected_compact_replay_requires_explicit_correct_provenance(tmp_path):
    summary, arrays = fixture(); compact=tmp_path/"input.npz"; np.savez_compressed(compact,**arrays)
    summary["artifact_sha256"] = {"native_failure_signals.npz": "0"*64}
    source=tmp_path/"summary.json"; source.write_text(json.dumps(summary))
    args=SimpleNamespace(summary=source,compact=compact,output_dir=tmp_path/"result",trusted_provenance=None)
    with pytest.raises(ValueError,match="trusted-provenance"): mod.run(args)
    proof=tmp_path/"trusted.json"; proof.write_text(json.dumps({"summary_input":{"sha256":mod.sha256(source)},
        "selected_compact":{"sha256":mod.sha256(compact)}}));args.trusted_provenance=proof
    result=mod.run(args)
    assert result["selected_compact"]["cases"]==4 and result["new_rollouts_or_training"] is False
