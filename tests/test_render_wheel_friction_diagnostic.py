"""Synthetic archived signals; real Agg PNG rendering, no physics or D data."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_wheel_friction_diagnostic as renderer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(folder, *, arrays=None, summary=None):
    folder.mkdir(exist_ok=True)
    if arrays is None:
        time = np.arange(2793)*.0025
        arrays = {
            "time_s": time,
            "coulomb_reference_rho_l2": np.column_stack((.6+.3*np.sin(time), .65+.25*np.cos(time))),
            "wheel_lateral_slip_abs_m_s": np.column_stack((.02+.08*np.sin(time)**2, .01+.05*np.cos(time)**2)),
            "cone_valid_loaded": np.ones((2793, 2), dtype=bool),
            "wheel_lateral_slip_valid": np.ones((2793, 2), dtype=bool),
        }
        # Deliberately bad stored values MUST remain missing in the plot, not
        # be shown as a 999 N-ratio or a zero-imputed observation.
        arrays["cone_valid_loaded"][20:30, 0] = False
        arrays["coulomb_reference_rho_l2"][20:30, 0] = 999.
        arrays["wheel_lateral_slip_valid"][40:50, 1] = False
        arrays["wheel_lateral_slip_abs_m_s"][40:50, 1] = 999.
    if summary is None:
        summary = {
            "status": "completed_offline_diagnostic",
            "data_scope": "wheel_resultant_lower_bound",
            "actual_per_contact_engine_boundary_computed": False,
            "reward_or_acceptance_changed": False,
            "exploratory_thresholds_NOT_acceptance": {
                "near_boundary_ratio_greater_equal": .9,
                "relative_slip_speed_m_s_strict": .05,
            },
            "provenance": {"effective_pair_mu": .7,
                "cohort_id": "synthetic_archived_fixture_NOT_an_actual_experiment"},
        }
    np.savez_compressed(folder / "friction_signals.npz", **arrays)
    summary = copy.deepcopy(summary)
    summary["artifact_sha256"] = {"friction_signals.npz": digest(folder / "friction_signals.npz")}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary, arrays


def test_actual_agg_png_has_four_horizontal_panels_and_hash_bound_provenance(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source, output = tmp_path / "archive", tmp_path / "render"
    archive(source)
    input_hashes = {name: digest(source / name) for name in ("summary.json", "friction_signals.npz")}
    captured = {}
    original_subplots = plt.subplots

    def subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured.update(figure=figure, axes=axes)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", subplots)
    returned = renderer.render(source, output)
    stored = json.loads((output / "provenance.json").read_text())
    assert stored == returned
    assert stored["kind"] == "wheel_resultant_friction_diagnostic_four_horizontal_panels"
    assert stored["input_sha256"] == input_hashes
    assert stored["renderer_sha256"] == digest(renderer.__file__)
    assert stored["output_sha256"]["comparison.png"] == digest(output / "comparison.png")
    for name in ("actual_per_contact_engine_boundary_claimed", "missing_data_imputed",
                 "new_physics_or_inference", "reward_or_acceptance_changed", "visual_reviewed"):
        assert stored[name] is False
    assert {name: digest(source / name) for name in input_hashes} == input_hashes
    with Image.open(output / "comparison.png") as png:
        assert png.format == "PNG" and png.size == (2880, 720)
        pixels = np.asarray(png.convert("RGB"))
        # Each image quarter contains real plot content, not a blank panel.
        for quarter in np.array_split(pixels[100:580], 4, axis=1):
            assert quarter.std() > 5.
    axes = captured["axes"]
    assert axes.shape == (4,) and len(captured["figure"].axes) == 4
    bounds = [axis.get_position() for axis in axes]
    assert all(bounds[i].x1 < bounds[i+1].x0 for i in range(3))
    assert max(b.y0 for b in bounds)-min(b.y0 for b in bounds) < .02
    assert np.isnan(axes[0].lines[0].get_ydata()).any()
    assert np.isnan(axes[1].lines[1].get_ydata()).any()
    assert not np.any(axes[0].lines[0].get_ydata() == 999.)
    assert not np.any(axes[1].lines[1].get_ydata() == 999.)
    assert not plt.fignum_exists(captured["figure"].number)


def test_existing_output_is_preserved_without_overwrite(tmp_path):
    source, output = tmp_path / "archive", tmp_path / "render"
    archive(source); output.mkdir()
    (output / "keep.txt").write_bytes(b"existing user evidence")
    before = digest(output / "keep.txt")
    with pytest.raises(ValueError, match="Preserve existing"):
        renderer.render(source, output)
    assert digest(output / "keep.txt") == before
    assert [p.name for p in output.iterdir()] == ["keep.txt"]


def test_cutoff_legend_uses_the_actual_summary_threshold(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    source, output = tmp_path / "archive", tmp_path / "render"
    summary, arrays = archive(source)
    summary["exploratory_thresholds_NOT_acceptance"]["near_boundary_ratio_greater_equal"] = .8
    archive(source, arrays=arrays, summary=summary)
    captured = {}
    original = plt.subplots

    def capture(*args, **kwargs):
        figure, axes = original(*args, **kwargs)
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture)
    renderer.render(source, output)
    line = next(line for line in captured["axes"][0].lines if line.get_label().startswith("Exploratory"))
    assert line.get_label() == "Exploratory 0.8"
    np.testing.assert_allclose(line.get_ydata(), .8)


def test_changed_archive_bytes_fail_hash_gate_before_output_creation(tmp_path):
    source, output = tmp_path / "archive", tmp_path / "render"
    archive(source)
    with (source / "friction_signals.npz").open("ab") as stream:
        stream.write(b"changed archive bytes")
    with pytest.raises(ValueError, match="hash-bound"):
        renderer.render(source, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["sample_count", "time_offset", "nan_clock"])
def test_hash_valid_but_wrong_native_clock_is_rejected(tmp_path, mutation):
    source, output = tmp_path / "archive", tmp_path / "render"
    summary, arrays = archive(source)
    if mutation == "sample_count": arrays["time_s"] = arrays["time_s"][:-1]
    elif mutation == "time_offset": arrays["time_s"][100] += .0001
    else: arrays["time_s"][100] = np.nan
    archive(source, arrays=arrays, summary=summary)
    with pytest.raises(ValueError, match="native 400 Hz clock"):
        renderer.render(source, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["coulomb_reference_rho_l2", "wheel_lateral_slip_abs_m_s",
                                 "cone_valid_loaded", "wheel_lateral_slip_valid"])
def test_two_wheel_shape_is_required_for_every_plotted_signal(tmp_path, name):
    source, output = tmp_path / "archive", tmp_path / "render"
    summary, arrays = archive(source)
    arrays[name] = arrays[name][:, :1]
    archive(source, arrays=arrays, summary=summary)
    with pytest.raises(ValueError, match="separate L/R"):
        renderer.render(source, output)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("status", "running"),
    ("data_scope", "per_contact"), ("actual_per_contact_engine_boundary_computed", True),
    ("reward_or_acceptance_changed", True)])
def test_uncompleted_or_wrong_scope_summary_is_not_relabelled(tmp_path, field, value):
    source, output = tmp_path / "archive", tmp_path / "render"
    summary, arrays = archive(source); summary[field] = value
    archive(source, arrays=arrays, summary=summary)
    with pytest.raises(ValueError, match="hash-bound completed"):
        renderer.render(source, output)
    assert not output.exists()
