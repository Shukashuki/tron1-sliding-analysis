"""Layout-only regressions: keep the published recordings and analysis frozen."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_physx_ppo_low_friction_horizontal as horizontal


comparison = horizontal.comparison


@pytest.fixture(scope="module")
def source():
    return horizontal.load_source(horizontal.DEFAULT_SOURCE)


def source_hashes():
    return {path.name: comparison.sha256(path) for path in horizontal.DEFAULT_SOURCE.iterdir()
            if path.is_file()}


def test_horizontal_axes_are_one_row_in_original_order_with_equal_shared_bounds(source):
    arrays, summary = source
    fig, axes = horizontal.horizontal_axes(arrays, summary)
    try:
        assert axes.shape == (1, 4)
        assert tuple(fig.get_size_inches()) == (20., 6.4)
        low, high = comparison._bounds(arrays)
        for column, (ax, mu) in enumerate(zip(axes.flat, comparison.SELECTED_MU)):
            spec = ax.get_subplotspec()
            assert spec.get_gridspec().get_geometry() == (1, 4)
            assert (spec.rowspan.start, spec.rowspan.stop) == (0, 1)
            assert (spec.colspan.start, spec.colspan.stop) == (column, column + 1)
            assert f"mu = {mu:g}" in ax.get_title()
            assert ax.get_aspect() == 1.
            np.testing.assert_allclose(ax.get_xlim(), [low[0], high[0]], rtol=0, atol=0)
            np.testing.assert_allclose(ax.get_ylim(), [low[1], high[1]], rtol=0, atol=0)
            assert ax.get_xlabel() == "World X [m]"
            assert ax.get_ylabel() == ("World Y [m]" if column == 0 else "")
        assert "posture loss" in fig._suptitle.get_text()
        assert len(fig.legends[0].get_texts()) == 6
    finally:
        comparison._plotting().close(fig)


@pytest.mark.parametrize("skip_video", [False, True])
def test_layout_wrapper_preserves_all_numerics_bytes_and_provenance(source, tmp_path, monkeypatch, skip_video):
    arrays, parent = source
    before = source_hashes()
    original_axes = comparison._base_axes
    calls = []

    def renderer(recorded, summary, destination):
        assert comparison._base_axes is horizontal.horizontal_axes
        assert summary == parent
        assert recorded.keys() == arrays.keys()
        assert summary["selected_mu"] == [.7, .15, .1, .05]
        calls.append(Path(destination).name)
        Path(destination).write_bytes(b"layout-only-test-render")

    monkeypatch.setattr(comparison, "render_topdown", renderer)
    monkeypatch.setattr(comparison, "render_video", renderer)
    output = tmp_path / ("still" if skip_video else "video")
    arguments = ["--output-dir", str(output)] + (["--skip-video"] if skip_video else [])
    args = horizontal.parser().parse_args(arguments)
    result = horizontal.run(args)
    assert comparison._base_axes is original_axes
    assert calls == (["topdown.png"] if skip_video else ["topdown.png", "topdown.mp4"])
    assert result["runs"] == parent["runs"] and len(result["runs"]) == 17
    assert result["large_xy_mu"] == parent["large_xy_mu"] == [.01, .03, .05, .1]
    assert result["evaluation_mu"] == parent["evaluation_mu"]
    assert result["checkpoint_sha256"] == parent["checkpoint_sha256"]
    assert result["layout"] == {
        "rows": 1, "columns": 4, "condition_order_mu": [.7, .15, .1, .05],
        "equal_world_xy_aspect": True, "shared_original_world_bounds": True}
    assert result["render_layout_only"] is True
    assert result["numerical_results_changed"] is False
    for name in ("selected_physx_trace.npz", "metrics.csv", "friction_sensitivity.png"):
        assert (output / name).read_bytes() == (horizontal.DEFAULT_SOURCE / name).read_bytes()
    assert result["source_result_sha256"] == {
        "summary.json": before["summary.json"],
        **{name: item["sha256"] for name, item in parent["artifacts"].items()}}
    assert result["source_analysis_kind"] == parent["kind"]
    assert result["analysis_source_sha256"] == {
        **parent["analysis_source_sha256"],
        "render_physx_ppo_low_friction_horizontal.py": comparison.sha256(horizontal.__file__)}
    for name, artifact in result["artifacts"].items():
        assert artifact["sha256"] == comparison.sha256(output / name)
    assert json.loads((output / "summary.json").read_text()) == result
    if skip_video:
        assert "video" not in result
    else:
        assert result["video"] == {**parent["video"], "size_pixels": [2000, 640]}
    assert source_hashes() == before
    output_hashes = {path.name: comparison.sha256(path) for path in output.iterdir()}
    with pytest.raises(ValueError, match="Preserve existing results"):
        horizontal.run(args)
    assert {path.name: comparison.sha256(path) for path in output.iterdir()} == output_hashes


def test_temporary_renderer_patch_is_restored_after_exception(tmp_path, monkeypatch):
    original_axes = comparison._base_axes
    before = source_hashes()

    def broken_renderer(*_args):
        assert comparison._base_axes is horizontal.horizontal_axes
        raise RuntimeError("deliberate render failure")

    monkeypatch.setattr(comparison, "render_topdown", broken_renderer)
    args = horizontal.parser().parse_args(["--output-dir", str(tmp_path / "failed-render")])
    with pytest.raises(RuntimeError, match="deliberate render failure"):
        horizontal.run(args)
    assert comparison._base_axes is original_axes
    assert source_hashes() == before


def test_changed_published_artifact_is_rejected_before_render(tmp_path):
    tampered = tmp_path / "tampered-source"
    tampered.mkdir()
    (tampered / "summary.json").write_bytes((horizontal.DEFAULT_SOURCE / "summary.json").read_bytes())
    parent = comparison.read_json(tampered / "summary.json")
    for name in parent["artifacts"]:
        if name == "metrics.csv":
            (tampered / name).write_bytes(b"changed validation data")
        else:
            (tampered / name).symlink_to(horizontal.DEFAULT_SOURCE / name)
    with pytest.raises(ValueError, match="Source artifact hash differs: metrics.csv"):
        horizontal.load_source(tampered)


def test_changed_frozen_renderer_source_is_rejected(monkeypatch):
    original_hash = comparison.sha256

    def changed_hash(path):
        if Path(path) == ROOT / "scripts/compare_physx_ppo_low_friction.py":
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(comparison, "sha256", changed_hash)
    with pytest.raises(ValueError, match="Frozen source differs: compare_physx_ppo_low_friction.py"):
        horizontal.load_source(horizontal.DEFAULT_SOURCE)
