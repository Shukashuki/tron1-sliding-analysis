"""Render the existing PhysX PPO comparison as one row of four panels.

Only the layout changes. Reuse the frozen plotting/animation logic and exact
recorded arrays, preserving the original two-by-two publication and hashes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

import compare_physx_ppo_low_friction as comparison


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/"results/physx_ppo_low_friction_20261021_v1"
ORIGINAL_BASE_AXES = comparison._base_axes


def horizontal_axes(arrays, summary):
    """Keep the original axes content and shared bounds; arrange it 1 x 4."""
    fig, axes = ORIGINAL_BASE_AXES(arrays, summary, figsize=(20, 6.4))
    grid = fig.add_gridspec(1, 4, left=.045, right=.992, bottom=.17,
                            top=.77, wspace=.13)
    for column, ax in enumerate(axes.flat):
        ax.set_subplotspec(grid[0, column])
        ax.set_position(grid[0, column].get_position(fig))
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=column == 0)
        if column:
            ax.set_ylabel("")
    legend = fig.legends[0]
    legend.set_ncols(6)
    legend.set_bbox_to_anchor((.5, .025), transform=fig.transFigure)
    return fig, np.asarray(axes).reshape(1, 4)


def load_source(source):
    source = Path(source)
    summary = comparison.read_json(source/"summary.json")
    comparison.require(
        summary["kind"] == "existing_physx_trained_physx_evaluated_ppo_low_friction_comparison"
        and summary["training_engine"] == summary["evaluation_engine"] == "PhysX"
        and summary["selected_mu"] == list(comparison.SELECTED_MU),
        "Require the original PhysX-only four-condition comparison")
    for name, item in summary["artifacts"].items():
        comparison.require(comparison.sha256(source/name) == item["sha256"],
                           "Source artifact hash differs: " + name)
    for name, digest in summary["analysis_source_sha256"].items():
        comparison.require(comparison.sha256(ROOT/"scripts"/name) == digest,
                           "Frozen source differs: " + name)
    with np.load(source/"selected_physx_trace.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    reference_time = arrays["reference_time_s"]
    by_mu = {row["mu"]: row for row in summary["runs"]}
    clock = None
    for mu in comparison.SELECTED_MU:
        label = comparison.prefix(mu)
        time = comparison.field(arrays, label, "time_s")
        comparison.require(time.shape == (1397,) and
                           np.allclose(time, np.arange(1397)*.005, rtol=0, atol=1e-9),
                           "Require unchanged original physical clocks")
        comparison.require(clock is None or np.array_equal(time, clock),
                           "Replay clocks differ")
        clock = time
        core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
        target = np.column_stack([np.interp(time[core], reference_time,
                     arrays["reference_position"][:, axis]) for axis in range(2)])
        error = np.linalg.norm(comparison.field(arrays, label, "position")[core, :2]-target, axis=1)
        comparison.require(int(core.sum()) == 513 and
            float(np.sqrt(np.mean(error**2))) == by_mu[mu]["reference_xy_rmse_m"] and
            float(error.max()) == by_mu[mu]["reference_xy_peak_m"],
            "Original XY metrics do not match selected recorded arrays")
    return arrays, summary


def run(args):
    output, source = Path(args.output_dir), Path(args.source_dir)
    comparison.require(not output.exists(), "Preserve existing results: choose a fresh output directory")
    arrays, parent = load_source(source)
    output.mkdir(parents=True)
    # Keep all 17 numerical results and their original bytes, not only the four
    # conditions selected for the spatial comparison.
    names = ["selected_physx_trace.npz", "metrics.csv", "friction_sensitivity.png"]
    for name in names:
        shutil.copyfile(source/name, output/name)
    with patch.object(comparison, "_base_axes", horizontal_axes):
        comparison.render_topdown(arrays, parent, output/"topdown.png")
        names.append("topdown.png")
        if not args.skip_video:
            comparison.render_video(arrays, parent, output/"topdown.mp4")
            names.append("topdown.mp4")
    summary = copy.deepcopy(parent)
    summary.update(
        kind="horizontal_layout_existing_physx_ppo_low_friction_comparison",
        source_analysis_kind=parent["kind"],
        layout={"rows": 1, "columns": 4, "condition_order_mu": list(comparison.SELECTED_MU),
                "equal_world_xy_aspect": True, "shared_original_world_bounds": True},
        source_result_sha256={"summary.json": comparison.sha256(source/"summary.json"),
                              **{name: item["sha256"] for name, item in parent["artifacts"].items()}},
        numerical_results_changed=False, render_layout_only=True)
    summary["analysis_source_sha256"][Path(__file__).name] = comparison.sha256(__file__)
    if args.skip_video:
        summary.pop("video", None)
    else:
        summary["video"]["size_pixels"] = [2000, 640]
    summary["artifacts"] = {name: {"file": name, "sha256": comparison.sha256(output/name)}
                            for name in names}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    summary = run(parser().parse_args())
    print(json.dumps({"status": "rendered_horizontal_layout", "layout": summary["layout"],
                      "numerical_results_changed": False}))
