"""Show recorded best-core paths without calling a core pass task success.

Original native trajectories only. Whole-path screening uses reference overlap
0–6.96s; the plotted physical trajectories continue unmodified through6.98s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_cem_friction_failure_400hz as frozen
import render_ppo_friction_failure_topdown_400hz as lowmu
from ppo_backend_400hz_contract import require, sha256, write_json

SELECTED = (("ppo_beta_on", .7), ("cem_beta_on", .7),
            ("ppo_beta_on", .15), ("cem_beta_on", .15))
GLYPH_TIMES = (0., 1.44, 2.5, 4., 5.5, 6.5)


def screen(tracking):
    require(all(np.isfinite(tracking[name]) and tracking[name] >= 0
                for name in ("xy_rmse_m", "xy_peak_m")), "Invalid XY screen metrics")
    return "LARGE" if tracking["xy_rmse_m"] > .15 or tracking["xy_peak_m"] > .30 else "BELOW"


def select_panels(summary, arrays):
    require(float(arrays["physics_dt_s"]) == .0025, "Require original400Hz compact data")
    ids = np.asarray(arrays["case_ids"]).tolist()
    require(len(ids) == len(set(ids)), "Duplicate compact ids")
    lookup = {}
    for row in summary["rows"]:
        require(row["id"] not in lookup, "Duplicate summary ids")
        lookup[row["id"]] = row
    panels = []
    for group, mu in SELECTED:
        key = frozen.case_key(group, mu)
        require(key in lookup and key in ids, "Selected completed case missing: " + key)
        row = lookup[key]
        require(row["evaluated"] is True and row["status"] in frozen.COMPLETE
                and row["group"] == group and row["mu"] == mu
                and row["condition"] == "beta_on"
                and row["controller"] == ("PPO" if group.startswith("ppo") else "MPC-CEM"),
                "Selected case is pending or a different controller: " + key)
        prefix = key.replace("/", "__") + "__"
        trace = {}
        for name, shape in lowmu.SHAPES.items():
            value = np.asarray(arrays[prefix + name])
            require(value.shape == shape and value.dtype.kind in "fiu" and np.isfinite(value).all(),
                    "Invalid native shape/data: " + name)
            trace[name] = value.copy()
        trace["physics_dt_s"] = .0025
        frozen.clock_audit(trace)
        trace["signals"] = frozen.heading_signals(trace["quaternion"], trace["com_velocity"],
            trace["wheel_quaternion_wxyz"], trace["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = frozen.metrics.heading_geometry_masks(trace)
        core, whole = (row["windows"][name]["tracking"] for name in ("core", "reference_overlap"))
        core_screen, whole_screen = screen(core), screen(whole)
        require(row["core_large_xy_screen"] == (core_screen == "LARGE")
                and row["overlap_large_xy_screen"] == (whole_screen == "LARGE"),
                "Stored XY screen disagrees with metrics")
        panels.append({"id": key, "group": group, "mu": mu, "row": row, "trace": trace,
                       "core_screen": core_screen, "whole_screen": whole_screen})
    reference = {name[len("reference_"):]: np.asarray(value).copy()
                 for name, value in arrays.items() if name.startswith("reference_")}
    for name, shape in {"time_s": (349,), "position": (349, 3),
                        "velocity": (349, 3), "quaternion": (349, 4)}.items():
        require(name in reference and reference[name].shape == shape
                and np.isfinite(reference[name]).all(), "Invalid reference shape: " + name)
    require(np.allclose(reference["time_s"], np.arange(349)*.02, rtol=0, atol=1e-10),
            "Require original reference clock")
    return panels, reference


def selected_arrays(panels, arrays):
    result = {name: np.asarray(value).copy() for name, value in arrays.items()
              if name.startswith("reference_") or name == "physics_dt_s"}
    result["case_ids"] = np.asarray([p["id"] for p in panels])
    for panel in panels:
        prefix = panel["id"].replace("/", "__") + "__"
        for name in lowmu.SHAPES:
            result[prefix + name] = panel["trace"][name].copy()
    return result


def render(panels, reference, limits, destination, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xlim, ylim = limits
    figure, axes = plt.subplots(1, len(panels), figsize=(5.2*len(panels), 6.9), squeeze=False)
    indices = np.rint(np.asarray(GLYPH_TIMES)/.0025).astype(int)
    evidence = []
    for axis, panel in zip(axes[0], panels):
        trace, row = panel["trace"], panel["row"]
        color = "#258177" if panel["group"].startswith("ppo") else "#dc8624"
        position, signals, masks = trace["position"], trace["signals"], lowmu.glyph_masks(trace)
        core_mask = (trace["time_s"] >= 1.44-1e-9) & (trace["time_s"] <= 4.+1e-9)
        axis.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color=".55", lw=1.4)
        axis.plot(position[:, 0], position[:, 1], color=color, alpha=.4, lw=1.5)
        axis.plot(position[core_mask, 0], position[core_mask, 1], color=color, lw=2.7)
        for index in indices:
            for name, angle, arrow_color in (
                ("body", signals["body_heading_rad"][index], "#1464b4"),
                ("velocity", signals["course_rad"][index], "#23934e")):
                if masks[name][index]:
                    delta = .14*np.array([np.cos(angle), np.sin(angle)])
                    axis.annotate("", xy=position[index, :2]+delta, xytext=position[index, :2],
                        arrowprops={"arrowstyle": "->", "color": arrow_color, "linewidth": 1.1})
            for side, wheel_color in enumerate(("#d62728", "#9467bd")):
                if masks["wheel"][index, side]:
                    angle = signals["wheel_rolling_plane_heading_rad"][index, side]
                    delta = .07*np.array([np.cos(angle), np.sin(angle)])
                    ends = trace["wheel_pos"][index, side, :2]+np.array([-delta, delta])
                    axis.plot(ends[:, 0], ends[:, 1], color=wheel_color, lw=1.2,
                              alpha=.9 if masks["bearing"][index, side] else .18)
        axis.plot(*position[1600, :2], "o", color=color, ms=5, zorder=4)
        axis.plot(*position[-1, :2], "s", color=color, ms=5, zorder=4)
        axis.plot(*reference["position"][200, :2], "o", color=".5", ms=4)
        core, whole = (row["windows"][name]["tracking"] for name in ("core", "reference_overlap"))
        label = "PPO + beta" if panel["group"].startswith("ppo") else "MPC-CEM + beta"
        axis.set(title=f"{label} | mu={panel['mu']:g}\n"
            f"Core XY screen: {panel['core_screen']} | whole: {panel['whole_screen']}\n"
            f"XY RMSE core / whole: {core['xy_rmse_m']:.3f} / {whole['xy_rmse_m']:.3f} m\n"
            f"Whole peak: {whole['xy_peak_m']:.3f} m | core speed: {100*core['speed_retention']:.0f}%",
            xlabel="World X (m)", ylabel="World Y (m)", xlim=xlim, ylim=ylim)
        axis.set_xticks(np.linspace(*xlim, 5)); axis.set_yticks(np.linspace(*ylim, 5))
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        axis.title.set_fontsize(9.5)
        evidence.append({"id": panel["id"], "core_xy_screen": panel["core_screen"],
            "whole_xy_screen": panel["whole_screen"], "world_xlim_m": list(axis.get_xlim()),
            "world_ylim_m": list(axis.get_ylim()), "world_xticks_m": axis.get_xticks().tolist(),
            "world_yticks_m": axis.get_yticks().tolist(), "aspect": float(axis.get_aspect()),
            "native_samples_plotted": len(position), "native_final_time_s": float(trace["time_s"][-1]),
            "core_samples_bold": int(core_mask.sum()), "plot_cropped_at_core_end": False})
    handles = [Line2D([], [], color=".55", ls="--", label="Reference"),
        Line2D([], [], color=".3", lw=2.7, label="Core1.44–4s (bold)"),
        Line2D([], [], color=".3", alpha=.4, label="Full0–6.98s path"),
        Line2D([], [], color="#1464b4", marker=">", label="Body heading"),
        Line2D([], [], color="#23934e", marker=">", label="COM direction (speed >=0.25m/s)"),
        Line2D([], [], color="#d62728", label="Left wheel axial plane"),
        Line2D([], [], color="#9467bd", label="Right wheel axial plane"),
        Line2D([], [], color=".3", marker="o", ls="", label="4s marker"),
        Line2D([], [], color=".3", marker="s", ls="", label="6.98s endpoint")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3, fontsize=8, frameon=False)
    figure.suptitle("Recorded400Hz | native CPU MuJoCo | identical world axes | N=1/cell\nCore screen BELOW is NOT whole-task success; CEM knows mu, PPO does not.", fontsize=11, y=.98)
    figure.text(.5, .035, "Whole=reference-overlap0–6.96s. LARGE: XY RMSE>0.15m OR peak>0.30m; no height/contact/tilt/beta failure gate.", ha="center", fontsize=8)
    figure.text(.5, .012, "Opaque wheel ticks: contact count>0 AND upward Fz>5N; faint is not flight. No smoothing, pose/time alignment or dynamics.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0., .21, 1., .90))
    figure.savefig(Path(destination)/filename, dpi=160); plt.close(figure)
    return {"panels": evidence, "glyph_times_s": list(GLYPH_TIMES),
            "glyph_native_indices": indices.tolist(), "poses_or_times_aligned_or_smoothed": False,
            "wheel_bearing_force_n_strict": 5., "whole_window_s": [0., 6.96]}


def run(args):
    summary_path, compact_path, output = Path(args.summary), Path(args.compact), Path(args.output_dir)
    require(not output.exists(), "Preserve existing result; fresh output directory required")
    summary_sha, compact_sha = sha256(summary_path), sha256(compact_path)
    summary = json.loads(summary_path.read_text())
    trusted_path = getattr(args, "trusted_provenance", None)
    if trusted_path:
        trusted = json.loads(Path(trusted_path).read_text())
        require(trusted["summary_input"]["sha256"] == summary_sha
                and trusted["selected_compact"]["sha256"] == compact_sha,
                "Explicit trusted provenance does not bind selected compact/summary")
    else:
        require(summary["artifact_sha256"]["native_failure_signals.npz"] == compact_sha,
                "Original compact SHA differs from frozen summary; selected replay needs --trusted-provenance")
    with np.load(compact_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    panels, reference = select_panels(summary, arrays)
    require(sha256(summary_path) == summary_sha and sha256(compact_path) == compact_sha,
            "Inputs changed while loading")
    output.mkdir(parents=True, exist_ok=False)
    limits = lowmu.shared_limits(panels, reference)
    names = ("ppo_cem_four_best_core_paths.png", "cem_two_best_core_paths.png")
    renderings = {names[0]: render(panels, reference, limits, output, names[0]),
                  names[1]: render([p for p in panels if p["group"].startswith("cem")],
                                   reference, limits, output, names[1])}
    compact = output/"selected_native_signals.npz"
    np.savez_compressed(compact, **selected_arrays(panels, arrays))
    provenance = {"status": "rendered_existing_completed_cases_not_new_successes",
        "selected_case_ids": [p["id"] for p in panels], "new_rollouts_or_training": False,
        "summary_input": {"path": str(summary_path), "sha256": summary_sha},
        "compact_input": {"path": str(compact_path), "sha256": compact_sha},
        "selected_compact": {"file": compact.name, "sha256": sha256(compact), "cases": 4},
        "source_sha256": {"renderer": sha256(__file__), "low_mu_renderer": sha256(lowmu.__file__),
                          "frozen_analyzer": sha256(frozen.__file__)},
        "artifact_sha256": {name: sha256(output/name) for name in names}, "renderings": renderings,
        "interpretation": "BELOW is a descriptive XY screen for its window, not complete task success."
    }
    write_json(output/"figure_provenance.json", provenance)
    return provenance


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--compact", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--trusted-provenance", type=Path,
                        help="Explicitly trust a prior provenance JSON for selected-compact replay.")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
