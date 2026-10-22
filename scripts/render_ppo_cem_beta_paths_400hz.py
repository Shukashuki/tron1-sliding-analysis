"""Recorded400Hz beta-on/off XY paths; no training or physical rollouts.

Only complete SHA-bound public records are selected. Late departures and large
tilts remain visible; height/contact/tilt/beta do not define XY failure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze_ppo_cem_friction_failure_400hz as frozen
import render_ppo_cem_success_paths_400hz as best
import render_ppo_friction_failure_topdown_400hz as native
from ppo_backend_400hz_contract import require, sha256, write_json

GROUPS = ("ppo_beta_on", "ppo_beta_off", "cem_beta_on", "cem_beta_off")
SELECTED = tuple((group, mu) for mu in (.7, .15) for group in GROUPS)
COLORS = {"ppo_beta_on": "#258177", "ppo_beta_off": "#3b78ae",
          "cem_beta_on": "#dc8624", "cem_beta_off": "#b55b9e"}


def label(panel):
    return ("PPO" if panel["group"].startswith("ppo") else "MPC-CEM") + \
        (" + beta" if panel["group"].endswith("beta_on") else " - beta")


def select_panels(summary, arrays):
    require(float(arrays["physics_dt_s"]) == .0025, "Require original400Hz compact clock")
    ids = np.asarray(arrays["case_ids"]).tolist()
    require(len(ids) == len(set(ids)), "Duplicate compact ids")
    rows = {}
    for row in summary["rows"]:
        require(row["id"] not in rows, "Duplicate summary ids")
        rows[row["id"]] = row
    panels = []
    for group, mu in SELECTED:
        key = frozen.case_key(group, mu)
        require(key in rows and key in ids, "Selected completed case missing: " + key)
        row = rows[key]
        require(row["evaluated"] is True and row["status"] in frozen.COMPLETE
                and row["group"] == group and row["mu"] == mu
                and row["condition"] == ("beta_on" if group.endswith("beta_on") else "beta_off")
                and row["controller"] == ("PPO" if group.startswith("ppo") else "MPC-CEM"),
                "Pending case or incorrect controller/beta identity: " + key)
        prefix = key.replace("/", "__") + "__"
        trace = {}
        for name, shape in native.SHAPES.items():
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
        require(row["core_large_xy_screen"] == (best.screen(core) == "LARGE")
                and row["overlap_large_xy_screen"] == (best.screen(whole) == "LARGE"),
                "Stored XY screening does not match numerical metrics")
        panels.append({"id": key, "group": group, "mu": mu, "row": row, "trace": trace})
    reference = {name[len("reference_"):]: np.asarray(value).copy()
                 for name, value in arrays.items() if name.startswith("reference_")}
    for name, shape in {"time_s": (349,), "position": (349, 3),
                        "velocity": (349, 3), "quaternion": (349, 4)}.items():
        require(name in reference and reference[name].shape == shape and np.isfinite(reference[name]).all(),
                "Invalid reference shape/data: " + name)
    require(np.allclose(reference["time_s"], np.arange(349)*.02, rtol=0, atol=1e-10), "Invalid reference clock")
    return panels, reference


def numerical_rows(panels):
    """Copy unrounded recorded metrics; beta error is target-relative, not failure."""
    result = []
    for panel in panels:
        row = panel["row"]
        value = {"id": panel["id"], "controller": row["controller"], "condition": row["condition"],
                 "mu": panel["mu"], "core_large_xy_screen": row["core_large_xy_screen"],
                 "whole_large_xy_screen": row["overlap_large_xy_screen"],
                 "maximum_native_tilt_deg_diagnostic": row["windows"]["full"]["maximum_tilt_deg"]}
        for window, source in (("core", "core"), ("whole", "reference_overlap")):
            tracking = row["windows"][source]["tracking"]
            for name in ("xy_rmse_m", "xy_peak_m", "beta_rmse_deg", "beta_valid_samples", "speed_retention"):
                value[window + "_" + name] = tracking.get(name)
        result.append(value)
    return result


def paired_comparisons(rows):
    result = []
    for mu in (.7, .15):
        for controller in ("PPO", "MPC-CEM"):
            on = next(r for r in rows if r["mu"] == mu and r["controller"] == controller and r["condition"] == "beta_on")
            off = next(r for r in rows if r["mu"] == mu and r["controller"] == controller and r["condition"] == "beta_off")
            fields = ("core_xy_rmse_m", "whole_xy_rmse_m", "whole_xy_peak_m",
                      "core_beta_rmse_deg", "core_speed_retention", "maximum_native_tilt_deg_diagnostic")
            result.append({"controller": controller, "mu": mu, "beta_on": on, "beta_off": off,
                           "off_minus_on": {name: off[name]-on[name] if off[name] is not None and on[name] is not None else None
                                            for name in fields}})
    return result


def render(panels, reference, limits, destination, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xlim, ylim = limits
    figure, axes = plt.subplots(1, 4, figsize=(21, 7.2))
    indices = np.rint(np.asarray(best.GLYPH_TIMES)/.0025).astype(int)
    evidence = []
    for axis, panel in zip(axes, panels):
        trace, row, color = panel["trace"], panel["row"], COLORS[panel["group"]]
        position, signals, masks = trace["position"], trace["signals"], native.glyph_masks(trace)
        core_mask = (trace["time_s"] >= 1.44-1e-9) & (trace["time_s"] <= 4.+1e-9)
        axis.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color=".55", lw=1.4)
        axis.plot(position[:, 0], position[:, 1], color=color, alpha=.4, lw=1.5)
        axis.plot(position[core_mask, 0], position[core_mask, 1], color=color, lw=2.7)
        for index in indices:
            for name, angle, arrow_color in (("body", signals["body_heading_rad"][index], "#1464b4"),
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
        tilt = row["windows"]["full"]["maximum_tilt_deg"]
        axis.set(title=f"{label(panel)} | mu={panel['mu']:g}\n"
            f"XY RMSE core / whole: {core['xy_rmse_m']:.3f} / {whole['xy_rmse_m']:.3f} m\n"
            f"Whole peak: {whole['xy_peak_m']:.3f} m | core speed: {100*core['speed_retention']:.0f}%\n"
            f"Max tilt: {tilt:.0f} deg (diagnostic only)",
            xlabel="World X (m)", ylabel="World Y (m)", xlim=xlim, ylim=ylim)
        axis.set_xticks(np.linspace(*xlim, 5)); axis.set_yticks(np.linspace(*ylim, 5))
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2); axis.title.set_fontsize(9.5)
        evidence.append({"id": panel["id"], "label": label(panel), "world_xlim_m": list(axis.get_xlim()),
            "world_ylim_m": list(axis.get_ylim()), "world_xticks_m": axis.get_xticks().tolist(),
            "world_yticks_m": axis.get_yticks().tolist(), "aspect": float(axis.get_aspect()),
            "native_samples_plotted": len(position), "native_final_time_s": float(trace["time_s"][-1]),
            "core_samples_bold": int(core_mask.sum()), "cropped_after4s": False,
            "maximum_native_tilt_deg_diagnostic": tilt})
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
    figure.suptitle("Beta reward on/off | recorded400Hz | native CPU MuJoCo | same absolute world axes across all three figures | N=1/cell\nPhysX-trained PPO; CEM receives exact mu. Full late departures/large tilts are retained.", fontsize=11, y=.98)
    figure.text(.5, .035, "Whole=reference-overlap0–6.96s. Height/contact/tilt/beta are not XY failure gates. No total reward comparison across objectives.", ha="center", fontsize=8)
    figure.text(.5, .012, "Opaque wheel ticks: count>0 AND upward Fz>5N; faint is not flight. No smoothing, pose/time alignment, dynamics or policy inference.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0., .21, 1., .90))
    figure.savefig(Path(destination)/filename, dpi=160); plt.close(figure)
    return {"panels": evidence, "glyph_times_s": list(best.GLYPH_TIMES), "glyph_native_indices": indices.tolist(),
            "body_axis_geometry_gate": True, "wheel_axis_geometry_gate": True,
            "COM_velocity_minimum_m_s_inclusive": .25, "wheel_tick_opacity_force_n_strict": 5.,
            "height_contact_tilt_or_beta_failure_gate": False, "smoothed_or_aligned": False}


def run(args):
    summary_path, compact_path, output = Path(args.summary), Path(args.compact), Path(args.output_dir)
    require(not output.exists(), "Preserve existing result; fresh output directory required")
    summary_sha, compact_sha = sha256(summary_path), sha256(compact_path)
    summary = json.loads(summary_path.read_text())
    require(summary["artifact_sha256"]["native_failure_signals.npz"] == compact_sha, "Compact SHA differs from frozen summary")
    with np.load(compact_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    panels, reference = select_panels(summary, arrays)
    require(sha256(summary_path) == summary_sha and sha256(compact_path) == compact_sha, "Inputs changed while loading")
    output.mkdir(parents=True, exist_ok=False)
    limits = native.shared_limits(panels, reference)
    selections = {"beta_on_off_mu0p700.png": panels[:4], "beta_on_off_mu0p150.png": panels[4:],
                  "cem_beta_on_off_two_mu.png": [p for p in panels if p["group"].startswith("cem")]}
    renderings = {name: render(selected, reference, limits, output, name) for name, selected in selections.items()}
    rows = numerical_rows(panels)
    comparison = {"cases": rows, "paired_comparisons": paired_comparisons(rows), "sample_size_per_case": 1,
                  "core_window_s": [1.44, 4.], "whole_window_s": [0., 6.96],
                  "beta_error_is_target_relative_not_zero_slip_failure": True,
                  "height_contact_tilt_or_beta_failure_gate": False, "reward_return_compared": False}
    write_json(output/"numerical_comparison.json", comparison)
    with (output/"numerical_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    artifacts = list(selections)+["numerical_comparison.json", "numerical_comparison.csv"]
    provenance = {"status": "rendered_existing_completed_on_off_records", "selected_case_ids": [p["id"] for p in panels],
        "new_training_policy_inference_or_physical_rollouts": False,
        "summary_input": {"path": str(summary_path), "sha256": summary_sha},
        "compact_input": {"path": str(compact_path), "sha256": compact_sha},
        "source_sha256": {"renderer": sha256(__file__), "native_render_helpers": sha256(native.__file__),
                          "best_paths_helpers": sha256(best.__file__), "frozen_analyzer": sha256(frozen.__file__)},
        "artifact_sha256": {name: sha256(output/name) for name in artifacts}, "renderings": renderings}
    write_json(output/"figure_provenance.json", provenance)
    return provenance


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--compact", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
