"""Four fixed-world topdowns from SHA-bound, recorded400Hz PPO+beta data.

No dynamics, policy inference, IK, smoothing, pose alignment or new scoring.
The displayed low-friction examples can co-occur with severe body tilt; the
contact gate affects wheel-tick opacity, never an airborne/failure label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_cem_friction_failure_400hz as frozen
from ppo_backend_400hz_contract import require, sha256, write_json

SELECTED_MU = (.01, .05, .1, .15)
GROUP = "ppo_beta_on"
SHAPES = {"time_s": (2793,), "position": (2793, 3),
          "quaternion": (2793, 4), "com_velocity": (2793, 3),
          "angular_velocity": (2793, 3), "joint_pos": (2793, 8),
          "joint_vel": (2793, 8), "wheel_pos": (2793, 2, 3),
          "contact_count": (2793, 2), "contact_force": (2793, 2, 3),
          "contact_slip": (2793, 2), "wheel_quaternion_wxyz": (2793, 2, 4),
          "wheel_center_velocity_w": (2793, 2, 3)}
COLORS = ("#d45a42", "#b55b9e", "#dd9c27", "#258177")


def select_panels(summary, arrays):
    """Only exact completed PPO+beta cells; never substitute a pending case."""
    require(float(arrays["physics_dt_s"]) == .0025, "Require original400Hz compact data")
    case_ids = np.asarray(arrays["case_ids"]).tolist()
    require(len(case_ids) == len(set(case_ids)), "Duplicate compact case ids")
    lookup = {}
    for row in summary["rows"]:
        require(row["id"] not in lookup, "Duplicate summary case ids")
        lookup[row["id"]] = row
    panels = []
    for mu in SELECTED_MU:
        key = frozen.case_key(GROUP, mu)
        require(key in lookup and key in case_ids, "Selected completed case missing: " + key)
        row = lookup[key]
        require(row["evaluated"] is True and row["status"] in frozen.COMPLETE
                and row["group"] == GROUP and row["controller"] == "PPO"
                and row["condition"] == "beta_on" and row["mu"] == mu,
                "Selected case is pending or a different controller: " + key)
        prefix = key.replace("/", "__") + "__"
        trace = {}
        for name, shape in SHAPES.items():
            value = np.asarray(arrays[prefix + name])
            require(value.shape == shape and value.dtype.kind in "fiu"
                    and np.isfinite(value).all(), "Invalid native shape/data: " + name)
            trace[name] = value.copy()
        trace["physics_dt_s"] = .0025
        frozen.clock_audit(trace)
        trace["signals"] = frozen.heading_signals(trace["quaternion"], trace["com_velocity"],
            trace["wheel_quaternion_wxyz"], trace["wheel_center_velocity_w"])
        body, wheel = frozen.metrics.heading_geometry_masks(trace)
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = body, wheel
        panels.append({"id": key, "mu": mu, "row": row, "trace": trace})
    reference = {name[len("reference_"):]: np.asarray(value).copy()
                 for name, value in arrays.items() if name.startswith("reference_")}
    for name, shape in {"time_s": (349,), "position": (349, 3),
                        "velocity": (349, 3), "quaternion": (349, 4)}.items():
        require(name in reference and reference[name].shape == shape
                and np.isfinite(reference[name]).all(), "Invalid reference shape: " + name)
    require(np.allclose(reference["time_s"], np.arange(349) * .02, rtol=0, atol=1e-10),
            "Require original0–6.96s reference clock")
    return panels, reference


def bearing_mask(trace):
    return (trace["contact_count"] > 0) & (trace["contact_force"][:, :, 2] > 5.)


def glyph_masks(trace):
    """COM arrows do not inherit body-axis validity; ticks are axial geometry."""
    return {"body": trace["body_heading_geometry_valid"],
            "velocity": trace["signals"]["speed_m_s"] >= .25,
            "wheel": trace["wheel_heading_geometry_valid"],
            "bearing": bearing_mask(trace)}


def shared_limits(panels, reference):
    xy = np.concatenate([reference["position"][:, :2]] +
                        [p["trace"]["position"][:, :2] for p in panels] +
                        [p["trace"]["wheel_pos"][:, :, :2].reshape(-1, 2) for p in panels])
    center = .5 * (xy.min(axis=0) + xy.max(axis=0))
    half = max(.5, .5 * float(np.ptp(xy, axis=0).max()) + .35)
    return tuple(float(x) for x in (center[0] - half, center[0] + half)), \
        tuple(float(x) for x in (center[1] - half, center[1] + half))


def render(panels, reference, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xlim, ylim = shared_limits(panels, reference)
    figure, axes = plt.subplots(1, 4, figsize=(20, 6.4), sharex=True, sharey=True)
    indices = np.unique(np.append(np.arange(0, 2793, 400), 2792))
    geometry = []
    for axis, panel, color in zip(axes, panels, COLORS):
        trace, row = panel["trace"], panel["row"]
        position, signals, masks = trace["position"], trace["signals"], glyph_masks(trace)
        axis.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color=".6", lw=1.4)
        axis.plot(position[:, 0], position[:, 1], color=color, lw=1.6)
        for index in indices:
            for name, angle, arrow_color in (
                ("body", signals["body_heading_rad"][index], "#1464b4"),
                ("velocity", signals["course_rad"][index], "#23934e")):
                if masks[name][index]:
                    delta = .16 * np.array([np.cos(angle), np.sin(angle)])
                    axis.annotate("", xy=position[index, :2] + delta, xytext=position[index, :2],
                        arrowprops={"arrowstyle": "->", "color": arrow_color, "linewidth": 1.1})
            for side, wheel_color in enumerate(("#d62728", "#9467bd")):
                if masks["wheel"][index, side]:
                    angle = signals["wheel_rolling_plane_heading_rad"][index, side]
                    delta = .085 * np.array([np.cos(angle), np.sin(angle)])
                    ends = trace["wheel_pos"][index, side, :2] + np.array([-delta, delta])
                    axis.plot(ends[:, 0], ends[:, 1], color=wheel_color, lw=1.2,
                              alpha=.9 if masks["bearing"][index, side] else .18)
        axis.plot(*position[1600, :2], "o", color=color, ms=5, zorder=4)
        axis.plot(*reference["position"][200, :2], "o", color=".5", ms=4)
        core = row["windows"]["core"]["tracking"]
        whole = row["windows"]["reference_overlap"]["tracking"]
        tilt = row["windows"]["full"]["maximum_tilt_deg"]
        axis.set(title=f"PPO + beta | mu={panel['mu']:g}\n"
            f"XY RMSE core / overlap: {core['xy_rmse_m']:.3f} / {whole['xy_rmse_m']:.3f} m\n"
            f"Core speed: {100*core['speed_retention']:.0f}% | max tilt: {tilt:.0f} deg",
            xlabel="World X (m)", ylabel="World Y (m)", xlim=xlim, ylim=ylim)
        axis.set_xticks(np.linspace(*xlim, 5)); axis.set_yticks(np.linspace(*ylim, 5))
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2)
        geometry.append({"id": panel["id"], "world_xlim_m": list(axis.get_xlim()),
                         "world_ylim_m": list(axis.get_ylim()), "aspect": float(axis.get_aspect()),
                         "world_xticks_m": axis.get_xticks().tolist(), "world_yticks_m": axis.get_yticks().tolist(),
                         "native_samples_plotted": len(position), "last_native_time_s": float(trace["time_s"][-1]),
                         "maximum_tilt_deg_diagnostic_only": tilt})
    handles = [Line2D([], [], color=".6", ls="--", label="Reference"),
        Line2D([], [], color="#1464b4", marker=">", label="Body heading"),
        Line2D([], [], color="#23934e", marker=">", label="COM velocity direction (speed >=0.25m/s)"),
        Line2D([], [], color="#d62728", label="Left wheel axial plane"),
        Line2D([], [], color="#9467bd", label="Right wheel axial plane"),
        Line2D([], [], color=".5", marker="o", ls="", label="4s marker")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3, fontsize=9, frameon=False)
    figure.suptitle("PhysX-trained PPO + beta -> native CPU MuJoCo | recorded400Hz | one shared absolute world frame | N=1/cell", fontsize=12, y=.98)
    figure.text(.5, .035, "Full original0–6.98s paths; core1.44–4s, reference-overlap0–6.96s. No pose/time alignment or smoothing. Severe tilt can accompany the XY failures.", ha="center", fontsize=9)
    figure.text(.5, .01, "Opaque wheel ticks: measured contact count>0 AND upward Fz>5N. Faint ticks are NOT a flight label. Height/contact/tilt/beta are not failure gates.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0., .18, 1., .94))
    figure.savefig(Path(destination) / "ppo_beta_on_low_mu_topdown.png", dpi=160)
    plt.close(figure)
    return {"panels": geometry, "glyph_native_indices": indices.tolist(),
            "wheel_bearing_gate": "contact_count>0 AND upward contact_force_z>5N; opacity only",
            "velocity_arrow_speed_threshold_m_s_inclusive": .25,
            "body_heading_requires_speed": False, "invalid_geometry_skipped": True,
            "pose_time_alignment_smoothing_or_dynamics": False,
            "height_airborne_tilt_or_beta_failure_gate": False}


def run(args):
    summary_path, compact_path, output = Path(args.summary), Path(args.compact), Path(args.output_dir)
    require(not output.exists(), "Preserve existing figure; fresh output directory required")
    summary_sha, compact_sha = sha256(summary_path), sha256(compact_path)
    summary = json.loads(summary_path.read_text())
    require(summary["artifact_sha256"]["native_failure_signals.npz"] == compact_sha,
            "Compact trace SHA differs from frozen summary")
    with np.load(compact_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    panels, reference = select_panels(summary, arrays)
    require(sha256(summary_path) == summary_sha and sha256(compact_path) == compact_sha,
            "Inputs changed while loading")
    output.mkdir(parents=True, exist_ok=False)
    rendering = render(panels, reference, output)
    figure = output / "ppo_beta_on_low_mu_topdown.png"
    provenance = {"status": "rendered_existing_completed_data", "selected_group": GROUP,
        "selected_mu": list(SELECTED_MU), "new_physical_rollouts_or_training": False,
        "summary_input": {"path": str(summary_path), "sha256": summary_sha},
        "compact_input": {"path": str(compact_path), "sha256": compact_sha},
        "renderer_sha256": sha256(__file__), "frozen_analyzer_sha256": sha256(frozen.__file__),
        "figure": {"file": figure.name, "sha256": sha256(figure)}, "rendering": rendering}
    write_json(output / "figure_provenance.json", provenance)
    return provenance


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--compact", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
