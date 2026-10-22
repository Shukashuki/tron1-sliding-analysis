"""Literal requested CEM+beta low-mu strip and labeled alternative, recorded only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_cem_friction_failure_400hz as frozen
import render_ppo_cem_success_paths_400hz as best
import render_ppo_friction_failure_topdown_400hz as native
from ppo_backend_400hz_contract import require, sha256, write_json

UNIQUE_MU = (.01, .05, .1, .15)
LITERAL_MU = (.1, .05, .1, .15)
ALTERNATIVE_MU = UNIQUE_MU
GROUP = "cem_beta_on"


def select_cases(summary, arrays):
    require(float(arrays["physics_dt_s"]) == .0025, "Require original400Hz clock")
    ids = np.asarray(arrays["case_ids"]).tolist()
    require(len(ids) == len(set(ids)), "Duplicate stored episodes are not independent trials")
    rows = {}
    for row in summary["rows"]:
        require(row["id"] not in rows, "Duplicate summary ids")
        rows[row["id"]] = row
    cases = []
    for mu in UNIQUE_MU:
        key = frozen.case_key(GROUP, mu)
        require(key in rows and key in ids, "Selected completed case missing: " + key)
        row = rows[key]
        require(row["evaluated"] is True and row["status"] in frozen.COMPLETE and row["group"] == GROUP
                and row["mu"] == mu and row["controller"] == "MPC-CEM" and row["condition"] == "beta_on"
                and row["prediction_friction_known"] is True, "Pending or incorrect CEM+beta/known-mu identity")
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
        cases.append({"id": key, "mu": mu, "group": GROUP, "row": row, "trace": trace})
    reference = {name[10:]: np.asarray(value).copy() for name, value in arrays.items() if name.startswith("reference_")}
    for name, shape in {"time_s": (349,), "position": (349, 3), "velocity": (349, 3), "quaternion": (349, 4)}.items():
        require(name in reference and reference[name].shape == shape and np.isfinite(reference[name]).all(), "Invalid reference: " + name)
    require(np.allclose(reference["time_s"], np.arange(349)*.02, rtol=0, atol=1e-10), "Invalid reference clock")
    return cases, reference


def panel_list(cases, order):
    lookup = {case["mu"]: case for case in cases}
    require(len(order) == 4 and all(mu in lookup for mu in order), "Require four available panel slots")
    return [lookup[mu] for mu in order]


def render(cases, reference, limits, destination, filename, *, alternative=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = ALTERNATIVE_MU if alternative else LITERAL_MU
    panels = panel_list(cases, order); xlim, ylim = limits
    figure, axes = plt.subplots(1, 4, figsize=(21, 7.2))
    glyph_indices = np.rint(np.asarray(best.GLYPH_TIMES)/.0025).astype(int)
    evidence = []
    for axis, case in zip(axes, panels):
        trace, row = case["trace"], case["row"]
        position, signals, masks = trace["position"], trace["signals"], native.glyph_masks(trace)
        core_mask = (trace["time_s"] >= 1.44-1e-9) & (trace["time_s"] <= 4.+1e-9)
        axis.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color=".55", lw=1.4)
        axis.plot(position[:, 0], position[:, 1], color="#dc8624", alpha=.4, lw=1.5)
        axis.plot(position[core_mask, 0], position[core_mask, 1], color="#dc8624", lw=2.7)
        for index in glyph_indices:
            for name, angle, color in (("body", signals["body_heading_rad"][index], "#1464b4"),
                                       ("velocity", signals["course_rad"][index], "#23934e")):
                if masks[name][index]:
                    delta = .16*np.array([np.cos(angle), np.sin(angle)])
                    axis.annotate("", xy=position[index, :2]+delta, xytext=position[index, :2],
                                  arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.1})
            for side, color in enumerate(("#d62728", "#9467bd")):
                if masks["wheel"][index, side]:
                    angle = signals["wheel_rolling_plane_heading_rad"][index, side]
                    delta = .08*np.array([np.cos(angle), np.sin(angle)])
                    ends = trace["wheel_pos"][index, side, :2]+np.array([-delta, delta])
                    axis.plot(ends[:, 0], ends[:, 1], color=color, lw=1.2,
                              alpha=.9 if masks["bearing"][index, side] else .18)
        axis.plot(*position[1600, :2], "o", color="#dc8624", ms=5)
        axis.plot(*position[-1, :2], "s", color="#dc8624", ms=5)
        axis.plot(*reference["position"][200, :2], "o", color=".5", ms=4)
        core, whole = (row["windows"][name]["tracking"] for name in ("core", "reference_overlap"))
        repeated = not alternative and case["mu"] == .1
        suffix = " | SAME recorded episode" if repeated else ""
        axis.set(title=f"MPC-CEM + beta | mu={case['mu']:g}{suffix}\n"
            f"XY RMSE core / whole: {core['xy_rmse_m']:.3f} / {whole['xy_rmse_m']:.3f} m\n"
            f"Whole peak: {whole['xy_peak_m']:.3f} m | core speed: {100*core['speed_retention']:.0f}%\n"
            f"Max tilt: {row['windows']['full']['maximum_tilt_deg']:.0f} deg (diagnostic)",
            xlabel="World X (m)", ylabel="World Y (m)", xlim=xlim, ylim=ylim)
        axis.set_xticks(np.linspace(*xlim, 5)); axis.set_yticks(np.linspace(*ylim, 5))
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2); axis.title.set_fontsize(9)
        evidence.append({"id": case["id"], "mu": case["mu"], "repeated_same_recorded_episode": repeated,
            "world_xlim_m": list(axis.get_xlim()), "world_ylim_m": list(axis.get_ylim()),
            "world_xticks_m": axis.get_xticks().tolist(), "world_yticks_m": axis.get_yticks().tolist(),
            "aspect": float(axis.get_aspect()), "native_samples_plotted": len(position),
            "native_final_time_s": float(trace["time_s"][-1]), "core_samples_bold": int(core_mask.sum())})
    descriptors = (("Reference", ".55", "--"), ("Core1.44–4s (bold)", ".3", "-"),
                   ("Full0–6.98s (light)", ".65", "-"), ("Body heading", "#1464b4", "-"),
                   ("COM direction (speed >=0.25m/s)", "#23934e", "-"),
                   ("Left wheel axial plane", "#d62728", "-"), ("Right wheel axial plane", "#9467bd", "-"))
    handles = [Line2D([], [], color=color, ls=style, label=text) for text, color, style in descriptors]
    handles += [Line2D([], [], color=".3", marker="o", ls="", label="4s marker"),
                Line2D([], [], color=".3", marker="s", ls="", label="6.98s endpoint")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3, fontsize=8, frameon=False)
    interpretation = "SUPPLEMENT: alternative interpretation [0.01,0.05,0.1,0.15]; NOT a replacement" if alternative else \
        "PRIMARY: literal requested order [0.1,0.05,0.1,0.15]; repeated0.1 is SAME episode; 3 unique episodes"
    figure.suptitle("MPC-CEM + beta | exact mu known | recorded400Hz native CPU MuJoCo | N=1 per unique coefficient\n"+interpretation, fontsize=11, y=.98)
    figure.text(.5, .035, "Same absolute axes in both figures. Whole=reference-overlap0–6.96s. Full late paths retained; no smoothing, pose/time alignment, inference or dynamics.", ha="center", fontsize=8)
    figure.text(.5, .012, "Opaque wheel ticks: count>0 AND upward Fz>5N; faint is not flight. Height/contact/tilt/beta are not XY failure gates.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0., .21, 1., .90)); figure.savefig(Path(destination)/filename, dpi=160); plt.close(figure)
    return {"panel_order_mu": list(order), "unique_episode_count": len(set(order)), "samples_per_unique_coefficient": 1,
            "alternative_interpretation_not_replacement": alternative, "panels": evidence,
            "glyph_native_indices": glyph_indices.tolist(), "poses_or_times_smoothed_or_aligned": False,
            "height_contact_tilt_or_beta_failure_gate": False}


def run(args):
    source, compact, output = Path(args.summary), Path(args.compact), Path(args.output_dir)
    require(not output.exists(), "Preserve existing result; fresh output directory required")
    source_sha, compact_sha = sha256(source), sha256(compact)
    summary = json.loads(source.read_text())
    require(summary["artifact_sha256"]["native_failure_signals.npz"] == compact_sha, "Compact SHA differs from summary")
    with np.load(compact, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    cases, reference = select_cases(summary, arrays)
    require(sha256(source) == source_sha and sha256(compact) == compact_sha, "Inputs changed while loading")
    output.mkdir(parents=True, exist_ok=False)
    limits = native.shared_limits(cases, reference)
    primary, supplement = "cem_low_mu_literal_requested.png", "cem_low_mu_alternative_interpretation.png"
    renderings = {primary: render(cases, reference, limits, output, primary),
                  supplement: render(cases, reference, limits, output, supplement, alternative=True)}
    selected = best.selected_arrays(cases, arrays)
    target = output/"selected_native_signals.npz"
    np.savez_compressed(target, **selected)
    selected_summary = {"rows": [case["row"] for case in cases], "selected_unique_cases": 4,
        "source_summary_sha256": source_sha, "source_compact_sha256": compact_sha,
        "artifact_sha256": {"native_failure_signals.npz": sha256(target)},
        "compact_filename": target.name,
        "source_scope": "Four unique completed CEM+beta episodes, not four new independent trials."}
    write_json(output/"selected_summary.json", selected_summary)
    artifact_names = [primary, supplement, target.name, "selected_summary.json"]
    provenance = {"status": "rendered_existing_completed_records", "new_rollouts_training_or_policy_inference": False,
        "summary_input": {"path": str(source), "sha256": source_sha},
        "compact_input": {"path": str(compact), "sha256": compact_sha},
        "selected_unique_case_ids": [case["id"] for case in cases],
        "source_sha256": {"renderer": sha256(__file__), "native_helpers": sha256(native.__file__),
                          "best_helpers": sha256(best.__file__), "analyzer": sha256(frozen.__file__)},
        "artifact_sha256": {name: sha256(output/name) for name in artifact_names}, "renderings": renderings,
        "small_replay": "Use selected_summary.json with selected_native_signals.npz and a fresh output directory."}
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
