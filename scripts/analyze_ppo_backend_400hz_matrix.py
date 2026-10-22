"""Offline descriptive analysis of the NEW completed 2x2x17 PPO cohort.

Native samples stay native. Wheel arrows use axle/world-up rolling planes from
isolated model FK, not spinning-wheel +X, and never step or align the plant.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import analyze_ppo_smoothness as smooth
from analyze_ppo_contact_stability import contact_masks, contact_statistics
from analyze_ppo_contact_timestep import band_spectrum
from analyze_ppo_ground_support_failure import persistent_events
from controller_metrics import wrap
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import quaternion_rotate
from ppo_backend_400hz_contract import (DEFAULT_PROTOCOL, INPUT_PINS, read_json,
    require, sha256, validate_output, verify_legacy, verify_protocol, write_json)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_LABELS = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}
TRAIN_LABELS = {"physx": "PhysX-trained", "mjlab": "MJWarp-trained"}
COLORS = {"reference": "#777777", "body": "#1464b4", "velocity": "#ed9518",
          "left": "#d62728", "right": "#9467bd", "path": "#15956c"}


def load_fk(model_file, legacy_evaluator):
    """Extract only the SHA-pinned mesh-free loader; no evaluator main/imports."""
    require(sha256(model_file) == INPUT_PINS["model"], "FK model is not the cohort model")
    source, _ = verify_legacy(legacy_evaluator,
        "2251d740d7568874a47a232556241cc8d1ee5f5b7be1351d1c657aea8fa85f87")
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    namespace = {"ET": ET, "Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    model, removed = namespace["load_model"](model_file, no_visual_mesh=True)
    return WheelKinematics(model), {"model_sha256": sha256(model_file),
        "loader_evaluator_sha256": sha256(source), "removed_proven_noncolliding_visual_meshes": removed,
        "isolated_FK_only": True, "dynamics_stepping": False}


def reference_data(path):
    require(sha256(path) == INPUT_PINS["motion"], "Reference is not the locked cohort trajectory")
    reference = smooth._load_reference(path)
    require(np.isclose(reference["time_s"][0], 0.) and np.isclose(reference["time_s"][-1], 6.96),
            "Reference clock coverage differs")
    return reference


def window_metrics(trace, reference, signals, interval):
    """Same SI/native COM conventions as the existing tracking diagnostics."""
    time = trace["time_s"]
    select = (time >= interval[0] - 1e-9) & (time <= interval[1] + 1e-9)
    require(select.sum() >= 4, "Insufficient native window samples")
    tracking = smooth._tracking(time, trace["position"], trace["com_velocity"], trace["quaternion"], reference, interval)
    # Tracking helper's speed threshold is retained, with an additional explicit
    # geometric validity gate near vertical body-forward axes.
    if tracking["available"]:
        query = time[select]
        _, ref_heading = smooth._orientation(reference["quaternion"])
        ref_beta = np.unwrap(np.arctan2(reference["velocity"][:, 1], reference["velocity"][:, 0]) - ref_heading)
        desired = np.interp(query, reference["time_s"], ref_beta)
        speed = np.interp(query, reference["time_s"], np.linalg.norm(reference["velocity"][:, :2], axis=1))
        valid = signals["root_beta_valid"][select] & (speed >= .25)
        difference = wrap(signals["root_directed_beta_rad"][select] - desired)
        tracking["beta_valid_samples"] = int(valid.sum())
        tracking["beta_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(difference[valid] ** 2)))) if valid.any() else None
    counts = trace["contact_count"][select]
    forces = trace["contact_force"][select]
    masks = contact_masks(counts, forces[:, :, 2])
    contact = {}
    for definition, group in masks.items():
        contact[definition] = {}
        for name, value in group.items():
            statistics = contact_statistics(time[select], value)
            contact[definition][name] = {key: statistics[key] for key in
                ("sample_duty_fraction", "time_duty_fraction_zoh", "state_switch_count", "state_switches_per_s")}
    q = trace["quaternion"][select]
    tilt = np.degrees(np.arccos(np.clip(1. - 2. * np.sum(q[:, 1:3] ** 2, axis=1), -1., 1.)))
    events = {str(threshold): persistent_events(time[select], tilt >= threshold) for threshold in (30, 45, 60)}
    return {"interval_s": list(interval), "native_samples": int(select.sum()), "tracking": tracking,
        "world_angular_xy_common10_100": band_spectrum(trace["angular_velocity"][select, :2], .0025),
        "contact": contact, "maximum_tilt_deg": float(tilt.max()),
        "persistent_tilt_diagnostics": {key: {name: value[name] for name in
            ("first_persistent_event", "persistent_event_count")} for key, value in events.items()},
        "height_or_airborne_failure_gate": False}


def analyze_case(raw, engine, reference, kinematics):
    trace = native_trace(raw, "policy_isaac" if engine == "physx" else "policy_mujoco")
    require(len(trace["time_s"]) == 2793 and trace["physics_dt_s"] == .0025, "Wrong native cohort clock")
    reconstructed = kinematics.reconstruct(trace)
    signals = reconstructed["signals"]
    windows = {"core": window_metrics(trace, reference, signals, (1.44, 4.)),
               "full": window_metrics(trace, reference, signals, (0., 6.98)),
               "reference_overlap": window_metrics(trace, reference, signals, (0., 6.96))}
    require(windows["core"]["native_samples"] == 1025, "Core400Hz sample count differs")
    core = windows["core"]["tracking"]
    require(core["available"], "Core tracking unavailable")
    screening = {"kind": "exploratory_descriptive_XY_screen", "rmse_threshold_m": .15, "peak_threshold_m": .30,
        "large_xy_error": bool(core["xy_rmse_m"] > .15 or core["xy_peak_m"] > .30),
        "height_contact_tilt_or_beta_used_as_failure_gate": False}
    trace["signals"] = signals
    trace["wheel_quaternion_wxyz"] = reconstructed["wheel_quaternion_wxyz"]
    return trace, {"windows": windows, "xy_screen": screening, "wheel_heading_FK_audit": reconstructed["audit"]}


def case_key(case):
    return f"{case['training_backend']}/{case['engine_tag']}/mu{case['mu']:.3f}".replace(".", "p")


def load_matrix(input_root, reference, model, legacy_evaluator, protocol_path=DEFAULT_PROTOCOL):
    root = Path(input_root).resolve()
    manifest_path = root / "validation_matrix_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "validation_matrix_completed" and manifest.get("counts") ==
            {"pipeline_smoke_expected": 4, "pipeline_smoke_completed": 4, "formal_expected": 68,
             "formal_completed": 68, "runtime_failed": 0}, "Matrix is not completed/verified")
    protocol = verify_protocol(protocol_path)
    require(manifest["protocol_sha256"] == sha256(protocol_path) and manifest["input_motion_sha256"] == sha256(reference)
            and manifest["input_model_sha256"] == sha256(model),
            "Matrix manifest protocol/reference identity differs")
    cases = manifest["cases"]
    formal = [case for case in cases if case["phase"] == "formal"]
    smoke = [case for case in cases if case["phase"] == "pipeline_smoke"]
    require(len(cases) == 72 and len(formal) == 68 and len(smoke) == 4, "Expected 68formal+4pipeline-smoke episodes")
    expected = {(origin, engine, mu) for origin in ("physx", "mjlab") for engine in ENGINE_LABELS
                for mu in protocol["validation"]["friction_grid"]}
    require({(c["training_backend"], c["engine_tag"], c["mu"]) for c in formal} == expected,
            "Formal matrix incomplete/duplicate/mixed cohort")
    require({(c["training_backend"], c["engine_tag"], c["mu"]) for c in smoke} ==
            {(origin, engine, .7) for origin in ("physx", "mjlab") for engine in ENGINE_LABELS}, "Smoke matrix differs")
    reference_values = reference_data(reference)
    kinematics, fk_audit = load_fk(model, legacy_evaluator)
    rows, nominal, checkpoint_by_training = [], {}, {}
    for case in cases:
        require(case.get("status") == "completed_verified", "Unverified matrix episode")
        folder = (root / case["relative_output_dir"]).resolve()
        require(folder.is_relative_to(root) and folder == Path(case["directory"]).resolve(), "Matrix case directory escaped/differs")
        backend = ENGINE_LABELS[case["engine_tag"]]
        require(case["validation_backend"] == backend, "Engine label differs")
        verified = validate_output(folder, case["cohort_dir"], protocol_path,
            validation_backend=backend, allow_smoke=case["phase"] == "pipeline_smoke")
        require(verified["smoke_only"] is (case["phase"] == "pipeline_smoke")
                and verified["training_backend"] == case["training_backend"]
                and verified["checkpoint_sha256"] == case["checkpoint_sha256"]
                and verified["evaluation_contact_friction"] == case["mu"], "Matrix cell identity differs")
        for key in ("recording_sha256", "report_sha256"):
            require(case["verification"][key] == verified[key], "Published queue verification hash differs: " + key)
        if case["phase"] != "formal":
            continue
        previous = checkpoint_by_training.setdefault(case["training_backend"], verified["checkpoint_sha256"])
        require(previous == verified["checkpoint_sha256"], "Epoch/actor changed within a training row")
        recording = folder / ("physics_trajectory.npz" if case["engine_tag"] == "physx" else "rollout.npz")
        with np.load(recording, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        trace, metrics = analyze_case(raw, case["engine_tag"], reference_values, kinematics)
        report = read_json(folder / "report.json")
        require(case["actor_sha256"] == report["backend_400hz_cohort_evaluation"]["normalized_actor_sha256"],
                "Matrix normalized actor identity differs")
        row = {"id": case_key(case), "training_backend": case["training_backend"], "validation_backend": backend,
            "engine_tag": case["engine_tag"], "mu": case["mu"], "directory": str(folder),
            "checkpoint_sha256": verified["checkpoint_sha256"], "actor_sha256": case["actor_sha256"],
            "input_hashes": {"report": verified["report_sha256"], "recording": verified["recording_sha256"]},
            "material_mapping": material_summary(report["contact_material_audit"], case["engine_tag"]), **metrics}
        rows.append(row)
        if case["mu"] == .7:
            nominal[(case["training_backend"], case["engine_tag"])] = trace
    require(len(set(checkpoint_by_training.values())) == 2, "Paired backends unexpectedly share a checkpoint")
    helper_names = ("analyze_ppo_smoothness.py", "analyze_ppo_contact_stability.py", "analyze_ppo_contact_timestep.py",
        "analyze_ppo_ground_support_failure.py", "controller_metrics.py", "controller_trace.py",
        "nmpc_tracking_objective.py", "height_smoothness.py", "ppo_backend_400hz_contract.py")
    return nominal, reference_values, {"schema_version": 1, "kind": "completed_new400hz_PPO_closed_loop_2x2x17_matrix",
        "cohort_id": protocol["cohort_id"], "rows": rows, "formal_episode_count": 68, "pipeline_smoke_episode_count": 4,
        "sample_size_per_cell": 1, "physics_dt_s": .0025, "policy_dt_s": .02,
        "reference_sha256": sha256(reference), "reference_real_last_time_s": 6.96,
        "manifest_sha256": sha256(manifest_path), "protocol_sha256": sha256(protocol_path),
        "analyzer_sha256": sha256(__file__), "FK_audit": fk_audit,
        "analysis_helpers_sha256": {name: sha256(Path(__file__).with_name(name)) for name in helper_names},
        "visual_reviewed": False, "new_simulation": False, "new_policy_inference": False,
        "limitations": ["N=1 deterministic episode per cell; descriptive, not a statistical robustness estimate.",
            "Closed-loop transfer changes actions through observations; this is not the same-action native-backend diagnostic.",
            "No XY alignment, root teleport, pose-height correction, state smoothing or resampling is applied.",
            "Full native horizon is6.98s; reference ends6.96s, so full tracking is unavailable and overlap is separate.",
            "Nonzero beta is intended behavior. Only reference-relative beta error is summarized with speed/axis validity.",
            "Zero measured ground load is not labeled flight or tracking failure. Tilt/contact are diagnostic only.",
            "Fallen-policy lower HF power is not evidence of better whole-body stability.",
            "Native effective-pair mu mapping is audited; contact laws and plant representations remain different.",
            "Contact counts/forces are backend-specific telemetry operators: the same count>0/Fz>5N rule is descriptive load eligibility, not an identical measurement operator; sampling, solver and warm-start behavior can contribute."]}


def material_summary(audit, engine):
    if engine == "physx":
        result = {key: copy.deepcopy(audit[key]) for key in
            ("live_physx_materials_verified", "ground_material_usd_verified", "wheel_mu_range", "effective_pair_mapping")}
        result["native_mapping"] = audit["recipe"]["wheel_contact_friction"]
    else:
        result = {key: copy.deepcopy(audit[key]) for key in
            ("native_mapping_verified", "pair_coefficient", "mapping", "geom_priority", "explicit_contact_pair_count")}
        result["sliding_friction_range"] = [min(row[0] for row in audit["geom_friction"]), max(row[0] for row in audit["geom_friction"])]
    result["native_plant_readback_in_source_report"] = "native_plant_readback" in audit
    return result


def heading_geometry_masks(trace):
    forward = quaternion_rotate(trace["quaternion"], np.array([1., 0., 0.]))
    axle = quaternion_rotate(trace["wheel_quaternion_wxyz"], np.array([0., 1., 0.]))
    rolling = np.cross(axle, np.array([0., 0., 1.]))
    return np.linalg.norm(forward[:, :2], axis=-1) > 1e-6, np.linalg.norm(rolling[:, :, :2], axis=-1) > 1e-6


def render_nominal(nominal, reference, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    xy = np.concatenate([reference["position"][:, :2], *[trace["position"][:, :2] for trace in nominal.values()]])
    low, high = xy.min(axis=0) - .35, xy.max(axis=0) + .35
    arrow_length = .18
    for i, origin in enumerate(("physx", "mjlab")):
        for j, engine in enumerate(ENGINE_LABELS):
            ax, trace = axes[i, j], nominal[(origin, engine)]
            ax.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color=COLORS["reference"], lw=1.1)
            ax.plot(trace["position"][:, 0], trace["position"][:, 1], color=COLORS["path"], lw=1.)
            signals = trace["signals"]
            body_valid, wheel_valid = heading_geometry_masks(trace)
            for timestamp in (2., 3., 4.):
                k = int(round(timestamp / .0025))
                root = trace["position"][k, :2]
                arrows = [(root, signals["body_heading_rad"][k], "body", body_valid[k]),
                    (root, signals["course_rad"][k], "velocity", signals["speed_m_s"][k] >= .25)]
                for wheel, color in enumerate(("left", "right")):
                    arrows.append((trace["wheel_pos"][k, wheel, :2], signals["wheel_rolling_plane_heading_rad"][k, wheel], color,
                                   wheel_valid[k, wheel]))
                for start, heading, color, valid in arrows:
                    if valid:
                        direction = arrow_length * np.array([np.cos(heading), np.sin(heading)])
                        axial = color in ("left", "right")
                        ax.annotate("", xy=start + direction * (.5 if axial else 1.),
                            xytext=start - direction * .5 if axial else start,
                            arrowprops={"arrowstyle": "<->" if axial else "->", "color": COLORS[color], "lw": 1.25})
                ax.text(*root, f" {timestamp:g}s", fontsize=7)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World X (m)", ylabel="World Y (m)",
                   title=TRAIN_LABELS[origin] + " → " + ENGINE_LABELS[engine])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
    handles = [Line2D([], [], color=COLORS[key], label=label) for key, label in
        (("reference", "Reference path"), ("path", "Measured root path"), ("body", "Body +X"),
         ("velocity", "COM velocity"), ("left", "Left rolling plane (axial)"), ("right", "Right rolling plane (axial)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("New400Hz PPO nominal μ=0.7: train rows × validation columns\nShared world bounds; no alignment; arrows at2/3/4s (fixed0.18m, not speed scale)", fontsize=12)
    fig.tight_layout(rect=(0., .07, 1., .94))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"shared_world_bounds": {"xlim": [float(low[0]), float(high[0])], "ylim": [float(low[1]), float(high[1])]},
            "arrow_length_m": arrow_length, "arrow_timestamps_s": [2., 3., 4.], "colors": COLORS,
            "wheel_arrow_kind": "axial rolling plane from wheel local+Y axle cross world+Z; spin invariant"}


def render_curves(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    fields = [("XY RMSE (m)", lambda c: c["tracking"]["xy_rmse_m"]),
        ("XY peak (m)", lambda c: c["tracking"]["xy_peak_m"]),
        ("Speed retention", lambda c: c["tracking"]["speed_retention"]),
        ("Reference-relative β RMSE (deg)", lambda c: c["tracking"]["beta_rmse_deg"]),
        ("β valid native samples", lambda c: c["tracking"]["beta_valid_samples"]),
        ("World ωXY10<f<100Hz RMS (rad/s)", lambda c: c["world_angular_xy_common10_100"]["common_band_rms"]),
        ("Both wheels load eligible fraction", lambda c: c["contact"]["load_eligibility"]["both"]["sample_duty_fraction"]),
        ("Neither wheel load eligible fraction (NOT flight)", lambda c: c["contact"]["load_eligibility"]["none"]["sample_duty_fraction"]),
        ("Maximum tilt (deg; diagnostic)", lambda c: c["maximum_tilt_deg"])]
    for origin in ("physx", "mjlab"):
        for engine in ENGINE_LABELS:
            rows = sorted((row for row in summary["rows"] if row["training_backend"] == origin and row["engine_tag"] == engine), key=lambda row: row["mu"])
            for ax, (label, getter) in zip(axes.flat, fields):
                ax.plot([row["mu"] for row in rows], [getter(row["windows"]["core"]) for row in rows], ".-",
                        label=TRAIN_LABELS[origin] + "→" + ENGINE_LABELS[engine], markersize=4)
                ax.set(xscale="log", ylabel=label, xlabel="Target effective pair μ")
                ax.grid(alpha=.2)
    axes[0, 0].axhline(.15, color="grey", ls="--", lw=.8)
    axes[0, 1].axhline(.30, color="grey", ls="--", lw=.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("New400Hz PPO:17μ closed-loop matrix | core1.44–4.0s, N=1/cell\nXY thresholds exploratory; no height, airborne, beta or contact failure gate", fontsize=12)
    fig.tight_layout(rect=(0., .07, 1., .94))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def compact_metrics(summary):
    """Concise reader-facing table; discrete tested mu, not inferred ranges."""
    table = []
    for row in summary["rows"]:
        core = row["windows"]["core"]
        tracking = core["tracking"]
        load = core["contact"]["load_eligibility"]
        event = core["persistent_tilt_diagnostics"]["30"]["first_persistent_event"]
        table.append({"training_backend": row["training_backend"], "validation_backend": row["validation_backend"],
            "mu": row["mu"], "xy_rmse_m": tracking["xy_rmse_m"], "xy_peak_m": tracking["xy_peak_m"],
            "speed_retention": tracking["speed_retention"], "beta_rmse_deg": tracking["beta_rmse_deg"],
            "beta_valid_samples": tracking["beta_valid_samples"],
            "world_angular_xy_common10_100_rms_rad_s": core["world_angular_xy_common10_100"]["common_band_rms"],
            "both_wheels_load_time_duty_fraction": load["both"]["time_duty_fraction_zoh"],
            "neither_wheel_load_time_duty_fraction": load["none"]["time_duty_fraction_zoh"],
            "first_core_tilt30_persistent_onset_s": event["onset_s"] if event else None,
            "large_xy_exploratory": row["xy_screen"]["large_xy_error"]})
    groups = []
    for origin in ("physx", "mjlab"):
        for engine in ENGINE_LABELS.values():
            selected = [row for row in table if row["training_backend"] == origin and row["validation_backend"] == engine]
            groups.append({"training_backend": origin, "validation_backend": engine,
                "tested_XY_pass_mu": sorted(row["mu"] for row in selected if not row["large_xy_exploratory"]),
                "tested_large_XY_mu": sorted(row["mu"] for row in selected if row["large_xy_exploratory"])})
    overview = {"kind": "new400hz_PPO_matrix_concise_overview", "sample_size_per_cell": 1,
        "visual_reviewed": False, "full_results": "summary.json", "all_cells_table": "metric_table.csv",
        "core_interval_s": [1.44, 4.], "nominal_mu": .7,
        "nominal_four_cells": [row for row in table if row["mu"] == .7], "tested_mu_lists": groups,
        "XY_screen": "exploratory: RMSE>0.15m OR peak>0.30m; no height/contact/tilt/beta failure gate",
        "interpretation": "N=1 descriptive. Lists are discrete tested coefficients, NOT continuous feasible ranges, statistical rates or stable-contact certification. Pending visual review.",
        "load_duty_definition": "native count>0 AND ground-normal Fz>5N, time-weighted ZOH; backend-specific telemetry, not flight",
        "tilt_onset_definition": "first core-window tilt>=30deg sustained>=0.1s; diagnostic only, window censoring in full summary"}
    return table, overview


def write_compact_outputs(summary, output):
    table, overview = compact_metrics(summary)
    write_json(output / "overview.json", overview)
    with (output / "metric_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-root", type=Path, required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, required=True)
    result.add_argument("--cohort-protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args):
    require(not args.output_dir.exists(), "Preserve existing matrix analysis; use a fresh output directory")
    nominal, reference, summary = load_matrix(args.input_root, args.reference, args.model,
                                            args.legacy_mujoco_evaluator, args.cohort_protocol)
    args.output_dir.mkdir(parents=True)
    summary["plot_settings"] = render_nominal(nominal, reference, summary, args.output_dir / "nominal_2x2_topview.png")
    render_curves(summary, args.output_dir / "friction_grid_curves.png")
    write_compact_outputs(summary, args.output_dir)
    summary["artifacts"] = {name: sha256(args.output_dir / name) for name in
        ("nominal_2x2_topview.png", "friction_grid_curves.png", "overview.json", "metric_table.csv")}
    write_json(args.output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run(parser().parse_args())
