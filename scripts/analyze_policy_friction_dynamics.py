"""Replay existing PPO/SAC grid dynamics, without training or policy inference.

Raw files must match the already published grid's trace/report identities.
Measured original-link COM velocities and recorded ground contacts are used;
wheel-plane geometry is reconstructed kinematically, never by stepping dynamics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_controller_pair import (
    audit_reports, audit_training, load_arrays, plant_source_provenance,
    posture_diagnostic, read_json, reference_trace, sha256, source_model,
    summarize_pair,
)
from analyze_nmpc_friction_development import (
    case_name, expected_grid, load_policy_grid,
)
from analyze_nmpc_nominal_development import require
from analyze_tvlqr_pilot import public_compact
from controller_references import DEFAULT_CATALOG, PRIMARY_REFERENCE_ID, ROOT, verify_reference_pair
from controller_trace import WheelKinematics, native_trace


def audit_policy_clock(trace):
    """Full first episode only; do not infer a speed from position differences."""
    require(len(trace["time_s"]) == 1397 and trace["physics_dt_s"] == .005
            and np.allclose(trace["time_s"], np.arange(1397)*.005, atol=1e-9, rtol=0),
            "Require original 1397 consecutive physical samples")


def audited_raw_case(raw_root, algorithm, mu, provenance):
    """Bind all four original archives to previously published case hashes."""
    folder = Path(raw_root)/algorithm/case_name(mu)
    files = {"isaac_trace": folder/"isaac/physics_trajectory.npz",
             "isaac_report": folder/"isaac/report.json",
             "mujoco_trace": folder/"mujoco/rollout.npz",
             "mujoco_report": folder/"mujoco/report.json"}
    hashes = {name: sha256(path) for name, path in files.items()}
    expected = {name: provenance[name+"_sha256"] for name in files}
    require(hashes == expected, "Original policy files differ from published grid capture")
    return files, hashes


def append_nonwheel(raw, trace, engine):
    """Only copy present recorded nonwheel arrays; absence is not zero contact."""
    count = "nonwheel_ground_contact_count"
    force = "nonwheel_ground_contact_force_norm_n"
    if count not in raw and force not in raw:
        return trace
    require(count in raw and force in raw, "Incomplete nonwheel-contact recording")
    for name in (count, force):
        value = np.asarray(raw[name])
        if engine == "isaac":
            require(value.shape == (1397, 1), "Unexpected Isaac nonwheel-contact shape")
            value = value[:, 0]
        require(value.shape == (1397,) and np.isfinite(value).all() and np.all(value >= 0),
                "Invalid measured nonwheel-contact field")
        if name == count:
            require(np.array_equal(value, np.round(value)), "Noninteger contact counts")
        trace[name] = value.copy()
    return trace


def reference_numeric_representation_audit(motion, robot_reference):
    """Prove the locked export's f64 -> f32 representation, not alignment.

    Identical timestamps and exactly cast knot values imply that the two
    piecewise-linear reference curves differ everywhere by at most delta:
    interpolation is a convex combination of the endpoint differences.
    Reverse triangle inequalities then bound both XY RMSE and peak changes
    by delta for the SAME measured positions and sample times.
    """
    bodies = tuple(np.asarray(motion["body_names"]).tolist())
    require("base_Link" in bodies and len(bodies) == len(set(bodies)), "Missing unique reference base body")
    original = np.asarray(robot_reference["root_target_position_m"])
    exported = np.asarray(motion["body_pos_w"])[:, bodies.index("base_Link"), :]
    times = np.asarray(motion["time_s"])
    require(original.dtype == np.float64 and exported.dtype == np.float32
            and original.shape == exported.shape == (len(times), 3)
            and times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all()
            and np.all(np.diff(times) > 0) and np.array_equal(times, robot_reference["time_s"])
            and np.isfinite(original).all() and np.isfinite(exported).all(),
            "Reference representation proof requires finite f64/f32 poses on identical native timestamps")
    cast = original[:, :2].astype(np.float32)
    require(np.array_equal(exported[:, :2].view(np.uint32), cast.view(np.uint32)),
            "Motion XY is not the exact float32 cast of the locked robot-reference XY; no alignment allowed")
    difference = exported[:, :2].astype(np.float64)-original[:, :2]
    delta = float(np.linalg.norm(difference, axis=1).max())
    return {"kind": "verified_locked_reference_float64_to_float32_xy_export",
        "motion_xy_dtype": "float32", "robot_reference_xy_dtype": "float64",
        "reference_knots": len(times), "identical_reference_timestamps": True,
        "float32_cast_bitwise_equal": True,
        "maximum_knot_xy_difference_norm_m": delta,
        "maximum_knot_xy_component_difference_abs_m": float(np.abs(difference).max()),
        "maximum_reference_coordinate_abs_m": float(max(np.abs(original[:, :2]).max(), np.abs(exported[:, :2]).max())),
        "analytic_metric_change_bound_m": delta,
        "bound_reason": "Identical-knot-time linear interpolation is convex; reverse triangle inequalities bound same-sample XY RMSE and peak changes by the maximum reference-knot vector difference.",
        "translation_rotation_or_time_warp_applied": False,
        "historical_reference_or_metrics_modified": False}


def audit_xy_replay(canonical, published, *, reference_audit=None, observed_xy=None):
    """Retain both numeric representations under a proven metric-change bound.

    Without the explicit reference proof this is an EXACT check, not a looser
    generic tolerance. The small additional float64 allowance uses gamma_n
    with a conservative arithmetic budget (16*N+64) and coordinate scale;
    it is not fitted to an observed metric discrepancy.
    """
    require(published.get("core_complete") is True and canonical.get("coverage_complete") is True,
            "Require complete original and canonical XY cores")
    delta = roundoff = 0.
    if reference_audit is not None:
        require(reference_audit.get("kind") == "verified_locked_reference_float64_to_float32_xy_export"
                and reference_audit.get("float32_cast_bitwise_equal") is True
                and reference_audit.get("identical_reference_timestamps") is True
                and reference_audit.get("translation_rotation_or_time_warp_applied") is False,
                "A verified reference-representation proof is required, not an arbitrary tolerance")
        delta = reference_audit.get("analytic_metric_change_bound_m")
        require(type(delta) in (int, float) and np.isfinite(delta) and delta >= 0
                and delta == reference_audit.get("maximum_knot_xy_difference_norm_m"), "Invalid analytic reference bound")
        count = canonical.get("samples_in_window")
        require(type(count) is int and count >= 2 and count == published.get("observed_core_samples"),
                "Canonical and legacy metric samples must be identical")
        xy = np.asarray(observed_xy, dtype=float)
        require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy) >= count and np.isfinite(xy).all(),
                "Measured original XY is required to derive a scale-dependent roundoff allowance")
        operations = 16*count+64
        epsilon = np.finfo(np.float64).eps
        gamma = operations*epsilon/(1.-operations*epsilon)
        scale = max(1., float(np.abs(xy).max()), reference_audit["maximum_reference_coordinate_abs_m"])
        # A factor of eight covers two coordinates, two reference evaluations,
        # subtraction/norm scaling and conservative accumulation. This is a
        # numerical replay guard, not an interval-arithmetic certificate.
        roundoff = float(8.*gamma*scale)
    differences = {}; legacy = {}; current = {}
    for actual, original in (("reference_xy_rmse_m", "full_core_reference_xy_rmse_m"),
                             ("reference_xy_peak_m", "full_core_reference_xy_peak_m")):
        values = (canonical[actual], published[original])
        require(all(type(value) in (int, float) and np.isfinite(value) and value >= 0 for value in values),
                "Invalid canonical/legacy XY metric")
        change = abs(values[0]-values[1])
        require(change <= delta+roundoff,
                "Canonical policy XY metric exceeds the proven reference-representation bound")
        differences[actual] = float(change); legacy[actual] = float(values[1]); current[actual] = float(values[0])
    require(canonical["large_xy_exploratory_screen"] == published["large_reference_xy_screen"],
            "Canonical policy XY verdict differs from original grid")
    return {"legacy_robot_reference_float64_metrics": legacy,
        "canonical_motion_float32_metrics": current, "absolute_metric_changes_m": differences,
        "analytic_reference_bound_m": float(delta), "float64_roundoff_allowance_m": roundoff,
        "combined_metric_change_bound_m": float(delta+roundoff),
        "roundoff_rule": "8 * gamma_(16*N+64) * max(1, maximum absolute measured/reference XY coordinate), binary64 epsilon; not fitted to metric differences",
        "xy_screen_unchanged": True, "legacy_xy_screen": published["large_reference_xy_screen"],
        "canonical_xy_screen": canonical["large_xy_exploratory_screen"],
        "historical_metrics_overwritten": False, "raw_xy_translation_rotation_or_time_warp_applied": False}


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = [.1, .3, .7, 1.6, 2.]
    fig, axes = plt.subplots(5, 3, figsize=(13.5, 16.5), squeeze=False)
    rt, rp = compact["reference_time_s"], compact["reference_position"]
    start, end = summary["reference_spec"]["source_demo_interval_s"]
    lo, hi = summary["reference_spec"]["evaluation_core_interval_s"]
    sel = (rt >= start-1e-9) & (rt <= end+1e-9)
    for row, mu in enumerate(selected):
        bounds = [rp[sel, :2]]
        for col, algorithm in enumerate(("reference", "PPO", "SAC")):
            ax = axes[row, col]
            ax.plot(rp[sel, 0], rp[sel, 1], "--", color=".75", lw=1)
            labels = [("reference", ".3")] if col == 0 else [
                (algorithm+"_"+case_name(mu)+"_isaac", "tab:blue"),
                (algorithm+"_"+case_name(mu)+"_mujoco", "tab:orange")]
            for label, color in labels:
                time, pos = compact[label+"_time_s"], compact[label+"_position"]
                selected_time = (time >= start-1e-9) & (time <= end+1e-9)
                core = selected_time & (time >= lo-1e-9) & (time <= hi+1e-9)
                ax.plot(pos[selected_time, 0], pos[selected_time, 1], color=color, lw=.8, alpha=.45)
                ax.plot(pos[core, 0], pos[core, 1], color=color, lw=1.8)
                indices = np.flatnonzero(selected_time)[::max(1, round(.4/float(compact[label+"_physics_dt_s"])))]
                wheels = compact[label+"_wheel_pos"]
                for side in range(2):
                    for k in indices:
                        if not compact[label+"_root_course_to_wheel_valid"][k, side]:
                            continue
                        grounded = label == "reference" or (compact[label+"_contact_count"][k, side] > 0
                            and compact[label+"_contact_force"][k, side, 2] > 5.)
                        heading = compact[label+"_wheel_rolling_plane_heading_rad"][k, side]
                        delta = .055*np.array([np.cos(heading), np.sin(heading)])
                        segment = wheels[k, side, :2]+np.array([-delta, delta])
                        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=.8, alpha=.8 if grounded else .12)
                arrows = indices[compact[label+"_root_beta_valid"][indices]]
                course = compact[label+"_course_rad"][arrows]
                ax.quiver(pos[arrows, 0], pos[arrows, 1], np.cos(course), np.sin(course), color=color,
                          angles="xy", scale_units="xy", scale=7, width=.003)
                bounds.extend((pos[selected_time, :2], wheels[selected_time, :, :2].reshape(-1, 2)))
            ax.set(title=f"μ={mu:g} · "+("Kinematic reference" if col == 0 else algorithm),
                   xlabel="World X [m]", ylabel="World Y [m]")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2)
        xy = np.concatenate(bounds); low, high = xy.min(axis=0), xy.max(axis=0)
        margin = max(.15, .06*float(np.max(high-low)))
        for ax in axes[row]:
            ax.set_xlim(low[0]-margin, high[0]+margin); ax.set_ylim(low[1]-margin, high[1]+margin)
    fig.suptitle("Existing policy recordings · measured COM speed/contact · no new inference / not ranked", fontsize=12)
    fig.legend(handles=[Line2D([], [], color="tab:blue", label="Isaac"),
        Line2D([], [], color="tab:orange", label="MuJoCo"),
        Line2D([], [], color=".3", label="Thick: core; arrows: velocity; ticks: rolling planes")],
        loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .97)); fig.savefig(output, dpi=130); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing output; choose a fresh directory")
    spec = verify_reference_pair(args.motion, args.reference_file, PRIMARY_REFERENCE_ID)
    motion = load_arrays(args.motion)
    reference = reference_trace(motion)
    numeric_reference_audit = reference_numeric_representation_audit(motion, load_arrays(args.reference_file))
    traces = {"reference": reference}; audits = {}; policies = {}
    grids = {"PPO": args.ppo_grid, "SAC": args.sac_grid}
    manifests = {"PPO": args.ppo_training_manifest, "SAC": args.sac_training_manifest}
    with source_model(args.source_root, args.model) as (model, loader):
        fk = WheelKinematics(model); recon = {"reference": fk.reconstruct(reference)}
        for algorithm, directory in grids.items():
            public, rows, xy_trace = load_policy_grid(directory, algorithm, spec)
            provenance = read_json(Path(directory)/"xy_trace.provenance.json")
            policies[algorithm] = {"source_xy_grid": public, "original_physical_archives_used": True,
                                   "measured_com_speed_and_ground_contact_available": True}
            for mu in expected_grid():
                case = case_name(mu); label = algorithm+"_"+case
                files, hashes = audited_raw_case(args.raw_root, algorithm, mu, provenance["cases"][case])
                ia, mj = read_json(files["isaac_report"]), read_json(files["mujoco_report"])
                audit = audit_reports(algorithm, spec, ia, mj,
                    isaac_trace_sha256=hashes["isaac_trace"], mujoco_trace_sha256=hashes["mujoco_trace"],
                    model_sha256=sha256(args.model), mujoco_capture_sha256=hashes["mujoco_trace"])
                require(audit["fixed_mu"] == mu, "Policy material coefficient differs from case")
                training = audit_training(manifests[algorithm], algorithm, spec, ia, mj)
                pair = {"reference": reference}; pair_recon = {"reference": recon["reference"]}
                xy_replay = {}
                for engine in ("isaac", "mujoco"):
                    raw = load_arrays(files[engine+"_trace"])
                    trace = append_nonwheel(raw, native_trace(raw, "policy_"+engine), engine)
                    audit_policy_clock(trace)
                    require(np.array_equal(trace["time_s"], xy_trace[case+"_"+engine+"_t"])
                            and np.array_equal(trace["position"][:, :2], xy_trace[case+"_"+engine+"_xy"]),
                            "Raw policy XY differs from previously published geometry")
                    full_label = label+"_"+engine
                    traces[full_label] = pair[engine] = trace
                    recon[full_label] = pair_recon[engine] = fk.reconstruct(trace)
                windows = summarize_pair(spec, pair, pair_recon)
                for engine in ("isaac", "mujoco"):
                    expected = rows[mu][engine]
                    actual = windows["core"][engine]
                    xy_replay[engine] = audit_xy_replay(actual, expected,
                        reference_audit=numeric_reference_audit, observed_xy=pair[engine]["position"][:, :2])
                audits[label] = {"algorithm": algorithm, "mu": mu, "input_sha256": hashes,
                    "audit": audit, "training": training, "windows": windows,
                    "xy_numeric_representation_replay": xy_replay,
                    "posture_diagnostics": {e: posture_diagnostic(pair[e]) for e in ("isaac", "mujoco")},
                    "reported_source_provenance": {"isaac": plant_source_provenance(ia), "mujoco": plant_source_provenance(mj)}}
                print(f"policy dynamics {algorithm} μ={mu:g}: audited original pair", flush=True)
    compact = public_compact(traces, recon)
    summary = {"schema_version": 1, "kind": "existing_policy_friction_dynamics_replay", "reference_spec": spec,
        "reference_id": PRIMARY_REFERENCE_ID, "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        "evaluation_mu": expected_grid(), "policies": policies, "runs": audits, "model_loader_audit": loader,
        "reference_numeric_representation_audit": numeric_reference_audit,
        "fk_audit": {k: v["audit"] for k, v in recon.items()},
        "input_sha256": {name: sha256(getattr(args, name)) for name in ("motion", "reference_file", "model")},
        "analysis_source_sha256": {name: sha256(ROOT/"scripts"/name) for name in (
            "analyze_policy_friction_dynamics.py", "analyze_controller_pair.py", "analyze_nmpc_friction_development.py",
            "analyze_tvlqr_pilot.py", "controller_trace.py", "controller_metrics.py", "controller_references.py")},
        **{name: False for name in ("new_policy_evaluation", "new_training", "dynamics_stepping", "learned_inference",
            "eligible_for_controller_ranking", "controlled_execution_success_claimed", "height_or_flight_failure_gate",
            "heading_offset_is_mismatch", "hardware_ready", "real_time_verified")},
        "com_speed_source": "Recorded original-link COM world velocity; not position finite differences",
        "contact_source": "Original recorded force-weighted ground-contact-point lateral slip; no contact resimulation",
        "limitations": ["Single seed/checkpoint and episode per coefficient/engine, not statistical success rates or controller-class ranking.",
            "Existing policy grids are hash-replayed; no new policy inference or change to historical results.",
            "The locked motion stores base XY as float32; the legacy grid used float64 robot-reference XY. Exact cast/timestamp checks prove a representation-only delta, and both metric versions are retained under its triangle-inequality bound; all observed XY verdicts are unchanged.",
            "Speed/slip during posture loss is descriptive, not successful upright turning.",
            "Equal numeric mu is not calibrated equivalence between engine contact/tire physics."]}
    output.mkdir(parents=True)
    np.savez_compressed(output/"policy_dynamics_trace.npz", **compact)
    render_topdown(compact, summary, output/"topdown.png")
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)}
                            for name in ("policy_dynamics_trace.npz", "topdown.png")}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "ppo-grid", "sac-grid", "ppo-training-manifest", "sac-training-manifest",
                 "motion", "reference-file", "model", "source-root", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    return result


if __name__ == "__main__":
    summary = run(parser().parse_args())
    print(json.dumps({"status": "replayed_original_recordings", "paired_policy_cases": len(summary["runs"])}))
