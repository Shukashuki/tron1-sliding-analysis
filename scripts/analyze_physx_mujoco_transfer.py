"""Replay published controller recordings by controller origin and plant engine.

Asset-free analysis only: no policy inference, optimization, or physics stepping.
The source bundle contains the original measured physical clocks, COM velocities,
contacts and FK wheel geometry. The PPO origin is PhysX training; the CEM origin
is a MuJoCo CPU predictor that replans from each plant's own measured state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analyze_controller_pair import pair_xy_gap
from controller_metrics import heading_signals, summarize_turn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/nmpc_primary_friction_pair_development_20261016_v1"
DEFAULT_OUTPUT = ROOT / "results/physx_mujoco_controller_transfer_20261021_v1"
SOURCE_SUMMARY_SHA256 = "f157e7894890560f330c3d3024cac9d4810c6db852f57b5dfe0939d5e5465191"
GRID = [.01, .03, .05, .10, .15, .20, .30, .45, .60, .70, .80, 1., 1.2, 1.4, 1.6, 1.8, 2.]
CORE = [1.44, 4.0]
ENGINES = {"isaac": "PhysX", "mujoco": "MuJoCo"}
ORIGINS = {
    "PPO": {"kind": "policy_training", "physics_engine": "PhysX",
            "native_plant": "PhysX", "cross_plant": "MuJoCo",
            "execution": "same frozen checkpoint; independent closed-loop plant episodes"},
    "CEM": {"kind": "mpc_prediction", "physics_engine": "MuJoCo CPU",
            "native_plant": "MuJoCo", "cross_plant": "PhysX", "predictor_ground_mu": .7,
            "execution": "same predictor/objective/search; online replanning from each plant's measured state"},
}
FIELDS = ("time_s", "physics_dt_s", "position", "quaternion", "com_velocity", "angular_velocity",
          "wheel_pos", "contact_count", "contact_force", "contact_slip")
COLORS = {"PPO": "#2775b6", "CEM": "#dd7628"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case(mu):
    return ("mu" + format(mu, ".3f")).replace(".", "p")


def read_trace(archive, prefix, *, reference=False):
    trace = {name: archive[prefix + "_" + name].copy() for name in FIELDS
             if prefix + "_" + name in archive.files}
    n, dt = (349, .02) if reference else (1397, .005)
    require(trace["time_s"].shape == (n,) and float(trace["physics_dt_s"]) == dt,
            "Wrong sample count/dt: " + prefix)
    require(np.allclose(trace["time_s"], np.arange(n) * dt, atol=1e-9, rtol=0),
            "Non-native or incomplete clock: " + prefix)
    require(all(not value.dtype.hasobject and np.isfinite(value).all() for value in trace.values()),
            "Invalid compact physical arrays: " + prefix)
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
                              archive[prefix + "_wheel_quaternion_wxyz"],
                              archive[prefix + "_wheel_center_velocity_w"])
    for name, value in signals.items():
        require(np.array_equal(value, archive[prefix + "_" + name]),
                "Heading/speed exact replay failed: " + prefix + "_" + name)
    return trace, signals


def replay_metrics(trace, signals, reference, reference_signals):
    contacts = {} if "contact_count" not in trace else {
        "wheel_ground_contact_count": trace["contact_count"],
        "wheel_ground_force_w": trace["contact_force"],
        "wheel_contact_lateral_slip_m_s": trace["contact_slip"],
    }
    return summarize_turn(trace["time_s"], trace["position"], signals,
                          reference["time_s"], reference["position"][:, :2],
                          reference_signals["speed_m_s"], CORE, **contacts)


def audited_inputs(source):
    summary_path = source / "summary.json"
    require(digest(summary_path) == SOURCE_SUMMARY_SHA256, "Published source summary identity changed")
    summary = json.loads(summary_path.read_text())
    require(summary["kind"] == "unqualified_two_engine_cem_friction_development"
            and summary["complete_two_engine_grid"] is True
            and summary["expected_mu"] == GRID
            and summary["prediction_mu_scenarios"] == [.7]
            and summary["policy_feature_scope"] == "original_physical_recording_dynamics"
            and summary["reference_spec"]["evaluation_core_interval_s"] == CORE,
            "Wrong controller/reference/grid scope")
    trace_path = source / "development_trace.npz"
    artifact = summary["artifacts"]["compact_trace"]
    require(artifact["file"] == trace_path.name and digest(trace_path) == artifact["sha256"],
            "Published compact trace identity changed")
    for name, expected in summary["analysis_source_sha256"].items():
        require(digest(ROOT / "scripts" / name) == expected, "Source analysis identity changed: " + name)
    policy_dir = ROOT / "results/policy_primary_friction_dynamics_20261016_v1"
    for name, expected in summary["policy_dynamics_bundle"]["input_sha256"].items():
        require(digest(policy_dir / name) == expected, "Published policy dynamics identity changed: " + name)
    policy = json.loads((policy_dir / "summary.json").read_text())
    require(policy["reference_spec"] == summary["reference_spec"], "Policy reference provenance differs")
    require(policy["input_sha256"]["model"] == summary["input_sha256"]["model"],
            "Policy and MPC FK model identities differ")
    for mu in GRID:
        evidence = summary["isaac_runs"][case(mu)]
        require(evidence["actual_plant_mu"] == mu and evidence["complete_physical_evidence"] is True
                and evidence["censored"] is False, "Incomplete CEM plant evidence")
        training = policy["runs"]["PPO_" + case(mu)]["training"]
        require(training["verified_formal_training"] is True
                and training["checkpoint_bytes_verified"] is True
                and training["checkpoint_sha256"] == summary["policy_grids"]["PPO"]["checkpoint_sha256"],
                "PPO frozen-checkpoint/training provenance differs")
    return summary, trace_path


def analyze(source):
    published, trace_path = audited_inputs(source)
    rows, traces, flat = [], {}, []
    with np.load(trace_path, allow_pickle=False) as archive:
        reference, ref_signals = read_trace(archive, "reference", reference=True)
        for original in published["rows"]:
            mu = original["mu"]
            require(mu in GRID, "Undeclared coefficient")
            for controller, label in (("PPO", "PPO"), ("CEM", "NMPC")):
                metrics, pair = {}, {}
                for engine, plant in ENGINES.items():
                    prefix = ("ppo_" + case(mu) + "_" + engine if controller == "PPO"
                              else engine + "_" + case(mu))
                    trace, signals = read_trace(archive, prefix)
                    measured = replay_metrics(trace, signals, reference, ref_signals)
                    require(measured == original[label][engine],
                            "Core metric exact replay failed: " + prefix)
                    metrics[plant], pair[engine] = measured, trace
                    traces[(controller, mu, plant)] = (trace, signals)
                    grounded = measured["ground_contact_heading_geometry"]
                    row = {"mu": mu, "controller": controller,
                           "controller_origin": ORIGINS[controller]["physics_engine"], "plant": plant,
                           "transfer": "native" if plant == ORIGINS[controller]["native_plant"] else "cross_engine",
                           "core_xy_rmse_m": measured["reference_xy_rmse_m"],
                           "core_xy_peak_m": measured["reference_xy_peak_m"],
                           "core_speed_retention_ratio": measured["mean_speed_retention_ratio"],
                           "core_mean_speed_m_s": measured["mean_speed_m_s"],
                           "core_net_course_change_deg": measured["net_course_change_deg"],
                           "core_xy_screen_exceeded": measured["large_xy_exploratory_screen"],
                           "core_body_beta_mean_abs_deg": measured["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"]}
                    for j, side in enumerate(("left", "right")):
                        row[side + "_contact_fraction"] = grounded["contact_fraction_left_right"][j]
                        row[side + "_grounded_wheel_center_heading_offset_mean_abs_deg"] = grounded["per_wheel"][side]["wheel_center_course_to_wheel_gamma"]["mean_absolute_deg"]
                        row[side + "_grounded_lateral_slip_rms_m_s"] = measured["ground_contact_lateral_slip_m_s"][side]
                    flat.append(row)
                gap = pair_xy_gap(pair["isaac"], pair["mujoco"], CORE)
                require(gap == original[label]["sim2sim"], "Pair gap exact replay failed")
                native, cross = (metrics[ORIGINS[controller][key]] for key in ("native_plant", "cross_plant"))
                rows.append({"mu": mu, "controller": controller, "plants": metrics, "paired_xy_gap": gap,
                             "cross_minus_native_xy_rmse_m": cross["reference_xy_rmse_m"] - native["reference_xy_rmse_m"],
                             "cross_minus_native_speed_retention_ratio": cross["mean_speed_retention_ratio"] - native["mean_speed_retention_ratio"]})
    require(len(rows) == 34 and len(flat) == 68, "Require all 68 original physical episodes")
    summary = {"schema_version": 1, "kind": "physx_mujoco_existing_controller_transfer_replay",
               "reference_id": published["reference_id"], "reference_spec": published["reference_spec"],
               "source_bundle": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
               "source_sha256": {"summary.json": digest(source / "summary.json"), "development_trace.npz": digest(trace_path)},
               "evaluation_mu": GRID, "core_window_s": CORE, "controller_origins": ORIGINS,
               "ppo_checkpoint_sha256": published["policy_grids"]["PPO"]["checkpoint_sha256"],
               "ppo_training_budget": published["policy_grids"]["PPO"]["training_budget"],
               "cem_effective_config": published["effective_cem_config"],
               "reference_and_model_sha256": published["input_sha256"],
               "metric_definitions": {
                   "body_beta": "wrapped world-XY COM course minus projected base +X heading; may differ from body-frame reward beta under pitch/roll",
                   "wheel_heading": "world(local wheel +Y axle) crossed with world +Z, axial modulo 180 degrees",
                   "ground_contact_gate": "measured wheel ground-contact count >0 and world upward force >5 N",
                   "wheel_center_offset": "axial wheel-center course minus wheel rolling-plane heading, evaluated only with speed/geometry/contact eligibility",
                   "contact_slip": "recorded force-weighted lateral contact-slip RMS; distinct from wheel-heading geometry",
                   "speed_retention": "mean measured base COM XY speed divided by mean interpolated reference COM XY speed on the same native samples",
               },
               "audit": {"source_npz_hash_verified": True, "source_analysis_hashes_verified": True,
                         "policy_provenance_bundle_verified": True, "physical_episode_metrics_exactly_replayed": 68,
                         "paired_xy_gaps_exactly_replayed": 34, "heading_and_com_speed_exactly_replayed": True,
                         "native_clock_s": .005, "samples_per_episode": 1397, "core_samples": 513,
                         "alignment": "same world coordinates and physical time; no position or time registration"},
               "xy_exploratory_screen_m": {"rmse": .15, "peak": .30}, "rows": rows,
               "counts_below_both_xy_limits": {
                   controller: {plant: sum(not row["plants"][plant]["large_xy_exploratory_screen"]
                                          for row in rows if row["controller"] == controller)
                                for plant in ENGINES.values()} for controller in ORIGINS},
               "new_training": False, "new_simulation": False, "new_policy_inference": False,
               "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
               "limitations": [
                   "One seed-42 episode per condition; counts are descriptive and are not success probabilities.",
                   "PPO uses a PhysX-trained checkpoint; CEM uses a MuJoCo CPU predictor. Algorithm and origin are confounded, so this is not a causal physics-backend-only comparison.",
                   "CEM replans independently in each plant; paired paths are closed-loop outcomes and do not share an identical action sequence.",
                   "CEM remains an offline development controller: solve times exceed the 20 ms control period, without wall-clock action delay injected.",
                   "Equal numeric ground friction does not calibrate equivalent wheel-ground contact laws across engines.",
                   "Body beta and wheel-heading offsets are descriptive features; geometric wheel-plane angles are not measured contact-patch slip.",
                   "No PhysX-predictor MPC or MuJoCo-trained PPO reverse-origin branch is available in this replay."]}
    return summary, flat, {"reference": (reference, ref_signals), "cases": traces}


def snapshot(ax, trace, signals, index, *, alpha=1.):
    xy = trace["position"][index, :2]
    for name, color, length in (("body_heading_rad", "black", .19), ("course_rad", "#24936a", .16)):
        if name == "course_rad" and not signals["root_beta_valid"][index]:
            continue
        angle = signals[name][index]
        ax.arrow(*xy, length * np.cos(angle), length * np.sin(angle), color=color, alpha=alpha,
                 width=.004, head_width=.04, length_includes_head=True, zorder=6)
    for side, color in enumerate(("#d54843", "#8a54aa")):
        angle = signals["wheel_rolling_plane_heading_rad"][index, side]
        direction = .085 * np.array([np.cos(angle), np.sin(angle)])
        center = trace["wheel_pos"][index, side, :2]
        grounded = trace["contact_count"][index, side] > 0 and trace["contact_force"][index, side, 2] > 5.
        line = center + np.array([-direction, direction])
        ax.plot(line[:, 0], line[:, 1], color=color, lw=2.2, alpha=alpha * (1. if grounded else .25), zorder=7)


def render_nominal(summary, bundle, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, sharex=True, sharey=True)
    reference, _ = bundle["reference"]
    visible_ref = reference["time_s"] <= 4.96 + 1e-9
    for row, controller in enumerate(ORIGINS):
        metrics = next(item for item in summary["rows"] if item["controller"] == controller and item["mu"] == .7)
        for col, plant in enumerate(ENGINES.values()):
            ax = axes[row, col]
            trace, signals = bundle["cases"][(controller, .7, plant)]
            t, xy = trace["time_s"], trace["position"][:, :2]
            visible = t <= 4.96 + 1e-9
            core = (t >= CORE[0] - 1e-9) & (t <= CORE[1] + 1e-9)
            ax.plot(reference["position"][visible_ref, 0], reference["position"][visible_ref, 1], "--", color=".4", lw=1.2)
            ax.plot(xy[visible, 0], xy[visible, 1], color=COLORS[controller], lw=1., alpha=.4)
            ax.plot(xy[core, 0], xy[core, 1], color=COLORS[controller], lw=2.)
            for time in (1.6, 2.2, 2.8, 3.4, 4.):
                snapshot(ax, trace, signals, int(np.argmin(np.abs(t - time))))
            m = metrics["plants"][plant]
            relation = "native" if plant == ORIGINS[controller]["native_plant"] else "cross engine"
            ax.set(title=f"{controller}: {ORIGINS[controller]['physics_engine']} origin → {plant} ({relation})\n"
                         f"XY RMSE {m['reference_xy_rmse_m']:.3f} m · speed {100*m['mean_speed_retention_ratio']:.1f}%",
                   xlabel="World X (m)", ylabel="World Y (m)")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([], [], ls="--", color=".4", label="Reference"),
                        Line2D([], [], color="black", marker=">", label="Body +X heading"),
                        Line2D([], [], color="#24936a", marker=">", label="COM velocity course"),
                        Line2D([], [], color="#d54843", label="Left wheel plane"),
                        Line2D([], [], color="#8a54aa", label="Right wheel plane; faint = contact gate not met")],
               loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Same original controllers, different physical plants · traj_4 · μ=0.70 · β reward on", fontsize=12)
    fig.savefig(path, dpi=150); plt.close(fig)


def render_friction(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for controller in ORIGINS:
        rows = [row for row in summary["rows"] if row["controller"] == controller]
        for plant, style in (("PhysX", "-"), ("MuJoCo", "--")):
            for ax, field in zip(axes[:2], ("reference_xy_rmse_m", "mean_speed_retention_ratio")):
                values = [row["plants"][plant][field] for row in rows]
                ax.plot(GRID, values, style, color=COLORS[controller], marker="o", ms=3., label=f"{controller} → {plant}")
        axes[2].plot(GRID, [row["paired_xy_gap"]["xy_rmse_m"] for row in rows],
                     color=COLORS[controller], marker="o", ms=3., label=controller)
    axes[0].axhline(.15, color=".4", ls=":", lw=1., label="RMSE screen 0.15 m")
    axes[0].set(yscale="log", ylabel="Core XY RMSE (m)", title="Reference tracking; peak screen also applies")
    axes[1].axhline(1., color=".4", ls=":", lw=1.)
    axes[1].set(ylabel="Core mean-speed retention ratio", title="Speed relative to the reference")
    axes[2].set(ylabel="Core paired XY RMSE (m)", title="PhysX–MuJoCo trajectory gap")
    for ax in axes:
        ax.set(xlabel="Plant ground friction μ", xscale="log")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Fixed controllers across all 17 ground coefficients · one original episode per cell", fontsize=12)
    fig.savefig(path, dpi=150); plt.close(fig)


def render_video(summary, bundle, path):
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return False
    from matplotlib.animation import FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = encoder
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained", sharex=True, sharey=True)
    reference, _ = bundle["reference"]
    writer = FFMpegWriter(fps=25, codec="libx264", extra_args=["-crf", "23", "-pix_fmt", "yuv420p"])
    all_xy = [reference["position"][:, :2]]
    for controller in ORIGINS:
        for plant in ENGINES.values():
            trace, _ = bundle["cases"][(controller, .7, plant)]
            all_xy.extend((trace["position"][:, :2], trace["wheel_pos"][:, :, :2].reshape(-1, 2)))
    points = np.concatenate(all_xy)
    low, high = points.min(axis=0) - .25, points.max(axis=0) + .25
    artists = {}
    for row, controller in enumerate(ORIGINS):
        for col, plant in enumerate(ENGINES.values()):
            ax = axes[row, col]
            ax.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color=".65", lw=1.)
            path_line, = ax.plot([], [], color=COLORS[controller], lw=1.8)
            body = ax.quiver([0.], [0.], [.19], [0.], color="black", angles="xy", scale_units="xy", scale=1., width=.006)
            course = ax.quiver([0.], [0.], [.16], [0.], color="#24936a", angles="xy", scale_units="xy", scale=1., width=.004)
            wheels = [ax.plot([], [], color=color, lw=2.2)[0] for color in ("#d54843", "#8a54aa")]
            dot, = ax.plot([], [], "o", color=COLORS[controller], ms=4.)
            artists[(controller, plant)] = (path_line, body, course, wheels, dot)
            native = plant == ORIGINS[controller]["native_plant"]
            ax.set(title=f"{controller}: {ORIGINS[controller]['physics_engine']} → {plant} ({'native' if native else 'cross'})",
                   xlabel="World X (m)", ylabel="World Y (m)",
                   xlim=(low[0], high[0]), ylim=(low[1], high[1]))
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2)
    title = fig.suptitle("traj_4 · μ=0.70 · t=0.00 s · recorded physical trajectories\n"
                         "Black: body | green: velocity | red/purple: wheels | faint: contact gate not met", fontsize=10)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    with writer.saving(fig, str(path), dpi=100):
        for time in np.arange(0., 7., .04):
            for controller in ORIGINS:
                for plant in ENGINES.values():
                    trace, signals = bundle["cases"][(controller, .7, plant)]
                    xy, t = trace["position"][:, :2], trace["time_s"]
                    current = int(np.argmin(np.abs(t - time)))
                    path_line, body, course, wheels, dot = artists[(controller, plant)]
                    path_line.set_data(xy[:current+1, 0], xy[:current+1, 1])
                    dot.set_data([xy[current, 0]], [xy[current, 1]])
                    for arrow, name, length in ((body, "body_heading_rad", .19), (course, "course_rad", .16)):
                        angle = signals[name][current]
                        arrow.set_offsets(xy[current][None, :])
                        arrow.set_UVC([length * np.cos(angle)], [length * np.sin(angle)])
                    course.set_visible(bool(signals["root_beta_valid"][current]))
                    for side, line in enumerate(wheels):
                        angle = signals["wheel_rolling_plane_heading_rad"][current, side]
                        direction = .085 * np.array([np.cos(angle), np.sin(angle)])
                        center = trace["wheel_pos"][current, side, :2]
                        segment = center + np.array([-direction, direction])
                        line.set_data(segment[:, 0], segment[:, 1])
                        grounded = trace["contact_count"][current, side] > 0 and trace["contact_force"][current, side, 2] > 5.
                        line.set_alpha(1. if grounded else .25)
            title.set_text(f"traj_4 · μ=0.70 · t={time:.2f} s · recorded physical trajectories\n"
                           "Black: body | green: velocity | red/purple: wheels | faint: contact gate not met")
            writer.grab_frame()
    plt.close(fig)
    return True


def run(args):
    output = args.output_dir
    require(not output.exists(), "Preserve existing outputs; choose a new output directory")
    summary, flat, bundle = analyze(args.source_dir)
    output.mkdir(parents=True)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat)
    render_nominal(summary, bundle, output / "nominal_topdown.png")
    render_friction(summary, output / "friction_sensitivity.png")
    if args.video:
        summary["video_created"] = render_video(summary, bundle, output / "nominal_topdown.mp4")
    summary["analysis_source_sha256"] = {
        name: digest(ROOT / "scripts" / name) for name in
        ("analyze_physx_mujoco_transfer.py", "analyze_controller_pair.py", "controller_metrics.py", "nmpc_tracking_objective.py")}
    summary["artifacts"] = {path.name: {"file": path.name, "sha256": digest(path)}
                            for path in sorted(output.iterdir())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output), "audit": summary["audit"],
                      "counts_below_both_xy_limits": summary["counts_below_both_xy_limits"],
                      "artifacts": list(summary["artifacts"])}, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", type=Path, default=SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--video", action="store_true", help="Render recorded nominal episodes to MP4 if ffmpeg is available")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
