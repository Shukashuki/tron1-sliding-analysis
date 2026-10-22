"""Offline PhysX-trained / PhysX-evaluated PPO friction comparison.

Replay published world-coordinate recordings only: no simulator, inference,
training, pose alignment, trajectory correction, or new retargeting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from controller_metrics import summarize_turn
from nmpc_tracking_objective import quaternion_rotate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/policy_primary_friction_dynamics_20261016_v1"
DEFAULT_TRAINING = ROOT / "results/controller_primary_ppo_height_smooth_seed42_v1/training_metadata.json"
SELECTED_MU = (.7, .15, .1, .05)
SIGNALS = ("speed_m_s", "course_rad", "root_directed_beta_rad",
           "root_course_to_wheel_gamma_rad", "wheel_center_course_to_wheel_gamma_rad",
           "root_beta_valid", "root_course_to_wheel_valid", "wheel_center_course_to_wheel_valid")
COLORS = {"path": "#0072B2", "body": "#161616", "velocity": "#009E73",
          "left": "#D55E00", "right": "#8B4BA6", "reference": "#757575"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_name(mu):
    return "mu" + f"{mu:.3f}".replace(".", "p")


def prefix(mu):
    return "PPO_" + case_name(mu) + "_isaac"


def field(arrays, label, name):
    return np.asarray(arrays[label + "_" + name])


def read_json(path):
    return json.loads(Path(path).read_text())


def build_summary(arrays, source_summary, training_metadata, *, input_hashes=None):
    """Recompute all 17 PhysX cases and audit against published canonical values."""
    spec = source_summary["reference_spec"]
    require(source_summary["kind"] == "existing_policy_friction_dynamics_replay",
            "Expected original, audited physical-recording replay")
    require(spec["reference_id"] == training_metadata["reference_id"] == "drift_drl_traj4",
            "Only locked primary traj_4 is supported")
    require(spec["evaluation_core_interval_s"] == [1.44, 4.0], "Unexpected evaluation core")
    require(training_metadata["algorithm"] == "PPO" and
            training_metadata["training_status"] == "training_completed" and
            training_metadata["trained_policy_claimed"] is True, "Require completed historical PPO")
    grid = source_summary["evaluation_mu"]
    require(grid == [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.],
            "Require complete historical 17-point grid")
    source_policy = source_summary["policies"]["PPO"]["source_xy_grid"]
    for key in ("checkpoint_sha256", "training_manifest_sha256"):
        require(source_policy[key] == training_metadata[key], "Historical identity mismatch: " + key)
    require(spec["motion_npz_sha256"] == training_metadata["motion_sha256"], "Motion identity mismatch")
    if input_hashes is not None:
        require(input_hashes["training_metadata.json"] == source_policy["input_sha256"]["training_metadata.json"],
                "Training metadata bytes differ from published source")
        require(input_hashes["policy_dynamics_trace.npz"] ==
                source_summary["artifacts"]["policy_dynamics_trace.npz"]["sha256"],
                "Published physical recording archive hash mismatch")
    require(np.array_equal(arrays["wheel_order"], ["left", "right"]), "Wheel order mismatch")
    rows = []
    for mu in grid:
        label = prefix(mu)
        run = source_summary["runs"]["PPO_" + case_name(mu)]
        require(run["mu"] == mu and run["algorithm"] == "PPO" and run["audit"]["fixed_mu"] == mu,
                "Coefficient identity mismatch")
        for key in ("checkpoint_sha256", "task_profile_sha256", "height_adapter_sha256"):
            require(run["training"][key] == training_metadata[key], "Per-case training identity mismatch: " + key)
        time = field(arrays, label, "time_s")
        require(time.shape == (1397,) and
                np.allclose(time, np.arange(1397) * .005, atol=1e-9, rtol=0),
                "Require unchanged complete native 200 Hz physical clock")
        require(float(field(arrays, label, "physics_dt_s")) == .005, "Unexpected physics dt")
        velocity = field(arrays, label, "com_velocity")
        require(np.allclose(field(arrays, label, "speed_m_s"),
                            np.linalg.norm(velocity[:, :2], axis=1), atol=1e-12, rtol=1e-12),
                "Speed must use recorded original-link COM velocity, not finite differences")
        signals = {key: field(arrays, label, key) for key in SIGNALS}
        metrics = summarize_turn(time, field(arrays, label, "position"), signals,
            arrays["reference_time_s"], arrays["reference_position"][:, :2], arrays["reference_speed_m_s"],
            spec["evaluation_core_interval_s"],
            wheel_ground_contact_count=field(arrays, label, "contact_count"),
            wheel_ground_force_w=field(arrays, label, "contact_force"),
            wheel_contact_lateral_slip_m_s=field(arrays, label, "contact_slip"))
        published = run["windows"]["core"]["isaac"]
        for key in ("reference_xy_rmse_m", "reference_xy_peak_m", "mean_speed_m_s",
                    "mean_speed_retention_ratio", "net_course_change_deg"):
            require(metrics[key] == published[key], "Exact canonical replay differs: " + key)
        require(metrics["samples_in_window"] == 513 and metrics["coverage_complete"] is True,
                "Core coverage mismatch")
        require(metrics["large_xy_exploratory_screen"] == published["large_xy_exploratory_screen"],
                "XY screen mismatch")
        rows.append({"mu": mu, "label": label, "engine": "PhysX", "training_engine": "PhysX",
                     **metrics,
                     "canonical_metrics_recomputed_exactly": True,
                     "source_physical_trace_sha256": run["input_sha256"]["isaac_trace"],
                     "source_physical_report_sha256": run["input_sha256"]["isaac_report"],
                     "posture_diagnostics_not_failure_gate": run["posture_diagnostics"]["isaac"],
                     "reported_source_provenance": run["reported_source_provenance"]["isaac"],
                     "original_float64_reference_replay": run["xy_numeric_representation_replay"]["isaac"]})
    return {
        "schema_version": 1,
        "kind": "existing_physx_trained_physx_evaluated_ppo_low_friction_comparison",
        "reference_id": spec["reference_id"], "reference_spec": spec,
        "historical_cohort": "primary_ppo_height_smooth_seed42_v1_not_new_matched_backend_pair",
        "training_engine": "PhysX", "evaluation_engine": "PhysX", "algorithm": "PPO",
        "training_seed": training_metadata["training_seed"],
        "checkpoint_sha256": training_metadata["checkpoint_sha256"],
        "task_profile_sha256": training_metadata["task_profile_sha256"],
        "training_manifest_sha256": training_metadata["training_manifest_sha256"],
        "training_budget": training_metadata["training_budget"],
        "training_domain_randomization": training_metadata["training_domain_randomization"],
        "validation_material_interpretation": {
            "robot_static_and_dynamic_friction": "Both overridden to each recorded mu",
            "ground_static_and_dynamic_friction": "Both overridden to each recorded mu",
            "friction_combine_mode": "average",
            "combine_mode_evidence": "Source-configured average; not an independent runtime combine-flag readback",
            "equal_pair_effective_mu": "(mu + mu) / 2 = mu; not mu squared",
            "road_only_perturbation": False,
            "training_ground_friction_fixed": 1.,
            "training_ideal_pair_bounds_static": [.65, 1.3],
            "training_ideal_pair_bounds_dynamic": [.65, 1.1],
            "bounds_note": "Ideal configured average-combine bounds, not independently measured coefficients or exact sampled bucket extrema"},
        "training_source_sha256": training_metadata["source_training_modules_sha256"],
        "evaluation_mu": grid, "selected_mu": list(SELECTED_MU), "runs": rows,
        "large_xy_mu": [row["mu"] for row in rows if row["large_xy_exploratory_screen"]],
        "xy_screen_limits_m": {"rmse": .15, "peak": .30},
        "full_physical_interval_s": [0., 6.98], "native_physics_sample_count": 1397,
        "reference_interval_s": [float(arrays["reference_time_s"][0]), float(arrays["reference_time_s"][-1])],
        "input_sha256": input_hashes or {},
        "com_speed_source": "Recorded original-link COM world XY velocity; no position differentiation",
        "wheel_heading_source": "Recorded audited kinematic reconstruction: wheel world +Y axle crossed with world +Z; not wheel spin",
        "ground_contact_mark_gate": {"contact_count_greater_than": 0, "upward_force_greater_than_n": 5.},
        "reference_after_final_knot": "Hold last reference knot for pointwise display through t=6.98 s; reference archive itself ends at 6.96 s",
        "new_training": False, "new_policy_evaluation": False, "dynamics_stepping": False,
        "learned_inference": False, "pose_alignment": False, "time_warp": False,
        "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
        "statistical_success_rate_claimed": False, "causal_friction_threshold_claimed": False,
        "limitations": [
            "One historical checkpoint, seed and episode per coefficient; descriptive sensitivity, not a statistical failure probability.",
            "Only PhysX training and PhysX evaluation are shown; this is not a sim2sim gap or new matched-DR backend comparison.",
            "Coefficient mu is the recorded validation setting; do not infer an independently measured tire-road effective coefficient.",
            "Low-mu behavior is not strictly monotonic; show all 17 coefficients, not a fitted universal friction threshold.",
            "Speed, headings and contact diagnostics after posture loss do not imply successful upright turning.",
            "The low-mu recordings include posture loss; wheel markings are projected rolling-plane headings, not proof of controlled grounded drifting.",
            "No height, flight, beta or wheel-heading failure gate is used; the XY screen is RMSE > 0.15 m OR peak > 0.30 m over 1.44--4.00 s."]}


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _rows(summary):
    return {row["mu"]: row for row in summary["runs"]}


def _bounds(arrays):
    coordinates = [arrays["reference_position"][:, :2]]
    for mu in SELECTED_MU:
        coordinates.extend((field(arrays, prefix(mu), "position")[:, :2],
                            field(arrays, prefix(mu), "wheel_pos")[:, :, :2].reshape(-1, 2)))
    xy = np.concatenate(coordinates)
    require(np.isfinite(xy).all(), "Cannot plot nonfinite world coordinates")
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = max(.2, float(np.max(high - low)) * .08)
    return low - margin, high + margin


def _base_axes(arrays, summary, *, figsize=(11, 10)):
    plt = _plotting()
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    low, high = _bounds(arrays)
    rows = _rows(summary)
    rt = arrays["reference_time_s"]
    ref = arrays["reference_position"]
    lo, hi = summary["reference_spec"]["evaluation_core_interval_s"]
    rc = (rt >= lo - 1e-9) & (rt <= hi + 1e-9)
    for ax, mu in zip(axes.flat, SELECTED_MU):
        row = rows[mu]
        ax.plot(ref[:, 0], ref[:, 1], "--", color=COLORS["reference"], lw=1.4)
        ax.plot(ref[rc, 0], ref[rc, 1], "--", color=COLORS["reference"], lw=2.6)
        verdict = "large XY error" if row["large_xy_exploratory_screen"] else "within XY screen"
        ax.set_title(f"mu = {mu:g} | {verdict}\nCore RMSE {row['reference_xy_rmse_m']:.3f} m | peak {row['reference_xy_peak_m']:.3f} m", fontsize=10)
        ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.18)
        ax.set_xlabel("World X [m]"); ax.set_ylabel("World Y [m]")
    fig.suptitle("PhysX-trained PPO -> PhysX evaluation | same checkpoint\nOriginal world coordinates; thick lines: core 1.44-4.00 s\nProjected headings; faint wheels: not contact-eligible; low-mu runs include posture loss", fontsize=11)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=COLORS["reference"], ls="--", label="Reference"),
        Line2D([], [], color=COLORS["path"], label="Measured base path"),
        Line2D([], [], color=COLORS["body"], label="Body heading"),
        Line2D([], [], color=COLORS["velocity"], label="COM velocity direction"),
        Line2D([], [], color=COLORS["left"], label="Left wheel rolling plane"),
        Line2D([], [], color=COLORS["right"], label="Right wheel rolling plane")],
        loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .93))
    return fig, axes


def _wheel_segment(arrays, label, index, side, length=.065):
    heading = field(arrays, label, "wheel_rolling_plane_heading_rad")[index, side]
    center = field(arrays, label, "wheel_pos")[index, side, :2]
    delta = length * np.array([np.cos(heading), np.sin(heading)])
    return center + np.array([-delta, delta])


def _contact(arrays, label, index, side):
    return (field(arrays, label, "contact_count")[index, side] > 0 and
            field(arrays, label, "contact_force")[index, side, 2] > 5.)


def plot_masks(arrays, label):
    """Direction geometry is valid independently of the COM speed cutoff."""
    body = quaternion_rotate(field(arrays, label, "quaternion"), np.array([1., 0., 0.]))
    axle = quaternion_rotate(field(arrays, label, "wheel_quaternion_wxyz"), np.array([0., 1., 0.]))
    return {"body": np.linalg.norm(body[:, :2], axis=-1) > 1e-6,
            "wheel": np.linalg.norm(axle[:, :, :2], axis=-1) > 1e-6,
            "velocity": field(arrays, label, "speed_m_s") >=
                        float(field(arrays, label, "speed_threshold_m_s"))}


def render_topdown(arrays, summary, output):
    """Static full-horizon paths, identical axes and sampled native headings."""
    plt = _plotting(); fig, axes = _base_axes(arrays, summary)
    lo, hi = summary["reference_spec"]["evaluation_core_interval_s"]
    for ax, mu in zip(axes.flat, SELECTED_MU):
        label = prefix(mu); time = field(arrays, label, "time_s")
        pos = field(arrays, label, "position")
        masks = plot_masks(arrays, label)
        core = (time >= lo - 1e-9) & (time <= hi + 1e-9)
        ax.plot(pos[:, 0], pos[:, 1], color=COLORS["path"], lw=.9, alpha=.5)
        ax.plot(pos[core, 0], pos[core, 1], color=COLORS["path"], lw=2.3)
        indices = np.arange(0, len(time), 80)
        for name, color, valid in (("body_heading_rad", COLORS["body"],
                                     masks["body"]),
                                    ("course_rad", COLORS["velocity"],
                                     masks["velocity"])):
            chosen = indices[valid[indices]]
            heading = field(arrays, label, name)[chosen]
            ax.quiver(pos[chosen, 0], pos[chosen, 1], .14*np.cos(heading), .14*np.sin(heading),
                      color=color, angles="xy", scale_units="xy", scale=1, width=.004)
        for index in indices:
            for side, color in enumerate((COLORS["left"], COLORS["right"])):
                if not masks["wheel"][index, side]:
                    continue
                segment = _wheel_segment(arrays, label, index, side)
                ax.plot(segment[:, 0], segment[:, 1], color=color, lw=1.1,
                        alpha=.9 if _contact(arrays, label, index, side) else .16)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=COLORS["path"], ms=4)
        ax.plot(pos[-1, 0], pos[-1, 1], "x", color=COLORS["path"], ms=6)
    fig.savefig(output, dpi=150); plt.close(fig)


def render_friction(arrays, summary, output):
    """All 17 coefficients, no interpolation or fitted friction threshold."""
    plt = _plotting()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    rows = summary["runs"]
    mu = np.array([row["mu"] for row in rows])
    for ax, key, threshold, title in (
            (axes[0], "reference_xy_rmse_m", .15, "Core XY RMSE [m]"),
            (axes[1], "reference_xy_peak_m", .30, "Core peak XY error [m]")):
        values = np.array([row[key] for row in rows])
        ax.plot(mu, values, "o-", color=COLORS["path"], lw=1, ms=5)
        failed = np.array([row["large_xy_exploratory_screen"] for row in rows])
        ax.scatter(mu[failed], values[failed], color="#CC3311", zorder=3, label="Large XY screen")
        ax.axhline(threshold, color="#CC3311", ls="--", lw=1,
                   label=f"{threshold:.2f} m screen")
        ax.set_ylabel(title); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.set_xscale("log"); ax.set_xticks(mu)
        ax.set_xticklabels([f"{value:g}" for value in mu], rotation=45, ha="right")
    axes[-1].set_xlabel("Recorded validation friction parameter mu (log scale)")
    fig.suptitle("PhysX -> PhysX | same PPO checkpoint | all 17 coefficients\nCore 1.44-4.00 s; one recorded episode per coefficient; not a statistical threshold")
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(output, dpi=150); plt.close(fig)


def render_video(arrays, summary, output):
    """50 fps replay of native samples 0,4,...,1396: t=0--6.98 s."""
    plt = _plotting()
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    if not FFMpegWriter.isAvailable():
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    require(FFMpegWriter.isAvailable(), "ffmpeg is required; no dependencies are installed automatically")
    fig, axes = _base_axes(arrays, summary)
    artists = []
    first_time = field(arrays, prefix(SELECTED_MU[0]), "time_s")
    for mu, ax in zip(SELECTED_MU, axes.flat):
        label = prefix(mu)
        require(np.array_equal(field(arrays, label, "time_s"), first_time), "Replay clocks must match exactly")
        path, = ax.plot([], [], color=COLORS["path"], lw=1, alpha=.65)
        core, = ax.plot([], [], color=COLORS["path"], lw=2.4)
        body = ax.quiver([0.], [0.], [.19], [0.], color=COLORS["body"],
                         angles="xy", scale_units="xy", scale=1., width=.006)
        velocity = ax.quiver([0.], [0.], [.24], [0.], color=COLORS["velocity"],
                             angles="xy", scale_units="xy", scale=1., width=.004)
        wheels = [ax.plot([], [], color=color, lw=3)[0]
                  for color in (COLORS["left"], COLORS["right"])]
        error, = ax.plot([], [], color=".55", ls=":", lw=.9)
        text = ax.text(.025, .025, "", transform=ax.transAxes, fontsize=9,
                       bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        artists.append((label, path, core, body, velocity, wheels, error, text, plot_masks(arrays, label)))
    lo, hi = summary["reference_spec"]["evaluation_core_interval_s"]
    writer = FFMpegWriter(fps=50, codec="libx264", bitrate=2400,
                         extra_args=["-pix_fmt", "yuv420p"],
                         metadata={"title": "Existing PhysX PPO low-friction replay", "comment": "No new simulation or pose correction"})
    with writer.saving(fig, str(output), dpi=100):
        for index in range(0, len(first_time), 4):
            t = float(first_time[index])
            target = np.array([np.interp(t, arrays["reference_time_s"], arrays["reference_position"][:, j])
                               for j in range(2)])
            for label, path, core, body, velocity, wheels, error, text, masks in artists:
                position = field(arrays, label, "position")
                path.set_data(position[:index+1, 0], position[:index+1, 1])
                selected = (first_time[:index+1] >= lo - 1e-9) & (first_time[:index+1] <= hi + 1e-9)
                core.set_data(position[:index+1, 0][selected], position[:index+1, 1][selected])
                xy = position[index, :2]
                heading = field(arrays, label, "body_heading_rad")[index]
                body.set_offsets(xy[None, :])
                body.set_UVC([.19*np.cos(heading)], [.19*np.sin(heading)])
                body.set_visible(bool(masks["body"][index]))
                heading = field(arrays, label, "course_rad")[index]
                velocity.set_offsets(xy[None, :])
                velocity.set_UVC([.24*np.cos(heading)], [.24*np.sin(heading)])
                velocity.set_visible(bool(masks["velocity"][index]))
                for side, wheel in enumerate(wheels):
                    segment = _wheel_segment(arrays, label, index, side)
                    wheel.set_data(segment[:, 0], segment[:, 1])
                    wheel.set_visible(bool(masks["wheel"][index, side]))
                    wheel.set_alpha(1. if _contact(arrays, label, index, side) else .16)
                error.set_data([xy[0], target[0]], [xy[1], target[1]])
                text.set_text(f"t = {t:.2f} s | {'CORE' if lo-1e-9 <= t <= hi+1e-9 else 'outside core'}\n"
                              f"XY error {np.linalg.norm(xy-target):.3f} m | COM speed {field(arrays, label, 'speed_m_s')[index]:.2f} m/s")
            writer.grab_frame()
    plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing results: choose a fresh output directory")
    source = Path(args.source_dir)
    source_summary = read_json(source / "summary.json")
    training = read_json(args.training_metadata)
    hashes = {"summary.json": sha256(source / "summary.json"),
              "policy_dynamics_trace.npz": sha256(source / "policy_dynamics_trace.npz"),
              "training_metadata.json": sha256(args.training_metadata)}
    with np.load(source / "policy_dynamics_trace.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    summary = build_summary(arrays, source_summary, training, input_hashes=hashes)
    output.mkdir(parents=True)
    selected = {key: value for key, value in arrays.items() if key in ("schema_version", "wheel_order")
                or key.startswith("reference_") or any(key.startswith(prefix(mu) + "_") for mu in SELECTED_MU)}
    np.savez_compressed(output / "selected_physx_trace.npz", **selected)
    columns = ["mu", "training_engine", "engine", "samples_in_window", "reference_xy_rmse_m",
               "reference_xy_peak_m", "large_xy_exploratory_screen", "mean_speed_m_s",
               "reference_mean_speed_m_s", "mean_speed_retention_ratio", "net_course_change_deg"]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summary["runs"])
    render_topdown(arrays, summary, output / "topdown.png")
    render_friction(arrays, summary, output / "friction_sensitivity.png")
    names = ["selected_physx_trace.npz", "metrics.csv", "topdown.png", "friction_sensitivity.png"]
    if not args.skip_video:
        render_video(arrays, summary, output / "topdown.mp4")
        names.append("topdown.mp4")
        summary["video"] = {"fps": 50, "frames": 350, "native_sample_stride": 4,
                            "first_native_time_s": 0., "last_native_time_s": 6.98,
                            "encoded_duration_s": 7., "final_sample_display_duration_s": .02,
                            "physics_or_reference_time_warp": False}
    summary["analysis_source_sha256"] = {"compare_physx_ppo_low_friction.py": sha256(__file__),
                                          "controller_metrics.py": sha256(ROOT / "scripts/controller_metrics.py")}
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output / name)} for name in names}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--training-metadata", type=Path, default=DEFAULT_TRAINING)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true", help="Static-only test run; full comparison includes video")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": "replayed_existing_physx_only_recordings", "cases": len(result["runs"]),
                      "large_xy_mu": result["large_xy_mu"], "new_simulation": False}))
