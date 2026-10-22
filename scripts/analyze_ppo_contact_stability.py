"""Offline nominal PhysX wheel force-bearing-contact stability diagnostics.

Counts already threshold individual measured ground points at >5 N. Zero count
does NOT prove geometric separation or flight. Aggregate upward force and the
additional count>0 AND Fz>5 N eligibility gate are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ppo_smoothness import analyze_trajectory, deduplicate_actions, rms, sha256, spectral_metrics, validate_clock
from summarize_ppo_smoothness import spectrum_arrays


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT/"outputs/policy_primary_grid_raw_20261016_v1/PPO/mu0p700/isaac"
DEFAULT_03 = ROOT/"outputs/ppo_smoothness_20261021_v2/action_rate_0p3/validation/mu0p700/isaac"
DEFAULT_05 = ROOT/"outputs/ppo_smoothness_20261021_v2/action_rate_0p5/validation/mu0p700/isaac"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
MOTION_SHA = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
CASES = {
    "baseline": {"weight": -.1, "checkpoint": "6ec411efeaea37fc61389285978b377aa7c5be80e326ce7f18581efc9acdf27b", "trace": "858a4dc5b11230d1ac5a894c9528060de90570265ebdea50f20a48005a021d61"},
    "action_rate_0p3": {"weight": -.3, "checkpoint": "060afb10b61a6cc6d494c70c8506475ffc6e4f0ac62885dd376f54701521dabb", "trace": "342b637767b65c1c23d3a90e11bcc6ea67cad89f199c48cd8b18b1b51de72da2"},
    "action_rate_0p5": {"weight": -.5, "checkpoint": "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a", "trace": "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780"}}
COLORS = {"baseline": "#0072B2", "action_rate_0p3": "#D55E00", "action_rate_0p5": "#009E73"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _field(raw, name):
    value = np.asarray(raw[name])
    return value[:, 0] if value.ndim >= 2 and value.shape[1] == 1 else value


def binary_runs(time_s, state):
    """Observed zero-order-held true runs, clipped at the last real timestamp.

    Boundary runs are window-censored. No duration extends past the last native
    sample; complete interior runs have duration n_samples*dt.
    """
    time = np.asarray(time_s, dtype=float); mask = np.asarray(state)
    require(time.ndim == 1 and len(time) >= 2 and np.isfinite(time).all() and np.all(np.diff(time) > 0), "Invalid run clock")
    require(mask.shape == time.shape and mask.dtype.kind == "b", "Run state must be one boolean per native sample")
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [{"start_s": float(time[a]), "end_s": float(time[b] if b < len(time) else time[-1]),
        "observed_duration_s": float((time[b] if b < len(time) else time[-1])-time[a]),
        "native_samples": int(b-a), "left_window_censored": bool(a == 0), "right_window_censored": bool(b == len(time))}
        for a, b in zip(starts, stops)]


def run_statistics(time_s, state):
    runs = binary_runs(time_s, state)
    values = np.array([run["observed_duration_s"] for run in runs])
    complete = np.array([run["observed_duration_s"] for run in runs
                         if not run["left_window_censored"] and not run["right_window_censored"]])
    return {"runs": len(runs), "max_observed_duration_s": float(values.max()) if len(values) else 0.,
        "mean_observed_duration_s": float(values.mean()) if len(values) else None,
        "p50_observed_duration_s": float(np.quantile(values, .5)) if len(values) else None,
        "p95_observed_duration_s": float(np.quantile(values, .95)) if len(values) else None,
        "complete_runs": len(complete), "p95_complete_run_duration_s": float(np.quantile(complete, .95)) if len(complete) else None,
        "left_window_censored_runs": sum(run["left_window_censored"] for run in runs),
        "right_window_censored_runs": sum(run["right_window_censored"] for run in runs),
        "longest_five_observed_runs": sorted(runs, key=lambda row: row["observed_duration_s"], reverse=True)[:5],
        "censoring_note": "Boundary durations are observed lower bounds, not completed separation/flight intervals"}


def contact_statistics(time_s, contact):
    time, mask = np.asarray(time_s, dtype=float), np.asarray(contact, dtype=bool)
    duration = float(time[-1]-time[0])
    transitions = np.diff(mask.astype(np.int8))
    return {"samples": len(time), "sample_duty_fraction": float(mask.mean()),
        "time_duty_fraction_zoh": float(np.sum(mask[:-1]*np.diff(time))/duration),
        "state_switch_count": int(np.count_nonzero(transitions)),
        "state_switches_per_s": float(np.count_nonzero(transitions)/duration),
        "onset_count": int(np.count_nonzero(transitions > 0)), "offset_count": int(np.count_nonzero(transitions < 0)),
        "condition_false_runs": run_statistics(time, ~mask), "condition_true_runs": run_statistics(time, mask)}


def contact_masks(counts, fz):
    measured = np.asarray(counts) > 0
    eligible = measured & (np.asarray(fz) > 5.)
    def masks(value):
        return {"left": value[:, 0], "right": value[:, 1], "both": value.all(axis=1), "none": ~value.any(axis=1)}
    return {"measured_force_bearing_contact": masks(measured), "load_eligibility": masks(eligible)}


def _force_statistics(force, dt):
    value = np.asarray(force, dtype=float)
    delta = np.diff(value)
    return {"mean_n": float(value.mean()), "std_n": float(value.std()), "rms_n": rms(value),
        "minimum_n": float(value.min()), "maximum_n": float(value.max()), "p95_n": float(np.quantile(value, .95)),
        "step_rms_n": rms(delta), "step_abs_peak_n": float(np.abs(delta).max()),
        "spectrum": spectral_metrics(value, dt, 10.)}


def analyze_signals(time_s, counts, force_w_n, base_com_velocity_w, base_angular_velocity_w, *, nonwheel_counts=None):
    """Pure native-array analysis, also usable for synthetic lifecycle tests."""
    time, dt = validate_clock(time_s)
    require(np.isclose(dt, .005, rtol=0, atol=1e-10), "Require original 200-Hz physical sampling")
    n = len(time); counts = np.asarray(counts); force = np.asarray(force_w_n, dtype=float)
    require(counts.shape == (n, 2) and np.isfinite(counts).all() and np.all(counts >= 0) and
            np.array_equal(counts, np.round(counts)), "Invalid measured wheel contact counts")
    require(force.shape == (n, 2, 3) and np.isfinite(force).all(), "Invalid measured ground-normal-force vectors")
    velocity, angular = np.asarray(base_com_velocity_w), np.asarray(base_angular_velocity_w)
    require(velocity.shape == angular.shape == (n, 3) and np.isfinite(velocity).all() and np.isfinite(angular).all(), "Invalid recorded base twists")
    fz = force[:, :, 2]
    intervals = {"full": [float(time[0]), float(time[-1])], "core": [1.44, 4.],
        "pre_core": [0., 1.44], "post_core": [4., float(time[-1])],
        "core_early": [1.44, 2.72], "core_late": [2.72, 4.]}
    windows = {}
    for name, interval in intervals.items():
        mask = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        if interval[1] <= interval[0] or mask.sum() < 4:
            continue
        selected_time, selected_counts, selected_fz = time[mask], counts[mask], fz[mask]
        states = contact_masks(selected_counts, selected_fz)
        row = {"interval_s": interval, "samples": int(mask.sum()), "duration_s": float(selected_time[-1]-selected_time[0]),
            "coverage_complete": bool(time[0] <= interval[0]+1e-9 and time[-1] >= interval[1]-1e-9),
            "contact": {kind: {side: contact_statistics(selected_time, value) for side, value in by_side.items()}
                        for kind, by_side in states.items()},
            "force_z": {side: _force_statistics(selected_fz[:, index], dt) for index, side in enumerate(("left", "right"))},
            "wheel_load_sum": {**_force_statistics(selected_fz.sum(axis=1), dt), "mg_normalized": None,
                "mass_normalization_available": False, "note": "No live measured total robot mass/gravity audit in these reports; keep N, do not infer mg from assets"},
            "base_com_vertical_velocity_spectrum": spectral_metrics(velocity[mask, 2], dt, 10.),
            "base_world_xy_angular_velocity_spectrum": spectral_metrics(angular[mask, :2], dt, 10.),
            "wheel_vertical_velocity": {"available": False, "reason": "Wheel COM/link-origin velocities are not recorded; no finite-difference position proxy is invented"}}
        for kind, statistics in row["contact"].items():
            row.setdefault("missing_contact_runs", {})[kind] = {
                "left_wheel_missing": statistics["left"]["condition_false_runs"],
                "right_wheel_missing": statistics["right"]["condition_false_runs"],
                "neither_wheel_has_contact": statistics["none"]["condition_true_runs"],
                "at_least_one_wheel_missing": statistics["both"]["condition_false_runs"]}
        for index, side in enumerate(("left", "right")):
            row["force_z"][side]["point_count_change_events_per_s"] = float(np.count_nonzero(np.diff(selected_counts[:, index]))/row["duration_s"])
        ids = np.arange(0, n, 4)
        ids = ids[(time[ids] >= interval[0]-1e-9) & (time[ids] <= interval[1]+1e-9)]
        if len(ids) >= 4:
            sub = contact_masks(counts[ids], fz[ids])
            row["policy_rate_subsample_diagnostic"] = {"sample_hz": 50., "native_stride": 4,
                "not_a_new_measurement": True, "aliasing_possible": True,
                "contact": {kind: {side: contact_statistics(time[ids], value) for side, value in by_side.items()}
                            for kind, by_side in sub.items()}}
        if nonwheel_counts is not None:
            value = np.asarray(nonwheel_counts)
            require(value.shape == (n,) and np.isfinite(value).all() and np.all(value >= 0), "Invalid recorded nonwheel counts")
            row["nonwheel_force_bearing_contact_sample_fraction"] = float((value[mask] > 0).mean())
        windows[name] = row
    return windows


def analyze_case(folder, label, reference=DEFAULT_REFERENCE):
    folder = Path(folder); report = json.loads((folder/"report.json").read_text())
    identity = CASES[label]
    require(report["simulator"] == "IsaacLab/PhysX" and report["status"] == "evaluated" and
            report["checkpoint_sha256"] == identity["checkpoint"], "Wrong completed PhysX policy cohort")
    require(sha256(folder/"physics_trajectory.npz") == report["physics_trajectory_sha256"] == identity["trace"], "Original physical trajectory SHA differs")
    require(report["motion_file_sha256"] == MOTION_SHA and sha256(reference) == MOTION_SHA and
            report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7 and
            report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False,
            "Require fixed nominal mu=.7/reference and deterministic evaluation")
    with np.load(folder/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    time = np.asarray(raw["time_s"])
    validity = _field(raw, "valid_mask")
    require(time.shape == (1397,) and np.array_equal(time, np.arange(1397)*.005) and
            validity.shape == (1397,) and validity.dtype.kind == "b" and validity.all(),
            "Require full original native first-episode clock, no reset splice")
    require(tuple(raw["wheel_body_names"].tolist()) == ("wheel_L_Link", "wheel_R_Link"), "Recorded wheel ordering differs")
    decision = deduplicate_actions(time, _field(raw, "actions_clipped"), _field(raw, "action_was_applied"),
        float(raw["policy_dt_s"]), frames=_field(raw, "reference_frame_for_action"), counters=raw["simulation_step_counter"])
    require(decision["decisions"] == 349, "Require all original 349 control decisions")
    require(report["contact_slip_measurement"]["active_point_normal_force_threshold_n"] == 5., "Recorded point-count threshold differs")
    counts, force = _field(raw, "wheel_ground_contact_count"), _field(raw, "wheel_ground_contact_force_w_n")
    velocity, angular = _field(raw, "root_linear_velocity_w_m_s"), _field(raw, "root_angular_velocity_w_rad_s")
    windows = analyze_signals(time, counts, force, velocity, angular,
        nonwheel_counts=_field(raw, "nonwheel_ground_contact_count") if "nonwheel_ground_contact_count" in raw else None)
    tracking = analyze_trajectory(folder/"physics_trajectory.npz", reference=reference, report=folder/"report.json")
    for window in ("full", "core"):
        windows[window]["tracking"] = tracking["windows"][window]["tracking"]
    arrays = {"time_s": time, "count": counts, "force_w_n": force,
        "base_com_velocity_w_m_s": velocity, "base_angular_velocity_w_rad_s": angular,
        "actions_clipped_native": _field(raw, "actions_clipped"), "decision_time_s": decision["time_s"]}
    core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
    for signal, value in (("force_left", force[:, 0, 2]), ("force_right", force[:, 1, 2]), ("base_com_vz", velocity[:, 2])):
        frequency, psd = spectrum_arrays(value[core, None])
        arrays[signal+"_core_psd_frequency_hz"], arrays[signal+"_core_psd"] = frequency, psd
    return arrays, {"label": label, "action_rate_weight": identity["weight"], "checkpoint_sha256": identity["checkpoint"],
        "input_sha256": {"report.json": sha256(folder/"report.json"), "physics_trajectory.npz": identity["trace"]},
        "evaluator_source_sha256": report["evaluator_source_sha256"], "contact_slip_helper_sha256": report["contact_slip_helper_sha256"],
        "reported_contact_metadata_sample_dt_s": report["contact_slip_measurement"]["sample_dt_s"],
        "actual_native_physical_sample_dt_s": .005, "windows": windows}, report


def load_comparison(baseline=DEFAULT_BASELINE, action_rate_03=DEFAULT_03, action_rate_05=DEFAULT_05, reference=DEFAULT_REFERENCE):
    arrays, cases, reports = {}, {}, {}
    for label, folder in (("baseline", baseline), ("action_rate_0p3", action_rate_03), ("action_rate_0p5", action_rate_05)):
        current, cases[label], reports[label] = analyze_case(folder, label, reference)
        arrays.update({label+"_"+key: value for key, value in current.items()})
    for name in ("contact_material_audit", "legacy_joint_friction_audit", "evaluator_source_sha256",
                 "drift_module_sha256", "contact_slip_helper_sha256", "asset_root_file_sha256", "physics_recording"):
        require(all(reports[label][name] == reports["baseline"][name] for label in CASES), "Shared physical/contact/evaluator setting differs: "+name)
    return arrays, {"schema_version": 1, "kind": "recorded_nominal_physx_ppo_wheel_contact_stability",
        "evaluation_mu": .7, "reference_sha256": MOTION_SHA, "cases": cases,
        "case_order": list(CASES), "point_count_source": "Actual ground-filtered points with individual normal force strictly >5 N; not all geometric contacts",
        "ground_force_source": "Sum of all recorded ground NORMAL force vectors, not friction force; upward component is Fz",
        "count_gate": "recorded_count > 0", "additional_load_gate": "recorded_count > 0 AND total upward Fz > 5 N",
        "wheel_vertical_velocity_available": False, "mass_normalization_available": False,
        "native_physical_hz": 200., "policy_hz": 50., "core_interval_s": [1.44, 4.],
        "height_or_flight_failure_gate": False, "nonzero_sideslip_target_preserved": True,
        "new_training": False, "new_simulation": False, "new_inference": False, "contact_failure_threshold_added": False,
        "limitations": ["Zero thresholded point count is NOT proof of geometric separation or flight; weak contacts <=5 N cannot be recovered.",
            "Native clock/recorder proves 5ms samples; legacy contact metadata still reports a 20ms policy interval and does not replace the native clock.",
            "Native-sample duty and zero-order-held time duty differ at endpoint samples; both are explicit. Window-boundary runs are censored lower bounds.",
            "50Hz contact summaries are deliberate stride-4 subsamples of the same 200Hz recording, not new measurements; they can miss contact switches.",
            "Post-core data include synthetic recovery/tail; early/late windows are descriptive, not new success gates.",
            "No measured live total-mass audit or wheel COM/link velocity is available. Do not infer mg or wheel velocity from assets/position differences.",
            "One seed/checkpoint/episode per policy; no causal or statistical certification of stable contact."]}


def render_time_figure(arrays, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey="col")
    max_force = max(float(arrays[label+"_force_w_n"][:, :, 2].max()) for label in CASES)
    max_angular = max(float(np.abs(arrays[label+"_base_angular_velocity_w_rad_s"][:, 0]).max()) for label in CASES)
    for row, label in enumerate(CASES):
        time = arrays[label+"_time_s"]; force = arrays[label+"_force_w_n"][:, :, 2]
        count = arrays[label+"_count"]
        axes[row, 0].plot(time, force[:, 0], color="#D55E00", lw=.65, label="Left Fz")
        axes[row, 0].plot(time, force[:, 1], color="#8B4BA6", lw=.65, label="Right Fz")
        axes[row, 0].set(ylim=(min(0., float(force.min())), max_force*1.05), ylabel=f"action-rate {CASES[label]['weight']:g}\nGround Fz [N]")
        contact, load = count > 0, (count > 0) & (force > 5.)
        for offset, value, color in ((3, contact[:, 0], "#D55E00"), (2, load[:, 0], "#D55E00"),
                                      (1, contact[:, 1], "#8B4BA6"), (0, load[:, 1], "#8B4BA6")):
            axes[row, 1].fill_between(time, offset, offset+.75, where=value, step="post", color=color, alpha=.85)
        axes[row, 1].set(yticks=[.35, 1.35, 2.35, 3.35], yticklabels=["R eligible", "R count>0", "L eligible", "L count>0"], ylim=(-.1, 4.))
        axes[row, 2].plot(time, arrays[label+"_base_angular_velocity_w_rad_s"][:, 0], color=COLORS[label], lw=.65)
        axes[row, 2].set(ylim=(-max_angular*1.05, max_angular*1.05), ylabel="World omega X [rad/s]")
        for ax in axes[row]:
            ax.axvspan(1.44, 4., color=".6", alpha=.12); ax.set_xlim(0, 6.98); ax.grid(alpha=.15)
            ax.set_xlabel("Original physical time [s]")
        axes[row, 0].legend(fontsize=8)
    for ax, title in zip(axes[0], ("Recorded wheel ground-normal forces", "Thresholded measured-contact / load eligibility", "Recorded base angular motion")):
        ax.set_title(title, fontsize=10)
    fig.suptitle("PhysX PPO mu=0.70 | native 200Hz contact stability | original unfiltered recordings\nCounts already threshold points at >5 N; missing eligibility is NOT a flight/failure label", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(output, dpi=150); plt.close(fig)


def render_spectrum_figure(arrays, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label in CASES:
        for axis, signal in zip(axes, ("force_left", "force_right", "base_com_vz")):
            f, psd = arrays[label+"_"+signal+"_core_psd_frequency_hz"], arrays[label+"_"+signal+"_core_psd"]
            axis.semilogy(f[1:], np.maximum(psd[1:], 1e-15), color=COLORS[label], label=f"action-rate {CASES[label]['weight']:g}")
    for axis, title, unit in zip(axes, ("Left wheel Fz", "Right wheel Fz", "Base COM vertical velocity (NOT wheel vz)"), ("N²/Hz", "N²/Hz", "(m/s)²/Hz")):
        axis.set(xlim=(0, 100), xlabel="Native frequency [Hz]", ylabel=unit, title=title)
        axis.axvline(10., color="black", ls="--", lw=.7); axis.axvspan(22.5, 27.5, color="#56B4E9", alpha=.15)
        axis.grid(alpha=.18); axis.legend(fontsize=8)
    fig.suptitle("Core 1.44-4.00 s | native 200Hz PSD | linear detrend, Hann, channel-mean Parseval normalization", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(output, dpi=150); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing contact evidence: choose a fresh directory")
    arrays, summary = load_comparison(args.baseline, args.action_rate_03, args.action_rate_05, args.reference)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_contact_traces.npz", **arrays)
    render_time_figure(arrays, summary, output/"contact_timeseries.png")
    render_spectrum_figure(arrays, summary, output/"contact_spectra.png")
    summary["analysis_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in
        ("analyze_ppo_contact_stability.py", "analyze_ppo_smoothness.py", "summarize_ppo_smoothness.py")}
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in
        ("native_contact_traces.npz", "contact_timeseries.png", "contact_spectra.png")}
    with (output/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--action-rate-03", type=Path, default=DEFAULT_03)
    result.add_argument("--action-rate-05", type=Path, default=DEFAULT_05)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    print(json.dumps({"status": "audited_existing_nominal_contacts", "cases": len(outcome["cases"]), "new_simulation": False}))
