"""Offline negative example: contact-reward replacement, not increased weight.

The matched comparison is two independently trained200Hz policies, with the
original binary grounded+0.4 versus dense all-substep grounded+0.4. Torque-rate
and400Hz cases are context, not additional matched reward ablations. Events
are descriptive observations, never new termination/flight/success gates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze_ppo_contact_stability as contact
import analyze_ppo_contact_timestep as timestep
import analyze_ppo_smoothness as smooth


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = contact.DEFAULT_REFERENCE
MOTION_SHA256 = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
PROFILE_SHA256 = "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6"
CASE_ORDER = ("baseline200", "ground200", "torque200", "baseline400")
CASE_SPECS = {
    "baseline200": {"folder": "outputs/contact_damping_20261021_v2/kd10", "dt_s": .005,
        "label": "Original grounded+0.4, action-rate-0.5 (200Hz)", "role": "matched_reward_baseline",
        "checkpoint_sha256": "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a",
        "raw_sha256": "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780",
        "report_sha256": "fdf3881b765276832dbbdfe0528446055a4af91a6147ca3520dc847f6906dc77"},
    "ground200": {"folder": "outputs/ppo_ground_support_20261021_v1/ground_support_sequence_seed42_v1/validation/mu0p700/isaac", "dt_s": .005,
        "label": "Dense grounded replacement+0.4 (200Hz)", "role": "matched_reward_negative_candidate",
        "checkpoint_sha256": "a42523048800ede9225429ec889d30bfa949499b134251a473048470d007fe9b",
        "raw_sha256": "1c302cab2be79e02895fc013b957a4df2f3f17e2f100c86569639f15018954e9",
        "report_sha256": "bac42f3063ad1e46a0650239f9db54607c045cc97e4baf09d5e8c170f5d5e7f6"},
    "torque200": {"folder": "outputs/ppo_torque_control_v2_20261021_v1/torque_control_sequence_seed42_v1/validation/mu0p700/isaac", "dt_s": .005,
        "label": "Added leg torque-rate-0.05 (200Hz, context)", "role": "different_reward_policy_context",
        "checkpoint_sha256": "d8530c5730b404ba1a5580ef52a9a51e3ceade91d0c56ceb13794d20ad8b1ff8",
        "raw_sha256": "f1a5ae3e3a23b9262a0caf1a3769530c240a341df571c30d10d53dcd031ef2fc",
        "report_sha256": "bb89b39a558b42942eb810724411f71de17124cff0421dc4e2e68a9ede69d69b"},
    "baseline400": {"folder": "outputs/ppo_contact_timestep_20261022_v1/dt0p0025", "dt_s": .0025,
        "label": "Original actor, refined physics/PD400Hz (context)", "role": "different_native_clock_context",
        "checkpoint_sha256": "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a",
        "raw_sha256": "8e7fc43c181b34b9ff87fe42d3c628ab483c7a6cddfb965562df250d77e46c1e",
        "report_sha256": "abe05d4ea69634e9794514fc60664cf4c67fe718ad841716d5518923671e778e"},
}
TRAINING_FILES = {
    "baseline_manifest": ("outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training/manifest.json", "066ddaeb5a23ff44dfca79e785520c57d8786cdc3a2d7b30d118e3a955a94d63"),
    "baseline_ablation": ("outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training/ablation_manifest.json", "4cf22f9c4e6b5f28aa1711f2e093516522046d259e0322b740e7cad5da05905e"),
    "ground_manifest": ("outputs/ppo_ground_support_20261021_v1/ground_support_sequence_seed42_v1/training/manifest.json", "630c40cf13f44e7b5f2bbd4b729abd3ee5a4f2df6fe6da9014bb70d6abff6d78"),
    "ground_ablation": ("outputs/ppo_ground_support_20261021_v1/ground_support_sequence_seed42_v1/training/ablation_manifest.json", "ba7d01f78ca5c0d3e732bb793e1f21bb970e1b4cbc8c9937322abcc4c59dfedb"),
}
SHARED_TRAINING_KEYS = ("algorithm", "source_training_modules_sha256", "motion_sha256", "asset_sha256", "task_profile_sha256",
    "task_profile", "contract", "seed", "num_envs", "iterations", "num_steps_per_env", "policy_steps_requested", "environment_transitions_requested", "recipe")
MIN_DURATION_S = .1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def persistent_events(time_s, condition, minimum_duration_s=MIN_DURATION_S):
    """Observed ZOH runs, with no extension past the last real timestamp.

    Onset is post-hoc; confirmation occurs only after0.1s of observed evidence.
    Censored durations remain lower bounds, never inferred flight durations.
    """
    require(type(minimum_duration_s) in (float, int) and np.isfinite(minimum_duration_s)
            and minimum_duration_s > 0, "Duration must be a finite positive number")
    time = np.asarray(time_s, dtype=float); state = np.asarray(condition)
    require(state.dtype.kind == "b", "Event condition must be boolean")
    runs = contact.binary_runs(time, state)
    events = []
    for run in runs:
        if run["observed_duration_s"] < minimum_duration_s-1e-9:
            continue
        events.append({"onset_s": run["start_s"], "confirmation_s": float(run["start_s"]+minimum_duration_s),
            "end_s": run["end_s"], "observed_duration_s": run["observed_duration_s"], "native_samples": run["native_samples"],
            "left_window_censored": run["left_window_censored"], "right_window_censored": run["right_window_censored"],
            "duration_is_lower_bound": run["left_window_censored"] or run["right_window_censored"]})
    return {"minimum_persistence_s": float(minimum_duration_s),
        "first_true_sample_s": float(time[np.flatnonzero(state)[0]]) if state.any() else None,
        "first_persistent_event": events[0] if events else None, "persistent_event_count": len(events),
        "longest_observed_run_s": max((run["observed_duration_s"] for run in runs), default=0.), "events": events}


def verify_training(root):
    data, hashes = {}, {}
    for key, (relative, expected) in TRAINING_FILES.items():
        path = root/relative
        require(smooth.sha256(path) == expected, "Pinned training metadata SHA differs: "+key)
        data[key], hashes[relative] = read_json(path), expected
    baseline, ground = data["baseline_manifest"], data["ground_manifest"]
    for key in SHARED_TRAINING_KEYS:
        require(key in baseline and baseline[key] == ground.get(key), "Matched200Hz training recipe differs: "+key)
    replacement = data["ground_ablation"].get("reward_replacement", {})
    require(replacement.get("term") == "drift_grounded" and replacement.get("weight") == .4
            and replacement.get("additional_reward_term") is False
            and replacement.get("reward_manager_dt_factor_applied_once") == .02
            and replacement.get("scale_fraction") == .2
            and ground.get("contact_reward_replacement") == replacement
            and ground.get("comparison_baseline_action_rate_weight") == -.5
            and baseline.get("training_reward_overrides") == {"action_rate_l2": -.5},
            "Negative comparison must replace grounded function at unchanged+0.4, not increase weight")
    for key, case_id in (("baseline_ablation", "baseline200"), ("ground_ablation", "ground200")):
        ablation = data[key]
        require(ablation.get("status") == "training_completed" and ablation.get("smoke_only") is False
                and ablation.get("initialization") == "from_scratch_no_checkpoint_resume"
                and ablation.get("actual_environment_transitions") == 12288000
                and ablation.get("checkpoint_sha256") == CASE_SPECS[case_id]["checkpoint_sha256"], "Training completion/checkpoint differs: "+key)
    require(baseline.get("seed") == 42 and baseline.get("num_envs") == 512 and baseline.get("iterations") == 1000
            and baseline.get("num_steps_per_env") == 24, "Historical matched training budget differs")
    return {"input_sha256": hashes, "seed": 42, "transitions_each": 12288000, "independent_from_scratch_policies": True,
        "both_native_training_hz": 200, "baseline_action_rate_weight": -.5, "original_grounded_weight": .4,
        "candidate_grounded_weight": .4, "reward_weight_increased": False, "reward_replacement": replacement,
        "extra_sensor_collector_implementation": True, "single_seed_causal_proof": False}


def analyze_case(folder, spec, reference, profile):
    raw_path, report_path = folder/"physics_trajectory.npz", folder/"report.json"
    require(smooth.sha256(raw_path) == spec["raw_sha256"] and smooth.sha256(report_path) == spec["report_sha256"], "Pinned original report/raw SHA differs")
    report = read_json(report_path)
    require(report.get("status") == "evaluated" and report.get("simulator") == "IsaacLab/PhysX"
            and report.get("checkpoint_sha256") == spec["checkpoint_sha256"] and report.get("motion_file_sha256") == MOTION_SHA256
            and report.get("full_horizon_diagnostic") is True and report.get("terminal_states_captured_before_auto_reset") is True
            and report.get("domain_randomization_enabled") is False and report.get("observation_noise_enabled") is False
            and report.get("physics_trajectory_sha256") == spec["raw_sha256"]
            and report.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7,
            "Require pinned completed nominal deterministic PhysX episode")
    require(report.get("task_profile") == profile, "Original training task profile differs")
    evaluated = dict(report.get("evaluation_task_profile", {})); evaluated["contact_friction"] = profile["contact_friction"]
    require(evaluated == profile and report["policy_export"]["leg_kp"] == 500. and report["policy_export"]["leg_kd"] == 10., "Evaluation profile or fixed PD differs")
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    dt = spec["dt_s"]; stride = round(.02/dt); n = 349*stride+1; time = np.asarray(raw["time_s"])
    require(time.shape == (n,) and np.array_equal(time, np.arange(n)*dt)
            and np.asarray(raw["valid_mask"]).shape == (n, 1) and raw["valid_mask"].dtype.kind == "b" and raw["valid_mask"].all()
            and float(raw["physics_dt_s"]) == dt and float(raw["policy_dt_s"]) == .02, "Native exact time/count/validity differs")
    counter = np.asarray(raw["simulation_step_counter"])
    require(counter.shape == (n,) and counter.dtype.kind in "iu" and np.array_equal(counter-counter[0], np.arange(n)), "Native counters differ")
    applied = contact._field(raw, "action_was_applied")
    require(applied.dtype.kind == "b" and not applied[0] and applied[1:].all(), "Initial/pre-action versus physical samples differ")
    generic = smooth.analyze_trajectory(raw_path, reference=reference, report=report_path)
    require(generic["control_decisions"] == 349 and generic["physical_samples"] == n, "Original50Hz control/native scope differs")
    signals = timestep.signal_windows(raw, dt)
    quaternion = contact._field(raw, "root_quaternion_wxyz").astype(float)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1-2*(quaternion[:, 1]**2+quaternion[:, 2]**2), -1, 1)))
    force = contact._field(raw, "wheel_ground_contact_force_w_n")[:, :, 2]
    counts = contact._field(raw, "wheel_ground_contact_count")
    bearing = (counts > 0) & (force > 5.)
    nonwheel = contact._field(raw, "nonwheel_ground_contact_force_norm_n")
    require(nonwheel.shape == (n,) and np.isfinite(nonwheel).all() and (nonwheel >= 0).all(), "Invalid recorded nonwheel ground normal-force norm")
    conditions = {**{f"tilt_ge_{threshold}_deg": tilt >= threshold for threshold in (30, 45, 60)},
        "nonwheel_ground_normal_force_gt1N": nonwheel > 1., "both_wheels_load_eligible": bearing.all(axis=1),
        "neither_wheel_load_eligible": ~bearing.any(axis=1), "at_least_one_wheel_not_load_eligible": ~bearing.all(axis=1)}
    events = {name: persistent_events(time, mask) for name, mask in conditions.items()}
    windows = {}
    for name, interval in (("core", (1.44, 4.)), ("full", (0., 6.98))):
        mask = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        tracking = generic["windows"][name]["tracking"]
        windows[name] = {"interval_s": list(interval), "native_samples": int(mask.sum()), "tracking": tracking,
            "both_load_eligible_fraction": float(bearing[mask].all(axis=1).mean()),
            "neither_load_eligible_fraction": float((~bearing[mask].any(axis=1)).mean()),
            "tilt_rms_deg": smooth.rms(tilt[mask]), "tilt_max_deg": float(tilt[mask].max()),
            "nonwheel_ground_gt1N_fraction": float((nonwheel[mask] > 1.).mean()), "nonwheel_ground_force_peak_n": float(nonwheel[mask].max()),
            "world_angular_xy_common10to100hz_rms_rad_s": signals[name]["spectra_common_band10_100"]["world_angular_xy"]["common_band_rms"]}
    return raw, {"label": spec["label"], "role": spec["role"], "source_folder": str(folder),
        "checkpoint_sha256": spec["checkpoint_sha256"], "input_sha256": {"physics_trajectory.npz": spec["raw_sha256"], "report.json": spec["report_sha256"]},
        "physics_dt_s": dt, "native_hz": 1./dt, "policy_hz": 50., "native_samples": n, "control_decisions": 349,
        "initial_row_is_pre_action_snapshot": True, "initial_nonwheel_ground_force_n": float(nonwheel[0]),
        "initial_load_eligibility_left_right": bearing[0].tolist(), "windows": windows, "events_full": events}


def load_comparison(input_root=ROOT, reference=DEFAULT_REFERENCE):
    root = Path(input_root); reference = Path(reference)
    require(smooth.sha256(reference) == MOTION_SHA256, "Pinned reference SHA differs")
    profile_path = root/"config/car_drift_height_smooth_task.json"
    require(smooth.sha256(profile_path) == PROFILE_SHA256, "Pinned historical task-profile SHA differs")
    profile = read_json(profile_path); training = verify_training(root)
    raw, cases = {}, {}
    for name in CASE_ORDER:
        spec = CASE_SPECS[name]
        raw[name], cases[name] = analyze_case(root/spec["folder"], spec, reference, profile)
    return raw, {"schema_version": 1, "kind": "recorded_ground_support_reward_replacement_negative_example",
        "status": "descriptive_negative_example_requires_visual_review", "case_order": list(CASE_ORDER), "cases": cases,
        "primary_matched_pair": ["baseline200", "ground200"], "context_only_cases": ["torque200", "baseline400"],
        "matched_training": training, "reference_sha256": MOTION_SHA256, "task_profile_sha256": PROFILE_SHA256,
        "evaluation_mu": .7, "event_persistence_s": MIN_DURATION_S, "new_simulation": False, "new_inference": False,
        "new_training": False, "height_or_contact_absence_failure_gate": False, "zero_contact_is_flight": False,
        "nonzero_sideslip_target_preserved": True, "absolute_height_constraint_added": False,
        "source_sha256": {Path(__file__).name: smooth.sha256(__file__), "analyze_ppo_contact_timestep.py": smooth.sha256(timestep.__file__),
            "analyze_ppo_contact_stability.py": smooth.sha256(contact.__file__), "analyze_ppo_smoothness.py": smooth.sha256(smooth.__file__)},
        "limitations": ["Grounded reward function/sampling was replaced at unchanged+0.4; it was NOT a weight increase or an additional reward term.",
            "The primary200Hz pair shares declared training recipe, seed42,12.288M transitions andPD500/10 but contains independent trained checkpoints and extra sensor/collector implementation; one seed is not statistical causal proof.",
            "Torque200 has another reward/checkpoint. Baseline400 changes native physics AND explicitPD rate. Comparing either directly with ground200 is context, not a single-factor matched reward ablation.",
            "Pose events describe observed tilt persistence, not simulator terminations. Thresholded load/contact absence does not prove flight or task failure.",
            "Onsets are post-hoc. Confirmation occurs0.1s later; boundary durations are censored lower bounds and never extended beyond6.98s.",
            "The initial row is a pre-action sensor snapshot, not a newly observed physical impact; brief startup transients are reported but not persistent events.",
            "A fallen policy can have lower full-horizon HF after settling. Do not rank fullHF as improvement without posture, contact, XY/speed/beta and pre/post-event context.",
            "HF uses shared10<f<100Hz band on each original native clock, without state resampling. Full6.98s tracking is unavailable beyond the6.96s reference; no endpoint extrapolation."]}


def render_timeline(raw, summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline200": "#0072B2", "ground200": "#D55E00", "torque200": "#009E73", "baseline400": "#7B61A8"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for case in CASE_ORDER:
        current = raw[case]; time = current["time_s"]; q = contact._field(current, "root_quaternion_wxyz").astype(float)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(1-2*(q[:, 1]**2+q[:, 2]**2), -1, 1)))
        color = colors[case]; opacity = 1. if case in summary["primary_matched_pair"] else .5
        axes[0].plot(time, tilt, color=color, alpha=opacity, lw=1., label=summary["cases"][case]["label"])
    for threshold in (30, 45, 60):
        axes[0].axhline(threshold, color="#777777", lw=.6, ls="--")
        event = summary["cases"]["ground200"]["events_full"][f"tilt_ge_{threshold}_deg"]["first_persistent_event"]
        if event:
            axes[0].axvline(event["onset_s"], color=colors["ground200"], lw=.6, ls=":")
            axes[0].annotate(f"{threshold}deg @{event['onset_s']:.3f}s", (event["onset_s"], threshold), xytext=(8, 5), textcoords="offset points", fontsize=8)
    for row, case in enumerate(summary["primary_matched_pair"]):
        current = raw[case]; time = current["time_s"]
        fz = contact._field(current, "wheel_ground_contact_force_w_n")[:, :, 2]
        bearing = (contact._field(current, "wheel_ground_contact_count") > 0) & (fz > 5.)
        axes[1].step(time, bearing.all(axis=1).astype(float)+row*1.3, where="post", color=colors[case], lw=.8, label=case+" both eligible")
        axes[1].step(time, (~bearing.any(axis=1)).astype(float)+row*1.3, where="post", color=colors[case], lw=.7, ls="--", label=case+" neither eligible")
        nonwheel = contact._field(current, "nonwheel_ground_contact_force_norm_n")
        axes[2].step(time, (nonwheel > 1.).astype(float)+row*1.3, where="post", color=colors[case], lw=.8, label=case+" nonwheel ground>1N")
    axes[0].set_ylabel("Original root tilt [deg]"); axes[1].set_ylabel("Both-wheel eligibility\n(offset rows)")
    axes[2].set_ylabel("Nonwheel contact indicator\n(offset rows)"); axes[2].set_xlabel("Original native physical time [s]")
    for axis in axes:
        axis.axvspan(1.44, 4., color="#888888", alpha=.08); axis.set_xlim(0., 6.98); axis.grid(alpha=.2); axis.legend(fontsize=7, loc="upper left")
    fig.suptitle("Matched200Hz negative example: replace grounded function +0.4, NOT increase its weight\n0.1s persistent tilt events; zero load is NOT a flight/failure label;200/400Hz contexts not resampled", fontsize=11)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def write_events_csv(summary, destination):
    columns = ("case", "condition", "onset_s", "confirmation_s", "end_s", "observed_duration_s", "native_samples", "left_window_censored", "right_window_censored", "duration_is_lower_bound")
    with Path(destination).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for case in CASE_ORDER:
            for condition, result in summary["cases"][case]["events_full"].items():
                for event in result["events"]:
                    writer.writerow({"case": case, "condition": condition, **event})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=ROOT, help="Repository root containing the pinned historical output folders")
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--save-native", action="store_true", help="Copy original native arrays into a separate compressed reproducibility archive")
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    raw, summary = load_comparison(args.input_root, args.reference)
    output.mkdir(parents=True, exist_ok=False)
    render_timeline(raw, summary, output/"event_timeline.png")
    write_events_csv(summary, output/"events.csv"); artifacts = ["event_timeline.png", "events.csv"]
    if args.save_native:
        np.savez_compressed(output/"native_states.npz", **{case+"__"+key: value for case, arrays in raw.items() for key, value in arrays.items()})
        artifacts.append("native_states.npz")
    summary["artifacts"] = {name: {"file": name, "sha256": smooth.sha256(output/name)} for name in artifacts}
    with (output/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    result = run(parser().parse_args()); print(json.dumps({"status": result["status"], "new_simulation": False}))
