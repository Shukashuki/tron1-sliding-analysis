"""Offline comparison of four recorded PPO training/deployment recipe stages.

No new dynamics, policy inference, retarget, filtering or frequency conversion.
The dense-contact 200Hz negative example is verified and reported separately;
none of the four main stages used that replacement contact reward.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

import analyze_ppo_smoothness as smooth
import analyze_ppo_ground_support_failure as ground
from analyze_ppo_contact_stability import contact_masks, contact_statistics
from analyze_ppo_contact_timestep import band_spectrum
from controller_metrics import wrap
from controller_trace import native_trace
from nmpc_tracking_objective import quaternion_rotate
from ppo_backend_400hz_contract import require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("original200", "smooth200", "deploy400", "retrain400")
LABELS = {"original200": "A Original200", "smooth200": "B Smooth200",
          "deploy400": "C Same actor400", "retrain400": "D Retrained400"}
COLORS = ("#0072B2", "#009E73", "#E69F00", "#7B61A8")
EXPECTED = {"original200": (200, 200, -.1), "smooth200": (200, 200, -.5),
            "deploy400": (200, 400, -.5), "retrain400": (400, 400, -.5)}
BUNDLE_FIELDS = ("time_s", "position", "quaternion", "com_velocity", "angular_velocity", "contact_count", "contact_force")


def root_signals(trace):
    """Only original root/COM geometry; no invented wheel or contact states."""
    q = np.asarray(trace["quaternion"], dtype=float)
    require(np.allclose(np.linalg.norm(q, axis=1), 1., rtol=0, atol=1e-5), "Root quaternions must be normalized")
    forward = quaternion_rotate(q / np.linalg.norm(q, axis=1, keepdims=True), np.array([1., 0., 0.]))
    velocity = np.asarray(trace["com_velocity"])
    return {"root_directed_beta_rad": wrap(np.arctan2(velocity[:, 1], velocity[:, 0]) - np.arctan2(forward[:, 1], forward[:, 0])),
        "root_beta_valid": (np.linalg.norm(velocity[:, :2], axis=1) >= .25) & (np.linalg.norm(forward[:, :2], axis=1) > 1e-6)}


def load_stages(reference, model, legacy_evaluator):
    from render_ppo_training_recipe_comparison import load_stages as strict_loader
    return strict_loader(reference, model, legacy_evaluator)


def window_metrics(trace, reference, interval):
    time = np.asarray(trace["time_s"])
    dt = float(trace["physics_dt_s"])
    require(dt in (.005, .0025) and len(time) == round(6.98 / dt) + 1
            and np.allclose(time, np.arange(len(time)) * dt, rtol=0, atol=1e-10),
            "Require the original consecutive native200/400Hz full episode")
    select = (time >= interval[0] - 1e-9) & (time <= interval[1] + 1e-9)
    require(select.sum() >= 4, "Insufficient native window samples")
    tracking = smooth._tracking(time, trace["position"], trace["com_velocity"], trace["quaternion"], reference, interval)
    if tracking["available"]:
        signals = trace["signals"]
        _, ref_heading = smooth._orientation(reference["quaternion"])
        ref_beta = np.unwrap(np.arctan2(reference["velocity"][:, 1], reference["velocity"][:, 0]) - ref_heading)
        desired = np.interp(time[select], reference["time_s"], ref_beta)
        speed = np.interp(time[select], reference["time_s"], np.linalg.norm(reference["velocity"][:, :2], axis=1))
        valid = signals["root_beta_valid"][select] & (speed >= .25)
        error = wrap(signals["root_directed_beta_rad"][select] - desired)
        tracking["beta_valid_samples"] = int(valid.sum())
        tracking["beta_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(error[valid] ** 2)))) if valid.any() else None
    counts, forces = trace["contact_count"][select], trace["contact_force"][select]
    require(np.isfinite(counts).all() and (counts >= 0).all() and np.equal(counts, np.floor(counts)).all()
            and np.isfinite(forces).all(), "Invalid native contact telemetry")
    groups = contact_masks(counts, forces[:, :, 2])
    contact = {definition: {name: {key: value for key, value in contact_statistics(time[select], mask).items()
        if key in ("sample_duty_fraction", "time_duty_fraction_zoh", "state_switch_count", "state_switches_per_s")}
        for name, mask in group.items()} for definition, group in groups.items()}
    q = trace["quaternion"][select]
    tilt = np.degrees(np.arccos(np.clip(1. - 2. * np.sum(q[:, 1:3] ** 2, axis=1), -1., 1.)))
    return {"interval_s": list(interval), "native_samples": int(select.sum()), "tracking": tracking,
        "world_angular_xy_common10_100": band_spectrum(trace["angular_velocity"][select, :2], dt),
        "contact": contact, "maximum_tilt_deg": float(tilt.max()),
        "height_contact_tilt_or_beta_used_as_failure_gate": False}


def windows(trace, reference):
    return {name: window_metrics(trace, reference, interval) for name, interval in
        (("core", (1.44, 4.)), ("full", (0., 6.98)), ("reference_overlap", (0., 6.96)))}


def load_historical_negative(reference):
    """Use the existing strict source/training/report/raw verifier, not a fifth stage."""
    raw, evidence = ground.load_comparison(ROOT, reference)
    ref = smooth._load_reference(reference)
    cases = []
    for name in ("baseline200", "ground200"):
        trace = native_trace(raw[name], "policy_isaac")
        trace["signals"] = root_signals(trace)
        original = evidence["cases"][name]
        events = {condition: {key: value for key, value in event.items() if key != "events"}
                  for condition, event in original["events_full"].items()}
        cases.append({"id": name, "label": original["label"], "checkpoint_sha256": original["checkpoint_sha256"],
            "directory": original["source_folder"], "input_hashes": original["input_sha256"],
            "windows": windows(trace, ref), "events_full": events})
    return {"kind": "separate_historical200Hz_grounded_reward_replacement_negative",
        "not_a_fifth_main_stage": True, "not_a400Hz_contact_training_result": True,
        "grounded_weight_both": .4, "reward_weight_increased": False, "reward_function_replaced": True,
        "cases": cases, "matched_training_evidence": evidence["matched_training"],
        "strict_analyzer_sha256": sha256(ground.__file__), "source_limitations": evidence["limitations"],
        "all_published_load_duties_are_native_ZOH_time_weighted": True,
        "wheel_fk_not_used_or_inferred_for_this_root_beta_analysis": True}


def analyze(traces, reference, provenance):
    require(set(traces) == set(STAGES) and provenance["stage_order"] == list(STAGES), "Expected four frozen recipe stages")
    require(provenance["grounded_reward_weight_all"] == .4 and provenance["dense_ground_support_reward_used"] is False,
            "Main stages must retain the original contact reward, not claim new dense400Hz training")
    require(provenance["all_main_stages_sideslip_reward_weight"] == 2., "Main comparison must contain only beta-ON stages")
    identities = {row["id"]: row for row in provenance["rows"]}
    require(len(identities) == 4 and set(identities) == set(STAGES), "Mixed/duplicate stage identities")
    rows = []
    for name in STAGES:
        original = identities[name]
        expected = EXPECTED[name]
        require((original["training_physics_hz"], original["evaluation_physics_hz"], original["action_rate_weight"]) == expected,
                "Stage training/deployment clock/action identity differs: " + name)
        trace = traces[name]
        require(float(trace["physics_dt_s"]) == 1. / expected[1], "Native stage clock differs")
        row = {**copy.deepcopy(original), "windows": windows(trace, reference)}
        require(row["windows"]["core"]["native_samples"] == (513 if expected[1] == 200 else 1025),
                "Native core sample count differs")
        rows.append(row)
    require(identities["smooth200"]["checkpoint_sha256"] == identities["deploy400"]["checkpoint_sha256"]
            and identities["original200"]["checkpoint_sha256"] != identities["smooth200"]["checkpoint_sha256"]
            and identities["retrain400"]["checkpoint_sha256"] != identities["smooth200"]["checkpoint_sha256"],
            "Only B/C must share the same frozen actor")
    return {"schema_version": 1, "kind": "recorded_PPO_training_recipe_four_stage_comparison",
        "stage_order": list(STAGES), "evaluation_mu": .7, "core_interval_s": [1.44, 4.],
        "reference_real_last_time_s": 6.96, "native_episode_last_time_s": 6.98,
        "sample_size_per_stage": 1, "rows": rows, "loader_provenance": provenance,
        "original_grounded_reward_weight_all": .4, "dense400Hz_contact_training_exists": False,
        "all_main_stages_sideslip_reward_weight": 2., "beta_on_off_ablation": False,
        "new_training": False, "new_simulation": False, "new_policy_inference": False,
        "state_resampling_filtering_or_alignment": False, "visual_reviewed": False,
        "causal_frequency_effect_claimed": False, "continuous_contact_certified": False,
        "definitions": {"beta": "COM course minus projected body+X; wrapped error against intended nonzero reference; speed/axis validity retained",
            "jitter": "World angular X/Y channel-mean RMS, linear detrend/Hann/Parseval, strictly10<f<100Hz at each actual native rate",
            "load_eligibility": "Recorded ground-contact count>0 AND world ground-normal Fz>5N, separately per wheel",
            "duty": "Time-weighted native zero-order hold, sum(mask[:-1]*diff(time))/(last-first); not sample mean",
            "failure_gates": "No height, tilt, zero-load or nonzero-beta failure gate; contact is diagnostic only"},
        "limitations": ["A/B are independently retrained historical200Hz checkpoints; one seed is not a statistical causal action-weight effect.",
            "B/C reuse one frozen200Hz-trained actor and change deployment physics AND explicitPD rate; policy stays50Hz and closed-loop actions can differ.",
            "D is retrained400Hz with shared representable MJLab-style DR and setup/reseed ordering, unlike historicalDR/setup. It does not isolate training frequency or contact handling.",
            "All four keep the ORIGINAL last-substep grounded+0.4 reward. No trained400Hz dense-contact replacement is available.",
            "Equal nominalmu does not calibrate different material mappings/contact laws; contact sample/history horizons also change with native rate.",
            "Full6.98s tracking is unavailable beyond the6.96s reference; reference_overlap is separate and no extrapolation is used.",
            "Similar XY does not establish equivalent dynamics, low jitter or stable continuous contact. Zero measured load is not a flight label."]}


def write_bundle(path, traces, reference):
    arrays = {name + "__" + field: np.asarray(traces[name][field]) for name in STAGES for field in BUNDLE_FIELDS}
    arrays.update({name + "__physics_dt_s": np.asarray(traces[name]["physics_dt_s"]) for name in STAGES})
    arrays.update({"reference__" + field: np.asarray(reference[field]) for field in ("time_s", "position", "quaternion", "velocity")})
    np.savez_compressed(path, **arrays)
    return {"format": "NPZ; allow_pickle=False", "native_fields_per_stage": list(BUNDLE_FIELDS) + ["physics_dt_s"],
        "reference_fields": ["time_s", "position", "quaternion", "velocity"],
        "sample_counts": {name: len(traces[name]["time_s"]) for name in STAGES},
        "reference_samples": len(reference["time_s"]), "native_states_copied_without_resampling": True,
        "checkpoint_actor_or_mesh_included": False,
        "scope": "Compact original numeric signals permit metric review, not independent reproduction of the strict source/metadata verification."}


def verify_bundle_metrics(path, summary):
    """Recompute published metrics from only the compact raw-signal bundle."""
    with np.load(path, allow_pickle=False) as archive:
        reference = {key: archive["reference__" + key] for key in ("time_s", "position", "quaternion", "velocity")}
        for row in summary["rows"]:
            name = row["id"]
            trace = {key: archive[name + "__" + key] for key in BUNDLE_FIELDS}
            trace["physics_dt_s"] = float(archive[name + "__physics_dt_s"])
            trace["signals"] = root_signals(trace)
            computed = windows(trace, reference)
            for label, metric in computed.items():
                expected = row["windows"][label]
                require(metric["native_samples"] == expected["native_samples"] and
                        metric["tracking"]["available"] == expected["tracking"]["available"], "Compact bundle window/clock differs")
                pairs = [(metric["world_angular_xy_common10_100"]["common_band_rms"], expected["world_angular_xy_common10_100"]["common_band_rms"])]
                for side in ("both", "none"):
                    pairs.append((metric["contact"]["load_eligibility"][side]["time_duty_fraction_zoh"],
                                  expected["contact"]["load_eligibility"][side]["time_duty_fraction_zoh"]))
                if metric["tracking"]["available"]:
                    require(metric["tracking"]["beta_valid_samples"] == expected["tracking"]["beta_valid_samples"],
                            "Compact bundle beta validity differs")
                    pairs.extend((metric["tracking"][key], expected["tracking"][key]) for key in
                                 ("xy_rmse_m", "xy_peak_m", "speed_retention", "beta_rmse_deg"))
                require(all((a is None and b is None) or (a is not None and b is not None and np.isclose(a, b, rtol=1e-9, atol=1e-10))
                            for a, b in pairs), "Compact raw-signal bundle does not recompute published metrics")
    return {"status": "verified", "stages_checked": 4, "windows_per_stage": 3,
        "uses_original_native_clocks": True, "new_physics_or_inference": False,
        "strict_training_metadata_verification_replaced": False}


def metric_table(summary):
    result = []
    for row in summary["rows"]:
        core = row["windows"]["core"]; tracking = core["tracking"]; load = core["contact"]["load_eligibility"]
        result.append({"stage": row["id"], "training_physics_hz": row["training_physics_hz"],
            "evaluation_physics_hz": row["evaluation_physics_hz"], "policy_hz": 50,
            "action_rate_weight": row["action_rate_weight"], "original_grounded_weight": .4,
            "xy_rmse_m": tracking["xy_rmse_m"], "xy_peak_m": tracking["xy_peak_m"],
            "speed_retention": tracking["speed_retention"], "beta_rmse_deg": tracking["beta_rmse_deg"],
            "beta_valid_samples": tracking["beta_valid_samples"],
            "world_angular_xy_common10_100_rms_rad_s": core["world_angular_xy_common10_100"]["common_band_rms"],
            "both_load_time_duty": load["both"]["time_duty_fraction_zoh"],
            "neither_load_time_duty": load["none"]["time_duty_fraction_zoh"], "checkpoint_sha256": row["checkpoint_sha256"]})
    return result


def render_metrics(table, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [LABELS[row["stage"]] + f"\n{row['training_physics_hz']}->{row['evaluation_physics_hz']}Hz / rate{row['action_rate_weight']:g}" for row in table]
    fields = (("xy_rmse_m", "Core XY RMSE (m)", 1.),
        ("world_angular_xy_common10_100_rms_rad_s", "World omegaXY common10<f<100Hz RMS (rad/s)", 1.),
        ("both_load_time_duty", "Both wheels load eligible (% time)", 100.),
        ("neither_load_time_duty", "Neither wheel load eligible (% time; NOT flight)", 100.))
    for axis, (key, title, scale) in zip(axes.flat, fields):
        values = [row[key] * scale for row in table]
        bars = axis.bar(np.arange(4), values, color=COLORS)
        axis.bar_label(bars, labels=[f"{value:.3f}" if scale == 1. else f"{value:.1f}" for value in values], padding=3)
        axis.set_xticks(np.arange(4), labels, fontsize=8)
        axis.set_title(title, fontsize=10); axis.set_ylim(0., max(max(values) * 1.22, 1e-6))
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    figure.suptitle("Recorded PPO recipe stages | mu=0.7, core1.44-4s, N=1/stage\nDescriptive combined recipes, NOT an isolated training-frequency or contact-reward effect", fontsize=12)
    figure.text(.5, .012, "A/B/C historicalDR; D sharedMJLab-styleDR. All original grounded+0.4. Dense200Hz negative reported separately. No contact/height failure gate.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0., .035, 1., .92)); figure.savefig(destination, dpi=160); plt.close(figure)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--reference", type=Path, default=ground.DEFAULT_REFERENCE)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--legacy-mujoco-evaluator", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve previous comparisons: use a fresh output directory")
    traces, reference, provenance = load_stages(args.reference, args.model, args.legacy_mujoco_evaluator)
    summary = analyze(traces, reference, provenance)
    negative = load_historical_negative(args.reference)
    summary["historical_contact_negative"] = {"file": "historical_ground_reward_negative.json",
        "scope": "Separate200Hz reward-function replacement at unchanged+0.4; never a fifth main400Hz stage"}
    source_names = (Path(__file__).name, "render_ppo_training_recipe_comparison.py", "analyze_ppo_smoothness.py",
        "analyze_ppo_contact_timestep.py", "analyze_ppo_contact_stability.py", "analyze_ppo_ground_support_failure.py",
        "controller_trace.py", "controller_metrics.py", "nmpc_tracking_objective.py")
    summary["analysis_sources_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in source_names}
    summary["inputs_sha256"] = {"reference": sha256(args.reference), "model": sha256(args.model),
        "legacy_mujoco_evaluator": sha256(args.legacy_mujoco_evaluator)}
    output.mkdir(parents=True)
    table = metric_table(summary)
    summary["compact_signal_bundle"] = write_bundle(output / "comparison_signals.npz", traces, reference)
    summary["compact_signal_bundle"]["recomputed_metrics"] = verify_bundle_metrics(output / "comparison_signals.npz", summary)
    with (output / "metric_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(table)
    render_metrics(table, output / "recipe_comparison.png")
    write_json(output / "historical_ground_reward_negative.json", negative)
    summary["artifacts"] = {name: sha256(output / name) for name in
        ("metric_table.csv", "recipe_comparison.png", "historical_ground_reward_negative.json", "comparison_signals.npz")}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run(parser().parse_args())
