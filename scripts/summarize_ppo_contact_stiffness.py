"""Offline fixed-actor Kp500/Kp250 diagnostic; Kd10 and nominal mu=.7 held.

The Kp500 recording is reused unchanged from the completed damping diagnostic.
The Kp250 episode is a zero-shot deployment change, never a retrained policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from analyze_ppo_contact_stability import analyze_signals, _field
from analyze_ppo_smoothness import analyze_trajectory, deduplicate_actions, sha256
from eval_ppo_contact_damping import BASE_EVALUATOR_SHA256, TRAINING_DEPENDENCIES, EVALUATION_DEPENDENCIES, PINS
from summarize_ppo_contact_damping import load_case as load_damping_baseline, DEFAULT_WRAPPER_SHA256 as BASE_WRAPPER_SHA256, require
from summarize_ppo_smoothness import posture_diagnostics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT/"outputs/contact_damping_20261021_v2/kd10"
DEFAULT_CANDIDATE = ROOT/"outputs/contact_stiffness_20261021_v1/kp250"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
DEFAULT_WRAPPER_SHA256 = "ff9b11f38f3f3a7bb69257abf99521e6b0f4535d9b9ad85348dd2559194dbac9"
BASE_REPORT_SHA256 = "fdf3881b765276832dbbdfe0528446055a4af91a6147ca3520dc847f6906dc77"
BASE_TRACE_SHA256 = "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780"
ORDER = ("kp500", "kp250")


def audit_stiffness_report(report, expected_wrapper_sha256=DEFAULT_WRAPPER_SHA256):
    require(len(expected_wrapper_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_wrapper_sha256), "Invalid expected wrapper SHA")
    audit = report["contact_stability_diagnostic"]
    expected = {"kind": "fixed_ppo_nominal_contact_stiffness_diagnostic", "diagnostic_only": True,
        "deployment_leg_kp": 250., "trained_leg_kp": 500., "unchanged_leg_kd": 10.,
        "checkpoint_sha256": PINS["checkpoint"], "additional_training_updates": 0,
        "physics_dt_s": .005, "policy_dt_s": .02, "evaluation_contact_friction": .7,
        "trained_action_rate_weight": -.5, "training_at_deployment_gain_claimed": False,
        "reward_changes": False, "reset_changes": False, "force_filter_changes": False,
        "only_changed_control_field": "leg_kp", "legacy_evaluator_sha256": BASE_EVALUATOR_SHA256,
        "training_dependency_sha256": TRAINING_DEPENDENCIES, "evaluation_dependency_sha256": EVALUATION_DEPENDENCIES,
        "training_evaluation_dependencies_identical": False, "wrapper_sha256": expected_wrapper_sha256,
        "provenance_adapter_sha256": BASE_WRAPPER_SHA256, "hardware_ready": False, "statistical_robustness_claimed": False}
    for key, value in expected.items():
        require(audit.get(key) == value, "Stiffness diagnostic contract differs: "+key)
    require(not any(key in audit for key in ("deployment_leg_kd", "trained_leg_kd", "unchanged_leg_kp")), "Misleading damping-only audit fields in stiffness case")
    require(all(audit["input_sha256"].get(key) == value for key, value in PINS.items()), "Pinned training/actor/reference hashes differ")
    require(report["simulator"] == "IsaacLab/PhysX" and report["status"] == "evaluated"
        and report["evaluator_source_sha256"] == expected_wrapper_sha256, "Completed evaluator wrapper identity differs")
    require(report["checkpoint_sha256"] == PINS["checkpoint"] and report["motion_file_sha256"] == PINS["motion_file"]
        and report["asset_root_file_sha256"] == PINS["asset_path"], "Recorded policy/motion/asset identity differs")
    require(report["drift_module_sha256"] == EVALUATION_DEPENDENCIES["training/tron1_drift.py"]
        and report["contact_slip_helper_sha256"] == EVALUATION_DEPENDENCIES["training/tron1_contact_slip.py"], "Evaluation dependency differs")
    require(report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
        and report["full_horizon_diagnostic"] is True and report["reference_frames"] == 349, "Require fixed nominal deterministic full horizon")
    episodes = report["summary"]["episodes"]
    require(len(episodes) == 1 and episodes[0]["completed_full_reference"] is True and episodes[0]["start_reference_frame"] == 0, "Require one completed first episode")
    override = report["evaluation_contact_friction_override"]
    require(override["evaluation_contact_friction"] == .7 and override.get("only_changed_material_field") == "contact_friction"
        and "only_changed_physics_field" not in override, "Material override declaration differs")
    export = report["policy_export"]
    require(export["contact_stability_diagnostic"] == audit and export["checkpoint_sha256"] == PINS["checkpoint"]
        and export["leg_kp"] == 250. and export["leg_kd"] == 10.
        and (export["physics_dt"], export["decimation"], export["policy_fps"]) == (.005, 4, 50.), "Exported deployed controller/clock differs")
    error = export["export_validation_max_abs_error"]
    require(isinstance(error, (int, float)) and not isinstance(error, bool) and np.isfinite(error)
        and 0 <= error <= 1e-4, "Actor export validation error exceeds tolerance")
    material = report["contact_material_audit"]
    require(material["expected_friction"] == .7 and material["leg_pd_gain_readback"] == {"kp_range": [250., 250.], "kd_range": [10., 10.]}, "Live actuator gain readback differs")
    require(np.allclose(material["robot_runtime_friction_range"], [.7, .7], rtol=0, atol=1e-6), "Live robot friction differs")
    require(len(material["ground_colliders"]) > 0 and all(np.allclose([row["static_friction"], row["dynamic_friction"]], [.7, .7], rtol=0, atol=1e-6) for row in material["ground_colliders"]), "Ground friction differs")
    require(report["physics_recording"]["sample_dt_s"] == .005 and report["physics_recording"]["samples"] == 1397
        and report["contact_slip_measurement"]["active_point_normal_force_threshold_n"] == 5., "Physical contact recorder/threshold differs")
    return copy.deepcopy(audit)


def load_candidate(folder=DEFAULT_CANDIDATE, reference=DEFAULT_REFERENCE, expected_wrapper_sha256=DEFAULT_WRAPPER_SHA256):
    folder = Path(folder); report_path, raw_path = folder/"report.json", folder/"physics_trajectory.npz"
    report = json.loads(report_path.read_text()); audit = audit_stiffness_report(report, expected_wrapper_sha256)
    require(sha256(reference) == PINS["motion_file"], "Locked reference SHA differs")
    require(sha256(raw_path) == report["physics_trajectory_sha256"], "Report-bound trajectory SHA differs")
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    time = np.asarray(raw["time_s"])
    require(time.shape == (1397,) and np.array_equal(time, np.arange(1397)*.005)
        and raw["valid_mask"].shape == (1397, 1) and raw["valid_mask"].dtype.kind == "b" and raw["valid_mask"].all(), "Require unchanged first-episode native clock; no reset splice")
    require(tuple(raw["wheel_body_names"].tolist()) == ("wheel_L_Link", "wheel_R_Link"), "Wheel ordering differs")
    decisions = deduplicate_actions(time, _field(raw, "actions_clipped"), _field(raw, "action_was_applied"), float(raw["policy_dt_s"]),
        frames=_field(raw, "reference_frame_for_action"), counters=raw["simulation_step_counter"])
    require(decisions["decisions"] == 349 and float(raw["policy_dt_s"]) == .02, "Control clock differs")
    windows = analyze_signals(time, _field(raw, "wheel_ground_contact_count"), _field(raw, "wheel_ground_contact_force_w_n"),
        _field(raw, "root_linear_velocity_w_m_s"), _field(raw, "root_angular_velocity_w_rad_s"),
        nonwheel_counts=_field(raw, "nonwheel_ground_contact_count") if "nonwheel_ground_contact_count" in raw else None)
    analysis = analyze_trajectory(raw_path, reference=reference, report=report_path)
    for name in ("full", "core"):
        windows[name].update({key: analysis["windows"][name][key] for key in ("tracking", "body", "action", "torque")})
    return raw, {"deployed_leg_kp": 250., "deployed_leg_kd": 10., "trained_leg_kp": 500., "trained_leg_kd": 10.,
        "action_rate_weight": -.5, "input_sha256": {"report.json": sha256(report_path), "physics_trajectory.npz": sha256(raw_path)},
        "diagnostic_contract": audit, "actor_export_validation_max_abs_error": report["policy_export"]["export_validation_max_abs_error"],
        "windows": windows, "posture": posture_diagnostics(raw)}, report


def comparable_export(report):
    value = copy.deepcopy(report["policy_export"])
    for key in ("contact_stability_diagnostic", "leg_kp", "export_validation_max_abs_error"):
        value.pop(key)
    value["task_profile"].pop("leg_kp", None)
    return value


def load_comparison(baseline=DEFAULT_BASELINE, candidate=DEFAULT_CANDIDATE, reference=DEFAULT_REFERENCE, expected_wrapper_sha256=DEFAULT_WRAPPER_SHA256):
    baseline = Path(baseline)
    require(sha256(baseline/"report.json") == BASE_REPORT_SHA256 and sha256(baseline/"physics_trajectory.npz") == BASE_TRACE_SHA256, "Reused baseline original artifact SHA differs")
    raw_base, base_case, base_report = load_damping_baseline(baseline, 10., reference, BASE_WRAPPER_SHA256)
    raw_new, new_case, new_report = load_candidate(candidate, reference, expected_wrapper_sha256)
    # Preserve the baseline's original damping-kind provenance, without adapting
    # or relabeling the recorded report as a new stiffness episode.
    base_case.update({"deployed_leg_kp": 500., "deployed_leg_kd": 10., "trained_leg_kp": 500.})
    for key in ("drift_module_sha256", "contact_slip_helper_sha256", "physics_recording", "legacy_joint_friction_audit", "contact_slip_measurement"):
        require(base_report[key] == new_report[key], "Common evaluator setting differs: "+key)
    materials = [copy.deepcopy(report["contact_material_audit"]) for report in (base_report, new_report)]
    for material in materials:
        material.pop("leg_pd_gain_readback")
    require(materials[0] == materials[1], "Additional material field differs")
    require(comparable_export(base_report) == comparable_export(new_report), "Additional exported control/task field differs beyond leg Kp")
    for key in ("input_sha256", "legacy_evaluator_sha256", "training_dependency_sha256", "evaluation_dependency_sha256"):
        require(base_case["diagnostic_contract"][key] == new_case["diagnostic_contract"][key], "Common source/training input differs: "+key)
    changes = {}
    for window in ("full", "core"):
        before, after = base_case["windows"][window], new_case["windows"][window]
        kind = "measured_force_bearing_contact"
        changes[window] = {"both_contact_duty_percentage_point_change": 100*(after["contact"][kind]["both"]["sample_duty_fraction"]-before["contact"][kind]["both"]["sample_duty_fraction"]),
            "neither_contact_duty_percentage_point_change": 100*(after["contact"][kind]["none"]["sample_duty_fraction"]-before["contact"][kind]["none"]["sample_duty_fraction"]),
            "angular_xy_hf_rms_percent_change": 100*(after["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"]/before["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"]-1) if before["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"] > 0 else None}
    return {"kp500": raw_base, "kp250": raw_new}, {"schema_version": 1, "kind": "fixed_actor_zero_shot_leg_stiffness_contact_diagnostic",
        "checkpoint_sha256": PINS["checkpoint"], "reference_sha256": PINS["motion_file"], "trained_action_rate_weight": -.5,
        "trained_leg_kp": 500., "trained_leg_kd": 10., "deployed_leg_kp": [500., 250.], "deployed_leg_kd": 10.,
        "evaluation_mu": .7, "evaluation_engine": "PhysX", "expected_candidate_wrapper_sha256": expected_wrapper_sha256,
        "baseline_wrapper_sha256": BASE_WRAPPER_SHA256, "reused_baseline_episode": True, "new_candidate_episodes": 1,
        "case_order": list(ORDER), "cases": {"kp500": base_case, "kp250": new_case}, "kp250_minus_kp500": changes,
        "native_physical_hz": 200., "control_hz": 50., "core_interval_s": [1.44, 4.], "new_training": False,
        "additional_training_updates": 0, "new_inference_by_analysis": False, "height_or_flight_failure_gate": False,
        "stability_success_claimed": False,
        "point_count_definition": "Ground points with individual normal force >5N; count>0 is distinct from count>0 AND aggregate upward Fz>5N eligibility",
        "force_definition": "Recorded summed ground normal force vectors in N; not friction force",
        "limitations": ["One reused baseline episode and one new nominal candidate episode of the same frozen actor; no statistical stability certification.",
            "Kp250 is zero-shot deployment; PPO was trained with Kp500/Kd10. The candidate is not retrained.",
            "Different adapter wrapper hashes are disclosed and independently checked; both retain the same SHA-pinned legacy evaluator and dependencies.",
            "Historical training/evaluation drift hashes differ, equally for both episodes; never claim training and evaluation code were identical.",
            "Zero thresholded contact count is not proof of geometric separation or flight; weak contacts cannot be recovered.",
            "50Hz subsampling can alias native200Hz contact switches; boundary run lengths are censored lower bounds.",
            "Post-core tail is not the maneuver core. Full tracking beyond the locked reference is unavailable rather than time-aligned or fabricated.",
            "No measured robot mass or recorded wheel vertical velocity is available; force remains N, no inferred mg or position-derived wheel velocity."]}


def render_time_figure(raw_by_case, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True, sharey="col")
    maximum_force = max(float(_field(raw, "wheel_ground_contact_force_w_n")[:, :, 2].max()) for raw in raw_by_case.values())
    maximum_omega = max(float(np.abs(_field(raw, "root_angular_velocity_w_rad_s")[:, 0]).max()) for raw in raw_by_case.values())
    for row, label in enumerate(ORDER):
        raw = raw_by_case[label]; time = raw["time_s"]
        force, count = _field(raw, "wheel_ground_contact_force_w_n")[:, :, 2], _field(raw, "wheel_ground_contact_count")
        for side, color in enumerate(("#D55E00", "#8B4BA6")):
            axes[row, 0].plot(time, force[:, side], lw=.65, color=color, label=("Left", "Right")[side])
        axes[row, 0].set(ylim=(0, maximum_force*1.05), ylabel=f"Deployed Kp={summary['cases'][label]['deployed_leg_kp']:g} / Kd10\nGround Fz [N]")
        axes[row, 1].imshow((count > 0).T, origin="lower", aspect="auto", interpolation="none", extent=(0, 6.98, -.5, 1.5), vmin=0, vmax=1, cmap="Greys")
        axes[row, 1].set(yticks=[0, 1], yticklabels=["L", "R"])
        axes[row, 2].plot(time, _field(raw, "root_angular_velocity_w_rad_s")[:, 0], lw=.65, color="#0072B2")
        axes[row, 2].set(ylim=(-maximum_omega*1.05, maximum_omega*1.05), ylabel="World omega X [rad/s]")
        for ax in axes[row]:
            ax.axvspan(1.44, 4., color="#f0b429", alpha=.15); ax.set_xlim(0, 6.98); ax.grid(alpha=.2); ax.set_xlabel("Original physical time [s]")
    for ax, title in zip(axes[0], ("Native200Hz ground-normal force", "Count>0 raster (not a flight label)", "Native angular velocity")):
        ax.set_title(title)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Same frozen -0.5 PPO | PhysX mu=.7 | trained Kp500/Kd10 | only deployment Kp changes")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    raw, summary = load_comparison(args.baseline, args.candidate, args.reference, args.expected_wrapper_sha256)
    output.mkdir(parents=True)
    render_time_figure(raw, summary, output/"contact_comparison.png")
    np.savez_compressed(output/"recorded_states.npz", **{label+"_"+key: value for label, current in raw.items() for key, value in current.items()})
    summary["analysis_source_sha256"] = sha256(__file__)
    summary["helper_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in
        ("summarize_ppo_contact_damping.py", "analyze_ppo_contact_stability.py", "analyze_ppo_smoothness.py", "summarize_ppo_smoothness.py", "eval_ppo_contact_damping.py", "eval_ppo_contact_stiffness.py")}
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in ("contact_comparison.png", "recorded_states.npz")}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--expected-wrapper-sha256", default=DEFAULT_WRAPPER_SHA256)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": "analyzed_existing_stiffness_recordings", "new_training": False, "case_order": result["case_order"]}))
