"""Offline, source-bound zero-shot Kd10/Kd5 contact diagnostic for one PPO.

No inference or simulation is performed. The policy was trained with Kd10 and
action-rate weight -0.5; Kd5 is a deployment-only intervention, not retraining.
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
from summarize_ppo_smoothness import posture_diagnostics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT/"outputs/contact_damping_20261021_v2"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
DEFAULT_WRAPPER_SHA256 = "5353f849d7d4f263c735d2cb000c8000403bf81280b51634065c9b721bb3b277"
ORDER = ("kd10", "kd5")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_report(report, kd, expected_wrapper_sha256):
    """Validate actual evaluator declarations/readbacks, not filename labels."""
    require(len(expected_wrapper_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_wrapper_sha256), "Invalid expected wrapper SHA")
    audit = report["contact_stability_diagnostic"]
    expected = {"kind": "fixed_ppo_nominal_contact_damping_diagnostic", "diagnostic_only": True,
        "deployment_leg_kd": kd, "trained_leg_kd": 10., "unchanged_leg_kp": 500.,
        "checkpoint_sha256": PINS["checkpoint"], "additional_training_updates": 0,
        "physics_dt_s": .005, "policy_dt_s": .02, "evaluation_contact_friction": .7,
        "trained_action_rate_weight": -.5, "training_at_deployment_gain_claimed": False,
        "reward_changes": False, "reset_changes": False, "force_filter_changes": False,
        "only_changed_control_field": "leg_kd" if kd == 5 else None,
        "legacy_evaluator_sha256": BASE_EVALUATOR_SHA256,
        "training_dependency_sha256": TRAINING_DEPENDENCIES,
        "evaluation_dependency_sha256": EVALUATION_DEPENDENCIES,
        "training_evaluation_dependencies_identical": False,
        "wrapper_sha256": expected_wrapper_sha256, "hardware_ready": False,
        "statistical_robustness_claimed": False}
    for key, value in expected.items():
        require(audit.get(key) == value, "Damping diagnostic contract differs: "+key)
    require(all(audit["input_sha256"].get(key) == value for key, value in PINS.items()), "Pinned policy/training/reference input hashes differ")
    require(report["simulator"] == "IsaacLab/PhysX" and report["status"] == "evaluated", "Require completed PhysX evaluation")
    require(report["checkpoint_sha256"] == PINS["checkpoint"] and report["motion_file_sha256"] == PINS["motion_file"]
        and report["asset_root_file_sha256"] == PINS["asset_path"], "Evaluation actor, motion or asset differs")
    require(report["evaluator_source_sha256"] == expected_wrapper_sha256, "Reported evaluator wrapper SHA differs")
    require(report["drift_module_sha256"] == EVALUATION_DEPENDENCIES["training/tron1_drift.py"]
        and report["contact_slip_helper_sha256"] == EVALUATION_DEPENDENCIES["training/tron1_contact_slip.py"], "Evaluation dependency differs")
    require(report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
        and report["full_horizon_diagnostic"] is True and report["reference_frames"] == 349, "Require deterministic complete nominal diagnostic")
    episodes = report["summary"]["episodes"]
    require(len(episodes) == 1 and episodes[0]["completed_full_reference"] is True and episodes[0]["start_reference_frame"] == 0, "Require one complete first episode")
    override = report["evaluation_contact_friction_override"]
    require(override["evaluation_contact_friction"] == .7 and override.get("only_changed_material_field") == "contact_friction"
        and "only_changed_physics_field" not in override, "Misleading or different material override")
    export = report["policy_export"]
    require(export["contact_stability_diagnostic"] == audit and export["checkpoint_sha256"] == PINS["checkpoint"]
        and export["leg_kp"] == 500. and export["leg_kd"] == kd
        and (export["physics_dt"], export["decimation"], export["policy_fps"]) == (.005, 4, 50.), "Exported deployed gains/clock/audit differ")
    export_error = export["export_validation_max_abs_error"]
    require(isinstance(export_error, (int, float)) and not isinstance(export_error, bool)
        and np.isfinite(export_error) and 0 <= export_error <= 1e-4, "Actor export validation error exceeds tolerance")
    material = report["contact_material_audit"]
    require(material["expected_friction"] == .7 and material["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [kd, kd]}, "Live actuator gain readback differs")
    require(np.allclose(material["robot_runtime_friction_range"], [.7, .7], rtol=0, atol=1e-6), "Live robot friction differs")
    require(len(material["ground_colliders"]) > 0 and all(np.allclose([row["static_friction"], row["dynamic_friction"]], [.7, .7], rtol=0, atol=1e-6) for row in material["ground_colliders"]), "Ground material differs")
    require(report["physics_recording"]["sample_dt_s"] == .005 and report["physics_recording"]["samples"] == 1397, "Native recording contract differs")
    require(report["contact_slip_measurement"]["active_point_normal_force_threshold_n"] == 5., "Point force threshold differs")
    return copy.deepcopy(audit)


def load_case(folder, kd, reference=DEFAULT_REFERENCE, expected_wrapper_sha256=DEFAULT_WRAPPER_SHA256):
    folder = Path(folder); report_path, raw_path = folder/"report.json", folder/"physics_trajectory.npz"
    report = json.loads(report_path.read_text())
    audit = audit_report(report, kd, expected_wrapper_sha256)
    require(sha256(reference) == PINS["motion_file"], "Locked reference SHA differs")
    require(sha256(raw_path) == report["physics_trajectory_sha256"], "Report-bound trajectory SHA differs")
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    time = np.asarray(raw["time_s"])
    require(time.shape == (1397,) and np.array_equal(time, np.arange(1397)*.005)
        and raw["valid_mask"].shape == (1397, 1) and _field(raw, "valid_mask").dtype.kind == "b"
        and _field(raw, "valid_mask").all(), "Require original complete native clock/episode, no reset splice")
    require(tuple(raw["wheel_body_names"].tolist()) == ("wheel_L_Link", "wheel_R_Link"), "Wheel ordering differs")
    decisions = deduplicate_actions(time, _field(raw, "actions_clipped"), _field(raw, "action_was_applied"), float(raw["policy_dt_s"]),
        frames=_field(raw, "reference_frame_for_action"), counters=raw["simulation_step_counter"])
    require(decisions["decisions"] == 349 and float(raw["policy_dt_s"]) == .02, "Control decision clock differs")
    contact = analyze_signals(time, _field(raw, "wheel_ground_contact_count"), _field(raw, "wheel_ground_contact_force_w_n"),
        _field(raw, "root_linear_velocity_w_m_s"), _field(raw, "root_angular_velocity_w_rad_s"),
        nonwheel_counts=_field(raw, "nonwheel_ground_contact_count") if "nonwheel_ground_contact_count" in raw else None)
    analysis = analyze_trajectory(raw_path, reference=reference, report=report_path)
    for name in ("full", "core"):
        contact[name]["tracking"] = analysis["windows"][name]["tracking"]
        contact[name]["body"] = analysis["windows"][name]["body"]
        contact[name]["action"] = analysis["windows"][name]["action"]
        contact[name]["torque"] = analysis["windows"][name]["torque"]
    return raw, {"deployed_leg_kd": kd, "trained_leg_kd": 10., "action_rate_weight": -.5,
        "input_sha256": {"report.json": sha256(report_path), "physics_trajectory.npz": sha256(raw_path)},
        "diagnostic_contract": audit, "actor_export_validation_max_abs_error": report["policy_export"]["export_validation_max_abs_error"],
        "windows": contact, "posture": posture_diagnostics(raw)}, report


def comparable_export(report):
    export = copy.deepcopy(report["policy_export"])
    export.pop("contact_stability_diagnostic"); export.pop("leg_kd")
    export.pop("export_validation_max_abs_error")  # Validated <=1e-4 separately; observation-dependent roundoff, not a control setting.
    export["task_profile"].pop("leg_kd", None)
    return export


def load_comparison(input_root=DEFAULT_INPUT, reference=DEFAULT_REFERENCE, expected_wrapper_sha256=DEFAULT_WRAPPER_SHA256):
    raw, cases, reports = {}, {}, {}
    for label, kd in (("kd10", 10.), ("kd5", 5.)):
        raw[label], cases[label], reports[label] = load_case(Path(input_root)/label, kd, reference, expected_wrapper_sha256)
    for key in ("drift_module_sha256", "contact_slip_helper_sha256", "physics_recording", "legacy_joint_friction_audit", "contact_slip_measurement"):
        require(reports["kd10"][key] == reports["kd5"][key], "Shared evaluation setting differs: "+key)
    materials = [copy.deepcopy(reports[label]["contact_material_audit"]) for label in ORDER]
    for material in materials:
        material.pop("leg_pd_gain_readback")
    require(materials[0] == materials[1], "Additional material setting differs")
    require(comparable_export(reports["kd10"]) == comparable_export(reports["kd5"]), "Additional exported control/task setting differs beyond leg Kd")
    for key in ("input_sha256", "generated_evaluator_sha256", "wrapper_sha256"):
        require(cases["kd10"]["diagnostic_contract"][key] == cases["kd5"]["diagnostic_contract"][key], "Shared evaluator or source input differs: "+key)
    changes = {}
    for window in ("full", "core"):
        before, after = cases["kd10"]["windows"][window], cases["kd5"]["windows"][window]
        kind = "measured_force_bearing_contact"
        changes[window] = {"both_contact_duty_percentage_point_change": 100*(after["contact"][kind]["both"]["sample_duty_fraction"]-before["contact"][kind]["both"]["sample_duty_fraction"]),
            "neither_contact_duty_percentage_point_change": 100*(after["contact"][kind]["none"]["sample_duty_fraction"]-before["contact"][kind]["none"]["sample_duty_fraction"]),
            "angular_xy_hf_rms_percent_change": 100*(after["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"]/before["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"]-1) if before["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"] > 0 else None}
    return raw, {"schema_version": 1, "kind": "fixed_actor_zero_shot_leg_damping_contact_diagnostic",
        "checkpoint_sha256": PINS["checkpoint"], "reference_sha256": PINS["motion_file"],
        "trained_action_rate_weight": -.5, "trained_leg_kp": 500., "trained_leg_kd": 10.,
        "deployed_leg_kp": 500., "deployed_leg_kd": [10., 5.], "evaluation_mu": .7,
        "evaluation_engine": "PhysX", "expected_evaluator_wrapper_sha256": expected_wrapper_sha256,
        "case_order": list(ORDER), "cases": cases, "kd5_minus_kd10": changes,
        "native_physical_hz": 200., "control_hz": 50., "core_interval_s": [1.44, 4.],
        "new_training": False, "additional_training_updates": 0, "new_inference_by_analysis": False,
        "height_or_flight_failure_gate": False, "stability_success_claimed": False,
        "point_count_definition": "Measured ground points with individual normal force >5 N; zero count is not proof of flight",
        "force_definition": "All recorded ground normal force vectors summed per wheel; not tangential friction force",
        "load_eligibility_definition": "count>0 AND aggregate upward Fz>5N; separate from count-only duty",
        "limitations": ["One episode at one nominal coefficient for one frozen actor; no robustness certification or retrained-policy comparison.",
            "Kd5 is zero-shot deployment outside the trained Kd10 setting. Only the deployment leg damping changes.",
            "Historical training and frozen evaluation drift-module hashes differ; both cases intentionally retain the same evaluator cohort.",
            "Thresholded contact counts cannot recover weak geometric contacts; missing count does not imply flight.",
            "50Hz stride-4 summaries can alias native 200Hz contact switches. Window-edge missing-contact runs are censored lower bounds.",
            "Full-horizon tracking outside reference support is explicitly unavailable; post-core tail must not substitute for maneuver-core contact.",
            "No measured live robot mass or wheel vertical velocity is available; forces remain N, no inferred mg or differentiated wheel kinematics."]}


def render_time_figure(raw_by_kd, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True, sharey="col")
    maximum_force = max(float(_field(raw, "wheel_ground_contact_force_w_n")[:, :, 2].max()) for raw in raw_by_kd.values())
    maximum_omega = max(float(np.abs(_field(raw, "root_angular_velocity_w_rad_s")[:, 0]).max()) for raw in raw_by_kd.values())
    for row, label in enumerate(ORDER):
        raw = raw_by_kd[label]; time = raw["time_s"]
        force, count = _field(raw, "wheel_ground_contact_force_w_n")[:, :, 2], _field(raw, "wheel_ground_contact_count")
        for side, color in enumerate(("#D55E00", "#8B4BA6")):
            axes[row, 0].plot(time, force[:, side], lw=.65, color=color, label=("Left", "Right")[side])
        axes[row, 0].set(ylim=(0, maximum_force*1.05), ylabel=f"Deployed Kd={summary['cases'][label]['deployed_leg_kd']:g}\nGround Fz [N]")
        axes[row, 1].imshow((count > 0).T, origin="lower", aspect="auto", interpolation="none", extent=(0, 6.98, -.5, 1.5), vmin=0, vmax=1, cmap="Greys")
        axes[row, 1].set(yticks=[0, 1], yticklabels=["L", "R"])
        axes[row, 2].plot(time, _field(raw, "root_angular_velocity_w_rad_s")[:, 0], lw=.65, color="#0072B2")
        axes[row, 2].set(ylim=(-maximum_omega*1.05, maximum_omega*1.05), ylabel="World omega X [rad/s]")
        for ax in axes[row]:
            ax.axvspan(1.44, 4., color="#f0b429", alpha=.15); ax.set_xlim(0, 6.98); ax.grid(alpha=.2); ax.set_xlabel("Original physical time [s]")
    for ax, title in zip(axes[0], ("Native 200Hz wheel ground-normal forces", "Count>0 contact raster (not a flight label)", "Native body angular velocity")):
        ax.set_title(title)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Same frozen -0.5 PPO actor | PhysX mu=0.7 | trained Kd10, deployment-only Kd10 vs Kd5")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing artifacts: choose a fresh output directory")
    raw, summary = load_comparison(args.input_root, args.reference, args.expected_wrapper_sha256)
    output.mkdir(parents=True)
    render_time_figure(raw, summary, output/"contact_comparison.png")
    arrays = {label+"_"+key: value for label, current in raw.items() for key, value in current.items()}
    np.savez_compressed(output/"recorded_states.npz", **arrays)
    summary["analysis_source_sha256"] = sha256(__file__)
    summary["helper_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in ("analyze_ppo_contact_stability.py", "analyze_ppo_smoothness.py", "summarize_ppo_smoothness.py", "eval_ppo_contact_damping.py")}
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in ("contact_comparison.png", "recorded_states.npz")}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--expected-wrapper-sha256", default=DEFAULT_WRAPPER_SHA256)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": "analyzed_existing_damping_recordings", "new_training": False, "cases": result["case_order"]}))
