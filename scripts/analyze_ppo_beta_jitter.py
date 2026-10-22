"""Offline nominal-mu PhysX PPO beta-reward ON/OFF jitter audit.

Use original 200-Hz physical recordings, not rendered motion or 50-Hz pose
differences. This is one seed/checkpoint pair, not a causal training study.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from analyze_ppo_smoothness import analyze_trajectory, spectral_metrics, sha256
from summarize_ppo_smoothness import extract_derived, spectrum_arrays


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ON = ROOT/"outputs/controller_primary_ppo_mu070_20261015_v1/mu0p700/isaac"
DEFAULT_OFF = ROOT/"outputs/no_sideslip_ppo_mu0p7_validation_20261016_v1/isaac"
DEFAULT_REFERENCE = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
MOTION_SHA = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
ASSET_SHA = "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e"
IDENTITIES = {
    "beta_on": {"checkpoint": "6ec411efeaea37fc61389285978b377aa7c5be80e326ce7f18581efc9acdf27b",
        "trajectory": "858a4dc5b11230d1ac5a894c9528060de90570265ebdea50f20a48005a021d61",
        "report": "c4d9ab3d9b2e0e1401b78d472057968bcdbd4f36f301015443b205c5d3659e92"},
    "beta_off": {"checkpoint": "cb336fa90a44ba73056fd6a1b4f5247a8a90fdcd636f0d05945ec973a24ec87f",
        "trajectory": "31dc01848e19623831c7754b6b9f8070cf8dd6a56d46222506cd278a5d9c1327",
        "report": "3dd5893e11533b4408fd2269e1cabb006ce0b90e20efecd26984f4ac1a190e62"}}
COLORS = {"beta_on": "#0072B2", "beta_off": "#D55E00"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _profile_pair(on, off):
    first, second = copy.deepcopy(on), copy.deepcopy(off)
    first.pop("description", None); second.pop("description", None)
    require(first["reward_weights"].pop("sideslip") == 2. and
            second["reward_weights"].pop("sideslip") == 0., "Require beta reward weights 2 -> 0")
    require(first == second, "Profiles differ beyond the isolated sideslip weight and prose description")


def audit_reports(on, off):
    for label, report in (("beta_on", on), ("beta_off", off)):
        require(report["checkpoint_sha256"] == IDENTITIES[label]["checkpoint"], "Wrong historical checkpoint")
        require(report["simulator"] == "IsaacLab/PhysX" and report["status"] == "evaluated", "Require completed PhysX reports")
        require(report["motion_file_sha256"] == MOTION_SHA and report["asset_root_file_sha256"] == ASSET_SHA,
                "Locked reference/asset identity differs")
        require(report["seed"] == 42 and report["deterministic_start_frame"] == 0 and
                report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False,
                "Require deterministic seed-42 fixed-condition evaluation")
        require(report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7 and
                report["physics_dt_s"] == .005 and report["reference_frames"] == 349 and
                report["full_horizon_diagnostic"] is True, "Nominal friction/native horizon differs")
    for field in ("task_profile", "evaluation_task_profile"):
        _profile_pair(on[field], off[field])
    equal_fields = ("contact_material_audit", "legacy_joint_friction_audit", "physics_recording",
        "evaluator_source_sha256", "drift_module_sha256", "contact_slip_helper_sha256", "runner_config_sha256")
    for field in equal_fields:
        require(field in on and on[field] == off[field], "Shared physical/source contract differs: "+field)
    first, second = copy.deepcopy(on["policy_export"]), copy.deepcopy(off["policy_export"])
    for field in ("task_profile", "trained_task_profile"):
        _profile_pair(first.pop(field), second.pop(field))
    for value in (first, second):
        value.pop("checkpoint_sha256")
        error = value.pop("export_validation_max_abs_error")
        require(np.isfinite(error) and 0 <= error < 1e-5, "Actor export readback error is excessive")
    require(first == second, "Policy/PD/action/observation/numerical export contracts differ")
    require(first["leg_kp"] == 500. and first["leg_kd"] == 10. and
            first["leg_position_scale"] == 1. and first["policy_fps"] == 50., "Historical actuator mapping differs")
    return {"profile_only_change_after_omitting_description": "drift_sideslip reward 2.0 -> 0.0",
        "equal_report_fields": list(equal_fields), "policy_contract_identical_except_checkpoint_profile_export_roundoff": True,
        "export_roundoff_limit": 1e-5, "wheel_xy_profile_changed": False,
        "evaluation_seed": 42, "evaluation_mu": .7,
        "training_recipe_parity_independently_verified_by_this_script": False,
        "scope": "Reported evaluation/source/profile identities; training DR/budget parity requires the separate training-manifest audit"}


def _signal(raw, name):
    value = np.asarray(raw[name])
    return value[:, 0] if value.ndim >= 2 and value.shape[1] == 1 else value


def _relative(on, off):
    return {"beta_on": on, "beta_off": off, "off_minus_on": off-on,
        "off_over_on_ratio": off/on if on > 1e-12 else None,
        "off_vs_on_change_percent": 100.*(off/on-1.) if on > 1e-12 else None,
        "off_vs_on_reduction_percent": 100.*(1.-off/on) if on > 1e-12 else None}


def analyze_pair(on_dir=DEFAULT_ON, off_dir=DEFAULT_OFF, reference=DEFAULT_REFERENCE):
    require(sha256(reference) == MOTION_SHA, "Require exact locked original motion, not a regenerated reference")
    reports = {label: json.loads((Path(folder)/"report.json").read_text())
               for label, folder in (("beta_on", on_dir), ("beta_off", off_dir))}
    parity = audit_reports(reports["beta_on"], reports["beta_off"])
    analyses, arrays, inputs = {}, {}, {"reference_motion.npz": MOTION_SHA}
    for label, folder in (("beta_on", on_dir), ("beta_off", off_dir)):
        folder = Path(folder); expected = IDENTITIES[label]
        require(sha256(folder/"report.json") == expected["report"] and
                sha256(folder/"physics_trajectory.npz") == expected["trajectory"], "Original artifact SHA differs: "+label)
        require(reports[label]["physics_trajectory_sha256"] == expected["trajectory"], "Report-to-recording SHA differs")
        analysis = analyze_trajectory(folder/"physics_trajectory.npz", reference=reference, report=folder/"report.json")
        require(analysis["physical_samples"] == 1397 and analysis["recorded_samples"] == 1397 and
                analysis["control_decisions"] == 349 and analysis["windows"]["core"]["physical_samples"] == 513 and
                analysis["windows"]["core"]["control_decisions"] == 129, "Incomplete native physical/control/core coverage")
        derived, raw = extract_derived(folder, label)
        require(np.array_equal(derived[label+"_time_s"], np.arange(1397)*.005), "Native clocks must remain exact, no resampling")
        arrays.update(derived)
        for name in ("actions_clipped", "action_was_applied", "reference_frame_for_action", "valid_mask", "simulation_step_counter"):
            arrays[label+"_native_"+name] = _signal(raw, name).copy()
        names = tuple(raw["joint_names"].tolist())
        legs = [index for index, name in enumerate(names) if not name.startswith("wheel_")]
        time = np.asarray(raw["time_s"])
        torque = _signal(raw, "applied_joint_torque_nm").astype(float)
        velocity = _signal(raw, "root_linear_velocity_w_m_s").astype(float)
        core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
        frequency, psd = spectrum_arrays(torque[core][:, legs])
        arrays[label+"_core_leg_torque_psd_frequency_hz"] = frequency
        arrays[label+"_core_leg_torque_psd_nm2_hz"] = psd
        for window, mask in (("full", np.ones(1397, dtype=bool)), ("core", core)):
            analysis["windows"][window]["body"]["com_vertical_velocity_spectrum"] = spectral_metrics(velocity[mask, 2], .005, 10.)
        analyses[label] = analysis
        inputs[label+"/report.json"] = expected["report"]
        inputs[label+"/physics_trajectory.npz"] = expected["trajectory"]
    require(np.array_equal(arrays["beta_on_time_s"], arrays["beta_off_time_s"]), "Recorded clocks differ")
    changes = []
    for window in ("full", "core"):
        before, after = analyses["beta_on"]["windows"][window], analyses["beta_off"]["windows"][window]
        for name, units, getter in (
            ("body_world_xy_angular_hf_rms", "rad/s", lambda value: value["body"]["angular_xy_spectrum"]["high_frequency_rms"]),
            ("leg_action_delta_rms", "normalized action (1 rad scale)", lambda value: value["action"]["leg_delta_rms"]),
            ("leg_applied_torque_hf_rms", "Nm", lambda value: value["torque"]["leg_spectrum"]["high_frequency_rms"]),
            ("com_vertical_velocity_hf_rms", "m/s", lambda value: value["body"]["com_vertical_velocity_spectrum"]["high_frequency_rms"])):
            changes.append({"window": window, "signal": name, "units": units, **_relative(getter(before), getter(after))})
    return arrays, {"schema_version": 1, "kind": "recorded_physx_ppo_beta_reward_nominal_jitter_audit",
        "reference_id": "drift_drl_traj4", "training_engine": "PhysX", "evaluation_engine": "PhysX", "evaluation_mu": .7,
        "beta_reward_weights": {"beta_on": 2., "beta_off": 0.}, "checkpoint_sha256": {key: value["checkpoint"] for key, value in IDENTITIES.items()},
        "report_parity_audit": parity, "cases": analyses, "relative_changes": changes,
        "training_run_confounds_from_separate_manifest_audit": {
            "completed_environment_transitions_each": 12288000,
            "beta_off_training_manifest_sha256": "ca865361294cad54678730ae4377866e294c3f247c7d471ee05b6d5259b2aa0f",
            "beta_off_training_run_status_sha256": "6965f6ee8da5d3504d084d354bdae6a168f8225ff0cac272456e97b09a409184",
            "external_evidence_note": "Separately read-only-verified training JSON identities, also documented in results/2026-10-16-no-sideslip-reward-ablation.md; not opened by this script",
            "setup_smoke_control_steps": {"beta_on": 0, "beta_off": 8},
            "reported_training_cuda_devices": {"beta_on": "cuda:2", "beta_off": "cuda:7"},
            "interpretation": "OFF reset after eight pretraining zero-action steps, but RNG/phase/event state may already have advanced; not a strict matched beta-only intervention",
            "independently_reverified_by_this_script": False},
        "input_sha256": inputs, "native_sample_hz": 200., "policy_sample_hz": 50.,
        "core_interval_s": [1.44, 4.], "high_frequency_cutoff_hz_exclusive": 10.,
        "plot_settings": {"layout": [2, 2], "colors": COLORS, "same_axes_for_both_conditions": True,
            "core_interval_s": [1.44, 4.], "psd_xlim_hz": [0., 100.], "highlight_band_hz": [22.5, 27.5]},
        "causal_beta_effect_claimed": False, "jitter_resolved_claimed": False,
        "new_training": False, "new_simulation": False, "new_policy_inference": False,
        "filtered_or_graphics_derived_signals": False, "height_or_flight_failure_gate": False,
        "limitations": ["One checkpoint and one deterministic episode per condition; no multi-seed causal inference.",
            "Training runs may use different GPUs; identical seeds do not imply bitwise-identical training or attribute every difference to beta.",
            "Separate training audit found ON setup smoke=0, OFF=8 zero-action steps and GPU cuda:2 versus cuda:7; the formal collection budget matches but initialization/RNG history is confounded.",
            "This script verifies original evaluator/report/profile identities; separate training-manifest evidence is needed for full DR/budget parity.",
            "Turning/yaw/speed and nonzero sideslip references remain; removing the explicit beta reward is not a zero-beta target.",
            "Physical 200-Hz recordings and 50-Hz actions are distinguished; video/pose aliasing is not used to infer physical 25-Hz oscillation.",
            "Mu=.7 is the nominal diagnostic only; low-friction success is not required or assessed here."]}


def render_diagnostics(arrays, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label in ("beta_on", "beta_off"):
        color = COLORS[label]; legend = "beta reward "+("ON (2)" if label == "beta_on" else "OFF (0)")
        time = arrays[label+"_time_s"]
        axes[0, 0].plot(time, arrays[label+"_angular_velocity_w_rad_s"][:, 0], color=color, lw=.6, label=legend)
        decision_t = arrays[label+"_decision_time_s"]
        core = (decision_t >= 1.44-1e-9) & (decision_t <= 4.+1e-9)
        axes[0, 1].plot(decision_t[core], arrays[label+"_hip_L_action"][core], color=color, lw=.8, label=legend)
        for axis, frequency, psd in (
            (axes[1, 0], arrays[label+"_core_psd_frequency_hz"], arrays[label+"_core_angular_xy_psd_rad2_s2_hz"]),
            (axes[1, 1], arrays[label+"_core_leg_torque_psd_frequency_hz"], arrays[label+"_core_leg_torque_psd_nm2_hz"])):
            axis.semilogy(frequency[1:], np.maximum(psd[1:], 1e-15), color=color, lw=1., label=legend)
    axes[0, 0].set(xlim=(0, 6.98), xlabel="Original physical time [s]", ylabel="Signed world omega X [rad/s]", title="Full original physical recording")
    axes[0, 0].axvspan(1.44, 4., color=".6", alpha=.15)
    axes[0, 1].set(xlim=(1.44, 4.), ylim=(-1.05, 1.05), xlabel="Original decision time [s]", ylabel="Clipped hip L action", title="Core 50-Hz actions; no filtering")
    for axis, title, ylabel in ((axes[1, 0], "Core world omega XY PSD", "PSD [(rad/s)^2/Hz]"),
                                (axes[1, 1], "Core leg applied-torque PSD", "PSD [Nm^2/Hz]")):
        axis.set(xlim=(0, 100), xlabel="Physical frequency [Hz]", ylabel=ylabel, title=title)
        axis.axvline(10., color="black", ls="--", lw=.8); axis.axvspan(22.5, 27.5, color="#56B4E9", alpha=.15)
    for axis in axes.flat:
        axis.grid(alpha=.18); axis.legend(fontsize=8)
    fig.suptitle("PhysX PPO beta ON/OFF | mu=0.70 | original native recordings\nOne seed/checkpoint pair; no causal or graphics-derived jitter claim", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(output, dpi=150); plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing evidence: choose a fresh output directory")
    arrays, summary = analyze_pair(args.on_dir, args.off_dir, args.reference)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_comparison.npz", **arrays)
    render_diagnostics(arrays, summary, output/"diagnostics.png")
    summary["analysis_source_sha256"] = {name: sha256(ROOT/"scripts"/name) for name in
        ("analyze_ppo_beta_jitter.py", "analyze_ppo_smoothness.py", "summarize_ppo_smoothness.py")}
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in ("native_comparison.npz", "diagnostics.png")}
    with (output/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--on-dir", type=Path, default=DEFAULT_ON)
    result.add_argument("--off-dir", type=Path, default=DEFAULT_OFF)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    summary = run(parser().parse_args())
    print(json.dumps({"status": "audited_existing_nominal_beta_pair", "mu": .7,
                      "causal_beta_effect_claimed": False, "changes": summary["relative_changes"]}))
