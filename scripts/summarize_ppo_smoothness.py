"""Publish an offline, metadata-bound PhysX PPO smoothing comparison.

Inputs are copied training JSON and recorded evaluations, not policy weights.
No simulator, policy inference, remote access, or new training is executed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_ppo_smoothness import analyze_trajectory, compare_candidate, deduplicate_actions, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = ROOT/"outputs/policy_primary_grid_raw_20261016_v1/PPO"
DEFAULT_PUBLIC_SUMMARY = ROOT/"results/policy_primary_friction_dynamics_20261016_v1/summary.json"
DEFAULT_BASELINE_METADATA = ROOT/"results/controller_primary_ppo_height_smooth_seed42_v1/training_metadata.json"
COEFFICIENTS = (.7, .15)
COLORS = {"baseline": "#0072B2", "candidate": "#D55E00"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def case_name(mu):
    return "mu"+f"{mu:.3f}".replace(".", "p")


def _digest(value, label):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "Invalid SHA-256: "+label)
    return value


def verify_candidate_training(training, *, motion_sha256, baseline_metadata):
    """Verify copied completion declarations; deliberately do not open weights."""
    training = Path(training)
    status = read_json(training/"run_status.json")
    manifest = read_json(training/"manifest.json")
    ablation = read_json(training/"ablation_manifest.json")
    runner = read_json(training/"runner_config.json")
    require(status.get("status") == ablation.get("status") == "training_completed"
            and status.get("trained_policy_claimed") is True and ablation.get("trained_policy_claimed") is True,
            "Require completed formal training status and ablation")
    require(ablation.get("initialization") == "from_scratch_no_checkpoint_resume"
            and ablation.get("smoke_only") is False and manifest.get("smoke_only") is False,
            "Require from-scratch formal candidate, not smoke or resumed weights")
    require(manifest.get("algorithm") == ablation.get("algorithm") == "PPO"
            and ablation.get("training_engine") == "PhysX", "Candidate algorithm or training engine differs")
    require(status.get("environment_transitions") == ablation.get("expected_environment_transitions")
            == ablation.get("actual_environment_transitions") == manifest.get("environment_transitions_requested") == 12288000,
            "Require exactly 12,288,000 completed candidate transitions")
    require(manifest.get("num_envs") == ablation.get("num_envs") == 512
            and manifest.get("iterations") == ablation.get("iterations") == 1000
            and manifest.get("seed") == ablation.get("seed") == 42
            and manifest.get("setup_smoke_control_steps") == 0
            and manifest.get("num_steps_per_env") == runner.get("num_steps_per_env") == 24,
            "Candidate budget, seed or setup-smoke recipe differs")
    require(ablation.get("final_training_manifest_sha256") == sha256(training/"manifest.json"),
            "Candidate training manifest SHA mismatch")
    override = ablation.get("reward_override", {})
    weight = override.get("weight")
    require(override.get("term") == "action_rate_l2" and override.get("baseline_weight") == -.1
            and isinstance(weight, (int, float)) and not isinstance(weight, bool) and np.isfinite(weight) and weight < -.1
            and override.get("applied_before_environment_creation") is True
            and override.get("function") == "isaaclab.envs.mdp.rewards.action_rate_l2"
            and override.get("divided_by_control_dt") is False,
            "Candidate is not the declared isolated normalized action-rate ablation")
    require(manifest.get("training_reward_overrides") == {"action_rate_l2": weight},
            "Training manifest and candidate override weight differ")
    require(manifest.get("motion_sha256") == ablation.get("motion_sha256") == motion_sha256,
            "Candidate training motion SHA differs")
    require(manifest.get("asset_sha256") == ablation.get("asset_sha256") == baseline_metadata["training_asset_sha256"],
            "Candidate training asset SHA differs")
    require(manifest.get("trainer_sha256") == ablation.get("historical_trainer_sha256") == baseline_metadata["trainer_sha256"]
            and manifest.get("height_adapter_sha256") == ablation.get("height_adapter_sha256") == baseline_metadata["height_adapter_sha256"],
            "Candidate historical trainer or height-adapter SHA differs")
    profile_hash = baseline_metadata["task_profile_sha256"]
    require(manifest.get("task_profile_sha256") == ablation.get("task_profile_sha256") == profile_hash,
            "Candidate baseline task-profile SHA differs")
    sources = baseline_metadata["source_training_modules_sha256"]
    require(manifest.get("source_training_modules_sha256") == ablation.get("baseline_source_sha256") == sources,
            "Candidate source module hashes differ from historical training")
    for name, value in baseline_metadata["training_domain_randomization"].items():
        if name != "robot_mapping":  # Historical human-readable wheel label differs; numerical recipe must not.
            require(manifest.get("recipe", {}).get(name) == value, "Candidate historical DR recipe differs: "+name)
    for name in ("DR_recipe_modified", "base_task_profile_file_modified", "baseline_checkpoint_modified"):
        require(ablation.get(name) is False, "Candidate declaration is not isolated: "+name)
    checkpoint = _digest(ablation.get("checkpoint_sha256"), "candidate checkpoint")
    require(checkpoint != baseline_metadata["checkpoint_sha256"], "Candidate checkpoint equals archived baseline")
    return {"action_rate_weight": float(weight), "checkpoint_sha256": checkpoint,
            "training_budget": {"num_envs": 512, "iterations": 1000, "steps_per_iteration": 24,
                                "environment_transitions": 12288000, "seed": 42},
            "initialization": "from_scratch_no_checkpoint_resume",
            "checkpoint_bytes_verified_locally": False,
            "checkpoint_evidence_scope": "Remote-computed SHA declared in copied ablation metadata and matched to recorded evaluator reports; local weight bytes are not opened or independently verified",
            "input_sha256": {name: sha256(training/name) for name in
                ("run_status.json", "manifest.json", "ablation_manifest.json", "runner_config.json")}}


def verify_evaluation(folder, *, mu, checkpoint_sha256, motion_sha256):
    folder = Path(folder)
    report = read_json(folder/"report.json")
    require(report.get("simulator") == "IsaacLab/PhysX" and report.get("status") == "evaluated",
            "Only completed PhysX recordings belong in this publication")
    require(report.get("checkpoint_sha256") == checkpoint_sha256 and report.get("motion_file_sha256") == motion_sha256,
            "Evaluation checkpoint/motion identity differs")
    require(report.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == mu,
            "Evaluation friction differs from case")
    require(report.get("physics_trajectory_sha256") == sha256(folder/"physics_trajectory.npz"),
            "Evaluation report-to-trajectory SHA mismatch")
    episodes = report.get("summary", {}).get("episodes", [])
    require(report.get("domain_randomization_enabled") is False and report.get("observation_noise_enabled") is False
            and report.get("full_horizon_diagnostic") is True and len(episodes) == 1
            and episodes[0].get("completed_full_reference") is True and episodes[0].get("start_reference_frame") == 0,
            "Require one deterministic full-horizon fixed-friction diagnostic episode")
    return report


def _array(raw, name):
    value = np.asarray(raw[name])
    if value.ndim >= 2 and value.shape[1] == 1:
        value = value[:, 0]
    return value


def spectrum_arrays(values, dt=.005):
    """Same channel-mean Hann/linear-detrend normalization as the analyzer."""
    value = np.asarray(values, dtype=float)
    n = len(value)
    design = np.column_stack((np.ones(n), np.arange(n, dtype=float)))
    value = value-design@np.linalg.lstsq(design, value, rcond=None)[0]
    window = np.hanning(n)
    power = np.mean(np.abs(np.fft.rfft(value*window[:, None], axis=0))**2, axis=1)/(n*np.sum(window**2))
    power[1:-1 if n % 2 == 0 else None] *= 2.
    power[0] = 0.
    frequency = np.fft.rfftfreq(n, dt)
    return frequency, power/(1./(n*dt))


def extract_derived(folder, label):
    with np.load(Path(folder)/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    time = np.asarray(raw["time_s"])
    require(time.shape == (1397,) and np.allclose(time, np.arange(1397)*.005, rtol=0, atol=1e-9),
            "Require unchanged complete 200-Hz physical recording")
    require(_array(raw, "valid_mask").dtype.kind == "b" and _array(raw, "valid_mask").all(),
            "Require full valid first episode, not reset-spliced or censored poses")
    names = tuple(raw["joint_names"].tolist())
    hip = names.index("hip_L_Joint")
    decisions = deduplicate_actions(time, _array(raw, "actions_clipped"), _array(raw, "action_was_applied"),
        float(raw["policy_dt_s"]), frames=_array(raw, "reference_frame_for_action"),
        counters=raw.get("simulation_step_counter"))
    require(decisions["decisions"] == 349, "Require all 349 original control intervals")
    angular = _array(raw, "root_angular_velocity_w_rad_s").astype(float)
    core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
    frequency, psd = spectrum_arrays(angular[core, :2])
    arrays = {"time_s": time, "position_m": _array(raw, "root_position_m"),
        "quaternion_wxyz": _array(raw, "root_quaternion_wxyz"),
        "com_velocity_w_m_s": _array(raw, "root_linear_velocity_w_m_s"),
        "angular_velocity_w_rad_s": angular, "joint_position_rad": _array(raw, "joint_position_rad"),
        "joint_velocity_rad_s": _array(raw, "joint_velocity_rad_s"),
        "applied_joint_torque_nm": _array(raw, "applied_joint_torque_nm"),
        "decision_time_s": decisions["time_s"], "decision_actions_clipped": decisions["actions"],
        "hip_L_action": decisions["actions"][:, hip], "joint_names": np.asarray(names),
        "core_psd_frequency_hz": frequency, "core_angular_xy_psd_rad2_s2_hz": psd}
    return {label+"_"+name: value for name, value in arrays.items()}, raw


def posture_diagnostics(raw):
    time = np.asarray(raw["time_s"])
    quaternion = _array(raw, "root_quaternion_wxyz").astype(float)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1.-2.*(quaternion[:, 1]**2+quaternion[:, 2]**2), -1., 1.)))
    height = _array(raw, "root_position_m")[:, 2]
    result = {}
    for name, mask in (("full", np.ones(len(time), dtype=bool)),
                       ("core", (time >= 1.44-1e-9) & (time <= 4.+1e-9))):
        selected = np.flatnonzero(mask)
        maximum = selected[int(np.argmax(tilt[mask]))]
        row = {"samples": len(selected), "max_tilt_deg": float(tilt[maximum]),
               "max_tilt_time_s": float(time[maximum]), "fraction_tilt_over_60_deg": float((tilt[mask] > 60).mean()),
               "height_range_m": [float(height[mask].min()), float(height[mask].max())],
               "height_or_flight_failure_gate": False, "posture_failure_gate": False}
        if "wheel_ground_contact_count" in raw and "wheel_ground_contact_force_w_n" in raw:
            counts, force = _array(raw, "wheel_ground_contact_count"), _array(raw, "wheel_ground_contact_force_w_n")
            contact = (counts > 0) & (force[:, :, 2] > 5.)
            row["eligible_wheel_contact_fraction_left_right"] = contact[mask].mean(axis=0).tolist()
            row["both_wheels_eligible_fraction"] = float(contact[mask].all(axis=1).mean())
            row["neither_wheel_eligible_fraction"] = float((~contact[mask].any(axis=1)).mean())
            row["contact_note"] = "Actual ground-contact count>0 AND upward force>5N; ineligibility alone is not flight"
        result[name] = row
    return result


def _reduction(candidate, baseline):
    return 100.*(1.-candidate/baseline) if baseline > 1e-12 else None


def load_publication(candidate_dir, reference, baseline_root=DEFAULT_BASELINE_ROOT, *,
                     public_summary=DEFAULT_PUBLIC_SUMMARY, baseline_metadata=DEFAULT_BASELINE_METADATA):
    candidate_dir, reference, baseline_root = Path(candidate_dir), Path(reference), Path(baseline_root)
    public, historical = read_json(public_summary), read_json(baseline_metadata)
    require(public.get("reference_id") == historical.get("reference_id") == "drift_drl_traj4",
            "Require the locked primary reference cohort")
    expected_training_hash = public["policies"]["PPO"]["source_xy_grid"]["input_sha256"]["training_metadata.json"]
    require(sha256(baseline_metadata) == expected_training_hash, "Historical training metadata SHA differs from published identity")
    reference_hash = sha256(reference)
    require(reference_hash == public["input_sha256"]["motion"] == historical["motion_sha256"],
            "Reference motion SHA differs from historical locked cohort")
    training = verify_candidate_training(candidate_dir/"training", motion_sha256=reference_hash,
                                         baseline_metadata=historical)
    inputs = {"baseline_public_summary.json": sha256(public_summary),
              "baseline_training_metadata.json": sha256(baseline_metadata), "reference_motion.npz": reference_hash,
              **{"candidate/training/"+name: value for name, value in training["input_sha256"].items()}}
    rows, cases, arrays = [], {}, {}
    for mu in COEFFICIENTS:
        name = case_name(mu)
        case = {}
        reports = {}
        for cohort, folder, checkpoint in (
            ("baseline", baseline_root/name/"isaac", historical["checkpoint_sha256"]),
            ("candidate", candidate_dir/"validation"/name/"isaac", training["checkpoint_sha256"])):
            report = verify_evaluation(folder, mu=mu, checkpoint_sha256=checkpoint, motion_sha256=reference_hash)
            reports[cohort] = report
            if cohort == "baseline":
                expected = public["runs"]["PPO_"+name]["input_sha256"]
                require(sha256(folder/"physics_trajectory.npz") == expected["isaac_trace"]
                        and sha256(folder/"report.json") == expected["isaac_report"], "Historical baseline artifact SHA differs")
            for filename in ("physics_trajectory.npz", "report.json"):
                inputs[cohort+"/"+name+"/isaac/"+filename] = sha256(folder/filename)
            analysis = analyze_trajectory(folder/"physics_trajectory.npz", reference=reference, report=folder/"report.json")
            require(analysis["physical_samples"] == 1397 and analysis["control_decisions"] == 349
                    and analysis["windows"]["core"]["tracking"].get("available") is True,
                    "Require complete physical/core tracking evidence")
            derived, raw = extract_derived(folder, name+"_"+cohort)
            arrays.update(derived)
            analysis["posture_diagnostics"] = posture_diagnostics(raw)
            case[cohort] = analysis
        for field in ("evaluator_source_sha256", "drift_module_sha256", "contact_slip_helper_sha256"):
            require(reports["baseline"].get(field) == reports["candidate"].get(field)
                    and isinstance(reports["baseline"].get(field), str),
                    "Baseline/candidate frozen evaluator source differs: "+field)
        comparison = compare_candidate(case["candidate"], case["baseline"], beta_extra_rmse_deg=5.)
        case["comparison"] = comparison
        cases[name] = case
        original, measured = case["baseline"]["windows"]["core"], case["candidate"]["windows"]["core"]
        for cohort, item, weight in (("baseline", original, -.1), ("candidate", measured, training["action_rate_weight"])):
            spectrum = item["body"]["angular_xy_spectrum"]
            tracking = item["tracking"]
            posture = case[cohort]["posture_diagnostics"]["full"]
            rows.append({"mu": mu, "cohort": cohort, "action_rate_weight": weight,
                "leg_action_delta_rms": item["action"]["leg_delta_rms"],
                "body_world_xy_hf_rms_rad_s": spectrum["high_frequency_rms"],
                "hf_rms_reduction_percent": _reduction(spectrum["high_frequency_rms"], original["body"]["angular_xy_spectrum"]["high_frequency_rms"]),
                "leg_action_delta_reduction_percent": _reduction(item["action"]["leg_delta_rms"], original["action"]["leg_delta_rms"]),
                "xy_rmse_m": tracking["xy_rmse_m"], "xy_peak_m": tracking["xy_peak_m"],
                "speed_retention": tracking["speed_retention"], "beta_rmse_deg": tracking["beta_rmse_deg"],
                "full_max_tilt_deg": posture["max_tilt_deg"], "full_fraction_tilt_over_60_deg": posture["fraction_tilt_over_60_deg"],
                "screen_status": "baseline_reference" if cohort == "baseline" else comparison["status"]})
    passed = all(item["comparison"]["passes"] for item in cases.values())
    summary = {"schema_version": 1, "kind": "recorded_physx_ppo_smoothness_publication",
        "training_engine": "PhysX", "evaluation_engine": "PhysX", "reference_id": "drift_drl_traj4",
        "baseline_checkpoint_sha256": historical["checkpoint_sha256"], "candidate_training": training,
        "coefficients": list(COEFFICIENTS), "table": rows, "cases": cases,
        "input_sha256": inputs, "engineering_screen_passes_both_conditions": bool(passed),
        "status": "screen_passed_requires_fullbody_visual_review" if passed else "screen_not_passed_requires_review",
        "requires_fullbody_visual_review": True, "jitter_resolved_claimed": False,
        "statistical_success_claimed": False, "causal_improvement_claimed": False,
        "new_training": False, "new_simulation": False, "new_policy_inference": False,
        "plot_settings": {"layout": [3, 2], "row_order": ["signed world omega X at 200 Hz", "hip L clipped normalized action at 50 Hz", "core world omega XY channel-mean PSD"],
            "column_mu": list(COEFFICIENTS), "colors": COLORS, "core_interval_s": [1.44, 4.],
            "high_frequency_cutoff_hz": 10., "highlighted_frequency_band_hz": [22.5, 27.5]},
        "limitations": ["One seed and one deterministic recorded episode per condition; no statistical or safety qualification.",
            "The screen uses only core 1.44--4.00s. Full-horizon posture/contact diagnostics and full-body visual review remain mandatory before interpreting execution quality.",
            "No height, flight, pose or zero-sideslip failure gate is introduced; posture diagnostics do not become screen criteria.",
            "Absolute HF RMS uses Hann-windowed, linear-detrended, channel-mean Parseval power above 10Hz, not the fraction of spectral power.",
            "Near-80-Nm sample fraction is not a measurement of speed-dependent motor saturation.",
            "Copied training declarations and matching evaluator hashes bind checkpoint identity, not independently verified local weight bytes.",
            "Shared source hashes do not alone prove identical driver/library versions or causal attribution to the changed weight.",
            "MuJoCo transfer recordings are not included or pooled into this PhysX smoothing screen."]}
    return arrays, summary


def render_diagnostic_figure(arrays, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="row", sharey="row")
    for column, mu in enumerate(COEFFICIENTS):
        for cohort in ("baseline", "candidate"):
            label = case_name(mu)+"_"+cohort
            weight = -.1 if cohort == "baseline" else summary["candidate_training"]["action_rate_weight"]
            legend = f"{cohort}: action-rate {weight:g}"
            axes[0, column].plot(arrays[label+"_time_s"], arrays[label+"_angular_velocity_w_rad_s"][:, 0],
                                 color=COLORS[cohort], lw=.7, alpha=.8, label=legend)
            axes[1, column].plot(arrays[label+"_decision_time_s"], arrays[label+"_hip_L_action"],
                                 color=COLORS[cohort], lw=.8, alpha=.8, label=legend)
            frequency, psd = arrays[label+"_core_psd_frequency_hz"], arrays[label+"_core_angular_xy_psd_rad2_s2_hz"]
            axes[2, column].semilogy(frequency[1:], np.maximum(psd[1:], 1e-15), color=COLORS[cohort], lw=1., label=legend)
        axes[0, column].set_title(f"PhysX -> PhysX | mu={mu:g}")
        for row in (0, 1):
            axes[row, column].axvspan(1.44, 4., color=".75", alpha=.2)
            axes[row, column].set_xlabel("Original physical/decision time [s]")
            axes[row, column].set_xlim(0, 6.98)
        axes[2, column].axvline(10, color="black", lw=.8, ls="--", label="10Hz cutoff")
        axes[2, column].axvspan(22.5, 27.5, color="#56B4E9", alpha=.16, label="Near 25Hz")
        axes[2, column].set_xlim(0, 100); axes[2, column].set_xlabel("Core frequency [Hz]")
        for ax in axes[:, column]:
            ax.grid(alpha=.18); ax.legend(fontsize=7, loc="upper right")
    axes[0, 0].set_ylabel("Signed world omega X [rad/s]")
    axes[1, 0].set_ylabel("Hip L normalized action [1]")
    axes[2, 0].set_ylabel("World omega XY PSD [(rad/s)^2/Hz]")
    fig.suptitle("Recorded PPO smoothing comparison | same reference, no state filtering\nNative physical signals; shaded time: core | engineering screen, not safety/statistical qualification", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output, dpi=150); plt.close(fig)


def summarize(candidate_dir, reference, baseline_root, output_dir, *,
              public_summary=DEFAULT_PUBLIC_SUMMARY, baseline_metadata=DEFAULT_BASELINE_METADATA):
    output = Path(output_dir)
    require(not output.exists(), "Preserve existing publication: choose a fresh output directory")
    arrays, summary = load_publication(candidate_dir, reference, baseline_root,
                                       public_summary=public_summary, baseline_metadata=baseline_metadata)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"derived_traces.npz", **arrays)
    with (output/"metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["table"][0]))
        writer.writeheader(); writer.writerows(summary["table"])
    render_diagnostic_figure(arrays, summary, output/"diagnostics.png")
    summary["analysis_source_sha256"] = {"summarize_ppo_smoothness.py": sha256(__file__),
        "analyze_ppo_smoothness.py": sha256(ROOT/"scripts/analyze_ppo_smoothness.py")}
    summary["artifacts"] = {name: {"file": name, "sha256": sha256(output/name)} for name in
                            ("derived_traces.npz", "metrics.csv", "diagnostics.png")}
    with (output/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-dir", type=Path, required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    summary = summarize(arguments.candidate_dir, arguments.reference, arguments.baseline_root, arguments.output_dir)
    print(json.dumps({"status": summary["status"], "requires_fullbody_visual_review": True,
                      "jitter_resolved_claimed": False}))
