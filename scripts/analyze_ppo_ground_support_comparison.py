"""Offline, provenance-bound nominal contact-reward PPO comparison.

Only recorded native physical states are analyzed. Thresholded force-bearing
contact is not geometric flight, and this report does not invent a contact
success threshold. Training, simulation, inference and pose smoothing are absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_body_jitter as body_jitter
import analyze_ppo_contact_stability as contact
import analyze_ppo_smoothness as smooth
from train_ppo_ground_support import BASELINE_CHECKPOINT_SHA256, BASELINE_METADATA, KIND, RUNTIME_FILE
from train_ppo_smoothness import (
    BASELINE_ASSET_SHA256, BASELINE_HEIGHT_SHA256, BASELINE_MOTION_SHA256,
    BASELINE_PROFILE_SHA256, BASELINE_SOURCES, BASELINE_TRAINER_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = contact.DEFAULT_05
DEFAULT_REFERENCE = contact.DEFAULT_REFERENCE
BASELINE_TRACE_SHA256 = contact.CASES["action_rate_0p5"]["trace"]
CASE_ORDER = ("baseline", "candidate")
PHYSICAL_KEYS = (
    "contact_material_audit", "legacy_joint_friction_audit", "evaluator_source_sha256",
    "drift_module_sha256", "contact_slip_helper_sha256", "asset_root_file_sha256",
    "physics_recording", "task_profile", "evaluation_task_profile", "domain_randomization_audit",
    "contact_slip_measurement", "terrain", "terrain_runtime_audit", "terrain_file_sha256",
)
TRAINING_SHARED_KEYS = (
    "algorithm", "reference_id", "reference_spec", "source_training_modules_sha256",
    "trainer_sha256", "motion_sha256", "asset_sha256", "task_profile_sha256", "task_profile",
    "contract", "seed", "num_envs", "iterations", "num_steps_per_env",
    "policy_steps_requested", "environment_transitions_requested", "setup_smoke_control_steps", "recipe",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_candidate_training(folder):
    """Read finalized parent audit without running or modifying the finalizer.

    Model bytes remain private/remote when absent locally. In that case identity
    is explicitly metadata-bound, not claimed to be a local checkpoint re-hash.
    """
    folder = Path(folder)
    names = ("manifest.json", "ablation_manifest.json", "run_status.json", RUNTIME_FILE, "runner_config.json")
    documents = {name: read_json(folder/name) for name in names}
    hashes = {name: smooth.sha256(folder/name) for name in names}
    manifest, ablation, status, audit = [documents[name] for name in names[:4]]
    require(ablation.get("kind") == KIND and ablation.get("status") == "training_completed" and
            ablation.get("smoke_only") is False and ablation.get("child_returncode") == 0 and
            ablation.get("trained_policy_claimed") is True and ablation.get("prepared_method_runtime_verified") is True,
            "Candidate must have finalized full training, not configured/smoke/partial metadata")
    for field, name in (("final_training_manifest_sha256", "manifest.json"),
                        ("runtime_audit_sha256", RUNTIME_FILE), ("run_status_sha256", "run_status.json")):
        require(ablation.get(field) == hashes[name], "Finalized candidate hash binding differs: "+name)
    require(ablation.get("initialization") == "from_scratch_no_checkpoint_resume" and
            ablation.get("num_envs") == 512 and ablation.get("iterations") == 1000 and
            ablation.get("seed") == 42 and ablation.get("expected_policy_steps") == 24000 and
            ablation.get("expected_environment_transitions") == ablation.get("actual_environment_transitions") == 12288000,
            "Candidate formal training scope changed")
    require(status.get("status") == "training_completed" and status.get("trained_policy_claimed") is True and
            status.get("environment_transitions") == 12288000 and status.get("policy_steps_completed") == 24000,
            "Candidate child did not complete full collection budget")
    require(ablation.get("baseline_checkpoint_sha256") == BASELINE_CHECKPOINT_SHA256 and
            ablation.get("baseline_metadata_sha256") == BASELINE_METADATA and
            ablation.get("baseline_action_rate_weight") == -.5 and
            ablation.get("source_default_action_rate_weight") == -.1,
            "Candidate was not the declared minus-0.5 same-recipe comparison")
    expected_sources = {
        "historical_trainer_sha256": BASELINE_TRAINER_SHA256, "height_adapter_sha256": BASELINE_HEIGHT_SHA256,
        "task_profile_sha256": BASELINE_PROFILE_SHA256, "motion_sha256": BASELINE_MOTION_SHA256,
        "asset_sha256": BASELINE_ASSET_SHA256, "baseline_source_sha256": BASELINE_SOURCES,
    }
    require(all(ablation.get(key) == value for key, value in expected_sources.items()),
            "Candidate historical task/source identity differs")
    replacement = ablation.get("reward_replacement", {})
    require(replacement.get("term") == "drift_grounded" and replacement.get("weight") == .4 and
            replacement.get("scale_fraction") == .20 and replacement.get("additional_reward_term") is False and
            replacement.get("raw_reward_bounds") == [0., 1.] and
            replacement.get("reward_manager_dt_factor_applied_once") == .02 and
            replacement.get("new_formula") == "1-mean_4_substeps(mean_2_wheels(exp(-clamp(ground_Fz_N,min=0)/(.20*mg_N))))",
            "Candidate grounded-term replacement contract differs")
    require(manifest.get("training_reward_overrides") == {"action_rate_l2": -.5, "drift_grounded": replacement} and
            manifest.get("contact_reward_replacement") == replacement and
            manifest.get("comparison_baseline_action_rate_weight") == -.5 and
            manifest.get("historical_profile_hash_does_not_include_explicit_reward_override") is True,
            "Final training manifest does not record the actual explicit replacement")
    for key, expected in (("trainer_sha256", BASELINE_TRAINER_SHA256), ("motion_sha256", BASELINE_MOTION_SHA256),
                          ("asset_sha256", BASELINE_ASSET_SHA256), ("task_profile_sha256", BASELINE_PROFILE_SHA256),
                          ("source_training_modules_sha256", BASELINE_SOURCES)):
        require(manifest.get(key) == expected, "Candidate manifest source identity differs: "+key)
    require(_digest(ablation.get("checkpoint_sha256")) and
            _digest(ablation.get("wrapper_sha256")) and _digest(ablation.get("sampler_module_sha256")) and
            audit.get("module_sha256") == ablation["sampler_module_sha256"] and
            audit.get("wrapper_sha256") == ablation["wrapper_sha256"], "Candidate runtime/source hash binding differs")
    require(audit.get("steps_started") == audit.get("steps_completed") == audit.get("unique_cost_windows") == 24000 and
            audit.get("physics_samples_total") == 96000 and audit.get("steps_failed") == 0 and
            audit.get("samples_last_window") == 4 and audit.get("num_envs") == 512 and
            audit.get("closed") is True and audit.get("device_health_checked") is True and
            audit.get("device_health_finite") is True and audit.get("cpu_fixture_only") is False,
            "Candidate native-force audit is incomplete or unhealthy")
    require(audit.get("physics_dt_s") == .005 and audit.get("policy_dt_s") == .02 and audit.get("decimation") == 4 and
            audit.get("scale_fraction") == .20 and audit.get("external_reward_weight") == .4 and
            audit.get("raw_reward_bounds") == [0., 1.] and audit.get("mass_gravity_reads") == 1,
            "Candidate native-force clock/reward normalization differs")
    require(audit.get("sensor_names") == ["wheel_L_ground", "wheel_R_ground"] and
            audit.get("wheel_body_names") == ["wheel_L_Link", "wheel_R_Link"] and
            audit.get("force_matrix_shape_each_sensor") == [512, 1, 1, 3] and
            audit.get("ground_filter", {}).get("path") == "/World/ground/terrain/GroundPlane/CollisionPlane" and
            audit.get("ground_filter", {}).get("collision_api_verified") is True and
            audit.get("gain_readback") == {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "Candidate measured sensor/filter/PD audit differs")
    require(audit.get("clock_helper_sha256") == smooth.sha256(ROOT/"scripts/ppo_leg_torque_smoothness.py"),
            "Candidate native clock dependency differs")
    for key in ("mass_kg_range", "mg_N_range"):
        value = np.asarray(audit.get(key), dtype=float)
        require(value.shape == (2,) and np.isfinite(value).all() and (value > 0).all() and value[0] <= value[1],
                "Invalid candidate runtime mass/gravity: "+key)
    gravity = np.asarray(audit.get("gravity_m_s2"), dtype=float)
    positive = np.asarray(audit.get("positive_normal_force_samples_left_right"), dtype=float)
    require(gravity.shape == (3,) and np.isfinite(gravity).all() and np.array_equal(gravity[:2], [0., 0.]) and gravity[2] < 0 and
            positive.shape == (2,) and np.isfinite(positive).all() and (positive > 0).all() and
            audit.get("negative_normal_force_samples") == 0, "Candidate upward ground force was not runtime-verified")
    checkpoint = folder/"model_final.pt"
    local_bytes = checkpoint.is_file()
    if local_bytes:
        require(checkpoint.stat().st_size > 0 and smooth.sha256(checkpoint) == ablation["checkpoint_sha256"],
                "Local candidate checkpoint SHA differs from finalized training")
    return {"checkpoint_sha256": ablation["checkpoint_sha256"], "input_sha256": hashes,
        "checkpoint_bytes_verified_locally": local_bytes,
        "checkpoint_identity_scope": "Local bytes re-hashed" if local_bytes else
            "Evaluation report bound to parent-finalized training metadata; private checkpoint bytes not downloaded/re-hashed locally",
        "reward_replacement": replacement, "native_training_audit": audit}, manifest


def _load_case(folder, label, reference):
    folder = Path(folder)
    report = read_json(folder/"report.json")
    require(report.get("status") == "evaluated" and report.get("simulator") == "IsaacLab/PhysX" and
            report.get("full_horizon_diagnostic") is True and report.get("terminal_states_captured_before_auto_reset") is True and
            report.get("domain_randomization_enabled") is False and report.get("observation_noise_enabled") is False and
            report.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7 and
            report.get("motion_file_sha256") == BASELINE_MOTION_SHA256 == smooth.sha256(reference),
            "Require completed nominal mu=.7 PhysX full-horizon/reference and deterministic DR-off evaluation")
    trace_sha = smooth.sha256(folder/"physics_trajectory.npz")
    require(trace_sha == report.get("physics_trajectory_sha256"), "Evaluation report-to-native-trajectory hash differs")
    if label == "baseline":
        require(report.get("checkpoint_sha256") == BASELINE_CHECKPOINT_SHA256 and trace_sha == BASELINE_TRACE_SHA256,
                "Wrong fixed minus-0.5 baseline checkpoint/trajectory")
    with np.load(folder/"physics_trajectory.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    time = np.asarray(raw["time_s"])
    valid = contact._field(raw, "valid_mask")
    require(time.shape == (1397,) and np.array_equal(time, np.arange(1397)*.005) and
            valid.shape == time.shape and valid.dtype.kind == "b" and valid.all() and
            float(raw["physics_dt_s"]) == .005 and float(raw["policy_dt_s"]) == .02,
            "Require full original native 200Hz first episode, no truncation/reset splice/resampling")
    require(tuple(raw["wheel_body_names"].tolist()) == ("wheel_L_Link", "wheel_R_Link") and
            report.get("contact_slip_measurement", {}).get("active_point_normal_force_threshold_n") == 5.,
            "Measured wheel ordering/point threshold differs")
    contract = report.get("policy_export", {})
    require(contract.get("actor_dim") == 51 and contract.get("critic_dim") == 120 and contract.get("leg_kp") == 500. and
            contract.get("leg_kd") == 10. and contract.get("physics_dt") == .005 and contract.get("decimation") == 4 and
            contract.get("checkpoint_sha256") == report.get("checkpoint_sha256"), "Evaluation policy/PD/clock binding differs")
    require(report.get("contact_material_audit", {}).get("leg_pd_gain_readback") ==
            {"kp_range": [500., 500.], "kd_range": [10., 10.]}, "Actual evaluation PD readback differs")
    smoothness = smooth.analyze_trajectory(folder/"physics_trajectory.npz", reference=reference, report=folder/"report.json")
    require(smoothness["control_decisions"] == 349, "Require original complete 349 control decisions")
    counts, forces = contact._field(raw, "wheel_ground_contact_count"), contact._field(raw, "wheel_ground_contact_force_w_n")
    position = contact._field(raw, "root_position_m")
    quaternion = contact._field(raw, "root_quaternion_wxyz")
    velocity = contact._field(raw, "root_linear_velocity_w_m_s")
    angular = contact._field(raw, "root_angular_velocity_w_rad_s")
    windows = contact.analyze_signals(time, counts, forces, velocity, angular,
        nonwheel_counts=contact._field(raw, "nonwheel_ground_contact_count") if "nonwheel_ground_contact_count" in raw else None)
    tilt = np.rad2deg(np.arccos(np.clip(1.-2.*(quaternion[:, 1]**2+quaternion[:, 2]**2), -1., 1.)))
    signals = {"angular_world_xy": angular[:, :2], "angular_world_z": angular[:, 2],
        "com_linear_velocity_world_xy": velocity[:, :2], "com_linear_velocity_world_z": velocity[:, 2],
        "root_link_origin_position_world_xy": position[:, :2], "root_link_origin_position_world_z": position[:, 2],
        "angular_body_xy": smooth._world_to_body(quaternion, angular)[:, :2]}
    for name in ("full", "core"):
        lo, hi = windows[name]["interval_s"]
        mask = (time >= lo-1e-9) & (time <= hi+1e-9)
        for key in ("tracking", "body", "action", "legs", "torque"):
            windows[name][key] = smoothness["windows"][name][key]
        windows[name]["tilt"] = {"mean_deg": float(tilt[mask].mean()), "rms_deg": smooth.rms(tilt[mask]),
            "peak_deg": float(tilt[mask].max()), "definition": "Angle of recorded base_Link +Z to world +Z; diagnostic, not a new success/height limit"}
        windows[name]["body_diagnostics"] = {key: {**body_jitter.SIGNALS[key],
            "raw_signal_rms": smooth.rms(value[mask]), "spectrum": smooth.spectral_metrics(value[mask], .005, 10.)}
            for key, value in signals.items()}
    ref = smooth._load_reference(reference)
    overlap = (max(float(time[0]), float(ref["time_s"][0])), min(float(time[-1]), float(ref["time_s"][-1])))
    overlap_tracking = {"interval_s": list(overlap), **smooth._tracking(time, position, velocity, quaternion, ref, overlap),
        "scope": "Explicit reference-overlap diagnostic; not full-horizon tracking and no endpoint hold/extrapolation"}
    case = {"label": label, "checkpoint_sha256": report["checkpoint_sha256"],
        "input_sha256": {"report.json": smooth.sha256(folder/"report.json"), "physics_trajectory.npz": trace_sha},
        "action_rate_weight": -.5, "native_physical_samples": len(time), "control_decisions": 349,
        "reported_contact_metadata_sample_dt_s": report["contact_slip_measurement"].get("sample_dt_s"),
        "actual_native_physical_sample_dt_s": .005, "windows": {name: windows[name] for name in ("full", "core")},
        "reference_overlap_tracking": overlap_tracking,
        "evaluation_provenance": {key: report[key] for key in PHYSICAL_KEYS if key in report}}
    return raw, case, report


def _changes(cases):
    changes = {}
    for window in ("full", "core"):
        before, after = (cases[label]["windows"][window] for label in CASE_ORDER)
        metrics = {}
        for kind in ("measured_force_bearing_contact", "load_eligibility"):
            for side in ("left", "right", "both", "none"):
                for field in ("sample_duty_fraction", "time_duty_fraction_zoh", "state_switch_count", "state_switches_per_s"):
                    a, b = before["contact"][kind][side][field], after["contact"][kind][side][field]
                    metrics[f"{kind}.{side}.{field}"] = {"baseline": a, "candidate": b, "difference": b-a}
        for side in ("left", "right"):
            for field in ("step_rms_n", "step_abs_peak_n"):
                a, b = before["force_z"][side][field], after["force_z"][side][field]
                metrics[f"force_z.{side}.{field}"] = {"baseline": a, "candidate": b, "difference": b-a}
        for key in body_jitter.SIGNALS:
            a, b = (row["body_diagnostics"][key]["spectrum"]["high_frequency_rms"] for row in (before, after))
            metrics["body_hf_rms."+key] = {"baseline": a, "candidate": b, "difference": b-a,
                **body_jitter._relative_change(b, a)}
        if before["tracking"].get("available") and after["tracking"].get("available"):
            for field in ("xy_rmse_m", "xy_peak_m", "speed_retention", "beta_rmse_deg"):
                a, b = before["tracking"].get(field), after["tracking"].get(field)
                metrics["tracking."+field] = {"baseline": a, "candidate": b,
                    "difference": b-a if a is not None and b is not None else None}
        for field in ("rms_deg", "peak_deg"):
            a, b = before["tilt"][field], after["tilt"][field]
            metrics["tilt."+field] = {"baseline": a, "candidate": b, "difference": b-a}
        changes[window] = metrics
    return changes


def load_comparison(candidate_dir, candidate_training_dir, reference=DEFAULT_REFERENCE,
                    baseline_dir=DEFAULT_BASELINE, *, candidate_status=None):
    """Return original raw archives by label and audited descriptive summary.

    Public for the FK renderer: arrays retain their original singleton env axes,
    clocks, body/joint ordering and values. No fake legacy report is constructed.
    """
    training, candidate_manifest = verify_candidate_training(candidate_training_dir)
    raw, cases, reports = {}, {}, {}
    for label, folder in (("baseline", baseline_dir), ("candidate", candidate_dir)):
        raw[label], cases[label], reports[label] = _load_case(folder, label, reference)
    require(reports["candidate"]["checkpoint_sha256"] == training["checkpoint_sha256"],
            "Candidate evaluation checkpoint is not the finalized training checkpoint")
    require(reports["candidate"].get("runner_config_sha256") == training["input_sha256"]["runner_config.json"],
            "Candidate evaluation runner config differs from finalized training bundle")
    for key in PHYSICAL_KEYS:
        require(key in reports["baseline"] and key in reports["candidate"] and
                reports["baseline"][key] == reports["candidate"][key], "Shared physical/contact/evaluator setting differs: "+key)
    contract_before, contract_after = (reports[label]["policy_export"] for label in CASE_ORDER)
    contract_keys = set(contract_before) | set(contract_after)
    for key in contract_keys-{"checkpoint_sha256", "actor_file", "export_validation_max_abs_error"}:
        require(contract_before.get(key) == contract_after.get(key), "Shared policy observation/body/PD contract differs: "+key)
    for key in ("body_names", "wheel_body_names", "joint_names", "time_s", "simulation_step_counter"):
        require(key in raw["baseline"] and key in raw["candidate"] and
                np.array_equal(raw["baseline"][key], raw["candidate"][key]), "Original matched body/clock schema differs: "+key)
    baseline_training_dir = Path(baseline_dir).parents[2]/"training"
    require(all(smooth.sha256(baseline_training_dir/name) == expected for name, expected in BASELINE_METADATA.items()),
            "Fixed baseline training provenance differs")
    baseline_manifest = read_json(baseline_training_dir/"manifest.json")
    for key in TRAINING_SHARED_KEYS:
        require(key in baseline_manifest and key in candidate_manifest and baseline_manifest[key] == candidate_manifest[key],
                "Training recipe changed beyond declared grounded-term replacement: "+key)
    cases["candidate"]["training_provenance"] = training
    cases["baseline"]["training_input_sha256"] = dict(BASELINE_METADATA)
    optional_status = None
    if candidate_status is not None:
        status = read_json(candidate_status)
        declared = status.get("training", {})
        require(declared.get("checkpoint_sha256") == training["checkpoint_sha256"], "Optional dispatcher checkpoint binding differs")
        for key, name in (("ablation_manifest_sha256", "ablation_manifest.json"), ("physical_sampling_audit_sha256", RUNTIME_FILE)):
            require(declared.get(key) == training["input_sha256"][name], "Optional dispatcher audit binding differs: "+key)
        optional_status = {"sha256": smooth.sha256(candidate_status), "checkpoint_binding_verified": True}
    summary = {"schema_version": 1, "kind": "recorded_nominal_physx_ppo_ground_support_reward_comparison",
        "status": "requires_visual_and_tracking_review", "case_order": list(CASE_ORDER), "cases": cases,
        "evaluation_mu": .7, "reference_sha256": BASELINE_MOTION_SHA256, "body": "base_Link",
        "friction_convention": "Legacy evaluation sets both robot and ground static/dynamic material coefficients to0.7; not a ground-only perturbation. Combine mode/effective pair coefficient is not inferred from this archived audit.",
        "native_physical_hz": 200., "policy_hz": 50., "core_interval_s": [1.44, 4.],
        "cutoff_hz_exclusive": 10., "optional_dispatcher_status": optional_status, "relative_changes": _changes(cases),
        "point_count_definition": "Actual ground-filtered contact points with individual normal force strictly >5 N; not all geometric contacts",
        "measured_contact_gate": "recorded point count > 0", "support_load_eligibility_gate": "recorded point count > 0 AND aggregate upward ground Fz > 5 N",
        "ground_force_definition": "Sum of measured ground-normal force vectors only, not tangential friction force; Fz is world upward component",
        "no_contact_is_geometric_flight": False, "contact_success_threshold_added": False,
        "height_or_flight_failure_gate": False, "absolute_height_target_added": False,
        "nonzero_sideslip_target_preserved": True, "new_training": False, "new_simulation": False,
        "new_inference": False, "pose_smoothing_or_state_correction": False, "resampling_or_alignment": False,
        "statistical_or_causal_success_claimed": False, "stable_contact_claimed": False,
        "requires_fullbody_visual_review": True,
        "analysis_source_sha256": {name: smooth.sha256(ROOT/"scripts"/name) for name in
            (Path(__file__).name, "analyze_ppo_contact_stability.py", "analyze_ppo_smoothness.py", "analyze_ppo_body_jitter.py")},
        "limitations": [
            "Zero thresholded count/load eligibility does not establish geometric separation, flight or task failure. Weak contacts <=5 N cannot be reconstructed.",
            "Ground sensors provide training reward only; unchanged actor/critic layout is checked, and no new sensor observation is assumed.",
            "Every contact statistic uses the original 5ms native clock. Legacy contact metadata may still say20ms; no clock is replaced by that label.",
            "Duty includes both native-sample and zero-order-held time fractions. Boundary missing-contact runs are censored observed durations.",
            "Force jumps are differences between adjacent 5ms recorded normal-force samples, in N, not a derivative or measured acceleration.",
            "Physical full horizon ends6.98s but reference ends6.96s. Full tracking is unavailable without extrapolation; explicit reference-overlap tracking is separate.",
            "Training mg cache is not an evaluation mass readback; evaluation force remains N, without inferred mass normalization.",
            "HF RMS uses linear detrend, Hann window and channel-mean one-sided Parseval normalization, frequencies strictly above10Hz; not a pose filter.",
            "Each body component retains its own frame/units. Intended yaw, nonzero sideslip and free absolute height are not new rejection criteria.",
            "One seed/checkpoint/episode per policy is descriptive only. Contact, tracking and full-body video must be reviewed together; no automatic pass threshold is invented."]}
    return raw, summary


def measured_state_arrays(raw_by_label):
    """Export only unchanged recorded physical signals; no policy/weights/assets."""
    fields = ("time_s", "valid_mask", "simulation_step_counter", "physics_dt_s", "policy_dt_s", "body_names",
        "wheel_body_names", "joint_names", "root_position_m", "root_quaternion_wxyz", "root_linear_velocity_w_m_s",
        "root_angular_velocity_w_rad_s", "joint_position_rad", "joint_velocity_rad_s", "applied_joint_torque_nm",
        "wheel_ground_contact_count", "wheel_ground_contact_force_w_n", "nonwheel_ground_contact_count")
    return {label+"_"+key: value for label, raw in raw_by_label.items() for key, value in raw.items() if key in fields}


def render_contact_plot(raw_by_label, summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, sharey="col")
    for row, label in enumerate(CASE_ORDER):
        raw = raw_by_label[label]; time = raw["time_s"]
        force = contact._field(raw, "wheel_ground_contact_force_w_n")[:, :, 2]
        counts = contact._field(raw, "wheel_ground_contact_count")
        angular = contact._field(raw, "root_angular_velocity_w_rad_s")
        for side, color, index in (("Left", "#D55E00", 0), ("Right", "#0072B2", 1)):
            axes[row, 0].plot(time, force[:, index], color=color, lw=.6, label=side+" Fz")
            axes[row, 1].step(time, (counts[:, index] > 0).astype(float)+1.3*index, where="post", color=color, lw=.6,
                            label=side+" count>0")
            gate = (counts[:, index] > 0) & (force[:, index] > 5.)
            axes[row, 2].step(time, gate.astype(float)+1.3*index, where="post", color=color, lw=.6,
                            label=side+" count>0 &Fz>5N")
        axes[row, 3].plot(time, angular[:, 0], lw=.6, label="World wx")
        axes[row, 3].plot(time, angular[:, 1], lw=.6, label="World wy")
        axes[row, 0].set_ylabel(("Baseline -0.5" if label == "baseline" else "Ground-support PPO")+"\nGround Fz [N]")
        axes[row, 3].set_ylabel("Recorded angular velocity [rad/s]")
        for axis in axes[row]:
            axis.axvspan(1.44, 4., color="#888888", alpha=.1)
            axis.grid(alpha=.2); axis.legend(fontsize=7, loc="upper right")
            axis.set_xlim(0., 6.98)
    for axis, title in zip(axes[0], ("Measured ground normal load", "Thresholded point count gate", "Support-load eligibility", "Raw body angular XY")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Native physical time [s]")
    fig.suptitle("Nominal PhysX mu=0.7 | native200Hz, no smoothing | zero gate is NOT geometric flight", fontsize=12)
    fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-dir", type=Path, required=True, help="Nominal Isaac directory with report and native trajectory")
    result.add_argument("--candidate-training-dir", type=Path, required=True, help="Finalized full-training metadata/audit directory")
    result.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--candidate-status", type=Path)
    result.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; never overwrite evidence")
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing comparison evidence: output directory must be fresh")
    raw, summary = load_comparison(args.candidate_dir, args.candidate_training_dir, args.reference,
        args.baseline_dir, candidate_status=getattr(args, "candidate_status", None))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"measured_states.npz", **measured_state_arrays(raw))
    render_contact_plot(raw, summary, output/"contact_comparison.png")
    summary["artifacts"] = {name: {"file": name, "sha256": smooth.sha256(output/name)} for name in
        ("measured_states.npz", "contact_comparison.png")}
    with (output/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "stable_contact_claimed": False, "new_simulation": False}))
