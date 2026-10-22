"""Offline nominal comparison of original-grounded PPO plus torque-rate v2.

The same from-scratch -0.5 recipe keeps grounded+0.4 and adds only the16th
leg-torque-rate term at -0.05. All metrics use original recorded states, not
new simulation, inferred contact sensors, pose smoothing or a success gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_support_comparison as shared
import analyze_ppo_contact_stability as contact
import analyze_ppo_smoothness as smooth
import train_ppo_torque_control_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE, DEFAULT_REFERENCE = shared.DEFAULT_BASELINE, shared.DEFAULT_REFERENCE
TRAINER_SHA256 = "2e82aba1058af213ebbbc8639c6d967db18992cb5c545a6b823438355e7ba4b0"
CASE_ORDER = ("baseline", "candidate")


def require(condition, message):
    if not condition: raise ValueError(message)


def verify_candidate_training(folder):
    folder = Path(folder)
    names = ("manifest.json", "ablation_manifest.json", "run_status.json", trainer.RUNTIME_FILE, "runner_config.json")
    documents = {name: shared.read_json(folder/name) for name in names}
    hashes = {name: smooth.sha256(folder/name) for name in names}
    manifest, metadata, status, runtime = [documents[name] for name in names[:4]]
    require(metadata.get("kind") == trainer.KIND and metadata.get("status") == "training_completed" and
        metadata.get("smoke_only") is False and metadata.get("child_returncode") == 0 and
        metadata.get("trained_policy_claimed") is True and metadata.get("prepared_method_runtime_verified") is True,
        "Require finalized formal torque-control v2, not smoke/configured/failed grounded replacement")
    for key, name in (("final_training_manifest_sha256", "manifest.json"), ("runtime_audit_sha256", trainer.RUNTIME_FILE),
        ("run_status_sha256", "run_status.json")):
        require(metadata.get(key) == hashes[name], "Finalized torque-control artifact SHA differs: "+name)
    require(metadata.get("wrapper_sha256") == TRAINER_SHA256 == smooth.sha256(trainer.__file__) and
        metadata.get("sampler_module_sha256") == trainer.COLLECTOR_SHA256 == smooth.sha256(trainer.torque.__file__),
        "Torque-control v2 wrapper/immutable collector source differs")
    require(metadata.get("initialization") == "from_scratch_no_checkpoint_resume" and
        metadata.get("baseline_checkpoint_used_for_initialization") is False and
        metadata.get("baseline_checkpoint_sha256") == trainer.BASELINE_CHECKPOINT_SHA256 and
        metadata.get("baseline_metadata_sha256") == trainer.BASELINE_METADATA and
        metadata.get("baseline_action_rate_weight") == -.5 and metadata.get("source_default_action_rate_weight") == -.1 and
        metadata.get("num_envs") == 512 and metadata.get("iterations") == 1000 and metadata.get("seed") == 42 and
        metadata.get("expected_policy_steps") == 24000 and metadata.get("expected_environment_transitions") ==
        metadata.get("actual_environment_transitions") == 12288000,
        "Torque-control v2 must use exact from-scratch baseline recipe and12.288M budget")
    require(status.get("status") == "training_completed" and status.get("trained_policy_claimed") is True and
        status.get("environment_transitions") == 12288000 and status.get("policy_steps_completed") == 24000 and
        documents["runner_config.json"].get("num_steps_per_env") == 24, "Formal child collection budget incomplete")
    require(metadata.get("original_grounded_reward_preserved") is True and metadata.get("original_grounded_weight") == .4 and
        metadata.get("original_grounded_function") == "training.tron1_drift.grounded_reward" and
        metadata.get("new_observation_terms") is False and metadata.get("new_absolute_height_target") is False and
        metadata.get("new_zero_yaw_or_beta_target") is False, "Original grounded/contact/observation/reference contract changed")
    additional = metadata.get("additional_reward", {})
    require(additional.get("term") == trainer.TERM_NAME and additional.get("weight") == -.05 and
        additional.get("additional_reward_term") is True and
        additional.get("formula") == "mean_4_substeps(mean_6_legs((delta_applied_tau_Nm/20_Nm)^2))" and
        additional.get("delta_torque_scale_nm") == 20. and additional.get("physics_dt_s") == .005 and
        additional.get("policy_dt_s") == .02 and additional.get("wheel_cost_included") is False and
        additional.get("first_actual_substep_after_reset_cost") == 0. and additional.get("denominator_after_reset") == 4 and
        additional.get("reward_manager_dt_factor_applied_once") == .02,
        "Additional16th torque-rate term changed")
    pre_before, pre_after = metadata.get("pre_height_active_reward_terms_before", []), metadata.get("pre_height_active_reward_terms_after", [])
    final_before, final_after = metadata.get("expected_final_active_reward_terms_before", []), metadata.get("expected_final_active_reward_terms_after", [])
    require(len(pre_before) == 14 and len(pre_after) == 15 and len(final_before) == 15 and len(final_after) == 16 and
        set(pre_after)-set(pre_before) == {trainer.TERM_NAME} and
        sorted(pre_before+["height_vertical_acceleration"]) == final_before and
        sorted(pre_after+["height_vertical_acceleration"]) == final_after,
        "Pre-height14->15/post-height15->16 lifecycle declaration differs")
    trainer.verify_runtime_audit(runtime, smoke=False, wrapper_sha256=TRAINER_SHA256)
    require(runtime.get("active_reward_terms") == final_after and
        "not native PhysX drive gains" in runtime.get("gain_readback_source", ""),
        "Actual post-height RewardManager/effective software PD provenance differs")
    require(manifest.get("training_reward_overrides") == {"action_rate_l2": -.5} and
        manifest.get("additional_training_reward_terms") == {trainer.TERM_NAME: additional} and
        manifest.get("original_grounded_reward_preserved") is True and manifest.get("baseline_action_rate_weight") == -.5 and
        "contact_reward_replacement" not in manifest and "reward_replacement" not in metadata,
        "Manifest is not the original-grounded plus one additional torque penalty")
    source_pins = {"trainer_sha256": trainer.BASELINE_TRAINER_SHA256, "motion_sha256": trainer.BASELINE_MOTION_SHA256,
        "asset_sha256": trainer.BASELINE_ASSET_SHA256, "task_profile_sha256": trainer.BASELINE_PROFILE_SHA256,
        "source_training_modules_sha256": trainer.BASELINE_SOURCES}
    for key, value in source_pins.items(): require(manifest.get(key) == value, "Historical manifest identity differs: "+key)
    for key, value in {"historical_trainer_sha256": trainer.BASELINE_TRAINER_SHA256,
        "height_adapter_sha256": trainer.BASELINE_HEIGHT_SHA256, "motion_sha256": trainer.BASELINE_MOTION_SHA256,
        "asset_sha256": trainer.BASELINE_ASSET_SHA256, "task_profile_sha256": trainer.BASELINE_PROFILE_SHA256,
        "baseline_source_sha256": trainer.BASELINE_SOURCES}.items():
        require(metadata.get(key) == value, "Ablation historical source identity differs: "+key)
    require(shared._digest(metadata.get("checkpoint_sha256")), "Missing finalized checkpoint SHA")
    checkpoint = folder/"model_final.pt"; local = checkpoint.is_file()
    if local:
        require(checkpoint.stat().st_size > 0 and smooth.sha256(checkpoint) == metadata["checkpoint_sha256"], "Local checkpoint differs")
    return {"checkpoint_sha256": metadata["checkpoint_sha256"], "input_sha256": hashes,
        "checkpoint_bytes_verified_locally": local, "checkpoint_identity_scope": "Local bytes re-hashed" if local else
            "Evaluation report bound to exact owned parent's finalized metadata; private checkpoint bytes not downloaded/re-hashed locally",
        "original_grounded_reward_preserved": True, "additional_reward": additional, "native_training_audit": runtime}, manifest


def torque_diagnostics(raw):
    """Measured changes and offline training-formula cost, not logged rewards.

    Only the first actual substep after the initial episode reset has zero cost;
    inter-control torque history is retained. Four-sample windows must lie fully
    inside each reporting interval, avoiding a cross-boundary reset invention.
    """
    time = np.asarray(raw["time_s"])
    names = tuple(raw["joint_names"].tolist())
    legs = [names.index(name) for name in trainer.torque.LEG_JOINT_NAMES]
    torque_nm = contact._field(raw, "applied_joint_torque_nm")[:, legs]
    require(torque_nm.shape == (len(time), 6) and np.isfinite(torque_nm).all(), "Need actual finite six-leg applied torques")
    decisions = smooth.deduplicate_actions(time, contact._field(raw, "actions_clipped"), contact._field(raw, "action_was_applied"),
        float(raw["policy_dt_s"]), frames=contact._field(raw, "reference_frame_for_action"), counters=raw["simulation_step_counter"])
    delta = np.diff(torque_nm, axis=0)
    physical_cost = np.mean((delta/20.)**2, axis=1)
    physical_cost[0] = 0.  # Training reset rule; initial stored no-action torque is not a predecessor sample.
    starts = decisions["record_indices"]
    require(np.all(starts+3 < len(time)), "Need complete four-substep control windows")
    costs = np.array([physical_cost[start-1:start+3].mean() for start in starts])
    result = {}
    for name, interval in (("full", (float(time[0]), float(time[-1]))), ("core", (1.44, 4.))):
        lo, hi = interval; selected = (time >= lo-1e-9) & (time <= hi+1e-9)
        raw_delta = np.diff(torque_nm[selected], axis=0)
        windows = (time[starts] >= lo-1e-9) & (time[starts+3] <= hi+1e-9)
        chosen = costs[windows]
        require(len(chosen) > 0 and len(raw_delta) > 0, "Insufficient actual torque samples")
        result[name] = {"interval_s": list(interval), "leg_joint_names": list(trainer.torque.LEG_JOINT_NAMES),
            "native_adjacent_leg_torque_change_rms_nm": smooth.rms(raw_delta),
            "native_adjacent_leg_torque_change_peak_nm": float(np.abs(raw_delta).max()),
            "native_adjacent_leg_torque_change_component_rms_nm": [smooth.rms(raw_delta[:, index]) for index in range(6)],
            "native_differences": len(raw_delta), "fully_contained_control_windows": int(windows.sum()),
            "offline_training_formula_cost_mean": float(chosen.mean()), "offline_training_formula_cost_peak": float(chosen.max()),
            "training_equivalent_weighted_dt_contribution_mean": -.05*.02*float(chosen.mean()),
            "cost_definition": "mean4native(mean6legs((delta applied torque/20Nm)^2)); first actual sample after initial reset contributes0 with denominator4; history otherwise retained",
            "weighted_contribution_scope": "Offline formula evaluated on recorded physical torques, NOT logged evaluation reward, not a new success gate"}
    return result


def load_comparison(candidate_dir, candidate_training_dir, reference=DEFAULT_REFERENCE, baseline_dir=DEFAULT_BASELINE,
                    *, candidate_status=None):
    training, manifest = verify_candidate_training(candidate_training_dir)
    raw, cases, reports = {}, {}, {}
    for label, directory in (("baseline", baseline_dir), ("candidate", candidate_dir)):
        raw[label], cases[label], reports[label] = shared._load_case(directory, label, reference)
        cases[label]["grounded_reward_kind"] = "Original binary grounded_reward+0.4 (unchanged)"
        for window, values in torque_diagnostics(raw[label]).items(): cases[label]["windows"][window]["torque_control"] = values
    require(reports["candidate"]["checkpoint_sha256"] == training["checkpoint_sha256"] and
        reports["candidate"].get("runner_config_sha256") == training["input_sha256"]["runner_config.json"],
        "Candidate report checkpoint/runner differs from finalized torque-control training")
    for key in shared.PHYSICAL_KEYS:
        require(key in reports["baseline"] and key in reports["candidate"] and reports["baseline"][key] == reports["candidate"][key],
            "Matched physical/contact/evaluator contract differs: "+key)
    before, after = [reports[label]["policy_export"] for label in CASE_ORDER]
    for key in (set(before)|set(after)) - {"checkpoint_sha256", "actor_file", "export_validation_max_abs_error"}:
        require(before.get(key) == after.get(key), "Policy/body/PD/observation contract differs: "+key)
    for key in ("body_names", "wheel_body_names", "joint_names", "time_s", "simulation_step_counter"):
        require(key in raw["baseline"] and key in raw["candidate"] and np.array_equal(raw["baseline"][key], raw["candidate"][key]),
            "Native measured body/clock schema differs: "+key)
    baseline_training = Path(baseline_dir).parents[2]/"training"
    require(all(smooth.sha256(baseline_training/name) == expected for name, expected in trainer.BASELINE_METADATA.items()),
        "Fixed -0.5 baseline provenance differs")
    baseline_manifest = shared.read_json(baseline_training/"manifest.json")
    for key in shared.TRAINING_SHARED_KEYS:
        require(key in manifest and key in baseline_manifest and manifest[key] == baseline_manifest[key],
            "Training changed beyond declared additional torque penalty: "+key)
    cases["candidate"]["training_provenance"] = training
    cases["baseline"]["training_input_sha256"] = dict(trainer.BASELINE_METADATA)
    status_proof = None
    if candidate_status is not None:
        proof = shared.read_json(candidate_status).get("training", {})
        require(proof.get("checkpoint_sha256") == training["checkpoint_sha256"] and
            proof.get("ablation_manifest_sha256") == training["input_sha256"]["ablation_manifest.json"] and
            proof.get("physical_sampling_audit_sha256") == training["input_sha256"][trainer.RUNTIME_FILE],
            "Optional torque-control dispatcher proof differs")
        status_proof = {"sha256": smooth.sha256(candidate_status), "exact_owned_parent_artifacts_verified": True}
    changes = shared._changes(cases)
    for window in ("core", "full"):
        for field in ("native_adjacent_leg_torque_change_rms_nm", "offline_training_formula_cost_mean"):
            a, b = [cases[label]["windows"][window]["torque_control"][field] for label in CASE_ORDER]
            changes[window]["torque_control."+field] = {"baseline": a, "candidate": b, "difference": b-a,
                "ratio": b/a if a > 1e-12 else None}
    return raw, {"schema_version": 1, "kind": "recorded_nominal_physx_ppo_torque_control_v2_comparison",
        "status": "requires_visual_and_tracking_review", "case_order": list(CASE_ORDER), "cases": cases,
        "evaluation_mu": .7, "reference_sha256": trainer.BASELINE_MOTION_SHA256, "body": "base_Link",
        "native_physical_hz": 200., "policy_hz": 50., "core_interval_s": [1.44, 4.], "cutoff_hz_exclusive": 10.,
        "optional_dispatcher_status": status_proof, "relative_changes": changes,
        "baseline_action_rate_weight": -.5, "original_grounded_reward_preserved": True, "original_grounded_weight": .4,
        "candidate_additional_reward": {"term": trainer.TERM_NAME, "weight": -.05}, "independent_from_scratch_policies": True,
        "point_count_definition": "Actual ground-filtered points with individual normal force>5N; not geometric flight",
        "support_load_eligibility_gate": "recorded point count>0 AND aggregate upward Fz>5N; diagnostic only",
        "ground_force_definition": "Measured ground-normal forces in N, not tangential friction force or evaluation mg normalization",
        "friction_convention": "Both robot and ground coefficients set0.7; not ground-only and no effective PhysX combine coefficient inferred",
        "new_training": False, "new_simulation": False, "new_inference": False, "pose_smoothing_or_state_correction": False,
        "resampling_or_alignment": False, "contact_success_threshold_added": False, "stable_contact_claimed": False,
        "height_or_flight_failure_gate": False, "nonzero_sideslip_target_preserved": True, "requires_fullbody_visual_review": True,
        "analysis_source_sha256": {Path(__file__).name: smooth.sha256(__file__),
            "analyze_ppo_ground_support_comparison.py": smooth.sha256(shared.__file__),
            "analyze_ppo_contact_stability.py": smooth.sha256(contact.__file__), "analyze_ppo_smoothness.py": smooth.sha256(smooth.__file__)},
        "limitations": [
            "One seed/checkpoint/episode per independent policy; no causal/statistical or stable-contact guarantee. Inspect full-body video with contact and tracking.",
            "Original grounded+0.4 is preserved. Added torque cost excludes wheels and is not a height/flight or zero-beta/yaw target.",
            "Effective gains500/10 are explicit DC motor software PD, not native PhysX drive gains. Native measured applied actuator torque is in Nm.",
            "Offline torque cost follows training's reset rule only at initial episode reset; it does not invent resets at reporting/control boundaries.",
            "Torque cost and weighted-dt contribution are recomputed offline from recorded torques, not claimed as logged evaluator reward.",
            "Zero thresholded count/load eligibility is not geometric separation or failure. Weak contacts<=5N cannot be recovered; engine force aggregation is specific.",
            "HF diagnostics use original native200Hz states with linear detrend/Hann/Parseval, not a pose filter; intended yaw/sideslip and free absolute height remain.",
            "Physical horizon6.98s exceeds reference6.96s; full tracking is unavailable without extrapolation and reference-overlap is separately labeled.",
            "Checkpoint identity is metadata-bound if private weights are absent locally; remote owned parent independently verified their bytes."]}


def render_contact_plot(raw, summary, destination):
    """Torque-specific captions; no use of the ground-replacement plot label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="col")
    for row, label in enumerate(CASE_ORDER):
        time = raw[label]["time_s"]; force = contact._field(raw[label], "wheel_ground_contact_force_w_n")[:, :, 2]
        count = contact._field(raw[label], "wheel_ground_contact_count"); angular = contact._field(raw[label], "root_angular_velocity_w_rad_s")
        for index, side, color in ((0, "Left", "#D55E00"), (1, "Right", "#0072B2")):
            axes[row, 0].plot(time, force[:, index], lw=.6, color=color, label=side+" Fz")
            axes[row, 1].step(time, ((count[:, index]>0)&(force[:, index]>5.)).astype(float)+1.3*index,
                where="post", lw=.6, color=color, label=side+" count>0 &Fz>5N")
        axes[row, 2].plot(time, angular[:, 0], lw=.6, label="World wx")
        axes[row, 2].plot(time, angular[:, 1], lw=.6, label="World wy")
        axes[row, 0].set_ylabel(("Baseline -0.5" if label == "baseline" else "Add torque-rate -0.05")+"\nGround Fz [N]")
        axes[row, 2].set_ylabel("Raw angular velocity [rad/s]")
        for axis in axes[row]:
            axis.axvspan(1.44, 4., color="#888888", alpha=.1); axis.grid(alpha=.2); axis.legend(fontsize=7)
            axis.set_xlim(0., 6.98)
    for axis, title in zip(axes[0], ("Measured ground normal load", "Support-load eligibility (not flight)", "Recorded body angular XY")): axis.set_title(title)
    for axis in axes[-1]: axis.set_xlabel("Native physical time [s]")
    fig.suptitle("Nominal PhysX mu=0.7 | original grounded+0.4 preserved | native200Hz, no smoothing")
    fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-dir", "candidate-training-dir", "output-dir"): result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--candidate-status", type=Path)
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing evidence: output directory must be fresh")
    raw, summary = load_comparison(args.candidate_dir, args.candidate_training_dir, args.reference, args.baseline_dir,
        candidate_status=args.candidate_status)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"measured_states.npz", **shared.measured_state_arrays(raw))
    render_contact_plot(raw, summary, output/"contact_comparison.png")
    summary["artifacts"] = {name: {"file": name, "sha256": smooth.sha256(output/name)} for name in ("measured_states.npz", "contact_comparison.png")}
    with (output/"summary.json").open("x") as stream: json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    result = run(parser().parse_args()); print(json.dumps({"status": result["status"], "stable_contact_claimed": False}))
