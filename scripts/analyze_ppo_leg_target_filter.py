"""Offline same-actor comparison of bypass versus processed leg-target LPF.

The physical states are never filtered by this analyzer. The deployment filter
acts on six software position targets at50Hz; unchanged requested actor actions
are NOT substituted for its actual applied targets in these diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_support_comparison as shared
import analyze_ppo_contact_stability as contact
import analyze_ppo_smoothness as smooth
import run_ppo_leg_target_filter as parent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT/"outputs/ppo_leg_target_filter_20261021_v1"
DEFAULT_REFERENCE = shared.DEFAULT_REFERENCE
CASE_ORDER = ("baseline", "candidate")
FOLDERS = {"baseline": "alpha1p0", "candidate": "alpha0p5"}
ALPHAS = {"baseline": 1., "candidate": .5}
AUTHORIZED_GPU_INDICES = (1, 3, 4, 5)
LEG_JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint")


def require(condition, message):
    if not condition: raise ValueError(message)


def _archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def target_control_diagnostics(raw, control, physics):
    """Software command changes and actual recorded physical torque changes.

    Native float timestamps are checked and preserved. Target-window selection
    uses the exact recorded control counters' nominal5ms schedule, not an
    accumulated float-clock boundary. No state/target is resampled or corrected.
    """
    time = np.asarray(raw["time_s"])
    counter = np.asarray(control.get("sim_step_counter"))
    require(counter.shape == (349,) and counter.dtype.kind in "iu" and (counter >= 0).all(),
        "Require original349 integer control physics counters")
    relative_counter = counter-counter[0]
    require(np.array_equal(relative_counter, np.arange(349)*4), "Control physics counters must advance exactly4 per decision")
    decision_time = relative_counter*.005
    native_time = np.asarray(control["time_s"], dtype=float)
    native_relative_time = native_time-native_time[0]
    requested, filtered = (np.asarray(control[name], dtype=float) for name in ("requested_targets_rad", "filtered_targets_rad"))
    require(native_time.shape == (349,) and np.isfinite(native_time).all() and requested.shape == filtered.shape == (349, 6) and
        np.allclose(native_relative_time, decision_time, rtol=0, atol=2e-6) and
        np.isfinite(requested).all() and np.isfinite(filtered).all(), "Need original349 target-control samples")
    applied = np.asarray(physics["applied_targets_rad"])
    indices = np.asarray(physics["control_index"])
    require(applied.shape == (1396, 6) and np.array_equal(indices, np.repeat(np.arange(349), 4)) and
        np.array_equal(applied, np.asarray(control["filtered_targets_rad"])[indices]),
        "Applied target sidecar must hold each actual filtered target for four native substeps")
    names = tuple(raw["joint_names"].tolist()); legs = [names.index(name) for name in LEG_JOINT_NAMES]
    torque = contact._field(raw, "applied_joint_torque_nm")[:, legs]
    require(torque.shape == (1397, 6) and np.isfinite(torque).all(), "Need recorded actual six-leg applied torque in Nm")
    result = {}
    for window, interval in (("full", (float(time[0]), float(time[-1]))), ("core", (1.44, 4.))):
        lo, hi = interval
        cmask = (decision_time >= lo-1e-9) & (decision_time <= hi+1e-9)
        pmask = (time >= lo-1e-9) & (time <= hi+1e-9)
        d_requested, d_filtered, d_torque = (np.diff(values[mask], axis=0) for values, mask in
            ((requested, cmask), (filtered, cmask), (torque, pmask)))
        require(min(len(d_requested), len(d_torque)) > 0, "Insufficient native control/physics samples")
        result[window] = {"interval_s": list(interval), "control_samples": int(cmask.sum()),
            "target_change_pairs": len(d_requested), "physical_samples": int(pmask.sum()), "torque_change_pairs": len(d_torque),
            "requested_target_step_rms_rad": smooth.rms(d_requested),
            "filtered_applied_target_step_rms_rad": smooth.rms(d_filtered),
            "requested_target_step_peak_rad": float(np.abs(d_requested).max()),
            "filtered_applied_target_step_peak_rad": float(np.abs(d_filtered).max()),
            "physical_leg_applied_torque_step_rms_nm": smooth.rms(d_torque),
            "physical_leg_applied_torque_step_peak_nm": float(np.abs(d_torque).max()),
            "physical_leg_applied_torque_rms_nm": smooth.rms(torque[pmask]),
            "target_definition": "Original process_actions output after scale/offset versus actual externally filtered software position target; independently read back at all4 pre-step applies",
            "torque_definition": "robot.data.applied_torque from post-step physical recording, six named legs only, Nm; adjacent5ms differences, not inferred target torque",
            "target_clock": "Counter-derived nominal schedule: (recorded sim_step_counter-control_counter0)*.005s; only for selecting50Hz target windows, NOT measured native float time or post-step200Hz pose time",
            "native_control_float_clock_max_abs_deviation_from_schedule_s": float(np.abs(native_relative_time-decision_time).max()),
            "native_control_float_clock_verification_tolerance_s": 2e-6,
            "native_control_float_clock_preserved": True,
            "no_target_is_inferred_from_actor_action": True,
            "reset_or_window_splice_invented": False}
    return result


def load_comparison(input_root, reference=DEFAULT_REFERENCE):
    """Return untouched original raw archives and audited same-actor metrics."""
    root = Path(input_root)
    declared_indices = {}
    for label in CASE_ORDER:
        report = shared.read_json(root/FOLDERS[label]/"report.json")
        index = report.get("leg_target_filter_diagnostic", {}).get("physical_gpu_index_requested")
        require(type(index) is int and index in AUTHORIZED_GPU_INDICES,
            "Both reports must explicitly declare an authorized physical GPU index; no implicit GPU1 fallback")
        declared_indices[label] = index
    require(len(set(declared_indices.values())) == 1, "Matched filter reports must declare the SAME authorized physical GPU")
    physical_gpu_index = declared_indices["baseline"]
    raw, cases, reports = {}, {}, {}
    for label in CASE_ORDER:
        folder = root/FOLDERS[label]; alpha = ALPHAS[label]
        # Keep the parent's full native/tensor/unmasked-device proof. Passing
        # the declared index explicitly prevents silently validating as GPU1.
        proof = parent.validate_output(folder, alpha, physical_gpu_index=physical_gpu_index)
        require(proof.get("status") == "verified" and proof.get("alpha") == alpha,
            "Exact parent filter-output verifier did not approve this alpha cohort")
        raw[label], cases[label], reports[label] = shared._load_case(folder, label, reference)
        control = _archive(folder/parent.evaluator.CONTROL_FILE)
        physics = _archive(folder/parent.evaluator.PHYSICS_FILE)
        target_windows = target_control_diagnostics(raw[label], control, physics)
        for window in ("core", "full"):
            # Generic action metrics are for requested actions, not the newly
            # filtered target. Omit them instead of mislabeling q0+action as applied.
            cases[label]["windows"][window].pop("action", None)
            cases[label]["windows"][window]["target_control"] = target_windows[window]
        cases[label].update(alpha=alpha, parent_filter_verification=proof,
            physical_gpu_index_requested=physical_gpu_index,
            input_telemetry_sha256={name: smooth.sha256(folder/name) for name in
                (parent.evaluator.CONTROL_FILE, parent.evaluator.PHYSICS_FILE, parent.evaluator.RUNTIME_FILE)},
            filter_state_recorded=True, generic_requested_action_target_metrics_omitted=True,
            deployment_filter_is_in_actor_export=False,
            target_clock_origins={"absolute_native_control_time_s": float(control["time_s"][0]),
                "absolute_sim_step_counter_origin": int(control["sim_step_counter"][0]),
                "absolute_native_pre_step_time_s": float(physics["time_s"][0]),
                "original_relative_physical_state_origin_s": float(raw[label]["time_s"][0])})
    require(cases["baseline"]["checkpoint_sha256"] == cases["candidate"]["checkpoint_sha256"] == parent.base.CHECKPOINT_SHA,
        "This is a SAME frozen -0.5 actor comparison, not separately trained policies")
    for key in shared.PHYSICAL_KEYS:
        require(key in reports["baseline"] and key in reports["candidate"] and reports["baseline"][key] == reports["candidate"][key],
            "Shared nominal physical/contact/evaluator contract differs: "+key)
    before, after = [reports[label]["policy_export"] for label in CASE_ORDER]
    ignored = {"leg_target_filter_diagnostic", "external_leg_target_filter_alpha", "export_validation_max_abs_error"}
    for key in (set(before)|set(after))-ignored:
        require(before.get(key) == after.get(key), "Shared actor/body/PD/observation contract differs: "+key)
    for key in ("body_names", "wheel_body_names", "joint_names", "time_s", "simulation_step_counter"):
        require(key in raw["baseline"] and key in raw["candidate"] and np.array_equal(raw["baseline"][key], raw["candidate"][key]),
            "Unchanged original native body/clock schema differs: "+key)
    changes = shared._changes(cases)
    for window in ("core", "full"):
        for field in ("requested_target_step_rms_rad", "filtered_applied_target_step_rms_rad", "physical_leg_applied_torque_step_rms_nm"):
            a, b = [cases[label]["windows"][window]["target_control"][field] for label in CASE_ORDER]
            changes[window]["target_control."+field] = {"baseline": a, "candidate": b, "difference": b-a,
                "ratio": b/a if a > 1e-12 else None}
    return raw, {"schema_version": 1, "kind": "recorded_nominal_ppo_processed_leg_target_filter_comparison",
        "status": "requires_visual_and_tracking_review", "case_order": list(CASE_ORDER), "cases": cases,
        "same_frozen_actor": True, "checkpoint_sha256": parent.base.CHECKPOINT_SHA,
        "physical_gpu_index_requested": physical_gpu_index, "requested_device": f"cuda:{physical_gpu_index}",
        "alpha_in_order": [1., .5], "baseline_is_exact_unfiltered_replay": True,
        "baseline_expected_original_trace_sha256": parent.BASELINE_RAW_SHA256,
        "reference_sha256": shared.BASELINE_MOTION_SHA256, "evaluation_mu": .7,
        "leg_kp": 500., "leg_kd": 10., "configured_solver_position_iterations": 8, "configured_solver_velocity_iterations": 4,
        "actor_previous_action_meaning": "Original requested clipped8-action vector; filter state is not added to unchanged51D actor observation",
        "filter_target_semantics": "Six post-scale/offset leg software position targets only, updated50Hz and held4 times at200Hz; no wheel filtering",
        "generic_requested_action_target_metrics_omitted": True, "relative_changes": changes,
        "native_physical_hz": 200., "control_hz": 50., "core_interval_s": [1.44, 4.], "cutoff_hz_exclusive": 10.,
        "filter_training_updates": 0, "new_simulation": False, "new_inference": False, "pose_smoothing_or_state_correction": False,
        "new_training": False, "resampling_or_state_alignment": False, "private_weights_downloaded": False,
        "contact_success_threshold_added": False, "stable_contact_claimed": False, "height_or_flight_failure_gate": False,
        "nonzero_sideslip_target_preserved": True, "requires_fullbody_visual_review": True,
        "point_count_definition": "Ground-filtered points with individual normal force>5N; zero count is not geometric flight",
        "support_load_eligibility_gate": "recorded count>0 AND aggregate upward Fz>5N; not a failure definition",
        "analysis_source_sha256": {Path(__file__).name: smooth.sha256(__file__),
            "run_ppo_leg_target_filter.py": smooth.sha256(parent.__file__),
            "eval_ppo_leg_target_filter.py": smooth.sha256(parent.evaluator.__file__),
            "analyze_ppo_ground_support_comparison.py": smooth.sha256(shared.__file__),
            "analyze_ppo_smoothness.py": smooth.sha256(smooth.__file__)},
        "limitations": [
            "This is zero-shot external controller filtering of one frozen actor, not retraining. Unobserved filter state/delay changes deployment dynamics and observation distribution.",
            "Requested actor actions and previous-action observations remain unchanged in meaning; subsequent requested actions can change because filtered targets change measured plant state.",
            "Actual target changes come only from separate post-scale/offset processed-target telemetry with4-substep software readback. q0+rawaction is not mislabeled as the filtered applied target.",
            "Physical joint positions/velocities, applied torque, contacts and body poses are original200Hz recordings; no visualization smoothing or new dynamics is performed.",
            "Software target telemetry is pre-step and physical recording post-step. Target-window masks use counter-derived nominal schedule (sim_step_counter-counter0)*.005s, not accumulated native float time; native timestamps are checked within2microseconds and preserved. No pose/target interpolation or resampling occurs.",
            "Contact absence at threshold5N does not establish flight or failure; weak contacts are unavailable and normal load is not tangential friction force or evaluation mg normalization.",
            "Physical horizon6.98s exceeds reference6.96s, so full tracking is unavailable without extrapolation; reference-overlap tracking is separately labeled.",
            "One checkpoint/seed/episode per setting is descriptive only. Review full-body video, contact, XY/speed/beta and tilt together; no automatic pass is invented."]}


def render_contact_plot(raw, summary, destination):
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
        axes[row, 0].set_ylabel(f"Same actor alpha={ALPHAS[label]:g}\nGround Fz [N]")
        axes[row, 2].set_ylabel("Raw angular velocity [rad/s]")
        for axis in axes[row]:
            axis.axvspan(1.44, 4., color="#888888", alpha=.1); axis.grid(alpha=.2); axis.legend(fontsize=7); axis.set_xlim(0., 6.98)
    for axis, title in zip(axes[0], ("Measured ground normal load", "Support-load eligibility, not flight", "Original body angular XY")): axis.set_title(title)
    for axis in axes[-1]: axis.set_xlabel("Native physical time [s]")
    fig.suptitle("Nominal PhysX mu=0.7 | same frozen actor, external6-leg target LPF | poses NOT smoothed")
    fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing analysis: output directory must be fresh")
    raw, summary = load_comparison(args.input_root, args.reference)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"measured_states.npz", **shared.measured_state_arrays(raw))
    render_contact_plot(raw, summary, output/"contact_comparison.png")
    summary["artifacts"] = {name: {"file": name, "sha256": smooth.sha256(output/name)} for name in ("measured_states.npz", "contact_comparison.png")}
    with (output/"summary.json").open("x") as stream: json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    result = run(parser().parse_args()); print(json.dumps({"status": result["status"], "stable_contact_claimed": False}))
