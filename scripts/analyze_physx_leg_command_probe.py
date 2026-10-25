"""Strict native contact/command analysis of the inference-time ramp probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_k_native_slices9 as prior
import ppo_backend_400hz_contract as recording

ROOT = Path(__file__).resolve().parents[1]
MODES = {"passthrough_v1": "passthrough", "linear_v1": "linear_position_8substeps"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def phase_statistics(raw, trace):
    gaps = prior.support_gap_diagnostics(raw, trace)
    starts = np.asarray([row["start_sample"] for row in gaps["interruptions"]], dtype=int)
    phase = (np.arange(2793)-1)%8
    time = trace["time_s"]
    selected = (time >= .02-1e-9) & (time <= 6.96+1e-9)
    lost = selected & (trace["floor_normal_force"][:, :, 2].max(-1) <= 5.)
    return dict(onset_phase_counts=np.bincount((starts-1)%8, minlength=8).tolist(),
        unsupported_samples_by_phase=[int(np.sum(lost & (phase == k))) for k in range(8)],
        observed_samples_by_phase=[int(np.sum(selected & (phase == k))) for k in range(8)],
        first_phase_definition="first post-step native sample after the50Hz policy update",
        support_interruption_count=len(starts), causal_attribution_from_timing_alone=False)


def verify_case(folder, mode, protocol):
    folder = Path(folder)
    metadata = read(folder/"leg_command_status.json")
    report = read(folder/"report.json")
    assert metadata == read(folder/"wheel_contact_model_status.json") == report["wheel_contact_model_transfer"]
    assert metadata["kind"] == "K_policy_PhysX_leg_command_execution_diagnostic"
    assert metadata["normalized_actor_sha256"] == protocol["actor_sha256"] and metadata["additional_training_updates"] == 0
    assert metadata["geometry"] == "authored" and metadata["law"] == "native_coulomb" and metadata["mu"] == .7
    assert metadata["native_states"] == 2793 and metadata["policy_decisions"] == 349
    assert metadata["command_probe_protocol_sha256"] == sha(ROOT/"config/physx_leg_command_probe_v1.json")
    assert report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
    assert report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
    assert report["terminal_states_captured_before_auto_reset"] is True
    assert report["runtime_device_audit"]["status"] == "verified"
    assert metadata["native_audit"]["masses_COM_inertias_armature_damping_match_baseline"]
    command = metadata["command_execution_diagnostic"]
    assert command == report["action_execution_intervention"] == report["policy_export"]["action_execution_intervention"]
    assert command["mode"] == mode and command["physical_steps"] == 2792
    assert command["validation_only_intervention"] and command["trained_for_interpolated_execution"] is False
    assert command["wheel_effort_interpolation"] is False and command["command_endpoints_exact"]
    assert command["actor_parameters_and_reward_functions_changed"] is False
    assert command["physical_normal_or_friction_model_changed"] is False
    for remote_path, digest in metadata["source_sha256"].items():
        path = ROOT/("config" if Path(remote_path).suffix == ".json" else "scripts")/Path(remote_path).name
        assert path.is_file() and sha(path) == digest, "Executed source differs: "+remote_path
    for name in ("physics_trajectory", "tire_trajectory", "leg_command_trajectory"):
        assert sha(folder/(name+".npz")) == metadata[name+"_sha256"]
    recording.verify_recording(folder/"physics_trajectory.npz", "PhysX")
    if mode == "passthrough":
        assert metadata["physics_trajectory_sha256"] == protocol["baseline_physics_trajectory_sha256"]
        assert metadata["read_only_baseline_byte_identical"]
    assert metadata["retraining_or_reward_action_reference_change"] == (mode != "passthrough")
    native = metadata["native_audit"]
    for wheel in native["wheels"].values():
        assert len(wheel["slots"]) == 1
        np.testing.assert_allclose(wheel["contact_offsets_m"], .001, rtol=0, atol=1e-8)
        np.testing.assert_allclose(wheel["rest_offsets_m"], 0., rtol=0, atol=1e-8)
    return metadata


def load_cases(collection, reference):
    protocol = read(ROOT/"config/physx_leg_command_probe_v1.json")
    criteria = read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    original = read(ROOT/"results/physx_tron1_wheel_models_20261025_v1/analysis.json")["original_baseline_metrics"]
    rows, traces, raws = [], {}, {}
    for name, mode in MODES.items():
        folder = Path(collection)/name
        metadata = verify_case(folder, mode, protocol)
        raw = prior.previous.npz(folder/"physics_trajectory.npz")
        commands = prior.previous.npz(folder/"leg_command_trajectory.npz")
        values, trace = prior.metrics.recorded_metrics(raw, "physx", reference)
        trace["contact_force"] = trace["floor_normal_force"].copy()
        target = commands["desired_policy_target_rad"].reshape(349, 8, 6)
        actual = commands["actual_leg_position_target_rad"].reshape(349, 8, 6)
        old = commands["previous_policy_target_rad"].reshape(349, 8, 6)
        np.testing.assert_array_equal(target, np.broadcast_to(target[:, :1], target.shape))
        np.testing.assert_array_equal(old, np.broadcast_to(old[:, :1], old.shape))
        np.testing.assert_array_equal(actual[:, -1], target[:, -1])
        np.testing.assert_array_equal(old[1:, 0], actual[:-1, -1])
        if mode == "passthrough":
            np.testing.assert_array_equal(actual, target)
        else:
            np.testing.assert_allclose(actual, old+(target-old)*np.arange(1, 9)[None, :, None]/8., rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(commands["unchanged_wheel_effort_target_nm"], raw["actions_clipped"][1:, 0, 6:]*12., rtol=1e-6, atol=1e-6)
        offset = commands["desired_policy_target_rad"]-raw["actions_clipped"][1:, 0, :6]
        np.testing.assert_allclose(offset, np.broadcast_to(offset[:1], offset.shape), rtol=0, atol=2e-7)
        np.testing.assert_allclose(commands["time_before_step_s"], trace["time_s"][:-1], rtol=0, atol=1e-12)
        step_changes = np.linalg.norm(np.diff(np.vstack((old[0, 0], actual.reshape(2792, 6))), axis=0), axis=-1)
        rows.append(dict(id=name, mode=mode, metrics=values,
            nominal_screen=prior.metrics.nominal_screen(values, original, criteria, "physx"),
            phase_statistics=phase_statistics(raw, trace),
            support_gap_diagnostics=prior.support_gap_diagnostics(raw, trace),
            actual_leg_target_step_L2_rad=dict(median=float(np.median(step_changes)), p95=float(np.percentile(step_changes, 95)), maximum=float(step_changes.max())),
            maximum_ground_normal_force_n=float(trace["floor_normal_force"][:, :, 2].max()),
            metadata=metadata, input_sha256={p.name: sha(p) for p in folder.iterdir() if p.suffix in (".json", ".npz")}))
        traces[name], raws[name] = trace, raw
    for field in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel"):
        np.testing.assert_array_equal(traces["passthrough_v1"][field][0], traces["linear_v1"][field][0])
    return rows, traces, raws


def run(args):
    assert not args.output_dir.exists()
    reference = prior.metrics.metrics.reference_data(args.reference)
    rows, traces, _ = load_cases(args.collection, reference)
    args.output_dir.mkdir(parents=True)
    result = dict(kind="frozen_actor_native_leg_target_execution_intervention", status="completed_analysis", rows=rows,
        reference_sha256=sha(args.reference), analyzer_sha256=sha(__file__),
        original_observer_recording_byte_identical=True, same_initial_mechanical_state=True,
        actor_parameters_reward_physics_and_reference_unchanged=True,
        policy_actions_can_change_through_closed_loop_state_feedback=True,
        interpolation_is_actual_actuator_input_change_not_pose_smoothing=True,
        trained_for_interpolated_execution=False, cross_engine_qualification_complete=False, task_success_claimed=False)
    (args.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), layout="constrained")
    axes[0].plot(reference["position"][:, 0], reference["position"][:, 1], "--", color="gray", label="reference")
    for row, color in zip(rows, ("#2271b2", "#d55e00"), strict=True):
        trace = traces[row["id"]]
        axes[0].plot(trace["position"][:, 0], trace["position"][:, 1], color=color, label=row["mode"])
        axes[1].plot(trace["time_s"], trace["floor_normal_force"][:, :, 2].max(-1), color=color, label=row["mode"], lw=.7)
        axes[2].plot(trace["time_s"], np.linalg.norm(trace["angular_velocity"][:, :2], axis=-1), color=color, label=row["mode"], lw=.7)
        shift = -.18 if row["mode"] == "passthrough" else .18
        axes[3].bar(np.arange(1, 9)+shift, row["phase_statistics"]["onset_phase_counts"], width=.36, color=color, label=row["mode"])
    axes[0].set(xlabel="World X (m)", ylabel="World Y (m)", title="Recorded XY", aspect="equal")
    axes[1].axhline(5., color="black", ls="--", lw=1)
    axes[1].set(xlabel="Time (s)", ylabel="Max wheel Fz (N)", title="Actual upward normal load", yscale="symlog")
    axes[2].set(xlabel="Time (s)", ylabel="Root angular XY norm (rad/s)", title="Native angular motion")
    axes[3].set(xlabel="Physical substep after policy update", ylabel="Support interruption onsets", xticks=np.arange(1, 9), title="Timing of support loss")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Same frozen K actor | PhysX mu=.7 | command execution diagnostic, no training")
    fig.savefig(args.output_dir/"comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps([dict(id=r["id"], core_XY=r["metrics"]["windows"]["core"]["tracking"]["xy_rmse_m"],
        support=r["metrics"]["any_wheel_ground_support"]["full_contact"]["any_wheel_support_sample_fraction"],
        gap_s=r["metrics"]["any_wheel_ground_support"]["full_contact"]["longest_no_support_observed_s"],
        HF_ratio=r["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"], failed=r["nominal_screen"]["failed_checks"]) for r in rows]))
    return result


if __name__ == "__main__":
    import run_cem_ground_contact_v2 as assets
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
