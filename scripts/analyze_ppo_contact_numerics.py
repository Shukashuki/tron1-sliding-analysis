"""Analyze fixed-actor solver4/8 recordings; no physics or inference is run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_contact_stability as contact
import analyze_ppo_smoothness as smooth
import run_ppo_contact_numerics as experiment
from summarize_ppo_smoothness import posture_diagnostics

ORDER = ("solver_v4", "solver_v8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_comparison(input_root, reference):
    root = Path(input_root)
    require(smooth.sha256(reference) == contact.MOTION_SHA, "Locked reference SHA differs")
    status = json.loads((root / "contact_numerics_status.json").read_text())
    require(status["status"] == "episodes_complete_requires_contact_and_tracking_review"
            and status["paired_native_offsets_and_materials_identical"] is True,
            "Require both completed, verified numerical episodes")
    raw, cases, proofs, reports = {}, {}, {}, {}
    for label, iterations in zip(ORDER, (4, 8)):
        directory = root / label
        proofs[label] = experiment.validate_output(directory, iterations)
        reports[label] = json.loads((directory / "report.json").read_text())
        require(reports[label]["motion_file_sha256"] == contact.MOTION_SHA,
                "Evaluation motion SHA differs from locked reference")
        with np.load(directory / "physics_trajectory.npz", allow_pickle=False) as archive:
            raw[label] = {key: archive[key] for key in archive.files}
        data = raw[label]
        trajectory = smooth.analyze_trajectory(directory / "physics_trajectory.npz",
            reference=reference, report=directory / "report.json", high_frequency_hz=10.)
        windows = contact.analyze_signals(data["time_s"], contact._field(data, "wheel_ground_contact_count"),
            contact._field(data, "wheel_ground_contact_force_w_n"),
            contact._field(data, "root_linear_velocity_w_m_s"),
            contact._field(data, "root_angular_velocity_w_rad_s"),
            nonwheel_counts=contact._field(data, "nonwheel_ground_contact_count"))
        cases[label] = {"configured_solver_velocity_iterations": iterations,
            "recording_sha256": proofs[label]["recording_sha256"],
            "report_sha256": proofs[label]["report_sha256"],
            "trajectory": trajectory, "contact": windows,
            "posture": posture_diagnostics(data), "runtime_readback": proofs[label]["runtime_readback"]}
    experiment.compare_readbacks(proofs[ORDER[0]], proofs[ORDER[1]])
    for key in ("task_profile", "evaluation_task_profile", "contact_material_audit", "physics_recording",
                "contact_slip_measurement", "domain_randomization_audit", "legacy_joint_friction_audit"):
        require(reports[ORDER[0]][key] == reports[ORDER[1]][key], "Additional setting changed: " + key)
    contracts = []
    for label in ORDER:
        contract = dict(reports[label]["policy_export"])
        for key in ("contact_numerics_diagnostic", "configured_solver_velocity_iteration_count"):
            contract.pop(key)
        parity = contract.pop("export_validation_max_abs_error", None)
        require(isinstance(parity, (int, float)) and np.isfinite(parity) and 0 <= parity <= 1e-4,
                "Actor export parity must be verified independently of physics")
        contracts.append(contract)
    require(contracts[0] == contracts[1], "Actor/control contract changed beyond solver setting")
    return raw, {"schema_version": 1, "kind": "fixed_actor_solver_velocity_numerical_sensitivity",
        "case_order": list(ORDER), "cases": cases, "mu": .7,
        "checkpoint_sha256": experiment.base.CHECKPOINT_SHA, "reference_sha256": smooth.sha256(reference),
        "additional_training_updates": 0, "native_sample_hz": 200, "policy_hz": 50,
        "height_or_airborne_failure_gate": False, "zero_sideslip_target": False,
        "stable_contact_claimed": False, "new_simulation_or_inference_by_analysis": False,
        "point_count_definition": "Ground contact points with individual normal force >5N; not all geometric contacts.",
        "load_eligibility_definition": "count>0 AND aggregate upward ground Fz>5N, separately for each wheel.",
        "limitations": ["Numerical sensitivity of a fixed policy trained with configured velocity count4; not a retrained-policy comparison.",
            "Requested and loaded USD solver counts are recorded; no native backend solver-count readback is available.",
            "Native robot contact/rest offsets and material arrays must match exactly across episodes; shape columns are not body indices.",
            "No-contact eligibility is not a geometric-flight classification or automatic failure gate.",
            "More solver iterations do not establish physical ground truth or numerical convergence.",
            "One deterministic episode per setting is descriptive; tracking, contact forces and full-body posture must be reviewed together."],
        "source_sha256": {name: smooth.sha256(Path(__file__).with_name(name)) for name in
            (Path(__file__).name, "eval_ppo_contact_numerics.py", "run_ppo_contact_numerics.py",
             "analyze_ppo_contact_stability.py", "analyze_ppo_smoothness.py", "summarize_ppo_smoothness.py")}}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing evidence; use a fresh output directory")
    raw, summary = load_comparison(args.input_root, args.reference)
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey="col")
    for row, label in enumerate(ORDER):
        time = raw[label]["time_s"]
        force = contact._field(raw[label], "wheel_ground_contact_force_w_n")[:, :, 2]
        omega = contact._field(raw[label], "root_angular_velocity_w_rad_s")
        for index, side in enumerate(("L", "R")):
            axes[row, 0].plot(time, force[:, index], lw=.6, label=side + " Fz")
            axes[row, 1].plot(time, omega[:, index], lw=.6, label=("wx", "wy")[index])
        axes[row, 0].set_ylabel(label + "\nGround Fz [N]")
        axes[row, 1].set_ylabel("World angular velocity [rad/s]")
        for axis in axes[row]:
            axis.axvspan(1.44, 4., alpha=.1, color="grey")
            axis.set_xlim(0, 6.98); axis.legend(); axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Original physical time [s]")
    figure.suptitle("Same PPO | mu=.7 | solver velocity4 vs8 | original200Hz signals, no smoothing")
    figure.tight_layout(); figure.savefig(output / "contact_comparison.png", dpi=140); plt.close(figure)
    np.savez_compressed(output / "recorded_states.npz", **{
        label + "_" + key: value for label, data in raw.items() for key, value in data.items()})
    summary["artifacts"] = {name: {"sha256": smooth.sha256(output / name)} for name in
        ("contact_comparison.png", "recorded_states.npz")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=contact.DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps({"status": "analyzed_recorded_solver_sensitivity",
                      "case_order": run(parser.parse_args())["case_order"]}))
