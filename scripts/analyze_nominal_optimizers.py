"""Audit and plot the primary μ=0.7 CEM / MPPI / SQP development episodes.

This deliberately compares recorded physical performance, not optimizer-only
causality: MPPI's Gaussian prior and the algorithms' evaluation budgets differ.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_run(path, kind, model_hash, motion_hash):
    path = Path(path)
    report = json.loads((path/"report.json").read_text())
    if (report.get("status") != "completed" or report.get("engine") != "MuJoCo"
            or report.get("reference_id") != "drift_drl_traj4"
            or report.get("model_sha256") != model_hash
            or report.get("motion_sha256") != motion_hash
            or report.get("plant_mu") != .7
            or report.get("prediction_mu_scenarios") != [.7]
            or report.get("completed_policy_steps") != 349):
        raise ValueError(f"Not a completed, identity-matched nominal {kind} run: {path}")
    if report.get("rollout_sha256") != digest(path/"rollout.npz"):
        raise ValueError(f"Trace hash mismatch: {kind}")
    if report.get("solver_log_sha256") != digest(path/"solver_log.json"):
        raise ValueError(f"Solver log hash mismatch: {kind}")
    solver = json.loads((path/"solver_log.json").read_text())
    if not isinstance(solver, list) or len(solver) != 349:
        raise ValueError(f"Solver decision count mismatch: {kind}")
    with np.load(path/"rollout.npz") as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    time = np.asarray(trace["time_s"], float)
    position = np.asarray(trace["root_pos"], float)
    reference_time = np.asarray(trace["reference_time_s"], float)
    reference_xy = np.asarray(trace["reference_root_xy_m"], float)
    if (time.shape != (1397,) or position.shape != (1397, 3)
            or reference_time.shape != (349,) or reference_xy.shape != (349, 2)
            or not np.allclose(time, np.arange(1397)*.005, atol=1e-8, rtol=0)
            or not np.allclose(reference_time, np.arange(349)*.02, atol=1e-8, rtol=0)):
        raise ValueError(f"Unexpected physical/reference clocks: {kind}")
    desired = np.column_stack([np.interp(time, reference_time, reference_xy[:, axis])
                               for axis in range(2)])
    error = np.linalg.norm(position[:, :2]-desired, axis=1)
    core = (time >= 1.44-1e-9) & (time <= 4.00+1e-9)
    rmse = float(np.sqrt(np.mean(error[core]**2)))
    peak = float(np.max(error[core]))
    if (not np.isclose(rmse, report["core_xy_rmse_m"], atol=1e-8, rtol=0)
            or not np.isclose(peak, report["core_xy_peak_m"], atol=1e-8, rtol=0)):
        raise ValueError(f"Recorded XY metrics do not replay: {kind}")
    q = np.asarray(trace["root_quaternion_wxyz"], float)
    tilt = np.degrees(np.arccos(np.clip(1.-2.*(q[:, 1]**2+q[:, 2]**2), -1., 1.)))
    count = np.asarray(trace["nonwheel_ground_contact_count"])
    compute = np.asarray(trace["compute_s"], float)
    if not np.array_equal(compute, [decision["compute_s"] for decision in solver]):
        raise ValueError(f"Solver and trace timing mismatch: {kind}")
    evaluations = sum(int(decision["solver"]["candidate_evaluations"])
                      for decision in solver)
    row = {
        "name": kind, "source_directory": str(path),
        "rollout_sha256": report["rollout_sha256"],
        "core_xy_rmse_m": rmse, "core_xy_peak_m": peak,
        "core_speed_retention_ratio": report["core_speed_retention_mean_ratio"],
        "core_speed_mean_m_s": report["core_speed_mean_m_s"],
        "max_tilt_deg": float(np.max(tilt)),
        "core_nonwheel_contact_fraction": float(np.mean(count[core] > 0)),
        "compute_ms_median_p95": (1000*np.quantile(compute, [.5, .95])).tolist(),
        "all_solves_exceed_20ms": bool(np.all(compute > .02)),
        "solver_failures": report.get("solver_failures"),
        "candidate_evaluations": evaluations,
        "sqp_successes": report.get("sqp_successes") if kind == "SQP" else None,
        "sqp_nonconverged_due_to_iteration_cap": (
            sum(decision["solver"].get("slsqp_status") == 9 for decision in solver)
            if kind == "SQP" else None),
    }
    return row, time, position[:, :2], desired, error


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Preserving prior comparison output: {output}")
    model_hash, motion_hash = digest(args.model), digest(args.motion)
    runs = {kind: load_run(getattr(args, kind.lower()), kind, model_hash, motion_hash)
            for kind in ("CEM", "MPPI", "SQP")}
    first = runs["CEM"]
    for kind, (_, times, _, desired, _) in runs.items():
        if not np.array_equal(times, first[1]) or not np.array_equal(desired, first[3]):
            raise ValueError(f"Reference or clock differs: {kind}")
    output.mkdir(parents=True)
    summary = {
        "kind": "nominal_optimizer_development_comparison", "status": "completed",
        "engine": "MuJoCo", "reference_id": "drift_drl_traj4", "plant_mu": .7,
        "prediction_mu": .7, "model_sha256": model_hash, "motion_sha256": motion_hash,
        "core_window_s": [1.44, 4.00], "cases": {key: value[0] for key, value in runs.items()},
        "limitations": [
            "MPPI includes a Gaussian action prior, unlike CEM and SQP; this is not a pure optimizer ablation.",
            "SQP uses finite-difference gradients and a three-iteration cap, not an exact converged solution.",
            "Rollout counts and episode timing environments are not matched; compute timings are descriptive.",
            "Only MuJoCo plant/predictor at nominal mu=0.7; no Isaac or real-time claim.",
            "One trajectory and one seed/configuration per method; no statistical ranking.",
        ],
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    axes[0].plot(first[3][:, 0], first[3][:, 1], color="black", linewidth=2.2, label="Reference")
    colors = {"CEM": "tab:blue", "MPPI": "tab:orange", "SQP": "tab:green"}
    for kind, (_, t, xy, _, error) in runs.items():
        axes[0].plot(xy[:, 0], xy[:, 1], color=colors[kind], linewidth=1.5, label=kind)
        axes[1].plot(t, error, color=colors[kind], linewidth=1.2, label=kind)
    axes[0].set(title="Top-down physical trajectory, μ=0.7", xlabel="World X (m)",
                ylabel="World Y (m)")
    axes[0].axis("equal")
    axes[0].legend(loc="best")
    axes[1].axvspan(1.44, 4.00, color="0.9", label="Evaluation core")
    axes[1].set(title="Unaligned XY tracking error", xlabel="Simulation time (s)",
                ylabel="XY error (m)")
    axes[1].legend(loc="best")
    figure.savefig(output/"topdown_and_error.png", dpi=160)
    plt.close(figure)
    print(json.dumps({key: value[0] for key, value in runs.items()}, indent=2), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cem", "mppi", "sqp", "model", "motion", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
