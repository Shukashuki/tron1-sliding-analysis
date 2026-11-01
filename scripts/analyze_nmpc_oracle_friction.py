"""Audit one fixed-model versus exact-plant-friction CEM MPC pair.

This comparison is an oracle upper-bound diagnostic, not friction estimation or
a deployable controller. It consumes completed physical rollouts only.
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


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_episode(directory: Path):
    report = json.loads((directory / "report.json").read_text())
    require(report["status"] == "completed" and not report["truncated_smoke"],
            f"{directory}: expected completed full-horizon episode")
    require(report["rollout_sha256"] == digest(directory / "rollout.npz")
            and report["solver_log_sha256"] == digest(directory / "solver_log.json"),
            f"{directory}: saved rollout/solver hash mismatch")
    with np.load(directory / "rollout.npz", allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in
                 ("time_s", "root_pos", "reference_xy_m", "speed_m_s")}
    time = trace["time_s"]
    require(time.ndim == 1 and len(time) == 1397 and np.allclose(np.diff(time), .005, atol=1e-9)
            and np.isclose(time[-1], 6.98, atol=1e-8), f"{directory}: incomplete physical clock")
    require(trace["root_pos"].shape == (len(time), 3)
            and trace["reference_xy_m"].shape == (len(time), 2)
            and trace["speed_m_s"].shape == (len(time),), f"{directory}: trace shape mismatch")
    require(all(np.isfinite(value).all() for value in trace.values()), f"{directory}: nonfinite trace")
    core = (time >= 1.44 - 1e-9) & (time <= 4.0 + 1e-9)
    error = np.linalg.norm(trace["root_pos"][:, :2] - trace["reference_xy_m"], axis=1)
    rmse = float(np.sqrt(np.mean(error[core] ** 2)))
    peak = float(error[core].max())
    require(np.isclose(rmse, report["core_xy_rmse_m"], atol=1e-8)
            and np.isclose(peak, report["core_xy_peak_m"], atol=1e-8),
            f"{directory}: report XY metrics do not reproduce from rollout")
    return report, trace, {"core_xy_rmse_m": rmse, "core_xy_peak_m": peak,
                           "core_speed_retention_ratio": report["core_speed_retention_mean_ratio"],
                           "solver_failures": report["solver_failures"],
                           "median_compute_ms": report["compute_ms_median_p95_max"][0]}


def analyze(fixed_dir: Path, oracle_dir: Path, output_dir: Path):
    fixed, fixed_trace, fixed_metrics = load_episode(fixed_dir)
    oracle, oracle_trace, oracle_metrics = load_episode(oracle_dir)
    require(fixed["engine"] == oracle["engine"] == "MuJoCo",
            "This audit currently supports same-engine MuJoCo pairs only")
    mu = float(fixed["plant_mu"])
    require(np.isclose(mu, oracle["plant_mu"]) and mu >= 0, "Plant μ differs between episodes")
    for name in ("engine", "model_sha256", "motion_sha256", "reference_id", "settings",
                 "effective_cem_config", "objective_contract", "action_parameterization",
                 "prediction_backend", "physics_dt_s", "control_dt_s"):
        require(fixed[name] == oracle[name], f"Controller/reference changed: {name}")
    require(fixed["prediction_mu_scenarios"] == [.7] and not fixed["prediction_receives_plant_mu"],
            "Fixed model is not the frozen μ=0.7 baseline")
    require(len(oracle["prediction_mu_scenarios"]) == 1
            and np.isclose(oracle["prediction_mu_scenarios"][0], mu)
            and oracle["prediction_receives_plant_mu"]
            and oracle["prediction_friction_source"] == "plant_friction_oracle",
            "Oracle model does not explicitly receive exact plant μ")
    for report in (fixed, oracle):
        require(np.allclose(report["plant_material_readback"]["compiled_sliding_friction_range"],
                            [mu, mu], atol=1e-9), "Plant material readback mismatch")
        require(report["completed_policy_steps"] == report["total_reference_policy_steps"] == 349,
                "Episode did not finish all policy steps")
    require(np.array_equal(fixed_trace["time_s"], oracle_trace["time_s"])
            and np.allclose(fixed_trace["reference_xy_m"], oracle_trace["reference_xy_m"], atol=1e-9),
            "Episodes do not share the same reference clock and XY target")
    # The runner source changed only to introduce the new labeled mode. All
    # other hashed source dependencies must remain bit-identical.
    def stable_sources(report):
        return {path: value for path, value in report["source_sha256"].items()
                if Path(path).name != "run_nmpc_tracking.py"}
    require(stable_sources(fixed) == stable_sources(oracle), "A non-runner source dependency changed")
    summary = {"kind": "nmpc_fixed_vs_exact_plant_friction_oracle", "engine": fixed["engine"],
               "plant_mu": mu, "prediction_mu_fixed": .7, "prediction_mu_oracle": mu,
               "reference_id": fixed["reference_id"], "seed": 42,
               "oracle_is_privileged_not_deployable": True,
               "fixed": fixed_metrics, "oracle": oracle_metrics,
               "oracle_minus_fixed_core_xy_rmse_m": oracle_metrics["core_xy_rmse_m"] - fixed_metrics["core_xy_rmse_m"],
               "oracle_minus_fixed_speed_retention_ratio": oracle_metrics["core_speed_retention_ratio"] - fixed_metrics["core_speed_retention_ratio"],
               "input_sha256": {"fixed_report": digest(fixed_dir / "report.json"),
                                "fixed_rollout": digest(fixed_dir / "rollout.npz"),
                                "oracle_report": digest(oracle_dir / "report.json"),
                                "oracle_rollout": digest(oracle_dir / "rollout.npz")}}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    reference = fixed_trace["reference_xy_m"]
    ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.5, label="reference")
    for label, trace in (("fixed predictor μ=0.70", fixed_trace),
                         (f"oracle predictor μ={mu:.2f}", oracle_trace)):
        xy = trace["root_pos"][:, :2]
        ax.plot(xy[:, 0], xy[:, 1], lw=1.5, label=label)
    ax.set(xlabel="X (m)", ylabel="Y (m)", title=f"CEM MuJoCo, plant μ={mu:.2f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "topdown.png", dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.fixed_dir, args.oracle_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("plant_mu", "fixed", "oracle",
                                                     "oracle_minus_fixed_core_xy_rmse_m")}, allow_nan=False))
