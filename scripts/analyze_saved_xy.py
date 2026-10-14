"""Audit and render saved matched-friction Isaac/MuJoCo root-XY traces.

Two modes: import three archived cases using --source-root, or regenerate the
figures/metrics solely from the compact trace shipped with this repository.
This script does not simulate, train, retarget, or infer tire contact/slip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = {
    "mu0p05": (0.05, "xy-failure-modes/mu0p05"),
    "mu0p70": (0.70, "matched-friction-mu0p70"),
    "mu1p00": (1.00, "xy-failure-modes/mu1p00"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def validate_path(times: np.ndarray, xy: np.ndarray, name: str) -> None:
    if (times.ndim != 1 or xy.shape != (len(times), 2) or len(times) < 2
            or not np.isfinite(times).all() or not np.isfinite(xy).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError(f"{name}: invalid native-clock XY path")


def observed_episode(data: dict[str, np.ndarray], engine: str) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(data["time_s"], dtype=float)
    if engine == "isaac":
        valid = np.asarray(data["valid_mask"][:, 0], dtype=bool)
        if not valid[0] or np.any(np.diff(valid.astype(int)) > 0):
            raise ValueError("Isaac first-episode mask is not a single prefix")
        time, xy = time[valid], np.asarray(data["root_position_m"][:, 0, :2], dtype=float)[valid]
    else:
        xy = np.asarray(data["root_pos"][:, :2], dtype=float)
    validate_path(time, xy, engine)
    return time, xy


def interpolate(time: np.ndarray, xy: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    if np.any(query < time[0] - 1e-9) or np.any(query > time[-1] + 1e-9):
        raise ValueError("Refusing to extrapolate beyond an observed trajectory")
    return np.column_stack([np.interp(query, time, xy[:, j]) for j in range(2)])


def path_error(time: np.ndarray, xy: np.ndarray, ref_t: np.ndarray,
               ref_xy: np.ndarray, core: tuple[float, float]) -> dict:
    # The physics recorder can include one terminal sample after the final
    # reference frame. Hold that last target only for the overall peak.
    if time[0] < ref_t[0] - 1e-9:
        raise ValueError("Trajectory starts before reference")
    target = interpolate(ref_t, ref_xy, np.minimum(time, ref_t[-1]))
    error = np.linalg.norm(xy - target, axis=1)
    mask = (time >= core[0] - 1e-9) & (time <= core[1] + 1e-9)
    complete = bool(time[-1] >= core[1] - 1e-9)
    return {
        "last_observed_time_s": float(time[-1]),
        "core_complete": complete,
        "observed_core_samples": int(mask.sum()),
        "observed_core_reference_xy_rmse_m": float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None,
        "full_core_reference_xy_rmse_m": float(np.sqrt(np.mean(error[mask] ** 2))) if complete else None,
        "full_core_reference_xy_peak_m": float(np.max(error[mask])) if complete else None,
        "observed_whole_clip_reference_xy_peak_m": float(np.max(error)),
    }


def pair_error(it: np.ndarray, ip: np.ndarray, mt: np.ndarray, mp: np.ndarray,
               core: tuple[float, float]) -> dict:
    overlap_end = min(it[-1], mt[-1])
    common = it[it <= overlap_end + 1e-9]
    if len(common) < 2:
        raise ValueError("Insufficient native-clock overlap")
    delta = np.linalg.norm(ip[:len(common)] - interpolate(mt, mp, common), axis=1)
    mask = (common >= core[0] - 1e-9) & (common <= core[1] + 1e-9)
    return {
        "overlap_end_s": float(overlap_end),
        "overlap_xy_rmse_m": float(np.sqrt(np.mean(delta ** 2))),
        "overlap_xy_peak_m": float(np.max(delta)),
        "observed_core_xy_rmse_m": float(np.sqrt(np.mean(delta[mask] ** 2))) if mask.any() else None,
        "full_core_overlap": bool(overlap_end >= core[1] - 1e-9),
        "full_core_xy_rmse_m": float(np.sqrt(np.mean(delta[mask] ** 2))) if overlap_end >= core[1] - 1e-9 else None,
    }


def read_source(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    base = root / "results/2026-10-14-car-drift"
    ref_file = base / "xy-failure-modes/robot_reference.npz"
    ref = arrays(ref_file)
    rt = np.asarray(ref["time_s"], dtype=float)
    rp = np.asarray(ref["root_target_position_m"][:, :2], dtype=float)
    validate_path(rt, rp, "reference")
    trace = {"reference_t": rt, "reference_xy": rp}
    provenance = {"reference_sha256": sha(ref_file), "cases": {}}
    motion_hashes = set()
    for name, (mu, relative) in CASES.items():
        directory = base / relative
        ia_file = directory / "isaac_physics_trajectory.npz"
        mj_file = directory / "mujoco_rollout.npz"
        ia_report = json.loads((directory / "isaac_report.json").read_text())
        mj_report = json.loads((directory / "mujoco_report.json").read_text())
        ia_mu = ia_report["evaluation_contact_friction_override"]["evaluation_contact_friction"]
        mj_range = mj_report["contact_material_audit"]["compiled_friction_range"]
        ia_expected = ia_report["contact_material_audit"]["expected_friction"]
        if (not np.isclose(ia_mu, mu) or not np.isclose(ia_expected, mu)
                or not np.allclose(mj_range, [mu, mu])
                or ia_report["physics_trajectory_sha256"] != sha(ia_file)
                or ia_report["motion_file_sha256"] != mj_report["motion_sha256"]):
            raise ValueError(f"{name}: friction/readback/trace/reference audit failed")
        motion_hashes.add(ia_report["motion_file_sha256"])
        it, ip = observed_episode(arrays(ia_file), "isaac")
        mt, mp = observed_episode(arrays(mj_file), "mujoco")
        trace.update({f"{name}_isaac_t": it, f"{name}_isaac_xy": ip,
                      f"{name}_mujoco_t": mt, f"{name}_mujoco_xy": mp})
        provenance["cases"][name] = {
            "mu": mu, "isaac_trace_sha256": sha(ia_file), "mujoco_trace_sha256": sha(mj_file),
            "isaac_report_sha256": sha(directory / "isaac_report.json"),
            "mujoco_report_sha256": sha(directory / "mujoco_report.json"),
            "isaac_contact_friction_readback": ia_report["contact_material_audit"],
            "mujoco_compiled_sliding_friction_range": mj_range,
            "isaac_termination": ia_report.get("episode_summaries", [{}])[0].get("termination"),
            "mujoco_termination": mj_report.get("termination"),
        }
    if len(motion_hashes) != 1:
        raise ValueError("Saved cases do not use one frozen reference motion")
    provenance["motion_sha256"] = motion_hashes.pop()
    return trace, provenance


def summarize(trace: dict[str, np.ndarray], provenance: dict, config: dict,
              cases: dict | None = None) -> dict:
    ref_t, ref_xy = trace["reference_t"], trace["reference_xy"]
    validate_path(ref_t, ref_xy, "reference")
    core = tuple(config["screening_core_interval_s"])
    if len(core) != 2 or not (ref_t[0] <= core[0] < core[1] <= ref_t[-1]):
        raise ValueError("Core interval must lie inside reference")
    rows = []
    for name, (mu, _) in (CASES if cases is None else cases).items():
        it, ip = trace[f"{name}_isaac_t"], trace[f"{name}_isaac_xy"]
        mt, mp = trace[f"{name}_mujoco_t"], trace[f"{name}_mujoco_xy"]
        validate_path(it, ip, f"{name} Isaac")
        validate_path(mt, mp, f"{name} MuJoCo")
        ia = path_error(it, ip, ref_t, ref_xy, core)
        mj = path_error(mt, mp, ref_t, ref_xy, core)
        pair = pair_error(it, ip, mt, mp, core)
        for metrics in (ia, mj):
            metrics["large_reference_xy_screen"] = (
                (metrics["full_core_reference_xy_rmse_m"] > config["large_reference_xy_rmse_m"]
                 or metrics["full_core_reference_xy_peak_m"] > config["large_reference_xy_peak_m"])
                if metrics["core_complete"] else None)
        rows.append({"case": name, "evaluation_mu_static_dynamic_equal": mu,
                     "isaac": ia, "mujoco": mj, "sim2sim": pair})
    return {"schema_version": 1, "kind": "historical_fixed_mu_policy_matched_friction_xy_baseline",
            "policy_training_friction_randomization": False,
            "planned_training_friction_randomization": bool(config["policy_training_friction_randomization"]),
            "core_interval_s": list(core), "screening_limits_m": {
                "reference_core_rmse": config["large_reference_xy_rmse_m"],
                "reference_core_peak": config["large_reference_xy_peak_m"]},
            "provenance": provenance, "rows": rows,
            "limitations": ["Three archived deterministic pairs, not a completed BeyondMimic-range grid or a friction threshold estimate.",
                            "The frozen policy was trained at mu=0.15; no friction randomization was added to training.",
                            "Early termination censors the full-core XY verdict; trajectories are never extrapolated.",
                            "Equal numerical PhysX and MuJoCo friction is not physical tire-ground calibration.",
                            "This root-XY view does not establish actual wheel-ground patch slip or human GMR feasibility."]}


def render(trace: dict[str, np.ndarray], output: Path, cases: dict | None = None) -> None:
    cases = CASES if cases is None else cases
    fig, axes = plt.subplots(1, len(cases), figsize=(max(4.4 * len(cases), 6), 4.4), dpi=150)
    axes = np.atleast_1d(axes)
    ref = trace["reference_xy"]
    for ax, (name, (mu, _)) in zip(axes, cases.items()):
        ia, mj = trace[f"{name}_isaac_xy"], trace[f"{name}_mujoco_xy"]
        ax.plot(ref[:, 0], ref[:, 1], color="#c68b00", lw=1.8, label="reference")
        ax.plot(ia[:, 0], ia[:, 1], color="#0574b6", lw=1.5, label="Isaac")
        ax.plot(mj[:, 0], mj[:, 1], color="#d55e00", lw=1.5, label="MuJoCo")
        ax.scatter([ia[-1, 0], mj[-1, 0]], [ia[-1, 1], mj[-1, 1]],
                   color=["#0574b6", "#d55e00"], marker="x", s=38)
        ax.set(title=f"evaluation µ={mu:.2f}", xlabel="world X (m)", ylabel="world Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Frozen-policy matched-friction XY paths (× = last observed state)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-root", type=Path, help="Local full experiment source checkout")
    mode.add_argument("--trace", type=Path, help="Compact published XY-only NPZ")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/evaluation_friction.json").read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty")
    if args.source_root:
        trace, provenance = read_source(args.source_root)
    else:
        trace = arrays(args.trace)
        provenance = json.loads(args.trace.with_suffix(".provenance.json").read_text())
    summary = summarize(trace, provenance, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source_root:
        np.savez_compressed(args.output_dir / "xy_trace.npz", **trace)
        (args.output_dir / "xy_trace.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(trace, args.output_dir / "topdown.png")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": summary["rows"]}, indent=2))


if __name__ == "__main__":
    main()
