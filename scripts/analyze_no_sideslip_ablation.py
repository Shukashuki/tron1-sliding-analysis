"""Audit and plot the fixed-mu primary-turn PPO/CEM no-sideslip ablation.

This reads recorded physical trajectories only; it never advances either simulator.
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

from analyze_saved_xy import arrays, observed_episode, pair_error, path_error, validate_path
from controller_references import verify_reference_pair


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def evaluate(args: argparse.Namespace) -> tuple[dict, dict]:
    spec = verify_reference_pair(args.motion_file, args.reference_file, "drift_drl_traj4")
    reference = arrays(args.reference_file)
    motion = arrays(args.motion_file)
    rt = np.asarray(reference["time_s"], dtype=float)
    rp = np.asarray(reference["root_target_position_m"][:, :2], dtype=float)
    validate_path(rt, rp, "reference")
    bodies = tuple(motion["body_names"].tolist())
    base = bodies.index("base_Link")
    reference_speed = np.linalg.norm(motion["body_lin_vel_w"][:, base, :2], axis=-1)
    core = tuple(spec["evaluation_core_interval_s"])
    sources = {
        "PPO": {"Isaac": (args.ppo_dir / "isaac", "physics_trajectory.npz", "isaac"),
                "MuJoCo": (args.ppo_dir / "mujoco", "rollout.npz", "mujoco")},
        "CEM NMPC": {"Isaac": (args.cem_isaac_dir, "rollout.npz", "nmpc"),
                     "MuJoCo": (args.cem_mujoco_dir, "rollout.npz", "nmpc")},
    }
    metrics, paths, audits = {}, {}, {}
    for controller, engines in sources.items():
        metrics[controller], paths[controller], audits[controller] = {}, {}, {}
        for engine, (directory, filename, kind) in engines.items():
            report_file, trace_file = directory / "report.json", directory / filename
            report, data = json.loads(report_file.read_text()), arrays(trace_file)
            if controller == "PPO":
                require(report.get("full_horizon_diagnostic") is True,
                        f"{engine} PPO must record a full-horizon diagnostic")
                require(report["motion_file_sha256" if engine == "Isaac" else "motion_sha256"] == digest(args.motion_file),
                        f"{engine} PPO motion differs from the locked reference")
                profile = report["task_profile"] if engine == "Isaac" else report["contract"]["trained_task_profile"]
                require(profile["reward_weights"]["sideslip"] == 0.0,
                        f"{engine} PPO sideslip reward is not zero")
                material = report["contact_material_audit"]
                coeff = material["robot_runtime_friction_range"] if engine == "Isaac" else material["compiled_friction_range"]
                require(np.allclose(coeff, [.7, .7], atol=1e-6), f"{engine} PPO friction readback differs from 0.7")
                if engine == "Isaac":
                    require(np.isclose(report["evaluation_contact_friction_override"]["evaluation_contact_friction"], .7)
                            and material["ground_colliders"]
                            and all(np.allclose((ground["static_friction"], ground["dynamic_friction"]), [.7, .7], atol=1e-6)
                                    for ground in material["ground_colliders"]),
                            "Isaac PPO ground material differs from 0.7")
                    require(report["physics_trajectory_sha256"] == digest(trace_file), "Isaac PPO trace hash mismatch")
                else:
                    isaac_report = json.loads((args.ppo_dir / "isaac/report.json").read_text())
                    require(report["contract"]["checkpoint_sha256"] == isaac_report["checkpoint_sha256"],
                            "PPO actor checkpoint differs between engines")
                t, xy = observed_episode(data, engine.lower())
                if engine == "Isaac":
                    valid = data["valid_mask"][:, 0]
                    velocity = data["root_linear_velocity_w_m_s"][:, 0, :2][valid]
                else:
                    velocity = data["root_com_lin_vel_w"][:, :2]
            else:
                require(report.get("status") == "completed" and report["motion_sha256"] == digest(args.motion_file),
                        f"{engine} CEM run is incomplete or uses another motion")
                require(report["objective_contract"]["weights"]["drift_sideslip"] == 0.0,
                        f"{engine} CEM sideslip reward is not zero")
                require(np.isclose(report["plant_mu"], .7) and np.allclose(report["prediction_mu_scenarios"], [.7]),
                        f"{engine} CEM friction differs from 0.7")
                if engine == "Isaac":
                    material = report["material_audit"]
                    require(np.isclose(material["expected_mu"], .7)
                            and np.allclose(material["robot_static_dynamic_range"], [.7, .7], atol=1e-6)
                            and material["ground_bound_materials"]
                            and all(np.allclose(ground["coefficients"][:2], [.7, .7], atol=1e-6)
                                    for ground in material["ground_bound_materials"]),
                            "Isaac CEM material readback differs from 0.7")
                else:
                    require(np.allclose(report["plant_material_readback"]["compiled_sliding_friction_range"], [.7, .7]),
                            "MuJoCo CEM material differs from 0.7")
                require(report["rollout_sha256"] == digest(trace_file), f"{engine} CEM trace hash mismatch")
                t, xy = observed_episode(data, "mujoco")
                velocity = data["base_com_velocity_w"][:, :2]
            require(velocity.shape == (len(t), 2) and np.isfinite(velocity).all(),
                    f"{controller}/{engine} has invalid COM XY velocity")
            error = path_error(t, xy, rt, rp, core)
            require(error["core_complete"], f"{controller}/{engine} did not cover the turn core")
            selected = (t >= core[0] - 1e-9) & (t <= core[1] + 1e-9)
            actual_speed = np.linalg.norm(velocity[selected], axis=-1)
            target_speed = np.interp(t[selected], rt, reference_speed)
            metrics[controller][engine] = {
                "core_xy_rmse_m": error["full_core_reference_xy_rmse_m"],
                "core_xy_peak_m": error["full_core_reference_xy_peak_m"],
                "core_speed_retention_ratio": float(actual_speed.mean() / target_speed.mean()),
                "core_samples": int(selected.sum()),
                "full_horizon_end_s": float(t[-1]),
            }
            paths[controller][engine] = (t, xy)
            audits[controller][engine] = {"report_sha256": digest(report_file), "trace_sha256": digest(trace_file)}
        it, ip = paths[controller]["Isaac"]
        mt, mp = paths[controller]["MuJoCo"]
        metrics[controller]["sim2sim"] = pair_error(it, ip, mt, mp, core)
    return (
        {"reference_id": spec["reference_id"], "core_window_s": list(core),
         "fixed_ground_mu": .7, "reference_sha256": digest(args.reference_file),
         "motion_sha256": digest(args.motion_file), "metrics": metrics,
         "artifact_audit": audits,
         "scope": "One seed, one fixed-friction episode per controller and engine; no statistical ranking or measured tire-slip claim"},
        {"reference": (rt, rp), "controllers": paths, "core": core},
    )


def render(traces: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True, constrained_layout=True)
    rt, rp = traces["reference"]
    core = (rt >= traces["core"][0] - 1e-9) & (rt <= traces["core"][1] + 1e-9)
    for ax, engine in zip(axes, ("Isaac", "MuJoCo")):
        ax.plot(rp[:, 0], rp[:, 1], color="0.45", lw=1.4, ls="--", label="Reference")
        ax.plot(rp[core, 0], rp[core, 1], color="black", lw=2.4, label="Turn core")
        for controller, color in (("PPO", "#1675b9"), ("CEM NMPC", "#db6b23")):
            _, xy = traces["controllers"][controller][engine]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.8, label=controller)
        ax.set(title=f"{engine} · μ=0.7", xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        ax.grid(alpha=.25)
    axes[1].legend(loc="best", frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-file", "reference-file", "ppo-dir", "cem-isaac-dir", "cem-mujoco-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Preserving nonempty output directory")
    summary, traces = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(traces, args.output_dir / "topdown.png")
    print(json.dumps({"status": "analyzed", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
