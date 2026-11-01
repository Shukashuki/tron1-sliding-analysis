"""Audit fixed-controller no-sideslip-reward friction pilot and render XY results.

Only reads recorded trajectories. It does not train, simulate, retune or infer tire slip.
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


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "config/no_sideslip_friction_pilot.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def case_name(mu: float) -> str:
    return f"mu{mu:.3f}".replace(".", "p")


def directories(args, mu: float) -> dict[str, dict[str, Path]]:
    if mu == .7:
        return {
            "PPO": {"Isaac": args.ppo_nominal_dir / "isaac", "MuJoCo": args.ppo_nominal_dir / "mujoco"},
            "CEM NMPC": {"Isaac": args.cem_isaac_nominal_dir, "MuJoCo": args.cem_mujoco_nominal_dir},
        }
    case = case_name(mu)
    return {
        "PPO": {"Isaac": args.ppo_pilot_dir / case / "isaac", "MuJoCo": args.ppo_pilot_dir / case / "mujoco"},
        "CEM NMPC": {"Isaac": args.cem_isaac_pilot_dir / case / "isaac",
                     "MuJoCo": args.cem_mujoco_pilot_dir / case / "mujoco"},
    }


def audit_and_measure(directory: Path, controller: str, engine: str, mu: float,
                      motion_sha: str, rt: np.ndarray, rp: np.ndarray,
                      reference_speed: np.ndarray, core: tuple[float, float]) -> tuple[dict, tuple[np.ndarray, np.ndarray], dict]:
    report_file = directory / "report.json"
    trace_file = directory / ("physics_trajectory.npz" if controller == "PPO" and engine == "Isaac" else "rollout.npz")
    report = json.loads(report_file.read_text())
    data = arrays(trace_file)
    if controller == "PPO":
        require(report.get("full_horizon_diagnostic") is True, f"{directory}: PPO is not full horizon")
        require(report["motion_file_sha256" if engine == "Isaac" else "motion_sha256"] == motion_sha,
                f"{directory}: PPO motion differs")
        profile = report["task_profile"] if engine == "Isaac" else report["contract"]["trained_task_profile"]
        require(profile["reward_weights"]["sideslip"] == 0., f"{directory}: PPO sideslip reward is nonzero")
        material = report["contact_material_audit"]
        if engine == "Isaac":
            require(np.isclose(report["evaluation_contact_friction_override"]["evaluation_contact_friction"], mu)
                    and np.allclose(material["robot_runtime_friction_range"], [mu, mu], atol=1e-6)
                    and material["ground_colliders"]
                    and all(np.allclose((g["static_friction"], g["dynamic_friction"]), [mu, mu], atol=1e-6)
                            for g in material["ground_colliders"]), f"{directory}: Isaac PPO material mismatch")
            require(report["physics_trajectory_sha256"] == sha(trace_file), f"{directory}: PPO trace hash mismatch")
            valid = np.asarray(data["valid_mask"][:, 0], dtype=bool)
            velocity = data["root_linear_velocity_w_m_s"][:, 0, :2][valid]
            policy_hash = report["checkpoint_sha256"]
        else:
            require(np.allclose(material["compiled_friction_range"], [mu, mu], atol=1e-6),
                    f"{directory}: MuJoCo PPO material mismatch")
            velocity = data["root_com_lin_vel_w"][:, :2]
            policy_hash = report["contract"]["checkpoint_sha256"]
        time, xy = observed_episode(data, engine.lower())
        identity = {"checkpoint_sha256": policy_hash, "task_profile_sha256": json_sha(profile)}
    else:
        require(report.get("status") == "completed" and report["motion_sha256"] == motion_sha,
                f"{directory}: CEM incomplete or motion differs")
        require(report["objective_contract"]["weights"]["drift_sideslip"] == 0.,
                f"{directory}: CEM sideslip reward is nonzero")
        require(report["prediction_backend"] == "native"
                and report.get("model_generated_nominal_plan") == {"enabled": False}
                and report.get("adaptation_contract") is None,
                f"{directory}: CEM controller is not the frozen nominal variant")
        require(np.isclose(report["plant_mu"], mu) and np.allclose(report["prediction_mu_scenarios"], [.7]),
                f"{directory}: CEM plant/prediction friction mismatch")
        require(report["rollout_sha256"] == sha(trace_file) and report["completed_policy_steps"] == 349,
                f"{directory}: CEM trace hash or duration mismatch")
        if engine == "Isaac":
            material = report["material_audit"]
            require(np.isclose(material["expected_mu"], mu)
                    and np.allclose(material["robot_static_dynamic_range"], [mu, mu], atol=1e-6)
                    and material["ground_bound_materials"]
                    and all(np.allclose(g["coefficients"][:2], [mu, mu], atol=1e-6)
                            for g in material["ground_bound_materials"]), f"{directory}: Isaac CEM material mismatch")
        else:
            require(np.allclose(report["plant_material_readback"]["compiled_sliding_friction_range"], [mu, mu], atol=1e-6),
                    f"{directory}: MuJoCo CEM material mismatch")
        time, xy = observed_episode(data, "mujoco")
        velocity = data["base_com_velocity_w"][:, :2]
        identity = {"effective_cem_config": report["effective_cem_config"],
                    "objective_contract_sha256": json_sha(report["objective_contract"]),
                    "model_sha256": report["mujoco_model_sha256" if engine == "Isaac" else "model_sha256"],
                    "solver_failures": report["solver_failures"]}
    require(velocity.shape == (len(time), 2) and np.isfinite(velocity).all(),
            f"{directory}: invalid base COM velocity")
    error = path_error(time, xy, rt, rp, core)
    require(error["core_complete"] and time[-1] >= rt[-1] - 1e-9,
            f"{directory}: physical trajectory does not cover full reference")
    selected = (time >= core[0] - 1e-9) & (time <= core[1] + 1e-9)
    speed = np.linalg.norm(velocity[selected], axis=-1)
    target_speed = np.interp(time[selected], rt, reference_speed)
    result = {"core_xy_rmse_m": error["full_core_reference_xy_rmse_m"],
              "core_xy_peak_m": error["full_core_reference_xy_peak_m"],
              "core_speed_retention_ratio": float(speed.mean() / target_speed.mean()),
              "large_xy_screen_exceeded": bool(error["full_core_reference_xy_rmse_m"] > .15
                                               or error["full_core_reference_xy_peak_m"] > .30),
              "core_samples": int(selected.sum()), "full_horizon_end_s": float(time[-1])}
    audit = {"report_sha256": sha(report_file), "trace_sha256": sha(trace_file), **identity}
    return result, (time, xy), audit


def analyze(args) -> tuple[dict, dict]:
    config = json.loads(PILOT.read_text())
    values = config["validation_ground_mu_equal_static_dynamic"]
    require(values == [.05, .1, .2, .3, .7, 1.6] and config["nmpc_prediction_mu_scenarios"] == [.7],
            "Pilot config differs from the predeclared grid/fixed predictor")
    profile = json.loads((ROOT / config["task_profile"]).read_text())
    require(profile["reward_weights"]["sideslip"] == 0., "Pilot profile still rewards sideslip")
    spec = verify_reference_pair(args.motion_file, args.reference_file, config["reference_id"])
    ref, motion = arrays(args.reference_file), arrays(args.motion_file)
    rt, rp = ref["time_s"], ref["root_target_position_m"][:, :2]
    validate_path(rt, rp, "reference")
    bodies = tuple(motion["body_names"].tolist())
    reference_speed = np.linalg.norm(motion["body_lin_vel_w"][:, bodies.index("base_Link"), :2], axis=-1)
    core, motion_hash = tuple(spec["evaluation_core_interval_s"]), sha(args.motion_file)
    rows, paths, audits = [], {}, {}
    ppo_hashes, profile_hashes, cem_configs, objective_hashes, model_hashes = set(), set(), set(), set(), set()
    for mu in values:
        name = case_name(mu)
        row, paths[name], audits[name] = {"mu": mu, "case": name}, {}, {}
        for controller, engine_dirs in directories(args, mu).items():
            row[controller], paths[name][controller], audits[name][controller] = {}, {}, {}
            for engine, directory in engine_dirs.items():
                metric, path, audit = audit_and_measure(directory, controller, engine, mu,
                                                        motion_hash, rt, rp, reference_speed, core)
                row[controller][engine], paths[name][controller][engine] = metric, path
                audits[name][controller][engine] = audit
                if controller == "PPO":
                    ppo_hashes.add(audit["checkpoint_sha256"])
                    profile_hashes.add(audit["task_profile_sha256"])
                else:
                    cem_configs.add(json.dumps(audit["effective_cem_config"], sort_keys=True))
                    objective_hashes.add(audit["objective_contract_sha256"])
                    model_hashes.add(audit["model_sha256"])
            it, ip = paths[name][controller]["Isaac"]
            mt, mp = paths[name][controller]["MuJoCo"]
            row[controller]["sim2sim"] = pair_error(it, ip, mt, mp, core)
        rows.append(row)
    require(all(len(group) == 1 for group in (ppo_hashes, profile_hashes, cem_configs, objective_hashes, model_hashes)),
            "Controller/checkpoint/profile/config/objective/model changed between friction values")
    require(profile_hashes == {json_sha(profile)}, "PPO validation profile differs from the pinned ablation")
    summary = {"kind": "no_sideslip_reward_fixed_controller_friction_pilot",
               "reference_id": spec["reference_id"], "core_window_s": list(core),
               "ground_mu_equal_static_dynamic": values,
               "nominal_mu_0p7_reused_from_completed_ablation": True,
               "cem_prediction_mu": .7, "policy_training_friction_randomization_unchanged": True,
               "pilot_config_sha256": sha(PILOT), "motion_sha256": motion_hash,
               "reference_sha256": sha(args.reference_file), "ppo_checkpoint_sha256": next(iter(ppo_hashes)),
               "rows": rows, "artifact_audit": audits,
               "limitations": ["One seed/episode per coefficient and engine; sampled crossings are not statistical thresholds.",
                               "Equal numeric PhysX/MuJoCo friction is not calibrated contact equivalence.",
                               "XY error is a tracking screen, not measured contact slip or a sideslip failure gate.",
                               "CEM solves are offline; no real-time deadline is enforced."]}
    return summary, {"reference": (rt, rp), "paths": paths, "core": core}


def render(summary: dict, trace: dict, directory: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    values = summary["ground_mu_equal_static_dynamic"]
    for controller, color in (("PPO", "#1675b9"), ("CEM NMPC", "#db6b23")):
        for engine, style in (("Isaac", "-"), ("MuJoCo", "--")):
            rmse = [row[controller][engine]["core_xy_rmse_m"] for row in summary["rows"]]
            speed = [row[controller][engine]["core_speed_retention_ratio"] for row in summary["rows"]]
            label = f"{controller} · {engine}"
            axes[0].plot(values, rmse, color=color, ls=style, marker="o", label=label)
            axes[1].plot(values, speed, color=color, ls=style, marker="o", label=label)
    axes[0].axhline(.15, color="0.4", ls=":", lw=1, label="XY RMSE screen")
    axes[0].set(xscale="log", yscale="log", xlabel="Ground friction μ", ylabel="Core XY RMSE (m)")
    axes[1].set(xscale="log", xlabel="Ground friction μ", ylabel="Core speed / reference speed")
    for ax in axes:
        ax.grid(alpha=.25, which="both")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(directory / "friction_metrics.png", dpi=180)
    plt.close(fig)

    selected = [.05, .2, 1.6]
    fig, axes = plt.subplots(3, 2, figsize=(9, 10), constrained_layout=True)
    rt, rp = trace["reference"]
    core = (rt >= trace["core"][0] - 1e-9) & (rt <= trace["core"][1] + 1e-9)
    for row_axes, mu in zip(axes, selected):
        case = case_name(mu)
        row_points = [rp]
        for ax, engine in zip(row_axes, ("Isaac", "MuJoCo")):
            ax.plot(rp[:, 0], rp[:, 1], color="0.5", lw=1, ls="--", label="Reference")
            ax.plot(rp[core, 0], rp[core, 1], color="black", lw=1.7, label="Turn core")
            for controller, color in (("PPO", "#1675b9"), ("CEM NMPC", "#db6b23")):
                _, xy = trace["paths"][case][controller][engine]
                row_points.append(xy)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.3, label=controller)
            ax.set(title=f"{engine} · μ={mu:g}", xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
            ax.grid(alpha=.2)
        all_points = np.concatenate(row_points)
        low, high = all_points.min(axis=0), all_points.max(axis=0)
        padding = np.maximum(.1, .05 * (high - low))
        for ax in row_axes:
            ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
            ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.savefig(directory / "topdown_selected.png", dpi=180)
    plt.close(fig)


def compact_xy(trace: dict, directory: Path) -> Path:
    rt, rp = trace["reference"]
    payload = {"reference_t": rt, "reference_xy": rp}
    for case, controllers in trace["paths"].items():
        for controller, engines in controllers.items():
            label = "ppo" if controller == "PPO" else "cem"
            for engine, (time, xy) in engines.items():
                prefix = f"{case}_{label}_{engine.lower()}"
                payload[prefix + "_t"] = time
                payload[prefix + "_xy"] = xy
    path = directory / "xy_trace.npz"
    np.savez_compressed(path, **payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-file", "reference-file", "ppo-pilot-dir", "ppo-nominal-dir",
                 "cem-isaac-pilot-dir", "cem-isaac-nominal-dir", "cem-mujoco-pilot-dir",
                 "cem-mujoco-nominal-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Preserving nonempty analysis output")
    summary, trace = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["compact_xy_trace_sha256"] = sha(compact_xy(trace, args.output_dir))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(summary, trace, args.output_dir)
    print(json.dumps({"status": "analyzed", "cases": len(summary["rows"]), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
