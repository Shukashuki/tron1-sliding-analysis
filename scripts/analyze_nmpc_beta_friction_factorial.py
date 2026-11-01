"""Audit the CEM sideslip-reward × known-friction grid from physical traces.

No simulation, policy training, retuning, or sideslip-based failure gate occurs.
The scalar plant-friction oracle is privileged; μ=0.70 reuses fixed episodes.
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

from controller_references import verify_reference_pair
from nmpc_tracking_objective import quaternion_inverse, quaternion_rotate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONFIG = ROOT / "config/nmpc_beta_friction_factorial.json"
CONFIG_17 = ROOT / "config/nmpc_beta_friction_factorial_17.json"
SIX_MU = (.05, .1, .2, .3, .7, 1.6)
ENGINES = ("Isaac", "MuJoCo")
PROFILES = ("side", "no_side")
MODES = ("fixed", "oracle")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case_name(mu: float) -> str:
    return f"mu{mu:.3f}".replace(".", "p")


def directory(profile: str, mode: str, engine: str, mu: float) -> Path:
    case = case_name(mu)
    if mu not in SIX_MU and (mode == "oracle" or profile == "no_side"):
        root = ("nmpc_beta_friction_17_isaac_20261017_v1" if engine == "Isaac" else
                "nmpc_beta_friction_17_mujoco_20261017_v1")
        return OUT / root / profile / mode / case
    if mode == "oracle" and mu != .7:
        if engine == "Isaac":
            return OUT / "nmpc_oracle_isaac_grid_20261017_v1" / profile / case
        if profile == "side":
            return OUT / "sideslip_cem_oracle_friction_grid_20261017_v1" / case / "mujoco"
        if mu in (.1, .2):
            return OUT / "no_sideslip_cem_oracle_friction_pilot_20261016_v1" / case / "mujoco"
        return OUT / "no_sideslip_cem_oracle_friction_grid_20261017_v1" / case / "mujoco"
    if profile == "side" and engine == "Isaac":
        return OUT / "nmpc_primary_isaac_friction_development_20261016_v1" / case
    if profile == "side" and engine == "MuJoCo":
        if mu == .7:
            return OUT / "nmpc_primary_nominal_residual_h30_b10_20261016_v1"
        if mu == .3:
            return OUT / "fixed_mpc_primary_mu0p3_20261016_v1"
        return OUT / "nmpc_primary_mujoco_friction_development_20261016_v1" / case
    if profile == "no_side" and engine == "Isaac":
        return (OUT / "no_sideslip_nmpc_isaac_mu0p7_20261016_v1" if mu == .7 else
                OUT / "no_sideslip_cem_isaac_friction_pilot_20261016_v1" / case / "isaac")
    return (OUT / "no_sideslip_nmpc_mu0p7_full_20261016_v1" if mu == .7 else
            OUT / "no_sideslip_cem_friction_pilot_20261016_v1" / case / "mujoco")


def load_episode(path: Path, profile: str, mode: str, engine: str, mu: float,
                 motion_hash: str, reference: dict, weight: float) -> tuple[dict, dict, dict]:
    report_path, trace_path = path / "report.json", path / "rollout.npz"
    report = json.loads(report_path.read_text())
    require(report.get("status") == "completed" and report.get("truncated_smoke") is False,
            f"{path}: incomplete or smoke run")
    require(report["reference_id"] == "drift_drl_traj4" and report["motion_sha256"] == motion_hash,
            f"{path}: reference changed")
    require(report["completed_policy_steps"] == report["total_reference_policy_steps"] == 349,
            f"{path}: policy horizon incomplete")
    require(report["rollout_sha256"] == file_sha(trace_path), f"{path}: raw trace hash mismatch")
    require(report["prediction_backend"] == "native"
            and report["model_generated_nominal_plan"] == {"enabled": False}
            and report.get("adaptation_contract") is None, f"{path}: CEM variant changed")
    require(np.isclose(report["objective_contract"]["weights"]["drift_sideslip"], weight),
            f"{path}: sideslip reward weight mismatch")
    predictor_mu = .7 if mode == "fixed" or mu == .7 else mu
    require(np.isclose(report["plant_mu"], mu)
            and np.allclose(report["prediction_mu_scenarios"], [predictor_mu], atol=1e-10),
            f"{path}: plant or predictor μ mismatch")
    require(bool(report["prediction_receives_plant_mu"]) == (mode == "oracle" and mu != .7),
            f"{path}: oracle disclosure metadata mismatch")
    if mode == "oracle" and mu != .7:
        require(report.get("prediction_friction_source") == "plant_friction_oracle",
                f"{path}: oracle source not explicitly recorded")
    require(report["solver_failures"] == 0 and not report["pose_height_failure_gate"],
            f"{path}: solver failure or height gate")
    require(np.isclose(report["physics_dt_s"], .005) and np.isclose(report["control_dt_s"], .02),
            f"{path}: physics/controller clock changed")
    if engine == "Isaac":
        material = report["material_audit"]
        require(np.isclose(material["expected_mu"], mu)
                and np.allclose(material["robot_static_dynamic_range"], [mu, mu], atol=1e-6)
                and material["ground_bound_materials"]
                and all(np.allclose(item["coefficients"], [mu, mu, 0.], atol=1e-6)
                        for item in material["ground_bound_materials"]),
                f"{path}: PhysX material readback mismatch")
    else:
        require(np.allclose(report["plant_material_readback"]["compiled_sliding_friction_range"],
                            [mu, mu], atol=1e-9), f"{path}: MuJoCo material readback mismatch")
    with np.load(trace_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in
                ("time_s", "root_pos", "root_quaternion_wxyz", "base_com_velocity_w",
                 "reference_frame_for_action", "reference_xy_m")}
    time, xy = data["time_s"], data["root_pos"][:, :2]
    require(len(time) == 1397 and np.isclose(time[0], 0., atol=1e-8)
            and np.isclose(time[-1], 6.98, atol=1e-8)
            and np.allclose(np.diff(time), .005, atol=1e-8), f"{path}: physical clock incomplete")
    require(xy.shape == (len(time), 2) and data["reference_xy_m"].shape == xy.shape
            and np.isfinite(xy).all(), f"{path}: invalid XY trace")
    core = (time >= 1.44 - 1e-9) & (time <= 4.0 + 1e-9)
    error = np.linalg.norm(xy - data["reference_xy_m"], axis=1)
    rmse, peak = float(np.sqrt(np.mean(error[core] ** 2))), float(error[core].max())
    require(core.sum() == 513 and np.isclose(rmse, report["core_xy_rmse_m"], atol=1e-6)
            and np.isclose(peak, report["core_xy_peak_m"], atol=1e-6),
            f"{path}: report XY metrics do not reproduce")
    index = data["reference_frame_for_action"].astype(int)
    require(index.shape == time.shape and np.all((0 <= index) & (index < 349)),
            f"{path}: invalid held reference indices")
    base = reference["base_index"]
    velocity = data["base_com_velocity_w"]
    target_velocity = reference["body_lin_vel_w"][index, base]
    actual_body_velocity = quaternion_rotate(quaternion_inverse(data["root_quaternion_wxyz"]), velocity)
    target_body_velocity = quaternion_rotate(quaternion_inverse(reference["body_quat_w"][index, base]),
                                             target_velocity)
    beta = np.arctan2(actual_body_velocity[:, 1], actual_body_velocity[:, 0])
    target_beta = np.arctan2(target_body_velocity[:, 1], target_body_velocity[:, 0])
    beta_error = np.arctan2(np.sin(beta - target_beta), np.cos(beta - target_beta))
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    target_speed = np.linalg.norm(target_velocity[:, :2], axis=1)
    eligible = core & (speed > .25) & (target_speed > .3)
    reference_speed = np.linalg.norm(reference["body_lin_vel_w"][:, base, :2], axis=1)
    interpolated_reference_speed = np.interp(time[core], reference["time_s"], reference_speed)
    ratio = float(speed[core].mean() / interpolated_reference_speed.mean())
    require(np.isclose(ratio, report["core_speed_retention_mean_ratio"], atol=1e-6),
            f"{path}: reported speed retention does not reproduce")
    metrics = {"core_xy_rmse_m": rmse, "core_xy_peak_m": peak,
               "core_speed_retention_ratio": ratio,
               "large_xy_screen_exceeded": bool(rmse > .15 or peak > .30),
               "beta_error_mean_abs_deg_eligible": (float(np.rad2deg(np.abs(beta_error[eligible])).mean())
                                                    if eligible.any() else None),
               "beta_reward_eligible_samples": int(eligible.sum()),
               "core_samples": int(core.sum()), "full_horizon_end_s": float(time[-1])}
    identity = {"model_sha256": report["mujoco_model_sha256" if engine == "Isaac" else "model_sha256"],
                "effective_cem_config": report["effective_cem_config"],
                "action_parameterization": report["action_parameterization"],
                "objective_without_sideslip_weight": {
                    **report["objective_contract"],
                    "weights": {**report["objective_contract"]["weights"], "drift_sideslip": None}},
                "objective_source_sha256": next(value for key, value in report["source_sha256"].items()
                                                if key.endswith("/nmpc_tracking_objective.py"))}
    evidence = {"directory": str(path), "report_sha256": file_sha(report_path),
                "rollout_sha256": file_sha(trace_path), "plant_mu": mu,
                "prediction_mu": predictor_mu, "reused_fixed_as_oracle": mode == "oracle" and mu == .7}
    return metrics, {"time": time, "xy": xy, "reference_xy": data["reference_xy_m"]}, {**evidence, **identity}


def analyze(motion_file: Path, reference_file: Path, config_file: Path = CONFIG) -> tuple[dict, dict]:
    config = json.loads(config_file.read_text())
    values = config["ground_mu_equal_static_dynamic"]
    allowed = (list(SIX_MU), [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7,
                               .8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
    require(values in allowed, "Unexpected friction grid")
    spec = verify_reference_pair(motion_file, reference_file, config["reference_id"])
    require(spec["evaluation_core_interval_s"] == [1.44, 4.0], "Core window changed")
    with np.load(motion_file, allow_pickle=False) as archive:
        names = tuple(archive["body_names"].tolist())
        reference = {"base_index": names.index("base_Link"),
                     "time_s": archive["time_s"].copy(),
                     "body_lin_vel_w": archive["body_lin_vel_w"].copy(),
                     "body_quat_w": archive["body_quat_w"].copy()}
    motion_hash = file_sha(motion_file)
    rows, paths, audits, identities = [], {}, {}, []
    for mu in values:
        case = case_name(mu)
        row, paths[case], audits[case] = {"mu": mu, "case": case}, {}, {}
        for profile in PROFILES:
            row[profile], paths[case][profile], audits[case][profile] = {}, {}, {}
            weight = config["profiles"][profile]["sideslip_reward_weight"]
            for mode in MODES:
                row[profile][mode], paths[case][profile][mode], audits[case][profile][mode] = {}, {}, {}
                for engine in ENGINES:
                    path = directory(profile, mode, engine, mu)
                    metrics, trace, audit = load_episode(path, profile, mode, engine, mu,
                                                         motion_hash, reference, weight)
                    row[profile][mode][engine] = metrics
                    paths[case][profile][mode][engine] = trace
                    audits[case][profile][mode][engine] = {key: value for key, value in audit.items()
                                                           if key not in ("effective_cem_config", "action_parameterization",
                                                                          "objective_without_sideslip_weight")}
                    identities.append({key: audit[key] for key in ("model_sha256", "effective_cem_config",
                                                              "action_parameterization", "objective_without_sideslip_weight",
                                                              "objective_source_sha256")})
                isaac, mujoco = (paths[case][profile][mode][engine] for engine in ENGINES)
                selected = (isaac["time"] >= 1.44 - 1e-9) & (isaac["time"] <= 4.0 + 1e-9)
                require(np.isclose(mujoco["time"][-1], 6.98, atol=1e-8), "MuJoCo core incomplete")
                mx = np.column_stack([np.interp(isaac["time"][selected], mujoco["time"], mujoco["xy"][:, i])
                                      for i in range(2)])
                row[profile][mode]["sim2sim_core_xy_gap_m"] = float(np.sqrt(np.mean(np.sum(
                    (isaac["xy"][selected] - mx) ** 2, axis=1))))
        rows.append(row)
    common = identities[0]
    require(all(identity == common for identity in identities),
            "Model, search configuration, actuator parameterization, or non-sideslip objective changed")
    summary = {"kind": "nmpc_sideslip_reward_by_exact_friction_oracle_factorial",
               "reference_id": spec["reference_id"], "ground_mu_equal_static_dynamic": values,
               "core_window_s": [1.44, 4.0], "profile_weights": {name: config["profiles"][name]["sideslip_reward_weight"]
                                                         for name in PROFILES},
               "prediction_modes": config["prediction_modes"],
               "nominal_mu_reuses_fixed_as_oracle": True,
               "config_sha256": file_sha(config_file), "motion_sha256": motion_hash,
               "reference_sha256": file_sha(reference_file),
               "shared_model_sha256": common["model_sha256"],
               "shared_effective_cem_config": common["effective_cem_config"],
               "rows": rows, "artifact_audit": audits,
               "limitations": ["One episode per cell, not a statistical success rate.",
                               "An exact numeric ground coefficient is privileged information, not an estimated tire-contact state.",
                               "PhysX and MuJoCo numeric friction are not calibrated physical equivalence.",
                               "The XY screen is not a sideslip, height or flight failure gate.",
                               "CEM remains offline relative to the 20 ms control period."]}
    return summary, paths


def render(summary: dict, paths: dict, output: Path):
    values = summary["ground_mu_equal_static_dynamic"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {"side": "#206ea8", "no_side": "#df752c"}
    for column, engine in enumerate(ENGINES):
        for profile in PROFILES:
            for mode, style in (("fixed", "-"), ("oracle", "--")):
                label = f"{'β reward' if profile == 'side' else 'no β reward'} · {mode}"
                xy = [row[profile][mode][engine]["core_xy_rmse_m"] for row in summary["rows"]]
                speed = [row[profile][mode][engine]["core_speed_retention_ratio"] for row in summary["rows"]]
                axes[0, column].plot(values, xy, color=colors[profile], ls=style, marker="o", label=label)
                axes[1, column].plot(values, speed, color=colors[profile], ls=style, marker="o", label=label)
        axes[0, column].axhline(.15, color="0.45", ls=":", lw=1)
        axes[0, column].set(xscale="log", yscale="log", title=engine,
                            ylabel="Core XY RMSE (m)", xlabel="Ground friction μ")
        axes[1, column].set(xscale="log", ylabel="Core speed / reference speed", xlabel="Ground friction μ")
        for ax in axes[:, column]:
            ax.grid(alpha=.25, which="both")
    axes[0, 1].legend(fontsize=7, frameon=False)
    fig.savefig(output / "friction_factorial.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for row_axes, mu in zip(axes, (.1, .2)):
        case = case_name(mu)
        for ax, engine in zip(row_axes, ENGINES):
            example = paths[case]["side"]["fixed"][engine]
            ref = example["reference_xy"]
            ax.plot(ref[:, 0], ref[:, 1], color="black", ls="--", lw=1.3, label="Reference")
            for profile in PROFILES:
                for mode, style in (("fixed", "-"), ("oracle", "--")):
                    xy = paths[case][profile][mode][engine]["xy"]
                    ax.plot(xy[:, 0], xy[:, 1], color=colors[profile], ls=style, lw=1.25,
                            label=f"{profile} · {mode}")
                    end = np.argmin(np.abs(paths[case][profile][mode][engine]["time"] - 4.0))
                    ax.plot(xy[end, 0], xy[end, 1], marker="o", ms=4,
                            mfc=colors[profile], mec="white", mew=.5, ls="none")
            ax.set(title=f"{engine} · μ={mu:.2f}", xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
            ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=7, frameon=False)
    fig.savefig(output / "topdown_mu0p10_mu0p20.png", dpi=180)
    plt.close(fig)

    for mu in values:
        case = case_name(mu)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, engine in zip(axes, ENGINES):
            ref = paths[case]["side"]["fixed"][engine]["reference_xy"]
            ax.plot(ref[:, 0], ref[:, 1], color="black", ls="--", lw=1.3, label="Reference")
            for profile in PROFILES:
                for mode, style in (("fixed", "-"), ("oracle", "--")):
                    xy = paths[case][profile][mode][engine]["xy"]
                    ax.plot(xy[:, 0], xy[:, 1], color=colors[profile], ls=style, lw=1.2,
                            label=f"{profile} · {mode}")
                    end = np.argmin(np.abs(paths[case][profile][mode][engine]["time"] - 4.0))
                    ax.plot(xy[end, 0], xy[end, 1], marker="o", ms=4,
                            mfc=colors[profile], mec="white", mew=.5, ls="none")
            ax.set(title=f"{engine} · μ={mu:.2f}", xlabel="World X (m)", ylabel="World Y (m)",
                   aspect="equal")
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=8, frameon=False, loc="best")
        fig.savefig(output / f"topdown_{case}.png", dpi=180)
        plt.close(fig)


def compact_trace(paths: dict, output: Path):
    data = {}
    for case, profiles in paths.items():
        for profile, modes in profiles.items():
            for mode, engines in modes.items():
                for engine, trace in engines.items():
                    prefix = f"{case}_{profile}_{mode}_{engine.lower()}"
                    data[prefix + "_t"] = trace["time"]
                    data[prefix + "_xy"] = trace["xy"]
                    data[prefix + "_ref_xy"] = trace["reference_xy"]
    np.savez_compressed(output / "xy_trace.npz", **data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, paths = analyze(args.motion_file, args.reference_file, args.config_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(summary, paths, args.output_dir)
    compact_trace(paths, args.output_dir)
    print(json.dumps({"status": "analyzed", "cases": len(summary["rows"]),
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
