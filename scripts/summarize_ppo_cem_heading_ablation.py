"""Audit saved nominal PPO/CEM β-reward pairs and draw measured heading geometry.

This is read-only with respect to physical episodes: FK reconstructs wheel-link
poses from recorded states, without stepping either simulator. Wheel-position
relaxation is not simulated or inferred from these baseline trajectories.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco
import numpy as np

from controller_metrics import axial, wrap
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import quaternion_inverse, quaternion_rotate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULTS = ROOT / "results"
MOTION = ROOT.parent / "tron1-learning-from-human/results/2026-10-14-car-drift/matched-friction-mu0p70/motion.npz"
MODEL = OUT / "car_failure_sideslip_20261015/raw/mujoco_model.xml"
CORE = (1.44, 4.0)

SOURCES = {
    ("PPO", "on", "Isaac"): (OUT / "controller_primary_ppo_mu070_20261015_v1/mu0p700/isaac", "physics_trajectory.npz", "policy_isaac"),
    ("PPO", "on", "MuJoCo"): (OUT / "controller_primary_ppo_mu070_20261015_v1/mu0p700/mujoco", "rollout.npz", "policy_mujoco"),
    ("CEM", "on", "Isaac"): (OUT / "nmpc_isaac_primary_nominal_residual_h30_b10_20261016_v1", "rollout.npz", "nmpc"),
    ("CEM", "on", "MuJoCo"): (OUT / "nmpc_primary_nominal_residual_h30_b10_20261016_v1", "rollout.npz", "nmpc"),
    ("PPO", "off", "Isaac"): (OUT / "no_sideslip_ppo_mu0p7_validation_20261016_v1/isaac", "physics_trajectory.npz", "policy_isaac"),
    ("PPO", "off", "MuJoCo"): (OUT / "no_sideslip_ppo_mu0p7_validation_20261016_v1/mujoco", "rollout.npz", "policy_mujoco"),
    ("CEM", "off", "Isaac"): (OUT / "no_sideslip_nmpc_isaac_mu0p7_20261016_v1", "rollout.npz", "nmpc"),
    ("CEM", "off", "MuJoCo"): (OUT / "no_sideslip_nmpc_mu0p7_full_20261016_v1", "rollout.npz", "nmpc"),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def published(key: tuple[str, str, str]) -> tuple[dict, dict]:
    controller, side, engine = key
    if side == "on":
        name = ("controller_primary_ppo_nominal_pair_v1" if controller == "PPO" else
                "controller_primary_nmpc_residual_h30_b10_nominal_pair_v1")
        data = json.loads((RESULTS / name / "summary.json").read_text())
        metrics = data["windows"]["core"][engine.lower()]
        hashes = {"report_sha256": data["input_sha256"][engine.lower() + "_report"],
                  "trace_sha256": data["input_sha256"][engine.lower() + "_trace"]}
        require(data["input_sha256"]["model"] == digest(MODEL), "Published FK model changed")
    else:
        data = json.loads((RESULTS / "no_sideslip_reward_mu0p7_v1/summary.json").read_text())
        label = "PPO" if controller == "PPO" else "CEM NMPC"
        metrics, hashes = data["metrics"][label][engine], data["artifact_audit"][label][engine]
    return metrics, hashes


def reference_arrays() -> dict:
    with np.load(MOTION, allow_pickle=False) as z:
        names = tuple(z["body_names"].tolist())
        require(names[0] == "base_Link" and "wheel_L_Link" in names and "wheel_R_Link" in names,
                "Unexpected motion body ordering")
        return {"time": z["time_s"].copy(), "position": z["body_pos_w"].copy(),
                "quaternion": z["body_quat_w"].copy(), "velocity": z["body_lin_vel_w"].copy(),
                "body_names": names}


def core_metrics(trace: dict, signals: dict, raw: dict, reference: dict, published_metrics: dict) -> dict:
    t, pos = trace["time_s"], trace["position"]
    selected = (t >= CORE[0] - 1e-9) & (t <= CORE[1] + 1e-9)
    require(len(t) == 1397 and selected.sum() == 513 and np.isclose(t[-1], 6.98, atol=1e-8),
            "Physical episode or turn core incomplete")
    base = reference["body_names"].index("base_Link")
    wheel_ids = [reference["body_names"].index(name) for name in ("wheel_L_Link", "wheel_R_Link")]
    rt, rp = reference["time"], reference["position"][:, base, :2]
    target_xy = np.column_stack([np.interp(t, rt, rp[:, j]) for j in range(2)])
    xy_error = np.linalg.norm(pos[:, :2] - target_xy, axis=1)
    rmse = float(np.sqrt(np.mean(xy_error[selected] ** 2)))
    peak = float(xy_error[selected].max())
    expected_rmse = published_metrics.get("core_xy_rmse_m", published_metrics.get("reference_xy_rmse_m"))
    expected_peak = published_metrics.get("core_xy_peak_m", published_metrics.get("reference_xy_peak_m"))
    require(np.isclose(rmse, expected_rmse, atol=5e-5) and np.isclose(peak, expected_peak, atol=5e-5),
            "Raw XY metrics disagree with published nominal comparison")
    raw_index = np.asarray(raw["reference_frame_for_action"])
    if raw_index.shape == (349,) and "reference_frame" in raw:
        # Legacy PPO MuJoCo records action-held indices at policy rate and
        # the same held phase at every native physics sample separately.
        require(np.array_equal(raw_index, np.arange(349)),
                "Unexpected PPO policy-rate reference phase sequence")
        raw_index = np.asarray(raw["reference_frame"])
    if raw_index.ndim == 2:
        require(raw_index.shape == (len(t), 1), "Unexpected multi-environment reference indices")
        raw_index = raw_index[:, 0]
    indices = raw_index.astype(int)
    require(indices.shape == t.shape and np.all((0 <= indices) & (indices < len(rt))),
            "Invalid held reference indices")
    ref_q = reference["quaternion"][indices, base]
    ref_vel = reference["velocity"][indices, base]
    actual_b = quaternion_rotate(quaternion_inverse(trace["quaternion"]), trace["com_velocity"])
    target_b = quaternion_rotate(quaternion_inverse(ref_q), ref_vel)
    actual_beta = np.arctan2(actual_b[:, 1], actual_b[:, 0])
    target_beta = np.arctan2(target_b[:, 1], target_b[:, 0])
    eligible = selected & (signals["speed_m_s"] > .25) & (np.linalg.norm(ref_vel[:, :2], axis=1) > .3)
    beta_error = wrap(actual_beta - target_beta)
    ref_forward = quaternion_rotate(ref_q, np.array([1., 0., 0.]))
    ref_yaw = np.arctan2(ref_forward[:, 1], ref_forward[:, 0])
    delta_yaw = wrap(signals["body_heading_rad"] - ref_yaw)
    reference_wheel_delta = reference["position"][indices][:, wheel_ids, :2] - reference["position"][indices, base, None, :2]
    c, s = np.cos(delta_yaw)[:, None], np.sin(delta_yaw)[:, None]
    target_delta = np.stack((c * reference_wheel_delta[:, :, 0] - s * reference_wheel_delta[:, :, 1],
                             s * reference_wheel_delta[:, :, 0] + c * reference_wheel_delta[:, :, 1]), axis=-1)
    relative_wheel_error = trace["wheel_pos"][:, :, :2] - (pos[:, None, :2] + target_delta)
    require(relative_wheel_error.shape == (len(t), 2, 2), "Relative wheel target shape changed")
    ground = (trace["contact_count"] > 0) & (trace["contact_force"][:, :, 2] > 5.)
    gamma = signals["root_course_to_wheel_gamma_rad"]
    body_wheel = axial(signals["body_heading_rad"][:, None] - signals["wheel_rolling_plane_heading_rad"])
    contact_valid = ground & signals["root_course_to_wheel_valid"] & selected[:, None]
    wheel_center_valid = ground & signals["wheel_center_course_to_wheel_valid"] & selected[:, None]
    geometry = {}
    for j, label in enumerate(("left", "right")):
        mask = contact_valid[:, j]
        center_mask = wheel_center_valid[:, j]
        geometry[label] = {
            "ground_valid_samples": int(mask.sum()),
            "mean_abs_course_to_wheel_deg": float(np.degrees(np.abs(gamma[mask, j])).mean()) if mask.any() else None,
            "wheel_center_ground_valid_samples": int(center_mask.sum()),
            "mean_abs_wheel_center_course_to_wheel_deg": (
                float(np.degrees(np.abs(signals["wheel_center_course_to_wheel_gamma_rad"][center_mask, j])).mean())
                if center_mask.any() else None),
            "mean_abs_body_to_wheel_deg": float(np.degrees(np.abs(body_wheel[mask, j])).mean()) if mask.any() else None,
            "relative_wheel_xy_rmse_m": float(np.sqrt(np.mean(np.sum(relative_wheel_error[selected, j] ** 2, axis=1)))),
        }
    return {"core_xy_rmse_m": rmse, "core_xy_peak_m": peak,
            "core_speed_retention_ratio": float(published_metrics.get("core_speed_retention_ratio",
                                                              published_metrics.get("mean_speed_retention_ratio"))),
            "beta_reference_error_mean_abs_deg": float(np.degrees(np.abs(beta_error[eligible])).mean()) if eligible.any() else None,
            "beta_reward_eligible_samples": int(eligible.sum()), "wheels": geometry,
            "wheel_xy_relaxation_effect_measured": False}


def analyze() -> tuple[dict, dict]:
    reference = reference_arrays()
    require(digest(MOTION) ==
            "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254",
            "Locked motion changed")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    fk = WheelKinematics(model)
    rows, traces = {}, {}
    for side in ("on", "off"):
        rows[side], traces[side] = {}, {}
        for controller in ("PPO", "CEM"):
            rows[side][controller], traces[side][controller] = {}, {}
            for engine in ("Isaac", "MuJoCo"):
                key = (controller, side, engine)
                directory, filename, kind = SOURCES[key]
                report_path, trace_path = directory / "report.json", directory / filename
                report, expected = published(key)
                require(digest(trace_path) == expected["trace_sha256"] and
                        digest(report_path) == expected["report_sha256"],
                        f"Published file identities changed: {key}")
                runtime_report = json.loads(report_path.read_text())
                if controller == "PPO":
                    profile = (runtime_report["task_profile"] if engine == "Isaac" else
                               runtime_report["contract"]["trained_task_profile"])
                    actual_weight = profile["reward_weights"]["sideslip"]
                else:
                    actual_weight = runtime_report["objective_contract"]["weights"]["drift_sideslip"]
                require(np.isclose(actual_weight, 2. if side == "on" else 0.),
                        f"Sideslip reward profile mismatch: {key}")
                with np.load(trace_path, allow_pickle=False) as z:
                    raw = {name: z[name].copy() for name in z.files}
                trace = native_trace(raw, kind)
                reconstruction = fk.reconstruct(trace)
                signals = reconstruction["signals"]
                metrics = core_metrics(trace, signals, raw, reference, report)
                metrics["fk_audit"] = reconstruction["audit"]
                metrics["source_trace_sha256"] = expected["trace_sha256"]
                rows[side][controller][engine] = metrics
                traces[side][controller][engine] = {"trace": trace, "signals": signals,
                                                     "wheel_quaternion": reconstruction["wheel_quaternion_wxyz"]}
    summary = {"reference_id": "drift_drl_traj4", "ground_mu": .7, "core_window_s": list(CORE),
            "wheel_heading_definition": "world(local +Y axle) crossed with world +Z, axial modulo 180 degrees",
            "body_heading_definition": "world-projected local +X of base_Link",
            "velocity_course_definition": "world XY direction of base_Link original COM velocity",
            "wheel_relative_xy_definition": "held-reference wheel origin rotated by actual-minus-reference base yaw, relative to actual base origin",
            "rows": rows, "wheel_xy_relaxation_effect_measured": False,
            "limitations": ["One seed and one nominal friction coefficient per condition; no success-rate inference.",
                            "Removing β retrained PPO; CEM objective changed without training. The changes are not identical interventions.",
                            "Wheel XY tracking is present in every source run; its removal needs new PPO training and CEM closed-loop evaluation.",
                            "Wheel rolling-plane heading is geometric, not measured tire contact slip."]}
    return summary, {"reference": reference, "cases": traces}


def render(bundle: dict, summary: dict, path: Path) -> None:
    reference, cases = bundle["reference"], bundle["cases"]
    body = reference["body_names"].index("base_Link")
    rt = reference["time"]
    ref_xy = reference["position"][:, body, :2]
    selected_ref = rt <= 4.96 + 1e-9
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True, sharex=True, sharey=True)
    colors = {"PPO": "#2675b8", "CEM": "#dc7229"}
    snapshot_times = (1.6, 2.2, 2.8, 3.4, 4.0)
    for row, side in enumerate(("on", "off")):
        for col, (controller, engine) in enumerate((("PPO", "Isaac"), ("PPO", "MuJoCo"),
                                                     ("CEM", "Isaac"), ("CEM", "MuJoCo"))):
            ax = axes[row, col]
            item = cases[side][controller][engine]
            trace, signals = item["trace"], item["signals"]
            t, pos, wheel = trace["time_s"], trace["position"][:, :2], trace["wheel_pos"][:, :, :2]
            core = (t >= CORE[0] - 1e-9) & (t <= CORE[1] + 1e-9)
            visible = t <= 4.96 + 1e-9
            ax.plot(ref_xy[selected_ref, 0], ref_xy[selected_ref, 1], color=".35", ls="--", lw=1.15)
            ax.plot(pos[visible, 0], pos[visible, 1], color=colors[controller], lw=1.0, alpha=.45)
            ax.plot(pos[core, 0], pos[core, 1], color=colors[controller], lw=2.0)
            for sample_time in snapshot_times:
                k = int(np.argmin(np.abs(t - sample_time)))
                x, y = pos[k]
                theta = signals["body_heading_rad"][k]
                course = signals["course_rad"][k]
                ax.arrow(x, y, .18*np.cos(theta), .18*np.sin(theta), color="black",
                         width=.004, head_width=.045, length_includes_head=True, zorder=6)
                if signals["root_beta_valid"][k]:
                    ax.arrow(x, y, .16*np.cos(course), .16*np.sin(course), color="#239268",
                             width=.003, head_width=.035, length_includes_head=True, zorder=5)
                for j, wheel_color in enumerate(("#d44842", "#8953a8")):
                    angle = signals["wheel_rolling_plane_heading_rad"][k, j]
                    d = .085*np.array([np.cos(angle), np.sin(angle)])
                    grounded = (trace["contact_count"][k, j] > 0 and trace["contact_force"][k, j, 2] > 5.)
                    center = wheel[k, j]
                    ax.plot(center[0] + np.array([-d[0], d[0]]), center[1] + np.array([-d[1], d[1]]),
                            color=wheel_color, lw=2.1, alpha=1. if grounded else .25, zorder=7)
            m = summary["rows"][side][controller][engine]
            ax.set(title=f"{controller} · {engine} · β {'on' if side == 'on' else 'off'}\n"
                         f"XY RMSE {m['core_xy_rmse_m']:.3f} m · speed {100*m['core_speed_retention_ratio']:.0f}%",
                   xlabel="World X (m)", ylabel="World Y (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.22)
    fig.legend(handles=[Line2D([], [], color=".35", ls="--", label="Retargeted reference"),
                        Line2D([], [], color="black", marker=">", label="Base heading +X"),
                        Line2D([], [], color="#239268", marker=">", label="Base COM course"),
                        Line2D([], [], color="#d44842", lw=2, label="Left wheel rolling plane"),
                        Line2D([], [], color="#8953a8", lw=2, label="Right wheel rolling plane"),
                        Line2D([], [], color=".6", alpha=.4, lw=2, label="Faint wheel tick: not measured grounded")],
               loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("TRON1 traj_4 · μ=0.70 · measured heading snapshots at 1.6–4.0 s", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh-derived", action="store_true",
                        help="Replace only this script's two generated files")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        present = {p.name for p in args.output_dir.iterdir()}
        if not args.refresh_derived or present != {"summary.json", "heading_topdown.png"}:
            parser.error("Preserving nonempty or unexpected output directory")
    summary, bundle = analyze()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(bundle, summary, args.output_dir / "heading_topdown.png")
    print(json.dumps({"status": "analyzed", "conditions": 8,
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
