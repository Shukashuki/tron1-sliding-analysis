"""Summarize checked, actor-free request replay without claiming policy transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(data):
    time = np.asarray(data["time_s"])
    if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
            or not np.allclose(time, np.arange(len(time))*.0025, atol=1e-12, rtol=0)):
        raise ValueError("Require original, monotone 400 Hz times starting at zero")
    xy = {}
    for engine in ("physx", "mujoco"):
        xy[engine] = np.asarray(data[engine+"_root_position_m"])
        if xy[engine].shape != (len(time), 3) or not np.isfinite(xy[engine]).all():
            raise ValueError("Invalid original root positions")
    gap = np.linalg.norm(xy["mujoco"][:, :2]-xy["physx"][:, :2], axis=1)
    result = {"sample_count": len(time), "native_physics_dt_s": .0025,
        "same_time_xy_gap_first_observed_above_m": {
            str(limit): float(time[np.flatnonzero(gap > limit)[0]]) if np.any(gap > limit) else None
            for limit in (.01, .05, .1, .3, 1.)}, "windows": {}}
    for end in (.2, .5, 1., 2., 6.98):
        if time[-1] < end-1e-9:
            continue
        mask = time <= end+1e-9
        physical = mask & (time > 0.)
        a = np.asarray(data["physx_applied_joint_torque_nm"])[physical]
        b = np.asarray(data["mujoco_applied_joint_torque_nm"])[physical]
        if a.shape != b.shape or a.shape != (physical.sum(), 8) or not np.isfinite(a-b).all():
            raise ValueError("Invalid torque signal")
        window = {"interval_s": [0., end], "samples": int(mask.sum()),
            "post_action_force_torque_samples": int(physical.sum()),
            "initial_pre_action_sensor_row_excluded_from_force_torque_metrics": True,
            "interengine_xy_rmse_m": float(np.sqrt(np.mean(gap[mask]**2))),
            "interengine_xy_peak_m": float(gap[mask].max()),
            "leg_actual_torque_interengine_rms_nm": float(np.sqrt(np.mean((a[:, :6]-b[:, :6])**2)))}
        for engine in ("physx", "mujoco"):
            force = np.asarray(data[engine+"_wheel_ground_contact_force_w_n"])[physical]
            count = np.asarray(data[engine+"_wheel_ground_contact_count"])[physical]
            if (force.shape != (physical.sum(), 2, 3) or count.shape != (physical.sum(), 2)
                    or not np.isfinite(force).all() or not np.isfinite(count).all()
                    or not np.equal(count, np.floor(count)).all() or np.any(count < 0)):
                raise ValueError("Invalid original wheel contact arrays")
            bearing = (count > 0) & (force[..., 2] > 5.)
            window[engine] = {"both_wheels_load_eligible_fraction": float(bearing.all(axis=1).mean()),
                "neither_wheel_load_eligible_fraction": float((~bearing.any(axis=1)).mean())}
        result["windows"][str(end)] = window
    return result


def render(data, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = data["time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"physx": "#0072B2", "mujoco": "#D55E00"}
    for engine, color in colors.items():
        xy = data[engine+"_root_position_m"]
        axes[0, 0].plot(xy[:, 0], xy[:, 1], color=color, label=engine)
        axes[0, 0].plot(xy[0, 0], xy[0, 1], "o", color=color, ms=4)
        short = (t > 0.) & (t <= .2+1e-9)
        force = data[engine+"_wheel_ground_contact_force_w_n"][short, :, 2]
        for index, style in enumerate(("-", "--")):
            axes[1, 0].plot(t[short], force[:, index], ls=style, color=color,
                label=engine+(" left" if index == 0 else " right"), lw=.8)
    gap = np.linalg.norm(data["mujoco_root_position_m"][:, :2]-data["physx_root_position_m"][:, :2], axis=1)
    axes[0, 1].plot(t, gap, color="#333333")
    delta = data["mujoco_applied_joint_torque_nm"]-data["physx_applied_joint_torque_nm"]
    axes[1, 1].plot(t[short], np.sqrt(np.mean(delta[short, :6]**2, axis=1)), color="#333333")
    axes[0, 0].set(xlabel="World X [m]", ylabel="World Y [m]", title="Full open-loop path; NOT closed-loop policy transfer")
    axes[0, 0].axis("equal"); axes[0, 0].legend()
    axes[0, 1].set(xlabel="Native time [s]", ylabel="Same-time XY gap [m]", title="Divergence under held identical normalized requests")
    axes[1, 0].set(xlabel="Native time [s]", ylabel="Wheel normal force [N]", title="First0.2s post-action forces (pre-action row excluded)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(xlabel="Native time [s]", ylabel="Actual leg torque gap RMS [Nm]", title="PD reacts to each plant: actual torques are NOT identical")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle("Same observable initial state + same50Hz requests / native400Hz\nHidden solver/contact states and contact laws are NOT matched", fontsize=12)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def run(args):
    source = args.replay_dir.resolve()
    metadata = json.loads((source/"request_replay_diagnostic.json").read_text())
    arrays = source/"same_time_comparison.npz"
    if (metadata.get("actor_calls") != 0 or metadata.get("torch_imported") is not False
            or metadata.get("same_normalized_requests") is not True
            or metadata.get("requested_prefix_completed") is not True
            or sha256(arrays) != metadata["artifacts_sha256"][arrays.name]):
        raise ValueError("Replay provenance or original arrays differ")
    with np.load(arrays, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    summary = {"schema_version": 1, "kind": "open_loop_request_replay_summary",
        "source_replay_report_sha256": sha256(source/"request_replay_diagnostic.json"),
        "source_arrays_sha256": sha256(arrays), "analysis_script_sha256": sha256(__file__),
        **summarize(data), "policy_transfer_result": False, "hidden_state_equivalence_claimed": False,
        "actual_torque_equivalence_claimed": False, "zero_load_is_airborne_or_failure": False,
        "limitations": metadata["limitations"]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    render(data, args.output_dir/"backend_request_replay.png")
    summary["plot_sha256"] = sha256(args.output_dir/"backend_request_replay.png")
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
