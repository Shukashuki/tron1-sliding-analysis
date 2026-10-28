"""Quantify posture fidelity separately from successful physical stopping.

Constant postures below are numerical reference-fit yardsticks, not simulated
controllers. No hindsight posture threshold changes the frozen stop gates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from evaluate_hockey_physx import load, sha, q_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run-dir", "qualification", "command-dir", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    qualified = json.loads((a.qualification/"report.json").read_text())
    trace_path = a.run_dir/"final_rollout.npz"
    if sha(trace_path) != qualified["input_sha256"]["final_rollout.npz"]:
        raise ValueError("Qualified physical trace changed")
    trace = load(trace_path); reference = load(a.command_dir/"motion.npz")
    command = json.loads((a.command_dir/"report.json").read_text())
    if sha(a.command_dir/"motion.npz") != command["motion_sha256"]:
        raise ValueError("Reference changed")
    rt = np.arange(len(reference["joint_pos"]))/float(reference["fps"])
    t = np.arange(len(trace["joint_pos"]))*float(trace["physics_dt_s"])
    valid = (t > 0)&(t <= rt[-1]+1e-9)
    brake = valid&(t >= command["brake_start_s"]-1e-9)&(t <= command["brake_end_s"]+1e-9)
    names = reference["joint_names"].astype(str).tolist()[:6]
    actual_ids = [trace["joint_names"].astype(str).tolist().index(name) for name in names]
    target = np.column_stack([np.interp(t, rt, reference["joint_pos"][:, k]) for k in range(6)])
    actual = trace["joint_pos"][:, actual_ids]
    initial = reference["joint_pos"][0, :6]
    rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
    windows = {}
    for label, mask in (("whole_reference", valid), ("braking", brake)):
        # The best constant uses the offline target mean in this window. It has
        # no actuator, dynamics, balance, or feasibility claim.
        oracle = target[mask].mean(0)
        executed_error = rmse(actual[mask]-target[mask])
        initial_error = rmse(initial-target[mask])
        rows = []
        for k, name in enumerate(names):
            x, y = actual[mask, k], target[mask, k]
            correlation = float(np.corrcoef(x, y)[0, 1]) if min(np.std(x), np.std(y)) > 1e-9 else None
            rows.append(dict(joint=name, reference_range_rad=float(np.ptp(y)), actual_range_rad=float(np.ptp(x)),
                mean_error_rad=float(np.mean(x-y)), rmse_rad=rmse(x-y), correlation=correlation))
        windows[label] = dict(actual_joint_rmse_rad=executed_error,
            initial_stance_joint_rmse_rad=initial_error,
            offline_best_constant_joint_rmse_rad=rmse(oracle-target[mask]),
            relative_rmse_reduction_vs_initial_stance=1-executed_error/initial_error if initial_error > 0 else None,
            per_joint=rows)
    tq = reference["body_quat_w"][:, 0]
    target_rotation = Slerp(rt, Rotation.from_quat(tq[:, [1, 2, 3, 0]]))(np.clip(t, rt[0], rt[-1]))
    actual_rotation = Rotation.from_quat(trace["root_quat"][:, [1, 2, 3, 0]])
    target_euler = target_rotation.as_euler("xyz", degrees=True)
    actual_euler = actual_rotation.as_euler("xyz", degrees=True)
    body_names = trace["body_names"].astype(str).tolist()
    reference_names = reference["body_names"].astype(str).tolist()
    actual_camber, target_camber = [], []
    for wheel in ("wheel_L_Link", "wheel_R_Link"):
        actual_axis = q_matrix(trace["body_quat"][:, body_names.index(wheel)])[..., :, 1]
        q = reference["body_quat_w"][:, reference_names.index(wheel)]
        reference_axis = Slerp(rt, Rotation.from_quat(q[:, [1, 2, 3, 0]]))(np.clip(t, rt[0], rt[-1])).as_matrix()[..., :, 1]
        actual_camber.append(np.degrees(np.arcsin(np.clip(actual_axis[:, 2], -1, 1))))
        target_camber.append(np.degrees(np.arcsin(np.clip(reference_axis[:, 2], -1, 1))))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    axes[0].plot(t, np.sqrt(np.mean((actual-target)**2, axis=1)), label="Executed PPO", color="#336db0")
    axes[0].plot(t, np.sqrt(np.mean((initial-target)**2, axis=1)), label="Fixed initial posture (kinematic)", color="#888888", ls=":")
    axes[0].set(title="Six-leg-joint posture error", ylabel="RMS [rad]")
    for j, name in enumerate(("Body roll", "Body pitch")):
        axes[j+1].plot(t, target_euler[:, j], color="#429345", label="GMR reference")
        axes[j+1].plot(t, actual_euler[:, j], color="#336db0", label="Executed PPO")
        axes[j+1].set(title=name, ylabel="Degrees (xyz Euler)")
    for k, color in enumerate(("#cc4d35", "#3476bd")):
        label = ("L", "R")[k]
        axes[3].plot(t, target_camber[k], color=color, ls="--", label=label+" reference")
        axes[3].plot(t, actual_camber[k], color=color, label=label+" executed")
    axes[3].set(title="Spin-invariant wheel camber", ylabel="asin(axle world-z) [deg]")
    for ax in axes:
        ax.set(xlabel="Time [s]", xlim=(0, rt[-1])); ax.grid(alpha=.2); ax.legend(fontsize=7)
        ax.axvspan(command["brake_start_s"], command["brake_end_s"], alpha=.07, color="gray")
    fig.suptitle("Physical task success and exact posture imitation are separate; constant postures are not controllers")
    a.output.mkdir(parents=True)
    image = a.output/"posture_fidelity_4x1.png"
    fig.savefig(image, dpi=160); plt.close(fig)
    report = dict(kind="hockey_posture_fidelity_vs_kinematic_constants", windows=windows,
        source_sha256=sha(__file__), trace_sha256=sha(trace_path), reference_sha256=sha(a.command_dir/"motion.npz"),
        qualification_sha256=sha(a.qualification/"report.json"), plot_sha256=sha(image),
        constants_are_simulated_controllers=False, new_posture_acceptance_threshold_introduced=False,
        physical_success_gate_changed=False, absolute_height_and_wheel_spin_used=False)
    (a.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(windows["braking"], indent=2))


if __name__ == "__main__":
    main()
