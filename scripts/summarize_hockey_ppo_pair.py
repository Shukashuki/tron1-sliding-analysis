"""Summarize the matched hockey PPO arms after independent qualification."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("evidence", "qualification", "command-dir", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    variants = ("tracking", "support")
    reports = {v: json.loads((a.qualification/v/"report.json").read_text()) for v in variants}
    trains = {v: json.loads((a.evidence/(v+"_formal")/"report.json").read_text()) for v in variants}
    configs = {v: json.loads((a.evidence/(v+"_formal")/"runner_config.json").read_text()) for v in variants}
    traces = {v: load(a.evidence/(v+"_eval")/"final_rollout.npz") for v in variants}
    signals = {v: load(a.qualification/v/"signals.npz") for v in variants}
    reference = load(a.command_dir/"motion.npz")
    rt = np.arange(len(reference["joint_pos"]))/float(reference["fps"])
    for key in ("initial_policy_state_sha256", "config_sha256", "motion_sha256", "asset_sha256", "source_sha256", "task_source_sha256", "transitions"):
        if trains["tracking"][key] != trains["support"][key]:
            raise ValueError("Training comparison differs: "+key)
    if configs["tracking"] != configs["support"]:
        raise ValueError("PPO hyperparameters differ")
    weights = {v: dict(trains[v]["reward_terms"]) for v in variants}
    if [weights[v].pop("hockey_support") for v in variants] != [0., .4] or weights["tracking"] != weights["support"]:
        raise ValueError("Treatment is not the declared single support-weight change")
    initial_error = {}
    for key in ("root_pos", "root_quat", "root_com_vel", "root_omega", "joint_pos", "joint_vel"):
        error = float(abs(traces["tracking"][key][0]-traces["support"][key][0]).max())
        if error > 1e-6:
            raise ValueError("Nominal initial state differs: "+key)
        initial_error[key] = error
    rows = []
    for v in variants:
        report = reports[v]
        if report["training_report_sha256"] != sha(a.evidence/(v+"_formal")/"report.json"):
            raise ValueError("Qualification refers to different training")
        if report["run_report_sha256"] != sha(a.evidence/(v+"_eval")/"report.json"):
            raise ValueError("Qualification refers to different validation")
        for name, digest in report["input_sha256"].items():
            if sha(a.evidence/(v+"_eval")/name) != digest:
                raise ValueError("Validation recording changed")
        q = report["qualification"]
        row = dict(variant=v, task_success=q["task_success_claimed"], xy_rmse_m=q["xy_RMSE_m"],
            xy_peak_m=q["xy_peak_m"], entry_speed_m_s=q["entry_mean_speed_last_0p1s_m_s"],
            approach_speed_ratio=q["approach_COM_speed_ratio"],
            body_heading_rmse_deg=q["braking_body_heading_RMSE_deg"],
            loaded_wheel_heading_rmse_deg=q["loaded_braking_wheel_heading_RMSE_deg"],
            support_fraction=q["support_fraction"], maximum_support_gap_s=q["longest_support_gap_s"],
            first_45deg_tilt_s=report["first_45deg_tilt_diagnostic_s"], terminal_stop_hold_s=q["terminal_stop_hold_s"])
        s = signals[v]
        t = s["time_s"]
        whole = (t > 0.) & (t <= rt[-1]+1e-9)
        # Each contact sample describes the interval ending at its timestamp.
        braking = (t-.0025 >= .4-1e-9) & (t <= 1.8+1e-9) & s["native_sample_valid"]
        force = s["normal_fz"][braking]
        loaded = force > 5.
        row["maximum_tilt_deg"] = float(s["root_tilt_deg"][whole].max())
        stop = q["stop_events"][-1] if q["stop_events"] else None
        row["supported_stop_start_s"] = stop["start_s"] if stop else None
        row["supported_stop_duration_s"] = stop["duration_s"] if stop else 0.
        row["braking_both_wheels_loaded_fraction"] = float(loaded.all(-1).mean())
        for k, side in enumerate(("left", "right")):
            row[side+"_braking_loaded_fraction"] = float(loaded[:, k].mean())
            row[side+"_braking_normal_force_p99_N"] = float(np.quantile(force[:, k], .99))
            row[side+"_braking_normal_force_peak_N"] = float(force[:, k].max())
        imitation = report["imitation"]["windows"]["braking"]
        row["braking_leg_joint_rmse_rad"] = imitation["leg_joint_RMSE_rad"]
        row["braking_body_up_axis_rmse_deg"] = imitation["body_up_axis_RMSE_deg"]
        for k, side in enumerate(("left", "right")):
            row[side+"_loaded_braking_camber_rmse_deg"] = imitation["loaded_wheel_camber_RMSE_deg"][k]
        for side in report["loaded_beta_summary"]:
            prefix = side["side"]+"_"
            row[prefix+"prefall_saturated_slip_longest_s"] = side["pre_fall_braking_saturated_dissipative_slip"]["longest_duration_s"]
            row[prefix+"prefall_high_beta_saturated_slip_longest_s"] = side["pre_fall_braking_high_beta_saturated_dissipative_slip"]["longest_duration_s"]
        rows.append(row)
    a.output.mkdir(parents=True)
    with (a.output/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    report = dict(kind="matched_direct_hockey_ppo_support_comparison", rows=rows,
        summarizer_sha256=sha(__file__),
        only_treatment="support deficit reward weight 0 versus +0.4 on negative cost",
        matched_ppo_hyperparameters=True, matched_initial_network=True,
        maximum_initial_physical_state_error=initial_error,
        failed_gates={v: reports[v]["qualification"]["failed_gates"] for v in variants},
        qualification_sha256={v: sha(a.qualification/v/"report.json") for v in variants},
        success_requires_all_existing_stop_and_slip_gates=True,
        pointwise_cone_saturation_certified=False, historical_car_initialization_used=False)
    report["additional_contact_diagnostics_are_new_success_gates"] = False
    (a.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    colors = ("#2673b8", "#d77325")
    axes[0].plot(*reference["body_pos_w"][:, 0, :2].T, color="#429345", label="Reference")
    axes[1].plot(rt, np.linalg.norm(reference["body_lin_vel_w"][:, 0, :2], axis=-1), color="#429345", label="Reference COM speed")
    for v, color in zip(variants, colors):
        t = signals[v]["time_s"]
        axes[0].plot(*traces[v]["root_pos"][:, :2].T, color=color, label=v)
        axes[1].plot(t, signals[v]["body_speed_m_s"], color=color, label=v)
        axes[2].plot(t, signals[v]["root_tilt_deg"], color=color, label=v)
        # Distinct vertical tracks avoid hiding shared unsupported intervals.
        supported = signals[v]["normal_fz"].max(-1) > 5
        valid = signals[v]["native_sample_valid"]
        offset = 1.5 if v == "tracking" else 0.
        axes[3].step(t, np.where(valid, supported.astype(float)+offset, np.nan), where="pre", color=color, label=v)
        first = reports[v]["first_45deg_tilt_diagnostic_s"]
        if first is not None:
            for ax in axes[1:]:
                ax.axvline(first, color=color, lw=.8, ls=":")
    axes[0].set(title="Base-link XY path", xlabel="X [m]", ylabel="Y [m]")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(title="Approach, braking, terminal hold", xlabel="Time [s]", ylabel="COM speed [m/s]")
    axes[1].axhline(.1, color="black", ls="--", lw=.7)
    axes[2].set(title="Tilt; dotted lines mark first fall", xlabel="Time [s]", ylabel="Degrees")
    axes[2].axhline(45, color="black", ls="--", lw=.7)
    axes[3].set(title="Any-wheel support at 400 Hz", xlabel="Time [s]", yticks=[0, 1, 1.5, 2.5],
                yticklabels=["support: air", "support: loaded", "tracking: air", "tracking: loaded"])
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in axes[1:]:
        ax.axvspan(.4, 1.8, color="gray", alpha=.08); ax.set_xlim(0, rt[-1])
    fig.suptitle("Direct hockey PPO | PhysX, mu=0.7 | inspect stop and physical sliding gates separately")
    fig.savefig(a.output/"tracking_and_support_4x1.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
