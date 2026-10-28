"""Four-panel braking evidence, including the wheel-aggregate friction disk.

The disk is a normalized section of aggregate Ft <= mu*Fz on a flat floor.
It is not a reconstruction of individual contact-point friction cones.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    s = load(a.qualification/"signals.npz")
    q = json.loads((a.qualification/"report.json").read_text())
    t = s["time_s"]; mu = q["mu"]
    braking = (t-.0025 >= .4-1e-9)&(t <= 1.8+1e-9)&s["native_sample_valid"]
    first_fall = q["first_45deg_tilt_diagnostic_s"]
    if first_fall is not None:
        braking &= t < first_fall
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    colors = ("#cc4d35", "#3476bd")
    theta = np.linspace(0, 2*np.pi, 361)
    axes[2].plot(np.cos(theta), np.sin(theta), "k--", lw=1., label="Aggregate Coulomb limit")
    details = []
    for k, (side, color) in enumerate(zip(("Left", "Right"), colors)):
        loaded = braking&(s["normal_fz"][:, k] > 5.)
        beta_valid = loaded&s["wheel_beta_valid"][:, k]
        axes[0].plot(t, np.where(beta_valid, abs(s["wheel_axial_beta_deg"][:, k]), np.nan), color=color, lw=1., label=side)
        axes[1].plot(t, np.where(loaded, s["normal_point_lateral_rms"][:, k], np.nan), color=color, lw=1., label=side)
        denominator = mu*s["normal_fz"][loaded, k]
        normalized = s["friction_force"][loaded, k, :2]/denominator[:, None]
        axes[2].scatter(normalized[:, 0], normalized[:, 1], s=7, color=color, alpha=.4, label=side+" loaded braking")
        axes[3].scatter(s["normal_point_lateral_rms"][loaded, k], s["aggregate_utilization"][loaded, k], s=7, color=color, alpha=.35, label=side)
        events = q["loaded_beta_summary"][k]["pre_fall_braking_high_beta_saturated_dissipative_slip"]["events"]
        if not events:
            details.append(dict(side=side, longest_qualified_event=None))
            continue
        event = max(events, key=lambda e: e["duration_s"])
        index = np.arange(event["first_index"], event["last_index"]+1)
        fraction = s["anchor_force_fraction_dissipative_lateral_slip"][index, k]
        quantiles = {key: np.quantile(s[key][index, k], [0, .5, 1.]).tolist()
                     for key in ("normal_fz", "normal_point_lateral_rms", "aggregate_utilization", "anchor_power")}
        quantiles["absolute_axial_beta_deg"] = np.quantile(abs(s["wheel_axial_beta_deg"][index, k]), [0, .5, 1.]).tolist()
        quantiles["same_anchor_lateral_dissipation_force_fraction"] = np.quantile(fraction, [0, .5, 1.]).tolist()
        details.append(dict(side=side, longest_qualified_event=event, quantiles_min_median_max=quantiles))
        for ax in axes[:2]:
            ax.axvspan(event["start_s"], event["end_s"], color=color, alpha=.1)
    axes[0].axhline(20, color="black", ls=":", lw=.9)
    axes[0].set(title="Loaded wheel |axial beta|", xlabel="Time [s]", ylabel="Degrees", ylim=(0, 95))
    axes[1].axhline(.05, color="black", ls=":", lw=.9)
    axes[1].set(title="Actual contact-material lateral slip", xlabel="Time [s]", ylabel="Loaded point RMS [m/s]")
    axes[2].set(title="Wheel-aggregate friction disk", xlabel="Sum(Fx) / (mu Fz)", ylabel="Sum(Fy) / (mu Fz)",
                xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
    axes[3].axhspan(.9, 1.05, color="gray", alpha=.1)
    axes[3].axvline(.05, color="black", ls=":", lw=.9)
    axes[3].set(title="Sliding and saturation jointly", xlabel="Loaded lateral slip RMS [m/s]", ylabel="Aggregate Ft / (mu Fz)")
    for ax in axes:
        ax.grid(alpha=.18); ax.legend(fontsize=7)
    for ax in axes[:2]:
        ax.set_xlim(.4, 1.8)
    fig.suptitle(f"PhysX hockey stop, mu={mu:g} | native braking contacts before any fall | aggregate, not pointwise cone proof")
    a.output.mkdir(parents=True)
    image = a.output/"beta_slip_friction_disk_4x1.png"
    fig.savefig(image, dpi=160); plt.close(fig)
    report = dict(qualification_sha256=sha(a.qualification/"report.json"),
        signals_sha256=sha(a.qualification/"signals.npz"), source_sha256=sha(__file__),
        plot_sha256=sha(image), native_braking_samples_only=True, pointwise_cone_certificate=False,
        raw_API_forces_already_N=True, extra_impulse_division_applied=False,
        power_is_poststep_force_velocity_proxy_not_exact_work=True, details=details)
    (a.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
