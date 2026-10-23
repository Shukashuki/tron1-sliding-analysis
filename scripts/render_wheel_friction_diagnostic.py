"""Plot archived wheel-resultant force utilization and measured lateral RMS.

No physics, contact-force reconstruction, inference, or reward changes. The
reference circle is not the recorded MuJoCo pyramidal boundary. Missing or
unloaded observations remain missing; wheel-level resultants can cancel local
contact forces and are not exact per-contact saturation measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(directory):
    directory = Path(directory)
    summary_path, signals_path = directory / "summary.json", directory / "friction_signals.npz"
    summary = json.loads(summary_path.read_text())
    if (summary.get("status") != "completed_offline_diagnostic"
            or summary.get("data_scope") != "wheel_resultant_lower_bound"
            or summary.get("actual_per_contact_engine_boundary_computed") is not False
            or summary.get("reward_or_acceptance_changed") is not False
            or sha256(signals_path) != summary["artifact_sha256"]["friction_signals.npz"]):
        raise ValueError("Require a hash-bound completed wheel-resultant diagnostic")
    with np.load(signals_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    time = arrays["time_s"]
    if time.shape != (2793,) or not np.allclose(time, np.arange(2793)*.0025, atol=1e-9, rtol=0):
        raise ValueError("Require the complete native 400 Hz clock")
    for name in ("coulomb_reference_rho_l2", "wheel_lateral_slip_abs_m_s",
                 "cone_valid_loaded", "wheel_lateral_slip_valid"):
        if arrays[name].shape != (2793, 2):
            raise ValueError("Missing separate L/R diagnostic signal: " + name)
    return summary, arrays, {"summary.json": sha256(summary_path),
                             "friction_signals.npz": sha256(signals_path)}


def render(input_dir, output_dir):
    summary, arrays, input_hashes = load_inputs(input_dir)
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Preserve existing renders; use a fresh output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = arrays["time_s"]
    selected = (time >= .02-1e-9) & (time <= 6.96+1e-9)
    rho = np.where(arrays["cone_valid_loaded"], arrays["coulomb_reference_rho_l2"], np.nan)
    lateral = np.where(arrays["wheel_lateral_slip_valid"], arrays["wheel_lateral_slip_abs_m_s"], np.nan)
    thresholds = summary["exploratory_thresholds_NOT_acceptance"]
    rho_cutoff = thresholds["near_boundary_ratio_greater_equal"]
    slip_cutoff = thresholds["relative_slip_speed_m_s_strict"]
    mu = summary["provenance"]["effective_pair_mu"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    colors = ("#2166ac", "#d95f02")
    for wheel, (side, color) in enumerate(zip(("L", "R"), colors)):
        axes[0].plot(time[selected], rho[selected, wheel], color=color, lw=.75, label=side)
        axes[1].plot(time[selected], lateral[selected, wheel], color=color, lw=.75, label=side)
        valid = selected & np.isfinite(rho[:, wheel]) & np.isfinite(lateral[:, wheel])
        core = valid & (time >= 1.44-1e-9) & (time <= 4.+1e-9)
        outside = valid & ~core
        panel = axes[wheel+2]
        panel.scatter(rho[outside, wheel], lateral[outside, wheel], s=5, color="#aaaaaa", alpha=.35, label="Outside core")
        panel.scatter(rho[core, wheel], lateral[core, wheel], s=7, color=color, alpha=.45, label="Turning core")
        panel.axvline(rho_cutoff, color="#a50f15", ls="--", lw=1)
        panel.axhline(slip_cutoff, color="#a50f15", ls="--", lw=1)
        panel.set(title=f"{side}: force vs measured slip", xlabel="Resultant Coulomb-reference rho",
                  ylabel="Patch lateral RMS (m/s)", xlim=(0, None), ylim=(0, None))
        panel.legend(fontsize=8)
    axes[0].axhline(rho_cutoff, color="#a50f15", ls="--", lw=1, label=f"Exploratory {rho_cutoff:g}")
    axes[0].axhline(1., color="black", ls=":", lw=.8)
    axes[0].set(title="Per-wheel force utilization", xlabel="Time (s)", ylabel="Resultant Coulomb-reference rho", ylim=(0, None))
    axes[1].axhline(slip_cutoff, color="#a50f15", ls="--", lw=1)
    axes[1].set(title="Measured contact slip", xlabel="Time (s)", ylabel="Patch lateral RMS (m/s)", ylim=(0, None))
    for axis in axes[:2]:
        axis.axvspan(1.44, 4., color="#969696", alpha=.12)
        axis.set_xlim(.02, 6.96)
        axis.legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    cohort = summary["provenance"]["cohort_id"]
    fig.suptitle(f"{cohort} | native MuJoCo, mu={mu:g}\nWheel-resultant reference ONLY; actual per-contact engine boundary unavailable", fontsize=12)
    fig.supxlabel("Missing / unloaded samples omitted. Cutoffs describe force and slip, NOT drift success or task acceptance.", fontsize=10)
    output.mkdir(parents=True, exist_ok=False)
    try:
        fig.savefig(output / "comparison.png", dpi=160)
    finally:
        plt.close(fig)
    provenance = {"kind": "wheel_resultant_friction_diagnostic_four_horizontal_panels",
        "input_sha256": input_hashes, "renderer_sha256": sha256(__file__),
        "output_sha256": {"comparison.png": sha256(output / "comparison.png")},
        "actual_per_contact_engine_boundary_claimed": False,
        "missing_data_imputed": False, "new_physics_or_inference": False,
        "reward_or_acceptance_changed": False, "visual_reviewed": False}
    with (output / "provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input_dir, args.output_dir), indent=2))
