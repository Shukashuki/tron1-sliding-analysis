"""Four horizontal native diagnostics for matched400Hz PPO/CEM stage rewards."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from analyze_ppo_contact_timestep import band_spectrum
from analyze_ppo_ground_support_failure import persistent_events
from ppo_backend_400hz_contract import require

ORDER = ("ppo_beta_on", "cem_beta_on", "ppo_beta_off", "cem_beta_off")
LABELS = {"ppo_beta_on": "PPO + beta", "cem_beta_on": "CEM + beta", "ppo_beta_off": "PPO - beta", "cem_beta_off": "CEM - beta"}
COLORS = dict(zip(ORDER, ("#0072B2", "#009E73", "#D55E00", "#CC79A7")))


def diagnostic_data(traces, reference):
    require(set(traces) == set(ORDER), "Require all four completed400Hz PPO/CEM beta conditions")
    ref_time = np.asarray(reference["time_s"])
    require(ref_time.shape == (349,) and np.isclose(ref_time[0], 0.) and np.isclose(ref_time[-1], 6.96),
            "Use the real reference0–6.96s, not a held/extrapolated endpoint")
    result = {}
    for key in ORDER:
        trace = traces[key]
        time = np.asarray(trace["time_s"], dtype=float)
        require(time.shape == (2793,) and float(trace["physics_dt_s"]) == .0025
                and np.allclose(time, np.arange(2793)*.0025, rtol=0, atol=1e-10),
                "Require full original400Hz records, not historical200Hz or cropped poses")
        position = np.asarray(trace["position"])
        q, omega = (np.asarray(trace[name]) for name in ("quaternion", "angular_velocity"))
        require(position.shape == (2793, 3) and q.shape == (2793, 4) and omega.shape == (2793, 3)
                and all(np.isfinite(value).all() for value in (position, q, omega))
                and np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-5, rtol=0), "Require finite original SI poses/velocities")
        q = q/np.linalg.norm(q, axis=-1, keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(1.-2.*np.sum(q[:, 1:3]**2, axis=-1), -1., 1.)))
        overlap = (time >= -1e-9) & (time <= ref_time[-1]+1e-9)
        core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
        query = time[overlap]
        desired = np.column_stack([np.interp(query, ref_time, reference["position"][:, i]) for i in (0, 1)])
        result[key] = {"native_time_s": time, "position_xy_m": position[:, :2],
            "reference_overlap_time_s": query, "xy_error_m": np.linalg.norm(position[overlap, :2]-desired, axis=-1),
            "tilt_deg": tilt, "persistent_tilt30_diagnostic": persistent_events(time, tilt >= 30.)["first_persistent_event"],
            "common10to100_omegaXY_RMS_rad_s": {"core": band_spectrum(omega[core, :2], .0025)["common_band_rms"],
                "full": band_spectrum(omega[:, :2], .0025)["common_band_rms"]}}
    return result


def draw_diagnostics(traces, reference, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = diagnostic_data(traces, reference)
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    axes[0].plot(reference["position"][:, 0], reference["position"][:, 1], color="#777777", label="Reference", lw=2.)
    for key in ORDER:
        value = values[key]; color = COLORS[key]
        style = "--" if key.startswith("cem") else "-"
        axes[0].plot(value["position_xy_m"][:, 0], value["position_xy_m"][:, 1], style, color=color, label=LABELS[key], lw=1.6)
        axes[1].plot(value["reference_overlap_time_s"], value["xy_error_m"], style, color=color, lw=1.4)
        axes[2].plot(value["native_time_s"], value["tilt_deg"], style, color=color, lw=1.4)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("Original world-XY paths\nSolid: PPO; dashed: CEM")
    axes[0].set_xlabel("World X (m)"); axes[0].set_ylabel("World Y (m)"); axes[0].legend(fontsize=8)
    axes[1].set_title("XY error against real reference\nNo endpoint hold after6.96s")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("XY error (m)")
    axes[2].set_title("Root tilt: diagnostic, NOT a failure gate")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Tilt (deg)")
    axes[2].axhline(30., color="#888888", ls=":", lw=1.)
    for axis in axes[1:3]:
        axis.set_xlim(0., 6.98); axis.set_ylim(bottom=0.)
        axis.axvspan(1.44, 4., color="#777777", alpha=.08)
        axis.axvline(4., color="#555555", ls="--", lw=1.)
    centers = np.arange(2)
    maximum = 0.
    for slot, key in enumerate(ORDER):
        data = [values[key]["common10to100_omegaXY_RMS_rad_s"][window] for window in ("core", "full")]
        maximum = max(maximum, max(data))
        bars = axes[3].bar(centers+(slot-1.5)*.2, data, width=.19, color=COLORS[key], label=LABELS[key])
        axes[3].bar_label(bars, labels=[f"{value:.2f}" for value in data], padding=3, fontsize=8)
    axes[3].set_xticks(centers, ["Core1.44-4s", "Full0-6.98s"])
    axes[3].set_ylim(0., max(maximum, .01)*1.45)
    axes[3].set_title("Common10 < f <100Hz jitter\nLower HF is NOT task stability")
    axes[3].set_ylabel("World omegaXY RMS (rad/s)"); axes[3].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2); axis.set_axisbelow(True)
    figure.suptitle("Matched400Hz stage reward | PhysX-trained PPO vs CEM in native CPU MuJoCo | mu=0.7 | beta2 vs0 | N=1/condition", fontsize=12)
    figure.text(.5, .016, "Same stage rewards and gamma=.99, not identical information/return: CEM has0.6s model/reference preview, no terminal value. Shading: predefined turn core. No pose alignment/filtering.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0., .055, 1., .92))
    figure.savefig(Path(destination), dpi=160)
    plt.close(figure)
    return {"layout": [1, 4], "native_samples_per_case": 2793, "native_physics_dt_s": .0025,
        "tracking_interval_s": [0., 6.96], "full_record_interval_s": [0., 6.98], "core_interval_s": [1.44, 4.],
        "same_complete_PPO_reward_returns_claimed": False, "stage_reward_matched": True,
        "state_filtering_resampling_or_alignment": False, "tilt_or_contact_failure_gate": False,
        "persistent_tilt30_diagnostic_by_case": {key: values[key]["persistent_tilt30_diagnostic"] for key in ORDER}}
