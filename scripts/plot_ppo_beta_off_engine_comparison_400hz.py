"""Native diagnostics for ONE PhysX-trained beta-OFF actor in two engines.

No physics, policy inference, state interpolation, alignment or filtering.
Tracking is measured only where the real reference exists; full recorded poses
remain visible so a fallen, quiet robot is never mistaken for successful control.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from analyze_ppo_contact_timestep import band_spectrum
from analyze_ppo_ground_support_failure import persistent_events
from ppo_backend_400hz_contract import require

ORDER = ("physx", "mujoco")
COLORS = {"physx": "#0072B2", "mujoco": "#D55E00"}
LABELS = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}


def diagnostic_data(traces, reference):
    """Use both full native clocks; interpolate only the locked reference."""
    require(set(traces) == set(ORDER), "Require exactly PhysX and MuJoCo validation recordings")
    ref_time = np.asarray(reference["time_s"])
    require(ref_time.ndim == 1 and len(ref_time) > 1 and np.isclose(ref_time[0], 0.)
            and np.isclose(ref_time[-1], 6.96), "Require the real locked reference interval")
    result = {}
    for key in ORDER:
        trace = traces[key]
        time = np.asarray(trace["time_s"], dtype=float)
        require(time.shape == (2793,) and float(trace["physics_dt_s"]) == .0025
                and np.allclose(time, np.arange(2793)*.0025, atol=1e-10, rtol=0),
                "Do not substitute old200Hz, cropped or resampled recordings")
        position = np.asarray(trace["position"])
        q = np.asarray(trace["quaternion"], dtype=float)
        omega = np.asarray(trace["angular_velocity"])
        require(position.shape == (2793, 3) and q.shape == (2793, 4) and omega.shape == (2793, 3)
                and all(np.isfinite(value).all() for value in (position, q, omega))
                and np.allclose(np.linalg.norm(q, axis=1), 1., rtol=0, atol=1e-5),
                "Require original finite SI root poses and angular velocity")
        unit = q/np.linalg.norm(q, axis=1, keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(1.-2.*np.sum(unit[:, 1:3]**2, axis=1), -1., 1.)))
        overlap = (time >= ref_time[0]-1e-9) & (time <= ref_time[-1]+1e-9)
        query = time[overlap]
        desired = np.column_stack([np.interp(query, ref_time, reference["position"][:, axis]) for axis in (0, 1)])
        error = np.linalg.norm(position[overlap, :2]-desired, axis=1)
        core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
        result[key] = {"native_time_s": time, "reference_overlap_time_s": query,
            "position_xy": position[:, :2], "xy_error_m": error, "tilt_deg": tilt,
            "first_persistent_tilt30_diagnostic": persistent_events(time, tilt >= 30.)["first_persistent_event"],
            "common10_100_rms_rad_s": {"core": band_spectrum(omega[core, :2], .0025)["common_band_rms"],
                                       "full": band_spectrum(omega[:, :2], .0025)["common_band_rms"]}}
    return result


def draw_diagnostics(traces, reference, destination):
    """Four horizontal panels for the same actor's native closed-loop runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = diagnostic_data(traces, reference)
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    axes[0].plot(reference["position"][:, 0], reference["position"][:, 1], color="#777777", lw=2., label="Reference")
    for key in ORDER:
        value, color = data[key], COLORS[key]
        axes[0].plot(value["position_xy"][:, 0], value["position_xy"][:, 1], color=color, lw=1.5, label=LABELS[key])
        nearest = int(np.argmin(np.abs(value["native_time_s"]-4.2)))
        axes[0].scatter(*value["position_xy"][nearest], color=color, s=25, marker="x")
        axes[1].plot(value["reference_overlap_time_s"], value["xy_error_m"], color=color, lw=1.2)
        axes[2].plot(value["native_time_s"], value["tilt_deg"], color=color, lw=1.2)
    axes[0].set_title("Original world-XY path\nCrosses mark t = 4.20 s")
    axes[0].set_xlabel("World X (m)"); axes[0].set_ylabel("World Y (m)")
    axes[0].set_aspect("equal", adjustable="datalim"); axes[0].legend(fontsize=8, loc="best")
    axes[1].set_title("XY tracking through the real reference\nNo endpoint hold after 6.96 s")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("XY error (m)")
    axes[1].set_xlim(0., 6.98); axes[1].set_ylim(bottom=0.)
    axes[2].set_title("Root tilt: diagnostic, NOT a failure gate")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Tilt (deg)")
    axes[2].set_xlim(0., 6.98); axes[2].set_ylim(bottom=0.)
    axes[2].axhline(30., color="#999999", ls=":", lw=1.)
    for axis in axes[1:3]:
        axis.axvspan(1.44, 4., color="#777777", alpha=.08)
        axis.axvline(4., color="#555555", ls="--", lw=1.)
    events = {}
    for slot, key in enumerate(ORDER):
        event = data[key]["first_persistent_tilt30_diagnostic"]
        events[key] = event
        if event is not None:
            for axis in axes[1:3]:
                axis.axvline(event["onset_s"], color=COLORS[key], ls=":", lw=1.)
            axes[2].annotate(f"{LABELS[key]}: tilt >= 30 deg for >= 0.10 s\nonset {event['onset_s']:.2f} s; confirmed {event['confirmation_s']:.2f} s",
                xy=(event["onset_s"], 30.), xytext=(.03, .95-slot*.18), textcoords="axes fraction", va="top", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": COLORS[key]}, color=COLORS[key])
    centers = np.arange(2)
    maximum = 0.
    for offset, key in ((-.18, "physx"), (.18, "mujoco")):
        values = [data[key]["common10_100_rms_rad_s"][window] for window in ("core", "full")]
        maximum = max(maximum, max(values))
        bars = axes[3].bar(centers+offset, values, width=.36, color=COLORS[key], label=LABELS[key])
        axes[3].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    axes[3].set_xticks(centers, ["Core 1.44-4.00 s", "Full 0-6.98 s"], fontsize=9)
    axes[3].set_ylabel("World omegaXY RMS (rad/s)")
    axes[3].set_title("Jitter: common 10 < f < 100 Hz\nLower HF is NOT task stability")
    axes[3].set_ylim(0., max(.01, maximum)*1.3); axes[3].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2); axis.set_axisbelow(True)
    figure.suptitle("NEW 400 Hz beta-OFF PPO | SAME PhysX-trained actor | PhysX vs native CPU MuJoCo | mu = 0.7 | policy 50 Hz | N = 1/engine", fontsize=12)
    figure.text(.5, .016, "Different native plants/contact laws; original full-horizon records, no alignment/smoothing. Shading: predefined turn core. Tilt/contact do not define XY failure.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0., .055, 1., .92))
    figure.savefig(Path(destination), dpi=160)
    plt.close(figure)
    return {"layout": [1, 4], "new_simulation": False, "native_states_per_condition": 2793,
        "xy_error_interval_s": [0., 6.96], "pose_diagnostic_interval_s": [0., 6.98],
        "core_interval_s": [1.44, 4.], "state_filtering_resampling_or_alignment": False,
        "tilt30_used_as_failure_gate": False, "persistent_tilt30_diagnostic_by_engine": events,
        "same_actor_different_native_plants": True, "identical_action_sequence_claimed": False,
        "MJWarp_trained_actor_included": False}
