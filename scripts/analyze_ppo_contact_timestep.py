"""Offline fixed-actor comparison of native5ms and2.5ms PhysX recordings.

No native states are aligned, resampled or filtered. Contact absence is not a
flight/failure label. Physics and explicit DC-PD rates both change, whereas the
actor, requested-action history, reference and50Hz policy clock stay fixed.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_contact_stability as contact
import analyze_ppo_smoothness as smooth


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = contact.DEFAULT_REFERENCE
MOTION_SHA256 = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
CHECKPOINT_SHA256 = "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a"
CASE_ORDER = ("baseline", "candidate")
FOLDERS = {"baseline": "dt0p005", "candidate": "dt0p0025"}
PHYSICS_DT = {"baseline": .005, "candidate": .0025}
AUTHORIZED_GPU_INDICES = (1, 3, 4, 5)
LEG_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint")
UNCHANGED_REPORT_FIELDS = ("contact_material_audit", "legacy_joint_friction_audit", "drift_module_sha256",
    "contact_slip_helper_sha256", "asset_root_file_sha256", "task_profile", "evaluation_task_profile",
    "domain_randomization_audit", "contact_slip_measurement", "terrain", "terrain_runtime_audit", "terrain_file_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def field(raw, name):
    value = np.asarray(raw[name])
    return value[:, 0] if value.ndim >= 2 and value.shape[1] == 1 else value


def band_spectrum(values, dt):
    """Same detrend/Hann/channel-mean Parseval bins as spectral_metrics.

    The comparable band is strictly10<f<100Hz in BOTH native recordings.
    Native>10Hz is also retained, but integrates different Nyquist limits.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    require(array.ndim == 2 and len(array) >= 4 and array.shape[1] > 0 and np.isfinite(array).all(), "Invalid spectral signal")
    require(type(dt) in (float, int) and np.isfinite(dt) and 0 < dt <= .005, "Invalid spectral sample dt")
    n = len(array); design = np.column_stack((np.ones(n), np.arange(n, dtype=float)))
    detrended = array-design@np.linalg.lstsq(design, array, rcond=None)[0]
    window = np.hanning(n)
    transform = np.fft.rfft(detrended*window[:, None], axis=0)
    power = np.mean(np.abs(transform)**2, axis=1)/(n*np.sum(window**2))
    power[1:] *= 2. if n % 2 else 1.
    if not n % 2:
        power[1:-1] *= 2.
    power[0] = 0.
    frequency = np.fft.rfftfreq(n, float(dt))
    common = float(power[(frequency > 10.) & (frequency < 100.)].sum())
    native = float(power[frequency > 10.].sum())
    extra = float(power[frequency >= 100.].sum())
    negligible = float(power.sum()) < 1e-24
    return {"samples": n, "sample_hz": 1./dt, "native_nyquist_hz": .5/dt,
        "frequency_resolution_hz": 1./(n*dt), "common_band_hz_exclusive": [10., 100.],
        "common_band_mean_square": 0. if negligible else common,
        "common_band_rms": 0. if negligible else float(np.sqrt(common)),
        "native_high_frequency_cutoff_hz_exclusive": 10.,
        "native_high_frequency_mean_square": 0. if negligible else native,
        "native_high_frequency_rms": 0. if negligible else float(np.sqrt(native)),
        "at_or_above100hz_mean_square": 0. if negligible else extra,
        "normalization": "one-sided Parseval; linear detrend; Hann; channel-mean power; DC excluded",
        "comparison_note": "Use common10<f<100Hz for shared-band comparisons; native>10Hz has different upper bandwidth"}


def signal_windows(raw, dt):
    time = np.asarray(raw["time_s"]); count = field(raw, "wheel_ground_contact_count")
    force = field(raw, "wheel_ground_contact_force_w_n")
    angular = field(raw, "root_angular_velocity_w_rad_s")
    quaternion = field(raw, "root_quaternion_wxyz").astype(float)
    names = tuple(raw["joint_names"].tolist()); legs = [names.index(name) for name in LEG_NAMES]
    torque = field(raw, "applied_joint_torque_nm")[:, legs]
    n = len(time)
    require(count.shape == (n, 2) and count.dtype.kind in "iu" and (count >= 0).all(), "Invalid native contact counts")
    require(force.shape == (n, 2, 3) and angular.shape == (n, 3) and torque.shape == (n, 6)
            and all(np.isfinite(value).all() for value in (force, angular, torque)), "Invalid native force/angular/torque arrays")
    require(quaternion.shape == (n, 4) and np.isfinite(quaternion).all()
            and np.allclose(np.linalg.norm(quaternion, axis=1), 1., rtol=0, atol=2e-5), "Invalid native quaternion")
    unit = quaternion/np.linalg.norm(quaternion, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1.-2.*(unit[:, 1]**2+unit[:, 2]**2), -1., 1.)))
    result = {}
    for name, interval in (("full", (0., float(time[-1]))), ("core", (1.44, 4.))):
        mask = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        t, fz = time[mask], force[mask, :, 2]
        masks = contact.contact_masks(count[mask], fz)
        row = {"interval_s": list(interval), "samples": int(mask.sum()), "native_dt_s": dt,
            "contact": {kind: {side: contact.contact_statistics(t, state) for side, state in states.items()}
                        for kind, states in masks.items()},
            "force_z": {side: {"mean_n": float(fz[:, index].mean()), "rms_n": smooth.rms(fz[:, index]),
                "minimum_n": float(fz[:, index].min()), "maximum_n": float(fz[:, index].max()),
                "p95_n": float(np.quantile(fz[:, index], .95)),
                "adjacent_native_step_rms_n": smooth.rms(np.diff(fz[:, index]))}
                for index, side in enumerate(("left", "right"))},
            "adjacent_native_leg_torque_step_rms_nm": smooth.rms(np.diff(torque[mask], axis=0)),
            "step_difference_comparison_note": "Adjacent force/torque changes use different native dt; do not directly rank these as equal-bandwidth rates",
            "tilt": {"rms_deg": smooth.rms(tilt[mask]), "maximum_deg": float(tilt[mask].max()),
                "median_deg": float(np.median(tilt[mask])), "not_a_failure_gate": True},
            "spectral_signal_units": {"world_angular_xy": "rad/s", "wheel_ground_fz": "N", "leg_applied_torque": "Nm"},
            "spectra_common_band10_100": {"world_angular_xy": band_spectrum(angular[mask, :2], dt),
                "wheel_ground_fz": band_spectrum(fz, dt), "leg_applied_torque": band_spectrum(torque[mask], dt)}}
        result[name] = row
    return result


def _load_case(folder, label, reference, parent, gpu_index):
    dt = PHYSICS_DT[label]; decimation = int(round(.02/dt)); expected_n = 349*decimation+1
    proof = parent.validate_output(folder, dt, physical_gpu_index=gpu_index)
    require(proof.get("status") == "verified" and proof.get("physics_dt_s") == dt
            and proof.get("decimation") == decimation and proof.get("native_states") == expected_n
            and proof.get("policy_decisions") == 349 and proof.get("native_clock_and_device_verified") is True,
            "Parent clock/device/source verifier did not approve exact native cohort")
    report = read_json(folder/"report.json")
    require(report.get("status") == "evaluated" and report.get("simulator") == "IsaacLab/PhysX"
            and report.get("checkpoint_sha256") == CHECKPOINT_SHA256
            and report.get("motion_file_sha256") == MOTION_SHA256
            and report.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7,
            "Require completed same-actor nominal/reference PhysX report")
    path = folder/"physics_trajectory.npz"
    digest = smooth.sha256(path)
    require(digest == report.get("physics_trajectory_sha256") == proof.get("recording_sha256")
            and smooth.sha256(folder/"report.json") == proof.get("report_sha256"), "Parent/report/native archive hash binding differs")
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    time = np.asarray(raw["time_s"]); valid = np.asarray(raw["valid_mask"])
    require(time.shape == (expected_n,) and np.array_equal(time, np.arange(expected_n)*dt)
            and valid.shape == (expected_n, 1) and valid.dtype.kind == "b" and valid.all()
            and float(raw["physics_dt_s"]) == dt and float(raw["policy_dt_s"]) == .02,
            "Require original complete native clock/valid first episode; no alignment or reset splice")
    counter = np.asarray(raw["simulation_step_counter"])
    require(counter.shape == (expected_n,) and counter.dtype.kind in "iu" and (counter >= 0).all()
            and np.array_equal(counter-counter[0], np.arange(expected_n)), "Physical integer counter sequence differs")
    applied = field(raw, "action_was_applied")
    require(applied.shape == (expected_n,) and applied.dtype.kind == "b" and not applied[0] and applied[1:].all(), "Invalid held-action application flags")
    requested, observation = field(raw, "actions_clipped"), field(raw, "action_observation")
    require(requested.shape == (expected_n, 8) and observation.shape == (expected_n, 51)
            and np.isfinite(requested).all() and np.isfinite(observation).all() and (np.abs(requested) <= 1.).all(), "Invalid original action/51D observation arrays")
    decision = smooth.deduplicate_actions(time, requested, applied, .02,
        frames=field(raw, "reference_frame_for_action"), counters=counter)
    require(decision["decisions"] == 349 and decision["decimation"] == decimation, "Require349 unchanged50Hz control decisions")
    previous = np.concatenate((np.zeros((1, 8), dtype=requested.dtype), decision["actions"][:-1]))
    require(np.array_equal(observation[decision["record_indices"], -8:], previous), "Actor previous action is not unchanged50Hz requested-action history")
    analyzed = smooth.analyze_trajectory(path, reference=reference, report=folder/"report.json")
    require(analyzed["control_decisions"] == 349 and analyzed["physical_samples"] == expected_n, "Generic analyzer native/control scope differs")
    additional = signal_windows(raw, dt)
    for window in ("core", "full"):
        analyzed["windows"][window].update(additional[window])
    ref = smooth._load_reference(reference)
    overlap = (max(float(time[0]), float(ref["time_s"][0])), min(float(time[-1]), float(ref["time_s"][-1])))
    overlap_tracking = {**smooth._tracking(time, field(raw, "root_position_m"), field(raw, "root_linear_velocity_w_m_s"),
        field(raw, "root_quaternion_wxyz"), ref, overlap), "interval_s": list(overlap),
        "scope": "Reference-overlap only; not full-horizon tracking; no endpoint hold or extrapolation"}
    return raw, {"checkpoint_sha256": CHECKPOINT_SHA256, "physics_dt_s": dt, "decimation": decimation,
        "native_sample_hz": 1./dt, "physical_samples": expected_n, "control_decisions": 349,
        "duration_s": float(time[-1]), "parent_verification": proof,
        "input_sha256": {"report.json": smooth.sha256(folder/"report.json"), "physics_trajectory.npz": digest},
        "windows": analyzed["windows"], "reference_overlap_tracking": overlap_tracking}, report


def load_comparison(input_root, reference=DEFAULT_REFERENCE):
    """Read both original native archives; preserve all sample counts/clocks."""
    require(smooth.sha256(reference) == MOTION_SHA256, "Locked reference SHA differs")
    root = Path(input_root); parent = importlib.import_module("run_ppo_contact_timestep")
    reports = {label: read_json(root/FOLDERS[label]/"report.json") for label in CASE_ORDER}
    indices = [reports[label].get("contact_timestep_diagnostic", {}).get("physical_gpu_index_requested") for label in CASE_ORDER]
    require(all(type(index) is int and index in AUTHORIZED_GPU_INDICES for index in indices)
            and indices[0] == indices[1], "Matched reports must declare SAME authorized unmasked physical GPU")
    gpu_index = indices[0]; raw, cases = {}, {}
    for label in CASE_ORDER:
        raw[label], cases[label], reports[label] = _load_case(root/FOLDERS[label], label, reference, parent, gpu_index)
    for key in UNCHANGED_REPORT_FIELDS:
        require(key in reports["baseline"] and key in reports["candidate"]
                and reports["baseline"][key] == reports["candidate"][key], "Shared material/source/profile contract differs: "+key)
    left, right = [reports[label]["policy_export"] for label in CASE_ORDER]
    allowed = {"physics_dt", "decimation", "physics_clock_refinement", "contact_timestep_diagnostic", "export_validation_max_abs_error"}
    for key in (set(left)|set(right))-allowed:
        require(left.get(key) == right.get(key), "Shared actor/observation/PD contract differs: "+key)
    for key in ("body_names", "wheel_body_names", "joint_names"):
        require(np.array_equal(raw["baseline"][key], raw["candidate"][key]), "Native named body/joint schema differs: "+key)
    changes = {}
    for window in ("core", "full"):
        changes[window] = {}
        for signal in ("world_angular_xy", "wheel_ground_fz", "leg_applied_torque"):
            values = [cases[label]["windows"][window]["spectra_common_band10_100"][signal]["common_band_rms"] for label in CASE_ORDER]
            changes[window][signal+"_common_band_rms"] = {"baseline": values[0], "candidate": values[1],
                "difference": values[1]-values[0], "ratio": values[1]/values[0] if values[0] > 1e-12 else None}
    return raw, {"schema_version": 1, "kind": "recorded_nominal_ppo_native_timestep_comparison",
        "status": "requires_visual_and_tracking_review", "case_order": list(CASE_ORDER), "cases": cases,
        "same_frozen_actor": True, "checkpoint_sha256": CHECKPOINT_SHA256, "reference_sha256": MOTION_SHA256,
        "physical_gpu_index_requested": gpu_index, "requested_device": f"cuda:{gpu_index}",
        "gpu_identity_scope": "Same explicitly declared unmasked ordinal; full native/tensor device proof delegated to parent verifier",
        "evaluation_mu": .7, "leg_kp": 500., "leg_kd": 10., "policy_hz": 50., "physics_dt_s_by_case": PHYSICS_DT,
        "core_interval_s": [1.44, 4.], "relative_common_band_changes": changes,
        "new_simulation": False, "new_inference": False, "additional_training_updates": 0,
        "resampling_or_state_alignment": False, "state_smoothing": False, "stable_contact_claimed": False,
        "reference_interpolation_only_for_tracking": True,
        "height_or_flight_failure_gate": False, "nonzero_sideslip_target_preserved": True,
        "requires_fullbody_visual_review": True,
        "analysis_source_sha256": {Path(__file__).name: smooth.sha256(__file__),
            "run_ppo_contact_timestep.py": smooth.sha256(parent.__file__),
            "analyze_ppo_smoothness.py": smooth.sha256(smooth.__file__), "analyze_ppo_contact_stability.py": smooth.sha256(contact.__file__)},
        "limitations": ["One fixed actor/seed/episode perdt; no statistical robustness, root-cause identification or stability certification.",
            "Both physical integration and explicitDC-PD update frequency change200→400Hz. This is not a contact-solver-only intervention or a newly trained policy.",
            "Native>10Hz spectra have different Nyquist limits100/200Hz; compare the separately reported common10<f<100Hz band, not unequal-bandwidth totals.",
            "Adjacent native force/torque differences cover different dt and bandwidth; smaller step differences alone do not establish improvement.",
            "Count>0 and upwardFz>5N eligibility are diagnostics. Missing thresholded points is not proof of geometric flight or an XY failure gate.",
            "Raw native1397/2793 states retain their own times; no alignment/resampling/pose correction. Both contain349original50Hz decisions.",
            "Tracking alone interpolates the locked reference at original native query times; raw physical states are not interpolated or resampled.",
            "Physical6.98s exceeds locked reference6.96s; full tracking remains unavailable without extrapolation; reference-overlap tracking is separately labeled."]}


def measured_state_arrays(raw):
    return {label+"__"+name: np.asarray(value).copy() for label, current in raw.items() for name, value in current.items()}


def render_contact_plot(raw, summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="col")
    for row, label in enumerate(CASE_ORDER):
        time = raw[label]["time_s"]; fz = field(raw[label], "wheel_ground_contact_force_w_n")[:, :, 2]
        counts = field(raw[label], "wheel_ground_contact_count"); angular = field(raw[label], "root_angular_velocity_w_rad_s")
        for index, side, color in ((0, "Left", "#D55E00"), (1, "Right", "#0072B2")):
            axes[row, 0].plot(time, fz[:, index], color=color, lw=.6, label=side+" Fz")
            axes[row, 1].step(time, ((counts[:, index]>0)&(fz[:, index]>5.)).astype(float)+1.3*index,
                where="post", color=color, lw=.6, label=side+" eligibility")
        axes[row, 2].plot(time, angular[:, 0], lw=.6, label="World wx")
        axes[row, 2].plot(time, angular[:, 1], lw=.6, label="World wy")
        axes[row, 0].set_ylabel(f"Native{1./PHYSICS_DT[label]:g}Hz\nGround Fz [N]")
        axes[row, 2].set_ylabel("Angular velocity [rad/s]")
        for axis in axes[row]:
            axis.axvspan(1.44, 4., color="#888888", alpha=.1); axis.set_xlim(0., 6.98); axis.grid(alpha=.2); axis.legend(fontsize=7)
    for axis, title in zip(axes[0], ("Original measured ground normal load", "Eligibility, NOT flight", "Original world angular XY")):
        axis.set_title(title)
    for axis in axes[-1]: axis.set_xlabel("Native physical time [s]")
    fig.suptitle("Same PhysX PPO actor | mu=.7 | policy50Hz | native physics + explicitPD200vs400Hz | no resampling")
    fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, required=True)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing analysis: choose a fresh output directory")
    raw, summary = load_comparison(args.input_root, args.reference)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"measured_states.npz", **measured_state_arrays(raw))
    render_contact_plot(raw, summary, output/"contact_comparison.png")
    summary["artifacts"] = {name: {"file": name, "sha256": smooth.sha256(output/name)}
        for name in ("measured_states.npz", "contact_comparison.png")}
    with (output/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    result = run(parser().parse_args()); print(json.dumps({"status": result["status"], "new_simulation": False}))
