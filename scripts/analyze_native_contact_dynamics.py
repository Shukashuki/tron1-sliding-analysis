"""Read-only native contact and command-transition diagnostics for frozen K/N.

Aggregated friction patches are not individual friction-cone measurements.
Body course-heading offset, wheel-hub angle and actual contact-point sliding
remain distinct. Associations within two recordings do not establish causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_cem_ground_contact_v2 import support_summary
from nonlinear_tire_contact import wheel_frame
from physx_tire_runtime import rotate_y

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
          "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"]
BASELINES = {key: ROOT/("outputs/ground_contact_v2"+key+"_"+date+"_v1/ppo_nominal_raw_v1/physx/physics_trajectory.npz")
             for key, date in (("k", "20261024"), ("n", "20261025"))}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def quantiles(value):
    value = np.asarray(value); value = value[np.isfinite(value)]
    return None if not value.size else np.quantile(value, [.5, .95, 1.]).tolist()


def empirical_auc(labels, values):
    """Descriptive pairwise AUC with midrank ties; no inferential independence."""
    labels = np.asarray(labels, dtype=bool); values = np.asarray(values, dtype=float)
    positive, negative = values[labels], values[~labels]
    if not len(positive) or not len(negative): return None
    return float(np.mean((positive[:, None] > negative).astype(float)
                         + .5*(positive[:, None] == negative)))


def contact_channels(detail, raw):
    time = np.asarray(detail["time_before_step_s"], dtype=float)
    assert len(time) == len(raw["time_s"])-1 and np.allclose(time, raw["time_s"][:-1], atol=1e-12, rtol=0)
    assert np.allclose(np.diff(time), .0025, atol=1e-12, rtol=0)
    # The observer's write counter names the UPCOMING step; values are read
    # before that step and match the preceding recorded native state.
    assert np.array_equal(detail["sim_counter"], raw["simulation_step_counter"][1:])
    normal = np.asarray(detail["preceding_normal_load_n"], dtype=float)
    assert np.allclose(normal, raw["wheel_ground_contact_force_w_n"][:-1, 0, :, 2], atol=1e-4, rtol=1e-6), "Normal force clock differs"
    assert not np.any(detail["applied_tangent_force_w_n"]) and not np.any(detail["applied_tangent_moment_COM_w_nm"])
    loaded = normal > 5.
    forward, lateral, _ = wheel_frame(rotate_y(detail["wheel_quaternion_wxyz"]), np.array([0., 0., 1.]))
    point_velocity = detail["COM_velocity_w_m_s"] + np.cross(detail["angular_velocity_w_rad_s"],
        detail["preceding_contact_position_w_m"]-detail["COM_position_w_m"])
    assert np.isfinite(point_velocity[loaded]).all(), "Loaded wheel lacks finite contact-point velocity"
    force = detail["native_friction_force_w_n"]
    utilization = np.divide(np.linalg.norm(force[..., :2], axis=-1), .7*normal,
        out=np.full_like(normal, np.nan), where=loaded)
    patch_sum = np.divide(detail["native_sum_patch_tangent_force_norm_n"], .7*normal,
        out=np.full_like(normal, np.nan), where=loaded)
    return {"time_s": time, "normal_n": normal, "loaded": loaded,
        "contact_velocity_w_m_s": point_velocity,
        "contact_longitudinal_slip_m_s": np.sum(point_velocity*forward, axis=-1),
        "contact_lateral_slip_m_s": np.sum(point_velocity*lateral, axis=-1),
        "aggregate_force_utilization": utilization, "sum_patch_force_utilization": patch_sum,
        "native_friction_force_w_n": force,
        "hub_side_angle_rad": detail["hub_side_angle_rad"],
        "base_COM_vz_m_s": raw["root_linear_velocity_w_m_s"][:-1, 0, 2]}


def command_transitions(raw, interval=(1.44, 4.)):
    assert raw["joint_names"].tolist() == JOINTS
    time = raw["time_s"]; action = raw["actions_clipped"][:, 0]
    force = raw["wheel_ground_contact_force_w_n"][:, 0, :, 2]
    assert np.isfinite(action).all() and np.all(np.abs(action) <= 1.+1e-7)
    assert np.allclose(np.diff(time), .0025, atol=1e-12, rtol=0)
    # All eight substeps retain the same actual applied target.
    for index in range(1, len(time), 8):
        assert np.array_equal(action[index:index+8], np.broadcast_to(action[index], action[index:index+8].shape))
    supported = np.max(force, axis=1) > 5.
    core = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
    onset = (~supported) & np.r_[False, supported[:-1]] & core
    phases = np.bincount(np.flatnonzero(onset) % 8, minlength=8)
    indices = np.arange(1, len(time), 8)
    indices = indices[core[indices] & supported[indices-1]]
    changed = action[indices]-action[indices-1]
    side_rms = np.stack([np.sqrt(np.mean(changed[:, ids]**2, axis=1))
                         for ids in ([0, 2, 4], [1, 3, 5])], axis=1)
    minimum = np.min(np.where(force[indices-1] > 5., side_rms, np.inf), axis=1)
    labels = ~supported[indices]
    groups = {}
    for name, mask in (("immediate_support_gap", labels), ("still_supported", ~labels)):
        groups[name] = {"samples": int(mask.sum()),
            "all_leg_target_jump_l2_rad_quantiles_50_95_100": quantiles(np.linalg.norm(changed[mask, :6], axis=1)),
            "minimum_previously_supporting_leg_target_jump_rms_rad_quantiles_50_95_100": quantiles(minimum[mask])}
    summary = {"core_gap_onsets": int(onset.sum()), "gap_onset_physics_phase_0_through7": phases.tolist(),
        "phase1_is_first_substep_after_control_command": True,
        "immediate_gap_after_previously_supported_control_updates": int(labels.sum()),
        "previously_supported_control_updates": len(indices), "groups": groups,
        "all_leg_jump_descriptive_auc": empirical_auc(labels, np.linalg.norm(changed[:, :6], axis=1)),
        "supporting_leg_minimum_rms_jump_descriptive_auc": empirical_auc(labels, minimum),
        "causal_effect_or_independent_frame_statistics_claimed": False}
    return summary, {"decision_time_s": time[indices], "immediate_support_gap": labels,
                     "minimum_supporting_leg_target_jump_rms_rad": minimum}


def run(args):
    assert not args.output_dir.exists(), "Fresh diagnostic analysis directory required"
    rows, series, evidence = [], {}, {}
    for tag in ("k", "n"):
        folder = args.collection/(tag+"_native_v1")
        metadata_path = folder/"native_contact_diagnostic.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["baseline_recording_byte_identical"] and metadata["cohort"] == "v2"+tag
        assert metadata["observer"]["law"] == "native_coulomb" and metadata["mu"] == .7
        assert metadata["observer"]["normal_load_delay_physical_steps"] == 0
        assert sha256(BASELINES[tag]) == metadata["baseline_recording_sha256"] == metadata["physics_trajectory_sha256"]
        detail_path = folder/"native_contact_detail.npz"
        assert sha256(detail_path) == metadata["native_contact_detail_sha256"]
        detail, raw = load_npz(detail_path), load_npz(BASELINES[tag])
        assert len(raw["time_s"]) == 2793 and len(detail["time_before_step_s"]) == 2792
        channels = contact_channels(detail, raw)
        time = channels["time_s"]; core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
        loaded = channels["loaded"] & core[:, None]
        slip = np.abs(channels["contact_lateral_slip_m_s"])
        rho = channels["aggregate_force_utilization"]
        q = raw["root_quaternion_wxyz"][:-1, 0]; w, x, y, z = q.T
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        velocity = raw["root_linear_velocity_w_m_s"][:-1, 0]
        beta = (np.arctan2(velocity[:, 1], velocity[:, 0])-yaw+np.pi) % (2*np.pi)-np.pi
        beta_mask = core & np.any(channels["loaded"], axis=1) & (np.linalg.norm(velocity[:, :2], axis=1) >= .2)
        commands, command_arrays = command_transitions(raw)
        support = support_summary(raw["time_s"], raw["wheel_ground_contact_force_w_n"][:, 0, :, 2], (.02, 6.96))
        events = []
        for gap in support["no_support_gaps"]:
            if not 1.44-1e-9 <= gap["onset_s"] <= 4.+1e-9: continue
            i = int(round(gap["onset_s"]/.0025)); previous = slice(max(0, i-8), i)
            before_loaded = channels["loaded"][previous]
            events.append({**gap, "physics_phase": i % 8,
                "preceding20ms_normal_peak_n": float(np.max(channels["normal_n"][previous])),
                "preceding20ms_loaded_aggregate_utilization_quantiles_50_95_100": quantiles(rho[previous][before_loaded]),
                "preceding20ms_loaded_lateral_slip_abs_m_s_quantiles_50_95_100": quantiles(slip[previous][before_loaded]),
                "base_COM_vertical_velocity_at_onset_m_s": float(velocity[i, 2])})
        rows.append({"cohort": "v2"+tag, "core_loaded_wheel_samples": int(loaded.sum()),
            "core_loaded_abs_body_beta_deg_quantiles_50_95_100": quantiles(np.abs(np.rad2deg(beta[beta_mask]))),
            "core_loaded_abs_wheel_hub_angle_deg_quantiles_50_95_100": quantiles(np.abs(np.rad2deg(channels["hub_side_angle_rad"][loaded]))),
            "core_loaded_lateral_contact_slip_abs_m_s_quantiles_50_95_100": quantiles(slip[loaded]),
            "core_loaded_net_aggregate_utilization_quantiles_50_95_100": quantiles(rho[loaded]),
            "core_loaded_sum_patch_utilization_quantiles_50_95_100": quantiles(channels["sum_patch_force_utilization"][loaded]),
            "core_loaded_fraction_aggregate_utilization_ge0p9": float(np.mean(rho[loaded] >= .9)),
            "core_loaded_fraction_lateral_contact_slip_ge0p1m_s": float(np.mean(slip[loaded] >= .1)),
            "core_loaded_fraction_joint_lateral_slip_and_near_limit": float(np.mean((slip[loaded] >= .1) & (rho[loaded] >= .9))),
            "support": support, "command_transitions": commands, "core_gap_context": events})
        series[tag] = {**channels, **command_arrays}
        evidence[tag] = {"baseline": {"path": str(BASELINES[tag]), "sha256": sha256(BASELINES[tag])},
            "observer_metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "contact_detail": {"path": str(detail_path), "sha256": sha256(detail_path)}}
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir/"native_contact_channels.npz", **{tag+"__"+name: value
        for tag, channels in series.items() for name, value in channels.items()})
    summary = {"kind": "native_PhysX_contact_and_command_dynamics_readonly_diagnostics", "rows": rows,
        "evidence": evidence, "analyzer_sha256": sha256(__file__), "source_policy_seeds": 1,
        "additional_independent_policy_seeds_or_physics_variants": 0,
        "counter_semantics": "Observer counter is the upcoming write; recorded forces and velocities match preceding native state, without a time shift.",
        "pointwise_friction_cone_edge_identified": False, "zero_sideslip_target_introduced": False,
        "causal_origin_of_support_loss_established": False, "task_success_claimed": False,
        "limitations": ["Same-seed policy candidates and overlapping within-episode samples; descriptive associations only.",
            "Body beta, wheel-hub direction and physical contact-point slip have different meanings.",
            "A friction patch resultant is not a per-point Coulomb cone measurement.",
            "Patch-speed and utilization thresholds are diagnostics, not new task failure gates.",
            "Base COM vertical velocity is not whole-system momentum; absence of force is not geometric flight.",
            "No actors, physical models, new rollouts, FK reconstruction or training are executed by this analyzer."],
        "artifacts": {"native_contact_channels.npz": sha256(args.output_dir/"native_contact_channels.npz")}}
    plot(series, args.output_dir/"contact_and_command_timing.png")
    summary["artifacts"]["contact_and_command_timing.png"] = sha256(args.output_dir/"contact_and_command_timing.png")
    (args.output_dir/"analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def plot(series, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), dpi=150, sharex=True, sharey="row")
    for col, tag in enumerate(("k", "n")):
        data = series[tag]; time = data["time_s"]
        for side, color in ((0, "tab:blue"), (1, "tab:orange")):
            label = ("Left" if side == 0 else "Right")
            axes[0, col].plot(time, data["normal_n"][:, side], color=color, label=label, lw=.8)
            axes[1, col].plot(time, data["aggregate_force_utilization"][:, side], color=color, lw=.8)
            values = np.where(data["loaded"][:, side], data["contact_lateral_slip_m_s"][:, side], np.nan)
            axes[2, col].plot(time, values, color=color, lw=.8)
        failed = data["immediate_support_gap"]
        axes[3, col].scatter(data["decision_time_s"][~failed], data["minimum_supporting_leg_target_jump_rms_rad"][~failed],
            s=12, color="tab:green", label="Still supported after command")
        axes[3, col].scatter(data["decision_time_s"][failed], data["minimum_supporting_leg_target_jump_rms_rad"][failed],
            s=18, color="tab:red", label="Immediate support gap")
        axes[0, col].set_title(tag.upper()+" frozen PPO | unchanged native PhysX, mu=.7")
        for row, label in enumerate(("Ground normal force (N)", "Aggregate |Ft|/(mu Fn)",
                                     "Loaded contact lateral slip (m/s)", "Stance-leg target jump RMS (rad)")):
            axes[row, col].set_ylabel(label); axes[row, col].grid(alpha=.2); axes[row, col].set_xlim(1.44, 4.)
        axes[0, col].legend(fontsize=8); axes[3, col].legend(fontsize=7); axes[3, col].set_xlabel("Native time (s)")
        axes[1, col].axhline(.9, color="gray", ls="--", lw=.8)
    fig.suptitle("Identical-recording contact readback | command-transition association, not a causal intervention")
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--collection", type=Path, default=ROOT/"outputs/native_contact_dynamics_20261025_v1/raw_v1")
    p.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps({"status": "completed_readonly_analysis", "cases": len(run(p.parse_args())["rows"])}))
