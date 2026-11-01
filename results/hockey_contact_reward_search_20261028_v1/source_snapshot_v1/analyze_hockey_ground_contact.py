"""Contact-first evaluation: geometric contact eligibility, then slip and forces.

Native manifolds can contain positive-distance points. Use separation, not a
normal-force threshold, to select eligible points. All velocities are recorded
post-step rigid-body material velocities, not new physics simulations.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_hockey_support_failure import read_support_case
from evaluate_hockey_physx import load, q_matrix, segments, sha


def contact_kinematics(trace, raw, tolerance_m):
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Nonnegative contact tolerance required")
    n = len(trace["body_com_pos"])
    if not np.array_equal(raw["physical_sample_indices"], np.arange(1, n)):
        raise ValueError("Incomplete physical-step record")
    result = {name: np.full((n, 2), np.nan) for name in
              ("min_separation_m", "tangent_rms_m_s", "lateral_rms_m_s")}
    result.update(contact_point_count=np.zeros((n, 2), dtype=int),
                  touching_point_count=np.zeros((n, 2), dtype=int),
                  velocity_point_count=np.zeros((n, 2), dtype=int),
                  invalid_geometry_point_count=np.zeros((n, 2), dtype=int))
    names = trace["body_names"].astype(str).tolist()
    origin = trace["environment_origin_w"]
    for side, name in enumerate(("wheel_L_Link", "wheel_R_Link")):
        bid = names.index(name)
        axles = q_matrix(trace["body_quat"][:, bid])[..., :, 1]
        for row, i in enumerate(raw["physical_sample_indices"]):
            count = int(raw[f"w{side}_normal_buffer_4"][row].item())
            start = int(raw[f"w{side}_normal_buffer_5"][row].item())
            buffer = raw[f"w{side}_normal_buffer_3"][row]
            if min(count, start) < 0 or start + count > len(buffer):
                raise ValueError("Contact buffer range invalid")
            result["contact_point_count"][i, side] = count
            if not count:
                continue
            distance = buffer[start:start+count].reshape(-1)
            if not np.isfinite(distance).all():
                raise ValueError("Nonfinite native separation")
            points = raw[f"w{side}_normal_buffer_1"][row, start:start+count].astype(float) - origin
            normal = raw[f"w{side}_normal_buffer_2"][row, start:start+count].astype(float)
            length = np.linalg.norm(normal, axis=-1, keepdims=True)
            if not np.isfinite(points).all() or not np.isfinite(normal).all():
                raise ValueError("Invalid touching-point geometry")
            geometry_valid = length[:, 0] >= 1e-8
            result["invalid_geometry_point_count"][i, side] = (~geometry_valid).sum()
            if not geometry_valid.any():
                # All-zero native entries are not evidence of zero separation.
                continue
            result["min_separation_m"][i, side] = distance[geometry_valid].min()
            touching = geometry_valid & (distance <= tolerance_m)
            result["touching_point_count"][i, side] = touching.sum()
            result["velocity_point_count"][i, side] = touching.sum()
            if not touching.any():
                continue
            points, normal, length = points[touching], normal[touching], length[touching]
            normal /= length
            com = trace["body_com_pos"][i, bid].astype(float)
            velocity = trace["body_com_vel"][i, bid].astype(float)
            omega = trace["body_omega"][i, bid].astype(float)
            vp = velocity + np.cross(omega, points - com)
            tangent = vp - (vp * normal).sum(-1, keepdims=True) * normal
            result["tangent_rms_m_s"][i, side] = np.sqrt(np.mean(np.sum(tangent**2, axis=-1)))
            lateral = axles[i] - (normal @ axles[i])[:, None] * normal
            length = np.linalg.norm(lateral, axis=-1, keepdims=True)
            if (length > 1e-8).all():
                lateral /= length
                result["lateral_rms_m_s"][i, side] = np.sqrt(np.mean(np.sum(vp * lateral, axis=-1)**2))
    result["touching"] = result["touching_point_count"] > 0
    result["contact_unknown"] = ~result["touching"] & (result["invalid_geometry_point_count"] > 0)
    return result


def contact_summary(t, valid, touching, unknown=None):
    unknown = np.zeros_like(touching) if unknown is None else unknown
    unknown_any = valid & ~touching.any(-1) & unknown.any(-1)
    gap = valid & ~touching.any(-1) & ~unknown.any(-1)
    events = segments(t, gap, interval_samples=True)
    dt = float(np.median(np.diff(t)))
    return dict(continuous_anywheel_contact=bool(touching[valid].any(-1).all()),
                anywheel_contact_fraction=float(touching[valid].any(-1).mean()),
                per_wheel_contact_fraction=touching[valid].mean(0).tolist(),
                total_no_contact_s=float(gap.sum() * dt), gap_count=len(events),
                unknown_anywheel_contact_s=float(unknown_any.sum() * dt),
                longest_no_contact_s=max((e["duration_s"] for e in events), default=0.), events=events)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/hockey_contact_evaluation_v2.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    names = ["beyondmimic_s100_seed42", "hockey_s100_seed42", "hockey_s150_seed42", "hockey_s200_seed42"]
    report = dict(kind=cfg["kind"], config=cfg, config_sha256=sha(args.config), source_sha256=sha(__file__),
                  dependency_sha256={name: sha(Path(__file__).with_name(name)) for name in
                                     ("analyze_hockey_support_failure.py", "evaluate_hockey_physx.py")},
                  scope="Four nominal recordings; contact eligibility and contact-conditioned sliding diagnostics",
                  full_task_success_recomputed=False, perturbed_success_rates_recomputed=False,
                  training_recipe_changed=False, physics_rerun=False, cells=[])
    recorded = []
    for name in names:
        cell = args.runs / name
        checked = read_support_case(cell)
        trace = load(cell / "eval-suite/nominal/final_rollout.npz")
        raw = load(cell / "eval-suite/nominal/native_contact_buffers.npz")
        refdir = args.references / name.split("_")[1]
        assert sha(refdir / "motion.npz") == checked["run"]["motion_sha256"]
        motion = load(refdir / "motion.npz")
        end = (len(motion["joint_pos"]) - 1) / float(motion["fps"])
        t = checked["signals"]["time_s"]
        dt = float(trace["physics_dt_s"])
        valid = checked["signals"]["native_sample_valid"] & (t-dt >= cfg["exclude_initial_s"]-1e-9) & (t <= end+1e-9)
        result = contact_kinematics(trace, raw, cfg["contact_separation_tolerance_m"])
        force = checked["signals"]["normal_fz"]
        result["capacity_N"] = .7 * force
        result["friction_utilization"] = np.full(force.shape, np.nan)
        usable = result["touching"] & (force > cfg["force_division_epsilon_N"])
        np.divide(np.linalg.norm(checked["signals"]["friction_force"][..., :2], axis=-1), .7*force,
                  out=result["friction_utilization"], where=usable)
        result["sliding"] = result["touching"] & (result["tangent_rms_m_s"] > cfg["sliding_speed_threshold_m_s"])
        result["lateral_sliding"] = result["touching"] & (result["lateral_rms_m_s"] > cfg["sliding_speed_threshold_m_s"])
        result["friction_corroborated_lateral_sliding"] = (result["lateral_sliding"] & usable &
            (result["friction_utilization"] >= .9) & (result["friction_utilization"] <= 1.05) &
            (checked["signals"]["anchor_power"] < -.05) & (checked["signals"]["dissipative_slipping_anchor_count"] >= 1))
        sensitivity = []
        for tolerance in cfg["contact_separation_sensitivity_m"]:
            touch = (result["contact_point_count"] > 0) & (result["min_separation_m"] <= tolerance)
            unknown = ~touch & (result["invalid_geometry_point_count"] > 0)
            sensitivity.append(dict(tolerance_m=tolerance, **contact_summary(t, valid, touch, unknown)))
        old_gap = valid & ~(force > 5).any(-1)
        result["time_s"] = t
        result["valid"] = valid
        summary = dict(cell=name, provenance=checked["provenance"], native_force_audit=checked["native_audit"],
            **contact_summary(t, valid, result["touching"], result["contact_unknown"]), tolerance_sensitivity=sensitivity,
            manifold_count_only=contact_summary(t, valid, result["contact_point_count"] > 0),
            former_force_gap_s=float(old_gap.sum()*dt),
            former_force_gap_still_touching_s=float((old_gap & result["touching"].any(-1)).sum()*dt),
            per_wheel_touching_with_zero_load_s=((result["touching"] & (force <= 1e-6) & valid[:, None]).sum(0)*dt).tolist(),
            per_wheel_contact_unknown_s=((result["contact_unknown"] & valid[:, None]).sum(0)*dt).tolist(),
            per_wheel_contact_sliding_s=((result["sliding"] & valid[:, None]).sum(0)*dt).tolist(),
            per_wheel_friction_corroborated_lateral_sliding_s=((result["friction_corroborated_lateral_sliding"] & valid[:, None]).sum(0)*dt).tolist(),
            material_velocity_point_weighting="Unweighted RMS over points meeting separation tolerance with a valid native normal; no force threshold or force weighting",
            limitations=["Native separation is collision geometry evidence, not an independently measured surface gap.",
                "Forces and separation belong to the completed substep; rigid twist is sampled after that step.",
                "Friction corroboration is wheel aggregate plus separate friction-anchor dissipation, not a pointwise cone certificate.",
                "Zero-normal entries, including all-zero placeholders, cannot prove contact or separation; absent another valid touching point, the state is unknown.",
                "Contact continuity alone is not full stop-task success; old force-gated slide/stop/heading gates are not reused as new success rates."])
        report["cells"].append(summary)
        recorded.append(result)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "contact_signals.npz", **{name+"__"+key:value for name,r in zip(names,recorded) for key,value in r.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(23, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, name, r, c in zip(axes, names, recorded, report["cells"]):
        ax.axvspan(.4, 1.8, color="gray", alpha=.08, label="Braking")
        for side, color in enumerate(("#2874b2", "#d77325")):
            ax.plot(r["time_s"], 1000*r["min_separation_m"][:, side], color=color, lw=.9, label=("Left", "Right")[side])
        for i,e in enumerate(c["events"]):
            ax.axvspan(e["start_s"], e["end_s"], color="crimson", alpha=.25, label="Neither wheel touching" if i == 0 else None)
        ax.axhline(1000*cfg["contact_separation_tolerance_m"], color="black", ls="--", lw=.8, label="0.1 mm tolerance")
        ax.set(xlim=(.02,2.76), ylim=(-.2,30), yscale="symlog", xlabel="Time [s]",
               title=name.replace("_seed42", "").replace("_", " ")+f"\nNo contact: total {1000*c['total_no_contact_s']:.1f}, max {1000*c['longest_no_contact_s']:.1f} ms")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Minimum native point separation [mm], symlog scale")
    fig.suptitle("Contact-first evaluation | Native points + separation <=0.1 mm; no load threshold | PhysX, mu=0.7, seed42")
    fig.savefig(args.output / "ground_contact_comparison_4x1.png", dpi=160)
    plt.close(fig)
    print(json.dumps([{k:c[k] for k in ("cell", "anywheel_contact_fraction", "total_no_contact_s", "longest_no_contact_s", "former_force_gap_still_touching_s")} for c in report["cells"]]))


if __name__ == "__main__":
    main()
