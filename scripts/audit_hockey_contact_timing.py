"""Audit contact timestamp alignment and recover same-time sliding evidence.

The archived native separation is compared with the locked USD cylinder at
five state offsets. Contact geometry at row i matches pose i-1. Compare legacy
post-step fixed-world-point velocity with aligned pre-step velocity and actual
material-point displacement over the step. No simulation or policy is changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_hockey_ground_contact import contact_kinematics
from analyze_hockey_support_failure import read_support_case
from evaluate_hockey_physx import load, q_matrix, segments, sha
from wheel_ground_geometry import cylinder_support

CELLS = ("beyondmimic_s100_seed42", "hockey_s100_seed42", "hockey_s150_seed42", "hockey_s200_seed42")


def previous_state_trace(trace):
    out = dict(trace)
    for key in ("body_com_pos", "body_com_vel", "body_omega", "body_quat"):
        out[key] = np.concatenate((trace[key][:1], trace[key][:-1]), axis=0)
    return out


def material_step_velocity(point, before_com, after_com, before_rotation, after_rotation, dt):
    """Transport the same wheel material point; never differentiate moving contact minima."""
    local = (point-before_com) @ before_rotation
    after_point = after_com + local @ after_rotation.T
    return (after_point-point)/dt


def vector_signals(trace, raw, tolerance):
    n = len(trace["root_pos"]); dt = float(trace["physics_dt_s"])
    result = {key: np.full((n, 2, 3), np.nan) for key in
              ("contact_point", "slip_velocity_pre", "slip_velocity_interval")}
    result["interval_tangent_rms"] = np.full((n, 2), np.nan)
    result["anchor_power_pre"] = np.zeros((n, 2))
    result["opposing_lateral_anchors_pre"] = np.zeros((n, 2), int)
    for side, name in enumerate(("wheel_L_Link", "wheel_R_Link")):
        bid = trace["body_names"].astype(str).tolist().index(name)
        R = q_matrix(trace["body_quat"][:, bid])
        for row, i in enumerate(raw["physical_sample_indices"]):
            before = i-1
            com, vel, omega = [trace[key][before, bid].astype(float) for key in
                               ("body_com_pos", "body_com_vel", "body_omega")]
            count = int(raw[f"w{side}_normal_buffer_4"][row].item())
            start = int(raw[f"w{side}_normal_buffer_5"][row].item())
            sl = slice(start, start+count)
            normals = raw[f"w{side}_normal_buffer_2"][row, sl].astype(float)
            distances = raw[f"w{side}_normal_buffer_3"][row, sl, 0]
            valid = (np.linalg.norm(normals, axis=-1) >= 1e-8) & (distances <= tolerance)
            if valid.any():
                points = raw[f"w{side}_normal_buffer_1"][row, sl][valid].astype(float)-trace["environment_origin_w"]
                normal = normals[valid]; normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
                pre = vel + np.cross(omega, points-com)
                interval = material_step_velocity(points, com, trace["body_com_pos"][i, bid], R[before], R[i], dt)
                pre -= (pre*normal).sum(-1, keepdims=True)*normal
                interval -= (interval*normal).sum(-1, keepdims=True)*normal
                result["contact_point"][i, side] = points.mean(0)
                result["slip_velocity_pre"][i, side] = pre.mean(0)
                result["slip_velocity_interval"][i, side] = interval.mean(0)
                result["interval_tangent_rms"][i, side] = np.sqrt(np.mean(np.sum(interval**2, axis=-1)))
            count = int(raw[f"w{side}_friction_buffer_2"][row].item())
            start = int(raw[f"w{side}_friction_buffer_3"][row].item())
            sl = slice(start, start+count)
            force = raw[f"w{side}_friction_buffer_0"][row, sl].astype(float)
            points = raw[f"w{side}_friction_buffer_1"][row, sl].astype(float)-trace["environment_origin_w"]
            velocity = vel + np.cross(omega, points-com)
            power = (force*velocity).sum(-1)
            axis = R[before, :, 1].copy(); axis[2] = 0
            axis /= max(np.linalg.norm(axis), 1e-8)
            vl, fl = velocity @ axis, force @ axis
            result["anchor_power_pre"][i, side] = power.sum()
            result["opposing_lateral_anchors_pre"][i, side] = ((abs(vl)>.05)&(vl*fl<0)&(power<0)).sum()
    return result


def quantiles(x):
    return np.quantile(x, [.5, .95, 1]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    geometry = json.loads(a.geometry.read_text())
    assert sha(a.asset) == geometry["asset_sha256"]
    result = dict(kind="hockey_contact_timing_audit_v3", script_sha256=sha(__file__),
        dependencies={name:sha(Path(__file__).with_name(name)) for name in
                      ("analyze_hockey_ground_contact.py", "analyze_hockey_support_failure.py", "evaluate_hockey_physx.py", "wheel_ground_geometry.py")},
        geometry_report_sha256=sha(a.geometry), asset_sha256=sha(a.asset),
        contact_tolerance_m=.0001, primary_contact_velocity="pre-step state at the native contact geometry timestamp",
        force_scope="Native force is averaged over the following 2.5 ms step; anchor power remains a corroboration proxy, not exact integrated work.",
        conclusions_scope="Recorded traces establish timestamp alignment and sliding evidence; no new engine calibration or training/evaluation action-parity experiment performed.",
        physics_steps=0, training_changed=False, historical_v2_outputs_preserved=True, cells=[])
    saved = {}
    for name in CELLS:
        cell = a.runs / name; checked = read_support_case(cell)
        tr = load(cell/"eval-suite/nominal/final_rollout.npz")
        raw = load(cell/"eval-suite/nominal/native_contact_buffers.npz")
        legacy = contact_kinematics(tr, raw, .0001)
        aligned = contact_kinematics(previous_state_trace(tr), raw, .0001)
        aligned.update(vector_signals(tr, raw, .0001))
        n = len(tr["root_pos"]); te = np.arange(n)*.0025; ts = te-.0025
        valid = (ts>=.02-1e-9)&(te<=2.76+1e-9)
        brake = valid&(ts>=.4-1e-9)&(te<=1.8+1e-9)
        pre = np.maximum(np.arange(n)-1, 0)
        ids = [tr["body_names"].astype(str).tolist().index("wheel_"+side+"_Link") for side in ("L","R")]
        force = checked["signals"]["normal_fz"]; ft = checked["signals"]["friction_force"]
        rho = np.full(force.shape, np.nan)
        np.divide(np.linalg.norm(ft[...,:2], axis=-1), .7*force, out=rho,
                  where=aligned["touching"]&(force>1e-6))
        sliding = aligned["touching"]&(aligned["lateral_rms_m_s"]>.05)
        corroborated = sliding&(rho>=.9)&(rho<=1.05)&(aligned["anchor_power_pre"]<-.05)&(aligned["opposing_lateral_anchors_pre"]>=1)
        summary = dict(cell=name, provenance=checked["provenance"], wheels=[],
            no_contact_count_changed=False, full_task_success_recomputed=False)
        assert np.array_equal(legacy["touching"], aligned["touching"])
        for side, bid in enumerate(ids):
            g=geometry["geometry"][side];R=q_matrix(tr["body_quat"][:,bid])
            center=tr["body_pos"][:,bid]+R@np.array(g["USD_center_in_link_m"])
            axis=R@np.array(g["USD_rotation_cylinder_to_link"])[:,2]
            gap=cylinder_support(center,axis,g["USD_radius_m"],g["USD_width_m"]/2)["authored_signed_gap_m"]
            sep=aligned["min_separation_m"][:,side]
            indices=np.flatnonzero(np.isfinite(sep)&(te>=.05)&(te<=2.75))
            indices=indices[(indices>=2)&(indices<n-2)]
            errors={str(lag):dict(mean_abs_error_m=float(np.mean(abs(sep[indices]-gap[indices+lag]))),
                                 p95_abs_error_m=float(np.quantile(abs(sep[indices]-gap[indices+lag]),.95))) for lag in (-2,-1,0,1,2)}
            best=min(errors,key=lambda k:errors[k]["mean_abs_error_m"])
            if best!="-1" or errors[best]["p95_abs_error_m"]>1e-6:
                raise ValueError("The independent geometry does not establish the expected timestamp")
            contact=brake&aligned["touching"][:,side]
            old=legacy["tangent_rms_m_s"][:,side];new=aligned["tangent_rms_m_s"][:,side]
            c=corroborated[:,side]&brake
            events=segments(te,c,interval_samples=True)
            fd=np.diff(tr["body_com_pos"][:,bid],axis=0)/.0025
            rotation_delta=R[1:]@np.swapaxes(R[:-1],1,2)
            wfd=Rotation.from_matrix(rotation_delta).as_rotvec()/.0025
            fdmask=valid[1:]
            summary["wheels"].append(dict(side=("left","right")[side],native_gap_pose_alignment=errors,
                best_pose_index_offset=-1, geometry_samples_compared=len(indices),
                braking_contact_s=float(contact.sum()*.0025),
                braking_lateral_sliding_legacy_s=float((contact&(legacy["lateral_rms_m_s"][:,side]>.05)).sum()*.0025),
                braking_lateral_sliding_aligned_s=float((contact&sliding[:,side]).sum()*.0025),
                braking_friction_corroborated_lateral_sliding_s=float(c.sum()*.0025),
                longest_corroborated_lateral_sliding_s=max((e["duration_s"] for e in events),default=0.),
                corroborated_sliding_events=events,
                aligned_contact_lateral_speed_p50_p95_max_m_s=quantiles(aligned["lateral_rms_m_s"][contact,side]),
                aligned_contact_tangent_speed_p50_p95_max_m_s=quantiles(new[contact]),
                interval_material_tangent_speed_p50_p95_max_m_s=quantiles(aligned["interval_tangent_rms"][contact,side]),
                legacy_vs_aligned_tangent_difference_p50_p95_max_m_s=quantiles(abs(old[contact]-new[contact])),
                legacy_chart_above_3m_s_fraction=float((old[contact]>3).mean()),
                backward_FD_vs_post_COM_velocity_error_p50_p95_max_m_s=quantiles(np.linalg.norm(fd[fdmask]-tr["body_com_vel"][1:,bid][fdmask],axis=-1)),
                rotation_FD_vs_post_omega_error_p50_p95_max_rad_s=quantiles(np.linalg.norm(wfd[fdmask]-tr["body_omega"][1:,bid][fdmask],axis=-1))))
        aligned.update(time_start_s=ts,time_end_s=te,valid=valid,braking=brake,
            rho=rho,lateral_sliding=sliding,friction_corroborated_lateral_sliding=corroborated,
            root_pos_pre=tr["root_pos"][pre],root_quat_pre=tr["root_quat"][pre],
            wheel_pos_pre=tr["body_pos"][pre][:,ids],wheel_quat_pre=tr["body_quat"][pre][:,ids],
            wheel_com_velocity_pre=tr["body_com_vel"][pre][:,ids],normal_force_N=force,
            legacy_tangent_rms=legacy["tangent_rms_m_s"],legacy_lateral_rms=legacy["lateral_rms_m_s"])
        saved.update({name+"__"+key:value for key,value in aligned.items()})
        result["cells"].append(summary)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"aligned_signals.npz",**saved)
    result["aligned_signals_sha256"]=sha(a.output/"aligned_signals.npz")
    (a.output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{ "cell":c["cell"],"wheels":[{k:w[k] for k in
        ("side","best_pose_index_offset","braking_lateral_sliding_aligned_s","braking_friction_corroborated_lateral_sliding_s",
         "longest_corroborated_lateral_sliding_s")} for w in c["wheels"]]} for c in result["cells"]]))


if __name__ == "__main__":
    main()
