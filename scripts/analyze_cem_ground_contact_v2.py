"""Read-only full nominal CEM V2 audit; no dynamics, inference, tuning or alignment.

An episode-level protocol screen is not completion of the wider PPO/engine
comparison or the required qualitative whole-body review.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as metrics
from controller_metrics import heading_signals
from controller_trace import native_trace
from ppo_backend_400hz_contract import require, sha256, read_json, write_json
import run_cem_ground_contact_v2 as runner

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL=ROOT/"config/ppo_ground_contact_v2_400hz_v1.json"
BASELINE_SUMMARY=ROOT/"results/ppo_cem_matched_reward_400hz_20261022_v1/summary.json"
BASELINE_FOLDER=ROOT/"outputs/ppo_cem_matched_reward_400hz_20261022_v1/beta_on"


def support_summary(time, force_z, interval, threshold=5.):
    time=np.asarray(time,dtype=float);force=np.asarray(force_z,dtype=float)
    require(time.ndim==1 and force.shape==(len(time),2) and len(time)>1
        and np.isfinite(time).all() and np.isfinite(force).all() and np.all(np.diff(time)>0),"Invalid support clock/forces")
    selected=(time>=interval[0]-1e-9)&(time<=interval[1]+1e-9)
    t=time[selected];fz=force[selected]
    require(len(t)>1 and abs(t[0]-interval[0])<1e-8 and abs(t[-1]-interval[1])<1e-8,"Support window incomplete")
    supported=np.max(fz,axis=1)>threshold
    unsupported=~supported
    starts=np.flatnonzero(unsupported & np.r_[True,~unsupported[:-1]])
    ends=np.flatnonzero(unsupported & np.r_[~unsupported[1:],True])+1
    gaps=[]
    for start,end in zip(starts,ends):
        finish=float(t[end]) if end<len(t) else float(t[-1])
        gaps.append({"onset_s":float(t[start]),"end_s":finish,"observed_duration_s":max(0.,finish-float(t[start])),
                     "native_samples":int(end-start),"right_window_censored":bool(end==len(t)),
                     "left_window_censored":bool(start==0)})
    duration=float(t[-1]-t[0]); widths=np.diff(t)
    return {"interval_s":list(interval),"force_threshold_n_strict":threshold,
        "definition":"Any wheel upward actual floor-normal Fz>threshold; no bilateral or count requirement",
        "native_samples":len(t),"any_wheel_support_sample_fraction":float(supported.mean()),
        "any_wheel_support_time_fraction_zoh":float(np.sum(widths*supported[:-1])/duration),
        "unsupported_physical_samples":int(unsupported.sum()),"every_physical_sample_supported":bool(supported.all()),
        "longest_no_support_observed_s":max((g["observed_duration_s"] for g in gaps),default=0.),
        "no_support_gaps":gaps,"minimum_anywheel_normal_force_n":float(np.max(fz,axis=1).min()),
        "single_wheel_lift_is_failure":False,"flight_or_height_inferred":False,
        "last_sample_no_future_extension":True}


def recorded_metrics(raw,reference):
    trace=native_trace(raw,"nmpc")
    require(trace["time_s"].shape==(2793,) and trace["physics_dt_s"]==.0025,"Require completed2793state400Hz recording")
    body_quat=np.asarray(raw["body_quat_w"])
    signals=heading_signals(trace["quaternion"],trace["com_velocity"],body_quat[:,-2:],np.zeros((2793,2,3)))
    windows={name:metrics.window_metrics(trace,reference,signals,interval) for name,interval in
             (("core",(1.44,4.)),("whole_reference_overlap",(0.,6.96)),("full_native",(0.,6.98)))}
    native_force=np.asarray(raw.get("ground_normal_force_w",raw["wheel_ground_contact_force_w_n"]))
    support={"core":support_summary(trace["time_s"],native_force[...,2],(1.44,4.)),
             "full_contact":support_summary(trace["time_s"],native_force[...,2],(.02,6.96))}
    return {"windows":windows,"any_wheel_ground_support":support,
        "nonzero_beta_is_intended":True,"height_contact_tilt_or_beta_used_as_XY_failure_gate":False,
        "whole_tracking_reference_end_s":6.96,"physical_recording_end_s":6.98,
        "trajectory_aligned_smoothed_or_extrapolated":False}


def nominal_screen(candidate,baseline,protocol):
    c=protocol["nominal_acceptance_before_training"]
    core=candidate["windows"]["core"]["tracking"]
    whole=candidate["windows"]["whole_reference_overlap"]["tracking"]
    baseline_hf=baseline["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]
    candidate_hf=candidate["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]
    ratio=candidate_hf/baseline_hf if baseline_hf>0 else None
    contact=candidate["any_wheel_ground_support"]["full_contact"]
    checks={"core_xy_rmse":core["xy_rmse_m"]<=c["core_and_whole_xy_rmse_m_max"],
        "core_xy_peak":core["xy_peak_m"]<=c["core_and_whole_xy_peak_m_max"],
        "whole_xy_rmse":whole["xy_rmse_m"]<=c["core_and_whole_xy_rmse_m_max"],
        "whole_xy_peak":whole["xy_peak_m"]<=c["core_and_whole_xy_peak_m_max"],
        "core_speed_retention":c["core_speed_ratio_range"][0]<=core["speed_retention"]<=c["core_speed_ratio_range"][1],
        "core_reference_beta":core["beta_rmse_deg"] is not None and core["beta_rmse_deg"]<=c["core_reference_beta_rmse_deg_max"],
        "any_wheel_every_physical_sample":contact["every_physical_sample_supported"]
            and contact["any_wheel_support_sample_fraction"]>=c["any_wheel_support_fraction_min"],
        "longest_no_support_gap":contact["longest_no_support_observed_s"]<=c["longest_no_support_interval_s_max"],
        "core_HF_reduction":ratio is not None and ratio<=c["core_10_to_100hz_angular_xy_rms_ratio_to_original_baseline_max"]}
    return {"criteria":copy.deepcopy(c),"checks":checks,"failed_checks":[name for name,value in checks.items() if not value],
        "numeric_nominal_episode_pass":bool(all(checks.values())),"core_HF_ratio_to_old_CEM":ratio,
        "baseline":"original400Hz CEM+beta nativeMuJoCo mu=.7; controller-matched oldcontactreward",
        "qualitative_whole_body_review_complete":False,"both_PPO_validation_backends_complete":False,
        "overall_task_goal_complete_claimed":False,
        "note":"This is the CEM-only numerical screen; completion also needs PPO/engine validation and qualitative review."}


def load_completed(folder):
    folder=Path(folder).resolve();report_path=folder/"report.json";raw_path=folder/"rollout.npz";log_path=folder/"solver_log.json"
    hashes={name:sha256(folder/name) for name in ("report.json","rollout.npz","solver_log.json","experiment.json")}
    report,experiment=read_json(report_path),read_json(folder/"experiment.json")
    require(report["status"]==experiment["status"]=="completed" and report["smoke_only"] is False
        and report["completed_policy_steps"]==349 and report["native_samples"]==2793
        and report["completed_full_reference"] is True,"Only terminal complete349-step V2 records may be scored")
    require(experiment["report_sha256"]==hashes["report.json"] and report["rollout_sha256"]==hashes["rollout.npz"]
        and report["solver_log_sha256"]==hashes["solver_log.json"],"Terminal report/raw/log bindings differ")
    require(report["kind"]=="400hz_CEM_ground_contact_reward_v2a_anywheel" and report["plant_mu"]==.7
        and report["prediction_friction_known"] is True and report["condition"]=="beta_on"
        and report["termination_or_reset_after_initialization"] is False and report["floor_filter_verified"] is True,
        "Wrong V2 controller/material/reset/floor contract")
    for name,digest in report["source_sha256"].items():require(sha256(ROOT/name)==digest,"Executed source changed: "+name)
    with np.load(raw_path,allow_pickle=False) as archive:raw={name:archive[name].copy() for name in archive.files}
    decisions=json.loads(log_path.read_text())
    require(len(decisions)==349,"Incomplete solver decisions")
    reward_audit=runner.recompute_actual_rewards(raw,report,decisions)
    require(reward_audit==report["actual_reward_audit"],"Published reward audit differs")
    require(all(sha256(folder/name)==digest for name,digest in hashes.items()),"Terminal files changed during analysis")
    return raw,report,hashes,reward_audit


def run(args):
    output=Path(args.output_dir);require(not output.exists(),"Preserve analysis; fresh output directory required")
    raw,report,hashes,reward_audit=load_completed(args.input_dir)
    protocol_path=Path(args.protocol);protocol=read_json(protocol_path)
    reference=metrics.reference_data(runner.paired.DEFAULT_REFERENCE)
    require(report["motion_sha256"]==sha256(runner.paired.DEFAULT_REFERENCE),"Changed motion reference")
    baseline_summary=read_json(BASELINE_SUMMARY)
    baseline_row=next(row for row in baseline_summary["rows"] if row["id"]=="cem_beta_on")
    baseline_hashes={"report.json":sha256(BASELINE_FOLDER/"report.json"),"rollout.npz":sha256(BASELINE_FOLDER/"rollout.npz")}
    require(baseline_hashes["report.json"]==baseline_row["report_sha256"]
        and baseline_hashes["rollout.npz"]==baseline_row["recording_sha256"],"Published original CEM baseline binding differs")
    with np.load(BASELINE_FOLDER/"rollout.npz",allow_pickle=False) as archive:
        baseline_raw={name:archive[name].copy() for name in archive.files}
    new_metrics,old_metrics=recorded_metrics(raw,reference),recorded_metrics(baseline_raw,reference)
    for name,original_name in (("core","core"),("whole_reference_overlap","reference_overlap")):
        require(old_metrics["windows"][name]["tracking"]==baseline_row["windows"][original_name]["tracking"],
                "Recomputed original baseline tracking differs")
    summary={"kind":"CEM_ground_contact_V2_nominal_readonly_audit","status":"completed_analysis",
        "input_directory":str(Path(args.input_dir).resolve()),"input_sha256":hashes,"reward_audit":reward_audit,
        "baseline_input_sha256":baseline_hashes,"baseline_summary_sha256":sha256(BASELINE_SUMMARY),
        "protocol_sha256":sha256(protocol_path),"analyzer_sha256":sha256(__file__),"sample_size_per_case":1,
        "new_V2":new_metrics,"original_grounded_baseline":old_metrics,
        "nominal_screen":nominal_screen(new_metrics,old_metrics,protocol),
        "runtime":{"solver_failures":report["solver_failures"],"plant_warning_counts":report["plant_warning_counts"],
                   "compute_ms_median_p95_max":report["compute_ms_median_p95_max"],"real_time_claimed":False},
        "no_new_dynamics_policy_inference_or_training":True,
        "limitations":["N=1: no statistical qualification or universal friction threshold.",
            "HF reduction alone does not establish balance or correct contact.",
            "Any-wheel support uses actual floor-normal Fz>5N only; the other wheel may lift.",
            "No contact/height/tilt/beta failure gate changes the independent XY error metric.",
            "Only the CEM nominal numerical arm is audited; full goal completion is not asserted."]}
    output.mkdir(parents=True,exist_ok=False);write_json(output/"summary.json",summary)
    return summary


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-dir",type=Path,required=True)
    result.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL)
    result.add_argument("--output-dir",type=Path,required=True)
    return result


if __name__=="__main__":run(parser().parse_args())
