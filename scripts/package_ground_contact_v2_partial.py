"""Publish a source/SHA-bound PARTIAL experiment record; never claim PPO results.

Copies only named simulation records/graphics and pure source. No models,
trained weights, credentials, native binaries, remote access or running-job edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

import analyze_cem_ground_contact_v2 as audit
from ppo_backend_400hz_contract import require,read_json,write_json,sha256

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_ROOT=ROOT/"outputs/ground_contact_v2_20261023_v1"
FIELDS=("time_s","physics_dt_s","root_pos","root_quaternion_wxyz","base_com_velocity_w",
    "base_angular_velocity_w","joint_pos","joint_vel","wheel_pos","action_joint_names",
    "wheel_ground_contact_count","wheel_ground_contact_force_w_n","wheel_lateral_contact_slip_m_s","body_quat_w")


def comparison_arrays(new_raw,old_raw,reference):
    result={"reference_"+name:np.asarray(value).copy() for name,value in reference.items() if isinstance(value,np.ndarray)}
    for key,raw in (("new_V2",new_raw),("original_grounded_baseline",old_raw)):
        for name in FIELDS:
            result[key+"__"+name]=np.asarray(raw[name]).copy()
        if "ground_normal_force_w" in raw:
            result[key+"__ground_normal_force_w"]=np.asarray(raw["ground_normal_force_w"]).copy()
    return result


def recompute_public_metrics(path):
    """No robot assets/weights/dynamics needed for native metric reproduction."""
    with np.load(path,allow_pickle=False) as archive:arrays={name:archive[name].copy() for name in archive.files}
    reference={name[10:]:value for name,value in arrays.items() if name.startswith("reference_")}
    result={}
    for case in ("new_V2","original_grounded_baseline"):
        prefix=case+"__";raw={name[len(prefix):]:value for name,value in arrays.items() if name.startswith(prefix)}
        result[case]=audit.recorded_metrics(raw,reference)
    return result


def copy_bound(source,destination,expected=None):
    source,destination=Path(source),Path(destination)
    require(source.is_file() and not destination.exists(),"Require existing source and fresh destination")
    digest=sha256(source)
    require(expected is None or digest==expected,"Source differs from analyzed artifact: "+str(source))
    shutil.copyfile(source,destination)
    require(sha256(source)==digest==sha256(destination),"Copied artifact changed during packaging")
    return {"source":str(source),"file":destination.name,"sha256":digest,"bytes":destination.stat().st_size}


def source_archive(output,ledger):
    paths=[]
    for name,digest in ledger.items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and path.suffix in (".py",".json",".cpp"),
                "Source archive accepts only allowlisted pure code/config inside repository")
        require(sha256(path)==digest,"Executed source changed: "+name)
        paths.append((name,path))
    target=output/"executed_source.tar.gz"
    with tarfile.open(target,"x:gz") as archive:
        for name,path in sorted(paths):archive.add(path,arcname=name,recursive=False)
    return {"file":target.name,"sha256":sha256(target),"pure_source_files":len(paths),
        "models_weights_credentials_native_libraries_included":False,
        "extraction":"Restore only into a fresh isolated checkout, never over a dirty working tree."}


def run(args):
    output=Path(args.output_dir);require(not output.exists(),"Preserve published records; fresh output directory required")
    root=Path(args.input_root);analysis_folder=root/"cem_nominal_analysis_v1";render_folder=root/"cem_render_v1"
    summary=read_json(analysis_folder/"summary.json");render=read_json(render_folder/"provenance.json")
    raw,report,hashes,reward_audit=audit.load_completed(root/"cem_mu0p7")
    require(summary["input_sha256"]==hashes and summary["reward_audit"]==reward_audit,"Analysis and terminal CEM bindings differ")
    require(summary["nominal_screen"]["numeric_nominal_episode_pass"] is False
        and set(summary["nominal_screen"]["failed_checks"])=={"any_wheel_every_physical_sample","longest_no_support_gap"},
        "This partial record must retain its observed contact failure, not relabel it")
    require(render["new_physics_policy_inference_training"] is False and render["video_generated"] is True
        and render["stable_contact_or_goal_achieved_claimed"] is False,"Wrong recorded-only complete render")
    for name,digest in render["artifacts"].items():require(sha256(render_folder/name)==digest,"Rendered artifact changed")
    with np.load(audit.BASELINE_FOLDER/"rollout.npz",allow_pickle=False) as archive:
        old_raw={name:archive[name].copy() for name in archive.files}
    require(sha256(audit.BASELINE_FOLDER/"rollout.npz")==summary["baseline_input_sha256"]["rollout.npz"],"Original CEM baseline changed")
    reference=audit.metrics.reference_data(audit.runner.paired.DEFAULT_REFERENCE)
    output.mkdir(parents=True,exist_ok=False)
    copies={}
    for source,name,expected in (
        (analysis_folder/"summary.json","cem_analysis.json",None),
        (render_folder/"comparison.png","comparison.png",render["artifacts"]["comparison.png"]),
        (render_folder/"fullbody_comparison.mp4","fullbody_comparison.mp4",render["artifacts"]["fullbody_comparison.mp4"]),
        (render_folder/"replay_states.npz","replay_states.npz",render["artifacts"]["replay_states.npz"]),
        (render_folder/"provenance.json","render_provenance.json",None),
        (root/"cem_mu0p7/report.json","cem_report.json",hashes["report.json"]),
        (root/"cem_mu0p7/experiment.json","cem_experiment.json",hashes["experiment.json"]),
        (root/"cem_launch.json","cem_launch.json",None),
        (audit.DEFAULT_PROTOCOL,"protocol.json",summary["protocol_sha256"])):
        copies[name]=copy_bound(source,output/name,expected)
    np.savez_compressed(output/"native_comparison_signals.npz",**comparison_arrays(raw,old_raw,reference))
    recomputed=recompute_public_metrics(output/"native_comparison_signals.npz")
    require(all(recomputed[name]==summary[name] for name in recomputed),"Public compact signals do not exactly reproduce metrics")
    ledger=dict(report["source_sha256"])
    for name in ("scripts/package_ground_contact_v2_partial.py","scripts/analyze_cem_ground_contact_v2.py",
                 "scripts/render_ground_contact_v2.py","scripts/render_ppo_fixed_follow.py",
                 "scripts/render_physx_ppo_fullbody.py","scripts/render_ppo_backend_400hz_nominal.py",
                 "scripts/analyze_ppo_backend_400hz_matrix.py","scripts/controller_trace.py","scripts/controller_metrics.py",
                 "config/ppo_ground_contact_v2_400hz_v1.json"):
        ledger[name]=sha256(ROOT/name)
    archive=source_archive(output,ledger)
    observed=datetime.now(timezone.utc).isoformat()
    support=summary["new_V2"]["any_wheel_ground_support"]
    status={"status":"PARTIAL_EXPERIMENT_NOT_GOAL_COMPLETION","snapshot_utc":observed,
        "task":"traj4 high-speed large turn with at least one continuously supporting wheel",
        "conditions":{"mu":.7,"physics_hz":400,"policy_hz":50,"beta_reward_on_both":True,"grounded_weight_old_and_V2":.4,
            "sample_size_per_completed_CEM_condition":1,"height_or_zero_beta_constraint_added":False,
            "original_CEM_contact_reward":"last-substep all-contact wheel net-force norm binary",
            "V2_CEM_contact_reward":"new eight-substep floor-only any-wheel support/impact/tracking-gated V2a"},
        "CEM":{"status":"COMPLETED_RECORDINGS_CONTACT_CRITERION_FAILED","full_policy_decisions":349,"native_states":2793,
            "numeric_nominal_episode_pass":False,"failed_checks":summary["nominal_screen"]["failed_checks"],
            "whole_contact_interval_s":[.02,6.96],"whole_anywheel_support_fraction":support["full_contact"]["any_wheel_support_sample_fraction"],
            "whole_longest_no_support_s":support["full_contact"]["longest_no_support_observed_s"],
            "core_contact_interval_s":[1.44,4.],"core_anywheel_support_fraction":support["core"]["any_wheel_support_sample_fraction"],
            "core_longest_no_support_s":support["core"]["longest_no_support_observed_s"],
            "actual_all15term_V2_reward_recomputed_exactly":True,"zero_solver_failures":report["solver_failures"]==0},
        "PPO":{"status":"FORMAL_TRAINING_RUNNING_RESULTS_NOT_AVAILABLE","new_formal_policy_results_claimed":False,
            "training_backend":"PhysX","physical_GPU_index":3,"parent_reported_process_id":3554880,
            "parent_reported_started_utc":"2026-10-23T17:24:59Z","declared_transitions":12288000,
            "snapshot_information_source":"Parent-agent launch report; packaging does not SSH/poll/restart jobs",
            "checkpoint_actor_validation_proof":"PENDING_NOT_PRESENT_IN_THIS_PARTIAL_BUNDLE",
            "nativeCPU_fullhorizon_smoke_integration":{"status":"PARENT_REPORTED_PASSED_NOT_FORMAL_PERFORMANCE",
                "native_states":2793,"policy_decisions":349,"formal_performance_claimed":False}},
        "comparison_fairness":{"PPO_training_objective":"new V2 ground_contact_reward+0.4; training underway",
            "CEM_prediction_objective":"new V2 formula+0.4","CEM_actual_stage_objective":"new V2 formula+0.4; exact recompute verified",
            "PPO_evaluation":"Normalized actor only; no numerical PPO-eval-return versus CEM-return comparison claimed",
            "legacy_PPO_validation_reward_logging":"Not used for cross-controller numerical return comparison",
            "comparison_metrics":"Recorded state/contact tracking metrics, not summed reward across recipes",
            "optimizer_only_or_statistical_superiority_claimed":False},
        "goal_complete":False,"strict_continuous_anywheel_contact_achieved":False,
        "warning":"CEM wholecontact70.47% is NOT its corecontact58.93%; less HF does not certify stable contact. PPO formal results are pending."}
    write_json(output/"status.json",status)
    reproduction={"kind":"PARTIAL_CEM_completed_PPO_running_reproduction_notes","status_file":"status.json",
        "executed_source_archive":archive,"graphics":"MuJoCo OSMesa CPU, opaque real TRON1 meshes; no physics stepping",
        "metrics_replay_without_robot_assets":{"compact":"native_comparison_signals.npz","exact_two_case_metric_parity_verified":True,
            "python":"import sys;sys.path.insert(0,'scripts');from package_ground_contact_v2_partial import recompute_public_metrics;result=recompute_public_metrics('results/ground_contact_v2_nominal_20261023_v1/native_comparison_signals.npz')"},
        "full_CEM_reproduction_requires":["Pinned MuJoCo3.8.1 and original TRON1 model/reference assets (not bundled).",
            "Historical completed PPO lineage/evaluation bundles required by nominal.inputs(); weights are not included or inferred.",
            "Explicitly build isolated ABI3 using nmpc_native_ground_contact_400hz.build_native_library(fresh_directory).",
            "Invoke scripts/run_cem_ground_contact_v2.py --native-library <ABI3.so> --output-dir <fresh_directory>; no automatic retries."],
        "full_reward_recompute_scope":"All349x15 actual CEM stages verified against original raw+solver log; public compact contains state/contact metric inputs, not all-body reward history.",
        "fullbody_replay_scope":"replay_states.npz preserves exact original2793poses/case and50Hz selections; original meshes required for3D replay.",
        "generic_renderer":"scripts/render_ground_contact_v2.py accepts explicit SHA-bound nmpc/policy_isaac/policy_mujoco case manifests; this poster caption refers to the present CEM pair.",
        "omitted":"No TRON1 models/meshes, policy weights, credentials, native shared libraries or speculative PPO results included.",
        "status_refresh":"When later PPO results arrive, update status and regenerate artifact-manifest hashes; this is a dated partial snapshot."}
    write_json(output/"reproduction.json",reproduction)
    manifest={"status":"PARTIAL","snapshot_utc":observed,"source_sha256":ledger,"source_archive":archive,
        "runtime":{"mujoco_version":report["mujoco_version"],"native_abi":3,"physics_dt_s":.0025,"policy_dt_s":.02,
            "native_library_sha256":report["native_library_sha256"],"native_build_manifest_sha256":report["native_build_manifest_sha256"],
            "reward_definition_sha256":report["source_sha256"]["scripts/ground_contact_reward_v2.py"]},
        "original_CEM_input_sha256":summary["baseline_input_sha256"],"new_CEM_input_sha256":hashes,"copied_artifacts":copies,
        "artifacts":{path.name:{"sha256":sha256(path),"bytes":path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file()},
        "packager_sha256":sha256(__file__),"models_weights_credentials_included":False}
    write_json(output/"artifact_manifest.json",manifest);return manifest


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root",type=Path,default=DEFAULT_ROOT)
    result.add_argument("--output-dir",type=Path,required=True)
    return result


if __name__=="__main__":run(parser().parse_args())
