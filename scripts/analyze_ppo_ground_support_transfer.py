"""Descriptive, offline PhysX-to-MuJoCo ground-support PPO transfer audit.

Private actor bytes are not downloaded: a separately supplied remote file-hash
audit binds each MuJoCo actor to its matching, already audited PhysX checkpoint.
No physics, inference, renderer, smoothing or automatic success gate is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_support_comparison as physical
import analyze_ppo_contact_stability as contact
import analyze_ppo_smoothness as smooth
from controller_trace import native_trace


MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
SHARED_KEYS = ("engine", "engine_version", "model_sha256", "model_hash_scope", "task_source_sha256",
    "evaluator_source_sha256", "contact_slip_helper_sha256", "contact_slip_measurement", "domain_source_sha256",
    "domain_parameters", "domain_parameters_resampled_during_episode", "contact_material_audit", "terrain",
    "terrain_runtime_audit", "terrain_module_sha256", "self_collision_enabled_in_memory", "self_collision_geoms",
    "actuator_model", "actual_joint_velocity_hard_clipped", "root_com_link_offset_m", "compiled_base_com_m")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_mujoco_case(folder, isaac_report, actor_identity, reference):
    """Validate one native MuJoCo recording against the matching PhysX export."""
    folder = Path(folder)
    report = physical.read_json(folder/"report.json")
    require(report.get("status") == "completed" and report.get("engine") == "MuJoCo" and
        report.get("completed_full_reference") is True and report.get("full_horizon_diagnostic") is True and
        report.get("physics_stepped") is True and report.get("termination") == "motion_end" and
        report.get("hidden_resets") == 0 and report.get("root_state_writes") == 1 and
        report.get("physics_steps") == 1396 and report.get("policy_steps") == 349 and
        report.get("start_reference_frame") == 0 and report.get("final_reference_frame") == 348,
        "Require completed MuJoCo full horizon with only one frame0 initialization, no reset/root writes during stepping")
    require(report.get("motion_sha256") == isaac_report.get("motion_file_sha256") == smooth.sha256(reference) and
        report.get("model_sha256") == MODEL_SHA256, "MuJoCo motion/model identity differs")
    require(report.get("contract") == isaac_report.get("policy_export") and
        report["contract"].get("root_prescribed_during_steps") is False and
        report["contract"].get("checkpoint_sha256") == isaac_report.get("checkpoint_sha256"),
        "MuJoCo contract is not the matching checkpoint's exact PhysX actor export")
    require(actor_identity.get("checkpoint_sha256") == isaac_report["checkpoint_sha256"] and
        physical._digest(actor_identity.get("actor_sha256")) and
        actor_identity["actor_sha256"] == report.get("policy_sha256") and
        report.get("policy_kind") == "normalized_torchscript_actor",
        "MuJoCo actor file SHA is not bound to the matching remotely audited PhysX export")
    material, domain = report.get("contact_material_audit", {}), report.get("domain_parameters", {})
    require(material.get("expected_friction") == .7 and material.get("compiled_friction_range") == [.7, .7] and
        report.get("domain_parameters_resampled_during_episode") is False and domain.get("is_nominal") is True and
        domain.get("torque_scale") == [1.]*8 and domain.get("velocity_scale") == [1.]*8 and
        domain.get("wheel_friction_nm") == [0., 0.] and
        report["contract"].get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7 and
        report["contract"].get("domain_randomization") == "disabled during evaluation; fixed actuator parameters",
        "MuJoCo must use actual nominal mu=.7 materials and disabled actuator domain randomization")
    require(report.get("contact_slip_measurement", {}).get("active_point_normal_force_threshold_n") == 5. and
        report["contact_slip_measurement"].get("sample_dt_s") == .005,
        "MuJoCo actual contact point threshold/native clock differs")
    with np.load(folder/"rollout.npz", allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    trace = native_trace(raw, "policy_mujoco")
    time = trace["time_s"]
    require(time.shape == (1397,) and np.allclose(time, np.arange(1397)*.005, rtol=0, atol=1e-9) and
        trace["physics_dt_s"] == .005, "Require all original1397 native200Hz MuJoCo states, no resampling/truncation")
    windows = contact.analyze_signals(time, trace["contact_count"], trace["contact_force"],
        trace["com_velocity"], trace["angular_velocity"], nonwheel_counts=raw.get("nonwheel_ground_contact_count"))
    reference_data = smooth._load_reference(reference)
    quaternion = trace["quaternion"]
    tilt = np.rad2deg(np.arccos(np.clip(1.-2.*(quaternion[:, 1]**2+quaternion[:, 2]**2), -1., 1.)))
    for name in ("core", "full"):
        interval = windows[name]["interval_s"]
        selected = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
        windows[name]["tracking"] = smooth._tracking(time, trace["position"], trace["com_velocity"],
            quaternion, reference_data, interval)
        windows[name]["tilt"] = {"mean_deg": float(tilt[selected].mean()), "rms_deg": smooth.rms(tilt[selected]),
            "peak_deg": float(tilt[selected].max()), "definition": "Recorded base_Link +Z to world +Z; not a success/height limit"}
    return {"checkpoint_sha256": report["contract"]["checkpoint_sha256"], "actor_sha256": report["policy_sha256"],
        "input_sha256": {"report.json": smooth.sha256(folder/"report.json"), "rollout.npz": smooth.sha256(folder/"rollout.npz")},
        "native_physical_samples": len(time), "root_state_writes": {"frame0_initialization": 1, "during_stepping": 0},
        "windows": {name: windows[name] for name in ("core", "full")},
        "evaluation_provenance": {key: report[key] for key in SHARED_KEYS},
        "actor_identity_scope": "Metadata-only remote file SHA audit, matched against MuJoCo policy SHA and PhysX checkpoint; no local actor bytes re-hashed",
        "rollout_identity_scope": "Local raw archive SHA computed here; legacy MuJoCo report does not include a report-to-rollout hash"}, report


def load_transfer(candidate_dir, candidate_training_dir, actor_sha_audit, reference=physical.DEFAULT_REFERENCE,
                  baseline_dir=physical.DEFAULT_BASELINE, *, candidate_status=None):
    _, parent = physical.load_comparison(candidate_dir, candidate_training_dir, reference, baseline_dir,
                                         candidate_status=candidate_status)
    audit = physical.read_json(actor_sha_audit)
    require(audit.get("schema_version") == 1 and audit.get("kind") == "remote_generated_actor_file_sha256_audit" and
        isinstance(audit.get("cases"), dict) and all(name in audit["cases"] for name in physical.CASE_ORDER),
        "Require explicit metadata-only remote actor SHA audit for both checkpoint exports")
    reports, cases = {}, {}
    for label, isaac_dir in (("baseline", baseline_dir), ("candidate", candidate_dir)):
        isaac = physical.read_json(Path(isaac_dir)/"report.json")
        mujoco, reports[label] = load_mujoco_case(Path(isaac_dir).parent/"mujoco", isaac, audit["cases"][label], reference)
        cases[label] = {"checkpoint_sha256": parent["cases"][label]["checkpoint_sha256"],
            "PhysX": parent["cases"][label], "MuJoCo": mujoco}
    for key in SHARED_KEYS:
        require(reports["baseline"][key] == reports["candidate"][key], "Shared MuJoCo physical/evaluator contract differs: "+key)
    return {"schema_version": 1, "kind": "recorded_nominal_ppo_ground_support_physx_mujoco_transfer_comparison",
        "status": "requires_visual_and_tracking_review", "training_engine": "PhysX", "evaluation_engines": ["PhysX", "MuJoCo"],
        "evaluation_mu": .7, "case_order": list(physical.CASE_ORDER), "cases": cases,
        "reference_sha256": parent["reference_sha256"], "actor_sha_audit_sha256": smooth.sha256(actor_sha_audit),
        "parent_physx_analysis_provenance": {key: value for key, value in parent.items() if key != "cases"},
        "analysis_source_sha256": {Path(__file__).name: smooth.sha256(__file__), "controller_trace.py": smooth.sha256(Path(__file__).with_name("controller_trace.py"))},
        "no_contact_is_geometric_flight": False, "contact_success_threshold_added": False, "stable_contact_claimed": False,
        "new_simulation": False, "new_inference": False, "new_training": False, "pose_smoothing_or_state_correction": False,
        "requires_fullbody_visual_review": True,
        "limitations": [
            "Same PhysX-trained checkpoint is evaluated in both engines; this is not a MuJoCo-trained PPO comparison.",
            "Both use recorded ground-filtered count>0 and the separate count>0 &aggregateFz>5N load gate; absence is not geometric flight.",
            "Contact patch discretization, instantaneous MuJoCo constraint forces and PhysX sensor aggregation remain engine-specific; equal nominal mu is not equal contact dynamics.",
            "MuJoCo root_state_writes=1 denotes frame0 initialization, then zero prescribed-root writes during stepping in the audited legacy evaluator.",
            "No actor/checkpoint bytes are downloaded; actor hash audit is metadata-bound. Local MuJoCo rollout hash is reported, but legacy report does not bind it by SHA.",
            "Full physical horizon6.98s exceeds reference6.96s; full tracking remains unavailable, never endpoint-extrapolated.",
            "One seed/checkpoint/episode is descriptive. Contact/load/HF differences do not define an automatic pass or stable-contact guarantee."]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-dir", type=Path, required=True)
    result.add_argument("--candidate-training-dir", type=Path, required=True)
    result.add_argument("--actor-sha-audit", type=Path, required=True)
    result.add_argument("--baseline-dir", type=Path, default=physical.DEFAULT_BASELINE)
    result.add_argument("--reference", type=Path, default=physical.DEFAULT_REFERENCE)
    result.add_argument("--candidate-status", type=Path)
    result.add_argument("--output", type=Path, required=True, help="Fresh JSON file only")
    return result


def run(args):
    output = Path(args.output)
    require(not output.exists(), "Preserve existing transfer evidence: output JSON must be fresh")
    summary = load_transfer(args.candidate_dir, args.candidate_training_dir, args.actor_sha_audit,
        args.reference, args.baseline_dir, candidate_status=args.candidate_status)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    summary = run(parser().parse_args())
    print(json.dumps({"status": summary["status"], "new_simulation": False, "stable_contact_claimed": False}))
