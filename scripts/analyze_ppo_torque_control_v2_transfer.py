"""Offline nominal PhysX-to-MuJoCo transfer of original-grounded torque-v2 PPO.

Uses the torque-v2 training verifier, not the older ground-support experiment's
training contract. Only its pure recorded MuJoCo parser is reused. No jobs,
actor downloads, physics, smoothing or automatic success claim are performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_ppo_torque_control_v2 as physical
import analyze_ppo_ground_support_transfer as mujoco
from analyze_ppo_smoothness import sha256


def require(condition, message):
    if not condition: raise ValueError(message)


def load_transfer(candidate_dir, candidate_training_dir, actor_sha_audit, reference=physical.DEFAULT_REFERENCE,
                  baseline_dir=physical.DEFAULT_BASELINE, *, candidate_status=None):
    """Method-bound parent audit plus original single-episode MuJoCo records."""
    _, parent = physical.load_comparison(candidate_dir, candidate_training_dir, reference, baseline_dir,
        candidate_status=candidate_status)
    require(parent["kind"] == "recorded_nominal_physx_ppo_torque_control_v2_comparison" and
        parent["original_grounded_reward_preserved"] is True and parent["candidate_additional_reward"] ==
        {"term": "leg_applied_torque_rate", "weight": -.05}, "Require the exact original-grounded torque-control v2 cohort")
    audit = physical.shared.read_json(actor_sha_audit)
    require(audit.get("schema_version") == 1 and audit.get("kind") == "remote_generated_actor_file_sha256_audit" and
        isinstance(audit.get("cases"), dict) and all(label in audit["cases"] for label in physical.CASE_ORDER),
        "Require metadata-only remote actor SHA audit for both independent checkpoint exports")
    cases, reports = {}, {}
    for label, isaac_dir in (("baseline", baseline_dir), ("candidate", candidate_dir)):
        isaac = physical.shared.read_json(Path(isaac_dir)/"report.json")
        current, reports[label] = mujoco.load_mujoco_case(Path(isaac_dir).parent/"mujoco", isaac, audit["cases"][label], reference)
        current["grounded_reward_kind"] = "Original binary grounded_reward+0.4 (unchanged)"
        cases[label] = {"checkpoint_sha256": parent["cases"][label]["checkpoint_sha256"],
            "PhysX": parent["cases"][label], "MuJoCo": current}
    for key in mujoco.SHARED_KEYS:
        require(reports["baseline"][key] == reports["candidate"][key], "Shared MuJoCo physical/evaluator contract differs: "+key)
    return {"schema_version": 1, "kind": "recorded_nominal_ppo_torque_control_v2_physx_mujoco_transfer_comparison",
        "status": "requires_visual_and_tracking_review", "training_engine": "PhysX", "evaluation_engines": ["PhysX", "MuJoCo"],
        "evaluation_mu": .7, "case_order": list(physical.CASE_ORDER), "cases": cases,
        "reference_sha256": parent["reference_sha256"], "actor_sha_audit_sha256": sha256(actor_sha_audit),
        "parent_physx_analysis_provenance": {key: value for key, value in parent.items() if key != "cases"},
        "baseline_action_rate_weight": -.5, "original_grounded_reward_preserved": True, "original_grounded_weight": .4,
        "candidate_additional_reward": {"term": "leg_applied_torque_rate", "weight": -.05},
        "independent_from_scratch_policies": True, "no_contact_is_geometric_flight": False,
        "contact_success_threshold_added": False, "stable_contact_claimed": False, "new_simulation": False,
        "new_training": False, "new_inference": False, "pose_smoothing_or_state_correction": False,
        "requires_fullbody_visual_review": True,
        "analysis_source_sha256": {Path(__file__).name: sha256(__file__),
            "analyze_ppo_torque_control_v2.py": sha256(physical.__file__),
            "analyze_ppo_ground_support_transfer.py": sha256(mujoco.__file__),
            "controller_trace.py": sha256(Path(__file__).with_name("controller_trace.py"))},
        "limitations": [
            "Two independent PhysX-trained policies share the -0.5/original-grounded+0.4 recipe; candidate adds only leg torque-rate -0.05. No MuJoCo-trained policy is compared.",
            "The torque-v2 verifier binds the full12.288M from-scratch training audit, native torque collector, explicit software PD500/10 and parent-finalized checkpoint metadata.",
            "Remote actor file SHA metadata binds each MuJoCo policy to its corresponding PhysX checkpoint. Private model/actor bytes are not downloaded or locally re-hashed unless already present.",
            "MuJoCo contact metrics use original200Hz data and the same count>0 and separate count>0 &aggregateFz>5N gates; absence is not geometric flight or task failure.",
            "Equal nominal materials do not remove engine-specific contact-patch discretization, constraint-force and sensor aggregation differences.",
            "The legacy MuJoCo report does not SHA-bind rollout bytes; this analyzer records their current local SHA. root_state_writes=1 denotes only frame0 initialization, with no prescribed root during stepping.",
            "Physical full horizon6.98s exceeds reference6.96s; full tracking remains unavailable rather than endpoint-extrapolated. Core tracking and full contact/HF/tilt are separate.",
            "One seed/checkpoint/episode per policy and backend is descriptive. Full-body video, contact, XY/speed/beta and tilt must be reviewed together; no automatic pass is invented."]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-dir", "candidate-training-dir", "actor-sha-audit", "output"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--baseline-dir", type=Path, default=physical.DEFAULT_BASELINE)
    result.add_argument("--reference", type=Path, default=physical.DEFAULT_REFERENCE)
    result.add_argument("--candidate-status", type=Path)
    return result


def run(args):
    output = Path(args.output)
    require(not output.exists(), "Preserve existing transfer evidence: output JSON must be fresh")
    summary = load_transfer(args.candidate_dir, args.candidate_training_dir, args.actor_sha_audit, args.reference,
        args.baseline_dir, candidate_status=args.candidate_status)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    return summary


if __name__ == "__main__":
    summary = run(parser().parse_args())
    print(json.dumps({"status": summary["status"], "new_simulation": False, "stable_contact_claimed": False}))
