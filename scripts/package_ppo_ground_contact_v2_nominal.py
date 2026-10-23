"""Package verified formal PPO contact-V2 recordings; no weights or new physics.

The old dated CEM partial package is never opened for writing. A fresh output
directory is mandatory. Copied JSON proofs distinguish remotely checked weights
from the compact native signals rehashed and recomputed locally.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tarfile

import numpy as np

import analyze_ppo_ground_contact_v2 as audit
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/ground_contact_v2_20261023_v1"
CASE_ORDER = ["baseline_physx", "new_physx", "baseline_mujoco", "new_mujoco"]
PROOF_FILES = ("runner_config.json", "cohort_completion_audit.json", "ground_contact_v2_startup_audit.json",
    "policy_contract.json", "final_audit.json", "run_status.json", "frozen_source_and_inputs.json",
    "cohort_startup_audit.json", "manifest.json", "ground_contact_v2_training_audit.json",
    "continuation_status.json", "final_policy_contract.json")
BOUND_PROOFS = ("ground_contact_v2_training_audit.json", "cohort_completion_audit.json", "manifest.json",
                "policy_contract.json", "final_policy_contract.json", "run_status.json")


def recompute_compact(path, summary):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    actual = audit.recompute_compact(arrays)
    require(set(actual) == set(CASE_ORDER), "Require all four real recorded cases")
    for row in summary["rows"]:
        engine = row["id"].removeprefix("physx_trained_to_")
        for variant, field in (("new", "new_V2"), ("baseline", "original_grounded_baseline")):
            expected = copy.deepcopy(row[field]); expected.pop("support_force_source", None)
            require(actual[variant+"_"+engine] == expected, "Compact metric recomputation differs")
    return actual


def verify_render(render, summary, folder):
    require(render["case_order"] == CASE_ORDER and render["full_four_panel_comparison"] is True
            and render["explicit_subset"] is None and render["video_generated"] is True,
            "Require completed real four-panel render, not placeholders")
    require(render["native_states_per_case"] == 2793 and render["video_frames"] == 350
            and render["video_fps"] == 50 and render["video_size_pixels"] == [2560, 850], "Wrong native video clock")
    require(render["camera_same_distance_all_cases_frames"] is True and render["camera_yaw_follow"] is False
            and render["camera_zoom"] is False and render["robot_mesh_alpha"] == 1., "Wrong camera/mesh recipe")
    require(render["new_physics_policy_inference_training"] is False
            and render["smoothing_interpolation_alignment_height_correction"] is False
            and render["goal_completion_claimed"] is False, "Recorded-only/no-goal scope changed")
    for row in summary["rows"]:
        engine = row["id"].removeprefix("physx_trained_to_")
        for variant, field in (("new", "new_V2"), ("baseline", "original_grounded_baseline")):
            case = variant+"_"+engine
            require(render["cases"][case]["analysis"] == row[field], "Render did not use actual case analysis")
            fk = render["replay_FK_audit"][case]
            require(fk["samples_checked"] == 2793 and fk["root_pose_copied_exactly"] is True
                    and fk["max_wheel_origin_error_m"] <= fk["tolerance_m"], "Native pose FK audit differs")
    for name, digest in render["artifacts"].items():
        require(sha256(folder/name) == digest, "Rendered artifact changed: "+name)


def verify_source_snapshot(archive_path, manifest_path):
    manifest = read_json(manifest_path)
    source_only = (manifest.get("no_models_weights_binaries_credentials") is True
        or manifest.get("includes_models_weights_credentials") is False
        or manifest.get("assets_weights_credentials_included") is False)
    require(manifest["archive_sha256"] == sha256(archive_path) and source_only, "Source snapshot SHA/scope differs")
    files = {entry["path"]: entry for entry in manifest.get("files", [])}
    with tarfile.open(archive_path, "r:gz") as archive:
        require(set(files) == {member.name for member in archive.getmembers()}, "Source archive member ledger differs")
        for member in archive.getmembers():
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and path.suffix in (".py", ".json", ".cpp"), "Unsafe or non-source archive member")
            entry = files[member.name]
            require(member.size == entry.get("size_bytes", entry.get("bytes"))
                    and hashlib.sha256(archive.extractfile(member).read()).hexdigest() == entry["sha256"],
                    "Source member bytes/hash differs")
    return manifest


def local_source_ledger():
    """Capture local helper dependencies separately from executed remote snapshots."""
    todo = [Path(__file__).resolve(), ROOT/"scripts/render_ppo_ground_contact_v2.py",
            ROOT/"scripts/analyze_ppo_ground_contact_v2.py"]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen: continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                candidate = ROOT/"scripts"/((name or "").split(".")[0]+".py")
                if candidate.is_file() and candidate not in seen: todo.append(candidate)
    seen.add(audit.DEFAULT_PROTOCOL)
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(seen)}


def run(args):
    output, root = Path(args.output_dir), Path(args.input_root)
    require(not output.exists(), "Preserve published results; require fresh output directory")
    analysis_folder, render_folder = root/"ppo_dual_engine_analysis_v1", root/"ppo_render_v1"
    summary, render = read_json(analysis_folder/"summary.json"), read_json(render_folder/"provenance.json")
    require(summary["both_PPO_validation_backends_complete"] is True
            and summary["completed_validation_engines"] == ["physx", "mujoco"]
            and summary["overall_task_goal_complete_claimed"] is False, "Require completed dual-engine analysis")
    evidence = {}
    for engine, row in zip(("physx", "mujoco"), summary["rows"], strict=True):
        require(row["id"] == "physx_trained_to_"+engine, "Wrong engine identity/order")
        _, actual = audit.load_completed(root/"ppo_nominal_raw_v1"/engine, engine)
        require(actual == row["input_evidence"], "Copied report/proof differs from actual analysis")
        require(row["nominal_screen"]["numeric_nominal_episode_pass"] is False
                and set(row["nominal_screen"]["failed_checks"]) == {
                    "any_wheel_every_physical_sample", "longest_no_support_gap"}, "Do not relabel observed contact failure")
        evidence[engine] = actual
    require(evidence["physx"]["actor_sha256"] == evidence["mujoco"]["actor_sha256"]
            and evidence["physx"]["checkpoint_sha256"] == evidence["mujoco"]["checkpoint_sha256"], "Different new actors")
    for name, digest in summary["artifact_sha256"].items():
        require(sha256(analysis_folder/name) == digest, "Analysis artifact changed")
    recompute_compact(analysis_folder/"native_states.npz", summary)
    verify_render(render, summary, render_folder)
    training = root/"ppo_training_proof_v1"
    reports = {engine: read_json(root/"ppo_nominal_raw_v1"/engine/"report.json") for engine in evidence}
    ledger = reports["physx"]["ground_contact_v2_evaluation"]["cohort_input_sha256"]
    require(ledger == reports["mujoco"]["ground_contact_v2_evaluation"]["cohort_input_sha256"], "Training proof ledger differs")
    for name in BOUND_PROOFS:
        require(sha256(training/name) == ledger[name], "Training JSON differs from evaluation binding: "+name)
    complete = read_json(training/"cohort_completion_audit.json")
    require(complete["status"] == "completed_verified" and complete["run_kind"] == "formal"
            and complete["training_environment_steps"] == 12288000, "Not completed formal12.288M training")
    for stem in ("source_snapshot_v1", "evaluation_dependencies_v1", "continuation_snapshot_v1"):
        verify_source_snapshot(root/(stem+".tar.gz"), root/(stem+".manifest.json"))
    output.mkdir(parents=True, exist_ok=False)
    copies = {}
    def save(source, name, expected=None):
        destination = output/name; destination.parent.mkdir(parents=True, exist_ok=True)
        copies[name] = copy_bound(source, destination, expected)
    for source, name in ((analysis_folder/"summary.json", "analysis.json"),
                         (audit.DEFAULT_PROTOCOL, "protocol.json"),
                         (root/"ppo_render_launch.json", "render_launch.json")):
        save(source, name)
    for name in ("native_states.npz", "metric_table.csv"):
        save(analysis_folder/name, name, summary["artifact_sha256"][name])
    for name in ("comparison.png", "fullbody_comparison.mp4", "replay_states.npz"):
        save(render_folder/name, name, render["artifacts"][name])
    save(render_folder/"provenance.json", "render_provenance.json")
    for engine in evidence:
        for name in ("report.json", "remote_verification.json"):
            save(root/"ppo_nominal_raw_v1"/engine/name, engine+"/"+name, evidence[engine]["input_sha256"][name])
    for name in PROOF_FILES: save(training/name, "training_proof/"+name, ledger.get(name))
    for stem in ("source_snapshot_v1", "evaluation_dependencies_v1", "continuation_snapshot_v1"):
        for suffix in (".tar.gz", ".manifest.json"): save(root/(stem+suffix), "executed_snapshots/"+stem+suffix)
    sources = local_source_ledger(); source_record = source_archive(output, sources)
    observed = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in summary["rows"]:
        contact = row["new_V2"]["any_wheel_ground_support"]["full_contact"]
        rows.append({"id": row["id"], "validation_backend": row["validation_backend"], "sample_size": 1,
            "whole_contact_interval_s": [.02, 6.96], "whole_anywheel_support_fraction": contact["any_wheel_support_sample_fraction"],
            "whole_longest_no_support_s": contact["longest_no_support_observed_s"],
            "numeric_nominal_episode_pass": False, "failed_checks": row["nominal_screen"]["failed_checks"]})
    status = {"status": "FORMAL_PPO_COMPLETED_NOMINAL_DUAL_ENGINE_CONTACT_FAILURE", "snapshot_utc": observed,
        "training_backend": "PhysX", "training_transitions": 12288000, "initialization": "from_scratch",
        "conditions": {"mu": .7, "physics_hz": 400, "policy_hz": 50, "beta_reward_on_both": True,
            "grounded_weight_old_and_new": .4, "action_rate_weight": -.5, "single_wheel_support_allowed": True,
            "bilateral_support_required": False, "absolute_height_or_zero_beta_gate_added": False},
        "same_new_normalized_actor_sha256_both_engines": evidence["physx"]["actor_sha256"],
        "new_checkpoint_sha256": evidence["physx"]["checkpoint_sha256"], "cases": rows,
        "goal_complete": False, "strict_continuous_anywheel_contact_achieved": False,
        "formal_training_and_validation_complete": True, "weights_included_or_executed_locally": False,
        "evidence_scope": "Training/export weights verified remotely; JSON proofs and native compact/state metrics rehashed and recomputed locally.",
        "fairness": {"PPO_training": "V2 floor-normal eight-substep any-wheel contact reward +0.4; other14 terms unchanged",
            "evaluation": "Normalized actor once; legacy evaluator reward logging is NOT numerically compared with CEM returns",
            "comparison_metrics": "Recorded state/contact metrics, not reward totals or optimizer-only superiority"},
        "limitations": ["N=1 per engine, mu0.7 only; no low-friction or seed qualification claimed.",
            "Lower high-frequency angular RMS and shorter maximum gaps do not establish continuous support.",
            "Old CEM partial remains an immutable earlier snapshot; this separate package does not rewrite it.",
            "The two failed support checks must not be silently relaxed after training."]}
    write_json(output/"status.json", status)
    write_json(output/"reproduction.json", {"kind": "verified_PPO_native_recording_replay_no_weights",
        "metrics": {"input": "native_states.npz", "four_case_exact_recompute_verified": True,
            "command": "import numpy as np,sys;sys.path.insert(0,'scripts');import analyze_ppo_ground_contact_v2 as a;z=np.load('results/ground_contact_v2_ppo_nominal_20261023_v1/native_states.npz',allow_pickle=False);r=a.recompute_compact({k:z[k] for k in z.files})"},
        "render": {"recorded_only": True, "350_frames_at_50hz": True, "native_samples_per_case": 2793,
            "same_fixed_relative_camera_no_yaw_or_zoom": True, "opaque_actual_TRON1_meshes": True,
            "FK_audit": "All2793 actual poses per case; no mj_step, IK, reinference, alignment, interpolation or height correction",
            "assets": "Original pinned TRON1 model/meshes required for3D rerender; not bundled"},
        "local_replay_source_archive": source_record,
        "remote_executed_source": "Original SHA-bound source/config snapshots and manifests in executed_snapshots/",
        "omitted": "No actor/checkpoint weights, robot models/meshes, credentials, native shared libraries or new physics",
        "visual_review": "Recorded renderer provenance is immutable and initially visual_reviewed=false; independent human/agent review is separate."})
    manifest = {"status": status["status"], "snapshot_utc": observed, "source_sha256": sources,
        "local_replay_source_archive": source_record, "original_analysis_sha256": sha256(analysis_folder/"summary.json"),
        "new_evaluation_evidence": evidence, "training_json_sha256_bound_to_evaluation": {name: ledger[name] for name in BOUND_PROOFS},
        "copied_artifacts": copies, "models_weights_credentials_included": False,
        "packager_sha256": sha256(__file__),
        "artifacts": {str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file()}}
    write_json(output/"artifact_manifest.json", manifest)
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__": run(parser().parse_args())
