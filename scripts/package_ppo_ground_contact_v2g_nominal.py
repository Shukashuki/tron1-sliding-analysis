"""Package formal G reference-beta-gated bonus and ungated-gap PPO proofs.

No weights, models, credentials, native libraries, physics, FK or inference are
loaded. A, B, C, D and CEM bundles are never written. Numerical pass/fail is derived
from actual locked screens; lower HF alone never becomes task completion.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

import analyze_ppo_ground_contact_v2g as audit
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"outputs/ground_contact_v2g_20261023_v1"
ANALYZER_SHA256 = "600aac38a2d21b323862d38ff51e1e9c4805f157c913bd954fbde70ec458e999"
RENDERER_SHA256 = "451dd4d6cbf199b8be64e0ce765a0b2f639e6886b2e7e30b21dad1ae79b51fe4"
COMPACT_CASE_ORDER = ["new_physx", "baseline_physx", "new_mujoco", "baseline_mujoco"]
CASE_ORDER = ["baseline_physx", "v2g_physx", "baseline_mujoco", "v2g_mujoco"]
PROOF_FILES = ("runner_config.json", "cohort_completion_audit.json", "ground_contact_v2g_startup_audit.json",
    "policy_contract.json", "final_policy_contract.json", "run_status.json", "manifest.json",
    "cohort_startup_audit.json", "ground_contact_v2g_training_audit.json", "continuation_status.json",
    "final_audit.json", "frozen_source_and_inputs.json", "physx_formal_v1_launch.json", "continuation_v1_launch.json")
BOUND_PROOFS = ("ground_contact_v2g_training_audit.json", "cohort_completion_audit.json", "manifest.json",
    "policy_contract.json", "final_policy_contract.json", "run_status.json")
SNAPSHOT_PINS = {
    "source_snapshot_v1": "0ce6f07650528b4a759832451302e403040c885820fe9ba9cf7024212815fc92",
    "continuation_snapshot_v1": "8878e81fee8934b0fed1bd31a4bee6226c9188872ff0e5d9a64b183b4401e1e8"}


SNAPSHOT_MEMBER_COUNTS = {"source_snapshot_v1": 22, "continuation_snapshot_v1": 2}


def verify_sources():
    require(sha256(audit.__file__) == ANALYZER_SHA256, "Frozen G analyzer changed")
    require(sha256(ROOT/"scripts/render_ppo_ground_contact_v2g.py") == RENDERER_SHA256,
            "Frozen G renderer changed")
    protocol = audit.load_protocol()
    settings = protocol["ground_contact_reward_v2"]
    require(settings["support_gap_weight"] == 1. and settings["support_force_threshold_n"] == 5.
            and settings["mean_weight"] == .75 and settings["minimum_weight"] == .25
            and settings["external_reward_weight"] == .4 and settings["beta_gate_low_reference_speed_m_s"] == .3,
            "G gates only F positive bonus by existing reference-beta quality; stop-neutral1; unchanged ungated gap cost")
    return protocol


def validate_summary(summary, protocol=None):
    protocol = protocol if protocol is not None else verify_sources()
    require(summary.get("kind") == "PPO_ground_contact_V2g_nominal_offline_audit"
            and summary.get("status") == "completed_analysis"
            and summary.get("cohort_id") == audit.evaluator.COHORT_ID
            and summary.get("reward_definition_version") == audit.evaluator.REWARD_VERSION
            and summary.get("protocol_sha256") == audit.PROTOCOL_SHA256
            and summary.get("reference_sha256") == audit.INPUT_PINS["motion"]
            and summary.get("analyzer_sha256") == ANALYZER_SHA256
            and summary.get("imported_A_B_C_D_F_and_G_source_sha256") == audit.verify_import_sources(), "Not actual frozen G analysis")
    require(summary.get("both_PPO_validation_backends_complete") is True
            and summary.get("completed_validation_engines") == ["physx", "mujoco"]
            and summary.get("overall_task_goal_complete_claimed") is False
            and summary.get("new_dynamics_inference_training_or_FK") is False
            and summary.get("height_or_flight_failure_gate") is False
            and summary.get("bilateral_support_required") is False
            and summary.get("sample_size_per_case") == 1
            and (summary.get("physics_dt_s"), summary.get("policy_dt_s"), summary.get("native_states_per_case")) == (.0025, .02, 2793),
            "Require completed actual G dual-engine analysis")
    rows = summary.get("rows")
    require(isinstance(rows, list) and len(rows) == 2, "Require exactly two real G engine rows")
    for engine, row in zip(audit.BACKENDS, rows, strict=True):
        require(row["id"] == "physx_trained_to_"+engine and row["validation_backend"] == audit.BACKENDS[engine]
                and row["sample_size"] == 1, "Wrong G engine identity/order")
        expected = audit.nominal_screen(row["new_V2"], row["original_grounded_baseline"], protocol, engine)
        require(row["nominal_screen"] == expected, "Published G nominal screen differs from frozen criteria")
        require(row["baseline_evidence"]["published_summary_sha256"] == audit.BASELINE_SUMMARY_SHA256,
                "Wrong original same-engine baseline publication")
    return protocol


def recompute_compact(path, summary):
    validate_summary(summary)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    require(np.asarray(arrays["case_ids"]).tolist() == COMPACT_CASE_ORDER, "Wrong four-case compact identity/order")
    actual = audit.recompute_compact(arrays)
    require(set(actual) == set(COMPACT_CASE_ORDER), "Require all four real recorded cases")
    for engine, row in zip(audit.BACKENDS, summary["rows"], strict=True):
        for variant, field in (("new", "new_V2"), ("baseline", "original_grounded_baseline")):
            expected = copy.deepcopy(row[field]); expected.pop("support_force_source", None)
            require(actual[variant+"_"+engine] == expected, "Compact metric recomputation differs")
    return actual


def verify_render(render, summary, folder):
    require(render.get("kind") == "PPO_contact_V2g_beta_gated_bonus_ungated_anywheel_gap_actual_case_caption_recorded_replay"
            and render.get("cohort_id") == audit.evaluator.COHORT_ID
            and render.get("reward_definition_version") == audit.evaluator.REWARD_VERSION
            and render.get("renderer_sha256") == RENDERER_SHA256
            and render.get("model_sha256") == audit.INPUT_PINS["model"]
            and render.get("protocol_sha256") == audit.PROTOCOL_SHA256
            and render.get("support_gap_weight") == 1. and render.get("support_force_threshold_n") == 5.
            and "support_deficit_weight" not in render and render.get("external_grounded_weight") == .4
            and render.get("reference_sha256") == summary["reference_sha256"], "Not actual frozen G renderer provenance")
    require(render.get("case_order") == CASE_ORDER and render.get("full_four_panel_comparison") is True
            and render.get("explicit_subset") is None and render.get("video_generated") is True,
            "Require completed real four-panel G render, not a subset or placeholders")
    require((render.get("native_states_per_case"), render.get("video_frames"), render.get("video_fps")) == (2793, 350, 50)
            and render.get("video_size_pixels") == [2560, 850], "Wrong G native video clock")
    require(render.get("camera_same_distance_all_cases_frames") is True and render.get("camera_yaw_follow") is False
            and render.get("camera_zoom") is False and render.get("robot_mesh_alpha") == 1., "Wrong fixed camera or opaque meshes")
    require(render.get("new_physics_policy_inference_training") is False
            and render.get("smoothing_interpolation_alignment_height_correction") is False
            and render.get("goal_completion_claimed") is False
            and render.get("training_audit_required_and_bound") is True
            and render.get("A_B_C_D_E_F_formal_results_changed_or_substituted") is False, "Recorded-only G identity/scope changed")
    for engine, row in zip(audit.BACKENDS, summary["rows"], strict=True):
        for variant, field, proof_key in (("baseline", "original_grounded_baseline", "baseline_evidence"),
                                         ("v2g", "new_V2", "input_evidence")):
            key = variant+"_"+engine; case = render["cases"][key]
            require(case["analysis"] == row[field] and render["input_evidence"][key] == row[proof_key]
                    and case.get("A_B_C_D_E_F_reward_policy_substituted") is False, "G render did not use actual analysis/proof")
            if variant == "v2g":
                require(case["new_nominal_screen"] == row["nominal_screen"]
                        and case["G_cohort_id"] == audit.evaluator.COHORT_ID, "G outcome caption or cohort differs")
            fk = render["replay_FK_audit"][key]
            require(fk["samples_checked"] == 2793 and fk["root_pose_copied_exactly"] is True
                    and fk["max_wheel_origin_error_m"] <= fk["tolerance_m"], "G native-pose FK audit differs")
    require(set(render["artifacts"]) == {"comparison.png", "fullbody_comparison.mp4", "replay_states.npz"},
            "Unexpected render artifact allowlist")
    for name, digest in render["artifacts"].items():
        require(sha256(Path(folder)/name) == digest, "G rendered artifact changed: "+name)


def verify_source_snapshot(archive_path, manifest_path, expected_archive_sha=None, *, expected_member_count=None):
    archive_path, manifest_path = Path(archive_path), Path(manifest_path)
    require(archive_path.is_file() and not archive_path.is_symlink()
            and manifest_path.is_file() and not manifest_path.is_symlink(), "Missing direct G source snapshot")
    manifest = read_json(manifest_path)
    require(manifest.get("archive_sha256") == sha256(archive_path)
            and (expected_archive_sha is None or manifest["archive_sha256"] == expected_archive_sha), "G source snapshot SHA differs")
    require(manifest.get("no_models_weights_binaries_credentials") is True
            or manifest.get("no_models_weights_credentials") is True, "Missing explicit source-only G snapshot scope")
    entries = manifest.get("files", [])
    require(isinstance(entries, list) and entries, "Missing G snapshot member ledger")
    require(expected_member_count is None or len(entries) == expected_member_count, "G snapshot member count differs")
    files = {entry["path"]: entry for entry in entries}
    require(len(files) == len(entries), "Duplicate snapshot member path")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len(files) and set(files) == {member.name for member in members}, "G snapshot members differ")
        for member in members:
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and path.suffix in (".py", ".json", ".cpp"), "Unsafe or non-source G snapshot member")
            entry = files[member.name]
            require(member.size == entry.get("size_bytes", entry.get("bytes"))
                    and hashlib.sha256(archive.extractfile(member).read()).hexdigest() == entry["sha256"],
                    "G source snapshot member bytes/hash differs")
    return manifest


def local_source_ledger():
    todo = [Path(__file__).resolve(), Path(audit.__file__).resolve(), ROOT/"scripts/render_ppo_ground_contact_v2g.py"]
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


def status_from_summary(summary, evidence, snapshot_utc):
    settings = audit.definition.validate_settings()
    formula = audit.definition.formula_metadata()["formula"]
    rows = []; strict_support = True
    for row in summary["rows"]:
        support = row["new_V2"]["any_wheel_ground_support"]["full_contact"]
        strict_support &= support["every_physical_sample_supported"] and support["longest_no_support_observed_s"] == 0.
        screen = row["nominal_screen"]
        rows.append({"id": row["id"], "validation_backend": row["validation_backend"], "sample_size": 1,
            "whole_contact_interval_s": [.02, 6.96], "whole_anywheel_support_fraction": support["any_wheel_support_sample_fraction"],
            "whole_longest_no_support_s": support["longest_no_support_observed_s"],
            "numeric_nominal_episode_pass": screen["numeric_nominal_episode_pass"], "failed_checks": screen["failed_checks"]})
    passed = all(row["numeric_nominal_episode_pass"] for row in rows)
    status = "FORMAL_G_PPO_COMPLETED_DUAL_ENGINE_NUMERICAL_SCREENS_PASSED_REVIEW_PENDING" if passed else "FORMAL_G_PPO_COMPLETED_DUAL_ENGINE_NUMERICAL_CRITERIA_FAILED"
    return {"status": status, "snapshot_utc": snapshot_utc, "cohort_id": audit.evaluator.COHORT_ID,
        "reward_definition_version": audit.evaluator.REWARD_VERSION, "support_aggregation": formula,
        "training_backend": "PhysX", "training_transitions": 12288000, "initialization": "from_scratch",
        "conditions": {"mu": .7, "physics_hz": 400, "policy_hz": 50, "beta_reward_on": True,
            "grounded_weight": settings["external_reward_weight"],
            "positive_mean_weight": settings["mean_weight"], "positive_minimum_weight": settings["minimum_weight"],
            "support_gap_weight": settings["support_gap_weight"], "support_force_threshold_n": settings["support_force_threshold_n"],
            "support_gap_penalty_active": True, "support_gap_tracking_or_impact_gated": False,
            "contact_reward_settings": settings,
            "positive_beta_gate_source": "existing unweighted reference-beta quality, not zero-beta target",
            "positive_beta_gate_low_reference_speed_m_s": settings["beta_gate_low_reference_speed_m_s"],
            "action_rate_weight": -.5, "single_wheel_support_allowed": True,
            "bilateral_support_required": False, "absolute_height_or_zero_beta_gate_added": False},
        "same_new_normalized_actor_sha256_both_engines": evidence["physx"]["actor_sha256"],
        "new_checkpoint_sha256": evidence["physx"]["checkpoint_sha256"], "cases": rows,
        "both_numerical_nominal_screens_passed": passed, "strict_continuous_anywheel_contact_achieved": bool(strict_support),
        "formal_training_and_validation_complete": True, "goal_complete": False,
        "weights_included_or_executed_locally": False,
        "evidence_scope": "Execution-host verified training/export weights; copied JSON proofs and compact native metrics rehashed locally.",
        "fairness": {"baseline": "Original same-engine PhysX-trained400Hz betaON PPO, not A contact policy",
            "training": "Frozen G formula: "+formula+"; external+0.4; other14 terms unchanged",
            "evaluation": "Normalized actor once; legacy evaluator reward logging is NOT numerically compared with CEM returns",
            "metrics": "Recorded state/contact tracking, not different-reward return totals or optimizer-only superiority"},
        "limitations": ["N=1 per engine, nominal mu0.7 only; no friction-range, seed, hardware or statistical qualification.",
            "Lower HF or shorter maximum gaps alone do not establish continuous support or task completion.",
            "Numerical pass/fail is actual-data-derived; unchanged contact/speed/beta/XY/HF criteria are not relaxed.",
            "Qualitative whole-body review and broader task completion remain separate; A/CEM packages are not overwritten."]}


def verify_training_proofs(training, reports, evidence):
    training = Path(training)
    require(all((training/name).is_file() and not (training/name).is_symlink() for name in PROOF_FILES),
            "Missing direct allowlisted G training/continuation/launch JSON proof")
    ledger = reports["physx"]["ground_contact_v2g_evaluation"]["cohort_input_sha256"]
    require(ledger == reports["mujoco"]["ground_contact_v2g_evaluation"]["cohort_input_sha256"], "G training ledger differs across engines")
    for name in BOUND_PROOFS:
        require(sha256(training/name) == ledger[name], "G training JSON differs from evaluation binding: "+name)
    complete = read_json(training/"cohort_completion_audit.json")
    require(complete["status"] == "completed_verified" and complete["run_kind"] == "formal"
            and complete["training_environment_steps"] == 12288000, "Not completed formal G12.288M training")
    for name in ("continuation_status.json", "final_audit.json"):
        state = read_json(training/name)
        require(state["status"] == "nominal_dual_engine_validation_completed_requires_task_review"
                and state["kind"] == "owned_ground_contact_v2g_formal_nominal_validation_continuation"
                and state["training_launch_record_sha256"] == sha256(training/"physx_formal_v1_launch.json")
                and state["automatic_retry"] is False and state["process_signals_sent"] is False, "G continuation not completed/proof-bound")
        require([child["backend"] for child in state["children"]] == ["mujoco", "physx"], "G continuation engine order differs")
        for child in state["children"]:
            require(child["status"] == "completed_verified" and child["returncode"] == 0
                    and child["verification"] == evidence[child["backend"]]["evaluation_proof"], "G child verification proof differs")
    return ledger


def run(args):
    output, root = Path(args.output_dir), Path(args.input_root)
    require(not output.exists(), "Preserve published results; require fresh output directory")
    protocol = verify_sources()
    analysis_folder, render_folder, training = root/"ppo_dual_engine_analysis_v1", root/"ppo_render_v1", root/"ppo_training_proof_v1"
    summary, render = read_json(analysis_folder/"summary.json"), read_json(render_folder/"provenance.json")
    validate_summary(summary, protocol)
    training_path = training/"ground_contact_v2g_training_audit.json"
    _, trained_evidence = audit.load_training_audit(training_path)
    evidence = {}
    for engine, row in zip(audit.BACKENDS, summary["rows"], strict=True):
        _, actual = audit.load_completed(root/"ppo_nominal_raw_v1"/engine, engine, training_audit=training_path)
        require(actual == row["input_evidence"], "Actual G report/proof differs from analysis")
        _, baseline = audit.load_baseline(engine)
        require(baseline == row["baseline_evidence"], "Original same-engine baseline proof differs")
        evidence[engine] = actual
    require(evidence["physx"]["actor_sha256"] == evidence["mujoco"]["actor_sha256"]
            and evidence["physx"]["checkpoint_sha256"] == evidence["mujoco"]["checkpoint_sha256"], "Different formal G actors")
    require(render["explicit_training_audit"] == trained_evidence, "G render explicit training audit differs")
    require(set(summary["artifact_sha256"]) == {"native_states.npz", "metric_table.csv"}, "Wrong G analysis artifact allowlist")
    for name, digest in summary["artifact_sha256"].items():
        require(name in ("native_states.npz", "metric_table.csv") and sha256(analysis_folder/name) == digest, "G analysis artifact changed")
    recompute_compact(analysis_folder/"native_states.npz", summary)
    verify_render(render, summary, render_folder)
    reports = {engine: read_json(root/"ppo_nominal_raw_v1"/engine/"report.json") for engine in evidence}
    ledger = verify_training_proofs(training, reports, evidence)
    for stem, digest in SNAPSHOT_PINS.items():
        verify_source_snapshot(root/(stem+".tar.gz"), root/(stem+".manifest.json"), digest,
            expected_member_count=SNAPSHOT_MEMBER_COUNTS[stem])
    require((root/"ppo_render_launch.json").is_file() and not (root/"ppo_render_launch.json").is_symlink(),
            "Missing direct actual G render launch JSON")
    output.mkdir(parents=True, exist_ok=False)
    copies = {}
    def save(source, name, expected=None):
        destination = output/name; destination.parent.mkdir(parents=True, exist_ok=True)
        copies[name] = copy_bound(source, destination, expected)
    for source, name in ((analysis_folder/"summary.json", "analysis.json"), (audit.DEFAULT_PROTOCOL, "protocol.json"),
                         (root/"ppo_render_launch.json", "render_launch.json"), (render_folder/"provenance.json", "render_provenance.json")):
        save(source, name)
    for name in ("native_states.npz", "metric_table.csv"): save(analysis_folder/name, name, summary["artifact_sha256"][name])
    for name in ("comparison.png", "fullbody_comparison.mp4", "replay_states.npz"): save(render_folder/name, name, render["artifacts"][name])
    for engine in evidence:
        for name in ("report.json", "remote_verification.json"):
            save(root/"ppo_nominal_raw_v1"/engine/name, engine+"/"+name, evidence[engine]["input_sha256"][name])
    for name in PROOF_FILES: save(training/name, "training_proof/"+name, ledger.get(name))
    for stem in SNAPSHOT_PINS:
        for suffix in (".tar.gz", ".manifest.json"): save(root/(stem+suffix), "executed_snapshots/"+stem+suffix)
    sources = local_source_ledger(); source_record = source_archive(output, sources)
    observed = datetime.now(timezone.utc).isoformat()
    status = status_from_summary(summary, evidence, observed)
    write_json(output/"status.json", status)
    write_json(output/"reproduction.json", {"kind": "verified_FORMAL_G_beta_gated_bonus_ungated_anywheel_gap_PPO_native_replay_no_weights",
        "metrics": {"input": "native_states.npz", "four_case_exact_recompute_verified": True,
            "verification_command": "python scripts/package_ppo_ground_contact_v2g_nominal.py --verify-public results/ground_contact_v2g_ppo_nominal_20261023_v1",
            "robot_reference_assets_or_weights_required": False},
        "render": {"recorded_only": True, "350_frames_at_50hz": True, "native_samples_per_case": 2793,
            "same_fixed_relative_camera_no_yaw_or_zoom": True, "opaque_actual_TRON1_meshes": True,
            "assets_for_3D_rerender": "Original pinned model/meshes required, not included; compact metrics need no assets"},
        "local_replay_source_archive": source_record, "remote_executed_snapshots": list(SNAPSHOT_PINS),
        "no_separate_evaluation_dependencies_snapshot_fabricated": True,
        "omitted": "No actor/checkpoint weights, models/meshes, credentials, native shared libraries, new physics or invented G results",
        "visual_review": "Renderer provenance is preserved; independent qualitative review is separate from numerical screens."})
    manifest = {"kind": "verified_FORMAL_G_beta_gated_bonus_ungated_anywheel_gap_PPO_completed_nominal_bundle", "status": status["status"], "snapshot_utc": observed,
        "source_sha256": sources, "local_replay_source_archive": source_record,
        "original_analysis_sha256": sha256(analysis_folder/"summary.json"), "new_evaluation_evidence": evidence,
        "training_json_sha256_bound_to_evaluation": {name: ledger[name] for name in BOUND_PROOFS},
        "copied_artifacts": copies, "models_weights_credentials_included": False, "packager_sha256": sha256(__file__),
        "artifacts": {str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file()}}
    write_json(output/"artifact_manifest.json", manifest)
    return manifest


def verify_published(folder):
    """Asset-free, read-only artifact/hash + exact compact/screen replay."""
    folder = Path(folder); manifest = read_json(folder/"artifact_manifest.json")
    require(manifest.get("kind") == "verified_FORMAL_G_beta_gated_bonus_ungated_anywheel_gap_PPO_completed_nominal_bundle"
            and manifest.get("models_weights_credentials_included") is False, "Not a formal G public proof bundle")
    verify_sources()
    for name, entry in manifest["artifacts"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and (folder/path).is_file()
                and not (folder/path).is_symlink(), "Unsafe/missing public G artifact")
        require(sha256(folder/path) == entry["sha256"] and (folder/path).stat().st_size == entry["bytes"], "Public G artifact changed: "+name)
    summary = read_json(folder/"analysis.json")
    require(sha256(folder/"analysis.json") == manifest["original_analysis_sha256"], "Copied G analysis SHA differs")
    evidence = {engine: row["input_evidence"] for engine, row in zip(audit.BACKENDS, summary["rows"], strict=True)}
    require(evidence == manifest["new_evaluation_evidence"]
            and evidence["physx"]["actor_sha256"] == evidence["mujoco"]["actor_sha256"]
            and evidence["physx"]["checkpoint_sha256"] == evidence["mujoco"]["checkpoint_sha256"], "Public same-actor G evidence differs")
    training = folder/"training_proof"
    trained, training_evidence = audit.load_training_audit(training/"ground_contact_v2g_training_audit.json", folder/"protocol.json")
    reports = {}
    for engine in audit.BACKENDS:
        report_path, proof_path = folder/engine/"report.json", folder/engine/"remote_verification.json"
        require(sha256(report_path) == evidence[engine]["input_sha256"]["report.json"]
                and sha256(proof_path) == evidence[engine]["input_sha256"]["remote_verification.json"], "Public report/proof G binding differs")
        reports[engine] = read_json(report_path)
        raw_name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        audit._check_new_report(reports[engine], read_json(proof_path), engine, folder/"protocol.json",
            evidence[engine]["input_sha256"][raw_name], sha256(report_path), trained, training_evidence)
    verify_training_proofs(training, reports, evidence)
    recompute_compact(folder/"native_states.npz", summary)
    verify_render(read_json(folder/"render_provenance.json"), summary, folder)
    status = read_json(folder/"status.json")
    expected = status_from_summary(summary, evidence, status["snapshot_utc"])
    require(status == expected and status["goal_complete"] is False, "Public G status inflated or changed")
    for stem, digest in SNAPSHOT_PINS.items():
        verify_source_snapshot(folder/"executed_snapshots"/(stem+".tar.gz"),
            folder/"executed_snapshots"/(stem+".manifest.json"), digest,
            expected_member_count=SNAPSHOT_MEMBER_COUNTS[stem])
    return {"status": "VERIFIED_ALL_FOUR_CASE_COMPACT_METRICS_AND_ACTUAL_G_OUTCOMES_EXACT",
        "numeric_screens_passed_both_engines": status["both_numerical_nominal_screens_passed"],
        "strict_anywheel_support_achieved": status["strict_continuous_anywheel_contact_achieved"],
        "goal_complete": False, "weights_models_dynamics_FK_inference_or_training_required": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-public", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(verify_published(args.verify_public) if args.verify_public else run(args), indent=2))

