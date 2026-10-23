"""Publish measured V1 contact-API recovery without relabeling its failure.

Only local files are read/copied. The original large warning log is omitted;
its full SHA and a path-redacted traceback excerpt remain. No formal report,
actor/model, missing live mass/gravity metadata or simulation is synthesized.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re

import recover_ground_force_api_probe_v1 as recovery
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, verify_recording, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"outputs/ground_contact_v2_20261023_v1"


def failure_excerpt(text):
    index = text.rfind("Traceback (most recent call last):")
    require(index >= 0, "Missing actual finalization traceback")
    excerpt = text[index:].strip()
    require("FileExistsError" in excerpt and "evaluation_source_audit.json" in excerpt, "Wrong failure excerpt")
    excerpt = re.sub(r'File "(/[^"\n]+)"', lambda match: 'File "<REMOTE>/'+Path(match[1]).name+'"', excerpt)
    excerpt = re.sub(r"File exists: '[^'\n]*/evaluation_source_audit\.json'",
                     "File exists: '<REMOTE_INPUT_DIR>/evaluation_source_audit.json'", excerpt)
    return {"kind": "actual_failed_finalization_traceback_excerpt", "excerpt": excerpt,
        "redactions": "Absolute remote runtime/workspace path prefixes only; exception type, code line and basename retained",
        "not_full_log": True}


def source_ledger():
    queue = [Path(__file__).resolve(), ROOT/"scripts/recover_ground_force_api_probe_v1.py",
             ROOT/"scripts/eval_ppo_ground_force_api_probe_v2r2.py"]
    seen = set()
    while queue:
        path = queue.pop()
        if path in seen: continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                dependency = ROOT/"scripts"/((name or "").split(".")[0]+".py")
                if dependency.is_file() and dependency not in seen: queue.append(dependency)
    for name in ("test_eval_ppo_ground_force_api_probe_v2.py", "test_eval_ppo_ground_force_api_probe_v2r2.py",
                 "test_recover_ground_force_api_probe_v1.py", "test_package_ground_force_api_probe_recovery_v1.py"):
        seen.add(ROOT/"tests"/name)
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(seen)}


def verify_public(folder):
    folder = Path(folder)
    manifest = read_json(folder/"artifact_manifest.json")
    require(manifest["status"] == recovery.STATUS and manifest["original_A_formal_result"] is False,
            "Published failed recovery identity differs")
    for name, item in manifest["artifacts"].items():
        path = folder/name
        require(path.is_file() and not path.is_symlink() and sha256(path) == item["sha256"]
                and path.stat().st_size == item["bytes"], "Published artifact SHA/size differs: "+name)
    audit = read_json(folder/"recovery_audit.json")
    diagnostics = recovery.validate_arrays(recovery.load_npz(folder/"probe_arrays.npz"),
                                           recovery.load_npz(folder/"physics_trajectory.npz"))
    require(diagnostics == audit["diagnostics"] and verify_recording(folder/"physics_trajectory.npz", "PhysX")
            == audit["raw_recording_verified"], "Published native/force numeric recompute differs")
    for name, file in (("raw", "physics_trajectory.npz"), ("probe_arrays", "probe_arrays.npz"),
                       ("source_audit", "evaluation_source_audit.json")):
        require(sha256(folder/file) == audit["input_files"][name]["sha256"], "Original input SHA differs")
    return {"status": recovery.STATUS, "all_artifact_hashes_verified": True,
        "all_actual_numeric_diagnostics_reproduced_exactly": True, "native_states": 2793, "probe_samples": 2792,
        "original_A_formal_result": False, "goal_completion_claimed": False}


def run(args):
    output, root = Path(args.output_dir), Path(args.input_root)
    require(not output.exists(), "Fresh public result directory required; preserve failed/original A evidence")
    original = root/"force_probe_recovery_v1/recovery_audit.json"
    audit = read_json(original)
    require(audit["status"] == recovery.STATUS and audit["finalization_completed"] is False
            and audit["original_A_formal_result"] is False and audit["goal_completion_claimed"] is False,
            "Recovery status must retain explicit failed finalization")
    for name, item in audit["input_files"].items():
        path = Path(item["path"])
        require(path.is_file() and sha256(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                "Actual recovery input hash/size differs: "+name)
    for name, digest in audit["source_sha256"].items(): require(sha256(name) == digest, "Recovery source changed")
    inputs = {name: Path(item["path"]) for name, item in audit["input_files"].items()}
    source_audit = read_json(inputs["source_audit"])
    require(source_audit["wrapper_sha256"] == recovery.V1_SHA and source_audit["recording_sha256"] == audit["input_files"]["raw"]["sha256"],
            "Source audit is not the failed V1 recording")
    actual = recovery.validate_arrays(recovery.load_npz(inputs["probe_arrays"]), recovery.load_npz(inputs["raw"]))
    require(actual == audit["diagnostics"] and verify_recording(inputs["raw"], "PhysX") == audit["raw_recording_verified"],
            "Actual recovered measurements differ from audited proof")
    excerpt = failure_excerpt(inputs["failure_log"].read_text(errors="replace"))
    excerpt["original_full_log_sha256"] = audit["input_files"]["failure_log"]["sha256"]
    excerpt["original_full_log_bytes"] = audit["input_files"]["failure_log"]["bytes"]
    output.mkdir(parents=True, exist_ok=False)
    copies = {}
    for source, name, expected in ((original, "recovery_audit.json", None),
        (inputs["raw"], "physics_trajectory.npz", audit["input_files"]["raw"]["sha256"]),
        (inputs["probe_arrays"], "probe_arrays.npz", audit["input_files"]["probe_arrays"]["sha256"]),
        (inputs["source_audit"], "evaluation_source_audit.json", audit["input_files"]["source_audit"]["sha256"]),
        (root/"force_probe_source_v1.tar.gz", "executed_probe_source_v1.tar.gz", None),
        (root/"force_probe_source_v1.manifest.json", "executed_probe_source_v1.manifest.json", None)):
        copies[name] = copy_bound(source, output/name, expected)
    executed = read_json(output/"executed_probe_source_v1.manifest.json")
    require(executed["archive_sha256"] == sha256(output/"executed_probe_source_v1.tar.gz")
            and executed["models_weights_credentials_included"] is False, "Executed probe source archive differs")
    write_json(output/"failure_excerpt.json", excerpt)
    ledger = source_ledger(); archive = source_archive(output, ledger)
    write_json(output/"status.json", {"status": recovery.STATUS, "finalization_completed": False,
        "role": "standalone_measurement_only_not_original_A_formal_comparison", "original_A_formal_result": False,
        "goal_completion_claimed": False, "same_A_actor_sha256": audit["actor_sha256_from_pre_failure_source_audit"],
        "conditions": {"mu": .7, "physics_hz": 400, "policy_hz": 50, "duration_s": 6.98, "sample_size": 1},
        "native_states": 2793, "actual_force_samples": 2792, "all_numeric_diagnostics_recomputed_exactly": True,
        "legacy32_capacity_L_R": actual["legacy32_capacity"],
        "legacy32_max_occupied_points_L_R": actual["legacy32_max_occupied_points_L_R"],
        "legacy32_at_capacity_samples_L_R": actual["legacy32_at_capacity_samples_L_R"],
        "legacy32_possible_saturation": actual["legacy32_possible_saturation"],
        "direct32_vs_filtered64": actual["comparisons"]["direct32_vs_filtered64"],
        "inference": "This episode does not support contact-buffer saturation or the32-vs64 matrix API as the cause of measured unsupported samples; not a universal engine claim.",
        "failure": "V1 failed while exclusively writing an already-created evaluation_source_audit.json AFTER complete probe/native arrays were saved.",
        "not_training_failure": True, "r2_fix_execution_claimed": False, "new_physics_or_retry_executed_for_recovery": False,
        "unavailable_not_fabricated": audit["unavailable_not_fabricated"]})
    write_json(output/"reproduction.json", {"kind": "offline_actual_recovery_numeric_replay", "status": recovery.STATUS,
        "command": "python scripts/package_ground_force_api_probe_recovery_v1.py --verify-package results/ground_force_api_probe_recovery_20261023_v1",
        "what_replays": "Every saved force comparison, >5N mask disagreement, point occupancy/capacity and all2793native state/action clock checks; no assets/actors/Isaac/GPU needed",
        "arrays": "Exact original small NPZ files retained instead of deriving a new compact or altering native samples",
        "source_archive": archive, "executed_source": "Original V1 source snapshot is preserved separately; r2 is an unexecuted bug-fix source, not a replacement result",
        "source_audit_warning": "evaluation_source_audit.json is pre-failure base-finish evidence with V1 wrapper SHA; it is NOT a completed A formal result",
        "log": "Full warning log deliberately omitted. Original SHA/size and path-redacted actual failure traceback in failure_excerpt.json.",
        "omissions": "No final report.json, probe_audit.json, missing live mass/gravity/USD data, models, policy weights, credentials or native binaries invented/included",
        "raw_recovery_cli": "recover_ground_force_api_probe_v1.py can reproduce a fresh recovery only when supplied the original hash-bound full failure log; the bundled --verify-package replay does not pretend the excerpt is that log."})
    manifest = {"status": recovery.STATUS, "original_A_formal_result": False, "models_weights_credentials_included": False,
        "original_raw_input_sha256": {name: item["sha256"] for name, item in audit["input_files"].items()},
        "original_recovery_audit_sha256": sha256(original), "copied_artifacts": copies,
        "source_sha256": ledger, "source_archive": archive, "packager_sha256": sha256(__file__),
        "artifacts": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir()) if path.is_file()}}
    write_json(output/"artifact_manifest.json", manifest)
    verify_public(output)
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--verify-package", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.verify_package is not None: print(json.dumps(verify_public(args.verify_package), indent=2))
    else: run(args)
