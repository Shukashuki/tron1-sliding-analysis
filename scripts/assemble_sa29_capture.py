"""Assemble private, portable SA29 captures from complete original evidence.

No simulation, actor loading, inference, FK, network or native compilation.
Fresh legitimate policy weights are accepted only with the fixed training and
source contracts checked by the frozen analyzer. NMPC assembly intentionally
requires the ORIGINAL published MuJoCo17 bundle: it does not generalize the
frozen comparison analyzer to arbitrary new MuJoCo recordings.

Plans use an explicit original_root and relative {path,sha256} file bindings.
Policy plan kind: sa29_policy_capture_assembly_plan, inputs(motion,
robot_reference,model), evaluation_source(root,audit,source_sha256,
reporting_only_dependencies), policies(PPO4/SAC6), cases34 using the frozen
files7 and execution.{isaac,mujoco}.{receipt,log} schema.
NMPC plan kind: sa29_nmpc_execution_assembly_plan, inputs3, source_root,
mujoco_bundle, cases17 of {mu,files:{report,trace,solver},receipt,log}.
Both plans additionally require schema_version=1 and reference_id=doa_sa29_scale2.

Capture trees contain raw states/weights and original private receipt paths.
Keep them PRIVATE; only the separate frozen analysis projection is publishable.
Only allowlisted original files are copied, never caches or arbitrary folders.
Publication uses Linux renameat2(RENAME_NOREPLACE), preserving existing targets.
The returned next-analysis argv is never executed.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import analyze_sa29_controller_comparison as comparison
import analyze_sa29_policy_friction_dynamics as policy
from analyze_controller_pair import (audit_reports, audit_training, load_arrays,
    read_json, reference_trace, sha256)
from controller_references import DEFAULT_CATALOG, ROOT, verify_reference_pair
from nmpc_validation_contract import validate_case_against_nominal

TRAIN_NAMES = {"manifest": "manifest.json", "status": "run_status.json", "checkpoint": "model_final.pt",
    "runner_config": "runner_config.json", "policy_contract": "policy_contract.json", "actor": "actor_normalized.pt"}
POLICY_NAMES = {"isaac_trace": "physics_trajectory.npz", "isaac_report": "report.json",
    "isaac_policy_trace": "trajectory.npz", "isaac_actor": "actor_normalized.pt", "isaac_contract": "policy_contract.json",
    "mujoco_trace": "rollout.npz", "mujoco_report": "report.json"}
NMPC_NAMES = {"report": "report.json", "trace": "rollout.npz", "solver": "solver_log.json"}
INPUT_NAMES = {"motion": "motion.npz", "robot_reference": "robot_reference.npz", "model": "model.xml"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(root, relative, *, directory=False):
    relative = Path(policy._relative(relative)); cursor = root
    for part in relative.parts:
        cursor /= part
        require(not cursor.is_symlink(), "Original paths must not traverse symlinks")
    target = cursor.resolve()
    require(target.is_relative_to(root) and (target.is_dir() if directory else target.is_file()),
        "Missing original or original-root escape")
    return target


def original_file(binding, root):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "Require explicit original path/SHA binding")
    _inside(root, binding["path"])
    return policy.checked_file(binding, root)


def _plan(path, kind, keys):
    path = Path(path).resolve(); document = read_json(path)
    require(set(document) == {"schema_version", "kind", "reference_id", "original_root", *keys}
        and type(document["schema_version"]) is int and document["schema_version"] == 1
        and document["kind"] == kind and document["reference_id"] == policy.REFERENCE_ID,
        "Unexpected assembly plan scope/schema")
    require(isinstance(document["original_root"], str) and document["original_root"], "Explicit original_root required")
    declared = Path(document["original_root"])
    declared = path.parent / declared if not declared.is_absolute() else declared
    require(not declared.is_symlink(), "Original root must not be a symlink")
    root = declared.resolve(); require(root.is_dir(), "Missing original root")
    return document, root


def _unique(paths):
    inodes = set()
    for path in paths:
        state = path.stat(); inode = (state.st_dev, state.st_ino)
        require(inode not in inodes, "Original file/path/inode reused across distinct roles")
        inodes.add(inode)


def _copy(source, target, expected):
    require(source.is_file() and sha256(source) == expected and not target.exists(), "Changed original or existing copy target")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, target.open("xb") as copied:
        shutil.copyfileobj(original, copied)
    require(sha256(source) == sha256(target) == expected, "Original changed while being copied")


def _bind(path, root):
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def _fresh(output):
    output = Path(output)
    require(not output.is_symlink() and not output.exists(), "Preserve existing assembly output")
    output = output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    return output, Path(tempfile.mkdtemp(prefix="."+output.name+".incomplete-", dir=output.parent))


def _publish(stage, output):
    require(not output.is_symlink() and not output.exists(), "Output appeared during assembly; preserve both trees")
    require(sys.platform.startswith("linux"), "Atomic exclusive directory publication requires Linux renameat2; staged tree preserved")
    library = ctypes.CDLL(None, use_errno=True)
    require(hasattr(library, "renameat2"), "renameat2 unavailable; staged tree preserved, no overwrite fallback")
    rename = library.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def _inputs(document, root):
    require(isinstance(document["inputs"], dict) and set(document["inputs"]) == set(INPUT_NAMES), "Require three explicit inputs")
    inputs = {name: original_file(value, root) for name, value in document["inputs"].items()}
    spec = verify_reference_pair(inputs["motion"], inputs["robot_reference"], policy.REFERENCE_ID)
    return inputs, spec


def _terminal_closed(path):
    receipt = read_json(path)
    require(receipt.get("capture_complete") is receipt.get("owned_processes_and_contexts_absent")
        is receipt.get("source_bindings_unchanged") is True, "Real terminal capture is incomplete/unclosed/changed")


def _audit_policy(document, root):
    parsed = policy.parse_manifest(document, root)
    inputs, spec = _inputs(document, root)
    source = document["evaluation_source"]
    policy.audit_source(parsed["source_root"], source["audit"], source["source_sha256"], root,
        reporting_only_dependencies=source["reporting_only_dependencies"])
    for algorithm, paths in parsed["policies"].items():
        training, status, runner = [read_json(paths[name]) for name in ("manifest", "status", "runner_config")]
        contract = training.get("policy_contract") if algorithm == "PPO" else read_json(paths["policy_contract"])
        policy.audit_training_budget(algorithm, training, status, runner, contract, sha256(paths["checkpoint"]),
            sha256(paths["actor"]) if "actor" in paths else None)
    for item in parsed["cases"]:
        algorithm, mu, files = item["algorithm"], item["mu"], item["files"]
        ia, mj = [read_json(files[engine+"_report"]) for engine in ("isaac", "mujoco")]
        hashes = {role: sha256(path) for role, path in files.items()}
        for engine in ("isaac", "mujoco"):
            policy.audit_execution(item["execution"][engine], algorithm, mu, engine, files, root)
            _terminal_closed(original_file(item["execution"][engine]["receipt"], root))
        audit = audit_reports(algorithm, spec, ia, mj, isaac_trace_sha256=hashes["isaac_trace"],
            mujoco_trace_sha256=hashes["mujoco_trace"], model_sha256=sha256(inputs["model"]), mujoco_capture_sha256=hashes["mujoco_trace"])
        training_files = parsed["policies"][algorithm]; training = read_json(training_files["manifest"])
        require(audit["fixed_mu"] == mu and ia.get("runner_config_sha256") == sha256(training_files["runner_config"]),
            "Actual coefficient/training runner differs")
        audit_training(training_files["manifest"], algorithm, spec, ia, mj, checkpoint_path=training_files["checkpoint"])
        policy.audit_case_sources(ia, mj, source["source_sha256"], files, training)
        raw_ia = load_arrays(files["isaac_trace"])
        policy.audit_policy_trace(raw_ia, "isaac", ia)
        policy.audit_policy_trace(load_arrays(files["mujoco_trace"]), "mujoco", mj)
        policy.audit_policy_trajectory(load_arrays(files["isaac_policy_trace"]), raw_ia)


def _policy_manifest(plan):
    return {"schema_version": 1, "kind": "sa29_policy_full_physical_grid_capture", "reference_id": policy.REFERENCE_ID,
        "reference_spec": policy.get_reference(policy.REFERENCE_ID), "reference_catalog_sha256": sha256(DEFAULT_CATALOG),
        "evaluation_mu": list(policy.GRID), **{name: copy.deepcopy(plan[name]) for name in ("inputs", "evaluation_source", "policies", "cases")}}


def _policy_bindings(document, root):
    paths = []
    for value in document["inputs"].values(): paths.append(original_file(value, root))
    for files in document["policies"].values():
        paths.extend(original_file(value, root) for value in files.values())
    for case in document["cases"]:
        paths.extend(original_file(value, root) for value in case["files"].values())
        paths.extend(original_file(value, root) for e in case["execution"].values() for value in e.values())
    source = document["evaluation_source"]; source_root = _inside(root, source["root"], directory=True)
    paths.append(original_file(source["audit"], root))
    for name, digest in {**source["source_sha256"], **source["reporting_only_dependencies"]}.items():
        target = _inside(source_root, name)
        require(sha256(target) == digest, "Changed original captured source/dependency")
        paths.append(target)
    _unique(paths)
    return paths


def assemble_policy(plan_path, output_dir, analysis_output):
    plan, root = _plan(plan_path, "sa29_policy_capture_assembly_plan", ("inputs", "evaluation_source", "policies", "cases"))
    document = _policy_manifest(plan); _policy_bindings(document, root); _audit_policy(document, root)
    require(not Path(analysis_output).exists(), "Preserve existing future analysis output")
    output, stage = _fresh(output_dir); copied = copy.deepcopy(document)
    def copy_binding(value, destination):
        _copy(original_file(value, root), stage/destination, value["sha256"])
        return _bind(stage/destination, stage)
    copied["inputs"] = {name: copy_binding(value, "inputs/"+INPUT_NAMES[name]) for name, value in document["inputs"].items()}
    copied["policies"] = {algorithm: {role: copy_binding(value, "training/"+algorithm+"/"+TRAIN_NAMES[role])
        for role, value in files.items()} for algorithm, files in document["policies"].items()}
    source = document["evaluation_source"]; original_source = _inside(root, source["root"], directory=True)
    for name, digest in {**source["source_sha256"], **source["reporting_only_dependencies"]}.items():
        _copy(_inside(original_source, name), stage/"evaluation_source"/name, digest)
    copied["evaluation_source"] = {**copy.deepcopy(source), "root": "evaluation_source",
        "audit": copy_binding(source["audit"], "evaluation_source/wide_friction_source.json")}
    copied["cases"] = []
    for item in document["cases"]:
        prefix = "cases/"+item["algorithm"]+"/"+policy.case_name(item["mu"])
        files = {role: copy_binding(value, prefix+"/"+role.split("_", 1)[0]+"/"+POLICY_NAMES[role]) for role, value in item["files"].items()}
        execution = {engine: {role: copy_binding(value, prefix+"/"+engine+"/"+("terminal_receipt.json" if role == "receipt" else "evaluation.log"))
            for role, value in capture.items()} for engine, capture in item["execution"].items()}
        copied["cases"].append({"algorithm": item["algorithm"], "mu": item["mu"], "files": files, "execution": execution})
    _policy_bindings(copied, stage); _audit_policy(copied, stage); _policy_bindings(document, root)
    _json(stage/"manifest.json", copied); _publish(stage, output)
    return {"status": "complete_private_policy_capture_assembled_not_analyzed", "pairs": 34, "engine_episodes": 68,
        "manifest": str(output/"manifest.json"), "manifest_sha256": sha256(output/"manifest.json"),
        "analysis_command": [sys.executable, str(ROOT/"scripts/analyze_sa29_policy_friction_dynamics.py"),
            "--manifest", str(output/"manifest.json"), "--output-dir", str(Path(analysis_output).resolve())],
        "analysis_executed": False, "simulation_performed": False}


def assemble_nmpc(plan_path, output_dir, analysis_output):
    plan, root = _plan(plan_path, "sa29_nmpc_execution_assembly_plan", ("inputs", "source_root", "mujoco_bundle", "cases"))
    inputs, spec = _inputs(plan, root)
    original_source = _inside(root, plan["source_root"], directory=True)
    bundle = _inside(root, plan["mujoco_bundle"], directory=True)
    source_paths = [_inside(original_source, name) for name in policy.BASE_SOURCE_SHA256]
    require(all(sha256(original_source/name) == digest for name, digest in policy.BASE_SOURCE_SHA256.items()), "Original source9 differs")
    reference = reference_trace(load_arrays(inputs["motion"]))
    hashes = {name: sha256(path) for name, path in inputs.items()}
    mj, _, _, _, _, _ = comparison.load_mujoco_bundle(bundle, spec, hashes, reference)
    require(isinstance(plan["cases"], list) and len(plan["cases"]) == 17, "Require complete17 NMPC cases")
    bundle_paths = [_inside(bundle, name) for name in comparison.MUJOCO_BUNDLE_SHA256]
    cases, original_paths, seen, first, identity = [], list(inputs.values())+source_paths+bundle_paths, set(), None, None
    for item in plan["cases"]:
        require(isinstance(item, dict) and set(item) == {"mu", "files", "receipt", "log"}
            and type(item["mu"]) in (int, float) and item["mu"] in comparison.GRID and item["mu"] not in seen
            and isinstance(item["files"], dict) and set(item["files"]) == set(NMPC_NAMES), "Missing/duplicate/undeclared NMPC case")
        mu = item["mu"]; seen.add(mu)
        files = {name: original_file(value, root) for name, value in item["files"].items()}
        require(len({p.parent for p in files.values()}) == 1 and all(p.name == NMPC_NAMES[role] for role, p in files.items()), "Preserve adjacent original NMPC names")
        capture = {role: original_file(item[role], root) for role in ("receipt", "log")}
        original_paths.extend((*files.values(), *capture.values()))
        evidence, _, audit = comparison.audit_isaac_episode(files["report"].parent, mu, spec, hashes["model"], mj["runs"][comparison.case_name(mu)], capture=capture)
        _terminal_closed(capture["receipt"])
        current = {name: audit[name] for name in ("controller_contract", "native_identity", "asset_sha256", "fk_source_helpers_sha256")}
        require(identity is None or identity == current, "NMPC source/controller identity changed across coefficients"); identity = current
        if first is None: first = evidence
        else: validate_case_against_nominal(evidence, first)
        cases.append((item, files, capture))
    require(seen == set(comparison.GRID), "Missing NMPC coefficients"); _unique(original_paths)
    require(not Path(analysis_output).exists(), "Preserve existing future analysis output")
    output, stage = _fresh(output_dir)
    for name, path in inputs.items(): _copy(path, stage/"inputs"/INPUT_NAMES[name], hashes[name])
    for name, digest in policy.BASE_SOURCE_SHA256.items(): _copy(original_source/name, stage/"original_source"/name, digest)
    for name, digest in comparison.MUJOCO_BUNDLE_SHA256.items(): _copy(bundle/name, stage/"mujoco_bundle"/name, digest)
    manifest = {"schema_version": 1, "kind": "sa29_nmpc_isaac_terminal_capture", "reference_id": policy.REFERENCE_ID, "cases": []}
    for item, files, capture in sorted(cases, key=lambda row: row[0]["mu"]):
        prefix = Path("cases")/comparison.case_name(item["mu"])
        for role, path in files.items(): _copy(path, stage/prefix/NMPC_NAMES[role], item["files"][role]["sha256"])
        record = {"mu": item["mu"]}
        for role, path in capture.items():
            destination = stage/prefix/("terminal_receipt.json" if role == "receipt" else "evaluation.log")
            _copy(path, destination, item[role]["sha256"]); record[role] = _bind(destination, stage)
        manifest["cases"].append(record)
    _json(stage/"execution_manifest.json", manifest); captured = comparison.load_execution_manifest(stage/"execution_manifest.json")
    for item, _, _ in cases:
        mu = item["mu"]
        comparison.audit_isaac_episode(stage/"cases"/comparison.case_name(mu), mu, spec, hashes["model"],
            mj["runs"][comparison.case_name(mu)], capture=captured["cases"][mu])
        for value in (*item["files"].values(), item["receipt"], item["log"]): original_file(value, root)
    require(all(sha256(path) == hashes[name] for name, path in inputs.items())
        and all(sha256(_inside(original_source, name)) == digest for name, digest in policy.BASE_SOURCE_SHA256.items())
        and all(sha256(_inside(bundle, name)) == digest for name, digest in comparison.MUJOCO_BUNDLE_SHA256.items()),
        "Original inputs/source/locked bundle changed during assembly")
    _publish(stage, output)
    command = [sys.executable, str(ROOT/"scripts/analyze_sa29_controller_comparison.py"), "--motion", str(output/"inputs/motion.npz"),
        "--reference-file", str(output/"inputs/robot_reference.npz"), "--model", str(output/"inputs/model.xml"),
        "--source-root", str(output/"original_source"), "--mujoco-bundle", str(output/"mujoco_bundle"),
        "--execution-manifest", str(output/"execution_manifest.json"), "--output-dir", str(Path(analysis_output).resolve())]
    for mu in comparison.GRID: command.extend(["--run", str(mu)+"="+str(output/"cases"/comparison.case_name(mu))])
    return {"status": "complete_private_nmpc_capture_assembled_not_analyzed", "engine_episodes": 17,
        "manifest": str(output/"execution_manifest.json"), "manifest_sha256": sha256(output/"execution_manifest.json"),
        "locked_published_mujoco_bundle_required": True, "arbitrary_new_mujoco_bundle_supported": False,
        "analysis_command": command, "analysis_executed": False, "simulation_performed": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="kind", required=True)
    for name in ("policy", "nmpc-execution"):
        command = commands.add_parser(name)
        for flag in ("plan", "output-dir", "analysis-output-dir"): command.add_argument("--"+flag, type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    operation = assemble_policy if args.kind == "policy" else assemble_nmpc
    print(json.dumps(operation(args.plan, args.output_dir, args.analysis_output_dir), indent=2, allow_nan=False))


if __name__ == "__main__": main()
