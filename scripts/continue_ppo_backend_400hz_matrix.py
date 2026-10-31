"""Complete an interrupted immutable PPO matrix in a new output directory.

Verified old episodes are copied byte-for-byte and explicitly marked reused.
Only pending cells run. Training provenance stays on its original GPU; the
new evaluation GPU is audited separately. No simulator/source/weight changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_verified_tree(source, target):
    """Reject symlinks; verify every copied file, not merely the report."""
    source, target = Path(source), Path(target)
    if target.exists() or source.is_symlink() or not source.is_dir():
        raise ValueError("Expected real source directory and fresh destination")
    before = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in historical evidence: " + str(path))
        if path.is_file():
            before[str(path.relative_to(source))] = digest(path)
    shutil.copytree(source, target)
    after = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*")) if p.is_file()}
    if before != after or any(digest(source / p) != h for p, h in before.items()):
        raise ValueError("Evidence changed during copy")
    return before


def wait_idle(probe, index, deadline, write_status):
    """Waiting does not allocate a GPU or retry a failed experiment."""
    while True:
        gpu = probe(index)
        valid = (gpu.get("idle") is True and gpu.get("physical_index") == index
                 and gpu.get("pci_bus_id_cuda_ordinal_verified") is True
                 and not gpu.get("all_reported_compute_and_graphics_processes"))
        if valid:
            return gpu
        write_status(gpu)
        if time.monotonic() >= deadline:
            raise TimeoutError("Requested GPU remained occupied; no fallback or process signals")
        time.sleep(15)


def run(options):
    old = options.previous.resolve()
    previous_path = old / "backend_400hz_matrix_status.json"
    previous_hash = digest(previous_path)
    previous = json.loads(previous_path.read_text())
    if previous["status"] != "stopped_without_retry" or previous["counts"]["runtime_failed"]:
        raise ValueError("Only an interrupted, runtime-failure-free matrix is eligible")
    driver_path = Path(previous["dispatcher_argv"][0]).resolve()
    sys.path.insert(0, str(driver_path.parent))
    spec = importlib.util.spec_from_file_location("frozen_matrix_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    args = driver.parser().parse_args(previous["dispatcher_argv"][1:])
    args.output_dir = options.output_dir.resolve()
    driver.resolve(args)  # Includes immutable source tree, protocol, fresh path.
    shared = driver.load_helper(args.source_snapshot / "scripts/ppo_backend_400hz_contract.py", "continuation_contract")
    guard = driver.load_helper(args.guard_source, "continuation_guard", driver.GUARD_SHA256)
    probe = driver.load_helper(args.gpu_probe_source, "continuation_gpu", driver.GPU_PROBE_SHA256).query_gpu
    pair = driver.read_json(args.training_pair_dir / "backend_400hz_pair_status.json")
    # Validate training using its actual original GPU before selecting eval GPU.
    cohorts = driver.completed_cohorts(args, pair, shared)
    frozen = driver.read_json(old / "frozen_source_and_inputs.json")
    driver.verify_snapshot(args, frozen, cohorts)
    protocol = shared.verify_protocol(args.protocol)
    expected = driver.planned_cases(args, protocol)
    by_id = {c["case_id"]: c for c in previous["cases"]}
    if len(by_id) != len(expected) or set(by_id) != {c["case_id"] for c in expected}:
        raise ValueError("Original matrix has missing or duplicate cells")
    for c in expected:
        for key in ("phase", "training_backend", "engine_tag", "validation_backend", "mu", "cohort_dir", "smoke_only"):
            if c[key] != by_id[c["case_id"]][key]:
                raise ValueError("Historical case identity differs: " + key)
    args.physical_gpu_index = options.physical_gpu_index
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "backend_400hz_matrix_status.json"
    status = dict(schema_version=1, kind="continued_matched_400hz_ppo_backend_validation_matrix",
                  status="verifying_reused_evidence", pid=os.getpid(), dispatcher_argv=sys.argv,
                  protocol_path=str(args.protocol), protocol_sha256=digest(args.protocol),
                  training_pair_dir=str(args.training_pair_dir), preflight_dir=str(args.preflight_dir),
                  physical_gpu_index=args.physical_gpu_index, training_physical_gpu_index=pair["physical_gpu_index"],
                  previous_status_path=str(previous_path), previous_status_sha256=previous_hash,
                  continuation_source_sha256=digest(__file__), started_unix_time=time.time(),
                  automatic_retry=False, automatic_gpu_fallback=False, process_signals_sent=False,
                  tracking_success_claimed=False, cases=expected)
    def save():
        status["counts"] = driver.counts(status["cases"])
        guard.write_json(status_path, status)
    guard.write_json(args.output_dir / "frozen_source_and_inputs.json", frozen)
    save()
    current = None
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    try:
        for case in expected:
            original = by_id[case["case_id"]]
            if original["status"] != "completed_verified":
                if original["status"] != "pending":
                    raise ValueError("Refuse ambiguous incomplete cell: " + case["case_id"])
                continue
            folder = Path(original["directory"]).resolve()
            if folder != (old / case["relative_output_dir"]).resolve() or not folder.is_relative_to(old):
                raise ValueError("Historical evidence escapes original root")
            proof = shared.validate_output(folder, case["cohort_dir"], args.protocol,
                       validation_backend=case["validation_backend"], allow_smoke=case["smoke_only"])
            if proof != original["verification"]:
                raise ValueError("Historical verification proof changed")
            copied = copy_verified_tree(folder, case["directory"])
            copied_proof = shared.validate_output(case["directory"], case["cohort_dir"], args.protocol,
                       validation_backend=case["validation_backend"], allow_smoke=case["smoke_only"])
            if copied_proof != proof:
                raise ValueError("Copied verification proof differs")
            case.update(copy.deepcopy(original))
            case.update(directory=str(args.output_dir / case["relative_output_dir"]), reused=True,
                        reused_from=str(folder), reused_file_sha256=copied)
            save()
        status["reused_case_count"] = sum(c.get("reused", False) for c in expected)
        pending = [c for c in expected if c["status"] == "pending"]
        status["new_case_count_planned"] = len(pending)
        identity = None
        for current in pending:
            driver.verify_snapshot(args, frozen, cohorts)
            cohort = cohorts[(current["phase"], current["training_backend"])]
            current.update(checkpoint_sha256=cohort["input_sha256"]["model_final.pt"], actor_sha256=cohort["input_sha256"]["policy.pt"])
            child_env = env.copy()
            if current["engine_tag"] == "physx":
                def waiting(gpu):
                    status.update(status="waiting_for_idle_gpu", latest_gpu_observation=gpu)
                    save()
                gpu = wait_idle(probe, args.physical_gpu_index, time.monotonic() + options.idle_wait_seconds, waiting)
                observed = {k: gpu[k] for k in ("uuid", "pci_bus_id")}
                if identity is not None and identity != observed:
                    raise ValueError("Evaluation GPU identity changed")
                identity = observed
                status["selected_gpu_identity"] = identity
                current["gpu_preflight"] = gpu
            else:
                child_env["CUDA_VISIBLE_DEVICES"] = ""
                current["gpu_allocation"] = False
            command = driver.case_command(args, current)
            current.update(status="running", command=command, reused=False, started_unix_time=time.time())
            status["status"] = "running_" + current["case_id"]
            log = args.output_dir / "logs" / (current["case_id"].replace("/", "__") + ".log")
            log.parent.mkdir(exist_ok=True)
            current["log"] = str(log)
            save()
            def spawned(pid):
                current["pid"] = pid
                save()
            code = guard.run_child(command, log, args.source_snapshot, child_env, spawned)
            current.update(returncode=code, finished_unix_time=time.time())
            if code:
                raise RuntimeError("Evaluation failed; no automatic retry")
            driver.verify_snapshot(args, frozen, cohorts)
            proof = shared.validate_output(current["directory"], current["cohort_dir"], args.protocol,
                        validation_backend=current["validation_backend"], allow_smoke=current["smoke_only"])
            current.update(status="completed_verified", verification=proof)
            save()
            time.sleep(2)
        current = None
        if driver.counts(expected) != dict(formal_expected=68, formal_completed=68, pipeline_smoke_expected=4, pipeline_smoke_completed=4, runtime_failed=0):
            raise ValueError("Matrix is incomplete")
        manifest = {k: status[k] for k in ("kind", "protocol_path", "protocol_sha256", "training_pair_dir", "preflight_dir", "physical_gpu_index", "selected_gpu_identity", "cases", "counts")}
        manifest.update(schema_version=1, status="validation_matrix_completed", input_motion_sha256=digest(args.motion),
                        input_model_sha256=digest(args.model), frozen_source_and_inputs_sha256=digest(args.output_dir / "frozen_source_and_inputs.json"),
                        smoke_pipeline_only=True, tracking_success_claimed=False)
        guard.write_json(args.output_dir / "validation_matrix_manifest.json", manifest)
        status["status"] = "all_validation_completed_analysis_pending"
        save()
        env["CUDA_VISIBLE_DEVICES"] = ""
        code = guard.run_child(driver.analysis_command(args), args.output_dir / "analysis.log", args.source_snapshot, env, lambda pid: None)
        if code or not (args.output_dir / "analysis/summary.json").is_file():
            raise RuntimeError("Analysis failed; validated episodes preserved")
        driver.verify_snapshot(args, frozen, cohorts)
        if digest(previous_path) != previous_hash:
            raise ValueError("Original matrix status changed")
        status.update(status="completed_with_analysis", analysis_summary_sha256=digest(args.output_dir / "analysis/summary.json"))
    except BaseException as error:
        if current is not None and current["status"] == "running":
            current["status"] = "failed_or_unverified"
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_unix_time"] = time.time()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--idle-wait-seconds", type=float, default=86400)
    run(parser.parse_args())
