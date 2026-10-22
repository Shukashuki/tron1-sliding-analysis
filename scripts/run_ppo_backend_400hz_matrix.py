"""Wait for the authorized training pair, then run sequential 400 Hz validation.

Waiting allocates no GPU and never controls the parent training process. Failed
jobs are not retried; a busy requested GPU aborts without fallback. Four smoke
episodes validate the pipeline only, followed by the 68 formal matrix cells.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GUARD_SHA256 = "d07e7b04af9554744ac70fa7dc149760171e55978da2791d9721ce4916f472e0"
GPU_PROBE_SHA256 = "cbc7bdbd9fd75a6ce98c04e0e0e9dc253956be3fa93acf5e246b73e5e11dd071"
TRAINING_DRIVER_SHA256 = "b26c779686eed3b12e713b5a40d06ad41dcda2a962a588c51c3b77366913545d"
PROTOCOL_SHA256 = "c54cd530884cffa22811ca2232cb2588eba3582deefc91d4d9675e4fca7fe30f"
ENGINE_NAMES = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}
ORIGINS = ("physx", "mjlab")
SUFFIXES = {".py", ".json", ".yaml", ".yml", ".cpp", ".c", ".h"}
IGNORED = {".git", "__pycache__", ".pytest_cache"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def load_helper(path, name, expected=None):
    path = Path(path).resolve()
    if expected is not None:
        require(sha256(path) == expected, "Pinned helper source differs: " + str(path))
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def planned_cases(args, protocol):
    cases = []
    def add(phase, mu, origin, engine):
        relative = f"{phase}/{origin}/{engine}/mu{mu:.3f}".replace(".", "p")
        cohort = (args.preflight_dir/(origin+"_smoke") if phase == "pipeline_smoke"
                  else args.training_pair_dir/(origin+"_formal"))
        cases.append({"case_id": relative, "phase": phase, "training_backend": origin,
            "engine_tag": engine, "validation_backend": ENGINE_NAMES[engine], "mu": float(mu),
            "relative_output_dir": relative, "directory": str(args.output_dir/relative),
            "cohort_dir": str(cohort), "smoke_only": phase == "pipeline_smoke", "status": "pending"})
    for origin in ORIGINS:
        for engine in ENGINE_NAMES:
            add("pipeline_smoke", .7, origin, engine)
    # Nominal four cells precede the remaining 64 fixed-friction cells.
    grid = [.7] + [mu for mu in protocol["validation"]["friction_grid"] if mu != .7]
    for mu in grid:
        for origin in ORIGINS:
            for engine in ENGINE_NAMES:
                add("formal", mu, origin, engine)
    require(len(cases) == 72 and len({case["case_id"] for case in cases}) == 72,
            "Expected four smoke plus 68 unique formal cells")
    return cases


def counts(cases):
    return {"pipeline_smoke_expected": 4, "formal_expected": 68,
        "pipeline_smoke_completed": sum(c["phase"] == "pipeline_smoke" and c["status"] == "completed_verified" for c in cases),
        "formal_completed": sum(c["phase"] == "formal" and c["status"] == "completed_verified" for c in cases),
        "runtime_failed": sum(c["status"] == "failed_or_unverified" for c in cases)}


def check_training_process(args, status):
    """Read only this explicitly identified owner's process, never signal it."""
    if args.training_dispatcher_pid is None:
        return {"requested": False, "process_signalled": False}
    pid = args.training_dispatcher_pid
    require(status.get("pid") == pid and pid > 0, "Training dispatcher PID differs from status")
    proc = Path("/proc")/str(pid)
    require(proc.exists() and proc.stat().st_uid == os.getuid(), "Running training dispatcher is absent or has a different owner")
    command = [word.decode() for word in (proc/"cmdline").read_bytes().split(b"\0") if word]
    require(any(word.endswith("/scripts/run_ppo_backend_400hz_pair.py") for word in command),
            "Owned PID is not the declared training dispatcher")
    require("--output-dir" in command and Path(command[command.index("--output-dir")+1]).resolve() == args.training_pair_dir,
            "Owned process belongs to a different training output")
    return {"requested": True, "pid": pid, "same_uid": True, "process_signalled": False,
            "command_matches_training_pair": True}


def source_snapshot(args, cohorts=None):
    paths = {args.protocol, args.model, args.motion, args.asset_path, args.guard_source, args.gpu_probe_source}
    roots = {args.source_snapshot, args.legacy_source, args.beyondmimic_source,
             args.gpu_probe_source.parent.parent}
    for root in roots:
        for path in root.rglob("*"):
            if IGNORED.intersection(path.relative_to(root).parts) or path.suffix not in SUFFIXES:
                continue
            require(not path.is_symlink(), "Refuse symlinked source: " + str(path))
            if path.is_file():
                paths.add(path)
    if cohorts:
        paths.update((args.training_pair_dir/"backend_400hz_pair_status.json",
                      args.preflight_dir/"backend_400hz_pair_status.json"))
        for cohort in cohorts.values():
            paths.update(cohort["paths"].values())
            paths.add(cohort["directory"]/"runner_config.json")
    records = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(paths)}
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"tree_sha256": hashlib.sha256(encoded).hexdigest(), "files": records,
            "formal_actor_bytes_frozen": bool(cohorts)}


def verify_snapshot(args, frozen, cohorts=None):
    require(source_snapshot(args, cohorts) == frozen, "Frozen source, input or completed actor bytes changed")


def wait_for_pair(args, status, guard, frozen, *, sleep=time.sleep, process_check=check_training_process):
    while True:
        verify_snapshot(args, frozen)
        pair = read_json(args.training_pair_dir/"backend_400hz_pair_status.json")
        state = pair.get("status")
        if state == "training_pair_completed_validation_pending":
            return pair
        require(state in ("initializing", "running_physx_formal", "running_mjlab_formal"),
                "Training pair failed/stopped or has an unexpected state; no validation/retry: " + str(state))
        status.update(status="waiting_for_training_pair", training_pair_last_status=state,
            training_process_readonly_audit=process_check(args, pair),
            wait_observations=status.get("wait_observations", 0)+1,
            no_gpu_allocation_while_waiting=True, counts=counts(status["cases"]))
        guard.write_json(args.output_dir/"backend_400hz_matrix_status.json", status)
        sleep(30.)


def completed_cohorts(args, pair, shared):
    require(pair.get("stage") == "formal" and pair.get("physical_gpu_index") == args.physical_gpu_index and
            [c.get("stage") for c in pair.get("children", [])] == ["physx_formal", "mjlab_formal"] and
            all(c.get("status") == "completed_verified" and c.get("returncode") == 0 for c in pair["children"]),
            "Actual formal pair is incomplete or used a different physical GPU")
    require(args.training_dispatcher_pid is None or pair.get("pid") == args.training_dispatcher_pid,
            "Completed dispatcher PID differs")
    argv = pair.get("dispatcher_argv", [])
    require(argv and sha256(argv[0]) == TRAINING_DRIVER_SHA256 and "--preflight-dir" in argv and
            Path(argv[argv.index("--preflight-dir")+1]).resolve() == args.preflight_dir,
            "Completed pair has different dispatcher/preflight provenance")
    preflight = read_json(args.preflight_dir/"backend_400hz_pair_status.json")
    require(preflight.get("status") == "preflight_completed_requires_formal_authorization" and
            preflight.get("physical_gpu_index") == args.physical_gpu_index and
            [c.get("stage") for c in preflight.get("children", [])] == ["parity", "physx_smoke", "mjlab_smoke"] and
            all(c.get("status") == "completed_verified" and c.get("returncode") == 0 for c in preflight["children"]),
            "The actual parity/two-smoke preflight is incomplete")
    cohorts = {}
    initial = pair.get("initial_policy_state_sha256")
    require(isinstance(initial, str) and len(initial) == 64, "Missing common initial network identity")
    for phase in ("pipeline_smoke", "formal"):
        for origin in ORIGINS:
            directory = args.preflight_dir/(origin+"_smoke") if phase == "pipeline_smoke" else args.training_pair_dir/(origin+"_formal")
            cohort = shared.verify_cohort(directory, args.protocol, allow_smoke=phase == "pipeline_smoke")
            require(cohort["training_backend"] == origin and cohort["smoke_only"] is (phase == "pipeline_smoke") and
                    cohort["manifest"]["initial_policy_state_sha256"] == initial,
                    "Cohort backend/mode or initialization differs")
            child = (preflight["children"][1+ORIGINS.index(origin)] if phase == "pipeline_smoke"
                     else pair["children"][ORIGINS.index(origin)])
            proof = child.get("verification", {})
            expected = {"status": "verified", "backend": origin,
                "run_kind": "smoke" if phase == "pipeline_smoke" else "formal",
                "training_environment_steps": 1536 if phase == "pipeline_smoke" else 12288000,
                "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
                "exported_policy_sha256": cohort["input_sha256"]["policy.pt"],
                "manifest_sha256": cohort["input_sha256"]["manifest.json"],
                "cohort_completion_audit_sha256": cohort["input_sha256"]["cohort_completion_audit.json"],
                "initial_policy_state_sha256": initial}
            require(all(proof.get(key) == value for key, value in expected.items()),
                    "Parent training/preflight child proof differs from completed actual artifacts")
            cohorts[(phase, origin)] = cohort
    return cohorts


def case_command(args, case):
    script = "eval_ppo_backend_400hz_isaac.py" if case["engine_tag"] == "physx" else "eval_ppo_backend_400hz_mujoco.py"
    legacy = "eval_tracking_friction_isaac.py" if case["engine_tag"] == "physx" else "eval_tracking_mujoco.py"
    command = [str(args.python), "-u", str(args.source_snapshot/"scripts"/script),
        "--cohort-dir", case["cohort_dir"], "--cohort-protocol", str(args.protocol),
        "--legacy-evaluator", str(args.legacy_source/"scripts"/legacy), "--motion-file", str(args.motion),
        "--output-dir", case["directory"], "--evaluation-contact-friction", str(case["mu"])]
    if case["engine_tag"] == "physx":
        command.extend(("--asset-path", str(args.asset_path), "--physical-gpu-index", str(args.physical_gpu_index), "--headless"))
    else:
        command.extend(("--model", str(args.model)))
    if case["smoke_only"]:
        command.append("--allow-smoke-policy")
    return command


def analysis_command(args):
    return [str(args.python), "-u", str(args.source_snapshot/"scripts/analyze_ppo_backend_400hz_matrix.py"),
        "--input-root", str(args.output_dir), "--reference", str(args.motion),
        "--cohort-protocol", str(args.protocol),
        "--model", str(args.model), "--legacy-mujoco-evaluator", str(args.legacy_source/"scripts/eval_tracking_mujoco.py"),
        "--output-dir", str(args.output_dir/"analysis")]


def resolve(args):
    for name in ("training_pair_dir", "preflight_dir", "source_snapshot", "legacy_source", "model", "motion",
                 "asset_path", "beyondmimic_source", "gpu_probe_source", "guard_source", "output_dir"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    args.python = Path(args.python).expanduser().absolute()
    args.protocol = (Path(args.protocol).expanduser().resolve() if args.protocol else
                     args.source_snapshot/"config/ppo_backend_400hz_comparison_v1.json")
    require(args.source_snapshot == ROOT.resolve(), "Run the queue from its declared new immutable validation snapshot")
    require(args.physical_gpu_index in (3, 4, 5), "Use an explicitly authorized physical GPU3/4/5")
    require(not args.output_dir.exists(), "Use a fresh validation output directory")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None and
            os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") == "PCI_BUS_ID", "Require unmasked PCI_BUS_ID CUDA ordering")
    require(sha256(args.protocol) == PROTOCOL_SHA256, "This matrix belongs to the unchanged declared baseline cohort")
    for root in (args.training_pair_dir, args.preflight_dir, args.source_snapshot, args.legacy_source, args.beyondmimic_source):
        require(root.is_dir() and not args.output_dir.is_relative_to(root) and not root.is_relative_to(args.output_dir),
                "Matrix output must not overlap training, preflight or source trees")


def run(args, *, shared=None, guard=None, gpu_probe=None, child_runner=None, sleep=time.sleep,
        process_check=check_training_process):
    resolve(args)
    shared = shared or load_helper(args.source_snapshot/"scripts/ppo_backend_400hz_contract.py", "_matrix400_shared")
    guard = guard or load_helper(args.guard_source, "_matrix400_guard", GUARD_SHA256)
    if gpu_probe is None:
        gpu_probe = load_helper(args.gpu_probe_source, "_matrix400_gpu", GPU_PROBE_SHA256).query_gpu
    protocol = shared.verify_protocol(args.protocol)
    for role, path in (("motion", args.motion), ("model", args.model), ("asset", args.asset_path)):
        require(sha256(path) == shared.INPUT_PINS[role], "Pinned matrix input differs: " + role)
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir/"backend_400hz_matrix_status.json"
    status = {"schema_version": 1, "kind": "queued_matched_400hz_ppo_backend_validation_matrix",
        "status": "initializing", "pid": os.getpid(), "dispatcher_argv": list(sys.argv),
        "started_unix_time": time.time(), "protocol_path": str(args.protocol), "protocol_sha256": sha256(args.protocol),
        "training_pair_dir": str(args.training_pair_dir), "preflight_dir": str(args.preflight_dir),
        "physical_gpu_index": args.physical_gpu_index, "cases": planned_cases(args, protocol),
        "automatic_retry": False, "automatic_gpu_fallback": False, "exclusive_gpu_reservation_claimed": False,
        "training_process_modified": False, "hardware_ready": False, "policy_success_claimed": False}
    status["counts"] = counts(status["cases"])
    guard.write_json(status_path, status)
    current = None
    try:
        static = source_snapshot(args)
        guard.write_json(args.output_dir/"static_frozen_source_and_inputs.json", static)
        pair = wait_for_pair(args, status, guard, static, sleep=sleep, process_check=process_check)
        cohorts = completed_cohorts(args, pair, shared)
        verify_snapshot(args, static)  # Never adopt a changed source while adding completed actor bytes.
        frozen = source_snapshot(args, cohorts)
        for cohort in cohorts.values():
            for name, path in cohort["paths"].items():
                require(frozen["files"][str(path)]["sha256"] == cohort["input_sha256"][name],
                        "Completed actor/metadata changed while freezing the verified cohorts")
        guard.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen)
        status["completed_training_actor_bytes_frozen"] = True
        identity = pair["selected_gpu_identity"]
        status["selected_gpu_identity"] = identity
        environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
            TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
        sleep(2.)  # Returned parent training only; let stale NVML utilization settle before the fresh read.
        for current in status["cases"]:
            verify_snapshot(args, frozen, cohorts)
            cohort = cohorts[(current["phase"], current["training_backend"])]
            current.update(checkpoint_sha256=cohort["input_sha256"]["model_final.pt"], actor_sha256=cohort["input_sha256"]["policy.pt"])
            child_env = environment.copy()
            if current["engine_tag"] == "physx":
                gpu = gpu_probe(args.physical_gpu_index)
                current["gpu_preflight"] = gpu
                require(gpu["idle"] is True and gpu["physical_index"] == args.physical_gpu_index and
                        gpu["pci_bus_id_cuda_ordinal_verified"] is True and not gpu["all_reported_compute_and_graphics_processes"] and
                        gpu["uuid"] == identity["uuid"] and gpu["pci_bus_id"] == identity["pci_bus_id"],
                        "Requested GPU is busy, ambiguous or changed; no fallback/retry")
            else:
                child_env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only TorchScript inference; never allocate a GPU.
                current["gpu_allocation"] = False
            command = case_command(args, current)
            current.update(status="running", command=command, started_unix_time=time.time())
            status["status"] = "running_"+current["case_id"]
            log = args.output_dir/"logs"/(current["case_id"].replace("/", "__")+".log")
            log.parent.mkdir(exist_ok=True)
            current["log"] = str(log)
            guard.write_json(status_path, status)
            def spawned(pid):
                current["pid"] = int(pid); guard.write_json(status_path, status)
            code = (child_runner or guard.run_child)(command, log, args.source_snapshot, child_env, spawned)
            current.update(returncode=code, finished_unix_time=time.time())
            require(code == 0, "Evaluation child failed; retain outputs and stop without retry")
            verify_snapshot(args, frozen, cohorts)
            proof = shared.validate_output(Path(current["directory"]), Path(current["cohort_dir"]), args.protocol,
                validation_backend=current["validation_backend"], allow_smoke=current["smoke_only"])
            require(proof.get("status") == "verified" and proof.get("native_states") == 2793 and
                    proof.get("policy_decisions") == 349 and proof.get("duration_s") == 6.98 and
                    proof.get("physics_dt_s") == .0025 and proof.get("policy_dt_s") == .02 and
                    proof.get("validation_backend") == current["validation_backend"] and
                    proof.get("training_backend") == current["training_backend"] and
                    proof.get("smoke_only") is current["smoke_only"], "Shared native completion proof differs")
            current.update(status="completed_verified", verification=proof)
            status["counts"] = counts(status["cases"])
            guard.write_json(status_path, status)
            sleep(2.)
        status.update(status="all_validation_completed_analysis_pending", counts=counts(status["cases"]))
        manifest = {key: status[key] for key in ("kind", "protocol_path", "protocol_sha256", "training_pair_dir",
            "preflight_dir", "physical_gpu_index", "selected_gpu_identity", "cases", "counts")}
        manifest.update(schema_version=1, status="validation_matrix_completed", input_motion_sha256=sha256(args.motion),
            input_model_sha256=sha256(args.model), frozen_source_and_inputs_sha256=sha256(args.output_dir/"frozen_source_and_inputs.json"),
            smoke_pipeline_only=True, tracking_success_claimed=False)
        guard.write_json(args.output_dir/"validation_matrix_manifest.json", manifest)
        guard.write_json(status_path, status)
        current = None
        verify_snapshot(args, frozen, cohorts)
        analysis_env = environment.copy(); analysis_env["CUDA_VISIBLE_DEVICES"] = ""
        status["analysis"] = {"status": "running", "command": analysis_command(args), "gpu_allocation": False}
        guard.write_json(status_path, status)
        code = (child_runner or guard.run_child)(analysis_command(args), args.output_dir/"analysis.log", args.source_snapshot,
            analysis_env, lambda pid: status["analysis"].update(pid=int(pid)))
        status["analysis"]["returncode"] = code
        require(code == 0 and (args.output_dir/"analysis/summary.json").is_file(), "Analysis failed; all validated episodes are preserved")
        verify_snapshot(args, frozen, cohorts)
        status["analysis"].update(status="completed", summary_sha256=sha256(args.output_dir/"analysis/summary.json"))
        status["status"] = "completed_with_analysis"
    except BaseException as error:
        if current is not None and current["status"] == "running":
            current["status"] = "failed_or_unverified"
        status.update(status="analysis_failed_validation_preserved" if status.get("counts", {}).get("formal_completed") == 68
            else "stopped_without_retry", error=f"{type(error).__name__}: {error}", counts=counts(status["cases"]))
        raise
    finally:
        status["finished_unix_time"] = time.time(); guard.write_json(status_path, status)
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("training-pair-dir", "preflight-dir", "source-snapshot", "legacy-source", "model", "motion",
                 "asset-path", "beyondmimic-source", "python", "gpu-probe-source", "guard-source", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    result.add_argument("--training-dispatcher-pid", type=int)
    result.add_argument("--protocol", type=Path)
    result.add_argument("--physical-gpu-index", type=int, choices=(3, 4, 5), default=4)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())["status"]))
