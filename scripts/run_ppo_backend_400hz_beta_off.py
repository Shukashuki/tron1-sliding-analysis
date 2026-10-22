"""Fresh beta-reward ablation: two PPO backends, then four nominal episodes.

The immutable beta-on dispatcher supplies its source/asset freeze, parity gate,
fresh idle-GPU checks and one-child-at-a-time process handling. Only the new
trainer/contract hooks differ. Preflight stops for review; a separate explicit
formal invocation runs the approved exact budgets and nominal validation.
No historical checkpoint initializes training, no GPU fallback or retry occurs,
and the previously stopped beta-on friction matrix is never resumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BASE_DRIVER_SHA256 = "b26c779686eed3b12e713b5a40d06ad41dcda2a962a588c51c3b77366913545d"
BASE_CONTRACT_SHA256 = "b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e"
EXPECTED_INITIAL_SHA256 = "7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"
ENGINES = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}
STATUS_FILE = "backend_400hz_beta_off_status.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name, expected=None):
    path = Path(path)
    require(expected is None or sha256(path) == expected, "Frozen helper source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_cohorts(args, shared):
    """Verify actual completed beta-on files, without loading checkpoint pickle."""
    result = {}
    for origin in ("physx", "mjlab"):
        cohort = shared.verify_cohort(args.beta_on_pair_dir / (origin + "_formal"), args.beta_on_protocol)
        require(cohort["training_backend"] == origin and cohort["smoke_only"] is False,
                "Expected both completed, formal beta-on baselines")
        require(cohort["manifest"]["initial_policy_state_sha256"] == EXPECTED_INITIAL_SHA256,
                "Beta-on initial network differs from the locked common initialization")
        require(cohort["contract"]["trained_task_profile"]["reward_weights"]["sideslip"] == 2.,
                "Reference baseline is not beta-reward-on")
        result[origin] = cohort
    return result


def cohort_files(cohorts):
    files = {}
    for cohort in cohorts.values():
        for name, path in cohort["paths"].items():
            digest = cohort["input_sha256"][name]
            require(sha256(path) == digest, "Verified training artifact changed before freezing: " + str(path))
            files[str(path)] = digest
    return files


def verify_files(files):
    require(all(sha256(path) == digest for path, digest in files.items()),
            "Frozen completed training artifact changed")


def compare_training(cohort, baseline, args):
    """Check the intervention and live backend identity beyond contract hashes."""
    profile = copy.deepcopy(cohort["contract"]["trained_task_profile"])
    require(profile["reward_weights"]["sideslip"] == 0., "Beta-off sideslip reward must be exactly zero")
    profile["reward_weights"]["sideslip"] = 2.
    require(profile == baseline["contract"]["trained_task_profile"],
            "More than the sideslip reward changed in the task profile")
    initial = cohort["manifest"]["initial_policy_state_sha256"]
    require(initial == baseline["manifest"]["initial_policy_state_sha256"] == EXPECTED_INITIAL_SHA256,
            "Beta-off and beta-on initial network identities differ")
    runtime = cohort["manifest"]["cohort"]["runtime"]
    device = runtime["device"]
    require(device["requested_device"] == f"cuda:{args.physical_gpu_index}"
            and device["torch_current_device"] == args.physical_gpu_index
            and device["cuda_visible_devices"] is None,
            "Training masked or changed the requested physical CUDA ordinal")
    if cohort["training_backend"] == "physx":
        require(device["status"] == "verified" and device["native_and_tensor_device_consistency_verified"] is True
                and device["environment_device"] == device["physics_sim_view_frontend_device"] ==
                device["robot_native_transform_tensor_device"] == f"cuda:{args.physical_gpu_index}"
                and device["native_physics_backend_device_ordinal"] == device["physics_cudaDevice_setting"] == args.physical_gpu_index,
                "PhysX native and tensor device ordinals differ")
    else:
        native = device["native_backend"]
        require(device["task_arrays_device"] == native["device"] == f"cuda:{args.physical_gpu_index}"
                and native["physics_dt_s"] == .0025 and native["decimation"] == 8
                and native["host_task_adapter"] is False,
                "MJWarp task is not the declared native GPU implementation")
        counters = json.loads((cohort["directory"] / "environment_final.json").read_text())["physical_counters"]
        transitions = cohort["completion"]["training_environment_steps"] + 8 * cohort["manifest"]["num_envs"]
        require(counters["environment_transitions"] == transitions and
                counters["physical_environment_substeps"] == transitions * 8,
                "Actual MJWarp transition/substep counters differ")
    return {"beta_on_checkpoint_sha256": baseline["input_sha256"]["model_final.pt"],
        "beta_on_actor_sha256": baseline["input_sha256"]["policy.pt"],
        "initial_network_identical_to_beta_on": True,
        "only_task_profile_difference": "reward_weights.sideslip:2.0->0.0",
        "beta_off_reward_weight": 0., "beta_on_reward_weight": 2.}


def training_command(base, args, backend, mode, folder):
    command = base(args, backend, mode, folder)
    old = str(args.snapshot / "scripts/train_ppo_backend_400hz.py")
    require(command.count(old) == 1, "Training command anchor is absent/ambiguous")
    command[command.index(old)] = str(args.snapshot / "scripts/train_ppo_backend_400hz_beta_off.py")
    return command


def validate_training(folder, backend, mode, args, shared, baselines, frozen_baseline):
    verify_files(frozen_baseline)
    cohort = shared.verify_cohort(folder, args.protocol, allow_smoke=mode == "smoke")
    require(cohort["training_backend"] == backend and cohort["completion"]["run_kind"] == mode,
            "Completed beta-off child backend or run kind differs")
    comparison = compare_training(cohort, baselines[backend], args)
    declared = folder.with_name(folder.name + "_declared_inputs")
    require({"contract": sha256(declared / "contract.json"),
             "task_profile": sha256(declared / "task_profile.json")} == args.derived_input_sha256,
            "Actual training inputs differ from the checked parity inputs")
    require(cohort["manifest"]["cohort"]["derived_input_sha256"] == args.derived_input_sha256,
            "Completed child declaration differs from parity inputs")
    for name, digest in cohort["manifest"]["cohort"]["source_files_sha256"].items():
        require(sha256(args.snapshot / name) == digest, "Training source changed after startup: " + name)
    return {"status": "verified", "backend": backend, "run_kind": mode,
        "training_environment_steps": cohort["completion"]["training_environment_steps"],
        "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
        "exported_policy_sha256": cohort["input_sha256"]["policy.pt"],
        "initial_policy_state_sha256": cohort["manifest"]["initial_policy_state_sha256"],
        "cohort_completion_audit_sha256": cohort["input_sha256"]["cohort_completion_audit.json"],
        "manifest_sha256": cohort["input_sha256"]["manifest.json"],
        "validation_matrix_status": "pending_not_evaluated", "beta_ablation": comparison}


def nominal_cases(args):
    return [{"case_id": f"formal/{origin}/{engine}/mu0p700", "phase": "formal",
        "training_backend": origin, "engine_tag": engine, "validation_backend": label, "mu": .7,
        "cohort_dir": str(args.output_dir / (origin + "_formal")),
        "directory": str(args.output_dir / "nominal_validation" / origin / engine / "mu0p700"),
        "status": "pending", "smoke_only": False}
        for origin in ("physx", "mjlab") for engine, label in ENGINES.items()]


def evaluation_command(args, case):
    engine = case["engine_tag"]
    command = [str(args.python), "-u", str(args.snapshot / "scripts" /
        ("eval_ppo_backend_400hz_beta_off_isaac.py" if engine == "physx" else "eval_ppo_backend_400hz_beta_off_mujoco.py")),
        "--cohort-dir", case["cohort_dir"], "--cohort-protocol", str(args.protocol),
        "--legacy-evaluator", str(args.legacy_source / "scripts" /
            ("eval_tracking_friction_isaac.py" if engine == "physx" else "eval_tracking_mujoco.py")),
        "--motion-file", str(args.motion), "--output-dir", case["directory"],
        "--evaluation-contact-friction", ".7"]
    if engine == "physx":
        command.extend(("--asset-path", str(args.asset_path), "--physical-gpu-index", str(args.physical_gpu_index), "--headless"))
    else:
        command.extend(("--model", str(args.model)))
    return command


def run_nominal(args, status, base, shared, baselines, guard, gpu_probe, child_runner, sleep):
    """One fixed-friction episode per cell, no optimization or success gate."""
    cohorts = {origin: shared.verify_cohort(args.output_dir / (origin + "_formal"), args.protocol)
               for origin in ("physx", "mjlab")}
    frozen = cohort_files({**{("off", k): v for k, v in cohorts.items()},
                           **{("on", k): v for k, v in baselines.items()}})
    source_frozen = base.source_snapshot(args)
    guard.write_json(args.output_dir / "nominal_frozen_completed_artifacts.json", frozen)
    cases = status["nominal_cases"] = nominal_cases(args)
    status["nominal_counts"] = {"expected": 4, "completed_verified": 0, "runtime_failed": 0}
    identity = status["selected_gpu_identity"]
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                       TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    current = None
    try:
        for current in cases:
            base.verify_snapshot(args, source_frozen)
            verify_files(frozen)
            child_env = environment.copy()
            if current["engine_tag"] == "physx":
                gpu = gpu_probe(args.physical_gpu_index)
                current["gpu_preflight"] = gpu
                require(gpu["idle"] is True and gpu["physical_index"] == args.physical_gpu_index
                        and gpu["pci_bus_id_cuda_ordinal_verified"] is True
                        and not gpu["all_reported_compute_and_graphics_processes"]
                        and gpu["uuid"] == identity["uuid"] and gpu["pci_bus_id"] == identity["pci_bus_id"],
                        "Requested GPU is not freshly completely idle; no fallback/retry")
            else:
                child_env["CUDA_VISIBLE_DEVICES"] = ""
                current["gpu_allocation"] = False
            current.update(command=evaluation_command(args, current), status="running", started_unix_time=time.time())
            status["status"] = "running_nominal_" + current["training_backend"] + "_" + current["engine_tag"]
            guard.write_json(args.output_dir / "backend_400hz_pair_status.json", status)
            log = args.output_dir / ("nominal_" + current["training_backend"] + "_" + current["engine_tag"] + ".log")
            def spawned(pid):
                current["pid"] = int(pid)
                guard.write_json(args.output_dir / "backend_400hz_pair_status.json", status)
            code = (child_runner or guard.run_child)(current["command"], log, args.snapshot, child_env, spawned)
            current.update(returncode=code, finished_unix_time=time.time())
            require(code == 0, "Nominal evaluation child failed; no retry")
            base.verify_snapshot(args, source_frozen)
            verify_files(frozen)
            proof = shared.validate_output(Path(current["directory"]), Path(current["cohort_dir"]), args.protocol,
                validation_backend=current["validation_backend"])
            require(proof["status"] == "verified" and proof["native_states"] == 2793 and
                    proof["policy_decisions"] == 349 and proof["duration_s"] == 6.98 and
                    proof["physics_dt_s"] == .0025 and proof["policy_dt_s"] == .02 and
                    proof["training_backend"] == current["training_backend"] and
                    proof["validation_backend"] == current["validation_backend"] and
                    proof["evaluation_contact_friction"] == .7 and proof["smoke_only"] is False,
                    "Native nominal completion proof differs")
            current.update(status="completed_verified", verification=proof,
                checkpoint_sha256=cohorts[current["training_backend"]]["input_sha256"]["model_final.pt"],
                actor_sha256=cohorts[current["training_backend"]]["input_sha256"]["policy.pt"])
            status["nominal_counts"]["completed_verified"] += 1
            guard.write_json(args.output_dir / "backend_400hz_pair_status.json", status)
            sleep(2.)
        status.update(status="training_and_nominal_validation_completed", validation_matrix_status="nominal4_completed_remaining_friction_not_launched")
    except BaseException as error:
        if current is not None and current["status"] == "running":
            current["status"] = "failed_or_unverified"
            status["nominal_counts"]["runtime_failed"] += 1
        status.update(status="stopped_without_retry_training_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_unix_time"] = time.time()
        guard.write_json(args.output_dir / "backend_400hz_pair_status.json", status)
    return status


def run(args, *, base=None, shared=None, on_shared=None, wrapper=None, guard=None,
        gpu_probe=None, child_runner=None, sleep=time.sleep):
    base = base or load(ROOT / "scripts/run_ppo_backend_400hz_pair.py", "_beta_off_base_dispatcher", BASE_DRIVER_SHA256)
    base._resolve(args)
    for name in ("beta_on_pair_dir", "beta_on_protocol", "legacy_source"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    require(args.beta_on_pair_dir.is_dir() and args.legacy_source.is_dir(), "Existing authorized baseline/evaluator source unavailable")
    require(not args.output_dir.is_relative_to(args.beta_on_pair_dir) and not args.beta_on_pair_dir.is_relative_to(args.output_dir),
            "Never write beta-off outputs into the completed beta-on cohort")
    shared = shared or load(ROOT / "scripts/ppo_backend_400hz_beta_contract.py", "_beta_off_contract")
    on_shared = on_shared or load(ROOT / "scripts/ppo_backend_400hz_contract.py", "_beta_on_contract", BASE_CONTRACT_SHA256)
    wrapper = wrapper or load(ROOT / "scripts/train_ppo_backend_400hz_beta_off.py", "_beta_off_trainer")
    guard = guard or base._load(args.guard_source, "_beta_off_guard", base.GUARD_SHA256)
    gpu_probe = gpu_probe or base._load(args.gpu_probe_source, "_beta_off_gpu", base.GPU_PROBE_SHA256).query_gpu
    baselines = baseline_cohorts(args, on_shared)
    frozen_baseline = cohort_files(baselines)
    original_write = guard.write_json
    def write(path, value):
        path = Path(path)
        if path.name == "backend_400hz_pair_status.json":
            value.update(kind="fresh_400hz_beta_reward_off_backend_ablation", beta_reward_enabled=False,
                beta_on_pair_dir=str(args.beta_on_pair_dir), beta_on_protocol_sha256=sha256(args.beta_on_protocol),
                beta_on_artifacts_readonly=True, beta_on_matrix_resumed=False,
                beta_on_checkpoint_initialization=False, tracking_success_claimed=False)
            original_write(path, value)
            original_write(path.with_name(STATUS_FILE), value)
        else:
            original_write(path, value)
    proxy = SimpleNamespace(write_json=write, run_child=guard.run_child)
    old_training = base.training_command
    old_validate = base.validate_training
    base.training_command = lambda *parts: training_command(old_training, *parts)
    base.validate_training = lambda folder, backend, mode, settings: validate_training(
        folder, backend, mode, settings, shared, baselines, frozen_baseline)
    try:
        status = base.run(args, guard=proxy, gpu_probe=gpu_probe, wrapper=wrapper,
                          child_runner=child_runner, quiesce=sleep)
    finally:
        base.training_command, base.validate_training = old_training, old_validate
    if args.stage == "formal":
        status = run_nominal(args, status, base, shared, baselines, proxy, gpu_probe, child_runner, sleep)
    return status


def parser():
    base = load(ROOT / "scripts/run_ppo_backend_400hz_pair.py", "_beta_off_parser", BASE_DRIVER_SHA256)
    result = base.parser()
    result.description = __doc__
    result.set_defaults(protocol=ROOT / "config/ppo_backend_400hz_beta_off_v1.json")
    result.add_argument("--beta-on-pair-dir", type=Path, required=True)
    result.add_argument("--beta-on-protocol", type=Path, default=ROOT / "config/ppo_backend_400hz_comparison_v1.json")
    result.add_argument("--legacy-source", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())["status"]))
