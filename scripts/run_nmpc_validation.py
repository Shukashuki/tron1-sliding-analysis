"""Plan, explicitly freeze, or execute a local fixed-friction NMPC grid.

Planning never launches a subprocess or simulator. Execution requires a
byte-bound, two-engine nominal qualification and a separate freeze artifact.
This tool does not upload files, use SSH, install software, compile a backend,
reserve GPUs, or grant permission to use a remote host.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from controller_references import DEFAULT_CATALOG, ROOT, get_reference, verify_reference_pair
from nmpc_validation_contract import (
    episode_evidence, file_sha, input_bindings, make_freeze, settings_contract,
    model_nominal_plan_contract, validate_case_against_nominal, validate_freeze,
)
from run_car_validation import case_name, require_beyondmimic_source


EVALUATION = ROOT / "config/evaluation_friction.json"
MATCHED_TASK_PROFILE = ROOT / "config/car_drift_height_smooth_task.json"
CONTROLLER_FILES = (
    "scripts/run_nmpc_tracking.py", "scripts/run_nmpc_tracking_isaac.py",
    "scripts/nmpc_shooting.py", "scripts/nmpc_tracking_objective.py",
    "scripts/nmpc_batch_objective.py", "scripts/nmpc_native_batch.py",
    "scripts/classical_tracking.py", "scripts/controller_references.py",
    "scripts/height_smoothness.py", "native/nmpc_batch.cpp",
)
MATCHED_HELPERS = (
    "scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
    "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py",
)
ISAAC_HELPERS = (
    "training/tron1_tracking.py", "training/tron1_drift.py",
    "training/tron1_runtime_device.py", "training/tron1_physics_clock.py",
)
VALIDATION_FILES = (
    "scripts/run_nmpc_validation.py", "scripts/nmpc_validation_contract.py",
    "scripts/analyze_controller_pair.py", "scripts/controller_metrics.py", "scripts/controller_trace.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def grid_values(values=None):
    expected = read_json(EVALUATION)["currently_supported_equal_static_dynamic_grid"]
    chosen = list(expected if values is None else values)
    names = [case_name(value) for value in chosen]
    require(len(chosen) == len(expected) and sorted(chosen) == sorted(expected)
            and len(set(names)) == len(names), "Use exactly the predeclared 17 unique fixed coefficients")
    return chosen


def libraries(args):
    if args.prediction_backend == "python":
        require(not any((args.native_library, args.isaac_native_library, args.mujoco_native_library)),
                "Native libraries are invalid for the Python backend")
        return {"isaac": None, "mujoco": None}
    result = {engine: getattr(args, engine + "_native_library") or args.native_library
              for engine in ("isaac", "mujoco")}
    require(all(result.values()), "Explicit native library paths are required for both engines; no automatic build")
    return result


def build_plan(args):
    """Construct commands only; nonexistent runtime paths remain unverified."""
    spec, config, libs = get_reference(args.reference_id), settings_contract(args.settings), libraries(args)
    require(config["seed"] == 42, "The matched grid uses CEM seed 42 as well as Isaac plant seed 42")
    require(re.fullmatch(r"cuda:[0-9]+", args.device) is not None, "Use an explicit unmasked cuda:N device")
    target = args.output_dir.resolve()
    common = ["--motion", str(args.motion.resolve()), "--reference-id", spec["reference_id"],
              "--contract", str(args.contract.resolve()), "--task-profile", str(args.task_profile.resolve()),
              "--settings", str(args.settings.resolve()), "--prediction-backend", args.prediction_backend,
              "--prediction-mu-scenarios", *map(str, config["prediction_mu_scenarios"])]
    optional_plan = getattr(args, "model_nominal_plan", None)
    if optional_plan is not None:
        common += ["--model-nominal-plan", str(optional_plan.resolve())]
    cases = []
    for mu in grid_values(args.mu):
        name, directory = case_name(mu), target / case_name(mu)
        ia = [str(args.isaac_python.resolve()), str(ROOT / "scripts/run_nmpc_tracking_isaac.py"),
              "--source-root", str(args.source_root.resolve()), "--asset-path", str(args.asset_path.resolve()),
              "--mujoco-model", str(args.model.resolve()), *common, "--plant-mu", str(mu),
              "--num-envs", "1", "--seed", "42", "--device", args.device, "--headless",
              "--kit_args", f"--/renderer/activeGpu={args.device.split(':')[1]} --/renderer/multiGpu/enabled=false",
              "--output-dir", str(directory / "isaac")]
        mj = [str(args.mujoco_python.resolve()), str(ROOT / "scripts/run_nmpc_tracking.py"),
              "--model", str(args.model.resolve()), *common, "--plant-mu", str(mu),
              "--output-dir", str(directory / "mujoco")]
        if args.prediction_backend == "native":
            ia += ["--native-library", str(libs["isaac"].resolve())]
            mj += ["--native-library", str(libs["mujoco"].resolve())]
        cases.append({"case": name, "mu": mu, "isaac_command": ia, "mujoco_command": mj})
    return {"schema_version": 1, "kind": "unexecuted_nmpc_fixed_mu_plan", "execution_performed": False,
            "freeze_verified": False, "remote_permission_granted": False, "reference_spec": spec,
            "settings_contract": config, "prediction_backend": args.prediction_backend,
            "model_nominal_plan_request": {"requested": optional_plan is not None,
                "metadata_path": str(optional_plan.resolve()) if optional_plan is not None else None,
                "identity_verified": False, "artifact_loaded": False},
            "output_directory": str(target), "output_already_exists": target.exists(),
            "device": args.device, "cases": cases, "engine_episodes_planned": 2*len(cases),
            "pooled_success_statistics": False, "height_or_flight_failure_gate": False,
            "heading_offset_is_mismatch": False,
            "requirements_before_execution": ["Two-engine, same-controller nominal qualification at mu=.70",
                "Separate matching freeze artifact and unchanged byte bindings", "Compatible installed runtimes and explicit native build",
                "Unmasked, idle requested GPU; independent permission for any remote transfer/host use"]}


def runtime_paths(args):
    paths = {"motion": args.motion, "reference": args.reference_file, "model": args.model,
             "asset": args.asset_path, "contract": args.contract, "settings": args.settings,
             "task_profile": args.task_profile, "reference_catalog": DEFAULT_CATALOG,
             "evaluation_protocol": EVALUATION}
    paths.update({"source:" + name: ROOT / name for name in CONTROLLER_FILES})
    paths.update({"validation_source:" + name: ROOT / name for name in VALIDATION_FILES})
    optional_plan = getattr(args, "model_nominal_plan", None)
    if optional_plan is not None:
        paths.update(model_nominal_plan=optional_plan,
                     model_nominal_plan_actions=optional_plan.parent / "actions.npz")
        paths["source:scripts/model_nominal_plan.py"] = ROOT / "scripts/model_nominal_plan.py"
    paths.update({"source_root:" + name: args.source_root / name for name in (*MATCHED_HELPERS, *ISAAC_HELPERS)})
    for name in MATCHED_HELPERS:
        local, supplied = ROOT / "upstream/tron1-learning-from-human" / name, args.source_root / name
        require(local.is_file() and supplied.is_file() and file_sha(local) == file_sha(supplied),
                "Isaac source-root differs from the MuJoCo runner's actual local helper: " + name)
        paths["mujoco_source:" + name] = local
    for engine, library in libraries(args).items():
        if library is not None:
            paths[engine + "_native_library"] = library
            paths[engine + "_native_build_manifest"] = library.with_suffix(".json")
    for engine in ("isaac", "mujoco"):
        directory = getattr(args, "nominal_" + engine + "_dir", None)
        require(directory is not None, "Both actual nominal episode directories are required before freezing/execution")
        for name in ("report.json", "rollout.npz", "solver_log.json"):
            paths["nominal:" + engine + ":" + name] = directory / name
    pair = getattr(args, "nominal_pair_dir", None)
    require(pair is not None, "Actual nominal pair analysis is required")
    for name in ("summary.json", "pair_trace.npz", "topdown.png"):
        paths["nominal_pair:" + name] = pair / name
    return paths


def requested_model_nominal_plan(args, spec):
    """Pure opt-in loader validation; never called by unverified plan mode."""
    path = getattr(args, "model_nominal_plan", None)
    if path is None:
        return {"enabled": False}
    from model_nominal_plan import load_model_nominal_plan
    from nmpc_tracking_objective import HeightSmoothTrackingObjective
    objective = HeightSmoothTrackingObjective(read_json(args.task_profile)).contract()
    plan = load_model_nominal_plan(path, expected_reference_id=spec["reference_id"],
        expected_motion_sha256=file_sha(args.motion), expected_robot_reference_sha256=file_sha(args.reference_file),
        expected_model_sha256=file_sha(args.model), expected_actuator_contract_sha256=file_sha(args.contract),
        expected_objective_contract=objective)
    metadata = plan.report_contract()
    require(metadata["contract"]["identity"]["reference_spec"] == spec
            and metadata["contract"]["identity"]["task_profile_sha256"] == file_sha(args.task_profile),
            "Model nominal plan reference/task-profile identity differs from requested inputs")
    return model_nominal_plan_contract({"model_generated_nominal_plan": metadata})


def qualification(args):
    require(file_sha(args.task_profile) == file_sha(MATCHED_TASK_PROFILE),
            "Matched-controller validation requires the shared height-smooth task profile")
    spec = verify_reference_pair(args.motion, args.reference_file, args.reference_id)
    config = settings_contract(args.settings)
    require(config["seed"] == 42, "Matched-controller qualification requires CEM seed 42")
    bindings = input_bindings(runtime_paths(args))
    requested_plan = requested_model_nominal_plan(args, spec)
    evidence = {engine: episode_evidence(getattr(args, "nominal_" + engine + "_dir"), engine, spec, config, .7)
                for engine in ("isaac", "mujoco")}
    require(evidence["isaac"]["report"].get("plant_seed") == 42, "Nominal Isaac plant seed must be 42")
    require(all(item["report"].get("prediction_backend") == args.prediction_backend for item in evidence.values()),
            "Requested predictor backend differs from nominal qualification")
    require(all(model_nominal_plan_contract(item["report"]) == requested_plan for item in evidence.values()),
            "Requested --model-nominal-plan is missing or differs from enabled nominal artifact/contract")
    summary = read_json(args.nominal_pair_dir / "summary.json")
    for name in ("analyze_controller_pair.py", "controller_metrics.py", "controller_trace.py",
                 "controller_references.py", "nmpc_tracking_objective.py"):
        require(summary.get("analysis_source_sha256", {}).get(name) == file_sha(ROOT / "scripts" / name),
                "Nominal pair needs reanalysis with the current analysis source: " + name)
    for name, key in (("pair_trace.npz", "compact_trace_sha256"), ("topdown.png", "topdown_sha256")):
        require(file_sha(args.nominal_pair_dir / name) == summary.get("artifacts", {}).get(key),
                "Nominal pair artifact SHA-256 mismatch")
    freeze = make_freeze(reference_spec=spec, settings_contract=config, bindings=bindings,
                         nominal=evidence, pair_summary=summary)
    freeze["evaluation_mu"] = grid_values(args.mu)
    freeze["validation_driver_sha256"] = file_sha(Path(__file__))
    return freeze, evidence


def freeze_controller(args):
    require(args.freeze_output is not None, "Choose an explicit --freeze-output")
    output = args.freeze_output.resolve()
    require(output != args.output_dir.resolve() and args.output_dir.resolve() not in output.parents,
            "Keep the freeze artifact outside the new grid output directory")
    require(not output.exists() and not args.freeze_output.is_symlink(), "Preserving existing freeze artifact")
    frozen, _ = qualification(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    return frozen


def verified_freeze(args):
    require(args.freeze_manifest is not None, "Execution requires an explicit --freeze-manifest")
    frozen = read_json(args.freeze_manifest)
    current, _ = qualification(args)
    validate_freeze(frozen, current["reference_spec"], current["settings_contract"], current["bindings"])
    require(frozen.get("evaluation_mu") == grid_values(args.mu), "Grid differs from the predeclared frozen plan")
    require(frozen.get("validation_driver_sha256") == file_sha(Path(__file__)), "Validation driver changed after freeze")
    require(frozen == current, "Frozen nominal evidence or qualification metadata was modified")
    return frozen


def require_idle_gpu(device):
    require("CUDA_VISIBLE_DEVICES" not in os.environ, "Unset CUDA_VISIBLE_DEVICES; masked/empty visibility is unsupported")
    index = device.split(":")[1]
    result = subprocess.run(["nvidia-smi", "-i", index,
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    values = result.stdout.strip().split(",")
    require(len(values) == 2 and float(values[0]) < 64 and float(values[1]) == 0,
            "Requested GPU is not idle; no sharing, switching, reservation or process termination performed")


def launch(command, log):
    with Path(log).open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def pair_arguments(args, directory):
    return SimpleNamespace(algorithm="NMPC", reference_id=args.reference_id, motion_file=args.motion,
        reference_file=args.reference_file, model=args.model, source_root=args.source_root,
        isaac_trace=directory / "isaac/rollout.npz", isaac_report=directory / "isaac/report.json",
        mujoco_trace=directory / "mujoco/rollout.npz", mujoco_report=directory / "mujoco/report.json",
        output_dir=directory / "analysis", training_manifest=None, checkpoint=None,
        mujoco_trace_sha256=None, integration_only=False)


def execute(args):
    plan = build_plan(args)
    frozen = verified_freeze(args)
    target = args.output_dir.resolve()
    require(not target.exists(), "Preserving existing grid output; choose a new directory")
    require_beyondmimic_source(args.source_root)
    require_idle_gpu(args.device)
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "kind": "frozen_nmpc_fixed_mu_validation_grid", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "reference_spec": plan["reference_spec"],
        "freeze_sha256": file_sha(args.freeze_manifest), "controller_freeze": frozen,
        "validation_driver_sha256": file_sha(Path(__file__)), "cases": [],
        "pooled_success_statistics": False, "learned_policy_used": False,
        "height_or_flight_failure_gate": False, "heading_offset_is_mismatch": False,
        "real_time_verified": False, "actuation_delay_injected": False}
    def save():
        (target / "grid_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    save()
    try:
        from analyze_controller_pair import run as analyze_pair
        for case in plan["cases"]:
            directory = target / case["case"]
            directory.mkdir()
            row = {"case": case["case"], "mu": case["mu"], "status": "running", "engines": {}, "executions": {}}
            manifest["cases"].append(row); save()
            for engine in ("isaac", "mujoco"):
                verified_freeze(args)  # Recheck all frozen bytes before EVERY launch.
                if engine == "isaac":
                    require_idle_gpu(args.device)
                print(f"NMPC {case['case']} {engine}", flush=True)
                log = directory / (engine + ".log")
                code = launch(case[engine + "_command"], log)
                row["executions"][engine] = {"exit_code": code, "log_file": log.name, "artifacts_validated": False}
                save()
                require(type(code) is int and code == 0, f"Child exit {code}; inspect {log.name}")
                verified_freeze(args)  # Also detect a file change during the child run.
                evidence = episode_evidence(directory / engine, engine, plan["reference_spec"],
                                             plan["settings_contract"], case["mu"])
                if engine == "isaac":
                    require(evidence["report"].get("plant_seed") == 42, "Isaac validation plant seed differs from 42")
                validate_case_against_nominal(evidence, frozen["nominal"][engine])
                # Exit 0 alone is insufficient. Fallback/large physical errors
                # remain measured outcomes, never runtime-error exclusions.
                row["engines"][engine] = {key: value for key, value in evidence.items()
                                           if key not in ("report", "_directory")}
                row["executions"][engine]["artifacts_validated"] = True
                save()
            summary = analyze_pair(pair_arguments(args, directory))
            row.update(status="completed", analysis_summary_sha256=file_sha(directory / "analysis/summary.json"),
                windows=summary["windows"], posture_diagnostics=summary["posture_diagnostics"])
            save()
        require(len(manifest["cases"]) == 17 and all(row["status"] == "completed" for row in manifest["cases"]),
                "Incomplete fixed-friction grid")
        manifest.update(status="completed", completed_engine_episodes=34)
        save()
        summary = {key: value for key, value in manifest.items() if key != "controller_freeze"}
        (target / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    except BaseException as error:
        manifest.update(status="runtime_failure_incomplete", error_type=type(error).__name__, error=str(error))
        save()
        raise
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "freeze", "execute"))
    for name in ("source-root", "motion", "reference-file", "asset-path", "model", "contract", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reference-id", default="drift_drl_traj4")
    p.add_argument("--settings", type=Path, default=ROOT / "config/nmpc_height_smooth_residual.json")
    p.add_argument("--task-profile", type=Path, default=MATCHED_TASK_PROFILE)
    p.add_argument("--prediction-backend", choices=("python", "native"), default="native")
    for name in ("native-library", "isaac-native-library", "mujoco-native-library", "nominal-isaac-dir",
                 "nominal-mujoco-dir", "nominal-pair-dir", "freeze-output", "freeze-manifest", "model-nominal-plan"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--isaac-python", type=Path, default=Path(sys.executable))
    p.add_argument("--mujoco-python", type=Path, default=Path(sys.executable))
    p.add_argument("--mu", nargs="+", type=float)
    p.add_argument("--device", default="cuda:0")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    result = {"plan": build_plan, "freeze": freeze_controller, "execute": execute}[args.action](args)
    print(json.dumps(result, indent=2, allow_nan=False))
