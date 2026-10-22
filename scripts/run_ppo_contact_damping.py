"""Sequential nominal, zero-shot damping diagnostics on freshly idle GPU 1.

The existing -0.5 actor is kept fixed. This runs two single-environment
episodes, Kd=10 then Kd=5, without training, retries, or foreign-process control.
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


GUARD_SHA = "d07e7b04af9554744ac70fa7dc149760171e55978da2791d9721ce4916f472e0"
CHECKPOINT_SHA = "8ba4178471c757b4d09f5bb44c0a508558f1bfd5cc15103195aeba420ebe6e9a"
BASE_EVAL_SHA = "76d78337d70519693bc7e8d819a6523a800e1e4889de5a0466c2d34356d2ae9d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_guard(path):
    require(sha256(path) == GUARD_SHA, "GPU guard source differs from the audited version")
    spec = importlib.util.spec_from_file_location("contact_damping_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(args):
    paths = set()
    for root in (Path(__file__).resolve().parent.parent, args.legacy_evaluator.parent.parent):
        for directory in (root / "scripts", root / "training", root / "config"):
            if directory.is_dir():
                for path in directory.rglob("*"):
                    if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".yml"):
                        require(not path.is_symlink(), "Refuse symlinked input: " + str(path))
                        paths.add(path)
    paths.update((args.guard_source, args.motion_file, args.asset_path, args.checkpoint,
                  args.runner_config, args.task_profile, args.legacy_evaluator))
    paths.update(args.checkpoint.parent / name for name in
                 ("manifest.json", "ablation_manifest.json", "run_status.json"))
    require(args.beyondmimic_source.is_dir(), "Explicit existing BeyondMimic source is missing")
    paths.update(path for path in args.beyondmimic_source.rglob("*.py") if path.is_file())
    return {str(path): sha256(path) for path in sorted(paths)}


def command(args, kd, output, kit_args):
    return [str(args.python), "-u", str(Path(__file__).with_name("eval_ppo_contact_damping.py")),
            "--legacy-evaluator", str(args.legacy_evaluator), "--leg-kd", str(kd),
            "--motion-file", str(args.motion_file), "--asset-path", str(args.asset_path),
            "--checkpoint", str(args.checkpoint), "--runner-config", str(args.runner_config),
            "--task-profile", str(args.task_profile), "--output-dir", str(output),
            "--evaluation-contact-friction", "0.7", "--num-envs", "1", "--seed", "42",
            "--contact-threshold-n", "5.0", "--record-physics", "--verify-unmasked-device",
            "--device", "cuda:1", "--headless", "--kit_args", kit_args]


def validate_output(folder, kd):
    import numpy as np

    report = json.loads((folder / "report.json").read_text())
    require(report["status"] == "evaluated" and report["simulator"] == "IsaacLab/PhysX"
            and report["checkpoint_sha256"] == CHECKPOINT_SHA,
            "Episode did not complete with the frozen -0.5 actor")
    require(report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7
            and report["physics_dt_s"] == .005 and report["reference_frames"] == 349,
            "Nominal material, physical clock or horizon differs")
    audit = report["contact_stability_diagnostic"]
    require(audit["deployment_leg_kd"] == kd and audit["trained_leg_kd"] == 10.
            and audit["unchanged_leg_kp"] == 500. and audit["additional_training_updates"] == 0
            and audit["checkpoint_sha256"] == CHECKPOINT_SHA and audit["diagnostic_only"] is True,
            "Damping override is missing or not isolated")
    gains = report["contact_material_audit"]["leg_pd_gain_readback"]
    require(gains["kp_range"] == [500., 500.] and gains["kd_range"] == [kd, kd],
            "Actual actuator gains differ from the diagnostic")
    raw = folder / "physics_trajectory.npz"
    require(sha256(raw) == report["physics_trajectory_sha256"], "Recording/report SHA mismatch")
    with np.load(raw, allow_pickle=False) as archive:
        require(archive["time_s"].shape == (1397,)
                and np.allclose(archive["time_s"], np.arange(1397) * .005, atol=1e-9, rtol=0)
                and archive["valid_mask"].shape == (1397, 1) and archive["valid_mask"].all(),
                "Require all original native first-episode states, not a reset-spliced trajectory")
    return {"status": "verified", "deployment_leg_kd": kd,
            "report_sha256": sha256(folder / "report.json"), "recording_sha256": sha256(raw)}


def run(args):
    for name in ("guard_source", "legacy_evaluator", "motion_file", "asset_path", "checkpoint",
                 "runner_config", "task_profile", "output_dir", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    require(not args.output_dir.exists(), "Preserve existing runs: choose a fresh output directory")
    require(sha256(args.checkpoint) == CHECKPOINT_SHA and sha256(args.legacy_evaluator) == BASE_EVAL_SHA,
            "Checkpoint or original evaluator identity differs")
    guard = load_guard(args.guard_source)
    frozen = freeze(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "damping_status.json"
    status = {"kind": "zero_shot_single_factor_leg_damping_diagnostic", "status": "initializing",
              "pid": os.getpid(), "started_unix_time": time.time(), "mu": .7,
              "dispatcher_argv": list(sys.argv), "physical_gpu_index": 1,
              "leg_kd_in_order": [10., 5.], "unchanged_leg_kp": 500.,
              "additional_training_updates": 0, "children": [], "automatic_retry": False,
              "exclusive_gpu_reservation_claimed": False, "contact_success_claimed": False}
    guard.write_json(args.output_dir / "frozen_source_and_inputs.json", frozen)
    guard.write_json(status_path, status)
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                       TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    try:
        for kd in (10., 5.):
            require(freeze(args) == frozen, "Frozen source or inputs changed")
            evidence = guard.query_gpu_one()
            require(evidence["idle"] is True, "Physical GPU 1 is not fully idle; no child launched")
            folder = args.output_dir / ("kd" + str(int(kd)))
            child = {"deployment_leg_kd": kd, "gpu_preflight": evidence,
                     "command": command(args, kd, folder, guard.KIT_ARGS), "status": "running"}
            status["children"].append(child)
            status["status"] = "running_kd" + str(int(kd))
            guard.write_json(status_path, status)

            def spawned(pid):
                child["pid"] = pid
                guard.write_json(status_path, status)

            code = guard.run_child(child["command"], args.output_dir / f"kd{int(kd)}.log",
                                   Path(__file__).resolve().parent.parent, environment, spawned)
            child.update(returncode=code, finished_unix_time=time.time())
            guard.write_json(status_path, status)
            require(code == 0, "Owned evaluator failed; no retry or next diagnostic")
            child["verification"] = validate_output(folder, kd)
            child["status"] = "completed_verified"
            guard.write_json(status_path, status)
            time.sleep(2.)  # Quiesce only after our exact child exits.
            require(freeze(args) == frozen, "Frozen source or inputs changed after evaluation")
        status["status"] = "episodes_complete_requires_contact_and_tracking_review"
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_unix_time"] = time.time()
        guard.write_json(status_path, status)
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("guard-source", "legacy-evaluator", "motion-file", "asset-path", "checkpoint",
                 "runner-config", "task-profile", "output-dir", "python", "beyondmimic-source"):
        result.add_argument("--" + name, type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())["status"]))
