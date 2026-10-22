"""Fresh-idle-GPU, fixed-actor Kp500/Kp250 nominal diagnostics, no training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import run_ppo_contact_damping as base


def validate_output(folder, kp):
    import numpy as np

    report = json.loads((folder / "report.json").read_text())
    base.require(report["status"] == "evaluated" and report["simulator"] == "IsaacLab/PhysX"
                 and report["checkpoint_sha256"] == base.CHECKPOINT_SHA,
                 "Episode did not complete with the frozen actor")
    audit = report["contact_stability_diagnostic"]
    base.require(audit["kind"] == "fixed_ppo_nominal_contact_stiffness_diagnostic"
                 and audit["deployment_leg_kp"] == kp and audit["trained_leg_kp"] == 500.
                 and audit["unchanged_leg_kd"] == 10. and audit["additional_training_updates"] == 0
                 and audit["diagnostic_only"] is True,
                 "Stiffness intervention is not isolated")
    gains = report["contact_material_audit"]["leg_pd_gain_readback"]
    base.require(gains == {"kp_range": [kp, kp], "kd_range": [10., 10.]}
                 and report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7,
                 "Actual PD or nominal material differs")
    raw = folder / "physics_trajectory.npz"
    base.require(base.sha256(raw) == report["physics_trajectory_sha256"], "Recording/report hash differs")
    with np.load(raw, allow_pickle=False) as archive:
        base.require(archive["time_s"].shape == (1397,)
                     and np.allclose(archive["time_s"], np.arange(1397)*.005, atol=1e-9, rtol=0)
                     and archive["valid_mask"].shape == (1397, 1) and archive["valid_mask"].all(),
                     "Require full native first episode without resets")
    if kp == 500.:
        base.require(base.sha256(raw) == "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780",
                     "New entrypoint must reproduce the frozen nominal -0.5 baseline before Kp250")
    return {"status": "verified", "report_sha256": base.sha256(folder/"report.json"),
            "recording_sha256": base.sha256(raw), "deployment_leg_kp": kp}


def command(args, kp, output, kit_args):
    result = base.command(args, 10., output, kit_args)
    result[2] = str(Path(__file__).with_name("eval_ppo_contact_stiffness.py"))
    index = result.index("--leg-kd")
    result[index:index+2] = ["--leg-kp", str(kp)]
    return result


def run(args):
    for name in ("guard_source", "legacy_evaluator", "motion_file", "asset_path", "checkpoint",
                 "runner_config", "task_profile", "output_dir", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    base.require(not args.output_dir.exists(), "Preserve existing outputs: use a fresh directory")
    base.require(base.sha256(args.checkpoint) == base.CHECKPOINT_SHA
                 and base.sha256(args.legacy_evaluator) == base.BASE_EVAL_SHA,
                 "Frozen actor or base evaluator changed")
    guard = base.load_guard(args.guard_source)
    frozen = base.freeze(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "stiffness_status.json"
    status = {"kind": "zero_shot_single_factor_leg_stiffness_diagnostic", "status": "initializing",
              "pid": os.getpid(), "dispatcher_argv": list(sys.argv), "started_unix_time": time.time(),
              "mu": .7, "leg_kp_in_order": [500., 250.], "unchanged_leg_kd": 10.,
              "physical_gpu_index": 1, "additional_training_updates": 0, "children": [],
              "automatic_retry": False, "exclusive_gpu_reservation_claimed": False,
              "contact_success_claimed": False}
    guard.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen)
    guard.write_json(status_path, status)
    environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                       TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    try:
        for kp in (500., 250.):
            base.require(base.freeze(args) == frozen, "Frozen source/input changed")
            gpu = guard.query_gpu_one()
            base.require(gpu["idle"] is True, "Physical GPU 1 is not freshly fully idle")
            folder = args.output_dir/f"kp{int(kp)}"
            child = {"deployment_leg_kp": kp, "gpu_preflight": gpu,
                     "command": command(args, kp, folder, guard.KIT_ARGS), "status": "running"}
            status["children"].append(child); status["status"] = f"running_kp{int(kp)}"
            guard.write_json(status_path, status)
            def spawned(pid):
                child["pid"] = pid; guard.write_json(status_path, status)
            code = guard.run_child(child["command"], args.output_dir/f"kp{int(kp)}.log",
                                   Path(__file__).resolve().parent.parent, environment, spawned)
            child.update(returncode=code, finished_unix_time=time.time())
            guard.write_json(status_path, status)
            base.require(code == 0, "Owned evaluator failed; no automatic retry or next diagnostic")
            child["verification"] = validate_output(folder, kp); child["status"] = "completed_verified"
            guard.write_json(status_path, status)
            time.sleep(2.)
            base.require(base.freeze(args) == frozen, "Frozen source/input changed after episode")
        status["status"] = "episodes_complete_requires_contact_and_tracking_review"
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}"); raise
    finally:
        status["finished_unix_time"] = time.time(); guard.write_json(status_path, status)
    return status


if __name__ == "__main__":
    print(json.dumps(run(base.parser().parse_args())["status"]))
