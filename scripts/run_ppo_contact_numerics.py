"""Sequential fixed-actor solver-velocity4 then8 diagnostic, no training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import run_ppo_contact_damping as base
import eval_ppo_contact_numerics as evaluator

BASELINE_RAW_SHA256 = "54e01f57b01da11c5b0736c572f8b2ebc81ebe2cac8f06c9b3b5be29b1637780"


def command(args, iterations, output, kit_args):
    result = base.command(args, 10., output, kit_args)
    result[2] = str(Path(__file__).with_name("eval_ppo_contact_numerics.py"))
    start = result.index("--leg-kd"); result[start:start+2] = ["--solver-velocity-iterations", str(iterations)]
    return result


def validate_output(folder, iterations):
    import numpy as np
    report = json.loads((folder/"report.json").read_text()); audit = report["contact_numerics_diagnostic"]
    base.require(report["status"] == "evaluated" and report["simulator"] == "IsaacLab/PhysX"
        and report["checkpoint_sha256"] == base.CHECKPOINT_SHA and report["physics_dt_s"] == .005
        and report["reference_frames"] == 349 and report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .7,
        "Completed nominal frozen-actor evaluation identity differs")
    base.require(audit["kind"] == evaluator.KIND and audit["configured_solver_velocity_iterations"] == iterations
        and audit["baseline_configured_solver_velocity_iterations"] == 4 and audit["unchanged_configured_solver_position_iterations"] == 8
        and audit["unchanged_leg_kp"] == 500. and audit["unchanged_leg_kd"] == 10. and audit["additional_training_updates"] == 0
        and audit["diagnostic_only"] is True and audit["only_changed_control_field"] is None
        and audit["only_changed_numerical_field"] == ("solver_velocity_iteration_count" if iterations == 8 else None), "Numerical intervention is not isolated")
    base.require(audit["wrapper_sha256"] == base.sha256(evaluator.__file__) and report["evaluator_source_sha256"] == audit["wrapper_sha256"]
        and audit["provenance_adapter_sha256"] == evaluator.BASE_ADAPTER_SHA256
        and audit["legacy_evaluator_sha256"] == base.BASE_EVAL_SHA
        and audit["evaluation_dependency_sha256"] == evaluator.base.EVALUATION_DEPENDENCIES
        and all(audit["input_sha256"][key] == value for key, value in evaluator.base.PINS.items()), "Frozen numerical evaluator/input provenance differs")
    base.require(report["policy_export"]["contact_numerics_diagnostic"] == audit
        and "contact_stability_diagnostic" not in report and "contact_stability_diagnostic" not in report["policy_export"], "Numerical audit mislabeled as damping")
    runtime = audit["runtime_readback"]
    base.require(runtime["gain_readback"] == report["contact_material_audit"]["leg_pd_gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]}
        and runtime["clock_before"] == runtime["clock_after"] and runtime["added_physics_frames"] == 0
        and runtime["native_solver_counts_available"] is False and runtime["usd_authored_solver"]["native_backend_readback"] is False,
        "Runtime gain/no-extra-frame/evidence-scope readback differs")
    base.require(runtime["cfg_requested"]["position"] == 8 and runtime["cfg_requested"]["velocity"] == iterations
        and all(row["position"] == 8 and row["velocity"] == iterations and row["position_has_authored_value"] is True
            and row["velocity_has_authored_value"] is True for row in runtime["usd_authored_solver"]["roots"]), "Loaded USD-authored solver values differ")
    native = runtime["native_shape_readback"]
    contact, rest, materials = (np.asarray(native[key]) for key in ("contact_offsets_m", "rest_offsets_m", "materials_static_dynamic_restitution"))
    base.require(tuple(native["shape"]) == contact.shape == rest.shape and contact.shape[0] == 1
        and materials.shape == contact.shape+(3,) and all(np.isfinite(value).all() for value in (contact, rest, materials))
        and np.all(contact >= rest-1e-7) and np.allclose(materials[..., :2], .7, rtol=0, atol=1e-6)
        and all(row["sha256"] == evaluator.TENSOR_API_SHA256 for row in native["getter_source"].values()), "Finite native offset/material/source readback differs")
    raw = folder/"physics_trajectory.npz"; digest = base.sha256(raw)
    base.require(digest == report["physics_trajectory_sha256"], "Report-bound raw hash differs")
    with np.load(raw, allow_pickle=False) as archive:
        base.require(archive["time_s"].shape == (1397,) and np.array_equal(archive["time_s"], np.arange(1397)*.005)
            and archive["valid_mask"].shape == (1397, 1) and archive["valid_mask"].dtype.kind == "b" and archive["valid_mask"].all()
            and int(archive["action_was_applied"].sum()) == 349*4, "Require original full200Hz first episode and349 decisions")
    if iterations == 4:
        base.require(digest == BASELINE_RAW_SHA256, "Same-readback-hook solver4 must reproduce original baseline before solver8")
    return {"status": "verified", "solver_velocity_iterations": iterations, "report_sha256": base.sha256(folder/"report.json"),
        "recording_sha256": digest, "runtime_readback": runtime}


def compare_readbacks(before, after):
    a, b = before["runtime_readback"], after["runtime_readback"]
    for key in ("native_shape_readback", "scene_physx_iteration_limits_cfg", "gain_readback"):
        base.require(a[key] == b[key], "Additional native/configured numerical property changed: "+key)
    base.require(a["cfg_requested"]["enabled_self_collisions"] == b["cfg_requested"]["enabled_self_collisions"], "Self-collision cfg changed")
    left, right = [dict(row) for row in a["usd_authored_solver"]["roots"]], [dict(row) for row in b["usd_authored_solver"]["roots"]]
    for rows in (left, right):
        for row in rows:
            row.pop("velocity")
    base.require(left == right, "Additional loaded USD articulation setting changed")


def run(args, *, guard=None, quiesce=time.sleep):
    for name in ("guard_source", "legacy_evaluator", "motion_file", "asset_path", "checkpoint", "runner_config", "task_profile", "output_dir", "beyondmimic_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()
    base.require(not args.output_dir.exists(), "Preserve existing outputs: choose a fresh directory")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Require unmasked physical GPU1")
    base.require(base.sha256(args.checkpoint) == base.CHECKPOINT_SHA and base.sha256(args.legacy_evaluator) == base.BASE_EVAL_SHA, "Frozen actor or original evaluator SHA differs")
    official = args.beyondmimic_source/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    base.require(official.is_file(), "Explicit official BeyondMimic config is missing")
    for source in (Path(__file__).resolve().parents[1], args.legacy_evaluator.parent.parent, args.beyondmimic_source, args.checkpoint.parent):
        base.require(not args.output_dir.is_relative_to(source), "Output must remain outside frozen source/input trees")
    guard = guard or base.load_guard(args.guard_source)
    frozen = base.freeze(args); args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir/"contact_numerics_status.json"
    status = {"kind": "single_factor_solver_velocity_iterations_diagnostic", "status": "initializing", "pid": os.getpid(),
        "dispatcher_argv": list(sys.argv), "started_unix_time": time.time(), "mu": .7, "solver_velocity_iterations_in_order": [4, 8],
        "solver_position_iterations": 8, "leg_kp": 500., "leg_kd": 10., "physical_gpu_index": 1,
        "additional_training_updates": 0, "automatic_retry": False, "exclusive_gpu_reservation_claimed": False,
        "native_solver_count_claimed": False, "contact_success_claimed": False, "children": []}
    guard.write_json(args.output_dir/"frozen_source_and_inputs.json", frozen); guard.write_json(status_path, status)
    environment = os.environ.copy(); environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source))
    try:
        before = None
        for iterations in (4, 8):
            base.require(base.freeze(args) == frozen, "Frozen source/input changed")
            gpu = guard.query_gpu_one(); status["last_gpu_preflight"] = gpu; guard.write_json(status_path, status)
            base.require(gpu["idle"] is True and gpu["physical_index"] == 1, "GPU1 is not freshly fully idle; no child started")
            folder = args.output_dir/f"solver_v{iterations}"
            child = {"solver_velocity_iterations": iterations, "gpu_preflight": gpu,
                "command": command(args, iterations, folder, guard.KIT_ARGS), "status": "running", "started_unix_time": time.time()}
            status["children"].append(child); status["status"] = f"running_solver_v{iterations}"; guard.write_json(status_path, status)
            def spawned(pid):
                child["pid"] = int(pid); guard.write_json(status_path, status)
            code = guard.run_child(child["command"], args.output_dir/f"solver_v{iterations}.log", Path(__file__).resolve().parents[1], environment, spawned)
            child.update(returncode=code, finished_unix_time=time.time()); guard.write_json(status_path, status)
            base.require(code == 0, "Owned evaluator failed; no retry or following child")
            proof = validate_output(folder, iterations)
            if before is None:
                before = proof
            else:
                compare_readbacks(before, proof)
            child.update(verification=proof, status="completed_verified"); guard.write_json(status_path, status)
            quiesce(2.); base.require(base.freeze(args) == frozen, "Frozen source/input changed after episode")
        status.update(status="episodes_complete_requires_contact_and_tracking_review", paired_native_offsets_and_materials_identical=True)
    except BaseException as error:
        status.update(status="stopped_without_retry", error=f"{type(error).__name__}: {error}")
        if status["children"] and status["children"][-1]["status"] == "running":
            status["children"][-1]["status"] = "failed_or_unverified"
        raise
    finally:
        status["finished_unix_time"] = time.time(); guard.write_json(status_path, status)
    return status


if __name__ == "__main__":
    print(json.dumps(run(base.parser().parse_args())["status"]))
