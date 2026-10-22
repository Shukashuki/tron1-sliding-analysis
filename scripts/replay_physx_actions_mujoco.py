"""Replay one pinned PhysX request sequence in native CPU MuJoCo, without an actor.

Same normalized requests imply the same leg position targets and wheel effort
requests, NOT identical applied leg torques: the explicit PD sees each plant's
own state. This is an open-loop backend diagnostic, not policy transfer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np


INPUT_PINS = {
    "physx_physics": "8e7fc43c181b34b9ff87fe42d3c628ab483c7a6cddfb965562df250d77e46c1e",
    "physx_report": "abe05d4ea69634e9794514fc60664cf4c67fe718ad841716d5518923671e778e",
    "motion": "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254",
    "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513",
}
SOURCE_PINS = {
    "scripts/eval_tracking_mujoco.py": "2251d740d7568874a47a232556241cc8d1ee5f5b7be1351d1c657aea8fa85f87",
    "scripts/eval_tracking.py": "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992",
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "training/tron1_drift.py": "5fee46238c825a2bcf474e969e93af65834aa1302ce857aa1c553e8a9fbaefaa",
    "training/tron1_physics_clock.py": "bcd89d6a24652d4501baf6744b46944d369fd224a9ab4727fc8b6073a20cf44e",
    "training/tron1_full_horizon_diagnostic.py": "2b30ac0f27af5b0e702c464392d9a74d1a34c35a92fc0e9643ceca46b5a1522d",
    "training/tron1_terrain.py": "dec819a3c3c526fce706fb6a5a6491388298884b32bfe73a5c331b166f976ebf",
}
JOINTS = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
          "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"]
FIELD_MAP = {
    "root_position_m": "root_pos", "root_quaternion_wxyz": "root_quat_wxyz",
    "root_linear_velocity_w_m_s": "root_com_lin_vel_w",
    "root_angular_velocity_w_rad_s": "root_ang_vel_w",
    "joint_position_rad": "joint_pos", "joint_velocity_rad_s": "joint_vel",
    "applied_joint_torque_nm": "joint_torque_nm", "wheel_position_m": "wheel_pos",
    "wheel_ground_contact_count": "wheel_ground_contact_count",
    "wheel_ground_contact_force_w_n": "wheel_ground_contact_force_w_n",
    "wheel_lateral_contact_slip_m_s": "wheel_lateral_contact_slip_m_s",
    "nonwheel_ground_contact_count": "nonwheel_ground_contact_count",
    "nonwheel_ground_contact_force_norm_n": "nonwheel_ground_contact_force_norm_n",
}
INITIAL_STATE_FIELDS = tuple(list(FIELD_MAP)[:6])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def extract_actions(physics):
    """Validate native 400 Hz held requests, retaining all 349 decisions."""
    count, decimation = 349, 8
    n = 1 + count * decimation
    t = np.asarray(physics["time_s"])
    require(t.shape == (n,) and np.allclose(t, np.arange(n)*.0025, atol=1e-12, rtol=0),
            "Expected one complete native400Hz episode")
    require(float(physics["physics_dt_s"]) == .0025 and float(physics["policy_dt_s"]) == .02,
            "Recorded physics/policy clocks differ")
    require(physics["joint_names"].astype(str).tolist() == JOINTS, "Named PhysX action order differs")
    for name in ("valid_mask", "action_was_applied"):
        value = np.asarray(physics[name])
        require(value.shape == (n, 1) and value.dtype.kind == "b", "Invalid " + name)
    require(physics["valid_mask"].all() and not physics["action_was_applied"][0, 0]
            and physics["action_was_applied"][1:, 0].all(), "Invalid reset/action coverage")
    frames = np.asarray(physics["reference_frame_for_action"])
    require(frames.shape == (n, 1) and np.array_equal(frames[1:, 0], np.repeat(np.arange(count), decimation)),
            "Held action reference frames differ")
    for name in ("actions_raw", "actions_clipped"):
        value = np.asarray(physics[name])
        require(value.shape == (n, 1, 8) and np.isfinite(value).all(), "Malformed " + name)
        require(np.array_equal(value[1:, 0].reshape(count, decimation, 8),
                               np.repeat(value[1::decimation, 0, None, :], decimation, axis=1)),
                "A request changed inside its eight-substep hold")
    require(np.array_equal(physics["actions_clipped"], np.clip(physics["actions_raw"], -1., 1.)),
            "Recorded clipping semantics differ")
    return physics["actions_raw"][1::decimation, 0].copy()


def audit_module_origins(source_root):
    """Fail on stale imports from the live checkout or an unpinned local helper."""
    source_root = Path(source_root).resolve()
    found = {}
    for name, module in list(sys.modules.items()):
        if name == "training":
            paths = list(getattr(module, "__path__", []))
            require(paths and all(Path(p).resolve() == source_root/"training" for p in paths),
                    "Foreign training namespace was already imported")
        if name == "eval_tracking" or name.startswith("training.") or name == "_pinned_request_replay_evaluator":
            path = getattr(module, "__file__", None)
            require(path is not None, "Runtime source path unavailable: " + name)
            path = Path(path).resolve()
            require(path.is_relative_to(source_root), "Foreign runtime module: " + name)
            relative = path.relative_to(source_root).as_posix()
            require(relative in SOURCE_PINS and sha256(path) == SOURCE_PINS[relative],
                    "Unpinned runtime source: " + name)
            found[name] = {"relative_path": relative, "sha256": SOURCE_PINS[relative]}
    require("torch" not in sys.modules, "This actor-free diagnostic must not import Torch")
    return found


def load_evaluator(path):
    """Import only pinned CPU sources; no source rewriting or simulator stepping."""
    path = Path(path).resolve()
    root = path.parent.parent
    require(path == root/"scripts/eval_tracking_mujoco.py", "Unexpected evaluator layout")
    for relative, expected in SOURCE_PINS.items():
        require((root/relative).is_file() and sha256(root/relative) == expected,
                "Source pin differs: " + relative)
    require("torch" not in sys.modules, "Use an actor-free fresh Python process")
    audit_module_origins(root)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root/"scripts"), str(root)]
    spec = importlib.util.spec_from_file_location("_pinned_request_replay_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit_module_origins(root)
    return module


def compare_native(physx, mujoco, steps, actions):
    n = 1 + steps*8
    require(np.asarray(mujoco["time_s"]).shape == (n,) and
            float(mujoco["physics_dt_s"]) == .0025 and float(mujoco["policy_dt_s"]) == .02 and
            np.allclose(mujoco["time_s"], physx["time_s"][:n], atol=1e-12, rtol=0),
            "Native clocks differ")
    require(mujoco["joint_names"].astype(str).tolist() == JOINTS, "MuJoCo joint order differs")
    require(np.array_equal(mujoco["actions_raw"], actions[:steps]) and
            np.array_equal(mujoco["actions_clipped"], np.clip(actions[:steps], -1., 1.)),
            "MuJoCo requests differ from the selected recorded sequence")
    arrays = {"time_s": physx["time_s"][:n].copy(), "requested_actions_raw": actions[:steps].copy(),
              "requested_actions_clipped": np.clip(actions[:steps], -1., 1.)}
    initial = {}
    for left, right in FIELD_MAP.items():
        a, b = np.asarray(physx[left])[:n, 0], np.asarray(mujoco[right])
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                "Native signal shape/value differs: " + left)
        arrays["physx_"+left], arrays["mujoco_"+left] = a.copy(), b.copy()
        if left in INITIAL_STATE_FIELDS:
            delta = float(np.max(np.abs(a[0]-b[0])))
            require(delta <= 1e-8, "Initial observable state differs: " + left)
            initial[left] = delta
    error = arrays["mujoco_root_position_m"][:, :2] - arrays["physx_root_position_m"][:, :2]
    return arrays, {"initial_observable_max_abs_difference": initial,
        "same_time_interengine_xy_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "same_time_interengine_xy_peak_m": float(np.linalg.norm(error, axis=1).max()),
        "initial_contact_force_difference_max_abs_n": float(np.max(np.abs(
            arrays["mujoco_wheel_ground_contact_force_w_n"][0]-arrays["physx_wheel_ground_contact_force_w_n"][0]))),
        "requests_equal_recorded_source_prefix": True, "native_states": n}


def run(args):
    require(type(args.max_policy_steps) is int and 1 <= args.max_policy_steps <= 349,
            "Choose 1..349 policy decisions (10=.20s;349=full 6.98s)")
    require(version("mujoco") == "3.8.1", "The recorded comparison uses native MuJoCo 3.8.1")
    output = args.output_dir.resolve()
    require(not output.exists(), "Preserving prior output: use a fresh directory")
    for role, expected in INPUT_PINS.items():
        path = Path(getattr(args, role)).resolve()
        require(path.is_file() and sha256(path) == expected, "Input pin differs: " + role)
    report = json.loads(args.physx_report.read_text())
    require(report["physics_trajectory_sha256"] == INPUT_PINS["physx_physics"] and
            report["motion_file_sha256"] == INPUT_PINS["motion"], "PhysX report binding differs")
    contract = copy.deepcopy(report["policy_export"])
    require((contract["physics_dt"], contract["decimation"], contract["leg_kp"], contract["leg_kd"]) ==
            (.0025, 8, 500., 10.) and contract["full_horizon_diagnostic"] is True and
            contract["task_profile"]["contact_friction"] == .7 and contract["actuator_model"] == "dc_motor",
            "Frozen nominal deployment contract differs")
    physx = read_npz(args.physx_physics)
    actions = extract_actions(physx)
    evaluator = load_evaluator(args.source_evaluator)
    output.mkdir(parents=True)
    contract_path = output/"recorded_deployment_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, allow_nan=False)+"\n")
    calls = 0

    def replay(observation):
        nonlocal calls
        require(np.asarray(observation).shape == (51,) and np.isfinite(observation).all()
                and calls < args.max_policy_steps, "Invalid callable state or excess request")
        request = actions[calls].copy()
        calls += 1
        return request

    result = evaluator.run_evaluation(replay, args.motion, args.model, output/"mujoco", policy_path=None,
        contact_threshold_n=5., max_policy_steps=args.max_policy_steps, no_visual_mesh=True,
        contract_path=contract_path)
    modules = audit_module_origins(args.source_evaluator.resolve().parent.parent)
    require(calls == args.max_policy_steps and result["policy_steps"] == calls and
            result["physics_steps"] == calls*8 and result["full_horizon_diagnostic"] is True,
            "Replay did not complete the requested finite prefix")
    require(result["termination"] == ("motion_end" if calls == 349 else "evaluation_step_limit"),
            "Unexpected early termination")
    arrays, comparison = compare_native(physx, read_npz(output/"mujoco/rollout.npz"), calls, actions)
    np.savez_compressed(output/"same_time_comparison.npz", **arrays)
    summary = {"schema_version": 1, "kind": "same_normalized_requests_native_physx_mujoco_open_loop_diagnostic",
        "requested_prefix_completed": True, "prefix_policy_steps": calls, "prefix_duration_s": calls*.02,
        "is_full_reference": calls == 349, "physics_dt_s": .0025, "policy_dt_s": .02,
        "actor_calls": 0, "torch_imported": "torch" in sys.modules,
        "same_normalized_requests": True, "same_leg_position_and_wheel_effort_requests": True,
        "same_applied_leg_torques_claimed": False, "closed_loop_policy_transfer_claimed": False,
        "contact_cache_or_warmstart_equality_claimed": False, "root_state_corrections_after_initialization": 0,
        "contact_sensor_scope": "Existing aggregate wheel ground-normal force/count/contact-point lateral slip; no new point sensor",
        "prefix_termination_interpretation": "evaluation_step_limit means the requested prefix ended, NOT a physical failure",
        "source_inputs_sha256": INPUT_PINS, "source_files_sha256": SOURCE_PINS,
        "runtime_source_modules": modules, "adapter_sha256": sha256(__file__), **comparison,
        "artifacts_sha256": {name: sha256(output/name) for name in
            ("recorded_deployment_contract.json", "mujoco/rollout.npz", "mujoco/report.json", "same_time_comparison.npz")},
        "limitations": ["Initial observable q, orientation, joint velocities and COM velocities are checked, but solver/contact hidden states are backend-specific.",
            "Explicit PD is recalculated in each plant; identical requests do not imply identical actual torques.",
            "Equal numeric friction is not calibrated contact-law equivalence.",
            "Open-loop divergence is a diagnostic, not a controller success rate or causal isolation of contact alone."]}
    (output/"request_replay_diagnostic.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-evaluator", "physx-physics", "physx-report", "motion", "model", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-policy-steps", type=int, default=10)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("kind", "prefix_policy_steps", "prefix_duration_s",
        "native_states", "same_time_interengine_xy_rmse_m", "actor_calls", "torch_imported")}, indent=2))


if __name__ == "__main__":
    main()
