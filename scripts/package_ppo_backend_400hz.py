"""Immutable, explicitly allowlisted source packet for the new 400 Hz cohort.

The audited smoothness packager supplies regular-file/hash/tar mechanics. Its
private FILES allowlist and manifest renderer are overridden only in memory;
no historical packager, source snapshot or result is edited.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DELEGATE_SHA256 = "d75fcc2891c041072ed3331697df6fc54875f981777aac8a43e0844aa17ae5b8"
MJLAB_PIN = "4360096b49c1bcc478668c11ce7d458db747e49d"
CORE_SCRIPTS = (
    "train_ppo_backend", "train_ppo_backend_400hz", "mjlab_tron1_env", "smoke_mjlab_tron1",
    "preflight_mjlab_runtime", "mujoco_ppo_env", "mujoco_ppo_native", "physx_mjlab_recipe",
    "classical_tracking", "nmpc_tracking_objective", "nmpc_batch_objective", "height_smoothness",
    "package_ppo_backend", "package_ppo_backend_400hz", "package_ppo_smoothness", "run_ppo_backend_pair",
    "mjlab_gpu_ppo_env", "mjlab_tron1_gpu", "tron1_torch_task", "benchmark_mjlab_gpu_task",
    "profile_mjlab_gpu_task",
    # The GPU-probe loader imports this exact pure-Python chain. Including it
    # does not install a filter, read its historical actor or start a job.
    "run_ppo_smoothness_sequence", "run_ppo_leg_target_filter", "eval_ppo_leg_target_filter",
    "run_ppo_contact_damping", "eval_ppo_contact_damping", "ppo_leg_target_filter",
)
CORE_TESTS = (
    "test_train_ppo_backend", "test_train_ppo_backend_400hz", "test_mjlab_tron1_env",
    "test_physx_mjlab_recipe", "test_mjlab_runtime_preflight", "test_mujoco_ppo_env",
    "test_mujoco_ppo_native", "test_run_ppo_backend_pair", "test_mjlab_gpu_ppo_env",
    "test_mjlab_tron1_gpu", "test_tron1_torch_task", "test_benchmark_mjlab_gpu_task",
    "test_profile_mjlab_gpu_task", "test_gpu_rsl_wrapper", "test_package_ppo_backend_400hz",
)
UPSTREAM_TRAINING = (
    "drift_mujoco_residual_reward", "drift_mujoco_streak", "tron1_axle_mid_trial", "tron1_axle_weight_trial",
    "tron1_contact_slip", "tron1_controller_damping_diagnostic", "tron1_controller_gain_diagnostic",
    "tron1_domain_randomization", "tron1_drift", "tron1_drift_sideslip_trial", "tron1_ground_load_reserve",
    "tron1_ground_load_substeps", "tron1_ground_load_substeps_trial", "tron1_load_weight_trial",
    "tron1_physics_clock", "tron1_resume_lr", "tron1_runtime_device", "tron1_substep_rewards",
    "tron1_terrain", "tron1_tight_axle_tracking", "tron1_tight_axle_trial", "tron1_tracking",
    "tron1_wheel_axle_tracking", "tron1_wheel_axle_trial", "tron1_worst_ground_load_substeps",
    "tron1_worst_ground_load_trial",
)
CONFIGS = ("nominal_actuator_contract", "car_drift_height_smooth_task",
           "ppo_backend_400hz_comparison_v1", "ppo_mjlab_backend_comparison_v1")
NOTICES = ("BeyondMimic-LICENCE.txt", "LimX-TRON1-LICENSE.txt", "DOA-LICENSE.txt")


def build_file_list(root=ROOT, *, calibration_only=False):
    root = Path(root).resolve()
    training_root = root/"upstream/tron1-learning-from-human/training"
    expected_training = {name+".py" for name in UPSTREAM_TRAINING}
    actual_training = {p.name for p in training_root.glob("*.py")}
    if actual_training != expected_training:
        raise ValueError("Upstream training file set changed; audit and update the explicit allowlist")
    files = (["scripts/"+name+".py" for name in CORE_SCRIPTS] +
        ["tests/"+name+".py" for name in CORE_TESTS] + ["config/"+name+".json" for name in CONFIGS] +
        ["native/ppo_batch.cpp"] +
        ["upstream/tron1-learning-from-human/training/"+name+".py" for name in UPSTREAM_TRAINING] +
        ["upstream/tron1-learning-from-human/scripts/"+name+".py" for name in
            ("eval_tracking", "eval_tracking_mujoco", "train_tracking")] +
        ["upstream/tron1-learning-from-human/notices/"+name for name in NOTICES])
    if not calibration_only:
        files.extend(("scripts/run_ppo_backend_400hz_pair.py", "tests/test_run_ppo_backend_400hz_pair.py"))
    for name in ("LICENSE", "LICENCE", "LICENSE.md"):
        if (root/"upstream/tron1-learning-from-human"/name).is_file():
            files.append("upstream/tron1-learning-from-human/"+name)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate packet entry")
    for name in files:
        path = root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing regular allowlisted source file: " + name)
    return tuple(files)


def _load_delegate():
    import hashlib
    path = ROOT/"scripts/package_ppo_smoothness.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DELEGATE_SHA256:
        raise ValueError("Audited package helper source differs")
    spec = importlib.util.spec_from_file_location("_private_backend400_packager", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def package(output, root=ROOT, *, calibration_only=False):
    helper = _load_delegate()
    helper.FILES = build_file_list(root, calibration_only=calibration_only)

    def render_manifest(value, *args, **kwargs):
        # A private json facade changes only the new manifest metadata, not the
        # delegated hash/tar/no-overwrite implementation or global json module.
        if isinstance(value, dict) and value.get("kind") == "ppo_smoothness_pure_source_snapshot":
            value = {**value, "kind": "matched_ppo_backend_400hz_pure_source_snapshot",
                "contents": "explicit Python/config/tests, uncompiled C++ source, and existing license notices only",
                "packet_scope": "calibration_without_dispatcher" if calibration_only else "training_and_calibration",
                "delegate_packager_sha256": DELEGATE_SHA256,
                "upstream_mjlab_commit": MJLAB_PIN,
                "external_mjlab_source_distribution": "existing separately provisioned pinned submodule; not bundled",
                "upstream_root_license_file_present": any("/"+name in item for name in ("LICENSE", "LICENCE", "LICENSE.md")
                    for item in helper.FILES if item.startswith("upstream/tron1-learning-from-human/") and "/notices/" not in item)}
        return json.dumps(value, *args, **kwargs)

    helper.json = SimpleNamespace(dumps=render_manifest)
    result = helper.package(output, root=root)
    return {**result, "packet_scope": "calibration_without_dispatcher" if calibration_only else "training_and_calibration"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-only", action="store_true",
                        help="Omit the not-yet-frozen new dispatcher; create a distinct immutable parity packet")
    args = parser.parse_args()
    print(json.dumps(package(args.output, calibration_only=args.calibration_only), indent=2))
