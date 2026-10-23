"""Strict nominal full-horizon evaluation of the NEW any-wheel-contact PPO.

Only a completed, source-bound normalized TorchScript export is executed. No
checkpoint pickle is loaded, no inference contact reward/target is introduced,
and one-wheel support is permitted rather than relabeled as bilateral support.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import sys

import ground_contact_reward_v2 as definition
import ppo_backend_400hz_contract as common
import eval_ppo_backend_400hz_mujoco as native


ROOT = Path(__file__).resolve().parents[1]
KIND = "ground_contact_v2_any_wheel_ppo_evaluation"
REWARD_VERSION = "ground_contact_v2_any_wheel"
COHORT_ID = "tron1_ground_contact_v2_any_wheel_400hz_v1"
NATIVE_HELPER_SHA256 = "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4"
COMMON_HELPER_SHA256 = "b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e"
PHYSX_HELPER_SHA256 = "e670d6c88bd85eb8aa36f402301e7f0c9ae5f7d3b97336a78d1a66750a642f99"
require, sha256, read_json, write_json = common.require, common.sha256, common.read_json, common.write_json


def verify_training(training_dir, protocol_path, *, allow_smoke=False):
    """Verify NEW training evidence; never pass it through the old cohort gate."""
    directory, protocol_path = Path(training_dir).resolve(), Path(protocol_path).resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("cohort_id") == COHORT_ID, "Wrong any-wheel contact cohort")
    require(protocol.get("reward_definition_version") == REWARD_VERSION, "Missing new protocol reward identity")
    for key, expected in {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
                          "action_rate_weight": -.5, "grounded_reward_weight": .4,
                          "shared_dr_recipe": "tron1_mjlab_tracking_dr_v1"}.items():
        require(protocol.get(key) == expected, "Ground-contact protocol field differs: " + key)
    require(isinstance(protocol.get("ground_contact_reward_v2"), dict), "Missing explicit any-wheel reward settings")
    settings = definition.validate_settings(protocol["ground_contact_reward_v2"])
    require(protocol["algorithm"]["num_envs"] == 512 and protocol["algorithm"]["iterations"] == 1000
            and protocol["algorithm"]["steps_per_env"] == 24
            and protocol["algorithm"]["transitions"] == 12288000
            and protocol["algorithm"]["seed"] == 42
            and protocol["smoke"]["num_envs"] == 32 and protocol["smoke"]["iterations"] == 2
            and protocol["smoke"]["transitions"] == 1536 and protocol["smoke"]["setup_smoke_steps"] == 8,
            "Declared PPO collection budget differs")
    names = ("ground_contact_v2_training_audit.json", "cohort_completion_audit.json", "manifest.json",
             "policy_contract.json", "final_policy_contract.json", "run_status.json", "policy.pt", "model_final.pt")
    paths = {name: directory / name for name in names}
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()),
            "Missing or indirect completed training artifact")
    hashes = {name: sha256(path) for name, path in paths.items()}
    audit, completion, manifest, contract, status = [read_json(paths[name]) for name in
        ("ground_contact_v2_training_audit.json", "cohort_completion_audit.json", "manifest.json",
         "policy_contract.json", "run_status.json")]
    require(paths["policy_contract.json"].read_bytes() == paths["final_policy_contract.json"].read_bytes(),
            "Canonical and compatibility policy contracts differ")
    smoke = contract.get("smoke_only") is True
    require(not smoke or allow_smoke, "Smoke policy requires explicit --allow-smoke-policy")
    mode, budget = ("smoke", 1536) if smoke else ("formal", 12288000)
    for item in (audit, completion, manifest, contract, status):
        require(item.get("smoke_only") is smoke and item.get("trained_policy_claimed") is (not smoke),
                "Smoke/formal policy identity differs")
    require(audit.get("status") == completion.get("status") == "completed_verified"
            and status.get("status") == "training_completed", "Training completion is not verified")
    require(audit.get("reward_definition_version") == REWARD_VERSION
            and audit.get("normalizer_in_actor") is True and audit.get("from_scratch") is True,
            "New reward/export initialization evidence differs")
    require(audit.get("other14_reward_terms_changed") is False and audit.get("new_reward_count") == 15,
            "The experiment must replace only the existing grounded reward")
    for item in (audit, completion):
        require(item.get("cohort_id") == COHORT_ID and item.get("protocol_sha256") == sha256(protocol_path)
                and item.get("training_environment_steps") == budget,
                "Training protocol identity or exact transition budget differs")
    require(status.get("training_environment_steps") == budget and manifest.get("transitions_requested") == budget
            and manifest.get("num_envs") == protocol["smoke" if smoke else "algorithm"]["num_envs"]
            and manifest.get("iterations") == protocol["smoke" if smoke else "algorithm"]["iterations"]
            and manifest.get("backend") == contract.get("policy_training_backend") == "physx",
            "Actual training collection/backend differs")
    require(contract.get("run_kind") == status.get("run_kind") == mode
            and manifest.get("seed") == 42 and manifest.get("versions", {}).get("rsl-rl-lib") == "3.0.1",
            "PPO mode/seed/implementation differs")
    for item in (audit, completion, contract, status):
        require(item.get("checkpoint_sha256") == hashes["model_final.pt"]
                and item.get("exported_policy_sha256") == hashes["policy.pt"], "Checkpoint/normalized actor hash differs")
    for key, name in (("manifest_sha256", "manifest.json"), ("policy_contract_sha256", "policy_contract.json")):
        require(audit.get(key) == completion.get(key) == hashes[name], "Completed metadata hash differs: " + name)
    require(paths["model_final.pt"].stat().st_size > 0 and paths["policy.pt"].stat().st_size > 0,
            "Empty actor/checkpoint artifact")
    initial = audit.get("initial_policy_state_sha256")
    require(isinstance(initial, str) and len(initial) == 64 and all(c in "0123456789abcdef" for c in initial)
            and initial == manifest.get("initial_policy_state_sha256") == status.get("initial_policy_state_sha256")
            == completion.get("initial_policy_state_sha256"), "Initial policy identity differs")
    ledger = audit.get("source_sha256")
    require(isinstance(ledger, dict) and len(ledger) >= 3, "Missing exact training source ledger")
    for filename, digest in ledger.items():
        path = Path(filename)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and sha256(path) == digest, "Training source ledger differs: " + filename)
    for basename in ("ground_contact_reward_v2.py", "ppo_ground_contact_v2.py", "train_ppo_ground_contact_v2.py"):
        require(any(Path(name).name == basename for name in ledger), "Missing new reward/trainer source: " + basename)
    require(any(Path(name).name == "ground_contact_reward_v2.py" and digest == sha256(definition.__file__)
                for name, digest in ledger.items()), "Evaluated reward definition source differs from training")
    profile = contract.get("trained_task_profile", {})
    require(profile == contract.get("task_profile") == manifest.get("task_profile")
            and profile.get("ground_contact_reward_v2") == settings
            and profile.get("contact_friction") == 1. and profile.get("action_rate_weight") == -.5
            and profile.get("reward_weights", {}).get("grounded") == .4
            and profile.get("reward_weights", {}).get("sideslip") == 2.
            and profile.get("height_regularization", {}).get("mode") == "vertical_acceleration_only",
            "Trained reward profile differs from the declared any-wheel experiment")
    expected = {"contract_status": "smoke_completed_normalized_actor_not_formal_policy" if smoke else "trained_completed_normalized_actor",
        "policy_algorithm": "PPO", "physics_dt": .0025, "decimation": 8, "policy_fps": 50.,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8, "leg_kp": 500., "leg_kd": 10.,
        "actor_includes_observation_normalizer": True, "actor_output_is_unclipped": True,
        "actor_observation_groups": ["policy"], "training_environment_transitions": budget,
        "reference_file_sha256": common.INPUT_PINS["motion"], "actuator_model": "dc_motor",
        "raw_action_clip": [-1., 1.], "leg_position_scale": 1., "wheel_torque_scale_nm": 12.,
        "leg_torque_limit_nm": 80., "wheel_torque_limit_nm": 12.,
        "leg_saturation_effort_nm": 80., "wheel_saturation_effort_nm": 12.,
        "leg_motor_velocity_limit_rad_s": 15., "wheel_motor_velocity_limit_rad_s": 100.,
        "solver_joint_velocity_limit_rad_s": 1000.}
    for key, value in expected.items():
        require(contract.get(key) == value, "New policy contract differs: " + key)
    require(contract.get("training_domain_randomization") == protocol["training_randomization"]
            and "evaluation_contact_friction_override" not in contract and manifest.get("contract") == contract,
            "Trained DR/contract changed or contains stale evaluation overrides")
    for name in ("motion", "model", "asset"):
        require(manifest.get("inputs", {}).get(name, {}).get("sha256") == common.INPUT_PINS[name],
                "Pinned training asset/reference differs: " + name)
    error = contract.get("export_validation_max_abs_error")
    require(isinstance(error, (int, float)) and not isinstance(error, bool) and math.isfinite(error)
            and 0 <= error <= 1.e-4, "Normalized export parity evidence missing")
    runtime = audit["contact_runtime_audit"]
    windows = (2 if smoke else 1000) * 24 + 8
    require(runtime.get("reward_definition_version") == REWARD_VERSION
            and runtime.get("settings") == settings and runtime.get("physics_dt_s") == .0025
            and runtime.get("policy_dt_s") == .02 and runtime.get("substeps") == 8
            and runtime.get("steps_started") == runtime.get("steps_completed") == windows
            and runtime.get("unique_reward_windows") == windows and runtime.get("physics_samples") == windows * 8
            and runtime.get("device_finite_normal_force_health") is True
            and runtime.get("device_finite_reward_health") is True
            and runtime.get("cpu_fixture_only") is False
            and runtime.get("single_wheel_support_allowed") is True and runtime.get("bilateral_constraint") is False
            and runtime.get("height_or_zero_slip_target") is False
            and runtime.get("observation_action_or_physics_change") is False,
            "Actual eight-substep any-wheel contact reward runtime proof differs")
    require(runtime.get("definition_sha256") == sha256(definition.__file__)
            and any(Path(name).name == "ppo_ground_contact_v2.py" and digest == runtime.get("module_sha256")
                    for name, digest in ledger.items()), "Actual collector/definition source readback differs")
    bearing = runtime.get("force_bearing_samples_L_R_any_both")
    require(isinstance(bearing, list) and len(bearing) == 4 and all(type(n) is int and n >= 0 for n in bearing)
            and bearing[2] > 0 and bearing[3] <= min(bearing[0], bearing[1])
            and bearing[2] + bearing[3] == bearing[0] + bearing[1], "Missing/inconsistent actual force-bearing observations")
    return {"directory": directory, "protocol_path": protocol_path, "protocol": protocol, "contract": contract,
            "completion": completion, "manifest": manifest, "paths": paths, "input_sha256": hashes,
            "training_backend": "physx", "smoke_only": smoke, "checkpoint": paths["model_final.pt"],
            "actor": paths["policy.pt"], "training_audit": audit}


def provenance(cohort):
    return {"kind": KIND, "cohort_id": COHORT_ID, "reward_definition_version": REWARD_VERSION,
        "training_backend": "physx", "smoke_only": cohort["smoke_only"],
        "trained_policy_claimed": not cohort["smoke_only"], "additional_training_updates": 0,
        "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
        "normalized_actor_sha256": cohort["input_sha256"]["policy.pt"],
        "protocol_sha256": sha256(cohort["protocol_path"]), "cohort_input_sha256": cohort["input_sha256"],
        "normalizer_included_in_actor": True, "normalization_applications_per_inference": 1,
        "extra_normalization": False, "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "single_wheel_support_allowed": True, "bilateral_support_required": False,
        "support_count_definition": "recorded legacy count includes contacts exceeding 5 N; not raw geometric point count",
        "absolute_height_gate": False, "airborne_failure": False, "hardware_ready": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("training-dir", "protocol", "legacy-evaluator", "motion-file", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--model", type=Path)
    result.add_argument("--asset-path", type=Path)
    result.add_argument("--backend", choices=("mujoco", "physx"), default="mujoco")
    result.add_argument("--mu", type=float, choices=(.7,), default=.7)
    result.add_argument("--physical-gpu-index", type=int, choices=(3, 4, 5))
    result.add_argument("--allow-smoke-policy", action="store_true")
    return result


def prepare_plan(argv):
    args = parser().parse_args(argv)
    require(sha256(common.__file__) == COMMON_HELPER_SHA256 and sha256(native.__file__) == NATIVE_HELPER_SHA256,
            "Pinned common/native inference helper changed")
    cohort = verify_training(args.training_dir, args.protocol, allow_smoke=args.allow_smoke_policy)
    if args.backend == "physx":
        import eval_ppo_backend_400hz_isaac as physx
        require(sha256(physx.__file__) == PHYSX_HELPER_SHA256, "Pinned PhysX inference helper changed")
        require(args.asset_path is not None and args.physical_gpu_index is not None, "PhysX needs asset and explicit approved GPU")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") == "PCI_BUS_ID",
                "PhysX requires unmasked PCI_BUS_ID GPU ordering")
        evaluator, legacy_root = common.verify_legacy(args.legacy_evaluator, physx.BASE_SHA256)
        plan = common.prepare_inputs(cohort, args.motion_file, args.output_dir, args.mu, "PhysX", asset=args.asset_path)
        from train_ppo_backend import ppo_config
        config = read_json(cohort["directory"] / "runner_config.json")
        require(config == ppo_config(42, config["save_interval"]), "PPO runner configuration differs")
        recipe = ROOT / "scripts/physx_mjlab_recipe.py"
        require(sha256(recipe) == physx.RECIPE_SHA256, "Frozen PhysX material recipe changed")
        index = args.physical_gpu_index
        device = f"cuda:{index}"
        kit = (f"--/renderer/activeGpu={index} --/renderer/multiGpu/enabled=false "
               f"--/physics/cudaDevice={index} --/physics/sceneMultiGPUMode=0")
        plan.update(requested_device=device, recipe_sha256=sha256(recipe),
                    runner_config_sha256=sha256(cohort["directory"] / "runner_config.json"))
        plan["inherited_argv"] = ["--motion-file", str(Path(args.motion_file).resolve()),
            "--asset-path", str(Path(args.asset_path).resolve()), "--checkpoint", str(cohort["checkpoint"]),
            "--runner-config", str(cohort["directory"] / "runner_config.json"), "--output-dir", str(plan["output"]),
            "--task-profile", str(plan["task_profile_path"]), "--evaluation-contact-friction", str(args.mu),
            "--num-envs", "1", "--seed", "42", "--record-physics", "--verify-unmasked-device",
            "--device", device, "--headless", "--kit_args", kit]
    else:
        require(args.model is not None, "Native MuJoCo needs --model")
        evaluator, legacy_root = common.verify_legacy(args.legacy_evaluator, native.BASE_SHA256)
        plan = common.prepare_inputs(cohort, args.motion_file, args.output_dir, args.mu, "native CPU MuJoCo", model=args.model)
    plan.update(args=args, evaluator=evaluator, legacy_root=legacy_root, wrapper_sha256=sha256(__file__))
    return plan


def _finish(report, arrays, plan, *, contract=None):
    backend = "PhysX" if plan["args"].backend == "physx" else "native CPU MuJoCo"
    recording = plan["output"] / ("physics_trajectory.npz" if backend == "PhysX" else "rollout.npz")
    common.verify_recording(recording, backend)
    if backend == "native CPU MuJoCo":
        require(report["status"] == "completed" and report["completed_full_reference"] is True
                and report["policy_steps"] == 349 and report["physics_steps"] == 2792
                and report["hidden_resets"] == 0 and report["root_state_writes"] == 1
                and report["start_reference_frame"] == 0 and report["domain_parameters_resampled_during_episode"] is False,
                "Native full-horizon lifecycle differs")
        require(report["policy_sha256"] == plan["cohort"]["input_sha256"]["policy.pt"], "Deployed normalized actor differs")
        deployed = report["contract"]
    else:
        require(report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
                and report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False,
                "PhysX deterministic lifecycle/DR differs")
        require(report["runtime_device_audit"]["status"] == "verified", "PhysX native device not verified")
        require(report["runtime_device_audit"]["requested_device"] == plan["requested_device"],
                "PhysX selected device differs")
        deployed = contract
    for name, path in plan["cohort"]["paths"].items():
        require(sha256(path) == plan["cohort"]["input_sha256"][name], "Training artifact changed during evaluation: " + name)
    for key in ("physics_dt", "decimation", "trained_task_profile", "task_profile", "full_horizon_diagnostic",
                "training_domain_randomization", "evaluation_contact_friction_override"):
        require(deployed[key] == plan["contract"][key], "Runtime inference contract differs: " + key)
    audit = provenance(plan["cohort"])
    audit.update(status="verified", validation_backend=backend, wrapper_sha256=plan["wrapper_sha256"],
        legacy_evaluator_sha256=sha256(plan["evaluator"]), generated_source_sha256=plan["generated_sha256"],
        recording_sha256=sha256(recording), native_states=2793, policy_decisions=349, duration_s=6.98)
    if backend == "PhysX":
        audit.update(physical_gpu_index_requested=plan["args"].physical_gpu_index,
                     deployment_actor_sha256=sha256(plan["output"] / "actor_normalized.pt"))
    report["ground_contact_v2_evaluation"] = audit
    deployed["ground_contact_v2_evaluation"] = copy.deepcopy(audit)
    write_json(plan["input_dir"] / "evaluation_source_audit.json", audit)


def validate_output(folder, training_dir, protocol, *, backend="mujoco", allow_smoke=False):
    """Parent-side proof: actual native records, no reset-spliced completion."""
    folder = Path(folder)
    cohort = verify_training(training_dir, protocol, allow_smoke=allow_smoke)
    report = read_json(folder / "report.json")
    audit = report["ground_contact_v2_evaluation"]
    for key, value in provenance(cohort).items():
        require(audit.get(key) == value, "Evaluation training provenance differs: " + key)
    require(audit.get("status") == "verified" and audit.get("wrapper_sha256") == sha256(__file__),
            "Completed evaluator source/status differs")
    native_backend = "native CPU MuJoCo" if backend == "mujoco" else "PhysX"
    require(backend in ("mujoco", "physx") and audit["validation_backend"] == native_backend,
            "Wrong validation physics backend")
    recording = folder / ("rollout.npz" if backend == "mujoco" else "physics_trajectory.npz")
    proof = common.verify_recording(recording, native_backend)
    require(audit["recording_sha256"] == sha256(recording), "Completed physical recording hash differs")
    if backend == "mujoco":
        require(report["status"] == "completed" and report["completed_full_reference"] is True
                and report["hidden_resets"] == 0 and report["root_state_writes"] == 1
                and report["policy_steps"] == 349 and report["physics_steps"] == 2792
                and report["start_reference_frame"] == 0
                and report["domain_parameters_resampled_during_episode"] is False
                and report["policy_sha256"] == cohort["input_sha256"]["policy.pt"], "Native lifecycle/actor differs")
        contract = report["contract"]
    else:
        require(report["status"] == "evaluated" and report["domain_randomization_enabled"] is False
                and report["observation_noise_enabled"] is False and report["deterministic_start_frame"] == 0
                and report["runtime_device_audit"]["status"] == "verified"
                and report["runtime_device_audit"]["requested_device"] == f"cuda:{audit['physical_gpu_index_requested']}",
                "PhysX lifecycle/device differs")
        require(sha256(folder / "actor_normalized.pt") == audit["deployment_actor_sha256"], "Deployed PhysX actor differs")
        contract = report["policy_export"]
    expected = common.evaluation_contract(cohort, .7, native_backend)
    for key in ("physics_dt", "decimation", "task_profile", "trained_task_profile", "full_horizon_diagnostic",
                "training_domain_randomization", "evaluation_contact_friction_override"):
        require(contract[key] == expected[key], "Completed inference contract differs: " + key)
    material = report["contact_material_audit"]
    if backend == "mujoco":
        require(material["native_mapping_verified"] is True and material["explicit_contact_pair_count"] == 0
                and material["pair_coefficient"] == .7
                and all(abs(row[0] - .7) < 1.e-12 for row in material["geom_friction"]), "Native material mapping differs")
        priorities = dict(zip(material["colliding_geom_ids"], material["geom_priority"]))
        require(all(priority == (1 if index in material["wheel_geom_ids"] else 0)
                    for index, priority in priorities.items()), "Native wheel material priorities differ")
    else:
        require(material["live_physx_materials_verified"] is True and material["ground_material_usd_verified"] is True
                and material["recipe"]["training"] is False
                and material["recipe"]["wheel_contact_friction"]["range"] == [.7, .7]
                and material["recipe"]["wheel_contact_friction"]["physx_ground_friction"] == 1.
                and material["encoder_bias_range_rad"] == [0., 0.], "PhysX material/DR mapping differs")
    return {**proof, "validation_backend": native_backend, "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
            "report_sha256": sha256(folder / "report.json"), "evaluation_mu": .7,
            "support_requirement": "at least one wheel; bilateral support is not required"}


def run(argv):
    plan = prepare_plan(argv)
    if plan["args"].backend == "mujoco":
        generated = native.materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
        namespace = {"__name__": "_ground_contact_v2_native_eval", "__file__": str(Path(__file__).resolve()),
            "_new_plan": plan, "_new_audit_materials": native.audit_materials, "_new_finish_report": _finish}
        plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
        previous = list(sys.path)
        try:
            sys.path[:0] = [str(plan["evaluator"].parent), str(plan["legacy_root"])]
            exec(compile(generated, "<ground_contact_v2_native_eval>", "exec"), namespace)
            policy = namespace["load_torchscript"](plan["cohort"]["actor"])
            return namespace["run_evaluation"](policy, plan["motion"], plan["args"].model, plan["output"],
                policy_path=plan["cohort"]["actor"], no_visual_mesh=True, contract_path=plan["contract_path"])
        finally:
            sys.path[:] = previous
    import eval_ppo_backend_400hz_isaac as physx
    generated = physx.materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()

    def runner(cohort, device):
        result = common.StrictPPOExportRunner(cohort, device)
        result.inference_provenance = provenance(cohort)
        return result

    namespace = {"__name__": "_ground_contact_v2_physx_eval", "__file__": str(Path(__file__).resolve()),
        "_new_plan": plan, "_new_configure": physx.configure, "_new_audit_materials": physx.audit_materials,
        "_new_runner": runner,
        "_new_finish_report": lambda report, contract, plan, raw: _finish(report, raw, plan, contract=contract)}
    previous = sys.argv
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<ground_contact_v2_physx_eval>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    run(sys.argv[1:])
