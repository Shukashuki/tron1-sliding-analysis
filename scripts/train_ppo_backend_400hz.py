"""Strict new 400 Hz / action-rate -0.5 cohort around the frozen backend trainer.

The original trainer, PPO implementation, seed-after-setup ordering, original
DR recipe, and previous experiments are not rewritten. Only counted in-memory
integration anchors and fresh, declared inputs are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = "65367115de9b5608dc2ef57cab701bea31c152c6af576bdb43a5cef025cc6878"
NOMINAL_SHA256 = "6e105cbd6efe7d35117a895fa097e8ec64b6ac822a9de07c00acfaef4d09642f"
PROFILE_SHA256 = "31de29a61eec847e0e39783f3a0d9d8f2db551acd679a1b38dfde2b24e8ae7d6"
DEVICE_AUDIT_SHA256 = "2a1460d1b48e884fc5608e0365edc43b815d1fc74c8439945d4e2cc18027f775"
MJLAB_PIN = "4360096b49c1bcc478668c11ce7d458db747e49d"
INPUT_PINS = {"motion": "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254",
              "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"}
SOURCE_FILES = ("scripts/train_ppo_backend.py", "scripts/train_ppo_backend_400hz.py",
    "scripts/mujoco_ppo_env.py", "scripts/mjlab_tron1_env.py", "scripts/mjlab_gpu_ppo_env.py",
    "scripts/mjlab_tron1_gpu.py", "scripts/tron1_torch_task.py", "scripts/physx_mjlab_recipe.py",
    "scripts/height_smoothness.py", "scripts/nmpc_batch_objective.py", "scripts/nmpc_tracking_objective.py",
    "scripts/mujoco_ppo_native.py", "scripts/classical_tracking.py")
STALE_FIELDS = ("checkpoint_sha256", "exported_policy_sha256", "policy_sha256", "actor_file",
    "actor_includes_observation_normalizer", "actor_output_is_unclipped", "actor_observation_groups",
    "export_validation_max_abs_error", "actual_joint_order", "actual_body_order", "trained_task_profile",
    "evaluation_contact_friction_override", "training_domain_randomization", "full_horizon_diagnostic",
    "physics_clock_refinement", "domain_parameters")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def validate_protocol(protocol):
    require(isinstance(protocol, dict) and protocol.get("schema_version") == 1 and
            isinstance(protocol.get("cohort_id"), str) and protocol["cohort_id"], "Invalid cohort protocol")
    expected = {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "policy_fps": 50,
        "action_rate_weight": -.5, "grounded_reward_weight": .4, "base_trainer_sha256": BASE_SHA256,
        "shared_dr_recipe": "tron1_mjlab_tracking_dr_v1", "mjlab_commit": MJLAB_PIN,
        "mjlab_task_implementation": "gpu"}
    for name, value in expected.items():
        require(protocol.get(name) == value, "Declared cohort field differs: " + name)
    algorithm = protocol.get("algorithm", {})
    for name, value in {"implementation": "rsl-rl-lib", "version": "3.0.1", "seed": 42,
            "num_envs": 512, "steps_per_env": 24, "iterations": 1000, "transitions": 12288000}.items():
        require(algorithm.get(name) == value, "Algorithm field differs: " + name)
    for name, value in {"num_envs": 32, "iterations": 2, "transitions": 1536, "setup_smoke_steps": 8}.items():
        require(protocol.get("smoke", {}).get(name) == value, "Smoke field differs: " + name)
    require(isinstance(protocol.get("training_randomization"), dict), "Shared DR declaration missing")
    dr = protocol["training_randomization"]
    for name, value in {"range": [.3, 1.2], "distribution": "uniform_continuous",
            "sampling": "startup_once_per_environment", "shared_between_wheels": True,
            "scope": "colliding_wheel_shapes", "mujoco_wheel_priority": 1,
            "physx_static_equals_dynamic": True, "physx_floor_mu": 1., "physx_combine": "multiply"}.items():
        require(dr.get("sliding_friction", {}).get(name) == value, "Shared friction declaration differs: " + name)
    require(dr.get("encoder_position_bias_rad") == [-.01, .01] and
            dr.get("encoder_bias_signs") == {"actor": "+b", "position_target": "-b", "critic": "unbiased"} and
            dr.get("base_com_xyz_range_m") == [[-.025, .025], [-.05, .05], [-.05, .05]] and
            dr.get("root_link_velocity_push_interval_s") == [1., 3.] and
            dr.get("restitution_randomization") is False, "Shared bias/COM/push/restitution declaration differs")
    return protocol


def derive_inputs(nominal, profile, protocol):
    """Remove historical actor/evaluation identities, retaining the motor/obs ABI."""
    validate_protocol(protocol)
    task = copy.deepcopy(profile)
    require(task["contact_friction"] == 1. and task["reward_weights"]["grounded"] == .4 and
            task["height_regularization"]["mode"] == "vertical_acceleration_only", "Original task differs")
    task["action_rate_weight"] = -.5
    task["description"] = ("New matched-backend native400Hz training cohort; action-rate weight -0.5, "
        "original last-substep grounded reward +0.4, free absolute height, shared mjlab-style DR. "
        "Training is from scratch; no historical checkpoint initializes either arm.")
    contract = copy.deepcopy(nominal)
    for name in STALE_FIELDS:
        contract.pop(name, None)
    require((contract["actor_dim"], contract["critic_dim"], contract["action_dim"],
        contract["leg_kp"], contract["leg_kd"]) == (51, 120, 8, 500., 10.), "Motor/observation ABI differs")
    contract.update(physics_dt=.0025, decimation=8, policy_fps=50., task_profile=copy.deepcopy(task),
        trained_task_profile=copy.deepcopy(task), training_domain_randomization=copy.deepcopy(protocol["training_randomization"]),
        domain_randomization=protocol["shared_dr_recipe"], cohort_id=protocol["cohort_id"],
        contract_status="declared_untrained_interface_not_an_exported_actor",
        physics_clock_origin="trained_at_native400Hz_not_evaluation_only_refinement")
    return contract, task


def prepare_declared_inputs(protocol_path, output_dir, *,
        nominal_path=ROOT/"config/nominal_actuator_contract.json",
        profile_path=ROOT/"config/car_drift_height_smooth_task.json"):
    """CPU-only, deterministic inputs shared by parity and every training arm."""
    protocol_path, output_dir = Path(protocol_path).resolve(), Path(output_dir).resolve()
    nominal_path, profile_path = Path(nominal_path), Path(profile_path)
    require(not output_dir.exists(), "Declared inputs need a fresh directory")
    require(sha256(nominal_path) == NOMINAL_SHA256 and sha256(profile_path) == PROFILE_SHA256,
            "Original nominal contract/task source differs")
    protocol = validate_protocol(json.loads(protocol_path.read_text()))
    contract, task = derive_inputs(json.loads(nominal_path.read_text()), json.loads(profile_path.read_text()), protocol)
    output_dir.mkdir(parents=True)
    write_json(output_dir/"contract.json", contract)
    write_json(output_dir/"task_profile.json", task)
    original = {"contract": NOMINAL_SHA256, "task_profile": PROFILE_SHA256}
    derived = {"contract": sha256(output_dir/"contract.json"), "task_profile": sha256(output_dir/"task_profile.json")}
    write_json(output_dir/"derivation.json", {"cohort_id": protocol["cohort_id"], "protocol_sha256": sha256(protocol_path),
        "source_inputs_sha256": original, "derived_inputs_sha256": derived,
        "stale_fields_removed": list(STALE_FIELDS), "old_source_files_modified": False,
        "historical_checkpoint_used": False, "backend_and_run_kind_independent_input_bytes": True})
    return {"contract": output_dir/"contract.json", "task_profile": output_dir/"task_profile.json",
        "protocol_sha256": sha256(protocol_path), "original_input_sha256": original,
        "derived_input_sha256": derived}


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--cohort-protocol", type=Path, required=True)
    parser.add_argument("--cohort-run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--declared-input-dir", type=Path)
    parser.add_argument("--isaac-device-audit-source", type=Path)
    own, inherited = parser.parse_known_args(argv)
    protocol_path = own.cohort_protocol.resolve()
    protocol = validate_protocol(json.loads(protocol_path.read_text()))
    base = ROOT/"scripts/train_ppo_backend.py"
    require(sha256(base) == BASE_SHA256, "Frozen base trainer differs")
    sources = {name: sha256(ROOT/name) for name in SOURCE_FILES}
    if own.isaac_device_audit_source is not None:
        own.isaac_device_audit_source = own.isaac_device_audit_source.resolve()
        require(sha256(own.isaac_device_audit_source) == DEVICE_AUDIT_SHA256, "Device audit source differs")
    return {"protocol": protocol, "protocol_path": protocol_path, "protocol_sha256": sha256(protocol_path),
        "run_kind": own.cohort_run_kind, "declared_input_dir": own.declared_input_dir,
        "device_audit_source": own.isaac_device_audit_source, "source_hashes": sources,
        "inherited_argv": inherited, "base": base, "wrapper_sha256": sha256(__file__)}


def before_start(args, plan):
    protocol, mode = plan["protocol"], plan["run_kind"]
    declaration = protocol["algorithm"] if mode == "formal" else protocol["smoke"]
    require((args.num_envs, args.iterations, args.seed, args.smoke_steps) ==
        (declaration["num_envs"], declaration["iterations"], 42, 8), "CLI collection/setup differs from declared mode")
    require(args.backend in ("physx", "mjlab") and args.mjlab_task_implementation == "gpu",
            "Both cohort arms require the original GPU task implementation")
    require(re.fullmatch(r"cuda:(0|[1-9][0-9]*)", args.device) is not None and
            os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Explicit unmasked CUDA device required")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    require(os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID", "Use unmasked PCI_BUS_ID CUDA order")
    if args.backend == "physx":
        require(plan["device_audit_source"] is not None, "PhysX requires the pinned native device audit source")
    require(sha256(args.contract) == NOMINAL_SHA256 and sha256(args.task_profile) == PROFILE_SHA256,
            "Original nominal contract/task source differs")
    for name, expected in INPUT_PINS.items():
        require(sha256(getattr(args, name)) == expected, "Pinned input differs: " + name)
    require(not args.output_dir.exists(), "Preserve old output: use a fresh directory")
    declared = (plan["declared_input_dir"] or args.output_dir.with_name(args.output_dir.name+"_declared_inputs")).resolve()
    output = args.output_dir.resolve()
    require(not declared.exists() and not declared.is_relative_to(output) and not output.is_relative_to(declared),
            "Declared inputs and output must be fresh, separate nonnested directories")
    prepared = prepare_declared_inputs(plan["protocol_path"], declared,
        nominal_path=args.contract, profile_path=args.task_profile)
    plan.update(expected_transitions=declaration["transitions"], declared_input_dir=declared,
        original_input_sha256=prepared["original_input_sha256"], derived_input_sha256=prepared["derived_input_sha256"])
    args.contract, args.task_profile = prepared["contract"], prepared["task_profile"]


def launch_app(AppLauncher, args, plan):
    index = int(args.device.partition(":")[2])
    kit = (f"--/renderer/activeGpu={index} --/renderer/multiGpu/enabled=false "
           f"--/physics/cudaDevice={index} --/physics/sceneMultiGPUMode=0")
    plan["kit_args"] = kit
    return AppLauncher(headless=True, device=args.device, kit_args=kit).app


def set_torch_device(torch, args, plan):
    torch.cuda.set_device(args.device)
    require(torch.cuda.current_device() == int(args.device.partition(":")[2]), "Torch device ordinal differs")


def configure_physx(cfg, args, plan, apply_recipe):
    apply_recipe(cfg)
    require((cfg.sim.dt, cfg.decimation) == (.005, 4), "Original PhysX task clock differs before cohort configuration")
    cfg.sim.dt, cfg.decimation = .0025, 8
    cfg.sim.render_interval = 8
    cfg.scene.contact_forces.update_period = .0025
    cfg.rewards.action_rate_l2.weight = -.5
    require(cfg.rewards.drift_grounded.weight == .4, "Original grounded reward changed")


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _all_equal(value, expected):
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_all_equal(item, expected) for item in value)
    return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) == expected


def validate_runtime_fields(values):
    for name, expected in {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
            "actor_dim": 51, "critic_dim": 120, "action_dim": 8,
            "action_rate_weight": -.5, "grounded_reward_weight": .4, "reward_term_count": 15}.items():
        require(values.get(name) == expected, "Runtime cohort field differs: " + name)
    require(_all_equal(values["leg_kp"], 500.) and _all_equal(values["leg_kd"], 10.), "Runtime explicit leg PD differs")
    require(values["absolute_height_termination"] is False and values["absolute_xy_termination"] is False,
            "Absolute height/XY training termination must remain absent")
    return values


def runtime_audit(env, args, metadata, plan):
    import torch
    require(torch.cuda.current_device() == int(args.device.partition(":")[2]), "Current Torch device changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "A CUDA mask appeared during startup")
    if args.backend == "physx":
        cfg = env.cfg
        weights = {name: float(env.reward_manager.get_term_cfg(name).weight) for name in env.reward_manager.active_terms}
        legs = env.scene["robot"].actuators["legs"]
        values = {"physics_dt_s": float(env.sim.get_physics_dt()), "decimation": cfg.decimation,
            "policy_dt_s": float(env.step_dt), "actor_dim": 51, "critic_dim": 120,
            "action_dim": env.action_manager.total_action_dim,
            "action_rate_weight": weights["action_rate_l2"], "grounded_reward_weight": weights["drift_grounded"],
            "reward_term_count": len(weights), "leg_kp": _plain(legs.stiffness), "leg_kd": _plain(legs.damping),
            "absolute_height_termination": cfg.terminations.anchor_pos is not None or cfg.terminations.ee_body_pos is not None,
            "absolute_xy_termination": cfg.terminations.anchor_pos is not None}
        require(float(env.physics_dt) == .0025 and cfg.sim.render_interval == 8 and
                float(env.scene.sensors["contact_forces"].cfg.update_period) == .0025,
                "Actual PhysX/sensor/render clocks differ")
        path = plan["device_audit_source"]
        require(sha256(path) == DEVICE_AUDIT_SHA256, "Device audit source changed")
        spec = importlib.util.spec_from_file_location("_cohort_native_device_audit", path)
        audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        device = audit.validate_simulation_device(env, requested_device=args.device, required_unmasked=True)
        recipe = metadata["recipe"]
        require(recipe["name"] == plan["protocol"]["shared_dr_recipe"] and recipe["training"] is True and
                recipe["wheel_contact_friction"]["range"] == [.3, 1.2] and
                recipe["restitution_randomization"] is False and metadata["live_physx_materials_verified"] is True and
                metadata["ground_material_usd_verified"] is True, "Native PhysX shared DR verification differs")
    else:
        weights = env.objective.weights
        values = {"physics_dt_s": env.physics_dt, "decimation": env.decimation,
            "policy_dt_s": env.step_dt, "actor_dim": metadata["actor_dim"], "critic_dim": metadata["critic_dim"],
            "action_dim": env.num_actions, "action_rate_weight": weights["action_rate_l2"],
            "grounded_reward_weight": weights["drift_grounded"], "reward_term_count": len(weights),
            "leg_kp": env.contract["leg_kp"], "leg_kd": env.contract["leg_kd"],
            "absolute_height_termination": metadata["absolute_height_termination"],
            "absolute_xy_termination": metadata["absolute_xy_termination"]}
        dr = metadata["dr"]
        require(dr["mode"] == "mjlab_v1p4" and dr["robot_single_sliding_mu_range"] == [.3, 1.2] and
                dr["encoder_bias_range_rad"] == [-.01, .01] and dr["friction_wheel_priority"] == 1,
                "MuJoCo-Warp shared DR declaration differs")
        require(str(env.device) == args.device and metadata["task_arrays_device"] == args.device and
                metadata["host_task_adapter"] is False, "GPU task/tensor device differs")
        require(metadata["native_backend"]["device"] == args.device and
                metadata["native_backend"]["physics_dt_s"] == .0025 and
                metadata["native_backend"]["decimation"] == 8,
                "Native MuJoCo-Warp device/clock metadata differs")
        require(all(math.isclose(float(model.opt.timestep), .0025, rel_tol=0, abs_tol=1e-15)
                for model in env.models) and env.batch.physics_dt == .0025 and env.batch.decimation == 8,
                "Compiled/batch MuJoCo clocks differ")
        device = {"requested_device": args.device, "torch_current_device": torch.cuda.current_device(),
            "task_arrays_device": metadata["task_arrays_device"], "native_backend": copy.deepcopy(metadata["native_backend"]),
            "cuda_visible_devices": None, "native_PhysX_device_audit_applicable": False}
    return {"status": "verified", "fields": validate_runtime_fields(values), "device": device,
        "shared_DR_recipe": plan["protocol"]["shared_dr_recipe"], "no_extra_environment_steps": True}


def before_learn(env, args, manifest, runner, plan):
    runtime = runtime_audit(env, args, manifest["environment"], plan)
    plan["runtime_audit"] = runtime
    manifest["evaluation_protocol"] = copy.deepcopy(plan["protocol"].get("validation", {}))
    manifest["smoke_only"] = plan["run_kind"] == "smoke"
    manifest["trained_policy_claimed"] = plan["run_kind"] == "formal"
    manifest["cohort"] = {"id": plan["protocol"]["cohort_id"], "run_kind": plan["run_kind"],
        "protocol_sha256": plan["protocol_sha256"], "wrapper_sha256": plan["wrapper_sha256"],
        "base_trainer_sha256": BASE_SHA256, "counted_generated_source_sha256": plan["generated_sha256"],
        "source_files_sha256": plan["source_hashes"], "original_input_sha256": plan["original_input_sha256"],
        "derived_input_sha256": plan["derived_input_sha256"], "runtime": runtime,
        "from_scratch": True, "historical_checkpoint_used": False,
        "smoke_only": plan["run_kind"] == "smoke", "trained_policy_claimed": plan["run_kind"] == "formal",
        "network_seed_after_environment_setup_and_smoke": True,
        "initial_policy_state_sha256": manifest["initial_policy_state_sha256"],
        "physical_substeps_per_environment_transition": 8}
    write_json(args.output_dir/"manifest.json", manifest)
    write_json(args.output_dir/"cohort_startup_audit.json", manifest["cohort"])


def finalize_contract(args, manifest, runner, exported, probe, expected, plan):
    import torch
    require(runner.tot_timesteps == plan["expected_transitions"], "Exact declared collection budget not completed")
    with torch.inference_mode():
        observed = exported(probe)
        require(torch.allclose(observed, expected.cpu(), rtol=1e-5, atol=1e-6), "Normalized export parity differs")
        export_error = float(torch.max(torch.abs(observed-expected.cpu())))
    contract = json.loads(args.contract.read_text())
    smoke = plan["run_kind"] == "smoke"
    contract.update(contract_status="smoke_completed_normalized_actor_not_formal_policy" if smoke else "trained_completed_normalized_actor",
        smoke_only=smoke, trained_policy_claimed=not smoke, run_kind=plan["run_kind"], policy_algorithm="PPO",
        policy_training_backend=args.backend, checkpoint_sha256=sha256(args.output_dir/"model_final.pt"),
        checkpoint_file="model_final.pt", exported_policy_sha256=sha256(args.output_dir/"policy.pt"), actor_file="policy.pt",
        actor_includes_observation_normalizer=True, actor_output_is_unclipped=True, actor_observation_groups=["policy"],
        export_validation_max_abs_error=export_error, reference_file_sha256=sha256(args.motion),
        trained_task_profile=json.loads(args.task_profile.read_text()),
        training_domain_randomization=copy.deepcopy(plan["protocol"]["training_randomization"]),
        training_cohort_protocol_sha256=plan["protocol_sha256"], training_environment_transitions=runner.tot_timesteps,
        runtime_clock_and_interface_verified=True, hardware_ready=False)
    require(contract["task_profile"] == contract["trained_task_profile"] and
            contract["task_profile"]["contact_friction"] == 1. and "evaluation_contact_friction_override" not in contract,
            "Exported training contract contains a stale evaluation profile")
    write_json(args.output_dir/"final_policy_contract.json", contract)
    write_json(args.output_dir/"policy_contract.json", contract)
    manifest["contract"] = contract
    manifest["cohort"]["normalized_actor_export_verified"] = True
    manifest["cohort"]["export_validation_max_abs_error"] = export_error
    write_json(args.output_dir/"manifest.json", manifest)


def complete(args, manifest, report, plan):
    require(report["status"] == "training_completed" and report["training_environment_steps"] == plan["expected_transitions"],
            "Completion/budget differs")
    for name, expected in plan["source_hashes"].items():
        require(sha256(ROOT/name) == expected, "Frozen cohort source changed while running: " + name)
    require(sha256(plan["protocol_path"]) == plan["protocol_sha256"], "Protocol changed during training")
    smoke = plan["run_kind"] == "smoke"
    report.update(cohort_id=plan["protocol"]["cohort_id"], run_kind=plan["run_kind"],
        smoke_only=smoke, trained_policy_claimed=not smoke, protocol_sha256=plan["protocol_sha256"])
    audit = {"status": "completed_verified", "cohort_id": plan["protocol"]["cohort_id"],
        "run_kind": plan["run_kind"], "backend": args.backend, "training_environment_steps": report["training_environment_steps"],
        "smoke_only": smoke, "trained_policy_claimed": not smoke, "seed": 42,
        "protocol_sha256": plan["protocol_sha256"], "physics_dt_s": .0025, "decimation": 8,
        "policy_dt_s": .02, "action_rate_weight": -.5, "grounded_reward_weight": .4,
        "checkpoint_sha256": report["checkpoint_sha256"], "exported_policy_sha256": report["exported_policy_sha256"],
        "manifest_sha256": sha256(args.output_dir/"manifest.json"),
        "policy_contract_sha256": sha256(args.output_dir/"policy_contract.json"),
        "final_policy_contract_sha256": sha256(args.output_dir/"final_policy_contract.json"),
        "checkpoint_file": "model_final.pt", "actor_file": "policy.pt",
        "final_policy_contract_file": "final_policy_contract.json",
        "startup_audit_sha256": sha256(args.output_dir/"cohort_startup_audit.json"),
        "initial_policy_state_sha256": manifest["initial_policy_state_sha256"],
        "source_files_unchanged": True, "additional_reward_terms": False, "historical_checkpoint_used": False}
    write_json(args.output_dir/"cohort_completion_audit.json", audit)


def materialize_source(text):
    replacements = [
        ("    args = parser.parse_args()\n", "    args = parser.parse_args()\n    _cohort_before_start(args, _cohort_plan)\n"),
        ("            app = AppLauncher(headless=True, device=args.device).app", "            app = _cohort_launch_app(AppLauncher, args, _cohort_plan)"),
        ("        import torch\n        from rsl_rl.runners import OnPolicyRunner", "        import torch\n        _cohort_set_torch_device(torch, args, _cohort_plan)\n        from rsl_rl.runners import OnPolicyRunner"),
        ("            apply_mjlab_recipe(cfg)", "            _cohort_configure_physx(cfg, args, _cohort_plan, apply_mjlab_recipe)"),
        ("        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)",
         "        _cohort_before_learn(env, args, manifest, runner, _cohort_plan)\n        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)"),
        ('        write_json(args.output_dir/"policy_contract.json", json.loads(args.contract.read_text()))',
         "        _cohort_finalize_contract(args, manifest, runner, exported, probe, expected, _cohort_plan)"),
        ('        write_json(args.output_dir/"environment_final.json",\n                   env.metadata() if args.backend == "mjlab" else recipe_audit(env))',
         '        write_json(args.output_dir/"environment_final.json",\n                   env.metadata() if args.backend == "mjlab" else recipe_audit(env))\n        _cohort_complete(args, manifest, report, _cohort_plan)'),
    ]
    require(hashlib.sha256(text.encode()).hexdigest() == BASE_SHA256, "Original trainer content differs")
    for old, new in replacements:
        require(text.count(old) == 1, "Frozen integration anchor absent or ambiguous: " + old[:80])
        text = text.replace(old, new, 1)
    compile(text, "<matched_400hz_backend_trainer>", "exec")
    return text


def run(argv):
    plan = prepare_plan(argv)
    generated = materialize_source(plan["base"].read_text())
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    namespace = {"__name__": "_frozen_400hz_backend_trainer", "__file__": str(plan["base"]), "_cohort_plan": plan,
        "_cohort_before_start": before_start, "_cohort_launch_app": launch_app, "_cohort_set_torch_device": set_torch_device,
        "_cohort_configure_physx": configure_physx, "_cohort_before_learn": before_learn,
        "_cohort_finalize_contract": finalize_contract, "_cohort_complete": complete}
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<matched_400hz_backend_trainer>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    run(sys.argv[1:])
