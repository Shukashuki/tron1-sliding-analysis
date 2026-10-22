"""Read-only provenance and inference contract for the NEW paired 400 Hz PPOs.

No checkpoint pickle is loaded. The completed trainer's normalized TorchScript
export is the only inference executable. Smoke actors require explicit opt-in.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config/ppo_backend_400hz_comparison_v1.json"
TRAINER_SHA256 = "5234070be219e59c61e6c4e4435e0dd0b2ab71df60d9c23886e507ed73692390"
INPUT_PINS = {
    "motion": "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254",
    "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513",
    "asset": "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e",
}
EVALUATION_DEPENDENCIES = {
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "5fee46238c825a2bcf474e969e93af65834aa1302ce857aa1c553e8a9fbaefaa",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _relative_file(directory, name, expected):
    require(name == expected, "Unexpected artifact filename: " + str(name))
    path = directory / name
    require(path.is_file() and not path.is_symlink(), "Missing/indirect artifact: " + str(path))
    return path


def verify_protocol(path=DEFAULT_PROTOCOL):
    # Importing this trainer is stdlib-only; its strict protocol validation is
    # reused, not its training entry point or simulator/task imports.
    require(sha256(Path(__file__).with_name("train_ppo_backend_400hz.py")) == TRAINER_SHA256,
            "Frozen completed-cohort trainer source differs")
    from train_ppo_backend_400hz import validate_protocol
    protocol = validate_protocol(read_json(path))
    require(protocol["kind"] == "matched_ppo_backend_400hz_action_rate_cohort", "Wrong cohort kind")
    validation = protocol["validation"]
    require(validation["friction_grid"] == [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
            and validation["full_horizon_s"] == 6.98 and validation["turn_core_s"] == [1.44, 4.]
            and validation["randomization"] is False and validation["observation_noise"] is False
            and validation["deterministic_start_frame"] == 0
            and validation["absolute_height_gate"] is False and validation["airborne_failure"] is False,
            "Validation protocol changed")
    return protocol


def verify_cohort(cohort_dir, protocol_path=DEFAULT_PROTOCOL, *, allow_smoke=False):
    """Verify completed files and their cross-links without executing weights."""
    directory, protocol_path = Path(cohort_dir).resolve(), Path(protocol_path).resolve()
    protocol = verify_protocol(protocol_path)
    names = ("cohort_completion_audit.json", "final_policy_contract.json", "policy_contract.json",
             "manifest.json", "cohort_startup_audit.json", "run_status.json", "model_final.pt", "policy.pt")
    paths = {name: _relative_file(directory, name, name) for name in names}
    hashes = {name: sha256(path) for name, path in paths.items()}
    completion, contract, manifest, startup, status = [read_json(paths[name]) for name in
        ("cohort_completion_audit.json", "final_policy_contract.json", "manifest.json",
         "cohort_startup_audit.json", "run_status.json")]
    require(paths["final_policy_contract.json"].read_bytes() == paths["policy_contract.json"].read_bytes(),
            "Canonical and compatibility contracts differ")
    require(completion.get("status") == "completed_verified", "Cohort did not complete verified")
    require(completion.get("run_kind") in ("smoke", "formal"), "Missing explicit cohort run_kind")
    smoke = completion["run_kind"] == "smoke"
    require(not smoke or allow_smoke, "Smoke actor is not a formal trained policy; explicit opt-in required")
    budget = 1536 if smoke else 12288000
    backend = completion.get("backend")
    require(backend in ("physx", "mjlab"), "Unknown training backend")
    for item in (completion, contract, manifest, manifest.get("cohort", {}), startup, status):
        require(item.get("smoke_only") is smoke and item.get("trained_policy_claimed") is (not smoke),
                "Smoke/formal identity missing or inconsistent")
    require(status.get("status") == "training_completed" and status.get("run_kind") == completion["run_kind"]
            and status.get("training_environment_steps") == budget, "Final training status/budget differs")
    for key, expected in {"cohort_id": protocol["cohort_id"], "protocol_sha256": sha256(protocol_path),
            "seed": 42, "training_environment_steps": budget, "physics_dt_s": .0025, "decimation": 8,
            "policy_dt_s": .02, "action_rate_weight": -.5, "grounded_reward_weight": .4,
            "source_files_unchanged": True, "additional_reward_terms": False,
            "historical_checkpoint_used": False}.items():
        require(completion.get(key) == expected, "Completion field differs: " + key)
    files = {"checkpoint": "model_final.pt", "actor": "policy.pt", "final_policy_contract": "final_policy_contract.json"}
    for key, filename in files.items():
        require(completion.get(key + "_file") == filename, "Completion artifact path differs: " + key)
    links = {"checkpoint_sha256": "model_final.pt", "exported_policy_sha256": "policy.pt",
        "manifest_sha256": "manifest.json", "policy_contract_sha256": "policy_contract.json",
        "final_policy_contract_sha256": "final_policy_contract.json", "startup_audit_sha256": "cohort_startup_audit.json"}
    for key, filename in links.items():
        require(completion.get(key) == hashes[filename], "Completed artifact SHA mismatch: " + filename)
    for item in (contract, status):
        require(item.get("checkpoint_sha256") == hashes["model_final.pt"] and
                item.get("exported_policy_sha256") == hashes["policy.pt"], "Actor/checkpoint identity mismatch")
    expected_contract = {"contract_status": "smoke_completed_normalized_actor_not_formal_policy" if smoke else "trained_completed_normalized_actor",
        "run_kind": completion["run_kind"], "cohort_id": protocol["cohort_id"], "policy_algorithm": "PPO",
        "policy_training_backend": backend, "checkpoint_file": "model_final.pt", "actor_file": "policy.pt",
        "physics_dt": .0025, "decimation": 8, "policy_fps": 50., "actor_dim": 51, "critic_dim": 120,
        "action_dim": 8, "leg_kp": 500., "leg_kd": 10., "actuator_model": "dc_motor",
        "leg_position_scale": 1., "wheel_torque_scale_nm": 12., "leg_torque_limit_nm": 80.,
        "wheel_torque_limit_nm": 12., "leg_saturation_effort_nm": 80., "wheel_saturation_effort_nm": 12.,
        "leg_motor_velocity_limit_rad_s": 15., "wheel_motor_velocity_limit_rad_s": 100.,
        "solver_joint_velocity_limit_rad_s": 1000.,
        "raw_action_clip": [-1., 1.], "actor_includes_observation_normalizer": True,
        "actor_output_is_unclipped": True, "actor_observation_groups": ["policy"],
        "reference_file_sha256": INPUT_PINS["motion"], "training_cohort_protocol_sha256": sha256(protocol_path),
        "training_environment_transitions": budget, "runtime_clock_and_interface_verified": True,
        "physics_clock_origin": "trained_at_native400Hz_not_evaluation_only_refinement"}
    for key, expected in expected_contract.items():
        require(contract.get(key) == expected, "Completed policy contract differs: " + key)
    legs = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"]
    wheels = ["wheel_L_Joint", "wheel_R_Joint"]
    require(contract.get("leg_joint_names") == legs and contract.get("wheel_joint_names") == wheels
            and contract.get("observed_joint_velocity_order") == legs + wheels, "Named action/state order differs")
    error = contract.get("export_validation_max_abs_error")
    require(isinstance(error, (int, float)) and not isinstance(error, bool) and math.isfinite(error)
            and 0 <= error <= 1e-4, "Export parity not verified")
    profile = contract.get("trained_task_profile", {})
    require(profile == contract.get("task_profile") and profile.get("contact_friction") == 1.
            and profile.get("action_rate_weight") == -.5 and profile.get("reward_weights", {}).get("grounded") == .4
            and profile.get("height_regularization", {}).get("mode") == "vertical_acceleration_only",
            "Training task/profile differs")
    require(contract.get("training_domain_randomization") == protocol["training_randomization"]
            and "evaluation_contact_friction_override" not in contract, "Stale/different training DR or friction override")
    cohort = manifest.get("cohort", {})
    require(manifest.get("contract") == contract and manifest.get("task_profile") == profile
            and manifest.get("backend") == backend and manifest.get("seed") == 42
            and manifest.get("transitions_requested") == budget
            and manifest.get("versions", {}).get("rsl-rl-lib") == "3.0.1", "Training manifest differs")
    declaration = protocol["smoke"] if smoke else protocol["algorithm"]
    require(manifest.get("num_envs") == declaration["num_envs"]
            and manifest.get("iterations") == declaration["iterations"], "Training collection shape differs")
    initial = manifest.get("initial_policy_state_sha256")
    require(isinstance(initial, str) and len(initial) == 64
            and all(c in "0123456789abcdef" for c in initial), "Initial network state SHA absent")
    for item in (cohort, startup, completion, status):
        require(item.get("initial_policy_state_sha256") == initial, "Initial network state identity differs")
    for item in (cohort, startup):
        require(item.get("id") == protocol["cohort_id"] and item.get("protocol_sha256") == sha256(protocol_path)
                and item.get("wrapper_sha256") == TRAINER_SHA256 and item.get("from_scratch") is True
                and item.get("historical_checkpoint_used") is False
                and item.get("network_seed_after_environment_setup_and_smoke") is True
                and item.get("physical_substeps_per_environment_transition") == 8
                and item.get("runtime", {}).get("status") == "verified", "Startup training provenance differs")
        require(item.get("base_trainer_sha256") == protocol["base_trainer_sha256"]
                and item.get("source_files_sha256", {}).get("scripts/train_ppo_backend_400hz.py") == TRAINER_SHA256,
                "Frozen training source provenance differs")
        from train_ppo_backend_400hz import validate_runtime_fields
        validate_runtime_fields(item["runtime"]["fields"])
    require(cohort["runtime"] == startup["runtime"], "Startup/final runtime audit differs")
    require(cohort.get("normalized_actor_export_verified") is True, "Normalized export audit absent")
    for name in ("motion", "model"):
        require(manifest.get("inputs", {}).get(name, {}).get("sha256") == INPUT_PINS[name], "Training input differs: " + name)
    if backend == "physx":
        require(manifest.get("inputs", {}).get("asset", {}).get("sha256") == INPUT_PINS["asset"], "Training USD asset differs")
    return {"directory": directory, "protocol_path": protocol_path, "protocol": protocol,
        "contract": contract, "manifest": manifest, "completion": completion, "paths": paths,
        "input_sha256": hashes, "training_backend": backend, "smoke_only": smoke,
        "checkpoint": paths["model_final.pt"], "actor": paths["policy.pt"]}


def evaluation_contract(cohort, mu, backend):
    require(not isinstance(mu, bool) and isinstance(mu, (int, float)) and math.isfinite(mu)
            and mu in cohort["protocol"]["validation"]["friction_grid"], "Friction is not in the frozen 17-point grid")
    require(not cohort["smoke_only"] or mu == .7, "Smoke policy is restricted to nominal pipeline validation")
    require(backend in ("PhysX", "native CPU MuJoCo"), "Unknown evaluation backend")
    result = copy.deepcopy(cohort["contract"])
    result["task_profile"]["contact_friction"] = float(mu)
    result.update(full_horizon_diagnostic=True, domain_randomization="disabled during evaluation; nominal fixed parameters",
        evaluation_backend=backend, evaluation_contact_friction_override={
            "trained_contact_friction": 1., "evaluation_contact_friction": float(mu),
            "meaning": "target effective pair coefficient; native material mapping, not contact-law equivalence",
            "physx_robot_mu": float(mu), "physx_ground_mu": 1., "physx_combine": "multiply",
            "mujoco_colliding_geom_mu": float(mu), "mujoco_wheel_priority": 1,
            "mujoco_other_priority": 0, "mujoco_equal_priority_combine": "max", "mu_squared": False})
    return result


def prepare_inputs(cohort, motion, output, mu, backend, *, model=None, asset=None):
    """Write fresh sibling evaluation inputs, never inside completed training."""
    output = Path(output).resolve()
    inputs = output.with_name(output.name + "_inputs")
    require(not output.exists() and not inputs.exists(), "Preserve existing output and evaluation inputs")
    require(not output.is_relative_to(cohort["directory"]) and not cohort["directory"].is_relative_to(output),
            "Evaluation must not overlap training artifacts")
    supplied = {"motion": motion, **({"model": model} if model is not None else {}),
                **({"asset": asset} if asset is not None else {})}
    for name, path in supplied.items():
        require(sha256(path) == INPUT_PINS[name], "Pinned evaluation input differs: " + name)
    contract = evaluation_contract(cohort, mu, backend)
    inputs.mkdir(parents=True)
    write_json(inputs / "contract.json", contract)
    write_json(inputs / "task_profile.json", cohort["contract"]["trained_task_profile"])
    return {"output": output, "input_dir": inputs, "contract_path": inputs / "contract.json",
            "task_profile_path": inputs / "task_profile.json", "contract": contract,
            "motion": Path(motion).resolve(), "mu": float(mu), "cohort": cohort}


def provenance(cohort):
    return {"kind": "completed_paired_backend_400hz_ppo_evaluation", "cohort_id": cohort["protocol"]["cohort_id"],
        "training_backend": cohort["training_backend"], "smoke_only": cohort["smoke_only"],
        "trained_policy_claimed": not cohort["smoke_only"], "run_kind": cohort["completion"]["run_kind"],
        "additional_training_updates": 0, "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
        "normalized_actor_sha256": cohort["input_sha256"]["policy.pt"],
        "cohort_input_sha256": dict(cohort["input_sha256"]), "protocol_sha256": sha256(cohort["protocol_path"]),
        "inference_algorithm": "PPO", "normalizer_included_in_actor": True,
        "outer_normalizer": "Identity", "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "old_200hz_actor_used": False, "absolute_height_gate": False, "airborne_failure": False,
        "contact_law_equivalence_claimed": False, "hardware_ready": False}


def verify_legacy(path, expected):
    path = Path(path).resolve()
    require(sha256(path) == expected, "Pinned evaluator source SHA mismatch")
    root = path.parent.parent
    for name, digest in EVALUATION_DEPENDENCIES.items():
        require(sha256(root / name) == digest, "Pinned legacy dependency differs: " + name)
    return path, root


def verify_recording(path, backend):
    """Validate native, unresampled states plus the actual 50 Hz action count."""
    import numpy as np
    require(backend in ("PhysX", "native CPU MuJoCo"), "Unknown recording backend")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    times = np.asarray(data["time_s"], dtype=float)
    require(times.shape == (2793,) and np.allclose(times, np.arange(2793) * .0025, rtol=0, atol=1e-10),
            "Native 400Hz recording clock/count differs")
    require(float(data["physics_dt_s"]) == .0025 and float(data["policy_dt_s"]) == .02, "Recording declared clock differs")
    if backend == "PhysX":
        require(np.asarray(data["valid_mask"]).shape == (2793, 1) and data["valid_mask"].all(),
                "PhysX recording contains invalid/reset-spliced samples")
        counters = np.asarray(data["simulation_step_counter"])
        require(counters.shape == (2793,) and np.all(np.diff(counters) == 1), "Native counters not consecutive")
        applied = np.asarray(data["action_was_applied"], dtype=bool)
        require(applied.shape == (2793, 1) and not applied[0, 0] and applied[1:].all(), "Held-action flags differ")
        expected = {"action_observation": (2793, 1, 51), "actions_raw": (2793, 1, 8), "actions_clipped": (2793, 1, 8),
            "root_position_m": (2793, 1, 3), "root_quaternion_wxyz": (2793, 1, 4),
            "joint_position_rad": (2793, 1, 8), "joint_velocity_rad_s": (2793, 1, 8), "applied_joint_torque_nm": (2793, 1, 8)}
        for key in ("actions_raw", "actions_clipped", "action_observation"):
            held = np.asarray(data[key])[1:].reshape(349, 8, -1)
            require(np.array_equal(held, np.repeat(held[:, :1], 8, axis=1)), "Policy context changed within native substeps")
    else:
        expected = {"action_observation": (349, 51), "actions_raw": (349, 8), "actions_clipped": (349, 8),
            "root_pos": (2793, 3), "root_quat_wxyz": (2793, 4), "joint_pos": (2793, 8),
            "joint_vel": (2793, 8), "joint_torque_nm": (2793, 8), "previous_clipped_actions": (349, 8)}
        require(np.allclose(data["policy_time_s"], np.arange(349) * .02, rtol=0, atol=1e-10), "Native policy clock differs")
        require(np.array_equal(data["previous_clipped_actions"][0], np.zeros(8)) and
                np.array_equal(data["previous_clipped_actions"][1:], data["actions_clipped"][:-1]),
                "Previous-action observation history differs")
    for key, shape in expected.items():
        value = np.asarray(data[key])
        require(value.shape == shape and np.isfinite(value).all(), "Invalid native field: " + key)
    require(np.array_equal(data["actions_clipped"], np.clip(data["actions_raw"], -1., 1.)), "Action clipping differs")
    return {"status": "verified", "native_states": 2793, "policy_decisions": 349,
        "duration_s": 6.98, "physics_dt_s": .0025, "policy_dt_s": .02, "recording_sha256": sha256(path)}


def validate_output(folder, cohort_dir, protocol_path=DEFAULT_PROTOCOL, *, validation_backend, allow_smoke=False):
    """Parent-side read-only check after the simulator child has exited."""
    cohort = verify_cohort(cohort_dir, protocol_path, allow_smoke=allow_smoke)
    folder = Path(folder)
    report = read_json(folder / "report.json")
    audit = report.get("backend_400hz_cohort_evaluation", {})
    for key, expected in provenance(cohort).items():
        require(audit.get(key) == expected, "Evaluation cohort identity differs: " + key)
    require(audit.get("status") == "verified" and audit.get("validation_backend") == validation_backend,
            "Wrong/unverified validation backend")
    wrapper_name = "eval_ppo_backend_400hz_isaac.py" if validation_backend == "PhysX" else "eval_ppo_backend_400hz_mujoco.py"
    require(audit.get("wrapper_sha256") == sha256(Path(__file__).with_name(wrapper_name)), "Evaluation wrapper source differs")
    if validation_backend == "PhysX":
        require(report.get("status") == "evaluated" and report.get("domain_randomization_enabled") is False
                and report.get("observation_noise_enabled") is False and report.get("deterministic_start_frame") == 0,
                "PhysX lifecycle/DR differs")
        require(report.get("runtime_device_audit", {}).get("status") == "verified"
                and report["runtime_device_audit"].get("requested_device") == "cuda:" + str(audit["physical_gpu_index_requested"]),
                "Native PhysX device identity not verified")
        materials = report["contact_material_audit"]
        require(materials.get("live_physx_materials_verified") is True and materials.get("ground_material_usd_verified") is True
                and materials["recipe"]["training"] is False, "Native PhysX material mapping unverified")
        contract = report["policy_export"]
        recording = folder / "physics_trajectory.npz"
        require(report.get("physics_trajectory_sha256") == sha256(recording), "Recorded PhysX hash differs")
        require(audit.get("deployment_actor_sha256") == sha256(folder / "actor_normalized.pt"), "Deployed actor hash differs")
    else:
        require(validation_backend == "native CPU MuJoCo" and report.get("status") == "completed"
                and report.get("completed_full_reference") is True and report.get("hidden_resets") == 0
                and report.get("root_state_writes") == 1 and report.get("start_reference_frame") == 0
                and report.get("domain_parameters_resampled_during_episode") is False, "Native MuJoCo lifecycle differs")
        require(report["contact_material_audit"].get("native_mapping_verified") is True
                and report["contact_material_audit"].get("explicit_contact_pair_count") == 0,
                "Native MuJoCo material mapping unverified")
        contract = report["contract"]
        recording = folder / "rollout.npz"
        require(audit.get("recording_sha256") == sha256(recording), "Recorded native MuJoCo hash differs")
        require(report.get("policy_sha256") == cohort["input_sha256"]["policy.pt"], "CPU deployed PPO actor differs")
    expected = evaluation_contract(cohort, contract["task_profile"]["contact_friction"], validation_backend)
    for key in ("physics_dt", "decimation", "task_profile", "trained_task_profile", "training_domain_randomization",
                "evaluation_contact_friction_override", "full_horizon_diagnostic"):
        require(contract.get(key) == expected[key], "Deployment contract changed: " + key)
    mu = contract["task_profile"]["contact_friction"]
    if validation_backend == "PhysX":
        require(materials["recipe"]["wheel_contact_friction"]["range"] == [mu, mu]
                and materials["recipe"]["wheel_contact_friction"]["physx_ground_friction"] == 1.
                and materials["recipe"]["wheel_contact_friction"]["physx_ground_friction_combine_mode"] == "multiply"
                and materials["recipe"]["nonwheel_friction"] == mu
                and materials.get("encoder_bias_range_rad") == [0., 0.], "Fixed PhysX material recipe differs")
    else:
        materials = report["contact_material_audit"]
        require(materials.get("pair_coefficient") == mu and all(abs(row[0] - mu) <= 1e-12 for row in materials["geom_friction"]),
                "Fixed native MuJoCo friction readback differs")
        priorities = dict(zip(materials["colliding_geom_ids"], materials["geom_priority"]))
        require(all(value == (1 if key in materials["wheel_geom_ids"] else 0) for key, value in priorities.items()),
                "Native MuJoCo geom priority mapping differs")
    verified = verify_recording(recording, validation_backend)
    return {**verified, "validation_backend": validation_backend, "training_backend": cohort["training_backend"],
        "smoke_only": cohort["smoke_only"], "report_sha256": sha256(folder / "report.json"),
        "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"], "evaluation_contact_friction": contract["task_profile"]["contact_friction"]}


class StrictPPOExportRunner:
    """Minimal RSL exporter interface, explicitly PPO and never a SAC adapter."""
    def __init__(self, cohort, device):
        self.cohort, self.device, self.alg = cohort, device, None
        self.inference_provenance = provenance(cohort)

    def load(self, checkpoint, *, load_optimizer=False, map_location=None):
        import torch
        require(self.alg is None and load_optimizer is False, "Load normalized PPO actor once; no optimizer loading")
        require(Path(checkpoint).resolve() == self.cohort["checkpoint"] and
                sha256(checkpoint) == self.cohort["input_sha256"]["model_final.pt"] and
                sha256(self.cohort["actor"]) == self.cohort["input_sha256"]["policy.pt"], "Actor/checkpoint changed before load")
        device = map_location or self.device
        exported = torch.jit.load(str(self.cohort["actor"]), map_location=device).eval()

        class PPOView(torch.nn.Module):
            is_recurrent = False
            obs_groups = {"policy": ["policy"]}

            def __init__(self, actor):
                super().__init__()
                self.actor, self.actor_obs_normalizer = actor, torch.nn.Identity()

            def get_actor_obs(self, obs):
                value = obs["policy"]
                require(value.ndim == 2 and value.shape[1] == 51 and torch.isfinite(value).all(), "Invalid 51D PPO observation")
                return value

            def forward(self, obs):
                value = self.actor(self.actor_obs_normalizer(self.get_actor_obs(obs)))
                require(value.ndim == 2 and value.shape[1] == 8 and torch.isfinite(value).all(), "Invalid raw PPO action")
                return value

        self.alg = SimpleNamespace(policy=PPOView(exported).to(device).eval())

    def get_inference_policy(self, device=None):
        require(self.alg is not None, "Load actor before inference")
        return self.alg.policy
