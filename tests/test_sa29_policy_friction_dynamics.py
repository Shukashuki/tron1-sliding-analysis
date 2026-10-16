"""Offline adversarial SA29 policy evidence tests; no physics or inference."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_sa29_policy_friction_dynamics as analysis
from controller_references import get_reference
from analyze_controller_pair import summarize_pair
from analyze_tvlqr_pilot import public_compact
from nmpc_tracking_objective import JOINT_NAMES
from test_controller_pair import policy_reports
from test_controller_trace import encoded_trace
from test_nmpc_nominal_development import canonical, geometry


SPEC = get_reference("doa_sa29_scale2")
GRID = [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
FRAMES, SAMPLES = 258, 1033


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, base):
    return {"path": str(Path(path).relative_to(base)), "sha256": digest(path)}


def policy_raw(engine):
    trace = canonical(SAMPLES, .005)
    raw = encoded_trace("policy_" + engine, trace, order=tuple(range(8)))
    phase = np.r_[0, np.repeat(np.arange(FRAMES), 4)]
    qpos = np.column_stack((trace["position"], trace["quaternion"], trace["joint_pos"]))
    raw.update(policy_dt_s=np.array(.02), qpos_joint_names=np.array(JOINT_NAMES))
    if engine == "isaac":
        raw.update(reference_frame=phase[:, None], reference_frame_for_action=phase[:, None],
            action_was_applied=np.r_[False, np.ones(SAMPLES - 1, dtype=bool)][:, None],
            terminal_sample=np.r_[np.zeros(SAMPLES - 1, dtype=bool), True][:, None],
            simulation_step_counter=np.arange(SAMPLES), qpos=qpos[:, None],
            actions_raw=np.zeros((SAMPLES, 1, 8)), actions_clipped=np.zeros((SAMPLES, 1, 8)),
            action_observation=np.zeros((SAMPLES, 1, 51)), applied_joint_torque_nm=np.zeros((SAMPLES, 1, 8)),
            nonwheel_ground_contact_count=np.zeros((SAMPLES, 1), dtype=int),
            nonwheel_ground_contact_force_norm_n=np.zeros((SAMPLES, 1)))
    else:
        raw.update(reference_frame=phase, reference_frame_for_action=np.arange(FRAMES),
            policy_time_s=raw["time_s"][::4][:-1].copy(), qpos=qpos,
            initial_qpos=qpos[0].copy(), initial_qvel=np.zeros(14),
            actions_raw=np.zeros((FRAMES, 8)), actions_clipped=np.zeros((FRAMES, 8)),
            previous_clipped_actions=np.zeros((FRAMES, 8)), action_observation=np.zeros((FRAMES, 51)),
            joint_torque_nm=np.zeros((SAMPLES, 8)),
            nonwheel_ground_contact_count=np.zeros(SAMPLES, dtype=int),
            nonwheel_ground_contact_force_norm_n=np.zeros(SAMPLES))
    return raw


def sa29_reports(algorithm="PPO", mu=.7):
    ia, mj = policy_reports(algorithm)
    ia.update(motion_file_sha256=SPEC["motion_npz_sha256"], reference_frames=FRAMES,
        reference_sample_span_s=5.14, physics_dt_s=.005, seed=42,
        summary={"episodes": [{"completed_full_reference": True, "recorded_samples": FRAMES + 1,
            "recorded_duration_s": 5.16}]})
    ia["physics_recording"]["samples"] = SAMPLES
    mj.update(motion_sha256=SPEC["motion_npz_sha256"], reference_frames=FRAMES,
        policy_steps=FRAMES, physics_steps=SAMPLES - 1, expected_duration_s=5.16,
        recorded_duration_s=5.16, completed_full_reference=True,
        start_reference_frame=0, final_reference_frame=FRAMES - 1)
    ia["evaluation_contact_friction_override"]["evaluation_contact_friction"] = mu
    ia["contact_material_audit"].update(expected_friction=mu, robot_runtime_friction_range=[mu, mu],
        ground_colliders=[{"static_friction": mu, "dynamic_friction": mu}])
    mj["contact_material_audit"].update(expected_friction=mu, compiled_friction_range=[mu, mu])
    return ia, mj


def write_training(tmp_path, algorithm):
    """Explicit synthetic metadata; no ignored outputs or real weights needed."""
    folder = tmp_path / algorithm; folder.mkdir()
    profile = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    height = {**profile["height_regularization"], "absolute_height_reward": False, "height_termination": False,
        "reference_vertical_velocity_reward": False, "position_and_linear_velocity_tracking_axes": ["x", "y"]}
    recipe = {"upstream_commit": "cd65172032893724b445448818c34165846d847d", "friction_static": [.3, 1.6],
        "friction_dynamic": [.3, 1.2], "restitution": [0., .5], "material_buckets": 64,
        "joint_default_position_offset_rad": [-.01, .01], "base_com_offset_m": {"x": [-.025, .025], "y": [-.05, .05], "z": [-.05, .05]},
        "push_interval_s": [1., 3.], "push_velocity_range": {"x": [-.5, .5], "y": [-.5, .5], "z": [-.2, .2],
            "roll": [-.52, .52], "pitch": [-.52, .52], "yaw": [-.78, .78]}, "ground_friction": 1.}
    device = {"status": "verified", "required_unmasked": True, "cuda_visible_devices": None,
        "native_and_tensor_device_consistency_verified": True, "requested_device": "cuda:2",
        "environment_device": "cuda:2", "physics_sim_view_frontend_device": "cuda:2", "robot_native_transform_tensor_device": "cuda:2",
        "native_physics_backend_device_ordinal": 2, "torch_current_device": 2, "physics_cudaDevice_setting": 2,
        "physics_sceneMultiGPUMode_setting": 0}
    contract = {"actor_dim": 51, "critic_dim": 120, "action_dim": 8, "height_regularization": copy.deepcopy(height),
        "training_domain_randomization": copy.deepcopy(recipe), "root_prescribed_during_steps": False, "motion_end_hidden_teleport": False}
    adapters = {name: digest(ROOT / "scripts" / name) for name in (
        "train_beyondmimic_style.py", "train_sac_tracking_isaac.py", "sac_tracking.py", "terminal_observation.py", "height_smoothness.py")}
    names = ["training/tron1_tracking.py", "training/tron1_drift.py", "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py"]
    if algorithm == "PPO": names.append("scripts/train_tracking.py")
    training = {"algorithm": algorithm, "smoke_only": False, "seed": 42, "num_envs": 512,
        "environment_transitions_requested": 12288000, "policy_steps_requested": 24000,
        "reference_catalog_sha256": digest(ROOT / "config/controller_references.json"), "device_audit": device,
        "height_adapter_sha256": adapters["height_smoothness.py"], "height_regularization": height,
        "task_profile": profile, "task_profile_sha256": digest(ROOT / "config/car_drift_height_smooth_task.json"),
        "recipe": recipe, "contract": copy.deepcopy(contract), "policy_contract": copy.deepcopy(contract),
        "source_training_modules_sha256": {name: digest(ROOT / "upstream/tron1-learning-from-human" / name) for name in names},
        "motion_file": "/private/original-reference/motion.npz", "asset_file": "/private/assets/WF_TRON1A.usda"}
    status = {"status": "training_completed", "smoke_only": False, "trained_policy_claimed": True,
        "environment_transitions": 12288000, "policy_steps_completed": 24000}
    checkpoint = folder / "model_final.pt"; checkpoint.write_bytes((algorithm + " synthetic opaque checkpoint token").encode())
    actor = folder / "actor_normalized.pt"; actor.write_bytes((algorithm + " synthetic opaque actor token").encode())
    training.update(reference_id=SPEC["reference_id"], reference_spec=copy.deepcopy(SPEC),
        motion_sha256=SPEC["motion_npz_sha256"], pooled_success_statistics=False)
    training["reference_spec"]["verification_scope"] = "motion_byte_identity_only_not_reference_pair_or_asset_validation"
    for name in training["source_training_modules_sha256"]:
        training["source_training_modules_sha256"][name] = digest(ROOT / "upstream/tron1-learning-from-human" / name)
    if algorithm == "SAC":
        sac = {"actor_dim": 51, "critic_dim": 120, "action_dim": 8, "hidden_sizes": [256, 128, 128], "activation": "elu",
            "learning_rate": .0003, "alpha_learning_rate": .0003, "initial_alpha": .004, "target_entropy": None,
            "tau": .005, "log_std_min": -5., "log_std_max": 2., "max_grad_norm": 10.}
        training.update(sac_config=sac, comparison_modules_sha256=adapters, batch_size=1024, gamma=.99,
            warmup_transitions=10000, updates_per_policy_step=1, replay_capacity=500000)
        status.update(checkpoint_sha256=digest(checkpoint), actor_sha256=digest(actor), learner_updates=23981)
        runner = {"algorithm": "SAC", "sac": copy.deepcopy(sac)}
    else:
        training.update(iterations=1000, num_steps_per_env=24, setup_smoke_control_steps=0,
            trainer_sha256=adapters["train_beyondmimic_style.py"])
        status.update(iterations_requested=1000, learning_iteration_index=999)
        runner = {"seed": 42, "num_steps_per_env": 24, "save_interval": 100, "logger": "tensorboard",
            "experiment_name": "tron1_wf_single_jump", "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
            "policy": {"class_name": "ActorCritic", "init_noise_std": .5, "noise_std_type": "log",
                "actor_obs_normalization": True, "critic_obs_normalization": True, "actor_hidden_dims": [256, 128, 128],
                "critic_hidden_dims": [256, 128, 128], "activation": "elu"},
            "algorithm": {"class_name": "PPO", "value_loss_coef": 1., "use_clipped_value_loss": True, "clip_param": .2,
                "entropy_coef": .005, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": .001,
                "schedule": "adaptive", "gamma": .99, "lam": .95, "desired_kl": .01, "max_grad_norm": 1.}}
    for name, value in (("manifest.json", training), ("run_status.json", status),
        ("runner_config.json", runner), ("policy_contract.json", contract)):
        (folder / name).write_text(json.dumps(value))
    return folder, training, status, runner, contract, digest(checkpoint), digest(actor)


@pytest.mark.parametrize("algorithm", ["PPO", "SAC"])
def test_complete_sa29_training_budget_is_actual_reference_specific_and_privacy_safe(tmp_path, algorithm):
    _, training, status, runner, contract, checkpoint, actor = write_training(tmp_path, algorithm)
    before = copy.deepcopy((training, status, runner, contract))
    result = analysis.audit_training_budget(algorithm, training, status, runner, contract, checkpoint,
        actor if algorithm == "SAC" else None)
    assert (training, status, runner, contract) == before
    serialized = json.dumps(result, allow_nan=False)
    assert "/data/" not in serialized and "/home/" not in serialized and str(tmp_path) not in serialized


@pytest.mark.parametrize("algorithm", ["PPO", "SAC"])
@pytest.mark.parametrize("problem", ["primary_reference", "primary_motion", "primary_spec", "wrong_seed", "wrong_envs",
    "requested_budget", "actual_budget", "actual_policy_steps", "smoke", "not_completed", "not_claimed",
    "height_gate", "height_adapter", "friction_recipe", "restitution_recipe", "source", "task_profile"])
def test_training_cannot_be_primary_transfer_partial_collection_or_a_changed_recipe(tmp_path, algorithm, problem):
    _, training, status, runner, contract, checkpoint, actor = write_training(tmp_path, algorithm)
    if problem == "primary_reference": training["reference_id"] = "drift_drl_traj4"
    elif problem == "primary_motion": training["motion_sha256"] = get_reference()["motion_npz_sha256"]
    elif problem == "primary_spec": training["reference_spec"] = get_reference()
    elif problem == "wrong_seed": training["seed"] = 43
    elif problem == "wrong_envs": training["num_envs"] = 256
    elif problem == "requested_budget": training["environment_transitions_requested"] -= 512
    elif problem == "actual_budget": status["environment_transitions"] -= 512
    elif problem == "actual_policy_steps": status["policy_steps_completed"] -= 1
    elif problem == "smoke": training["smoke_only"] = True
    elif problem == "not_completed": status["status"] = "training"
    elif problem == "not_claimed": status["trained_policy_claimed"] = False
    elif problem == "height_gate": training["height_regularization"]["height_termination"] = True
    elif problem == "height_adapter": training["height_adapter_sha256"] = "0" * 64
    elif problem == "friction_recipe": training["recipe"]["friction_static"] = [.01, 2.]
    elif problem == "restitution_recipe": training["recipe"]["restitution"] = [0., 0.]
    elif problem == "source": training["source_training_modules_sha256"]["training/tron1_drift.py"] = "0" * 64
    else: training["task_profile"]["reward_weights"]["speed"] = 0.
    with pytest.raises(ValueError):
        analysis.audit_training_budget(algorithm, training, status, runner, contract, checkpoint,
            actor if algorithm == "SAC" else None)


def test_interrupted_iteration_199_ppo_is_not_completed_training_even_if_supervisor_exit_was_zero(tmp_path):
    _, training, status, runner, contract, checkpoint, _ = write_training(tmp_path, "PPO")
    status.update(status="training", trained_policy_claimed=False,
        environment_transitions=2457600, policy_steps_completed=4800, learning_iteration_index=199)
    with pytest.raises(ValueError, match="completed|budget"):
        analysis.audit_training_budget("PPO", training, status, runner, contract, checkpoint)


@pytest.mark.parametrize("problem", ["learning_rate", "network", "rollout"])
def test_ppo_fixed_training_recipe_rejects_changed_runner_knobs(tmp_path, problem):
    _, training, status, runner, contract, checkpoint, _ = write_training(tmp_path, "PPO")
    if problem == "learning_rate": runner["algorithm"]["learning_rate"] = .01
    elif problem == "network": runner["policy"]["actor_hidden_dims"] = [32, 32]
    else: runner["num_steps_per_env"] = 12
    with pytest.raises(ValueError):
        analysis.audit_training_budget("PPO", training, status, runner, contract, checkpoint)


@pytest.mark.parametrize("problem", ["learning_rate", "batch_size", "gamma", "warmup", "updates", "replay"])
def test_sac_coherent_runner_and_manifest_mutations_do_not_create_a_new_tuned_recipe(tmp_path, problem):
    _, training, status, runner, contract, checkpoint, actor = write_training(tmp_path, "SAC")
    if problem == "learning_rate":
        training["sac_config"]["learning_rate"] = runner["sac"]["learning_rate"] = .01
    elif problem == "batch_size": training["batch_size"] = 512
    elif problem == "gamma": training["gamma"] = .9
    elif problem == "warmup": training["warmup_transitions"] = 1000
    elif problem == "updates": training["updates_per_policy_step"] = 2
    else: training["replay_capacity"] = 100000
    with pytest.raises(ValueError):
        analysis.audit_training_budget("SAC", training, status, runner, contract, checkpoint, actor)


@pytest.mark.parametrize("engine", ["isaac", "mujoco"])
def test_policy_trace_uses_all_original_258_decisions_and_1033_physical_samples(engine):
    raw = policy_raw(engine); report = sa29_reports()[0 if engine == "isaac" else 1]
    before = {key: value.copy() for key, value in raw.items()}
    trace = analysis.audit_policy_trace(raw, engine, report)
    assert trace["time_s"].shape == (SAMPLES,)
    np.testing.assert_array_equal(trace["time_s"], np.arange(SAMPLES) * .005)
    assert trace["physics_dt_s"] == .005
    for key, original in before.items(): np.testing.assert_array_equal(raw[key], original)


@pytest.mark.parametrize("engine", ["isaac", "mujoco"])
@pytest.mark.parametrize("problem", ["missing_sample", "wrong_dt", "offset_clock", "clock_gap", "reset_splice",
    "action_phase", "physical_phase", "unbounded_action", "nan_action", "missing_contact", "negative_slip",
    "fractional_count", "qpos_root", "qpos_joint", "missing_nonwheel", "wrong_report_frames"])
def test_original_native_measurements_and_action_phase_fail_closed(engine, problem):
    raw = policy_raw(engine); report = sa29_reports()[0 if engine == "isaac" else 1]
    if problem == "missing_sample": raw["time_s"] = raw["time_s"][:-1]
    elif problem == "wrong_dt": raw["physics_dt_s"] = np.array(.02)
    elif problem == "offset_clock": raw["time_s"] += .005
    elif problem == "clock_gap": raw["time_s"][500:] += .005
    elif problem == "reset_splice": raw["time_s"][500:] -= 2.
    elif problem == "action_phase": raw["reference_frame_for_action"][100] += 1
    elif problem == "physical_phase": raw["reference_frame"][100] += 1
    elif problem == "unbounded_action": raw["actions_clipped"][100] = 1.01
    elif problem == "nan_action": raw["actions_raw"][100] = np.nan
    elif problem == "missing_contact": del raw["wheel_ground_contact_count"]
    elif problem == "negative_slip": raw["wheel_lateral_contact_slip_m_s"][100] = -1.
    elif problem == "fractional_count": raw["wheel_ground_contact_count"][100] = .5
    elif problem == "qpos_root": raw["qpos"][100, ..., 0] += .1
    elif problem == "qpos_joint": raw["qpos"][100, ..., 7] += .1
    elif problem == "missing_nonwheel": del raw["nonwheel_ground_contact_count"]
    else: report["reference_frames"] = 349
    with pytest.raises((ValueError, KeyError)):
        analysis.audit_policy_trace(raw, engine, report)


@pytest.mark.parametrize("problem", ["prefix", "int_mask", "extra_env", "wrong_sim_counter", "no_terminal", "extra_terminal",
    "no_applied_action", "int_applied", "action_not_held"])
def test_isaac_prefixes_reset_or_forward_samples_and_bad_masks_are_not_full_episodes(problem):
    raw = policy_raw("isaac"); report = sa29_reports()[0]
    if problem == "prefix": raw["valid_mask"][-1] = False
    elif problem == "int_mask": raw["valid_mask"] = raw["valid_mask"].astype(int)
    elif problem == "extra_env": raw["valid_mask"] = np.ones((SAMPLES, 2), dtype=bool)
    elif problem == "wrong_sim_counter": raw["simulation_step_counter"][100] += 1
    elif problem == "no_terminal": raw["terminal_sample"][-1] = False
    elif problem == "extra_terminal": raw["terminal_sample"][100] = True
    elif problem == "no_applied_action": raw["action_was_applied"][100] = False
    elif problem == "int_applied": raw["action_was_applied"] = raw["action_was_applied"].astype(int)
    else: raw["actions_raw"][100, 0, 0] = raw["actions_clipped"][100, 0, 0] = .2
    with pytest.raises(ValueError): analysis.audit_policy_trace(raw, "isaac", report)


@pytest.mark.parametrize("problem", ["policy_clock", "previous_action", "wrong_decision_count", "wrong_physics_count", "incomplete"])
def test_mujoco_full_playback_checks_decisions_history_and_physics_counts(problem):
    raw = policy_raw("mujoco"); report = sa29_reports()[1]
    if problem == "policy_clock": raw["policy_time_s"][100] += .005
    elif problem == "previous_action": raw["previous_clipped_actions"][100, 0] = .2
    elif problem == "wrong_decision_count": report["policy_steps"] = 257
    elif problem == "wrong_physics_count": report["physics_steps"] = 1031
    else: report["completed_full_reference"] = False
    with pytest.raises(ValueError): analysis.audit_policy_trace(raw, "mujoco", report)


def test_portable_binding_checks_actual_bytes_without_deserializing_checkpoint(tmp_path):
    path = tmp_path / "opaque.pt"; path.write_bytes(b"opaque, not a pickle or a model")
    result = analysis.checked_file(bind(path, tmp_path), tmp_path)
    assert result == path.resolve() and path.read_bytes() == b"opaque, not a pickle or a model"


@pytest.mark.parametrize("problem", ["digest", "missing", "absolute", "parent", "extra_key", "bad_digest", "symlink", "directory"])
def test_missing_changed_or_nonportable_capture_file_binding_fails_closed(tmp_path, problem):
    path = tmp_path / "evidence.bin"; path.write_bytes(b"recorded evidence")
    binding = bind(path, tmp_path)
    if problem == "digest": binding["sha256"] = "0" * 64
    elif problem == "missing": binding["path"] = "missing.bin"
    elif problem == "absolute": binding["path"] = str(path.resolve())
    elif problem == "parent": binding["path"] = "../" + tmp_path.name + "/evidence.bin"
    elif problem == "extra_key": binding["claimed_exit_code"] = 0
    elif problem == "bad_digest": binding["sha256"] = "not-a-sha256"
    elif problem == "symlink":
        outside = tmp_path.parent / (tmp_path.name + "-outside.bin"); outside.write_bytes(path.read_bytes())
        link = tmp_path / "alias.bin"; link.symlink_to(outside); binding["path"] = link.name
    else: binding["path"] = "."
    with pytest.raises((ValueError, FileNotFoundError, IsADirectoryError)):
        analysis.checked_file(binding, tmp_path)


def compact_fixture():
    reference = canonical(FRAMES, .02); measured = canonical(SAMPLES, .005)
    for key in ("contact_count", "contact_force", "contact_slip"):
        del reference[key]
    measured.update(nonwheel_ground_contact_count=np.zeros(SAMPLES), nonwheel_ground_contact_force_norm_n=np.zeros(SAMPLES))
    measured["quaternion"][:] = [np.cos(np.pi / 6), 0., 0., np.sin(np.pi / 6)]
    measured["contact_slip"][:, 0] = 3.
    traces = {"reference": reference, "PPO_mu0p700_isaac": measured}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    return public_compact(traces, recon)


def test_public_compact_replays_measured_com_and_heading_without_adding_height_or_slip_gates():
    data = compact_fixture()
    reference, reference_geometry = analysis.compact_case(data, "reference")
    trace, geom = analysis.compact_case(data, "PPO_mu0p700_isaac", require_nonwheel=True)
    pair = {"reference": reference, "isaac": trace, "mujoco": trace}
    recon = {"reference": reference_geometry, "isaac": geom, "mujoco": geom}
    result = summarize_pair(SPEC, pair, recon)
    value = result["core"]["isaac"]
    assert value["reference_xy_rmse_m"] == pytest.approx(0., abs=1e-14)
    assert value["mean_speed_retention_ratio"] == 1.
    assert value["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"] == pytest.approx(60.)
    assert value["ground_contact_lateral_slip_m_s"] == {"left": 3., "right": None}
    assert value["large_xy_exploratory_screen"] is value["heading_offset_is_mismatch"] is value["height_or_flight_failure_gate"] is False
    assert result["core"]["reference"]["ground_contact_lateral_slip_m_s"] is None


@pytest.mark.parametrize("problem", ["forged_speed", "forged_gamma", "forged_mask", "short_clock", "bad_quaternion",
    "missing_force", "fractional_count", "negative_slip", "missing_nonwheel"])
def test_compact_vectors_are_recomputed_not_trusted_summary_features(problem):
    data = compact_fixture(); prefix = "PPO_mu0p700_isaac_"
    if problem == "forged_speed": data[prefix + "speed_m_s"][100] += .2
    elif problem == "forged_gamma": data[prefix + "root_course_to_wheel_gamma_rad"][100, 0] += .2
    elif problem == "forged_mask": data[prefix + "root_beta_valid"][100] = False
    elif problem == "short_clock": data[prefix + "time_s"] = data[prefix + "time_s"][:-1]
    elif problem == "bad_quaternion": data[prefix + "quaternion"][100] *= 2.
    elif problem == "missing_force": del data[prefix + "contact_force"]
    elif problem == "fractional_count": data[prefix + "contact_count"][100] = .5
    elif problem == "negative_slip": data[prefix + "contact_slip"][100] = -.1
    else: del data[prefix + "nonwheel_ground_contact_count"]
    with pytest.raises((ValueError, KeyError)):
        analysis.compact_case(data, "PPO_mu0p700_isaac", require_nonwheel=True)


def manifest_fixture(tmp_path):
    def file(name, data=b"independent same-content capture token"):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        return bind(path, tmp_path)
    source = tmp_path / "evaluation_source"; source.mkdir()
    policies = {}
    for algorithm in ("PPO", "SAC"):
        folder, *_ = write_training(tmp_path, algorithm)
        roles = {"manifest": "manifest.json", "status": "run_status.json", "checkpoint": "model_final.pt",
            "runner_config": "runner_config.json", "policy_contract": "policy_contract.json"}
        if algorithm == "SAC": roles["actor"] = "actor_normalized.pt"
        policies[algorithm] = {role: bind(folder / name, tmp_path) for role, name in roles.items()}
    cases = []
    for algorithm in ("PPO", "SAC"):
        for mu in GRID:
            prefix = algorithm + "/raw/mu" + format(mu, ".3f").replace(".", "p")
            files = {role: file(prefix + "/" + role + ".bin") for role in analysis.FILE_ROLES}
            execution = {engine: {role: file(prefix + "/" + engine + "_" + role + ".bin")
                for role in ("receipt", "log")} for engine in ("isaac", "mujoco")}
            cases.append({"algorithm": algorithm, "mu": mu, "files": files, "execution": execution})
    return {"schema_version": 1, "kind": "sa29_policy_full_physical_grid_capture", "reference_id": SPEC["reference_id"],
        "reference_spec": copy.deepcopy(SPEC), "reference_catalog_sha256": digest(ROOT / "config/controller_references.json"),
        "evaluation_mu": GRID.copy(), "inputs": {role: file("inputs/" + role + ".bin") for role in ("motion", "robot_reference", "model")},
        "evaluation_source": {"root": "evaluation_source", "audit": file("evaluation_source/wide_friction_source.json", b"{}"),
            "source_sha256": {}}, "policies": policies, "cases": cases}


def test_manifest_requires_all_34_pairs_but_allows_independent_identical_content(tmp_path):
    manifest = manifest_fixture(tmp_path); before = copy.deepcopy(manifest)
    result = analysis.parse_manifest(manifest, tmp_path)
    assert len(result["cases"]) == 34 and manifest == before
    assert {(row["algorithm"], row["mu"]) for row in result["cases"]} == {(a, mu) for a in ("PPO", "SAC") for mu in GRID}


@pytest.mark.parametrize("problem", ["missing_pair", "duplicate_pair", "primary_reference", "wrong_grid", "unknown_algorithm",
    "missing_engine", "missing_role", "missing_receipt", "extra_field", "path_alias", "inode_alias"])
def test_manifest_rejects_partial_grid_relabeling_missing_execution_or_file_aliases(tmp_path, problem):
    manifest = manifest_fixture(tmp_path)
    if problem == "missing_pair": manifest["cases"].pop()
    elif problem == "duplicate_pair": manifest["cases"][-1] = copy.deepcopy(manifest["cases"][0])
    elif problem == "primary_reference": manifest["reference_id"] = "drift_drl_traj4"
    elif problem == "wrong_grid": manifest["evaluation_mu"] = GRID[:-1]
    elif problem == "unknown_algorithm": manifest["cases"][0]["algorithm"] = "NMPC"
    elif problem == "missing_engine": del manifest["cases"][0]["execution"]["mujoco"]
    elif problem == "missing_role": del manifest["cases"][0]["files"]["mujoco_report"]
    elif problem == "missing_receipt": del manifest["cases"][0]["execution"]["isaac"]["receipt"]
    elif problem == "extra_field": manifest["qualified"] = True
    elif problem == "path_alias": manifest["cases"][1]["files"]["isaac_trace"] = manifest["cases"][0]["files"]["isaac_trace"]
    else:
        original = tmp_path / manifest["cases"][0]["files"]["isaac_trace"]["path"]
        target = tmp_path / manifest["cases"][1]["files"]["isaac_trace"]["path"]
        target.unlink(); os.link(original, target)
    with pytest.raises(ValueError): analysis.parse_manifest(manifest, tmp_path)


def execution_fixture(tmp_path, engine):
    files = {}
    for role in analysis.FILE_ROLES:
        path = tmp_path / (role + ".bin"); path.write_bytes((role + " captured artifact").encode()); files[role] = path
    ia, mj = sa29_reports()
    ia.update(motion_file="/private/reference/motion.npz", checkpoint="/private/training/model_final.pt",
        runner_config="/private/training/runner_config.json", asset_path="/private/assets/WF_TRON1A.usda",
        runtime_device_audit={"requested_device": "cuda:2"})
    mj.update(motion_file=ia["motion_file"], policy="/private/evaluation/actor_normalized.pt",
        contract_source="/private/evaluation/policy_contract.json", model="/private/assets/model.xml",
        isaac_observation_validation={"status": "passed", "trajectory_sha256": digest(files["isaac_policy_trace"])})
    for name, value in (("isaac_report", ia), ("mujoco_report", mj)):
        files[name].write_text(json.dumps(value))
    command = ["/private/runtime/python", "/private/source/scripts/eval_tracking_" + ("friction_isaac.py" if engine == "isaac" else "mujoco.py"),
        "--motion-file", ia["motion_file"], "--output-dir", "/private/results/" + engine]
    if engine == "isaac":
        command += ["--checkpoint", ia["checkpoint"], "--runner-config", ia["runner_config"], "--asset-path", ia["asset_path"],
            "--num-envs", "1", "--seed", "42", "--evaluation-contact-friction", "0.7", "--device", "cuda:2",
            "--record-physics", "--verify-unmasked-device", "--headless"]
    else:
        command += ["--policy", mj["policy"], "--contract", mj["contract_source"], "--model", mj["model"],
            "--verify-isaac-trajectory", "/private/results/isaac/trajectory.npz", "--no-visual-mesh"]
    log = tmp_path / "engine.log"; log.write_text("Evaluator completed and all captured artifacts are present.\n")
    receipt = {"schema_version": 1, "kind": "sa29_policy_engine_terminal_receipt", "algorithm": "PPO", "mu": .7,
        "reference_id": SPEC["reference_id"], "engine": engine, "terminal_status": "exited", "exit_code": 0,
        "command": command, "log_sha256": digest(log), "requested_device": "cuda:2" if engine == "isaac" else "cpu",
        "artifact_sha256": {role: digest(path) for role, path in files.items() if role.startswith(engine + "_")}}
    receipt_file = tmp_path / "receipt.json"; receipt_file.write_text(json.dumps(receipt))
    execution = {"receipt": bind(receipt_file, tmp_path), "log": bind(log, tmp_path)}
    return execution, receipt, files, receipt_file, log


@pytest.mark.parametrize("engine", ["isaac", "mujoco"])
def test_execution_is_bound_to_original_terminal_receipt_log_and_artifact_bytes(tmp_path, engine):
    execution, _, files, _, _ = execution_fixture(tmp_path, engine)
    result = analysis.audit_execution(execution, "PPO", .7, engine, files, tmp_path)
    assert result["exit_code"] == 0 and result["traceback_error_scan_passed"] is True
    assert "/private/" not in json.dumps(result) and "command" not in result


@pytest.mark.parametrize("engine", ["isaac", "mujoco"])
@pytest.mark.parametrize("problem", ["nonzero", "bool_exit", "not_terminal", "wrong_reference", "wrong_mu", "wrong_engine",
    "wrong_log_hash", "wrong_artifact_hash", "traceback_exit_zero", "fatal_exit_zero", "cuda_exit_zero",
    "smoke", "wrong_motion", "wrong_entrypoint", "entrypoint_decoy", "wrong_device"])
def test_exit_zero_cannot_hide_traceback_wrong_episode_or_nonterminal_execution(tmp_path, engine, problem):
    execution, receipt, files, receipt_file, log = execution_fixture(tmp_path, engine)
    if problem == "nonzero": receipt["exit_code"] = 1
    elif problem == "bool_exit": receipt["exit_code"] = False
    elif problem == "not_terminal": receipt["terminal_status"] = "running"
    elif problem == "wrong_reference": receipt["reference_id"] = "drift_drl_traj4"
    elif problem == "wrong_mu": receipt["mu"] = .1
    elif problem == "wrong_engine": receipt["engine"] = "other"
    elif problem == "wrong_log_hash": receipt["log_sha256"] = "0" * 64
    elif problem == "wrong_artifact_hash": receipt["artifact_sha256"][engine + "_trace"] = "0" * 64
    elif problem in {"traceback_exit_zero", "fatal_exit_zero", "cuda_exit_zero"}:
        text = {"traceback_exit_zero": "Traceback (most recent call last):\n  failed()\nKeyboardInterrupt\n",
            "fatal_exit_zero": "Fatal Python error: interrupted\n", "cuda_exit_zero": "CUDA error: invalid device\n"}[problem]
        log.write_text(text); receipt["log_sha256"] = digest(log); execution["log"] = bind(log, tmp_path)
    elif problem == "smoke": receipt["command"].append("--smoke-only")
    elif problem == "wrong_motion": receipt["command"][receipt["command"].index("--motion-file") + 1] = "/private/primary/motion.npz"
    elif problem == "wrong_entrypoint": receipt["command"][1] = "/private/source/scripts/other.py"
    elif problem == "entrypoint_decoy": receipt["command"].insert(1, "/private/evil.py")
    else: receipt["requested_device"] = "cpu" if engine == "isaac" else "cuda:2"
    receipt_file.write_text(json.dumps(receipt)); execution["receipt"] = bind(receipt_file, tmp_path)
    with pytest.raises(ValueError): analysis.audit_execution(execution, "PPO", .7, engine, files, tmp_path)


@pytest.mark.parametrize("problem", [None, "observation", "action_phase", "samples", "dt"])
def test_policy_rate_observation_capture_matches_the_same_physical_episode(problem):
    physical = policy_raw("isaac")
    policy = {name: value[::4].copy() if value.ndim and len(value) == SAMPLES else value.copy()
        for name, value in physical.items()}
    if problem == "observation": policy["action_observation"][30, 0, 0] += .1
    elif problem == "action_phase": policy["reference_frame_for_action"][30] += 1
    elif problem == "samples": policy["time_s"] = policy["time_s"][:-1]
    elif problem == "dt": policy["policy_dt_s"] = np.array(.005)
    if problem is None: analysis.audit_policy_trajectory(policy, physical)
    else:
        with pytest.raises(ValueError): analysis.audit_policy_trajectory(policy, physical)


@pytest.mark.parametrize("problem", [None, "source_bytes", "original_pin", "controller_capability"])
def test_real_pure_materializer_closure_is_bound_without_changing_original_sources(tmp_path, problem):
    from materialize_controller_source import materialize
    original = tmp_path / "packaged_original"
    before = {}
    for name in analysis.BASE_SOURCE_SHA256:
        source = ROOT / "upstream/tron1-learning-from-human" / name
        before[source] = source.read_bytes()
        target = original / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(before[source])
    generated = tmp_path / "evaluation"
    audit = materialize(original, generated)
    path = generated / "wide_friction_source.json"
    expected, _ = analysis.expected_evaluation_sources()
    if problem == "source_bytes": (generated / "training/tron1_tracking.py").write_bytes(b"changed runtime code")
    elif problem == "original_pin": audit["source_hashes"]["training/tron1_drift.py"] = "0" * 64
    elif problem == "controller_capability": audit["controller_algorithms"] = ["PPO"]
    if problem in {"original_pin", "controller_capability"}: path.write_text(json.dumps(audit))
    if problem is None:
        result = analysis.audit_source(generated, bind(path, tmp_path), expected, tmp_path)
        assert "/tmp/" not in json.dumps(result) and "5eb40" in result["source_difference_disclosure"]
    else:
        with pytest.raises(ValueError): analysis.audit_source(generated, bind(path, tmp_path), expected, tmp_path)
    assert all(path.read_bytes() == data for path, data in before.items())


def publication_stub():
    """Synthetic IO-only shape; never a physical experiment or saved result."""
    return {"complete_predeclared_grid": True, "paired_policy_cases": 34, "physical_engine_episodes": 68,
        "reference_spec": SPEC, "input_sha256": {"motion": "1" * 64, "robot_reference": "2" * 64, "model": "3" * 64}}


def test_atomic_publication_stages_three_files_and_calls_semantic_replay_before_rename(tmp_path, monkeypatch):
    output = tmp_path / "publication"; checked = []
    def render(compact, summary, path): path.write_bytes(b"synthetic IO-only raster token")
    def replay(stage, *unused):
        assert stage != output and not output.exists()
        assert {path.name for path in stage.iterdir()} == {"summary.json", "policy_dynamics_trace.npz", "topdown.png"}
        checked.append(True)
    monkeypatch.setattr(analysis, "render_topdown", render)
    monkeypatch.setattr(analysis, "load_public_bundle", replay)
    result = analysis.publish_bundle(publication_stub(), {"schema_version": np.array(1)}, output, {})
    assert checked == [True] and output.is_dir()
    assert {path.name for path in output.iterdir()} == {"summary.json", "policy_dynamics_trace.npz", "topdown.png"}
    assert all(digest(output / item["file"]) == item["sha256"] for item in result["artifacts"].values())


@pytest.mark.parametrize("problem", ["partial", "private", "render_failure", "replay_failure", "target_appeared"])
def test_failed_or_incomplete_publication_never_exposes_a_partial_or_overwritten_target(tmp_path, monkeypatch, problem):
    output = tmp_path / "publication"; summary = publication_stub()
    if problem == "partial": summary["paired_policy_cases"] = 33
    elif problem == "private": summary["private_path"] = "/private/credentials/token"
    def render(compact, unused, path):
        if problem == "render_failure": raise RuntimeError("synthetic render failure")
        path.write_bytes(b"synthetic raster")
        if problem == "target_appeared":
            output.mkdir(); (output / "user-owned").write_bytes(b"preserve this unrelated output")
    def replay(*unused):
        if problem == "replay_failure": raise ValueError("synthetic semantic replay failure")
    monkeypatch.setattr(analysis, "render_topdown", render)
    monkeypatch.setattr(analysis, "load_public_bundle", replay)
    with pytest.raises((ValueError, RuntimeError)):
        analysis.publish_bundle(summary, {"schema_version": np.array(1)}, output, {})
    if problem == "target_appeared":
        assert (output / "user-owned").read_bytes() == b"preserve this unrelated output"
        assert {path.name for path in output.iterdir()} == {"user-owned"}
    else: assert not output.exists()


def test_run_and_publish_preserve_preexisting_destinations_before_reading_any_capture(tmp_path):
    (tmp_path / "sentinel").write_bytes(b"existing user file")
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.run(SimpleNamespace(output_dir=tmp_path))
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.publish_bundle(publication_stub(), {}, tmp_path, {})
    assert (tmp_path / "sentinel").read_bytes() == b"existing user file"
