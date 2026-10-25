"""Offline audits of formal N: K margin cost1 plus binary any-gap cost3.

No actors, checkpoint pickles, physics, FK models or rendering are loaded. A
copied parent verification proof binds each completed report/recording; formal
training artifacts were verified on the execution host, not rehashed locally.
The old same-engine PhysX-trained beta-ON baselines are separately SHA-pinned.
Numerical episode screens never claim the broader task or visual review done.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as metrics
import analyze_ppo_ground_contact_v2 as original
from analyze_cem_ground_contact_v2 import support_summary
from controller_trace import native_trace
import eval_ppo_ground_contact_v2n as evaluator
import ground_contact_reward_v2n as definition
from ground_contact_beta_contract import BETA_SOURCE_SHA256, verify_beta_runtime
from ppo_backend_400hz_contract import (INPUT_PINS, read_json, require, sha256,
                                       verify_protocol, verify_recording, write_json)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config/ppo_ground_contact_v2n_400hz_v1.json"
PROTOCOL_SHA256 = "370237b187ece94f6a95d6a1997fef95418ea3cb07df251ddf73e4495cfd3383"
DEFAULT_INPUT_ROOT = ROOT / "outputs/ground_contact_v2n_20261025_v1/ppo_nominal_raw_v1"
DEFAULT_TRAINING_AUDIT = ROOT / "outputs/ground_contact_v2n_20261025_v1/ppo_training_proof_v1/ground_contact_v2n_training_audit.json"
TRAINING_SOURCE_PINS = {
    "ground_contact_reward_v2g.py": "81056b23fbf2052f2e0630d1888aa6b71556027ca108ab3b08dd768a912e493a",
    "ground_contact_reward_v2.py": "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5",
    "ground_contact_reward_v2b.py": "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c",
    "ground_contact_reward_v2c.py": "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb",
    "ground_contact_reward_v2d.py": "1f6245ca3ca08483704d4f42a044ba38b8d09f01d81dc6e71bbf423d206898a7",
    "ground_contact_reward_v2f.py": "a9e561a09ca2f0cbcd6c412a8e1860817aeccaff50660e9796866996a844d3e1",
    "ground_contact_reward_v2n.py": "e2b3fb0a401e9f3b92740e4732a7df14f784b02fae205922062bfe90f465ca89",
    "ground_contact_beta_contract.py": "6168e4369ffaae16f06fe375bec78813d8a1e7854d20f4e6ff8c073e1233d126",
    "ppo_ground_contact_v2n.py": "7ce3edd7ec888db675824fcb4b1225697b8704155282edc563aa904b3eacdaa8",
    "train_ppo_ground_contact_v2n.py": "436be3bd1a4f3e66ca1bdd5985846b85f01aa54503b49bda9452c2a978295000",
    "train_ppo_backend_400hz.py": "5234070be219e59c61e6c4e4435e0dd0b2ab71df60d9c23886e507ed73692390",
    "train_ppo_backend.py": "65367115de9b5608dc2ef57cab701bea31c152c6af576bdb43a5cef025cc6878",
    "ground_contact_reward_v2i.py": "2d805c5d878541da7a7ccc1f6af421ebe4b425fb9006756081f786ae79bf5101",
    "ground_contact_reward_v2k.py": "dbea1148c00a394b990964f7e7b26d3a7e9bdbe75e43975b3e9e79e7e0ebe18f"
}
IMPORT_SOURCE_PINS = {
    "eval_ppo_ground_contact_v2n.py": "0bfe76419af65bcc9caf14fff8e7d5484006eb34aea109722c4d7cc819b325c8",
    "analyze_ppo_ground_contact_v2.py": "f7edea803cfe409bc049b6447aef02577b2fa0d579013ecadcf90c307cf6abcd",
    "analyze_cem_ground_contact_v2.py": "540f85cb960669d30ef212a4d1ab68c0c1c6abe8508505ac8103dedbe3b1a0ed",
    "analyze_ppo_backend_400hz_matrix.py": "2645c26150b4bbca5a5631beee3d1cf1178ee365bf0a11044d7e8c618220c0a1",
    "analyze_ppo_ground_support_failure.py": "cc0acd232a240618cd55aa5de8fcad0817b71696e0f9f285db4bd11c679c5219",
    "controller_trace.py": "673d17bda4d9f6c93b76b0b963b199aef3e3abe8982cfc6cc6580184e5b0a3e6",
    "controller_metrics.py": "40921dad0c268d4bd871e0910086957555116fdd07188ba7cb4f8c76ca4343de",
    "nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
    "ppo_backend_400hz_contract.py": "b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e",
    "eval_ppo_backend_400hz_mujoco.py": "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4",
    "eval_ppo_backend_400hz_isaac.py": "e670d6c88bd85eb8aa36f402301e7f0c9ae5f7d3b97336a78d1a66750a642f99"
}
DEFAULT_REFERENCE = ROOT.parent / "tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz"
BASELINE_ROOT = ROOT / "outputs/ppo_backend_400hz_nominal_20261022_v1"
BASELINE_SUMMARY = ROOT / "results/ppo_backend_400hz_nominal_20261022_v1/summary.json"
BASELINE_SUMMARY_SHA256 = "6dcc65fb2b863d856190e452bcea744dfeb2bdd253d1d31389cb51146fcac5ac"
BASELINE_CHECKPOINT = "c508dc3ba50db81f56f538bf23c550aca12f3279ddd15749d60582a487d2d8c7"
BASELINE_ACTOR = "ab3c486ef779deb0cff52bfb33415f5c3a779381740a2c12898506e4e80f08bc"
BACKENDS = {"physx": "PhysX", "mujoco": "native CPU MuJoCo"}
WINDOWS = {"core": (1.44, 4.), "whole_reference_overlap": (0., 6.96), "full_native": (0., 6.98)}
TRACE_FIELDS = ("time_s", "position", "quaternion", "com_velocity", "angular_velocity",
                "joint_pos", "joint_vel", "wheel_pos", "contact_count", "contact_force", "contact_slip")


def load_protocol(path=DEFAULT_PROTOCOL):
    verify_import_sources()
    require(sha256(path) == PROTOCOL_SHA256, "Frozen V2n protocol/acceptance bytes changed")
    protocol = verify_protocol(path)
    require(protocol["cohort_id"] == evaluator.COHORT_ID
            and protocol["reward_definition_version"] == evaluator.REWARD_VERSION,
            "Wrong declared new any-wheel cohort")
    definition.validate_settings(protocol["ground_contact_reward_v2"])
    return protocol


def verify_import_sources():
    """Bind frozen A calculations and the A/B/C/D/F/G/N reward chain; never execute actors."""
    pins = {**TRAINING_SOURCE_PINS, **IMPORT_SOURCE_PINS}
    for name, digest in pins.items():
        require(sha256(ROOT/"scripts"/name) == digest,
                "Frozen imported A/B/C/D/F/G/N source differs: " + name)
    definition.verify_base()
    return copy.deepcopy(pins)


def load_training_audit(path, protocol_path=DEFAULT_PROTOCOL):
    """Bind copied formal training metadata, not local checkpoint re-hashing."""
    path = _file(Path(path).parent, Path(path).name)
    value = read_json(path)
    protocol = load_protocol(protocol_path)
    expected = {"status": "completed_verified", "cohort_id": evaluator.COHORT_ID,
        "reward_definition_version": evaluator.REWARD_VERSION, "protocol_sha256": sha256(protocol_path),
        "training_environment_steps": 12288000, "normalizer_in_actor": True,
        "from_scratch": True, "smoke_only": False, "trained_policy_claimed": True,
        "other14_reward_terms_changed": False, "new_reward_count": 15,
        "settings": protocol["ground_contact_reward_v2"]}
    expected["initial_policy_state_sha256"] = evaluator.EXPECTED_INITIAL_POLICY_SHA256
    for name, expected_value in expected.items():
        require(value.get(name) == expected_value, "Copied N formal training audit differs: " + name)
    for name in ("checkpoint_sha256", "exported_policy_sha256", "manifest_sha256",
                 "policy_contract_sha256", "initial_policy_state_sha256"):
        _digest(value.get(name), name)
    ledger = value.get("source_sha256")
    require(isinstance(ledger, dict) and ledger, "Missing N formal training source ledger")
    for filename, digest in ledger.items():
        require(isinstance(filename, str) and Path(filename).is_absolute(), "Training source path is not absolute")
        _digest(digest, filename)
    for basename, digest in TRAINING_SOURCE_PINS.items():
        entries = [value for name, value in ledger.items() if Path(name).name == basename]
        require(entries == [digest], "Copied A/B/C/D/F/G/N training source ledger differs: " + basename)
    beta_entries = [(name, digest) for name, digest in ledger.items() if Path(name).name == "tron1_drift.py"]
    require(len(beta_entries) == 1 and beta_entries[0][1] == BETA_SOURCE_SHA256,
            "Copied N executed existing beta source ledger differs")
    runtime = value.get("contact_runtime_audit", {})
    expected_runtime = {"reward_definition_version": evaluator.REWARD_VERSION,
        "settings": protocol["ground_contact_reward_v2"], "physics_dt_s": .0025,
        "policy_dt_s": .02, "substeps": 8, "steps_started": 24008, "steps_completed": 24008,
        "physics_samples": 192064, "unique_reward_windows": 24008,
        "device_finite_normal_force_health": True, "device_finite_reward_health": True,
        "cpu_fixture_only": False, "single_wheel_support_allowed": True, "bilateral_constraint": False,
        "height_or_zero_slip_target": False, "observation_action_or_physics_change": False,
        "module_sha256": TRAINING_SOURCE_PINS["ppo_ground_contact_v2n.py"],
        "definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2n.py"],
        "immutable_v2a_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2.py"],
        "immutable_v2b_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2b.py"],
        "immutable_v2c_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2c.py"],
        "immutable_v2d_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2d.py"],
        "immutable_v2f_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2f.py"],
        "immutable_v2g_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2g.py"],
        "immutable_v2i_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2i.py"],
        "immutable_v2k_definition_sha256": TRAINING_SOURCE_PINS["ground_contact_reward_v2k.py"],
        "support_gap_coefficient_previous": 1., "support_gap_coefficient_current": 1., "binary_any_gap_coefficient": 3.,
        "hybrid_margin_plus_binary_gap": True,
        "force_slew_cost_added": False,
        "support_gap_aggregation": "max8", "support_gap_is_window_event_not_duration_fraction": False,
        "support_gap_is_continuous_load_margin": True,
        "support_gap_cost_mode": "worst_substep_margin_plus_binary_anygap"}
    for name, expected_value in expected_runtime.items():
        require(runtime.get(name) == expected_value, "Copied N contact runtime audit differs: " + name)
    target = runtime.get("load_margin_target_range_n")
    mass = runtime.get("mass_kg_range")
    gravity = runtime.get("load_margin_gravity_w_m_s2")
    require(isinstance(target, list) and len(target) == 2
            and isinstance(mass, list) and len(mass) == 2
            and isinstance(gravity, list) and len(gravity) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in [*target, *mass, *gravity])
            and gravity[:2] == [0., 0.] and gravity[2] < 0.
            and 5. < target[0] <= target[1] and 0. < mass[0] <= mass[1]
            and all(math.isclose(t, m*abs(gravity[2])*.5, rel_tol=2.e-6, abs_tol=2.e-5)
                    for t, m in zip(target, mass, strict=True)),
            "N actual mass/gravity load margin readback differs")
    verify_beta_runtime(runtime, expected_envs=512, expected_windows=24008)
    require(runtime.get("beta_quality_source_path") == beta_entries[0][0],
            "Copied N beta reference source differs from executed source ledger")
    bearing = runtime.get("force_bearing_samples_L_R_any_both")
    require(isinstance(bearing, list) and len(bearing) == 4
            and all(type(n) is int and n >= 0 for n in bearing)
            and bearing[2] > 0 and bearing[3] <= min(bearing[0], bearing[1])
            and bearing[2]+bearing[3] == bearing[0]+bearing[1], "Invalid copied force-bearing evidence")
    return value, {"path": str(path), "sha256": sha256(path), "source_sha256": ledger,
        "imported_A_B_C_D_F_G_I_K_L_sources_sha256": verify_import_sources(),
        "checkpoint_bytes_locally_rehashed": False,
        "execution_host_verification_metadata_only": True}


def _digest(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value), "Invalid SHA256: " + label)
    return value


def _file(folder, name):
    path = Path(folder).resolve() / name
    require(path.is_file() and not path.is_symlink(), "Missing/indirect completed artifact: " + str(path))
    return path


# Use the frozen A metric and acceptance implementations, not a re-tuned copy.
_clock = original._clock
_root_signals = original._root_signals
analyze_trace = original.analyze_trace
recorded_metrics = original.recorded_metrics
nominal_screen = original.nominal_screen


def _check_new_report(report, proof, engine, protocol_path, raw_sha, report_sha,
                      training_audit, training_evidence):
    protocol = load_protocol(protocol_path)
    audit = report.get("ground_contact_v2n_evaluation", {})
    expected = {"kind": evaluator.KIND, "cohort_id": evaluator.COHORT_ID,
        "reward_definition_version": evaluator.REWARD_VERSION, "training_backend": "physx",
        "smoke_only": False, "trained_policy_claimed": True, "additional_training_updates": 0,
        "status": "verified", "validation_backend": BACKENDS[engine], "protocol_sha256": sha256(protocol_path),
        "wrapper_sha256": sha256(evaluator.__file__), "normalizer_included_in_actor": True,
        "normalization_applications_per_inference": 1, "extra_normalization": False,
        "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "initial_policy_state_sha256": evaluator.EXPECTED_INITIAL_POLICY_SHA256,
        "recording_sha256": raw_sha, "single_wheel_support_allowed": True,
        "bilateral_support_required": False, "absolute_height_gate": False, "airborne_failure": False}
    for key, value in expected.items():
        require(audit.get(key) == value, "NEW formal evaluator provenance differs: " + key)
    actor = _digest(audit.get("normalized_actor_sha256"), "new normalized actor")
    checkpoint = _digest(audit.get("checkpoint_sha256"), "new checkpoint")
    require(actor != BASELINE_ACTOR and checkpoint != BASELINE_CHECKPOINT,
            "Do not relabel the old baseline actor as the NEW contact experiment")
    for key in ("legacy_evaluator_sha256", "generated_source_sha256"):
        _digest(audit.get(key), key)
    ledger = audit.get("cohort_input_sha256", {})
    require(isinstance(ledger, dict) and ledger.get("model_final.pt") == checkpoint
            and ledger.get("policy.pt") == actor
            and {"ground_contact_v2n_training_audit.json", "cohort_completion_audit.json", "manifest.json",
                 "policy_contract.json", "final_policy_contract.json", "run_status.json"} <= ledger.keys(),
            "Missing remotely verified formal training/export metadata bindings")
    for name, value in ledger.items(): _digest(value, name)
    require(ledger["ground_contact_v2n_training_audit.json"] == training_evidence["sha256"]
            and checkpoint == training_audit["checkpoint_sha256"]
            and actor == training_audit["exported_policy_sha256"]
            and ledger["manifest.json"] == training_audit["manifest_sha256"]
            and ledger["policy_contract.json"] == training_audit["policy_contract_sha256"],
            "F report is not bound to the copied formal training/source audit")
    expected_proof = {"status": "verified", "native_states": 2793, "policy_decisions": 349,
        "duration_s": 6.98, "physics_dt_s": .0025, "policy_dt_s": .02,
        "recording_sha256": raw_sha, "validation_backend": BACKENDS[engine],
        "checkpoint_sha256": checkpoint, "report_sha256": report_sha, "evaluation_mu": .7}
    for key, value in expected_proof.items():
        require(proof.get(key) == value, "Copied parent verification proof differs: " + key)
    contract = report.get("policy_export" if engine == "physx" else "contract", {})
    for key, value in {"physics_dt": .0025, "decimation": 8, "policy_fps": 50.,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8, "leg_kp": 500., "leg_kd": 10.,
        "full_horizon_diagnostic": True, "actor_includes_observation_normalizer": True,
        "training_environment_transitions": 12288000, "reward_definition_version": evaluator.REWARD_VERSION}.items():
        require(contract.get(key) == value, "New formal inference contract differs: " + key)
    profile, trained = contract.get("task_profile", {}), contract.get("trained_task_profile", {})
    compare = copy.deepcopy(profile); compare["contact_friction"] = 1.
    require(compare == trained and profile.get("contact_friction") == .7
            and profile.get("ground_contact_reward_v2") == protocol["ground_contact_reward_v2"]
            and profile.get("reward_weights", {}).get("sideslip") == 2.
            and profile.get("reward_weights", {}).get("grounded") == .4
            and profile.get("action_rate_weight") == -.5
            and profile.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
            and contract.get("training_domain_randomization") == protocol["training_randomization"],
            "New deployed/trained reward, DR, beta or height profile differs")
    require(contract.get("evaluation_contact_friction_override", {}).get("evaluation_contact_friction") == .7,
            "Wrong nominal evaluation coefficient")
    material = report.get("contact_material_audit", {})
    if engine == "physx":
        gpu = audit.get("physical_gpu_index_requested")
        require(report.get("status") == "evaluated" and report.get("deterministic_start_frame") == 0
                and report.get("domain_randomization_enabled") is False and report.get("observation_noise_enabled") is False
                and report.get("terminal_states_captured_before_auto_reset") is True
                and report.get("runtime_device_audit", {}).get("status") == "verified"
                and type(gpu) is int and gpu in (3, 4, 5)
                and report["runtime_device_audit"].get("requested_device") == "cuda:"+str(gpu)
                and report.get("motion_file_sha256") == INPUT_PINS["motion"], "PhysX lifecycle/reference differs")
        _digest(audit.get("deployment_actor_sha256"), "remotely verified PhysX deployment actor")
        require(material.get("live_physx_materials_verified") is True and material.get("ground_material_usd_verified") is True
                and material.get("recipe", {}).get("training") is False
                and material.get("recipe", {}).get("wheel_contact_friction", {}).get("range") == [.7, .7]
                and material["recipe"]["wheel_contact_friction"].get("physx_ground_friction") == 1.
                and material.get("encoder_bias_range_rad") == [0., 0.], "PhysX fixed native materials differ")
    else:
        require(report.get("status") == "completed" and report.get("completed_full_reference") is True
                and report.get("policy_steps") == 349 and report.get("physics_steps") == 2792
                and report.get("hidden_resets") == 0 and report.get("root_state_writes") == 1
                and report.get("start_reference_frame") == 0
                and report.get("domain_parameters_resampled_during_episode") is False
                and report.get("policy_sha256") == actor and report.get("motion_sha256") == INPUT_PINS["motion"]
                and report.get("model_sha256") == INPUT_PINS["model"], "MuJoCo lifecycle/reference/model differs")
        require(material.get("native_mapping_verified") is True and material.get("explicit_contact_pair_count") == 0
                and material.get("pair_coefficient") == .7 and material.get("compiled_native_timestep_s") == .0025,
                "MuJoCo native material coefficient/clock differs")
        friction = np.asarray(material.get("geom_friction"), dtype=float)
        require(friction.ndim == 2 and friction.shape[1] == 3 and len(friction) > 0
                and np.isfinite(friction).all() and np.all(friction[:, 0] == .7), "MuJoCo compiled sliding coefficients differ")
        priorities = dict(zip(material.get("colliding_geom_ids", []), material.get("geom_priority", [])))
        wheels = material.get("wheel_geom_ids", [])
        require(priorities and wheels and all(priorities.get(index) == 1 for index in wheels)
                and all(value == (1 if index in wheels else 0) for index, value in priorities.items()),
                "MuJoCo effective wheel material priority differs")
    return contract, audit


def load_completed(folder, engine, protocol_path=DEFAULT_PROTOCOL, *, training_audit=None):
    """Only a completed NEW formal episode; smoke/old/missing proofs fail closed."""
    require(engine in BACKENDS, "Select physx or mujoco explicitly")
    raw_name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
    paths = {name: _file(folder, name) for name in ("report.json", raw_name, "remote_verification.json")}
    hashes = {name: sha256(path) for name, path in paths.items()}
    report, proof = read_json(paths["report.json"]), read_json(paths["remote_verification.json"])
    training_path = training_audit if training_audit is not None else Path(folder)/"ground_contact_v2n_training_audit.json"
    trained, training_evidence = load_training_audit(training_path, protocol_path)
    contract, audit = _check_new_report(report, proof, engine, protocol_path,
        hashes[raw_name], hashes["report.json"], trained, training_evidence)
    verify_recording(paths[raw_name], BACKENDS[engine])
    with np.load(paths[raw_name], allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    native_trace(raw, "policy_isaac" if engine == "physx" else "policy_mujoco")
    require(all(sha256(paths[name]) == value for name, value in hashes.items()),
            "Completed candidate files changed during offline analysis")
    require(sha256(training_evidence["path"]) == training_evidence["sha256"],
            "Copied formal training audit changed during offline analysis")
    return raw, {"input_directory": str(Path(folder).resolve()), "input_sha256": hashes,
        "checkpoint_sha256": audit["checkpoint_sha256"], "actor_sha256": audit["normalized_actor_sha256"],
        "contract": contract, "evaluation_proof": proof,
        "formal_training_source_evidence": training_evidence,
        "training_artifacts_verified_on_execution_host_not_locally_rehashed": True}


load_baseline = original.load_baseline
_compare_profiles = original._compare_profiles
compact_arrays = original.compact_arrays
recompute_compact = original.recompute_compact


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing analysis; use a fresh output directory")
    protocol = load_protocol(args.protocol)
    reference = metrics.reference_data(args.reference)
    folders = {engine: getattr(args, engine+"_dir", None) for engine in BACKENDS}
    require(not getattr(args, "input_root", None) or not any(value is not None for value in folders.values()),
            "Choose input-root or explicit engine directories, not ambiguous mixed inputs")
    if not any(value is not None for value in folders.values()):
        input_root = Path(getattr(args, "input_root", None) or DEFAULT_INPUT_ROOT)
        folders = {engine: input_root/engine for engine in BACKENDS}
    require(any(value is not None for value in folders.values()), "At least one verified NEW completed PPO evaluation is required")
    rows, traces = [], {}
    for engine, folder in folders.items():
        if folder is None: continue
        raw, evidence = load_completed(folder, engine, args.protocol,
            training_audit=getattr(args, "training_audit", None))
        old_raw, baseline_evidence = load_baseline(engine, args.baseline_root, args.baseline_summary)
        _compare_profiles(evidence["contract"], baseline_evidence["contract"])
        candidate, trace = recorded_metrics(raw, engine, reference)
        baseline, old_trace = recorded_metrics(old_raw, engine, reference)
        for name, original in (("core", "core"), ("whole_reference_overlap", "reference_overlap"), ("full_native", "full")):
            require(baseline["windows"][name] == baseline_evidence["published_row"]["windows"][original],
                    "Recomputed original same-engine baseline metrics differ from published result")
        initial = {key: float(np.max(np.abs(trace[key][0]-old_trace[key][0])))
                   for key in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")}
        traces["new_"+engine], traces["baseline_"+engine] = trace, old_trace
        rows.append({"id": "physx_trained_to_"+engine, "validation_backend": BACKENDS[engine], "sample_size": 1,
            "new_V2": candidate, "original_grounded_baseline": baseline,
            "nominal_screen": nominal_screen(candidate, baseline, protocol, engine),
            "input_evidence": evidence, "baseline_evidence": baseline_evidence,
            "initial_recorded_state_max_abs_difference": initial})
    if len(rows) == 2:
        require(rows[0]["input_evidence"]["actor_sha256"] == rows[1]["input_evidence"]["actor_sha256"]
                and rows[0]["input_evidence"]["checkpoint_sha256"] == rows[1]["input_evidence"]["checkpoint_sha256"],
                "Both physics backends must evaluate the SAME newly trained PPO actor")
    summary = {"schema_version": 1, "kind": "PPO_ground_contact_V2n_nominal_offline_audit",
        "status": "completed_analysis", "protocol_sha256": sha256(args.protocol), "reference_sha256": sha256(args.reference),
        "analyzer_sha256": sha256(__file__), "rows": rows, "expected_validation_engines": list(BACKENDS),
        "cohort_id": evaluator.COHORT_ID, "reward_definition_version": evaluator.REWARD_VERSION,
        "imported_A_B_C_D_F_G_I_K_L_source_sha256": verify_import_sources(),
        "completed_validation_engines": [engine for engine, folder in folders.items() if folder is not None],
        "both_PPO_validation_backends_complete": len(rows) == 2, "sample_size_per_case": 1,
        "physics_dt_s": .0025, "policy_dt_s": .02, "native_states_per_case": 2793,
        "new_dynamics_inference_training_or_FK": False, "height_or_flight_failure_gate": False,
        "bilateral_support_required": False, "qualitative_whole_body_review_complete": False,
        "overall_task_goal_complete_claimed": False,
        "limitations": ["N=1 per actor/engine: not statistical, hardware, or universal-friction qualification.",
            "Formal new training/export evidence is remotely parent-verified; weights are not fetched or executed here.",
            "Any-wheel support uses actual summed floor-normal Fz>5N, not total body net force or legacy thresholded point count.",
            "A single-wheel lift and alternating loaded wheels are allowed; no bilateral criterion is added.",
            "All physical samples stay on their original400Hz clocks; no alignment, smoothing, resets or state correction.",
            "The reference ends6.96s. Tracking over full native0–6.98s remains unavailable; no endpoint extrapolation.",
            "Reference-relative beta errors disclose valid samples; nonzero beta is intended, not a mismatch or zero-slip target.",
            "Contact, tilt and nonwheel-contact diagnostics do not change the independent XY failure metric.",
            "Passing a numerical episode screen does not complete the broader PPO/CEM comparison or qualitative whole-body review."]}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_states.npz", **compact_arrays(traces, reference))
    fields = ("id", "core_xy_rmse_m", "whole_xy_rmse_m", "whole_xy_peak_m", "core_speed_retention",
              "core_beta_rmse_deg", "core_beta_valid_samples", "any_support_sample_fraction",
              "unsupported_samples", "longest_gap_s", "core_HF_ratio", "numeric_episode_pass")
    with (output/"metric_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows:
            core = row["new_V2"]["windows"]["core"]["tracking"]
            whole = row["new_V2"]["windows"]["whole_reference_overlap"]["tracking"]
            support = row["new_V2"]["any_wheel_ground_support"]["full_contact"]
            writer.writerow(dict(zip(fields, (row["id"], core["xy_rmse_m"], whole["xy_rmse_m"], whole["xy_peak_m"],
                core["speed_retention"], core["beta_rmse_deg"], core["beta_valid_samples"],
                support["any_wheel_support_sample_fraction"], support["unsupported_physical_samples"],
                support["longest_no_support_observed_s"], row["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"],
                row["nominal_screen"]["numeric_nominal_episode_pass"]))))
    summary["artifact_sha256"] = {name: sha256(output/name) for name in ("native_states.npz", "metric_table.csv")}
    write_json(output/"summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--physx-dir", type=Path)
    result.add_argument("--mujoco-dir", type=Path)
    result.add_argument("--input-root", type=Path, help="Directory containing physx/ and mujoco/; default is the N nominal download")
    result.add_argument("--training-audit", type=Path, default=DEFAULT_TRAINING_AUDIT,
                        help="Copied formal N training audit, hash-bound to both remotely verified reports")
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    result.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
