"""Synthetic provenance/recording fixtures; no training, simulation or GPU."""
import copy
from q_geometry_fixture import geometry_audit_fixture
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ppo_ground_contact_v2q as module
import ground_contact_reward_v2q as definition
from ground_contact_beta_contract import BETA_SOURCE_SHA256, BETA_QUALITY_SOURCE


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def make_training(tmp_path, *, smoke=False):
    directory = tmp_path / "training"
    directory.mkdir()
    protocol = json.loads((ROOT / "config/ppo_backend_400hz_comparison_v1.json").read_text())
    protocol.update(cohort_id=module.COHORT_ID, ground_contact_reward_v2=definition.validate_settings(),
                    reward_definition_version=module.REWARD_VERSION)
    protocol_path = tmp_path / "protocol.json"
    put(protocol_path, protocol)
    profile = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    profile.update(action_rate_weight=-.5, ground_contact_reward_v2=definition.validate_settings())
    contract = json.loads((ROOT / "config/nominal_actuator_contract.json").read_text())
    contract.pop("evaluation_contact_friction_override", None)
    (directory / "model_final.pt").write_bytes(b"synthetic checkpoint; never unpickled")
    (directory / "policy.pt").write_bytes(b"synthetic actor bytes; never executed")
    flags = {"smoke_only": smoke, "trained_policy_claimed": not smoke}
    mode, budget, iterations, num_envs = ("smoke", 1536, 2, 32) if smoke else ("formal", 12288000, 1000, 512)
    actor, checkpoint = module.sha256(directory / "policy.pt"), module.sha256(directory / "model_final.pt")
    contract.update(contract_status="smoke_completed_normalized_actor_not_formal_policy" if smoke else "trained_completed_normalized_actor",
        physics_dt=.0025, decimation=8, policy_fps=50., task_profile=profile, trained_task_profile=profile,
        training_domain_randomization=protocol["training_randomization"], policy_training_backend="physx",
        policy_algorithm="PPO", actor_includes_observation_normalizer=True, actor_output_is_unclipped=True,
        actor_observation_groups=["policy"], training_environment_transitions=budget,
        reference_file_sha256=module.common.INPUT_PINS["motion"], checkpoint_sha256=checkpoint,
        exported_policy_sha256=actor, run_kind=mode, export_validation_max_abs_error=0., **flags)
    manifest = {"contract": contract, "task_profile": profile, "backend": "physx", "seed": 42,
        "versions": {"rsl-rl-lib": "3.0.1"}, "transitions_requested": budget,
        "num_envs": num_envs, "iterations": iterations, "initial_policy_state_sha256": module.EXPECTED_INITIAL_POLICY_SHA256,
        "inputs": {name: {"sha256": module.common.INPUT_PINS[name]} for name in ("motion", "model", "asset")}, **flags}
    status = {"status": "training_completed", "training_environment_steps": budget, "run_kind": mode,
        "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor,
        "initial_policy_state_sha256": module.EXPECTED_INITIAL_POLICY_SHA256, **flags}
    for name, value in (("manifest.json", manifest), ("policy_contract.json", contract),
                        ("final_policy_contract.json", contract), ("run_status.json", status)):
        put(directory / name, value)
    trainer = tmp_path / "train_ppo_ground_contact_v2q.py"
    trainer.write_text("# synthetic training source fixture, not executable policy evidence\n")
    source_paths = [ROOT / "scripts/ground_contact_reward_v2.py", ROOT / "scripts/ground_contact_reward_v2b.py", ROOT / "scripts/ground_contact_reward_v2c.py", ROOT / "scripts/ground_contact_reward_v2d.py", ROOT / "scripts/ground_contact_reward_v2f.py",
                    ROOT / "scripts/ground_contact_reward_v2g.py",
                    ROOT / "scripts/ground_contact_reward_v2i.py", ROOT / "scripts/ground_contact_reward_v2q.py", ROOT / "scripts/ground_contact_beta_contract.py",
                    ROOT / "upstream/tron1-learning-from-human/training/tron1_drift.py",
                    ROOT / "scripts/ppo_ground_contact_v2q.py",
                    *[ROOT/"scripts"/n for n in ("ground_contact_reward_v2k.py", "ppo_ground_contact_v2k.py",
                      "wheel_ground_geometry.py", "wheel_ground_geometry_torch.py", "tron1_wheel_geometry.py")], trainer]
    windows = iterations * 24 + 8
    runtime = {"reward_definition_version": module.REWARD_VERSION, "settings": definition.validate_settings(),
        "physics_dt_s": .0025, "policy_dt_s": .02, "substeps": 8, "steps_started": windows,
        "steps_completed": windows, "unique_reward_windows": windows, "physics_samples": windows * 8,
        "device_finite_normal_force_health": True, "device_finite_reward_health": True,
        "definition_sha256": module.sha256(definition.__file__),
        "module_sha256": module.sha256(ROOT / "scripts/ppo_ground_contact_v2q.py"), "cpu_fixture_only": False,
        "immutable_v2a_definition_sha256": definition.base.BASE_DEFINITION_SHA256,
        "immutable_v2b_definition_sha256": definition.base.BASE_B_DEFINITION_SHA256,
        "immutable_v2c_definition_sha256": definition.base.BASE_C_DEFINITION_SHA256,
        "immutable_v2d_definition_sha256": definition.base.BASE_D_DEFINITION_SHA256,
        "single_wheel_support_allowed": True, "bilateral_constraint": False,
        "height_or_zero_slip_target": False, "observation_action_or_physics_change": False,
        # One wheel alone is explicitly allowed: no bilateral-bearing samples.
        "force_bearing_samples_L_R_any_both": [100, 0, 100, 0]}
    runtime.update(immutable_v2g_definition_sha256=definition.BASE_G_DEFINITION_SHA256,
        support_gap_aggregation="max8", support_gap_is_window_event_not_duration_fraction=False,
        support_gap_is_continuous_load_margin=True,
        support_gap_cost_mode=definition.DEFAULT_SETTINGS["support_gap_cost_mode"],
        immutable_v2i_definition_sha256=definition.BASE_I_DEFINITION_SHA256,
        load_margin_target_range_n=[98.1, 98.1], mass_kg_range=[20., 20.],
        load_margin_gravity_w_m_s2=[0., 0., -9.81],
        immutable_v2f_definition_sha256=definition.BASE_F_DEFINITION_SHA256,
        beta_quality_source=BETA_QUALITY_SOURCE, beta_quality_source_sha256=BETA_SOURCE_SHA256,
        beta_quality_source_path=str((ROOT/"upstream/tron1-learning-from-human/training/tron1_drift.py").resolve()),
        beta_quality_source_verified=True, beta_quality_std_rad=.12, beta_low_reference_speed_m_s=.3,
        beta_num_envs=num_envs, beta_unique_reward_reads=windows,
        beta_moving_reference_samples=windows*num_envs, beta_low_reference_samples=0,
        beta_environment_reward_samples=windows*num_envs, device_held_reference_phase_health=True,
        device_low_reference_speed_gate_health=True, beta_quality_is_unweighted_existing_term=True,
        reference_beta_not_zero_beta_is_intended=True)
    runtime.update(geometry_physics_samples=windows*8, geometry_unique_reward_reads=windows,
        device_finite_geometry_health=True, geometry_force_sampling_synchronous=True,
        geometry_gap_scale_m=.02, geometry_gap_scale_is_acceptance_tolerance=False,
        unsupported_geometry_cost_only_below5N=True,
        immutable_v2k_definition_sha256=definition.BASE_K_DEFINITION_SHA256,
        immutable_v2k_runtime_sha256=definition.BASE_K_RUNTIME_SHA256,
        authored_geometry_audit=geometry_audit_fixture(num_envs))
    common = {"status": "completed_verified", "cohort_id": module.COHORT_ID,
        "protocol_sha256": module.sha256(protocol_path), "training_environment_steps": budget,
        "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor,
        "manifest_sha256": module.sha256(directory / "manifest.json"),
        "policy_contract_sha256": module.sha256(directory / "policy_contract.json"),
        "initial_policy_state_sha256": module.EXPECTED_INITIAL_POLICY_SHA256, "run_kind": mode, **flags}
    audit = {**common, "reward_definition_version": module.REWARD_VERSION, "normalizer_in_actor": True,
        "from_scratch": True, "source_sha256": {str(path): module.sha256(path) for path in source_paths},
        "contact_runtime_audit": runtime, "other14_reward_terms_changed": False, "new_reward_count": 15}
    put(directory / "cohort_completion_audit.json", common)
    put(directory / "ground_contact_v2q_training_audit.json", audit)
    return directory, protocol_path


def refresh_metadata(directory):
    for name in ("cohort_completion_audit.json", "ground_contact_v2q_training_audit.json"):
        value = module.read_json(directory / name)
        value["manifest_sha256"] = module.sha256(directory / "manifest.json")
        value["policy_contract_sha256"] = module.sha256(directory / "policy_contract.json")
        put(directory / name, value)


def test_new_training_verified_without_checkpoint_execution_and_one_wheel_allowed(tmp_path):
    directory, protocol = make_training(tmp_path)
    result = module.verify_training(directory, protocol)
    assert result["training_backend"] == "physx" and result["smoke_only"] is False
    audit = module.provenance(result)
    assert audit["single_wheel_support_allowed"] is True
    assert audit["bilateral_support_required"] is False
    assert audit["normalization_applications_per_inference"] == 1
    assert "geometric" in audit["support_count_definition"]
    assert audit["initial_policy_state_sha256"] == module.EXPECTED_INITIAL_POLICY_SHA256


def test_e_cannot_substitute_a_different_initial_network_even_with_matching_local_metadata(tmp_path):
    directory,protocol=make_training(tmp_path)
    for name in ("manifest.json","run_status.json","cohort_completion_audit.json","ground_contact_v2q_training_audit.json"):
        value=module.read_json(directory/name);value["initial_policy_state_sha256"]="0"*64
        put(directory/name,value)
    refresh_metadata(directory)
    with pytest.raises(ValueError,match="Initial policy"):
        module.verify_training(directory,protocol)


def test_smoke_requires_explicit_opt_in_not_formal_success(tmp_path):
    directory, protocol = make_training(tmp_path, smoke=True)
    with pytest.raises(ValueError, match="Smoke policy"):
        module.verify_training(directory, protocol)
    result = module.verify_training(directory, protocol, allow_smoke=True)
    assert module.provenance(result)["trained_policy_claimed"] is False


@pytest.mark.parametrize("name", ["policy.pt", "model_final.pt", "manifest.json", "policy_contract.json"])
def test_changed_completed_bytes_rejected(tmp_path, name):
    directory, protocol = make_training(tmp_path)
    with (directory / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        module.verify_training(directory, protocol)


@pytest.mark.parametrize("field,value", [("from_scratch", False), ("normalizer_in_actor", False),
    ("training_environment_steps", 12288), ("reward_definition_version", "old_bilateral_grounded")])
def test_new_audit_semantic_fields_fail_closed(tmp_path, field, value):
    directory, protocol = make_training(tmp_path)
    audit = module.read_json(directory / "ground_contact_v2q_training_audit.json")
    audit[field] = value
    put(directory / "ground_contact_v2q_training_audit.json", audit)
    with pytest.raises(ValueError):
        module.verify_training(directory, protocol)


@pytest.mark.parametrize("field,value", [("physics_samples", 192000), ("unique_reward_windows", 24000),
    ("device_finite_normal_force_health", False), ("device_finite_reward_health", False), ("cpu_fixture_only", True),
    ("bilateral_constraint", True), ("single_wheel_support_allowed", False),
    ("force_bearing_samples_L_R_any_both", [0, 0, 0, 0])])
def test_actual_reward_clock_and_force_runtime_proof_required(tmp_path, field, value):
    directory, protocol = make_training(tmp_path)
    audit = module.read_json(directory / "ground_contact_v2q_training_audit.json")
    audit["contact_runtime_audit"][field] = value
    put(directory / "ground_contact_v2q_training_audit.json", audit)
    with pytest.raises(ValueError):
        module.verify_training(directory, protocol)


def test_absolute_source_ledger_checked_not_only_declared(tmp_path):
    directory, protocol = make_training(tmp_path)
    (tmp_path / "train_ppo_ground_contact_v2q.py").write_text("# source changed after training\n")
    with pytest.raises(ValueError, match="source ledger differs"):
        module.verify_training(directory, protocol)


def test_e_requires_immutable_d_runtime_and_source_ancestry(tmp_path):
    directory, protocol = make_training(tmp_path)
    audit = module.read_json(directory / "ground_contact_v2q_training_audit.json")
    audit["contact_runtime_audit"]["immutable_v2d_definition_sha256"] = "0" * 64
    put(directory / "ground_contact_v2q_training_audit.json", audit)
    with pytest.raises(ValueError, match="source readback"):
        module.verify_training(directory, protocol)


def test_zero_gap_settings_cannot_be_relabelled_as_f(tmp_path):
    directory, protocol_path = make_training(tmp_path)
    protocol = module.read_json(protocol_path)
    protocol["ground_contact_reward_v2"]["support_gap_weight"] = 0.
    put(protocol_path, protocol)
    with pytest.raises(ValueError, match="Frozen Q setting differs"):
        module.verify_training(directory, protocol_path)


def test_rehashed_zero_beta_or_wrong_clock_still_rejected(tmp_path):
    directory, protocol = make_training(tmp_path)
    contract = module.read_json(directory / "policy_contract.json")
    contract["trained_task_profile"]["reward_weights"]["sideslip"] = 0.
    contract["task_profile"] = copy.deepcopy(contract["trained_task_profile"])
    for name in ("policy_contract.json", "final_policy_contract.json"):
        put(directory / name, contract)
    manifest = module.read_json(directory / "manifest.json")
    manifest["contract"], manifest["task_profile"] = contract, contract["task_profile"]
    put(directory / "manifest.json", manifest)
    refresh_metadata(directory)
    with pytest.raises(ValueError, match="Trained reward profile differs"):
        module.verify_training(directory, protocol)


def recording():
    actions = np.zeros((349, 8))
    return {"time_s": np.arange(2793) * .0025, "physics_dt_s": np.array(.0025),
        "policy_dt_s": np.array(.02), "policy_time_s": np.arange(349) * .02,
        "action_observation": np.zeros((349, 51)), "actions_raw": actions,
        "actions_clipped": actions.copy(), "previous_clipped_actions": actions.copy(),
        "root_pos": np.zeros((2793, 3)), "root_quat_wxyz": np.tile([1., 0., 0., 0.], (2793, 1)),
        "joint_pos": np.zeros((2793, 8)), "joint_vel": np.zeros((2793, 8)), "joint_torque_nm": np.zeros((2793, 8))}


def finish_fixture(tmp_path):
    directory, protocol = make_training(tmp_path)
    cohort = module.verify_training(directory, protocol)
    output, inputs = tmp_path / "evaluation", tmp_path / "evaluation_inputs"
    output.mkdir(); inputs.mkdir()
    raw = recording()
    np.savez_compressed(output / "rollout.npz", **raw)
    contract = module.common.evaluation_contract(cohort, .7, "native CPU MuJoCo")
    report = {"status": "completed", "completed_full_reference": True, "policy_steps": 349,
        "physics_steps": 2792, "hidden_resets": 0, "root_state_writes": 1, "start_reference_frame": 0,
        "domain_parameters_resampled_during_episode": False, "contract": contract,
        "policy_sha256": cohort["input_sha256"]["policy.pt"], "contact_material_audit": {
            "native_mapping_verified": True, "explicit_contact_pair_count": 0, "pair_coefficient": .7,
            "geom_friction": [[.7, 0., 0.]] * 3, "colliding_geom_ids": [0, 1, 2],
            "geom_priority": [0, 1, 1], "wheel_geom_ids": [1, 2]}}
    evaluator = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
    plan = {"cohort": cohort, "args": SimpleNamespace(backend="mujoco"), "output": output,
        "input_dir": inputs, "contract": copy.deepcopy(contract), "wrapper_sha256": module.sha256(module.__file__),
        "evaluator": evaluator, "generated_sha256": "a" * 64}
    return report, raw, plan


def test_complete_native_recording_and_metadata_no_fake_bilateral_success(tmp_path):
    report, raw, plan = finish_fixture(tmp_path)
    module._finish(report, raw, plan)
    audit = report["ground_contact_v2q_evaluation"]
    assert audit["native_states"] == 2793 and audit["policy_decisions"] == 349
    assert audit["bilateral_support_required"] is False and audit["hardware_ready"] is False
    assert "backend_400hz_cohort_evaluation" not in report
    put(plan["output"] / "report.json", report)
    proof = module.validate_output(plan["output"], plan["cohort"]["directory"], plan["cohort"]["protocol_path"])
    assert proof["status"] == "verified" and proof["native_states"] == 2793
    assert "at least one" in proof["support_requirement"]


@pytest.mark.parametrize("defect", ["hash", "mu", "priority", "clock", "actor", "reset"])
def test_parent_recheck_rejects_mixed_recording_or_simulator_proof(tmp_path, defect):
    report, raw, plan = finish_fixture(tmp_path)
    module._finish(report, raw, plan)
    if defect == "hash": report["ground_contact_v2q_evaluation"]["recording_sha256"] = "f" * 64
    if defect == "mu": report["contact_material_audit"]["geom_friction"][0][0] = .49
    if defect == "priority": report["contact_material_audit"]["geom_priority"][1] = 0
    if defect == "clock": report["contract"]["physics_dt"] = .005
    if defect == "actor": report["policy_sha256"] = "f" * 64
    if defect == "reset": report["hidden_resets"] = 1
    put(plan["output"] / "report.json", report)
    with pytest.raises(ValueError):
        module.validate_output(plan["output"], plan["cohort"]["directory"], plan["cohort"]["protocol_path"])


@pytest.mark.parametrize("field,value", [("policy_steps", 348), ("physics_steps", 2791),
    ("root_state_writes", 2), ("hidden_resets", 1), ("completed_full_reference", False),
    ("domain_parameters_resampled_during_episode", True), ("policy_sha256", "f" * 64)])
def test_no_missing_return_or_full_horizon_fabrication(tmp_path, field, value):
    report, raw, plan = finish_fixture(tmp_path)
    report[field] = value
    with pytest.raises(ValueError):
        module._finish(report, raw, plan)


def test_frozen_native_source_adapter_does_not_edit_original_or_change_physics(tmp_path):
    source = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
    if not source.is_file():
        pytest.skip("Frozen source not downloaded")
    before = module.sha256(source)
    text = module.native.materialize_source(source.read_text(), source.parent.parent)
    compile(text, "<any-wheel-native-eval>", "exec")
    assert module.sha256(source) == before == module.native.BASE_SHA256
    assert "if terms and not diagnostic_mode:" in text
    assert "for substep in range(contract[\"decimation\"]):" in text
    assert "_new_finish_report(report, arrays, _new_plan)" in text


def test_cli_is_nominal_only_and_gpu_is_never_an_implicit_fallback():
    args = ["--training-dir", "x", "--protocol", "p", "--legacy-evaluator", "e",
            "--motion-file", "m", "--output-dir", "o", "--model", "xml"]
    parsed = module.parser().parse_args(args)
    assert parsed.backend == "mujoco" and parsed.mu == .7
    with pytest.raises(SystemExit):
        module.parser().parse_args(args + ["--mu", ".15"])


def test_prepare_plan_reuses_native_counted_legacy_and_fresh_inputs(tmp_path, monkeypatch):
    directory, protocol = make_training(tmp_path)
    evaluator = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
    if not evaluator.is_file():
        pytest.skip("Frozen evaluator snapshot unavailable")
    cohort = module.verify_training(directory, protocol)
    monkeypatch.setattr(module, "verify_training", lambda *args, **kwargs: cohort)
    motion, model = tmp_path / "motion.npz", tmp_path / "model.xml"
    motion.write_bytes(b"reference fixture"); model.write_bytes(b"model fixture")
    monkeypatch.setitem(module.common.INPUT_PINS, "motion", module.sha256(motion))
    monkeypatch.setitem(module.common.INPUT_PINS, "model", module.sha256(model))
    argv = ["--training-dir", str(directory), "--protocol", str(protocol), "--legacy-evaluator", str(evaluator),
        "--motion-file", str(motion), "--model", str(model), "--output-dir", str(tmp_path / "validation")]
    plan = module.prepare_plan(argv)
    assert plan["contract"]["full_horizon_diagnostic"] is True
    assert plan["contract"]["trained_task_profile"]["contact_friction"] == 1.
    assert plan["contract"]["task_profile"]["contact_friction"] == .7
    assert plan["input_dir"] == tmp_path / "validation_inputs"
    assert not plan["output"].exists()
    with pytest.raises(ValueError, match="Preserve"):
        module.prepare_plan(argv)


def test_optional_physx_adapter_uses_counted_legacy_without_gpu_initialization(tmp_path):
    import eval_ppo_backend_400hz_isaac as physx
    assert module.sha256(physx.__file__) == module.PHYSX_HELPER_SHA256
    source = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py"
    if not source.is_file():
        pytest.skip("Frozen PhysX evaluator unavailable")
    generated = physx.materialize_source(source.read_text(), source.parent.parent)
    compile(generated, "<ground-contact-v2-physx-fixture>", "exec")
    assert "runner = _new_runner(" in generated
    assert "_new_finish_report(report, contract, _new_plan, physical_arrays)" in generated

@pytest.mark.parametrize("field,value", [
    ("beta_quality_source_verified", False), ("beta_quality_source_sha256", "0"*64),
    ("device_held_reference_phase_health", False), ("device_low_reference_speed_gate_health", False),
    ("beta_unique_reward_reads", 24000), ("beta_moving_reference_samples", 1),
    ("beta_environment_reward_samples", 1), ("beta_quality_std_rad", .2),
    ("beta_low_reference_speed_m_s", 0.), ("immutable_v2f_definition_sha256", "0"*64),
    ("immutable_v2g_definition_sha256", "0"*64), ("support_gap_aggregation", "mean8"),
    ("support_gap_is_window_event_not_duration_fraction", True),
    ("support_gap_is_continuous_load_margin", False),
    ("support_gap_cost_mode", "binary_any_gap"),
    ("immutable_v2i_definition_sha256", "0"*64),
    ("load_margin_target_range_n", [5., 5.]),
    ("load_margin_gravity_w_m_s2", [0., 0., 0.]),
    ("mass_kg_range", [200., 200.]),
    ("beta_quality_source_path", "/unexecuted/source.py"), ("beta_quality_is_unweighted_existing_term", False)])
def test_g_rejects_missing_or_relabelled_beta_phase_and_source_proof(tmp_path, field, value):
    directory, protocol = make_training(tmp_path)
    audit = module.read_json(directory/"ground_contact_v2q_training_audit.json")
    audit["contact_runtime_audit"][field] = value
    put(directory/"ground_contact_v2q_training_audit.json", audit)
    with pytest.raises(ValueError): module.verify_training(directory, protocol)

@pytest.mark.parametrize("basename", ["ground_contact_reward_v2f.py", "ground_contact_reward_v2g.py", "ground_contact_beta_contract.py", "tron1_drift.py"])
def test_g_requires_actual_f_beta_and_contract_source_ledger(tmp_path, basename):
    directory, protocol = make_training(tmp_path)
    audit = module.read_json(directory/"ground_contact_v2q_training_audit.json")
    audit["source_sha256"] = {name: sha for name, sha in audit["source_sha256"].items() if Path(name).name != basename}
    put(directory/"ground_contact_v2q_training_audit.json", audit)
    with pytest.raises(ValueError): module.verify_training(directory, protocol)


@pytest.mark.parametrize('field,value',[
    ('geometry_physics_samples',1),('geometry_unique_reward_reads',1),
    ('device_finite_geometry_health',False),('geometry_force_sampling_synchronous',False),
    ('geometry_gap_scale_m',.04),('geometry_gap_scale_is_acceptance_tolerance',True),
    ('unsupported_geometry_cost_only_below5N',False),
    ('immutable_v2k_definition_sha256','0'*64),('immutable_v2k_runtime_sha256','0'*64)])
def test_q_rejects_missing_native_geometry_sampling_proof(tmp_path,field,value):
    directory,protocol=make_training(tmp_path)
    audit=module.read_json(directory/'ground_contact_v2q_training_audit.json')
    audit['contact_runtime_audit'][field]=value
    put(directory/'ground_contact_v2q_training_audit.json',audit)
    with pytest.raises(ValueError):module.verify_training(directory,protocol)


@pytest.mark.parametrize('field,value',[
    ('cpu_fixture_only',True),('verified_environment_count',1),('all_environment_frames_checked',False),
    ('original_one_cylinder_per_wheel',False),('native_wheel_offsets',{'contact_m':[.02,.02],'rest_m':[0.,0.]}),
    ('startup_CPU_Torch_max_gap_difference_m',.0001),('source_sha256',{})])
def test_q_rejects_fixture_partial_or_wrong_geometry_audit(tmp_path,field,value):
    directory,protocol=make_training(tmp_path)
    audit=module.read_json(directory/'ground_contact_v2q_training_audit.json')
    audit['contact_runtime_audit']['authored_geometry_audit'][field]=value
    put(directory/'ground_contact_v2q_training_audit.json',audit)
    with pytest.raises(ValueError):module.verify_training(directory,protocol)
