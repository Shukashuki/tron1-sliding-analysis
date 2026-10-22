"""CPU-only source/configuration tests; never launch Isaac, Warp or training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cohort_trainer", ROOT/"scripts/train_ppo_backend_400hz.py")
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def protocol():
    return {"schema_version": 1, "cohort_id": "test_native400hz_matched_arms",
        "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "policy_fps": 50,
        "action_rate_weight": -.5, "grounded_reward_weight": .4,
        "base_trainer_sha256": trainer.BASE_SHA256, "shared_dr_recipe": "tron1_mjlab_tracking_dr_v1",
        "mjlab_commit": trainer.MJLAB_PIN, "mjlab_task_implementation": "gpu",
        "algorithm": {"implementation": "rsl-rl-lib", "version": "3.0.1", "seed": 42,
            "num_envs": 512, "steps_per_env": 24, "iterations": 1000, "transitions": 12288000},
        "smoke": {"num_envs": 32, "iterations": 2, "transitions": 1536, "setup_smoke_steps": 8},
        "training_randomization": {"sliding_friction": {"range": [.3, 1.2], "distribution": "uniform_continuous",
            "sampling": "startup_once_per_environment", "shared_between_wheels": True,
            "scope": "colliding_wheel_shapes", "mujoco_wheel_priority": 1,
            "physx_static_equals_dynamic": True, "physx_floor_mu": 1., "physx_combine": "multiply"},
            "encoder_position_bias_rad": [-.01, .01],
            "encoder_bias_signs": {"actor": "+b", "position_target": "-b", "critic": "unbiased"},
            "base_com_xyz_range_m": [[-.025, .025], [-.05, .05], [-.05, .05]],
            "root_link_velocity_push_interval_s": [1., 3.], "restitution_randomization": False}}


def test_protocol_and_base_source_are_fixed():
    assert trainer.validate_protocol(protocol())
    assert trainer.sha256(ROOT/"scripts/train_ppo_backend.py") == trainer.BASE_SHA256


@pytest.mark.parametrize("key,value", [("physics_dt_s", .005), ("decimation", 4),
    ("action_rate_weight", -.1), ("grounded_reward_weight", .5), ("shared_dr_recipe", "old_bm"),
    ("mjlab_task_implementation", "host")])
def test_reject_changed_declared_cohort(key, value):
    p = protocol(); p[key] = value
    with pytest.raises(ValueError, match="field differs"):
        trainer.validate_protocol(p)


def test_reject_different_formal_budget():
    p = protocol(); p["algorithm"]["transitions"] = 1000
    with pytest.raises(ValueError, match="Algorithm"):
        trainer.validate_protocol(p)


def test_reject_false_common_dr_declaration():
    p = protocol(); p["training_randomization"]["sliding_friction"]["range"] = [.3, 1.6]
    with pytest.raises(ValueError, match="friction declaration"):
        trainer.validate_protocol(p)


def test_derived_inputs_remove_all_historical_actor_and_evaluation_identity():
    nominal = json.loads((ROOT/"config/nominal_actuator_contract.json").read_text())
    task = json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text())
    old_n, old_t = copy.deepcopy(nominal), copy.deepcopy(task)
    contract, derived = trainer.derive_inputs(nominal, task, protocol())
    assert nominal == old_n and task == old_t
    assert contract["physics_dt"] == .0025 and contract["decimation"] == 8
    assert derived["action_rate_weight"] == -.5 and derived["contact_friction"] == 1.
    assert contract["task_profile"] == contract["trained_task_profile"] == derived
    assert "f803a850" not in json.dumps(contract) and "actor_normalized.pt" not in json.dumps(contract)
    for name in ("checkpoint_sha256", "actor_file", "actual_joint_order", "actual_body_order",
            "export_validation_max_abs_error", "evaluation_contact_friction_override", "physics_clock_refinement"):
        assert name not in contract
    for name in ("actor_dim", "critic_dim", "action_dim", "leg_kp", "leg_kd",
                 "leg_torque_limit_nm", "wheel_torque_scale_nm", "leg_joint_names"):
        assert contract[name] == nominal[name]


def test_counted_source_adapter_preserves_ppo_configuration_and_seed_order():
    original = (ROOT/"scripts/train_ppo_backend.py").read_text()
    generated = trainer.materialize_source(original)
    compile(generated, "<test>", "exec")
    assert generated.count("_cohort_before_start(") == 1
    assert generated.count("_cohort_before_learn(") == 1
    assert generated.count("_cohort_finalize_contract(") == 1
    assert "runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)" in generated
    assert generated.index("torch.manual_seed(args.seed)") < generated.index("runner = OnPolicyRunner") < generated.index("_cohort_before_learn")
    assert generated[generated.index("def ppo_config"):generated.index("def write_json")] == original[original.index("def ppo_config"):original.index("def write_json")]
    assert "runner.load" not in generated


def test_reject_edited_base_before_anchor_matching():
    original = (ROOT/"scripts/train_ppo_backend.py").read_text()
    with pytest.raises(ValueError, match="content differs"):
        trainer.materialize_source(original+"\n")


def test_configure_physx_changes_only_declared_clocks_and_action_weight():
    cfg = NS(sim=NS(dt=.005, render_interval=4), decimation=4,
        scene=NS(contact_forces=NS(update_period=.005)),
        rewards=NS(action_rate_l2=NS(weight=-.1), drift_grounded=NS(weight=.4)))
    calls = []
    trainer.configure_physx(cfg, None, None, lambda c: calls.append(c))
    assert calls == [cfg]
    assert (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval) == (.0025, 8, 8)
    assert cfg.scene.contact_forces.update_period == .0025
    assert cfg.rewards.action_rate_l2.weight == -.5 and cfg.rewards.drift_grounded.weight == .4


def runtime_values():
    return {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8,
        "action_rate_weight": -.5, "grounded_reward_weight": .4, "reward_term_count": 15,
        "leg_kp": [[500.]*6]*32, "leg_kd": [[10.]*6]*32,
        "absolute_height_termination": False, "absolute_xy_termination": False}


def test_runtime_pd_checks_array_values_not_physx_drive_gains():
    assert trainer.validate_runtime_fields(runtime_values())


@pytest.mark.parametrize("key,value", [("physics_dt_s", .005), ("decimation", 4),
    ("actor_dim", 50), ("reward_term_count", 16), ("action_rate_weight", -.1),
    ("grounded_reward_weight", .8), ("leg_kp", [0]*6), ("leg_kd", [[5]*6]),
    ("absolute_height_termination", True), ("absolute_xy_termination", True)])
def test_runtime_fields_fail_closed(key, value):
    values = runtime_values(); values[key] = value
    with pytest.raises(ValueError):
        trainer.validate_runtime_fields(values)


def test_launch_arguments_pin_native_and_renderer_to_same_explicit_device():
    calls = []
    class FakeLauncher:
        def __init__(self, **kwargs):
            calls.append(kwargs); self.app = "owned_app"
    plan = {}
    assert trainer.launch_app(FakeLauncher, NS(device="cuda:3"), plan) == "owned_app"
    assert calls[0]["device"] == "cuda:3" and calls[0]["headless"] is True
    assert "--/physics/cudaDevice=3" in calls[0]["kit_args"]
    assert "--/physics/sceneMultiGPUMode=0" in calls[0]["kit_args"]
    assert "--/renderer/multiGpu/enabled=false" in calls[0]["kit_args"]


def test_prepare_plan_accepts_final_nested_protocol_without_importing_gpu_runtime(tmp_path):
    path = tmp_path/"protocol.json"; path.write_text(json.dumps(protocol()))
    plan = trainer.prepare_plan(["--cohort-protocol", str(path), "--cohort-run-kind", "smoke",
                                "--backend", "physx", "--num-envs", "32"])
    assert plan["run_kind"] == "smoke"
    assert plan["inherited_argv"] == ["--backend", "physx", "--num-envs", "32"]
    assert plan["source_hashes"]["scripts/train_ppo_backend.py"] == trainer.BASE_SHA256


def test_public_preparer_is_deterministic_across_parity_and_training_stages(tmp_path):
    p = tmp_path/"protocol.json"; p.write_text(json.dumps(protocol()))
    parity = trainer.prepare_declared_inputs(p, tmp_path/"parity")
    formal = trainer.prepare_declared_inputs(p, tmp_path/"formal")
    assert parity["derived_input_sha256"] == formal["derived_input_sha256"]
    assert parity["contract"].read_bytes() == formal["contract"].read_bytes()
    assert parity["task_profile"].read_bytes() == formal["task_profile"].read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        trainer.prepare_declared_inputs(p, tmp_path/"parity")


@pytest.mark.parametrize("mode,budget", [("smoke", 1536), ("formal", 12288000)])
def test_finalize_and_complete_contract_have_truthful_mode_and_current_identity(tmp_path, monkeypatch, mode, budget):
    import sys
    class FakeTensor(np.ndarray):
        def cpu(self):
            return self
    fake = NS(inference_mode=nullcontext, allclose=np.allclose, abs=np.abs, max=np.max)
    monkeypatch.setitem(sys.modules, "torch", fake)
    p = tmp_path/"protocol.json"; p.write_text(json.dumps(protocol()))
    prepared = trainer.prepare_declared_inputs(p, tmp_path/"declared")
    output = tmp_path/"training"; output.mkdir()
    (output/"model_final.pt").write_bytes(b"new-test-checkpoint")
    (output/"policy.pt").write_bytes(b"new-test-normalized-actor")
    (output/"cohort_startup_audit.json").write_text("{}")
    motion = tmp_path/"motion.npz"; motion.write_bytes(b"current-test-reference")
    args = NS(contract=prepared["contract"], task_profile=prepared["task_profile"],
        output_dir=output, backend="physx", motion=motion)
    plan = {"run_kind": mode, "expected_transitions": budget, "protocol": protocol(),
        "protocol_path": p, "protocol_sha256": trainer.sha256(p),
        "source_hashes": {name: trainer.sha256(ROOT/name) for name in trainer.SOURCE_FILES}}
    manifest = {"cohort": {}, "initial_policy_state_sha256": "a"*64}
    probe = np.zeros((2, 51)).view(FakeTensor)
    expected = np.ones((2, 8)).view(FakeTensor)
    trainer.finalize_contract(args, manifest, NS(tot_timesteps=budget), lambda _: expected, probe, expected, plan)
    contract = json.loads((output/"final_policy_contract.json").read_text())
    assert (output/"final_policy_contract.json").read_bytes() == (output/"policy_contract.json").read_bytes()
    assert contract["actor_file"] == "policy.pt" and contract["checkpoint_file"] == "model_final.pt"
    assert contract["checkpoint_sha256"] == trainer.sha256(output/"model_final.pt")
    assert contract["exported_policy_sha256"] == trainer.sha256(output/"policy.pt")
    assert contract["smoke_only"] is (mode == "smoke")
    assert contract["trained_policy_claimed"] is (mode == "formal")
    assert contract["training_environment_transitions"] == budget
    assert contract["task_profile"] == contract["trained_task_profile"]
    assert contract["task_profile"]["contact_friction"] == 1.
    assert "evaluation_contact_friction_override" not in contract
    report = {"status": "training_completed", "training_environment_steps": budget,
        "checkpoint_sha256": contract["checkpoint_sha256"], "exported_policy_sha256": contract["exported_policy_sha256"]}
    trainer.complete(args, manifest, report, plan)
    completed = json.loads((output/"cohort_completion_audit.json").read_text())
    assert completed["status"] == "completed_verified" and completed["run_kind"] == mode
    assert completed["training_environment_steps"] == budget
    assert completed["final_policy_contract_sha256"] == completed["policy_contract_sha256"]
    assert completed["smoke_only"] is report["smoke_only"] is (mode == "smoke")


def test_supplied_actual_protocol_matches_when_root_publishes_it():
    path = ROOT/"config/ppo_backend_400hz_comparison_v1.json"
    if not path.exists():
        pytest.skip("Root-owned protocol not published yet")
    trainer.validate_protocol(json.loads(path.read_text()))
