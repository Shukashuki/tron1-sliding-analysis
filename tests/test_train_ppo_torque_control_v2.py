"""Pure wrapper/finalization fixtures; no Isaac/Torch policy job is launched."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_ppo_torque_control_v2 as trainer


def action_rate_l2(): pass
action_rate_l2.__module__ = "isaaclab.envs.mdp.rewards"
def grounded_reward(): pass
grounded_reward.__module__ = "training.tron1_drift"
def other_reward(): pass


@pytest.fixture
def managers(monkeypatch):
    package, module = ModuleType("isaaclab"), ModuleType("isaaclab.managers")
    class Base:
        def __init__(self, cfg, env): self.cfg, self._env = cfg, env
    class Term:
        def __init__(self, *, func, weight, params=None): self.func, self.weight, self.params = func, weight, params
    module.ManagerTermBase, module.RewardTermCfg = Base, Term
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", module)
    return module


@pytest.fixture
def plan(tmp_path):
    return {"output_dir": tmp_path, "inherited_argv": [], "smoke_only": True, "num_envs": 2, "iterations": 1,
        "expected_policy_steps": 24, "expected_environment_transitions": 48}


def configuration(managers):
    rewards = SimpleNamespace(action_rate_l2=managers.RewardTermCfg(func=action_rate_l2, weight=-.1),
        drift_grounded=managers.RewardTermCfg(func=grounded_reward, weight=.4))
    for index in range(12):
        setattr(rewards, "original_"+str(index), managers.RewardTermCfg(func=other_reward, weight=1.))
    return SimpleNamespace(rewards=rewards, scene=SimpleNamespace(robot=SimpleNamespace(
        actuators={"legs": SimpleNamespace(stiffness=500., damping=10.)})), sim=SimpleNamespace(dt=.005), decimation=4)


def configured(plan, managers):
    cfg = configuration(managers); old_grounded = cfg.rewards.drift_grounded
    recipe = {"original_DR": True}
    assert trainer.make_pre_env_hook(lambda *a, **kw: recipe, plan)(cfg) is recipe
    assert cfg.rewards.drift_grounded is old_grounded
    return cfg


def test_add_only16th_reward_on_minus0p5_recipe_keep_original_grounded(plan, managers):
    cfg = configured(plan, managers)
    assert cfg.rewards.action_rate_l2.weight == -.5
    assert cfg.rewards.drift_grounded.weight == .4
    assert cfg.rewards.drift_grounded.func is grounded_reward
    assert cfg.rewards.leg_applied_torque_rate.weight == -.05
    metadata = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())
    assert len(metadata["pre_height_active_reward_terms_before"]) == 14
    assert len(metadata["pre_height_active_reward_terms_after"]) == 15
    assert len(metadata["expected_final_active_reward_terms_before"]) == 15
    assert len(metadata["expected_final_active_reward_terms_after"]) == 16
    assert "height_vertical_acceleration" in metadata["expected_final_active_reward_terms_after"]
    assert metadata["baseline_checkpoint_used_for_initialization"] is False
    assert metadata["original_grounded_reward_preserved"] is True
    assert not metadata["new_observation_terms"]
    assert metadata["additional_reward"]["delta_torque_scale_nm"] == 20.
    assert metadata["additional_reward"]["weight"] == -.05


@pytest.mark.parametrize("change", ["wrong_ground", "changed_ground_weight", "already_minus0p5", "pd", "clock", "sensor", "extra_reward"])
def test_pre_env_rejects_other_method_or_nonisolated_recipe(plan, managers, change):
    cfg = configuration(managers)
    if change == "wrong_ground": cfg.rewards.drift_grounded.func = other_reward
    if change == "changed_ground_weight": cfg.rewards.drift_grounded.weight = .8
    if change == "already_minus0p5": cfg.rewards.action_rate_l2.weight = -.5
    if change == "pd": cfg.scene.robot.actuators["legs"].damping = 5.
    if change == "clock": cfg.decimation = 8
    if change == "sensor": cfg.scene.wheel_L_ground = object()
    if change == "extra_reward": cfg.rewards.extra = managers.RewardTermCfg(func=other_reward, weight=1.)
    with pytest.raises(ValueError): trainer.make_pre_env_hook(lambda *a, **kw: {}, plan)(cfg)


def test_reward_manager_reset_before_install_is_safe(plan, managers):
    cfg = configured(plan, managers); env = SimpleNamespace()
    term = cfg.rewards.leg_applied_torque_rate.func(None, env)
    term.reset([0])
    with pytest.raises(ValueError, match="explicitly installed"): term(env)


def test_real_pinned_height_adapter_lifecycle14_to15_then16(plan, managers, monkeypatch):
    import height_smoothness
    cfg = configuration(managers)
    for index, name in enumerate(("motion_global_anchor_pos", "motion_body_pos", "motion_body_lin_vel")):
        value = getattr(cfg.rewards, "original_"+str(index)); delattr(cfg.rewards, "original_"+str(index))
        setattr(cfg.rewards, name, value)
    cfg.terminations = SimpleNamespace(anchor_pos=object(), ee_body_pos=object())
    monkeypatch.setattr(height_smoothness, "vertical_acceleration_term", lambda: other_reward)
    trainer.make_pre_env_hook(lambda *a, **kw: {}, plan)(cfg)
    assert len(trainer._reward_names(cfg.rewards, managers.RewardTermCfg)) == 15
    height_smoothness.install_height_smoothness(cfg, {"mode": "vertical_acceleration_only", "weight": -.1, "acceleration_scale_m_s2": 5.})
    metadata = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())
    assert trainer._reward_names(cfg.rewards, managers.RewardTermCfg) == metadata["expected_final_active_reward_terms_after"]
    assert len(metadata["expected_final_active_reward_terms_after"]) == 16
    assert cfg.rewards.drift_grounded.func is grounded_reward


def test_post_env_close_audit_before_real_isaac_exit(plan, monkeypatch):
    events = []
    class Scalar:
        def __init__(self, value): self.value = value
        def item(self): return self.value
    class Buffer:
        def abs(self): return self
        def max(self): return Scalar(35.)
        def min(self): return Scalar(.1)
    class Handler:
        previous = cached_cost = Buffer()
        def audit(self, *, check_device_health=False):
            events.append("audit"); return {"device_health_finite": True, "closed": False}
        def close(self): events.append("detach")
    handler = Handler()
    monkeypatch.setattr(trainer.torque, "install_accumulator", lambda env: handler)
    monkeypatch.setattr(trainer, "_runtime_recipe", lambda env, installed: {"original_grounded_reward_preserved": True})
    def old_close():
        assert (plan["output_dir"]/trainer.RUNTIME_FILE).is_file()
        events.append("isaac_close")
    env = SimpleNamespace(scene=SimpleNamespace(), close=old_close)
    hook = trainer.make_post_env_hook(lambda env, recipe: recipe, plan)
    assert hook(env, {"oldDR": True}) == {"oldDR": True}
    wrapped = env.close; hook(env, {})
    assert env.close is wrapped
    env.close()
    assert events == ["audit", "detach", "isaac_close"]
    assert env.close is old_close and not hasattr(env, trainer.CLOSE_MARKER)
    audit = json.loads((plan["output_dir"]/trainer.RUNTIME_FILE).read_text())
    assert audit["closed"] and audit["last_native_leg_applied_torque_abs_peak_nm"] == 35.
    assert "Last actual native sample" in audit["torque_telemetry_scope"]


def test_startup_native_gain_and_manager_audit_with_cpu_tensors(plan, managers):
    torch = pytest.importorskip("torch")
    cfg = configured(plan, managers)
    names = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())["expected_final_active_reward_terms_after"]
    legs = SimpleNamespace(stiffness=torch.full((2, 6), 500.), damping=torch.full((2, 6), 10.))
    robot = SimpleNamespace(data=SimpleNamespace(joint_stiffness=torch.zeros((2, 8)), joint_damping=torch.zeros((2, 8))), actuators={"legs": legs})
    env = SimpleNamespace(cfg=cfg, scene={"robot": robot}, reward_manager=SimpleNamespace(active_terms=names))
    audit = trainer._runtime_recipe(env, SimpleNamespace(leg_index=torch.arange(6)))
    assert audit["gain_readback"] == {"kp_range": [500., 500.], "kd_range": [10., 10.]}
    assert "not native PhysX drive" in audit["gain_readback_source"]
    legs.damping[0, 0] = 5.
    with pytest.raises(ValueError, match="DC motor leg PD"):
        trainer._runtime_recipe(env, SimpleNamespace(leg_index=torch.arange(6)))


def runtime(smoke=True):
    steps, envs = (24, 2) if smoke else (24000, 512)
    names = sorted(["original_"+str(x) for x in range(12)]+[trainer.TERM_NAME, "drift_grounded", "action_rate_l2", "height_vertical_acceleration"])
    return {"module_sha256": trainer.COLLECTOR_SHA256, "wrapper_sha256": trainer.sha256(trainer.__file__),
        "steps_started": steps, "steps_completed": steps, "unique_cost_windows": steps,
        "physics_samples_total": 4*steps, "steps_failed": 0, "samples_last_window": 4, "num_envs": envs,
        "closed": True, "device_health_checked": True, "device_health_finite": True, "device": "cuda:1",
        "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4, "delta_torque_scale_nm": 20.,
        "torque_rate_scale_nm_s": 4000., "wheel_cost_included": False, "first_actual_substep_after_reset_cost": 0.,
        "denominator_after_reset": 4, "leg_joint_names": list(trainer.torque.LEG_JOINT_NAMES),
        "gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]}, "active_reward_count": 16,
        "active_reward_terms": names, "original_grounded_reward_preserved": True, "original_grounded_weight": .4,
        "original_grounded_function": "training.tron1_drift.grounded_reward",
        "action_rate_weight": -.5, "additional_reward_weight": -.05, "new_contact_sensors_added": False,
        "new_observation_terms": False, "last_native_leg_applied_torque_abs_peak_nm": 35., "last_window_raw_cost_range": [.1, .2]}


def complete(plan, managers):
    configured(plan, managers); out = plan["output_dir"]
    smoke = plan["smoke_only"]
    for name, value in {
        "run_status.json": {"status": "smoke_passed" if smoke else "training_completed", "trained_policy_claimed": not smoke,
            "policy_steps_completed": plan["expected_policy_steps"], "environment_transitions": plan["expected_environment_transitions"]},
        trainer.RUNTIME_FILE: runtime(smoke),
        "manifest.json": {"trainer_sha256": trainer.BASELINE_TRAINER_SHA256, "task_profile_sha256": trainer.BASELINE_PROFILE_SHA256,
            "source_training_modules_sha256": trainer.BASELINE_SOURCES, "motion_sha256": trainer.BASELINE_MOTION_SHA256,
            "asset_sha256": trainer.BASELINE_ASSET_SHA256},
    }.items(): (out/name).write_text(json.dumps(value))
    (out/("model_smoke.pt" if smoke else "model_final.pt")).write_bytes(b"fixture checkpoint not a trained model")
    return out


def test_parent_finalize_after_isaac_exit_preserves_original_grounded_and_no_smoke_claim(plan, managers):
    out = complete(plan, managers); result = trainer.finalize_manifest(out, child_returncode=0)
    assert result["status"] == "smoke_passed" and not result["trained_policy_claimed"]
    assert result["prepared_method_runtime_verified"]
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["training_reward_overrides"] == {"action_rate_l2": -.5}
    assert manifest["additional_training_reward_terms"][trainer.TERM_NAME]["weight"] == -.05
    assert manifest["original_grounded_reward_preserved"]
    assert "contact_reward_replacement" not in manifest
    assert trainer.finalize_manifest(out, child_returncode=0)["checkpoint_sha256"] == result["checkpoint_sha256"]


def test_formal_exact_budget_from_scratch(plan, managers):
    plan.update(smoke_only=False, num_envs=512, iterations=1000, expected_policy_steps=24000, expected_environment_transitions=12288000)
    out = complete(plan, managers); result = trainer.finalize_manifest(out, child_returncode=0)
    assert result["status"] == "training_completed" and result["trained_policy_claimed"]
    assert result["actual_environment_transitions"] == 12288000


@pytest.mark.parametrize("key,value", [("steps_completed", 23), ("physics_samples_total", 95), ("device_health_finite", False),
    ("wheel_cost_included", True), ("delta_torque_scale_nm", 80.), ("device", "cpu"), ("additional_reward_weight", -.01),
    ("original_grounded_reward_preserved", False), ("last_native_leg_applied_torque_abs_peak_nm", 0.),
    ("last_native_leg_applied_torque_abs_peak_nm", 800.), ("active_reward_count", 15), ("last_window_raw_cost_range", [float("nan"), .1])])
def test_parent_rejects_bad_torque_runtime(plan, managers, key, value):
    out = complete(plan, managers); audit = runtime(); audit[key] = value
    (out/trainer.RUNTIME_FILE).write_text(json.dumps(audit))
    with pytest.raises(ValueError): trainer.finalize_manifest(out, child_returncode=0)
    assert json.loads((out/"ablation_manifest.json").read_text())["trained_policy_claimed"] is False


def test_failed_owned_child_never_becomes_completed(plan, managers):
    out = complete(plan, managers)
    result = trainer.finalize_manifest(out, child_returncode=7)
    assert result["status"] == "failed" and not result["trained_policy_claimed"]


def test_partial_budget_cannot_finalize(plan, managers):
    out = complete(plan, managers)
    status = json.loads((out/"run_status.json").read_text()); status["environment_transitions"] = 47
    (out/"run_status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="budget"): trainer.finalize_manifest(out, child_returncode=0)


def test_pin_immutable_collector_and_historical_trainer():
    assert trainer.sha256(trainer.torque.__file__) == trainer.COLLECTOR_SHA256
    assert trainer.sha256(trainer.ROOT/"scripts/train_beyondmimic_style.py") == trainer.BASELINE_TRAINER_SHA256


@pytest.mark.parametrize("extra", [["--resume"], ["--checkpoint", "old.pt"], ["--action-rate-weight", "-.1"],
    ["--leg-torque-rate-weight", "-.01"], ["--seed", "7"], ["--num-envs", "2"], ["--smoke-steps", "8"]])
def test_prepare_rejects_old_protocol_or_resume_before_loading_source(tmp_path, extra):
    args = ["--baseline-training-dir", str(tmp_path/"baseline"), "--source-root", str(tmp_path/"source"),
        "--motion-file", "motion.npz", "--asset-path", "robot.usda", "--task-profile", "profile.json",
        "--output-dir", str(tmp_path/"fresh"), *extra]
    with pytest.raises(ValueError): trainer.prepare_plan(args)
