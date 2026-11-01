"""Preparation/finalization tests, with no simulator or training execution."""
from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_ppo_torque_smoothness as trainer


def action_rate_l2():
    pass


action_rate_l2.__module__ = "isaaclab.envs.mdp.rewards"


@pytest.fixture
def managers(monkeypatch):
    package, module = ModuleType("isaaclab"), ModuleType("isaaclab.managers")
    class Base:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
    module.ManagerTermBase = Base
    module.RewardTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", module)
    return module


@pytest.fixture
def plan(tmp_path):
    return {"weight": -.01, "output_dir": tmp_path, "inherited_argv": [], "smoke_only": True,
            "num_envs": 2, "iterations": 1, "expected_policy_steps": 24,
            "expected_environment_transitions": 48}


def cfg():
    return SimpleNamespace(rewards=SimpleNamespace(action_rate_l2=SimpleNamespace(weight=-.1, func=action_rate_l2)),
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={"legs": SimpleNamespace(stiffness=500., damping=10.)})))


def configured(plan, managers):
    value = cfg(); recipe = {"historical_DR": True}
    result = trainer.make_pre_env_hook(lambda *args, **kwargs: recipe, plan)(value)
    assert result is recipe
    return value


def test_add_one_reward_preserving_baseline_action_rate(plan, managers):
    configuration = configured(plan, managers)
    assert configuration.rewards.action_rate_l2.weight == -.1
    assert configuration.rewards.leg_applied_torque_rate.weight == -.01
    report = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())
    assert report["additional_reward"]["delta_torque_scale_nm"] == 20.
    assert report["additional_reward"]["denominator_after_reset"] == 4
    assert report["additional_reward"]["wheel_cost_included"] is False
    assert report["baseline_action_rate_weight"] == -.1
    assert report["trained_policy_claimed"] is False
    assert report["prepared_method_runtime_verified"] is False
    assert report["new_zero_yaw_or_beta_target"] is False


@pytest.mark.parametrize("change", ["action_weight", "kp", "kd", "duplicate"])
def test_pre_env_guard_rejects_nonisolated_configuration(plan, managers, change):
    value = cfg()
    if change == "action_weight": value.rewards.action_rate_l2.weight = -.3
    if change == "kp": value.scene.robot.actuators["legs"].stiffness = 300.
    if change == "kd": value.scene.robot.actuators["legs"].damping = 30.
    if change == "duplicate": value.rewards.leg_applied_torque_rate = object()
    with pytest.raises(ValueError):
        trainer.make_pre_env_hook(lambda *args, **kwargs: {}, plan)(value)


def test_reward_reset_before_post_env_install_is_safe(plan, managers):
    configuration = configured(plan, managers)
    env = SimpleNamespace()
    term = configuration.rewards.leg_applied_torque_rate.func(None, env)
    term.reset([0])
    with pytest.raises(ValueError, match="explicitly installed"):
        term(env)


def test_runtime_audit_written_before_underlying_close_and_posthook_idempotent(plan, monkeypatch):
    events = []
    class Handler:
        def audit(self, *, check_device_health=False):
            events.append("audit")
            return {"device_health_finite": True, "closed": False}
        def close(self): events.append("detach")
    handler = Handler()
    monkeypatch.setattr(trainer.torque, "install_accumulator", lambda env: handler)
    def old_close():
        assert (plan["output_dir"]/"torque_rate_runtime_audit.json").is_file()
        events.append("old_close")
    env = SimpleNamespace(close=old_close)
    old_audit = lambda env, recipe: recipe
    hook = trainer.make_post_env_hook(old_audit, plan)
    recipe = {"DR": "unchanged"}
    assert hook(env, recipe) is recipe
    first = env.close
    assert hook(env, recipe) is recipe and env.close is first
    env.close()
    assert events == ["audit", "detach", "old_close"]
    assert env.close is old_close
    assert not hasattr(env, trainer.CLOSE_MARKER)
    report = json.loads((plan["output_dir"]/"torque_rate_runtime_audit.json").read_text())
    assert report["closed"] is True


def completed_fixture(plan, managers):
    configured(plan, managers)
    output = plan["output_dir"]
    metadata = json.loads((output/"ablation_manifest.json").read_text())
    (output/"run_status.json").write_text(json.dumps({"status": "smoke_passed", "trained_policy_claimed": False,
                                                   "environment_transitions": 48}))
    (output/"model_smoke.pt").write_bytes(b"unit fixture, not a trained checkpoint")
    (output/"manifest.json").write_text(json.dumps({"task_profile_sha256": trainer.BASELINE_PROFILE_SHA256}))
    runtime = {"module_sha256": metadata["sampler_module_sha256"], "wrapper_sha256": metadata["wrapper_sha256"],
        "steps_started": 24, "steps_completed": 24, "unique_cost_windows": 24,
        "physics_samples_total": 96, "steps_failed": 0, "samples_last_window": 4,
        "device_health_finite": True, "closed": True, "num_envs": 2,
        "delta_torque_scale_nm": 20., "wheel_cost_included": False}
    (output/"torque_rate_runtime_audit.json").write_text(json.dumps(runtime))
    return output


def test_parent_finalizer_verifies_smoke_without_claiming_policy(plan, managers):
    output = completed_fixture(plan, managers)
    result = trainer.finalize_manifest(output, child_returncode=0)
    assert result["status"] == "smoke_passed" and result["trained_policy_claimed"] is False
    assert result["prepared_method_runtime_verified"] is True
    assert result["actual_environment_transitions"] == 48
    assert result["checkpoint_sha256"] == trainer.sha256(output/"model_smoke.pt")
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["baseline_action_rate_weight"] == -.1
    assert manifest["additional_training_reward_terms"][trainer.TERM_NAME]["weight"] == -.01
    assert result["final_training_manifest_sha256"] == trainer.sha256(output/"manifest.json")
    assert trainer.finalize_manifest(output, child_returncode=0)["checkpoint_sha256"] == result["checkpoint_sha256"]


@pytest.mark.parametrize("key,value", [("steps_completed", 23), ("physics_samples_total", 95),
    ("unique_cost_windows", 23), ("steps_failed", 1), ("device_health_finite", False),
    ("closed", False), ("wheel_cost_included", True), ("delta_torque_scale_nm", 80.)])
def test_parent_finalizer_rejects_partial_or_changed_sampler(plan, managers, key, value):
    output = completed_fixture(plan, managers)
    path = output/"torque_rate_runtime_audit.json"
    runtime = json.loads(path.read_text()); runtime[key] = value
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        trainer.finalize_manifest(output, child_returncode=0)
    assert json.loads((output/"ablation_manifest.json").read_text())["trained_policy_claimed"] is False


def test_nonzero_child_exit_is_failure_even_if_fake_complete_files_exist(plan, managers):
    output = completed_fixture(plan, managers)
    result = trainer.finalize_manifest(output, child_returncode=7)
    assert result["status"] == "failed" and result["trained_policy_claimed"] is False
    assert result["child_returncode"] == 7


def test_formal_completion_requires_full_12288000_transition_scope(plan, managers):
    plan.update(smoke_only=False, num_envs=512, iterations=1000,
                expected_policy_steps=24000, expected_environment_transitions=12288000)
    output = completed_fixture(plan, managers)
    (output/"run_status.json").write_text(json.dumps({"status": "training_completed", "trained_policy_claimed": True,
                                                   "environment_transitions": 12288000}))
    (output/"model_final.pt").write_bytes(b"formal-sized test fixture, not an actual policy")
    path = output/"torque_rate_runtime_audit.json"
    runtime = json.loads(path.read_text())
    runtime.update(steps_started=24000, steps_completed=24000, unique_cost_windows=24000,
                   physics_samples_total=96000, num_envs=512)
    path.write_text(json.dumps(runtime))
    result = trainer.finalize_manifest(output, child_returncode=0)
    assert result["trained_policy_claimed"] is True
    assert result["actual_environment_transitions"] == 12288000


def test_no_checkpoint_no_completion_claim(plan, managers):
    output = completed_fixture(plan, managers)
    (output/"model_smoke.pt").unlink()
    with pytest.raises(ValueError, match="save"):
        trainer.finalize_manifest(output, child_returncode=0)


def test_empty_checkpoint_never_counts_as_completion(plan, managers):
    output = completed_fixture(plan, managers)
    (output/"model_smoke.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        trainer.finalize_manifest(output, child_returncode=0)


def arguments(tmp_path):
    return ["--source-root", str(tmp_path/"source"), "--motion-file", "motion.npz", "--asset-path", "robot.usda",
            "--task-profile", "profile.json", "--output-dir", str(tmp_path/"fresh")]


@pytest.mark.parametrize("value", ["nan", "inf", "0", "1", "-1"])
def test_invalid_new_reward_weight_rejected_before_import(tmp_path, value):
    with pytest.raises(ValueError, match="weight must"):
        trainer.prepare_plan(["--leg-torque-rate-weight", value, *arguments(tmp_path)])


@pytest.mark.parametrize("extra", [["--action-rate-weight", "-.3"], ["--resume"], ["--seed", "7"],
    ["--reference-id", "sa29"], ["--num-envs", "2"], ["--ground-friction", ".7"], ["--smoke-steps", "8"]])
def test_nonisolated_configuration_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        trainer.prepare_plan([*arguments(tmp_path), *extra])


def test_historical_file_sha_gate_runs_before_delegate_import(tmp_path):
    with pytest.raises(ValueError, match="SHA mismatch"):
        trainer.prepare_plan(arguments(tmp_path))


def test_existing_output_rejected(tmp_path):
    (tmp_path/"fresh").mkdir()
    with pytest.raises(ValueError, match="Preserve existing"):
        trainer.prepare_plan(arguments(tmp_path))


def test_historical_trainer_and_first_wrapper_are_unchanged():
    assert trainer.sha256(trainer.ROOT/"scripts/train_beyondmimic_style.py") == trainer.BASELINE_TRAINER_SHA256
    assert trainer.sha256(trainer.ROOT/"scripts/train_ppo_smoothness.py") == "a71f9c9fe84919a27ceadae499af9dc4b5ab14c18bfc04fcc42fab3829556093"
