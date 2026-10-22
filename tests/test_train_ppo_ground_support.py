"""CPU-only preparation/finalization tests; no Isaac job or policy inference."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_ppo_ground_support as t


def action_rate_l2():
    pass


def grounded_reward():
    pass


action_rate_l2.__module__ = "isaaclab.envs.mdp.rewards"
grounded_reward.__module__ = "training.tron1_drift"


@pytest.fixture
def module(monkeypatch):
    def configure(scene):
        assert not hasattr(scene, "ground_support_L")
        scene.ground_support_L = "left-ground-only"
        scene.ground_support_R = "right-ground-only"
    def term(**kwargs):
        return NS(func="new-ground-support-function", params={"scale_fraction": kwargs["scale_fraction"]}, weight=kwargs["weight"])
    module = NS(__file__=t.__file__, configure_ground_support_sensors=configure, make_reward_term=term,
                CLOCK_HELPER_SHA256=t.sha256(t.ROOT / "scripts/ppo_leg_torque_smoothness.py"))
    monkeypatch.setattr(t, "_COLLECTOR", module)
    return module


@pytest.fixture
def plan(tmp_path, module):
    return {"output_dir": tmp_path, "inherited_argv": [], "smoke_only": True,
        "num_envs": 2, "iterations": 1, "expected_policy_steps": 24, "expected_environment_transitions": 48,
        "baseline_metadata_sha256": dict(t.BASELINE_METADATA), "sampler_module_sha256": t.sha256(module.__file__)}


def cfg():
    return NS(rewards=NS(action_rate_l2=NS(func=action_rate_l2, weight=-.1),
                         drift_grounded=NS(func=grounded_reward, weight=.4), motion_xy="unchanged"),
              scene=NS(robot=NS(actuators={"legs": NS(stiffness=500., damping=10.),
                                          "wheels": NS(stiffness=0., damping=0.)})),
              sim=NS(dt=.005), decimation=4)


def configured(plan):
    configuration = cfg()
    recipe = {"historical_DR": "unchanged"}
    assert t.make_pre_env_hook(lambda *a, **kw: recipe, plan)(configuration) is recipe
    return configuration


def test_replace_existing_grounded_not_add_term(plan):
    configuration = cfg()
    terms_before = set(vars(configuration.rewards))
    original = copy.deepcopy(configuration)
    t.make_pre_env_hook(lambda *a, **kw: {}, plan)(configuration)
    assert set(vars(configuration.rewards)) == terms_before
    assert configuration.rewards.drift_grounded.weight == .4
    assert configuration.rewards.drift_grounded.params == {"scale_fraction": .20}
    assert configuration.rewards.action_rate_l2.weight == -.5
    assert configuration.rewards.motion_xy == original.rewards.motion_xy
    assert configuration.scene.robot == original.scene.robot
    assert configuration.sim == original.sim and configuration.decimation == 4
    metadata = json.loads((plan["output_dir"] / "ablation_manifest.json").read_text())
    assert metadata["reward_replacement"]["additional_reward_term"] is False
    assert metadata["baseline_action_rate_weight"] == -.5
    assert metadata["source_default_action_rate_weight"] == -.1
    assert metadata["baseline_checkpoint_sha256"] == t.BASELINE_CHECKPOINT_SHA256
    assert metadata["prepared_method_runtime_verified"] is False
    assert metadata["initialization"] == "from_scratch_no_checkpoint_resume"
    assert metadata["new_absolute_height_target"] is False
    assert metadata["new_zero_yaw_or_beta_target"] is False


@pytest.mark.parametrize("change", ["action", "grounded_weight", "grounded_func", "kp", "kd", "clock"])
def test_strict_pre_env_guards(plan, change):
    configuration = cfg()
    if change == "action": configuration.rewards.action_rate_l2.weight = -.5
    if change == "grounded_weight": configuration.rewards.drift_grounded.weight = 1.
    if change == "grounded_func": configuration.rewards.drift_grounded.func = action_rate_l2
    if change == "kp": configuration.scene.robot.actuators["legs"].stiffness = 250.
    if change == "kd": configuration.scene.robot.actuators["legs"].damping = 5.
    if change == "clock": configuration.sim.dt = .0025
    with pytest.raises(ValueError):
        t.make_pre_env_hook(lambda *a, **kw: {}, plan)(configuration)


def test_postenv_after_material_dr_and_audit_before_close(plan, module):
    events = []
    class Handler:
        def audit(self, *, check_device_health=False):
            assert check_device_health
            events.append("force_audit")
            return {"device_health_finite": True}
        def close(self): events.append("detach")
    handler = Handler()
    def install(env):
        assert events == ["original_material_DR_audit"]
        events.append("install_after_startupDR")
        return handler
    module.install_ground_support_accumulator = install
    def original_audit(env, recipe):
        events.append("original_material_DR_audit")
        return recipe
    def original_close():
        assert (plan["output_dir"] / t.RUNTIME_FILE).is_file()
        events.append("underlying_close")
    env = NS(close=original_close)
    hook = t.make_post_env_hook(original_audit, plan)
    recipe = {"DR": "unchanged"}
    assert hook(env, recipe) is recipe
    env.close()
    assert events == ["original_material_DR_audit", "install_after_startupDR", "force_audit", "detach", "underlying_close"]
    assert env.close is original_close and not hasattr(env, t.CLOSE_MARKER)


def completed(plan):
    configured(plan)
    output = plan["output_dir"]
    metadata = json.loads((output / "ablation_manifest.json").read_text())
    (output / "run_status.json").write_text(json.dumps({"status": "smoke_passed", "trained_policy_claimed": False,
                                                      "environment_transitions": 48, "policy_steps_completed": 24}))
    (output / "model_smoke.pt").write_bytes(b"test fixture; not a trained policy")
    (output / "manifest.json").write_text(json.dumps({"task_profile_sha256": t.BASELINE_PROFILE_SHA256,
        "source_training_modules_sha256": t.BASELINE_SOURCES, "motion_sha256": t.BASELINE_MOTION_SHA256,
        "asset_sha256": t.BASELINE_ASSET_SHA256, "trainer_sha256": t.BASELINE_TRAINER_SHA256}))
    runtime = {"module_sha256": metadata["sampler_module_sha256"], "wrapper_sha256": metadata["wrapper_sha256"],
        "steps_started": 24, "steps_completed": 24, "unique_cost_windows": 24, "physics_samples_total": 96,
        "steps_failed": 0, "samples_last_window": 4, "device_health_finite": True, "closed": True, "num_envs": 2,
        "scale_fraction": .20, "raw_reward_bounds": [0., 1.], "gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]},
        "mass_kg_range": [22., 22.], "mg_N_range": [215.82, 215.82], "gravity_m_s2": [0., 0., -9.81],
        "negative_normal_force_samples": 0, "positive_normal_force_samples_left_right": [90, 90],
        "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4,
        "sensor_names": ["wheel_L_ground", "wheel_R_ground"], "wheel_body_names": ["wheel_L_Link", "wheel_R_Link"],
        "force_matrix_shape_each_sensor": [plan["num_envs"], 1, 1, 3],
        "ground_filter": {"path": "/World/ground/terrain/GroundPlane/CollisionPlane", "collision_api_verified": True},
        "cpu_fixture_only": False, "device_health_checked": True, "external_reward_weight": .4, "mass_gravity_reads": 1,
        "clock_helper_sha256": t.collector().CLOCK_HELPER_SHA256}
    (output / t.RUNTIME_FILE).write_text(json.dumps(runtime))
    return output


def test_parent_finalization_no_smoke_policy_claim(plan):
    output = completed(plan)
    result = t.finalize_manifest(output, child_returncode=0)
    assert result["status"] == "smoke_passed" and result["trained_policy_claimed"] is False
    assert result["prepared_method_runtime_verified"] is True
    assert result["checkpoint_sha256"] == t.sha256(output / "model_smoke.pt")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_reward_overrides"]["action_rate_l2"] == -.5
    assert manifest["contact_reward_replacement"]["term"] == "drift_grounded"
    assert "additional_training_reward_terms" not in manifest
    assert t.finalize_manifest(output, child_returncode=0)["checkpoint_sha256"] == result["checkpoint_sha256"]


@pytest.mark.parametrize("key,value", [("steps_completed", 23), ("physics_samples_total", 95),
    ("unique_cost_windows", 23), ("steps_failed", 1), ("device_health_finite", False), ("closed", False),
    ("scale_fraction", .30), ("raw_reward_bounds", [0., 2.]), ("negative_normal_force_samples", 1),
    ("positive_normal_force_samples_left_right", [90, 0]), ("mass_kg_range", [0., 0.]),
    ("gravity_m_s2", [0., 0., 9.81]), ("gain_readback", {"kp_range": [250., 250.], "kd_range": [10., 10.]})])
def test_partial_or_invalid_runtime_never_completion(plan, key, value):
    output = completed(plan)
    path = output / t.RUNTIME_FILE
    runtime = json.loads(path.read_text()); runtime[key] = value
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        t.finalize_manifest(output, child_returncode=0)
    assert json.loads((output / "ablation_manifest.json").read_text())["trained_policy_claimed"] is False


def test_nonzero_child_exit_fails(plan):
    output = completed(plan)
    result = t.finalize_manifest(output, child_returncode=7)
    assert result["status"] == "failed" and not result["trained_policy_claimed"]


def test_full_formal_budget_required(plan):
    plan.update(smoke_only=False, num_envs=512, iterations=1000, expected_policy_steps=24000,
                expected_environment_transitions=12288000)
    output = completed(plan)
    (output / "model_final.pt").write_bytes(b"formal budget fixture, not actual trained weights")
    (output / "run_status.json").write_text(json.dumps({"status": "training_completed", "trained_policy_claimed": True,
        "environment_transitions": 12288000, "policy_steps_completed": 24000}))
    path = output / t.RUNTIME_FILE
    runtime = json.loads(path.read_text())
    runtime.update(steps_started=24000, steps_completed=24000, unique_cost_windows=24000,
                   physics_samples_total=96000, num_envs=512, force_matrix_shape_each_sensor=[512, 1, 1, 3])
    path.write_text(json.dumps(runtime))
    result = t.finalize_manifest(output, child_returncode=0)
    assert result["trained_policy_claimed"] is True and result["actual_environment_transitions"] == 12288000


def arguments(tmp_path):
    return ["--baseline-training-dir", str(tmp_path / "baseline"), "--source-root", str(tmp_path / "source"),
        "--motion-file", "motion.npz", "--asset-path", "asset.usda", "--task-profile", "profile.json",
        "--output-dir", str(tmp_path / "fresh")]


@pytest.mark.parametrize("extra", [["--action-rate-weight", "-.3"], ["--resume"], ["--leg-kp", "250"],
    ["--leg-kd", "5"], ["--ground-support-weight", "1"], ["--scale-fraction", ".3"], ["--seed", "7"],
    ["--reference-id", "sa29"], ["--num-envs", "2"], ["--ground-friction", ".7"], ["--smoke-steps", "8"]])
def test_nonisolated_configuration_blocked(tmp_path, extra):
    with pytest.raises(ValueError):
        t.prepare_plan([*arguments(tmp_path), *extra])


def test_existing_output_preserved(tmp_path):
    (tmp_path / "fresh").mkdir()
    with pytest.raises(ValueError, match="Preserve existing"):
        t.prepare_plan(arguments(tmp_path))


def test_frozen_trainer_and_wrapper_hashes_unchanged():
    assert t.sha256(t.ROOT / "scripts/train_beyondmimic_style.py") == t.BASELINE_TRAINER_SHA256
    assert t.sha256(t.ROOT / "scripts/train_ppo_smoothness.py") == t.ACTION_WRAPPER_SHA256


def test_local_baseline_metadata_pins():
    baseline = t.ROOT / "outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training"
    if not baseline.is_dir(): pytest.skip("Historical baseline metadata unavailable")
    for name, digest in t.BASELINE_METADATA.items():
        assert t.sha256(baseline / name) == digest


def test_postenv_hook_precedes_first_wrapped_reset_in_frozen_trainer():
    source = (t.ROOT / "scripts/train_beyondmimic_style.py").read_text()
    assert source.index("material_audit = audit_material_draw(env, recipe)") < source.index("obs, _ = wrapped.reset()")


def prepared_inputs(tmp_path, monkeypatch):
    source, baseline = tmp_path / "source", tmp_path / "baseline"
    baseline.mkdir()
    expected = {}
    for name, digest in t.BASELINE_SOURCES.items():
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture")
        expected[path] = digest
    ablation = {"checkpoint_sha256": t.BASELINE_CHECKPOINT_SHA256, "reward_override": {"weight": -.5},
                "actual_environment_transitions": 12288000, "initialization": "from_scratch_no_checkpoint_resume"}
    for name, digest in t.BASELINE_METADATA.items():
        path = baseline / name
        path.write_text(json.dumps(ablation if name == "ablation_manifest.json" else {}))
        expected[path] = digest
    flags = {"--motion-file": (tmp_path / "motion.npz", t.BASELINE_MOTION_SHA256),
             "--asset-path": (tmp_path / "asset.usda", t.BASELINE_ASSET_SHA256),
             "--task-profile": (tmp_path / "profile.json", t.BASELINE_PROFILE_SHA256)}
    argv = ["--baseline-training-dir", str(baseline), "--source-root", str(source),
            "--output-dir", str(tmp_path / "fresh")]
    for flag, (path, digest) in flags.items():
        path.write_text("fixture")
        expected[path] = digest
        argv += [flag, str(path)]
    actual_sha = t.sha256
    monkeypatch.setattr(t, "sha256", lambda path: expected.get(Path(path), actual_sha(path)))
    return argv, expected


@pytest.mark.parametrize("smoke", [False, True])
def test_preparation_exact_formal_or_smoke_scope(tmp_path, monkeypatch, module, smoke):
    argv, _ = prepared_inputs(tmp_path, monkeypatch)
    if smoke: argv += ["--smoke-only", "--num-envs", "2", "--iterations", "1"]
    result = t.prepare_plan(argv)
    assert result["expected_policy_steps"] == (24 if smoke else 24000)
    assert result["expected_environment_transitions"] == (48 if smoke else 12288000)
    assert result["smoke_only"] is smoke
    assert result["inherited_argv"][-2:] == ["--smoke-steps", "0"]
    assert "--baseline-training-dir" not in result["inherited_argv"]
    assert not result["output_dir"].exists()


def test_preparation_rejects_baseline_metadata_mutation(tmp_path, monkeypatch, module):
    argv, expected = prepared_inputs(tmp_path, monkeypatch)
    expected[tmp_path / "baseline/ablation_manifest.json"] = "0" * 64
    with pytest.raises(ValueError, match="baseline metadata SHA"):
        t.prepare_plan(argv)


def test_preparation_rejects_training_source_mutation(tmp_path, monkeypatch, module):
    argv, expected = prepared_inputs(tmp_path, monkeypatch)
    expected[tmp_path / "source/training/tron1_drift.py"] = "0" * 64
    with pytest.raises(ValueError, match="Historical source/input SHA"):
        t.prepare_plan(argv)
