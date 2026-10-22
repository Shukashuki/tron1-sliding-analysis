import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contact_damping", ROOT / "scripts/eval_ppo_contact_damping.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
LEGACY = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py"


def fixture_plan(tmp_path, monkeypatch, kd="5"):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    evaluator = source / "scripts/evaluator.py"
    evaluator.write_text("pinned synthetic evaluator\n")
    train = tmp_path / "training"
    train.mkdir()
    paths = {name: train / name for name in m.PINS}
    profile = {"contact_friction": 1.0}
    paths["task_profile"].write_text(json.dumps(profile))
    paths["manifest"].write_text(json.dumps({"task_profile": profile,
        "contract": {"leg_kp": 500., "leg_kd": 10., "physics_dt": .005, "decimation": 4, "policy_fps": 50.},
        "source_training_modules_sha256": dict(m.TRAINING_DEPENDENCIES)}))
    for name in ("checkpoint", "motion_file", "asset_path", "runner_config"):
        paths[name].write_text(name)
    checkpoint_hash = m.sha256(paths["checkpoint"])
    paths["ablation_manifest"].write_text(json.dumps({"status": "training_completed", "smoke_only": False,
        "initialization": "from_scratch_no_checkpoint_resume", "reward_override": {"weight": -.5},
        "actual_environment_transitions": 12288000, "checkpoint_sha256": checkpoint_hash}))
    # The real file names are intentionally required beside the checkpoint.
    for name in ("manifest", "ablation_manifest"):
        paths[name].rename(train / (name + ".json"))
        paths[name] = train / (name + ".json")
    (train / "run_status.json").write_text(json.dumps({"status": "training_completed", "smoke_only": False,
        "environment_transitions": 12288000, "policy_steps_completed": 24000}))
    monkeypatch.setattr(m, "PINS", {name: m.sha256(path) for name, path in paths.items()})
    monkeypatch.setattr(m, "BASE_EVALUATOR_SHA256", m.sha256(evaluator))
    monkeypatch.setattr(m, "EVALUATION_DEPENDENCIES", {})
    argv = ["--legacy-evaluator", str(evaluator), "--leg-kd", kd,
        "--evaluation-contact-friction", ".7", "--num-envs", "1", "--seed", "42",
        "--record-physics", "--verify-unmasked-device", "--output-dir", str(tmp_path / "new_output")]
    for name in ("checkpoint", "motion_file", "asset_path", "task_profile", "runner_config"):
        argv += ["--" + name.replace("_", "-"), str(paths[name])]
    return argv, paths


@pytest.mark.parametrize("kd", [5., 10.])
def test_one_only_physical_config_change(kd):
    cfg = NS(scene=NS(robot=NS(actuators={"legs": NS(stiffness=500., damping=10., effort_limit=80.),
                                            "wheels": NS(stiffness=0., damping=0.)})),
             sim=NS(dt=.005), decimation=4, reward="unchanged")
    before = copy.deepcopy(cfg)
    profile = {"contact_friction": .7, "reward_weights": {"sideslip": 2.}}
    original = copy.deepcopy(profile)
    m.apply_config(cfg, profile, {"leg_kd": kd})
    assert cfg.scene.robot.actuators["legs"].damping == kd
    cfg.scene.robot.actuators["legs"].damping = 10.
    assert cfg == before
    assert profile == (original if kd == 10. else {**original, "leg_kd": kd})


def test_reject_unknown_original_gains():
    cfg = NS(scene=NS(robot=NS(actuators={"legs": NS(stiffness=500., damping=12.)})), sim=NS(dt=.005), decimation=4)
    with pytest.raises(ValueError, match="original PD"):
        m.apply_config(cfg, {"contact_friction": .7}, {"leg_kd": 5.})


def test_plan_and_input_immutability(tmp_path, monkeypatch):
    argv, paths = fixture_plan(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in paths.values()}
    plan = m.prepare_plan(argv)
    assert plan["leg_kd"] == 5.
    assert plan["audit"]["additional_training_updates"] == 0
    assert plan["audit"]["trained_action_rate_weight"] == -.5
    assert not plan["audit"]["training_at_deployment_gain_claimed"]
    assert before == {p: p.read_bytes() for p in paths.values()}
    assert "--leg-kd" not in plan["inherited_argv"]
    assert not plan["paths"]["output_dir"].exists()


@pytest.mark.parametrize("flag,value", [("--deployment-leg-kd", "5"), ("--deployment-leg-kp", "250"),
    ("--physics-dt-s", ".0025"), ("--domain-parameters", "draw.json"), ("--terrain-file", "terrain.json")])
def test_block_combined_flags(tmp_path, monkeypatch, flag, value):
    argv, _ = fixture_plan(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Blocked"):
        m.prepare_plan([*argv, flag, value])


def test_no_overwrite(tmp_path, monkeypatch):
    argv, _ = fixture_plan(tmp_path, monkeypatch)
    (tmp_path / "new_output").mkdir()
    with pytest.raises(ValueError, match="fresh"):
        m.prepare_plan(argv)


def test_mutated_checkpoint_rejected(tmp_path, monkeypatch):
    argv, paths = fixture_plan(tmp_path, monkeypatch)
    paths["checkpoint"].write_text("changed")
    with pytest.raises(ValueError, match="checkpoint"):
        m.prepare_plan(argv)


def test_contract_honest_trained_vs_deployed():
    original_profile = {"contact_friction": 1.}
    contract = {"leg_kp": 500., "leg_kd": 10., "trained_task_profile": original_profile,
                "task_profile": {"contact_friction": .7, "leg_kd": 5.}}
    m.decorate_contract(contract, {"leg_kd": 5., "audit": {"diagnostic_only": True}})
    assert contract["leg_kd"] == 5.
    assert contract["trained_task_profile"] == {"contact_friction": 1.}
    assert contract["contact_stability_diagnostic"]["diagnostic_only"]


@pytest.mark.skipif(not LEGACY.is_file(), reason="Historical source snapshot not published locally")
def test_pinned_source_materialization_and_legacy_guards():
    source = LEGACY.read_text()
    generated = m.materialize_source(source, LEGACY.parent.parent)
    ast.parse(generated)
    assert m.sha256(LEGACY) == m.BASE_EVALUATOR_SHA256
    assert 'or args.deployment_leg_kp is not None or args.deployment_leg_kd is not None' in generated
    assert 'trained_manifest.get("task_profile") != task_profile' in generated
    assert generated.index('_contact_apply_config(cfg, evaluation_profile, _contact_plan)') < generated.index('env = TerminalCaptureEnv(cfg=cfg)')
    assert 'audit_isaac_materials(env, evaluation_profile)' in generated
    assert '"only_changed_physics_field"' not in generated
    assert '"only_changed_material_field": "contact_friction"' in generated
    assert '"contact_stability_diagnostic": copy.deepcopy(_contact_plan["audit"])' in generated
    assert 'ROOT = Path(' + repr(str(LEGACY.parent.parent)) + ')' in generated
    with pytest.raises(ValueError, match="source changed"):
        m.materialize_source(source + "\n", LEGACY.parent.parent)


def test_local_training_metadata_pins():
    train = ROOT / "outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training"
    if not train.is_dir():
        pytest.skip("Private completed training metadata unavailable")
    for name in ("manifest", "ablation_manifest", "runner_config"):
        assert m.sha256(train / (name + ".json")) == m.PINS[name]


@pytest.mark.skipif(not LEGACY.is_file(), reason="Historical evaluation source unavailable")
def test_actual_training_and_evaluation_dependencies_are_separately_pinned(tmp_path, monkeypatch):
    train = ROOT / "outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training"
    if not train.is_dir():
        pytest.skip("Private training metadata unavailable")
    training = json.loads((train / "manifest.json").read_text())
    assert training["source_training_modules_sha256"] == m.TRAINING_DEPENDENCIES
    assert m.TRAINING_DEPENDENCIES != m.EVALUATION_DEPENDENCIES
    for name, digest in m.EVALUATION_DEPENDENCIES.items():
        assert m.sha256(LEGACY.parent.parent / name) == digest
    original_eval = dict(m.EVALUATION_DEPENDENCIES)
    argv, _ = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "BASE_EVALUATOR_SHA256", hashlib.sha256(LEGACY.read_bytes()).hexdigest())
    monkeypatch.setattr(m, "EVALUATION_DEPENDENCIES", original_eval)
    argv[argv.index("--legacy-evaluator") + 1] = str(LEGACY)
    plan = m.prepare_plan(argv)
    assert plan["audit"]["training_dependency_sha256"] == m.TRAINING_DEPENDENCIES
    assert plan["audit"]["evaluation_dependency_sha256"] == original_eval
    assert plan["audit"]["training_evaluation_dependencies_identical"] is False
