import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contact_stiffness", ROOT / "scripts/eval_ppo_contact_stiffness.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
LEGACY = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py"


@pytest.mark.parametrize("kp", [500., 250.])
def test_only_stiffness_config_changes(kp):
    cfg = NS(scene=NS(robot=NS(actuators={"legs": NS(stiffness=500., damping=10., effort_limit=80.),
          "wheels": NS(stiffness=0., damping=0., effort_limit=12.)})), sim=NS(dt=.005), decimation=4,
          rewards=NS(action_rate_l2=NS(weight=-.5)))
    before = copy.deepcopy(cfg)
    profile = {"contact_friction": .7, "height_regularization": {"mode": "vertical_acceleration_only"}}
    original_profile = copy.deepcopy(profile)
    m.apply_config(cfg, profile, {"leg_kp": kp})
    assert cfg.scene.robot.actuators["legs"].stiffness == kp
    assert cfg.scene.robot.actuators["legs"].damping == 10.
    cfg.scene.robot.actuators["legs"].stiffness = 500.
    assert cfg == before
    assert profile == (original_profile if kp == 500. else {**original_profile, "leg_kp": 250.})


@pytest.mark.parametrize("kp,kd,dt", [(400., 10., .005), (500., 5., .005), (500., 10., .0025)])
def test_original_config_required(kp, kd, dt):
    cfg = NS(scene=NS(robot=NS(actuators={"legs": NS(stiffness=kp, damping=kd)})), sim=NS(dt=dt), decimation=4)
    with pytest.raises(ValueError):
        m.apply_config(cfg, {"contact_friction": .7}, {"leg_kp": 250.})


def fake_base_plan(argv):
    assert argv[:2] == ["--leg-kd", "10"]
    return {"leg_kd": 10., "inherited_argv": argv[2:], "audit": {
        "kind": "fixed_ppo_nominal_contact_damping_diagnostic", "trained_leg_kd": 10.,
        "deployment_leg_kd": 10., "unchanged_leg_kp": 500., "additional_training_updates": 0,
        "training_at_deployment_gain_claimed": False, "checkpoint_sha256": m.base.PINS["checkpoint"]}}


def test_guard_delegation_and_honest_metadata(monkeypatch):
    monkeypatch.setattr(m.base, "prepare_plan", fake_base_plan)
    plan = m.prepare_plan(["--leg-kp", "250", "--legacy-evaluator", "retained"])
    assert plan["inherited_argv"] == ["--legacy-evaluator", "retained"]
    assert plan["leg_kp"] == 250.
    assert "leg_kd" not in plan
    audit = plan["audit"]
    assert audit["kind"] == "fixed_ppo_nominal_contact_stiffness_diagnostic"
    assert audit["trained_leg_kp"] == 500. and audit["deployment_leg_kp"] == 250.
    assert audit["unchanged_leg_kd"] == 10.
    assert audit["only_changed_control_field"] == "leg_kp"
    assert audit["wrapper_sha256"] == m.base.sha256(m.__file__)
    assert audit["provenance_adapter_sha256"] == m.BASE_ADAPTER_SHA256
    assert not any(k in audit for k in ["deployment_leg_kd", "trained_leg_kd", "unchanged_leg_kp"])
    contract = {"leg_kp": 500., "leg_kd": 10., "trained_task_profile": {"contact_friction": 1.},
                "task_profile": {"contact_friction": .7, "leg_kp": 250.}}
    original = copy.deepcopy(contract)
    m.decorate_contract(contract, plan)
    assert contract["leg_kp"] == 250. and contract["leg_kd"] == 10.
    assert contract["trained_task_profile"] == original["trained_task_profile"]
    assert contract["contact_stability_diagnostic"]["additional_training_updates"] == 0


@pytest.mark.parametrize("option", ["--leg-kd", "--leg-kd=5"])
def test_damping_cannot_be_combined(option):
    with pytest.raises(ValueError, match="cannot accept"):
        m.prepare_plan(["--leg-kp", "250", option, "5"])


def test_provenance_adapter_mutation_rejected(tmp_path):
    path = tmp_path / "adapter.py"
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.load_base_adapter(path)


def test_real_training_and_evaluation_pins_unchanged():
    assert m.base.sha256(m.BASE_ADAPTER_PATH) == m.BASE_ADAPTER_SHA256
    train = ROOT / "outputs/ppo_smoothness_20261021_v2/action_rate_0p5/training"
    if not train.is_dir() or not LEGACY.is_file():
        pytest.skip("Local historical source/metadata unavailable")
    assert m.base.sha256(LEGACY) == m.base.BASE_EVALUATOR_SHA256
    for name in ("manifest", "ablation_manifest", "runner_config"):
        assert m.base.sha256(train / (name + ".json")) == m.base.PINS[name]
    assert json.loads((train / "manifest.json").read_text())["source_training_modules_sha256"] == m.base.TRAINING_DEPENDENCIES
    for name, sha in m.base.EVALUATION_DEPENDENCIES.items():
        assert m.base.sha256(LEGACY.parent.parent / name) == sha


@pytest.mark.skipif(not LEGACY.is_file(), reason="Historical evaluator unavailable")
def test_same_generated_source_guards_and_native_readback():
    generated = m.base.materialize_source(LEGACY.read_text(), LEGACY.parent.parent)
    ast.parse(generated)
    assert generated.count("_contact_apply_config(cfg, evaluation_profile, _contact_plan)") == 1
    assert generated.index("_contact_apply_config(cfg, evaluation_profile, _contact_plan)") < generated.index("env = TerminalCaptureEnv(cfg=cfg)")
    assert 'audit_isaac_materials(env, evaluation_profile)' in generated
    assert 'or args.deployment_leg_kp is not None or args.deployment_leg_kd is not None' in generated
    assert '"only_changed_material_field": "contact_friction"' in generated
    assert '"evaluator_source_sha256": sha256(__file__)' in generated


def test_bad_kp_rejected():
    with pytest.raises(SystemExit):
        m.prepare_plan(["--leg-kp", "300"])


def test_original_plan_guard_failure_propagates(monkeypatch):
    def fail(argv):
        raise ValueError("Preserve previous output: directory must be fresh")
    monkeypatch.setattr(m.base, "prepare_plan", fail)
    with pytest.raises(ValueError, match="fresh"):
        m.prepare_plan(["--leg-kp", "250"])
