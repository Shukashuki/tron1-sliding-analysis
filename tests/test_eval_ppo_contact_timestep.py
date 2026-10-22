"""CPU stages of the pinned clock adapter; no app, GPU, policy or jobs."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"scripts"))
import eval_ppo_contact_timestep as module
LEGACY = ROOT/"outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py"


def clock_helper():
    path = LEGACY.parent.parent/"training/tron1_physics_clock.py"
    spec = importlib.util.spec_from_file_location("test_original_timestep_clock", path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def fake_base_plan(argv):
    assert argv[:2] == ["--leg-kd", "10"] and "--physics-dt-s" not in argv and "--physical-gpu-index" not in argv
    return {"leg_kd": 10., "legacy_evaluator": LEGACY, "inherited_argv": argv[2:], "audit": {
        "deployment_leg_kd": 10., "trained_leg_kd": 10., "diagnostic_only": True,
        "input_sha256": dict(module.base.PINS), "evaluation_dependency_sha256": module.base.EVALUATION_DEPENDENCIES}}


def plan(dt, monkeypatch, index=3):
    monkeypatch.setattr(module.base, "prepare_plan", fake_base_plan)
    return module.prepare_plan(["--physics-dt-s", str(dt), "--physical-gpu-index", str(index),
        "--device", f"cuda:{index}", "--kit_args", module.devices.kit_args_for(index)])


@pytest.mark.parametrize("dt,decimation", [(.005, 4), (.0025, 8)])
def test_prepare_removes_clock_for_old_provenance_then_forwards_declared_dt(dt, decimation, monkeypatch):
    result = plan(dt, monkeypatch)
    assert result["physics_dt_s"] == dt and result["decimation"] == decimation
    assert result["inherited_argv"][-2:] == ["--physics-dt-s", str(dt)]
    assert result["audit"]["unchanged_policy_fps"] == 50.
    assert result["audit"]["low_level_explicit_pd_update_hz"] == 1./dt
    assert not result["audit"]["leg_target_filter_installed"] and not result["audit"]["additional_torque_reward"]


@pytest.mark.parametrize("index", [1, 3, 4, 5])
def test_declared_selected_device_allowed_no_gpu_calls(index, monkeypatch):
    result = plan(.0025, monkeypatch, index)
    assert result["audit"]["requested_device"] == f"cuda:{index}"


def test_materialization_removes_only_main_clock_conjunct_and_replaces_only_final_call():
    source = LEGACY.read_text(); result = module.materialize_source(source, LEGACY.parent.parent)
    ast.parse(result)
    assert "or args.physics_dt_s is not None or not args.record_physics" not in result
    assert "or not args.record_physics" in result
    assert "or args.domain_parameters is not None" in result
    assert 'if args.physics_dt_s is not None or args.domain_parameters is not None:' in result
    assert result.count("_timestep_refine_export(") == 1
    assert result.count("refined_contract = apply_physics_clock_to_contract(") == 1
    assert result.count("configure_physics_clock(cfg, args.physics_dt_s)") == 1
    assert '"contact_timestep_diagnostic": copy.deepcopy(_contact_plan["audit"]),' in result
    assert module.base.sha256(LEGACY) == module.base.BASE_EVALUATOR_SHA256


def profiles():
    profile = {"task": "car_drift", "contact_friction": 1., "reward_weights": {"sideslip": 2., "grounded": .4},
        "height_regularization": {"mode": "vertical_acceleration_only"}}
    trained = {"actor_dim": 51, "action_dim": 8, "policy_fps": 50., "physics_dt": .005, "decimation": 4, "task_profile": profile}
    actual = copy.deepcopy(trained); actual["task_profile"]["contact_friction"] = .7
    actual["trained_task_profile"] = copy.deepcopy(profile)
    actual["evaluation_contact_friction_override"] = {"trained_contact_friction": 1., "evaluation_contact_friction": .7,
        "only_changed_material_field": "contact_friction"}
    return trained, actual


@pytest.mark.parametrize("dt,decimation", [(.005, 4), (.0025, 8)])
def test_early_config_and_final_export_stages_call_original_strict_helper(dt, decimation, monkeypatch):
    p = plan(dt, monkeypatch); helper = clock_helper(); trained, actual = profiles()
    early = helper.apply_physics_clock_to_contract(trained, dt, trained_contract=trained)
    assert (early["physics_dt"], early["decimation"]) == (dt, decimation)
    cfg = NS(sim=NS(dt=.005, render_interval=4), decimation=4, scene=NS(contact_forces=NS(update_period=.005),
        robot=NS(spawn=NS(articulation_props=NS(solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
            actuators={"legs": NS(stiffness=500., damping=10.)})))
    module.apply_config(cfg, actual["task_profile"], p)
    helper.configure_physics_clock(cfg, dt)
    assert (cfg.sim.dt, cfg.decimation, cfg.scene.contact_forces.update_period) == (dt, decimation, dt)
    trained_before, actual_before = copy.deepcopy(trained), copy.deepcopy(actual)
    with pytest.raises(ValueError, match="trained task profile"):
        helper.apply_physics_clock_to_contract(actual, dt, trained_contract=trained)
    out = module.refine_export_contract(actual, dt, trained_contract=trained,
        clock_helper=helper.apply_physics_clock_to_contract, plan=p)
    assert helper.validate_physics_clock(out) == (dt, decimation)
    assert actual == actual_before and trained == trained_before
    assert out["task_profile"]["contact_friction"] == .7 and out["trained_task_profile"]["contact_friction"] == 1.
    assert out["evaluation_contact_friction_override"] == actual_before["evaluation_contact_friction_override"]
    bridge = out["contact_timestep_diagnostic"]["final_export_profile_bridge"]
    assert bridge["trained_task_profile_canonical_sha256"] != bridge["evaluation_task_profile_canonical_sha256"]
    assert not bridge["trained_profile_modified"] and not bridge["runtime_material_modified_by_bridge"]
    assert out["physics_clock_refinement"]["trained_physics_dt_s"] == .005


@pytest.mark.parametrize("change", ["reward", "trained_mu", "eval_mu", "override", "actor_dimension"])
def test_profile_bridge_does_not_suppress_other_contract_assertions(monkeypatch, change):
    p = plan(.0025, monkeypatch); helper = clock_helper(); trained, actual = profiles()
    if change == "reward": actual["task_profile"]["reward_weights"]["sideslip"] = 0.
    if change == "trained_mu": trained["task_profile"]["contact_friction"] = .7
    if change == "eval_mu": actual["task_profile"]["contact_friction"] = .15
    if change == "override": actual["evaluation_contact_friction_override"]["evaluation_contact_friction"] = .15
    if change == "actor_dimension": actual["actor_dim"] = 52
    with pytest.raises(ValueError):
        module.refine_export_contract(actual, .0025, trained_contract=trained, clock_helper=helper.apply_physics_clock_to_contract, plan=p)


@pytest.mark.parametrize("extra", [["--alpha", ".5"], ["--leg-kd", "5"], ["--solver-velocity-iterations", "8"],
    ["--physics-dt-s", ".005"]])
def test_combined_interventions_or_duplicate_dt_rejected_before_launch(monkeypatch, extra):
    monkeypatch.setattr(module.base, "prepare_plan", fake_base_plan)
    args = ["--physics-dt-s", ".0025", "--physical-gpu-index", "3", "--device", "cuda:3", "--kit_args", module.devices.kit_args_for(3), *extra]
    with pytest.raises(ValueError): module.prepare_plan(args)
