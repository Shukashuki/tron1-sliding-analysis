"""Synthetic target-telemetry fixtures; no actor, GPU or physics is run."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import copy
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"scripts"))
import eval_ppo_leg_target_filter as m
LEGACY = ROOT/"outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_friction_isaac.py"
BASE_REPORT = ROOT/"outputs/contact_damping_20261021_v2/kd10/report.json"


def telemetry(alpha, requested=None, initial=None):
    if requested is None: requested = np.random.default_rng(42).uniform(-.1, .1, (349, 8)).astype(np.float32)
    if initial is None: initial = np.zeros(6, dtype=np.float32)
    targets = requested[:, :6]+initial; filtered = targets.copy()
    if alpha < 1:
        history = initial.copy()
        for index in range(349):
            filtered[index] = .5*history+.5*targets[index]; history = filtered[index].copy()
    control = {"time_s": 1.25+np.arange(349)*.02, "sim_step_counter": 100+np.arange(349)*4,
        "native_step_index": 50+np.arange(349)*4, "requested_actions_clipped": requested,
        "requested_targets_rad": targets, "filtered_targets_rad": filtered,
        "last_action_unchanged": np.ones(349, dtype=bool), "initialized_from_measured": np.r_[True, np.zeros(348, dtype=bool)]}
    physics = {"time_s": 1.25+np.arange(1396)*.005, "sim_step_counter": 101+np.arange(1396),
        "native_step_index": 50+np.arange(1396), "control_index": np.repeat(np.arange(349), 4),
        "substep_index": np.tile(np.arange(1, 5), 349), "applied_targets_rad": np.repeat(filtered, 4, axis=0),
        "filtered_targets_rad": np.repeat(filtered, 4, axis=0)}
    runtime = {"control_steps": 349, "physics_apply_calls": 1396, "samples_last_control": 4, "failures": 0,
        "requested_action_history_unchanged": True, "wheel_requests_unchanged": True, "applied_targets_match_filtered": True,
        "device_health_finite": True, "closed": True, "alpha_one_exact_identity": True, "alpha": alpha,
        "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4, "module_sha256": m.FILTER_MODULE_SHA256,
        "cpu_fixture_only": False, "num_envs": 1, "device": "cuda:1", "device_health_checked": True,
        "source_api_provenance": {"cpu_fixture_only": False, "source_pins_checked": True},
        "additional_physics_steps": 0, "additional_reset_calls": 0, "additional_training_updates": 0,
        "filter_update_rate_hz": 50, "apply_rate_hz": 200, "wheel_outputs_filtered": False}
    return control, physics, runtime


def fake_base_plan(argv):
    assert argv[:2] == ["--leg-kd", "10"]
    return {"leg_kd": 10., "inherited_argv": argv[2:], "audit": json.loads(BASE_REPORT.read_text())["contact_stability_diagnostic"]}


def make_plan(alpha, monkeypatch, physical_gpu_index=1, *, device=None, kit_args=None):
    monkeypatch.setattr(m.base, "prepare_plan", fake_base_plan)
    return m.prepare_plan(["--alpha", str(alpha), "--physical-gpu-index", str(physical_gpu_index),
        "--device", device or f"cuda:{physical_gpu_index}", "--kit_args", kit_args or m.kit_args_for(physical_gpu_index)])


@pytest.mark.parametrize("alpha", [1., .5])
def test_final_not_installation_proof_and_cfg_unchanged(alpha, monkeypatch, tmp_path):
    plan = make_plan(alpha, monkeypatch); plan["paths"] = {"output_dir": tmp_path}
    cfg = NS(scene=NS(robot=NS(spawn=NS(articulation_props=NS(solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
        actuators={"legs": NS(stiffness=500., damping=10.)})), sim=NS(dt=.005), decimation=4)
    before = copy.deepcopy(cfg); profile = {"contact_friction": .7}; m.apply_config(cfg, profile, plan)
    assert cfg == before and profile == {"contact_friction": .7}
    with pytest.raises(ValueError, match="Final—not installation"):
        m.decorate_contract({"leg_kp": 500., "leg_kd": 10.}, plan)
    control, physics, runtime = telemetry(alpha)
    class Handler:
        def control_arrays(self): return control
        def physics_arrays(self): return physics
        def close(self): self.closed = True
        def audit(self, *, check_device_health):
            assert check_device_health and self.closed; return runtime
    final = m.finalize_filter(None, Handler(), plan)
    contract = {"leg_kp": 500., "leg_kd": 10.}; m.decorate_contract(contract, plan)
    assert final["finalized_after_rollout"] is True
    assert final["verification_status"] == "verified"
    assert contract["leg_target_filter_diagnostic"]["final_runtime_evidence"]["control_steps"] == 349
    assert "contact_stability_diagnostic" not in contract and "contact_numerics_diagnostic" not in contract
    assert contract["actor_export_includes_leg_target_filter"] is False and contract["previous_action_is_requested_not_filtered"] is True
    assert plan["audit"]["deployment_requires_same_external_filter"] is (alpha < 1)
    for name in (m.CONTROL_FILE, m.PHYSICS_FILE, m.RUNTIME_FILE): assert (tmp_path/name).is_file()


@pytest.mark.parametrize("key,value", [("control_steps", 0), ("physics_apply_calls", 1395), ("samples_last_control", 3),
    ("requested_action_history_unchanged", False), ("wheel_requests_unchanged", False), ("applied_targets_match_filtered", False),
    ("device_health_finite", False), ("closed", False), ("cpu_fixture_only", True), ("additional_reset_calls", 1), ("wheel_outputs_filtered", True)])
def test_invalid_runtime_proof_rejected(key, value):
    control, physics, runtime = telemetry(.5); runtime[key] = value
    with pytest.raises(ValueError): m.verify_telemetry(control, physics, runtime, .5)


@pytest.mark.parametrize("bad", ["history", "recurrence", "applied", "substep", "time", "counter", "reset_splice", "nan"])
def test_targets_history_all_applies_and_clocks_verified(bad):
    control, physics, runtime = telemetry(.5)
    if bad == "history": control["last_action_unchanged"][1] = False
    if bad == "recurrence": control["filtered_targets_rad"][3, 0] += .01
    if bad == "applied": physics["applied_targets_rad"][5, 0] += .01
    if bad == "substep": physics["substep_index"][0] = 0
    if bad == "time": control["time_s"][3] += .005
    if bad == "counter": physics["sim_step_counter"][3] += 1
    if bad == "reset_splice": control["initialized_from_measured"][10] = True
    if bad == "nan": control["filtered_targets_rad"][1, 0] = np.nan
    with pytest.raises(ValueError): m.verify_telemetry(control, physics, runtime, .5)


def test_first_history_is_measured_and_alpha1_is_bit_identity():
    initial = np.arange(6, dtype=np.float32)/10; control, physics, runtime = telemetry(.5, initial=initial)
    m.verify_telemetry(control, physics, runtime, .5, initial_joint_position=initial)
    with pytest.raises(ValueError, match="measured initial"):
        m.verify_telemetry(control, physics, runtime, .5, initial_joint_position=initial+.01)
    control, physics, runtime = telemetry(1.); control["filtered_targets_rad"][0, 0] += np.float32(1e-7)
    physics["filtered_targets_rad"][:4] = control["filtered_targets_rad"][0]; physics["applied_targets_rad"][:4] = control["filtered_targets_rad"][0]
    with pytest.raises(ValueError, match="exact requested-target"):
        m.verify_telemetry(control, physics, runtime, 1.)


def test_source_hooks_final_capture_and_cleanup_order():
    generated = m.materialize_source(LEGACY.read_text(), LEGACY.parent.parent); ast.parse(generated)
    assert generated.count("_leg_filter_install(env, _contact_plan)") == 1
    assert generated.index("_leg_filter_install(env, _contact_plan)") < generated.index("wrapped = RslRlVecEnvWrapper(")
    assert generated.index("_leg_filter_finalize(env, _leg_filter_handler, _contact_plan)") < generated.index("        contract = observation_action_contract()", generated.index("        summary = summarize_episodes"))
    assert '"contact_stability_diagnostic"' not in generated and '"leg_target_filter_diagnostic"' in generated
    assert "_leg_filter_handler.close()" in generated


@pytest.mark.parametrize("flag", ["--leg-kp=250", "--leg-kd=5", "--solver-velocity-iterations=8"])
def test_combined_changes_blocked(flag):
    with pytest.raises(ValueError, match="blocked"):
        m.prepare_plan(["--alpha", ".5", "--device", "cuda:1", "--kit_args", m.kit_args_for(1), flag])


def test_gpu3_explicit_device_kit_and_runtime_binding(monkeypatch):
    plan = make_plan(.5, monkeypatch, 3)
    assert plan["requested_device"] == plan["audit"]["requested_device"] == "cuda:3"
    assert plan["physical_gpu_index"] == plan["audit"]["physical_gpu_index_requested"] == 3
    control, physics, runtime = telemetry(.5); runtime["device"] = "cuda:3"
    m.verify_telemetry(control, physics, runtime, .5, requested_device=plan["requested_device"])
    with pytest.raises(ValueError, match="GPU/native API"):
        m.verify_telemetry(control, physics, runtime, .5)
    with pytest.raises(ValueError, match="Explicit device"):
        make_plan(.5, monkeypatch, 3, device="cuda:1")
    with pytest.raises(ValueError, match="Kit ordinals"):
        make_plan(.5, monkeypatch, 3, kit_args=m.kit_args_for(1))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(ValueError, match="unmasked"):
        make_plan(.5, monkeypatch, 3)


def test_sdk_float32_step_accumulation_is_measured_not_rounded():
    control, physics, runtime = telemetry(1.)
    step = float(np.float32(.005)); origin = float(control["time_s"][0])
    control["time_s"] = origin+np.arange(349)*4*step
    physics["time_s"] = origin+np.arange(1396)*step
    before = physics["time_s"].copy()
    proof = m.verify_telemetry(control, physics, runtime, 1.)
    clocks = proof["clock_diagnostics"]
    observed = clocks["measured"]["physics_time"]["maximum_absolute_cumulative_drift_s"]
    assert 1e-8 < observed < 2e-6
    assert clocks["all_components_passed"] is True
    assert np.array_equal(physics["time_s"], before)
    assert clocks["previous_unsaved_v2_drift_measured"] is False


@pytest.mark.parametrize("bad", ["missing_counter", "native_index", "5ms_jump", "large_drift"])
def test_quantization_bound_does_not_accept_missing_steps_or_large_drift(bad):
    control, physics, runtime = telemetry(1.)
    if bad == "missing_counter": physics.pop("sim_step_counter")
    if bad == "native_index": physics["native_step_index"][30] += 1
    if bad == "5ms_jump": physics["time_s"][30:] += .005
    if bad == "large_drift": physics["time_s"] += np.linspace(0., 3e-6, 1396)
    with pytest.raises(ValueError, match="clock component"):
        m.verify_telemetry(control, physics, runtime, 1.)


def test_failed_clock_validation_preserves_original_unverified_sidecars(tmp_path, monkeypatch):
    plan = make_plan(1., monkeypatch); plan["paths"] = {"output_dir": tmp_path}
    control, physics, runtime = telemetry(1.); physics["time_s"][30:] += .005
    class Handler:
        def control_arrays(self): return control
        def physics_arrays(self): return physics
        def close(self):
            assert (tmp_path/m.CONTROL_FILE).is_file() and (tmp_path/m.PHYSICS_FILE).is_file()
            assert json.loads((tmp_path/m.RUNTIME_FILE).read_text())["verification_status"] == "not_verified_yet"
        def audit(self, *, check_device_health): return runtime
    with pytest.raises(ValueError, match="physics_time_relative_grid"):
        m.finalize_filter(None, Handler(), plan)
    final = json.loads((tmp_path/m.RUNTIME_FILE).read_text())
    assert final["verification_status"] == "failed" and final["telemetry_verification"] is None
    assert final["clock_diagnostics"]["measured"]["physics_time"]["maximum_absolute_cumulative_drift_s"] > .004
    assert "physics_time_relative_grid" in final["verification_error"]["message"]
    assert "final_runtime_evidence" not in plan["audit"]
    assert not (tmp_path/"report.json").exists()
    with np.load(tmp_path/m.PHYSICS_FILE, allow_pickle=False) as archive:
        assert np.array_equal(archive["time_s"], physics["time_s"])
