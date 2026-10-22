from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_cem_matched_reward_400hz as mod


def test_fresh_output_guard_precedes_inputs_and_work(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "inputs", lambda *a: pytest.fail("Inputs must not be loaded"))
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path))


@pytest.mark.parametrize("field,value", [("physics_dt_s", .005), ("decimation", 4),
    ("action_rate_weight", -.1), ("grounded_reward_weight", 1.), ("discount_factor", 1.),
    ("validation_backend", "PhysX"), ("reference_id", "sa29")])
def test_stale_or_altered_recipe_rejected_before_loading_PPO(field, value, tmp_path, monkeypatch):
    settings = mod.read_json(mod.DEFAULT_SETTINGS); settings[field] = value
    target = tmp_path/"changed.json"; mod.write_json(target, settings)
    monkeypatch.setattr(mod.paired, "load_comparison", lambda *a: pytest.fail("Reject before loaded PPO"))
    with pytest.raises(ValueError, match="frozen400Hz"):
        mod.inputs(target)


def test_fixed_optimizer_budget_gamma_and_two_local_CPU_episodes():
    settings = mod.read_json(mod.DEFAULT_SETTINGS)
    assert settings["discount_factor"] == .99 and settings["concurrent_local_CPU_episodes"] == 2
    assert settings["nmpc"]["horizon_steps"] == 30 and settings["nmpc"]["control_blocks"] == 10
    assert settings["nmpc"]["candidates"] == 64 and settings["nmpc"]["iterations"] == 3
    assert settings["nmpc"]["elite_count"] == 8


def test_source_never_loads_policy_actions_or_claims_PPO_episode_return():
    source = Path(mod.__file__).read_text()
    assert 'raw[name].copy() for name in ("initial_qpos", "initial_qvel")' in source
    assert '"PPO_actions_or_future_states_used_for_initial_plan": False' in source
    assert '"PPO_total_reward_recomputed": False' in source
    assert '"termination_or_reset_after_initialization": False' in source
    assert "torch.jit.load(" not in source and "torch.load(" not in source


@pytest.fixture(scope="module")
def actual():
    if not (mod.paired.DEFAULT_OFF/"manifest.json").is_file():
        pytest.skip("Optional local strict PPO retrieval bundles")
    return mod.inputs()


def test_actual_completed_PPO_task_contract_only_beta_differs(actual):
    settings, contracts, traces, reference, evidence = actual
    on, off = [copy.deepcopy(contracts[key]["task_profile"]) for key in ("beta_on", "beta_off")]
    assert on["reward_weights"]["sideslip"] == 2. and off["reward_weights"]["sideslip"] == 0.
    off["reward_weights"]["sideslip"] = 2.
    assert on == off
    assert on["action_rate_weight"] == -.5 and on["reward_weights"]["grounded"] == .4
    assert len(traces["beta_on"]["time_s"]) == len(traces["beta_off"]["time_s"]) == 2793
    assert reference["time_s"][-1] == pytest.approx(6.96)
    assert settings["physics_dt_s"] == .0025 and len(evidence["rows"]) == 2


def test_actual_model_nuisance_and_priority_arrays_match_PPO_then_poison_detected(actual):
    settings, contracts, *_ = actual
    provider = mod.build_provider(mod.paired.DEFAULT_MODEL, mod.paired.DEFAULT_REFERENCE, contracts["beta_on"], settings, (.7,))
    expected = mod.read_json(mod.paired.DEFAULT_ON/"cases/physx/mujoco/report.json")["contact_material_audit"]
    model = provider.model
    observed = mod.configure_native_model(model, expected)
    assert observed["geom_priority"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    model.body_mass[1] += .1
    with pytest.raises(ValueError, match="nuisance mismatch: body_mass"):
        mod.configure_native_model(model, expected)


def test_optional_actual_two_step_smoke_schema():
    root = ROOT/"outputs/ppo_cem_matched_reward_400hz_smoke_20261022_v3"
    if not (root/"experiment.json").is_file() or mod.read_json(root/"experiment.json")["status"] != "smoke_completed":
        pytest.skip("Optional completed two-arm local smoke")
    for label in ("beta_on", "beta_off"):
        report = mod.read_json(root/label/"report.json")
        assert report["status"] == "smoke_completed" and report["solver_failures"] == 0
        assert report["native_samples"] == 17 and report["duration_s"] == pytest.approx(.04)
        assert report["native_plant_arrays_match_PPO_validation_exactly"] is True
        assert report["PPO_total_reward_recomputed"] is False
        with np.load(root/label/"rollout.npz", allow_pickle=False) as raw:
            assert raw["body_net_forces_w"].shape == (17, 9, 3)
            assert raw["contact_force_history_w"].shape == (17, 3, 9, 3)
            assert raw["actions"].shape == (2, 8)
            assert np.all(np.isfinite(raw["qpos"]))
