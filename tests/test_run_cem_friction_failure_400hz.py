from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_cem_friction_failure_400hz as mod


def test_fresh_output_guard_precedes_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "inputs", lambda *a: pytest.fail("No simulator/input work"))
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path, resume=False))


def test_resume_requires_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "inputs", lambda *a: pytest.fail("No simulator/input work"))
    with pytest.raises(ValueError, match="resume requires existing"):
        mod.run(SimpleNamespace(output_dir=tmp_path/"missing", resume=True))


@pytest.mark.parametrize("field,value", [("mu_grid", [.1, .7]), ("representative_first", [.3]),
    ("beta_conditions", ["beta_on"]), ("prediction_friction_known", False),
    ("new_training", True), ("concurrent_local_CPU_episodes", 3)])
def test_invalid_scope_rejected_before_completed_input_loading(tmp_path, monkeypatch, field, value):
    settings = mod.read_json(mod.DEFAULT_SETTINGS)
    settings[field] = value
    path = tmp_path/"settings.json"; mod.write_json(path, settings)
    monkeypatch.setattr(mod.nominal, "inputs", lambda: pytest.fail("Reject before loading PPO"))
    with pytest.raises(ValueError, match="seventeen-point"):
        mod.inputs(path)


@pytest.mark.parametrize("values", [[], [.15, .15], [.11], [float("nan")], [float("inf")]])
def test_subset_must_be_unique_declared_values(values):
    with pytest.raises(ValueError, match="unique frozen-grid"):
        mod.selected_grid(values)


def test_stable_requested_scope_and_representative_first_order():
    settings = mod.read_json(mod.DEFAULT_SETTINGS)
    assert mod.selected_grid(None) == list(mod.GRID)
    assert mod.selected_grid([1.2, .15, .3]) == [.15, .3, 1.2]
    ordered = mod.ordered_grid(list(mod.GRID), settings)
    assert ordered[:3] == [.15, .3, 1.2]
    assert len(ordered) == len(set(ordered)) == 17
    assert set(ordered) == set(mod.GRID)
    assert mod.job_id("beta_on", .15) == "cem_beta_on_mu0p15"


def test_same_frozen_seed_search_budget_and_not_automatic_training():
    settings = mod.read_json(mod.DEFAULT_SETTINGS)
    baseline = mod.read_json(mod.nominal.DEFAULT_SETTINGS)
    assert settings["nmpc"] == baseline["nmpc"]
    assert settings["nmpc"]["seed"] == 42
    assert settings["discount_factor"] == .99
    assert settings["new_training"] is False
    source = Path(mod.__file__).read_text()
    assert "ThreadPoolExecutor(max_workers=2" in source
    assert '"automatic_failed_job_retry": False' in source
    assert '"height_contact_tilt_failure_gate": False' in source
    assert 'profile["contact_friction"] = float(mu)' in source
    assert '"legacy_contract_training_friction_retained": True' in source
    assert 'mu_scenarios=(mu,)' in source
    assert 'scalar.models[mu]' in source
    assert 'model.geom_friction[:, 0] = mu' in source
    assert "torch.jit.load(" not in source and "torch.load(" not in source


def test_literal_nominal_hash_bindings_are_original_evidence():
    if not mod.DEFAULT_NOMINAL.is_dir():
        pytest.skip("Optional nominal completed bundle")
    result = mod.verify_nominal(mod.DEFAULT_NOMINAL)
    assert set(result) == {"beta_on", "beta_off"}
    assert all(row["status"] == "reused_completed_nominal" and row["new_episode"] is False
               for row in result.values())
    assert all(len(row["file_sha256"]) == 3 for row in result.values())


@pytest.fixture(scope="module")
def actual():
    if not (mod.paired.DEFAULT_OFF/"manifest.json").is_file():
        pytest.skip("Optional local strict completed PPO bundles")
    return mod.inputs()


@pytest.mark.parametrize("mu", [.01, .15, .3, 1.2, 2.])
def test_material_mapping_changes_only_sliding_coefficient(actual, mu):
    settings, contracts, *_ = actual
    contract = copy.deepcopy(contracts["beta_on"])
    provider = mod.build_provider(mod.paired.DEFAULT_MODEL, mod.paired.DEFAULT_REFERENCE, contract, settings, (mu,))
    expected = mod.read_json(mod.paired.DEFAULT_ON/"cases/physx/mujoco/report.json")["contact_material_audit"]
    model = provider.model
    model.geom_friction[:, 0] = mu
    audit = mod.configure_native_model(model, expected, mu)
    assert audit["pair_coefficient"] == mu
    assert all(row[0] == mu for row in audit["geom_friction"])
    assert audit["native_plant_readback"] == expected["native_plant_readback"]
    assert np.array_equal(np.asarray(audit["geom_friction"])[:, 1:], np.asarray(expected["geom_friction"])[:, 1:])
    model.body_mass[1] += .1
    with pytest.raises(ValueError, match="nuisance mismatch"):
        mod.configure_native_model(model, expected, mu)


def test_validation_never_promotes_running_or_failed_checkpoint(tmp_path):
    mod.write_json(tmp_path/"report.json", {"kind": "400hz_friction_sweep_CEM_with_completed_PPO_stage_reward",
        "status": "running"})
    with pytest.raises(ValueError, match="genuine completed"):
        mod.validate_completed(tmp_path, "beta_on", .15, {}, {})


def test_resume_preserves_failed_prefix_and_never_retries_it(tmp_path, monkeypatch):
    settings = mod.read_json(mod.DEFAULT_SETTINGS)
    monkeypatch.setattr(mod, "inputs", lambda *_: (settings, {key: {} for key in settings["beta_conditions"]}, {}, {}, {}))
    monkeypatch.setattr(mod, "initial_states", lambda: {key: {} for key in settings["beta_conditions"]})
    monkeypatch.setattr(mod, "verify_nominal", lambda *_: {})
    attempts = []
    def fake_episode(directory, label, *args):
        attempts.append(label)
        directory.mkdir()
        if label == "beta_on":
            mod.write_json(directory/"report.json", {"status": "failed", "error_type": "FloatingPointError", "error": "preserved"})
            np.savez_compressed(directory/"failed_prefix.npz", qpos=np.zeros((2, 15)))
            raise FloatingPointError("preserved")
        mod.write_json(directory/"report.json", {"status": "smoke_completed"})
    monkeypatch.setattr(mod, "run_episode", fake_episode)
    monkeypatch.setattr(mod, "validate_completed", lambda *_: {"status": "smoke_completed"})
    library = tmp_path/"fake.so"; library.write_bytes(b"not-executed")
    args = SimpleNamespace(output_dir=tmp_path/"new", settings=mod.DEFAULT_SETTINGS,
        native_library=library, mu=[.15], max_policy_steps=1, resume=False)
    first = mod.run(args)
    assert sorted(attempts) == ["beta_off", "beta_on"]
    assert first["status"] == "completed_with_failures"
    prefix = args.output_dir/mod.job_id("beta_on", .15)/"failed_prefix.npz"
    digest = mod.sha256(prefix)
    args.resume = True
    second = mod.run(args)
    assert sorted(attempts) == ["beta_off", "beta_on"]
    assert second["jobs"][0]["status"] == "failed"
    assert second["completed_scope_includes_full17grid"] is False
    assert mod.sha256(prefix) == digest


def test_resume_does_not_promote_interrupted_report(tmp_path, monkeypatch):
    settings = mod.read_json(mod.DEFAULT_SETTINGS)
    monkeypatch.setattr(mod, "inputs", lambda *_: (settings, {key: {} for key in settings["beta_conditions"]}, {}, {}, {}))
    monkeypatch.setattr(mod, "initial_states", lambda: {key: {} for key in settings["beta_conditions"]})
    monkeypatch.setattr(mod, "verify_nominal", lambda *_: {})
    attempts = []
    def interrupted(directory, label, *args):
        attempts.append(label)
        directory.mkdir()
        mod.write_json(directory/"report.json", {"status": "running"})
        raise RuntimeError("interrupted")
    monkeypatch.setattr(mod, "run_episode", interrupted)
    library = tmp_path/"fake.so"; library.write_bytes(b"not-executed")
    args = SimpleNamespace(output_dir=tmp_path/"new", settings=mod.DEFAULT_SETTINGS,
        native_library=library, mu=[.3], max_policy_steps=1, resume=False)
    mod.run(args)
    args.resume = True
    second = mod.run(args)
    assert len(attempts) == 2
    assert all(job["status"] == "interrupted_preserved" for job in second["jobs"])
    assert second["status"] == "completed_with_failures"


def test_optional_two_arm_actual_mu_smoke_schema(actual):
    root = ROOT/"outputs/cem_friction_failure_400hz_smoke_20261022_v2"
    if not (root/"experiment.json").is_file():
        pytest.skip("Optional explicitly launched local one-step smoke")
    experiment = mod.read_json(root/"experiment.json")
    assert experiment["status"] == "smoke_completed"
    assert experiment["completed_scope_includes_full17grid"] is False
    assert experiment["max_policy_steps"] == 1
    assert len(experiment["jobs"]) == 2
    initials = mod.initial_states()
    for job in experiment["jobs"]:
        checked = mod.validate_completed(job["directory"], job["condition"], job["mu"],
            experiment["settings"], initials[job["condition"]], 1, experiment["native_library_sha256"])
        assert checked["status"] == "smoke_completed"
        assert checked["native_samples"] == 9
        assert checked["completed_policy_steps"] == 1
