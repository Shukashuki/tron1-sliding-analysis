from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ppo_backend_400hz_contract as mod
import train_ppo_backend_400hz as trainer


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_cohort(tmp_path, *, smoke=False, backend="physx", actor_bytes=b"normalized-export"):
    directory = tmp_path / (backend + ("_smoke" if smoke else "_formal"))
    directory.mkdir()
    protocol = json.loads(mod.DEFAULT_PROTOCOL.read_text())
    contract, profile = trainer.derive_inputs(json.loads((ROOT / "config/nominal_actuator_contract.json").read_text()),
        json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text()), protocol)
    (directory / "model_final.pt").write_bytes(b"checkpoint-not-unpickled")
    (directory / "policy.pt").write_bytes(actor_bytes)
    checkpoint, actor = mod.sha256(directory / "model_final.pt"), mod.sha256(directory / "policy.pt")
    mode, budget = ("smoke", 1536) if smoke else ("formal", 12288000)
    flags = {"smoke_only": smoke, "trained_policy_claimed": not smoke}
    contract.update(contract_status="smoke_completed_normalized_actor_not_formal_policy" if smoke else "trained_completed_normalized_actor",
        **flags, run_kind=mode, policy_algorithm="PPO", policy_training_backend=backend,
        checkpoint_sha256=checkpoint, exported_policy_sha256=actor, checkpoint_file="model_final.pt", actor_file="policy.pt",
        actor_includes_observation_normalizer=True, actor_output_is_unclipped=True, actor_observation_groups=["policy"],
        export_validation_max_abs_error=0., reference_file_sha256=mod.INPUT_PINS["motion"],
        training_cohort_protocol_sha256=mod.sha256(mod.DEFAULT_PROTOCOL), training_environment_transitions=budget,
        runtime_clock_and_interface_verified=True)
    cohort = {"id": protocol["cohort_id"], "protocol_sha256": mod.sha256(mod.DEFAULT_PROTOCOL),
        "wrapper_sha256": mod.TRAINER_SHA256, "from_scratch": True, "historical_checkpoint_used": False,
        "network_seed_after_environment_setup_and_smoke": True, "physical_substeps_per_environment_transition": 8,
        "runtime": {"status": "verified", "fields": {"physics_dt_s": .0025, "decimation": 8,
            "policy_dt_s": .02, "actor_dim": 51, "critic_dim": 120, "action_dim": 8,
            "action_rate_weight": -.5, "grounded_reward_weight": .4, "reward_term_count": 15,
            "leg_kp": 500., "leg_kd": 10., "absolute_height_termination": False, "absolute_xy_termination": False}},
        "base_trainer_sha256": protocol["base_trainer_sha256"],
        "source_files_sha256": {"scripts/train_ppo_backend_400hz.py": mod.TRAINER_SHA256},
        "initial_policy_state_sha256": "f" * 64, **flags, "run_kind": mode}
    put(directory / "cohort_startup_audit.json", cohort)
    manifest = {"contract": contract, "task_profile": profile, "backend": backend, "seed": 42,
        "transitions_requested": budget, "versions": {"rsl-rl-lib": "3.0.1"},
        "cohort": {**cohort, "normalized_actor_export_verified": True}, **flags,
        "num_envs": 32 if smoke else 512, "iterations": 2 if smoke else 1000,
        "initial_policy_state_sha256": "f" * 64,
        "inputs": {name: {"sha256": mod.INPUT_PINS[name]} for name in ("motion", "model", "asset")}}
    put(directory / "manifest.json", manifest)
    for name in ("final_policy_contract.json", "policy_contract.json"):
        put(directory / name, contract)
    put(directory / "run_status.json", {"status": "training_completed", "run_kind": mode,
        "initial_policy_state_sha256": "f" * 64,
        "training_environment_steps": budget, "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor, **flags})
    from train_ppo_backend import ppo_config
    put(directory / "runner_config.json", ppo_config())
    completion = {"status": "completed_verified", "cohort_id": protocol["cohort_id"], "backend": backend,
        "protocol_sha256": mod.sha256(mod.DEFAULT_PROTOCOL), "seed": 42, "training_environment_steps": budget,
        "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "action_rate_weight": -.5,
        "grounded_reward_weight": .4, "source_files_unchanged": True, "additional_reward_terms": False,
        "historical_checkpoint_used": False, "run_kind": mode, **flags, "checkpoint_file": "model_final.pt",
        "actor_file": "policy.pt", "final_policy_contract_file": "final_policy_contract.json"}
    completion["initial_policy_state_sha256"] = "f" * 64
    for key, name in {"checkpoint_sha256": "model_final.pt", "exported_policy_sha256": "policy.pt",
        "manifest_sha256": "manifest.json", "policy_contract_sha256": "policy_contract.json",
        "final_policy_contract_sha256": "final_policy_contract.json", "startup_audit_sha256": "cohort_startup_audit.json"}.items():
        completion[key] = mod.sha256(directory / name)
    put(directory / "cohort_completion_audit.json", completion)
    return directory


def refresh(directory):
    completion = json.loads((directory / "cohort_completion_audit.json").read_text())
    for key, name in {"checkpoint_sha256": "model_final.pt", "exported_policy_sha256": "policy.pt",
        "manifest_sha256": "manifest.json", "policy_contract_sha256": "policy_contract.json",
        "final_policy_contract_sha256": "final_policy_contract.json", "startup_audit_sha256": "cohort_startup_audit.json"}.items():
        completion[key] = mod.sha256(directory / name)
    put(directory / "cohort_completion_audit.json", completion)


@pytest.mark.parametrize("backend", ["physx", "mjlab"])
def test_completed_new_cohort_verified_without_loading_pickle(tmp_path, backend):
    path = make_cohort(tmp_path, backend=backend)
    result = mod.verify_cohort(path)
    assert result["training_backend"] == backend
    assert result["smoke_only"] is False
    assert mod.provenance(result)["old_200hz_actor_used"] is False


def test_smoke_explicitly_labeled_and_nominal_only(tmp_path):
    path = make_cohort(tmp_path, smoke=True)
    with pytest.raises(ValueError, match="Smoke actor"):
        mod.verify_cohort(path)
    result = mod.verify_cohort(path, allow_smoke=True)
    assert mod.evaluation_contract(result, .7, "PhysX")["trained_policy_claimed"] is False
    with pytest.raises(ValueError, match="restricted"):
        mod.evaluation_contract(result, .15, "PhysX")


@pytest.mark.parametrize("name", ["policy.pt", "model_final.pt", "manifest.json", "cohort_startup_audit.json"])
def test_completed_artifact_mutation_rejected(tmp_path, name):
    path = make_cohort(tmp_path)
    with (path / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA mismatch"):
        mod.verify_cohort(path)


@pytest.mark.parametrize("key,value", [("physics_dt", .005), ("decimation", 4), ("policy_algorithm", "SAC"),
    ("leg_kp", 250.), ("actor_includes_observation_normalizer", False), ("actor_output_is_unclipped", False)])
def test_rehashed_incompatible_contract_still_rejected(tmp_path, key, value):
    path = make_cohort(tmp_path)
    contract = json.loads((path / "final_policy_contract.json").read_text())
    contract[key] = value
    for name in ("final_policy_contract.json", "policy_contract.json"):
        put(path / name, contract)
    refresh(path)
    with pytest.raises(ValueError, match="policy contract differs"):
        mod.verify_cohort(path)


def test_evaluation_retains_training_profile_and_native_mu_semantics(tmp_path):
    result = mod.verify_cohort(make_cohort(tmp_path))
    before = copy.deepcopy(result["contract"])
    for backend in ("PhysX", "native CPU MuJoCo"):
        contract = mod.evaluation_contract(result, .03, backend)
        assert contract["trained_task_profile"]["contact_friction"] == 1.
        assert contract["task_profile"]["contact_friction"] == .03
        assert contract["full_horizon_diagnostic"] is True
        assert contract["evaluation_contact_friction_override"]["mu_squared"] is False
    assert result["contract"] == before
    with pytest.raises(ValueError, match="17-point"):
        mod.evaluation_contract(result, .11, "PhysX")


def test_inputs_fresh_sibling_and_no_input_mutation(tmp_path, monkeypatch):
    result = mod.verify_cohort(make_cohort(tmp_path))
    motion = tmp_path / "reference.npz"
    motion.write_bytes(b"locked")
    monkeypatch.setitem(mod.INPUT_PINS, "motion", mod.sha256(motion))
    old = {name: mod.sha256(path) for name, path in result["paths"].items()}
    prepared = mod.prepare_inputs(result, motion, tmp_path / "validation", .7, "PhysX")
    assert prepared["input_dir"] == tmp_path / "validation_inputs"
    assert not prepared["output"].exists()
    assert old == {name: mod.sha256(path) for name, path in result["paths"].items()}
    with pytest.raises(ValueError, match="Preserve"):
        mod.prepare_inputs(result, motion, tmp_path / "validation", .7, "PhysX")


def test_ppo_runner_normalizes_exactly_once_without_tanh_or_checkpoint_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "export.pt"
    actor = torch.nn.Sequential(torch.nn.Linear(51, 51), torch.nn.Linear(51, 8)).eval()
    with torch.no_grad():
        actor[0].weight.copy_(torch.eye(51) * 2.)
        actor[0].bias.fill_(1.)
        actor[1].weight.fill_(1.)
        actor[1].bias.fill_(.5)
    torch.jit.script(actor).save(str(path))
    cohort = mod.verify_cohort(make_cohort(tmp_path, actor_bytes=path.read_bytes()))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Checkpoint unpickling forbidden"))
    runner = mod.StrictPPOExportRunner(cohort, "cpu")
    runner.load(cohort["checkpoint"], map_location="cpu")
    obs = {"policy": torch.ones(2, 51)}
    result = runner.get_inference_policy()(obs)
    assert torch.equal(result, actor(obs["policy"]))
    assert torch.all(result > 1.)
    assert isinstance(runner.alg.policy.actor_obs_normalizer, torch.nn.Identity)
    with pytest.raises(ValueError, match="once"):
        runner.load(cohort["checkpoint"])
    with pytest.raises(ValueError, match="51D"):
        runner.get_inference_policy()({"policy": torch.zeros(1, 50)})
