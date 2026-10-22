"""CPU-only source, exact-ablation and synthetic completed-artifact tests."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_ppo_backend_400hz_beta_off as trainer
import ppo_backend_400hz_beta_contract as contract
import train_ppo_backend_400hz as baseline
from test_ppo_backend_400hz_contract import make_cohort as make_baseline, put, refresh


def protocol():
    return json.loads(trainer.DEFAULT_PROTOCOL.read_text())


def runtime(backend="physx"):
    count = 15
    return {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8, "action_rate_weight": -.5,
        "grounded_reward_weight": .4, "reward_term_count": count, "leg_kp": 500., "leg_kd": 10.,
        "absolute_height_termination": False, "absolute_xy_termination": False,
        "training_backend": backend, "sideslip_reward_weight": 0., "declared_reward_term_count": 15,
        "listed_reward_term_count": count, "nonzero_reward_term_count": 14,
        "sideslip_term_listed": True, "effective_sideslip_reward_enabled": False,
        "declared_reward_weights": copy.deepcopy(trainer.REWARD_WEIGHTS)}


def make_cohort(tmp_path, *, backend="physx", smoke=False):
    path = make_baseline(tmp_path, backend=backend, smoke=smoke)
    p, identity = protocol(), trainer.ablation_identity()
    sha = contract.sha256(trainer.DEFAULT_PROTOCOL)
    files = {name: json.loads((path / name).read_text()) for name in ("final_policy_contract.json", "manifest.json",
        "cohort_startup_audit.json", "run_status.json", "cohort_completion_audit.json")}
    policy = files["final_policy_contract.json"]
    declared, task = trainer.derive_inputs(json.loads((ROOT / "config/nominal_actuator_contract.json").read_text()),
        json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text()), p)
    policy.update(cohort_id=p["cohort_id"], beta_ablation=identity, sideslip_reward_weight=0.,
        task_profile=copy.deepcopy(task), trained_task_profile=copy.deepcopy(task),
        training_cohort_protocol_sha256=sha)
    startup = files["cohort_startup_audit.json"]
    startup.update(id=p["cohort_id"], protocol_sha256=sha, wrapper_sha256=contract.TRAINER_SHA256,
        source_files_sha256={"scripts/train_ppo_backend_400hz_beta_off.py": contract.TRAINER_SHA256},
        beta_ablation=identity, initial_policy_matches_beta_on=True,
        initial_policy_state_sha256=trainer.INITIAL_POLICY_SHA256,
        runtime={"status": "verified", "fields": runtime(backend)})
    manifest = files["manifest.json"]
    manifest.update(contract=policy, task_profile=copy.deepcopy(task), beta_ablation=identity,
        initial_policy_state_sha256=trainer.INITIAL_POLICY_SHA256,
        cohort={**copy.deepcopy(startup), "normalized_actor_export_verified": True})
    for name in ("run_status.json", "cohort_completion_audit.json"):
        files[name].update(cohort_id=p["cohort_id"], protocol_sha256=sha,
            beta_ablation=identity, sideslip_reward_weight=0., initial_policy_matches_beta_on=True,
            initial_policy_state_sha256=trainer.INITIAL_POLICY_SHA256)
    files["cohort_completion_audit.json"].update(declared_reward_term_count=15,
        listed_reward_term_count=15, nonzero_reward_term_count=14, sideslip_term_listed=True,
        effective_sideslip_reward_enabled=False)
    for name, value in files.items():
        put(path / name, value)
    put(path / "policy_contract.json", policy)
    refresh(path)
    return path


def test_protocol_exact_baseline_copy_except_identity_and_declared_ablation():
    off = trainer.validate_protocol(protocol())
    on = json.loads(trainer.BASELINE_PROTOCOL.read_text())
    off = copy.deepcopy(off); off.pop("beta_ablation"); off["cohort_id"] = on["cohort_id"]
    assert off == on
    assert contract.verify_protocol()["cohort_id"] == trainer.COHORT_ID


@pytest.mark.parametrize("change", [lambda p:p["algorithm"].update(seed=43),
    lambda p:p.update(action_rate_weight=-.1), lambda p:p["training_randomization"].update(restitution_randomization=True),
    lambda p:p["beta_ablation"].update(ablated_weight=1e-9), lambda p:p["validation"].update(full_horizon_s=1.)])
def test_protocol_rejects_any_other_change(change):
    p = protocol(); change(p)
    with pytest.raises(ValueError): trainer.validate_protocol(p)


def test_derived_profile_is_exactly_one_reward_change_and_preserves_description():
    nominal = json.loads((ROOT / "config/nominal_actuator_contract.json").read_text())
    source = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    before = copy.deepcopy(source)
    on_protocol = json.loads(trainer.BASELINE_PROTOCOL.read_text())
    on_contract, on = baseline.derive_inputs(nominal, source, on_protocol)
    off_contract, off = trainer.derive_inputs(nominal, source, protocol())
    restored = copy.deepcopy(off); restored["reward_weights"]["sideslip"] = 2.
    assert restored == on and source == before
    assert off["action_rate_weight"] == -.5 and off["reward_weights"]["grounded"] == .4
    assert off_contract["sideslip_reward_weight"] == 0. and off_contract["task_profile"] == off_contract["trained_task_profile"]
    trainer.validate_task_profile(off, protocol())
    changed = copy.deepcopy(off); changed["reward_weights"]["speed"] = 0.
    with pytest.raises(ValueError, match="single sideslip"):trainer.validate_task_profile(changed, protocol())


def test_prepared_inputs_deterministic_across_smoke_formal_and_parity(tmp_path):
    a = trainer.prepare_declared_inputs(trainer.DEFAULT_PROTOCOL, tmp_path / "a")
    b = trainer.prepare_declared_inputs(trainer.DEFAULT_PROTOCOL, tmp_path / "b")
    assert a["derived_input_sha256"] == b["derived_input_sha256"]
    assert json.loads(a["task_profile"].read_text())["reward_weights"]["sideslip"] == 0.
    with pytest.raises(ValueError, match="fresh"):
        trainer.prepare_declared_inputs(trainer.DEFAULT_PROTOCOL, tmp_path / "a")


def test_source_preserves_original_fifteen_listed_term_guard_and_seed_order():
    original = (ROOT / "scripts/train_ppo_backend.py").read_text()
    old = baseline.materialize_source(original)
    new = trainer.materialize_source(original)
    assert new == old
    assert 'len(metadata["reward_terms"]) != 15' in new
    assert new.index("torch.manual_seed(args.seed)") < new.index("runner = OnPolicyRunner") < new.index("_cohort_before_learn")
    assert "runner.load(" not in new
    with pytest.raises(ValueError, match="content differs"):
        trainer.materialize_source(original + "\n")


@pytest.mark.parametrize("backend", ["physx", "mjlab"])
def test_exact_zero_and_listed_nonzero_declared_reward_counts(backend):
    trainer.validate_runtime_fields(runtime(backend))
    v = runtime(backend); v["declared_reward_weights"]["drift_sideslip"] = 1e-10
    with pytest.raises(ValueError, match="runtime weight"): trainer.validate_runtime_fields(v)
    v = runtime(backend); v["declared_reward_weights"]["drift_speed"] = .5
    with pytest.raises(ValueError, match="runtime weight"): trainer.validate_runtime_fields(v)


def test_installed_sdk_retains_zero_weight_in_list_but_only_fourteen_weights_are_nonzero():
    value = runtime("physx")
    assert len(value["declared_reward_weights"]) == 15
    assert sum(weight != 0. for weight in value["declared_reward_weights"].values()) == 14
    trainer.validate_runtime_fields(value)
    for key, wrong in (("reward_term_count", 14), ("listed_reward_term_count", 14),
                       ("nonzero_reward_term_count", 15), ("sideslip_term_listed", False),
                       ("effective_sideslip_reward_enabled", True)):
        changed = copy.deepcopy(value); changed[key] = wrong
        with pytest.raises(ValueError): trainer.validate_runtime_fields(changed)


def test_initial_network_guard_fails_before_runtime_or_learning(tmp_path, monkeypatch):
    monkeypatch.setitem(trainer._ORIGINAL, "before_learn", lambda *a: pytest.fail("No baseline hook after seed mismatch"))
    with pytest.raises(ValueError, match="Initial policy"):
        trainer.before_learn(None, None, {"initial_policy_state_sha256": "0" * 64}, None, {})


@pytest.mark.parametrize("backend", ["physx", "mjlab"])
def test_actual_runtime_weights_and_before_learn_startup_guard(tmp_path, monkeypatch, backend):
    fields = runtime(backend)
    for key in ("training_backend", "sideslip_reward_weight", "declared_reward_weights", "listed_reward_term_count",
                "nonzero_reward_term_count", "declared_reward_term_count", "sideslip_term_listed", "effective_sideslip_reward_enabled"):
        fields.pop(key)
    monkeypatch.setitem(trainer._ORIGINAL, "runtime_audit", lambda *a: {"status": "verified", "fields": copy.deepcopy(fields)})
    weights = copy.deepcopy(trainer.REWARD_WEIGHTS)
    # Actual installed PhysX SDK regression: active_terms retains the zero key.
    manager = NS(active_terms=list(weights), get_term_cfg=lambda name: NS(weight=weights[name]))
    env = NS(reward_manager=manager, cfg=NS(rewards=NS(drift_sideslip=NS(weight=0.))), objective=NS(weights=weights))
    declared = trainer.prepare_declared_inputs(trainer.DEFAULT_PROTOCOL, tmp_path / "declared")
    output = tmp_path / "output"; output.mkdir()
    args = NS(backend=backend, output_dir=output, task_profile=declared["task_profile"])
    plan = {"protocol": protocol(), "run_kind": "smoke", "protocol_sha256": trainer.sha256(trainer.DEFAULT_PROTOCOL),
        "wrapper_sha256": trainer.sha256(trainer.__file__), "generated_sha256": "0" * 64,
        "source_hashes": {}, "original_input_sha256": declared["original_input_sha256"],
        "derived_input_sha256": declared["derived_input_sha256"]}
    manifest = {"initial_policy_state_sha256": trainer.INITIAL_POLICY_SHA256, "environment": {}}
    trainer.before_learn(env, args, manifest, None, plan)
    startup = json.loads((output / "cohort_startup_audit.json").read_text())
    assert startup["initial_policy_matches_beta_on"] is True
    assert startup["beta_ablation"] == trainer.ablation_identity()
    assert startup["runtime"]["fields"]["sideslip_reward_weight"] == 0.
    if backend == "physx":
        env.cfg.rewards.drift_sideslip.weight = 1e-12
    else:
        weights["drift_sideslip"] = 1e-12
    with pytest.raises(ValueError): trainer.runtime_audit(env, args, {}, plan)


@pytest.mark.parametrize("smoke,budget", [(True, 1536), (False, 12288000)])
def test_actual_finalize_complete_preserves_true_beta_off_export(tmp_path, monkeypatch, smoke, budget):
    class FakeTensor(np.ndarray):
        def cpu(self):return self
    fake = NS(inference_mode=nullcontext, allclose=np.allclose, abs=np.abs, max=np.max)
    monkeypatch.setitem(sys.modules, "torch", fake)
    prepared = trainer.prepare_declared_inputs(trainer.DEFAULT_PROTOCOL, tmp_path / "declared")
    output = tmp_path / "output"; output.mkdir()
    (output / "model_final.pt").write_bytes(b"synthetic-off-checkpoint")
    (output / "policy.pt").write_bytes(b"synthetic-off-normalized-actor")
    (output / "cohort_startup_audit.json").write_text("{}")
    motion = tmp_path / "motion.npz"; motion.write_bytes(b"synthetic-reference")
    args = NS(backend="physx", output_dir=output, contract=prepared["contract"], task_profile=prepared["task_profile"], motion=motion)
    plan = {"protocol": protocol(), "protocol_path": trainer.DEFAULT_PROTOCOL,
        "protocol_sha256": trainer.sha256(trainer.DEFAULT_PROTOCOL), "run_kind": "smoke" if smoke else "formal",
        "expected_transitions": budget, "source_hashes": {name: trainer.sha256(ROOT / name) for name in trainer._CORE.SOURCE_FILES}}
    manifest = {"cohort": {}, "initial_policy_state_sha256": trainer.INITIAL_POLICY_SHA256}
    expected = np.ones((2, 8)).view(FakeTensor)
    probe = np.zeros((2, 51)).view(FakeTensor)
    trainer.finalize_contract(args, manifest, NS(tot_timesteps=budget), lambda _: expected, probe, expected, plan)
    c = json.loads((output / "final_policy_contract.json").read_text())
    assert c["sideslip_reward_weight"] == c["task_profile"]["reward_weights"]["sideslip"] == 0.
    assert c["smoke_only"] is smoke and c["trained_policy_claimed"] is (not smoke)
    report = {"status": "training_completed", "training_environment_steps": budget,
        "checkpoint_sha256": c["checkpoint_sha256"], "exported_policy_sha256": c["exported_policy_sha256"]}
    trainer.complete(args, manifest, report, plan)
    completion = json.loads((output / "cohort_completion_audit.json").read_text())
    assert completion["listed_reward_term_count"] == 15 and completion["nonzero_reward_term_count"] == 14
    assert completion["declared_reward_term_count"] == 15 and completion["sideslip_term_listed"] is True
    assert completion["initial_policy_matches_beta_on"] is True and completion["sideslip_reward_weight"] == 0.
    assert report["beta_ablation"] == trainer.ablation_identity()


@pytest.mark.parametrize("backend", ["physx", "mjlab"])
@pytest.mark.parametrize("smoke", [False, True])
def test_completed_artifacts_have_beta_off_identity_without_unpickling(tmp_path, backend, smoke):
    path = make_cohort(tmp_path, backend=backend, smoke=smoke)
    result = contract.verify_cohort(path, allow_smoke=smoke)
    assert result["training_backend"] == backend and result["smoke_only"] is smoke
    assert result["contract"]["trained_task_profile"]["reward_weights"]["sideslip"] == 0.
    assert contract.provenance(result)["sideslip_reward_weight"] == 0.
    deployed = contract.evaluation_contract(result, .7, "native CPU MuJoCo")
    assert deployed["task_profile"]["reward_weights"]["sideslip"] == 0.
    assert deployed["trained_task_profile"]["contact_friction"] == 1.
    if smoke:
        with pytest.raises(ValueError, match="Smoke actor"): contract.verify_cohort(path)


def test_cannot_relabel_beta_on_old_cohort(tmp_path):
    with pytest.raises(ValueError): contract.verify_cohort(make_baseline(tmp_path))


def test_rehashed_profile_cannot_restore_beta_on(tmp_path):
    path = make_cohort(tmp_path)
    value = json.loads((path / "final_policy_contract.json").read_text())
    for key in ("task_profile", "trained_task_profile"):
        value[key]["reward_weights"]["sideslip"] = 2.
    for name in ("final_policy_contract.json", "policy_contract.json"):put(path / name, value)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(contract=value, task_profile=value["task_profile"]);put(path / "manifest.json", manifest)
    refresh(path)
    with pytest.raises(ValueError, match="single sideslip"):contract.verify_cohort(path)


def test_contract_adapter_rejects_changed_baseline_source_and_new_eval_names_bound():
    source = (ROOT / "scripts/ppo_backend_400hz_contract.py").read_text()
    generated = contract.materialize_source(source)
    assert '"eval_ppo_backend_400hz_beta_off_isaac.py"' in generated
    assert '"eval_ppo_backend_400hz_beta_off_mujoco.py"' in generated
    with pytest.raises(ValueError, match="contract SHA"):
        contract.materialize_source(source + "\n")


def test_baseline_module_and_files_unmodified_after_beta_imports():
    assert baseline.sha256(ROOT / "scripts/train_ppo_backend_400hz.py") == trainer.BASELINE_WRAPPER_SHA256
    assert baseline.validate_protocol(json.loads(trainer.BASELINE_PROTOCOL.read_text()))
    assert baseline.derive_inputs is not trainer.derive_inputs
    import ppo_backend_400hz_contract as old
    assert old.DEFAULT_PROTOCOL != contract.DEFAULT_PROTOCOL
    assert old.TRAINER_SHA256 == trainer.BASELINE_WRAPPER_SHA256
