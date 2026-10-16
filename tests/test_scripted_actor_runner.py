"""Artifact provenance and raw-observation export tests, without Isaac startup."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripted_actor_runner import ScriptedActorRunner, sha256


class TinyNormalizedActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.ones(51))
        self.register_buffer("scale", torch.full((51,), 2.))

    def forward(self, obs):
        return torch.tanh(((obs - self.mean) / self.scale)[:, :8])


class UnboundedActor(torch.nn.Module):
    def forward(self, obs):
        return obs[:, :8] + 10.


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_run(tmp_path, smoke=False, actor=None):
    root = tmp_path / "source_run"
    root.mkdir()
    # Intentionally NOT a valid pickle: inference must only hash this file.
    checkpoint = root / ("model_smoke.pt" if smoke else "model_final.pt")
    checkpoint.write_bytes(b"optimizer checkpoint is opaque to the inference adapter")
    actor_path = root / ("actor_smoke.pt" if smoke else "actor_normalized.pt")
    torch.jit.script(TinyNormalizedActor() if actor is None else actor).save(str(actor_path))
    settings = {"mode": "vertical_acceleration_only", "weight": -.1, "acceleration_scale_m_s2": 5.}
    height = {**settings, "absolute_height_reward": False, "height_termination": False,
              "reference_vertical_velocity_reward": False}
    contract = {"actor_dim": 51, "action_dim": 8, "critic_dim": 120,
                "policy_algorithm": "SAC", "actor_format": "torchscript_raw_observation_deterministic_tanh",
                "height_regularization": height}
    config = {"algorithm": "SAC", "sac": {"actor_dim": 51, "action_dim": 8, "critic_dim": 120}}
    manifest = {"algorithm": "SAC", "smoke_only": smoke, "sac_config": config["sac"],
                "contract": contract, "policy_contract": contract,
                "height_regularization": height, "height_adapter_sha256": "a" * 64,
                "task_profile": {"height_regularization": settings}}
    status = {"status": "smoke_passed" if smoke else "training_completed", "smoke_only": smoke,
              "trained_policy_claimed": not smoke, "actor_sha256": sha256(actor_path),
              "checkpoint_sha256": sha256(checkpoint)}
    for name, value in (("manifest.json", manifest), ("run_status.json", status),
                        ("policy_contract.json", contract), ("runner_config.json", config)):
        write_json(root / name, value)
    return checkpoint, config


def change_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    write_json(path, value)


def test_final_sac_actor_uses_raw_obs_once_and_never_unpickles_checkpoint(tmp_path, monkeypatch):
    checkpoint, config = make_run(tmp_path)
    runner = ScriptedActorRunner(None, config, device="cpu")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Inference must not unpickle a SAC checkpoint"))
    provenance = runner.load(checkpoint, load_optimizer=False, map_location="cpu")
    assert isinstance(runner.alg.policy, torch.nn.Module)
    assert isinstance(runner.alg.policy.actor_obs_normalizer, torch.nn.Identity)
    assert runner.alg.policy.is_recurrent is False
    raw = torch.randn(3, 51)
    actual = runner.get_inference_policy(device="cpu")({"policy": raw, "critic": torch.randn(3, 120)})
    torch.testing.assert_close(actual, TinyNormalizedActor()(raw))
    torch.testing.assert_close(runner.alg.policy.get_actor_obs({"policy": raw}), raw)
    assert not actual.requires_grad
    assert provenance["checkpoint_deserialized"] is False and provenance["trained_policy_claimed"] is True
    assert runner.actor_sha256 == sha256(Path(runner.actor_file_path))


def test_unloaded_runner_and_optimizer_loading_are_not_silently_supported(tmp_path):
    checkpoint, config = make_run(tmp_path)
    runner = ScriptedActorRunner(None, config)
    with pytest.raises(RuntimeError, match="Load"):
        runner.get_inference_policy()
    with pytest.raises(ValueError, match="optimizer"):
        runner.load(checkpoint, load_optimizer=True)


def test_pinned_evaluator_final_report_observation_group_expression(tmp_path):
    checkpoint, config = make_run(tmp_path)
    runner = ScriptedActorRunner(None, config)
    runner.load(checkpoint)
    actor = runner.alg.policy
    # Exact field access in pinned eval_tracking.py:671. Export/inference tests
    # alone do not reach this final report-building path.
    metadata = {"actor_observation_groups": actor.obs_groups["policy"]}
    assert metadata == {"actor_observation_groups": ["policy"]}
    assert actor.get_actor_obs({"policy": torch.zeros(2, 51)}).shape == (2, 51)


@pytest.mark.parametrize("field", ["checkpoint", "actor"])
def test_changed_artifact_hash_is_rejected_before_actor_load(tmp_path, field):
    checkpoint, config = make_run(tmp_path)
    changed = checkpoint if field == "checkpoint" else checkpoint.parent / "actor_normalized.pt"
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        ScriptedActorRunner(None, config).load(checkpoint)


def test_smoke_is_explicitly_opt_in_and_never_a_trained_policy_claim(tmp_path):
    checkpoint, config = make_run(tmp_path, smoke=True)
    with pytest.raises(ValueError, match="Smoke actor"):
        ScriptedActorRunner(None, config).load(checkpoint)
    runner = ScriptedActorRunner(None, {**config, "allow_smoke_actor": True})
    provenance = runner.load(checkpoint)
    assert provenance["smoke_only"] is True and provenance["trained_policy_claimed"] is False
    assert runner.actor_file_path.endswith("actor_smoke.pt")


@pytest.mark.parametrize("status", ["training", "failed", "initializing", "smoke_running"])
def test_incomplete_or_failed_training_is_rejected(tmp_path, status):
    checkpoint, config = make_run(tmp_path)
    change_json(checkpoint.parent / "run_status.json", lambda item: item.update(status=status))
    with pytest.raises(ValueError, match="completed"):
        ScriptedActorRunner(None, config).load(checkpoint)


@pytest.mark.parametrize("field,value", [("actor_dim", 50), ("action_dim", 9),
                                         ("policy_algorithm", "PPO")])
def test_wrong_or_inconsistent_policy_contract_is_rejected(tmp_path, field, value):
    checkpoint, config = make_run(tmp_path)
    change_json(checkpoint.parent / "policy_contract.json", lambda item: item.update({field: value}))
    with pytest.raises(ValueError, match="contract"):
        ScriptedActorRunner(None, config).load(checkpoint)


@pytest.mark.parametrize("field", ["height_adapter_sha256", "height_regularization"])
def test_height_free_provenance_is_mandatory(tmp_path, field):
    checkpoint, config = make_run(tmp_path)
    change_json(checkpoint.parent / "manifest.json", lambda item: item.pop(field))
    with pytest.raises(ValueError, match="height-smoothness"):
        ScriptedActorRunner(None, config).load(checkpoint)


def test_wrong_runner_algorithm_and_boolean_smoke_flag_are_rejected():
    with pytest.raises(ValueError, match="SAC"):
        ScriptedActorRunner(None, {"algorithm": "PPO"})
    with pytest.raises(ValueError, match="boolean"):
        ScriptedActorRunner(None, {"algorithm": "SAC", "allow_smoke_actor": "false"})


def test_valid_hash_does_not_excuse_unbounded_actor_outputs(tmp_path):
    checkpoint, config = make_run(tmp_path, actor=UnboundedActor())
    with pytest.raises(ValueError, match="normalized actions"):
        ScriptedActorRunner(None, config).load(checkpoint)


def test_actual_isaac_lab_exporter_deepcopies_and_reexports_scripted_actor(tmp_path):
    source = os.environ.get("ISAACLAB_RSL_EXPORTER")
    if source is None:
        pytest.skip("Set ISAACLAB_RSL_EXPORTER to test the installed Isaac Lab exporter without simulator startup")
    spec = importlib.util.spec_from_file_location("_test_installed_isaac_exporter", source)
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)
    checkpoint, config = make_run(tmp_path)
    runner = ScriptedActorRunner(None, config)
    runner.load(checkpoint)
    exporter.export_policy_as_jit(runner.alg.policy, normalizer=runner.alg.policy.actor_obs_normalizer,
                                 path=str(tmp_path / "evaluation"), filename="actor_normalized.pt")
    reexported = torch.jit.load(str(tmp_path / "evaluation/actor_normalized.pt"))
    raw = torch.randn(4, 51)
    expected = runner.get_inference_policy()({"policy": raw})
    torch.testing.assert_close(reexported(raw), expected)
    reexported.reset()
