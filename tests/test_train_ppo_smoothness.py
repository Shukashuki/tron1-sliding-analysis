"""CPU-only contract tests; no Isaac import or training job is launched."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_ppo_smoothness as trainer


def action_rate_l2():
    pass


action_rate_l2.__module__ = "isaaclab.envs.mdp.rewards"


@pytest.fixture
def plan(tmp_path):
    return {"action_rate_weight": -.3, "output_dir": tmp_path, "trainer": Path(trainer.__file__).with_name("train_beyondmimic_style.py"),
            "source_root": tmp_path/"source", "smoke_only": False, "num_envs": 512,
            "iterations": 1000, "seed": 42, "baseline_source_sha256": trainer.BASELINE_SOURCES,
            "inherited_argv": ["--seed", "42"]}


def cfg(weight=-.1, func=action_rate_l2):
    return SimpleNamespace(rewards=SimpleNamespace(action_rate_l2=SimpleNamespace(weight=weight, func=func)))


def test_hook_only_changes_one_reward_and_preserves_recipe(plan):
    configuration = cfg()
    configuration.unrelated = object()
    before = configuration.unrelated
    recipe = {"friction_static": [.3, 1.6], "push_interval_s": [1., 3.]}
    calls = []
    def original(value, *, ground_friction):
        calls.append((value, ground_friction))
        return recipe
    hook = trainer.make_override_hook(original, plan)
    assert hook(configuration, ground_friction=1.) is recipe
    assert calls == [(configuration, 1.)]
    assert configuration.unrelated is before
    assert configuration.rewards.action_rate_l2.weight == -.3
    report = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())
    assert report["reward_override"]["baseline_weight"] == -.1
    assert report["reward_override"]["weight"] == -.3
    assert report["reward_override"]["applied_before_environment_creation"] is True
    assert report["initialization"] == "from_scratch_no_checkpoint_resume"
    assert report["DR_recipe_modified"] is False
    assert report["expected_environment_transitions"] == 12288000
    assert report["trained_policy_claimed"] is False


@pytest.mark.parametrize("weight", [0., -.2, -1.])
def test_rejects_unexpected_baseline_weight(plan, weight):
    with pytest.raises(ValueError, match="baseline action-rate weight"):
        trainer.make_override_hook(lambda *args, **kwargs: {}, plan)(cfg(weight))
    assert not (plan["output_dir"]/"ablation_manifest.json").exists()


def test_rejects_unexpected_reward_function(plan):
    with pytest.raises(ValueError, match="reward function"):
        trainer.make_override_hook(lambda *args, **kwargs: {}, plan)(cfg(func=lambda: None))


def test_hook_preserves_existing_ablation_record(plan):
    path = plan["output_dir"]/"ablation_manifest.json"
    path.write_text("old evidence")
    with pytest.raises(FileExistsError):
        trainer.make_override_hook(lambda *args, **kwargs: {}, plan)(cfg())
    assert path.read_text() == "old evidence"


def test_final_manifest_explicitly_discloses_override(plan):
    trainer.make_override_hook(lambda *args, **kwargs: {}, plan)(cfg())
    output = plan["output_dir"]
    (output/"run_status.json").write_text(json.dumps({"status": "training_completed", "trained_policy_claimed": True,
                                                    "environment_transitions": 12288000}))
    (output/"manifest.json").write_text(json.dumps({"task_profile_sha256": trainer.BASELINE_PROFILE_SHA256}))
    (output/"model_final.pt").write_bytes(b"test fixture, not a policy")
    trainer.finalize_manifest(plan)
    metadata = json.loads((output/"ablation_manifest.json").read_text())
    manifest = json.loads((output/"manifest.json").read_text())
    assert metadata["status"] == "training_completed"
    assert metadata["trained_policy_claimed"] is True
    assert metadata["checkpoint_sha256"] == trainer.sha256(output/"model_final.pt")
    assert manifest["training_reward_overrides"] == {"action_rate_l2": -.3}
    assert manifest["historical_profile_hash_does_not_include_explicit_reward_override"] is True
    assert metadata["final_training_manifest_sha256"] == trainer.sha256(output/"manifest.json")


def test_failed_run_never_claims_trained_policy(plan):
    trainer.make_override_hook(lambda *args, **kwargs: {}, plan)(cfg())
    trainer.finalize_manifest(plan, failed=True)
    report = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())
    assert report["status"] == "failed" and report["trained_policy_claimed"] is False
    assert "checkpoint_sha256" not in report


def _arguments(tmp_path):
    return ["--source-root", str(tmp_path/"source"), "--motion-file", "motion.npz", "--asset-path", "robot.usda",
            "--task-profile", str(tmp_path/"profile.json"), "--output-dir", str(tmp_path/"new-output")]


@pytest.mark.parametrize("value", ["nan", "inf", "-.1", "0", "-11"])
def test_invalid_override_rejected_before_import_or_training(tmp_path, value):
    with pytest.raises(ValueError, match="stronger negative"):
        trainer.prepare_plan(["--action-rate-weight", value, *_arguments(tmp_path)])


@pytest.mark.parametrize("extra", [["--seed", "7"], ["--ground-friction", ".7"],
                                   ["--smoke-steps", "8"], ["--iterations", "1"], ["--resume"]])
def test_nonisolated_formal_configuration_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        trainer.prepare_plan([*_arguments(tmp_path), *extra])


def test_existing_output_rejected_before_training(tmp_path):
    (tmp_path/"new-output").mkdir()
    with pytest.raises(ValueError, match="output directory must not exist"):
        trainer.prepare_plan(_arguments(tmp_path))


def test_source_hash_failure_rejected_before_delegate_import(tmp_path):
    with pytest.raises(ValueError, match="Historical source/profile SHA mismatch"):
        trainer.prepare_plan(_arguments(tmp_path))


def test_duplicate_source_option_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate option"):
        trainer.prepare_plan([*_arguments(tmp_path), "--source-root", "another"])


def test_smoke_wrong_scope_rejected(tmp_path):
    with pytest.raises(ValueError, match="Smoke must use exactly"):
        trainer.prepare_plan([*_arguments(tmp_path), "--smoke-only"])


def test_historical_trainer_and_height_files_are_unchanged():
    assert trainer.sha256(trainer.ROOT/"scripts/train_beyondmimic_style.py") == trainer.BASELINE_TRAINER_SHA256
    assert trainer.sha256(trainer.ROOT/"scripts/height_smoothness.py") == trainer.BASELINE_HEIGHT_SHA256


def test_option_equals_and_default():
    assert trainer._option(["--seed=42"], "--seed") == "42"
    assert trainer._option([], "--seed", "42") == "42"


def test_other_reference_is_rejected_before_training(tmp_path):
    with pytest.raises(ValueError, match="primary traj_4"):
        trainer.prepare_plan([*_arguments(tmp_path), "--reference-id", "doa_sa_traj29"])


def test_locked_motion_and_asset_identities_are_recorded(plan):
    trainer.make_override_hook(lambda *args, **kwargs: {}, plan)(cfg())
    report = json.loads((plan["output_dir"]/"ablation_manifest.json").read_text())
    assert report["motion_sha256"] == trainer.BASELINE_MOTION_SHA256
    assert report["asset_sha256"] == trainer.BASELINE_ASSET_SHA256
