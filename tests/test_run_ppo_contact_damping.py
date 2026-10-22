"""No-GPU checks for the fixed-policy damping diagnostic dispatcher."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ppo_contact_damping as module


@pytest.fixture
def output(tmp_path):
    raw = tmp_path / "physics_trajectory.npz"
    np.savez_compressed(raw, time_s=np.arange(1397) * .005,
                        valid_mask=np.ones((1397, 1), dtype=bool))
    report = {"status": "evaluated", "simulator": "IsaacLab/PhysX",
              "checkpoint_sha256": module.CHECKPOINT_SHA,
              "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
              "physics_dt_s": .005, "reference_frames": 349,
              "contact_stability_diagnostic": {
                  "deployment_leg_kd": 5., "trained_leg_kd": 10., "unchanged_leg_kp": 500.,
                  "additional_training_updates": 0, "checkpoint_sha256": module.CHECKPOINT_SHA,
                  "diagnostic_only": True},
              "contact_material_audit": {"leg_pd_gain_readback": {
                  "kp_range": [500., 500.], "kd_range": [5., 5.]}},
              "physics_trajectory_sha256": module.sha256(raw)}
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path, report


def test_command_is_single_env_fixed_nominal_clock(tmp_path):
    args = SimpleNamespace(**{name: tmp_path / name for name in
        ("python", "legacy_evaluator", "motion_file", "asset_path", "checkpoint", "runner_config", "task_profile")})
    result = module.command(args, 5., tmp_path / "output", "KIT")
    assert result[result.index("--leg-kd") + 1] == "5.0"
    assert result[result.index("--evaluation-contact-friction") + 1] == "0.7"
    assert result[result.index("--num-envs") + 1] == "1"
    assert result[result.index("--device") + 1] == "cuda:1"
    assert "--record-physics" in result and "--verify-unmasked-device" in result
    assert "--deployment-leg-kd" not in result
    assert "--physics-dt-s" not in result


def test_complete_hash_bound_episode(output):
    folder, _ = output
    assert module.validate_output(folder, 5.)["status"] == "verified"


@pytest.mark.parametrize("problem", ["wrong_gain", "wrong_mu", "trained", "wrong_actor", "not_complete"])
def test_reject_inconsistent_episode(output, problem):
    folder, report = output
    if problem == "wrong_gain":
        report["contact_material_audit"]["leg_pd_gain_readback"]["kd_range"] = [10., 10.]
    elif problem == "wrong_mu":
        report["evaluation_contact_friction_override"]["evaluation_contact_friction"] = .15
    elif problem == "trained":
        report["contact_stability_diagnostic"]["additional_training_updates"] = 1
    elif problem == "wrong_actor":
        report["checkpoint_sha256"] = "0" * 64
    else:
        report["status"] = "evaluation_incomplete"
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module.validate_output(folder, 5.)


def test_reject_recording_with_resets(output):
    folder, report = output
    valid = np.ones((1397, 1), dtype=bool)
    valid[-10:] = False
    np.savez_compressed(folder / "physics_trajectory.npz", time_s=np.arange(1397) * .005,
                        valid_mask=valid)
    report["physics_trajectory_sha256"] = module.sha256(folder / "physics_trajectory.npz")
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="native first-episode"):
        module.validate_output(folder, 5.)


def test_explicit_beyondmimic_source_is_required():
    options = module.parser()._option_string_actions
    assert options["--beyondmimic-source"].required is True
    # No implicit fallback to a missing archive-local third_party checkout.
    import inspect
    assert "TRON1_BEYONDMIMIC_ROOT=str(args.beyondmimic_source)" in inspect.getsource(module.run)
