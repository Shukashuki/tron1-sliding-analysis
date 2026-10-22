"""No-simulator stiffness-dispatch command and evidence checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ppo_contact_stiffness as module


def test_single_factor_command(tmp_path):
    args = SimpleNamespace(**{name: tmp_path / name for name in
        ("python", "legacy_evaluator", "motion_file", "asset_path", "checkpoint", "runner_config", "task_profile")})
    cmd = module.command(args, 250., tmp_path / "new", "KIT")
    assert cmd[2].endswith("eval_ppo_contact_stiffness.py")
    assert cmd[cmd.index("--leg-kp") + 1] == "250.0"
    assert "--leg-kd" not in cmd and "--deployment-leg-kp" not in cmd
    assert cmd[cmd.index("--evaluation-contact-friction") + 1] == "0.7"
    assert "--verify-unmasked-device" in cmd


@pytest.fixture
def output(tmp_path):
    raw = tmp_path / "physics_trajectory.npz"
    np.savez_compressed(raw, time_s=np.arange(1397)*.005, valid_mask=np.ones((1397, 1), dtype=bool))
    report = {"status": "evaluated", "simulator": "IsaacLab/PhysX",
              "checkpoint_sha256": module.base.CHECKPOINT_SHA,
              "contact_stability_diagnostic": {
                  "kind": "fixed_ppo_nominal_contact_stiffness_diagnostic", "deployment_leg_kp": 250.,
                  "trained_leg_kp": 500., "unchanged_leg_kd": 10., "additional_training_updates": 0,
                  "diagnostic_only": True},
              "contact_material_audit": {"leg_pd_gain_readback": {
                  "kp_range": [250., 250.], "kd_range": [10., 10.]}},
              "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
              "physics_trajectory_sha256": module.base.sha256(raw)}
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path, report


def test_complete_candidate(output):
    folder, _ = output
    assert module.validate_output(folder, 250.)["status"] == "verified"


@pytest.mark.parametrize("problem", ["damping", "mu", "kp", "updates"])
def test_reject_combined_or_wrong_override(output, problem):
    folder, report = output
    if problem == "damping":
        report["contact_material_audit"]["leg_pd_gain_readback"]["kd_range"] = [5., 5.]
    elif problem == "mu":
        report["evaluation_contact_friction_override"]["evaluation_contact_friction"] = .15
    elif problem == "kp":
        report["contact_stability_diagnostic"]["deployment_leg_kp"] = 300.
    else:
        report["contact_stability_diagnostic"]["additional_training_updates"] = 1
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module.validate_output(folder, 250.)
