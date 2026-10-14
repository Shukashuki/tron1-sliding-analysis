import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import full_horizon_diagnostic as diagnostic
import materialize_full_horizon_source as materializer


def test_pose_conditions_are_observed_and_not_returned_as_stops():
    def original(env, threshold):
        return np.array([env.error > threshold])

    terms = SimpleNamespace(**{name: SimpleNamespace(func=original)
                               for name in diagnostic.POSE_TERMS})
    cfg = SimpleNamespace(terminations=terms)
    env = SimpleNamespace(num_envs=1, _sim_step_counter=0, physics_dt=.005, error=1.)
    diagnostic.reset_isaac_diagnostics(env)
    diagnostic.install_isaac_diagnostics(cfg)
    env._sim_step_counter = 4
    for name in diagnostic.POSE_TERMS:
        func = getattr(terms, name).func
        assert inspect.signature(func) == inspect.signature(original)
        assert not func(env, threshold=.35).any()
    report = diagnostic.isaac_diagnostic_report(env)
    assert all(item["active_policy_steps"] == 1 and item["first_trigger_time_s"] == .02
               for item in report.values())


def test_full_horizon_materialization_preserves_sources_and_compiles(tmp_path):
    source = tmp_path / "base"
    for name in ("training/tron1_drift.py", "scripts/eval_tracking.py",
                 "scripts/eval_tracking_mujoco.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "upstream/tron1-learning-from-human" / name).read_bytes())
    old_hash = materializer.wide.sha(source / "scripts/eval_tracking_mujoco.py")
    output = tmp_path / "diagnostic"
    audit = materializer.materialize(source, output)
    assert audit["full_horizon_diagnostic"] is True
    assert audit["training_unchanged"] is True
    assert materializer.wide.sha(source / "scripts/eval_tracking_mujoco.py") == old_hash
    for name, expected in audit["changed_output_sha256"].items():
        assert materializer.wide.sha(output / name) == expected
        compile((output / name).read_text(), name, "exec")
    code = (output / "scripts/eval_tracking_mujoco.py").read_text()
    assert "if terms and not diagnostic_mode:" in code
    assert "Unexpected MuJoCo time reset" in code
    with pytest.raises(FileExistsError):
        materializer.materialize(source, output)


def test_nonfinite_snapshot_is_rejected():
    keys = ("root_position_m", "root_quaternion_wxyz", "root_linear_velocity_w_m_s",
            "root_angular_velocity_w_rad_s", "joint_position_rad", "joint_velocity_rad_s")
    state = {key: np.ones((1, 3)) for key in keys}
    diagnostic.validate_snapshot(state)
    state["root_position_m"][0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        diagnostic.validate_snapshot(state)
