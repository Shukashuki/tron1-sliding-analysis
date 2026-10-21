"""Pure-Python interface checks; actual PhysX execution requires remote smoke."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from physx_batch_predictor import (
    PhysXBatchTrackingPredictor, canonical_measurement, validate_physx_config,
)


def test_canonical_measurement_uses_link_pose_and_world_com_velocity():
    qpos = np.arange(15, dtype=float)
    qvel = np.arange(14, dtype=float)+100.
    history = np.zeros((3, 9, 3))
    measured = {
        "body_pos_w": np.tile([1., 2., 3.], (9, 1)),
        "body_quat_w": np.tile([1., 0., 0., 0.], (9, 1)),
        "body_lin_vel_w": np.tile([9., 8., 7.], (9, 1)),
        "body_ang_vel_w": np.tile([6., 5., 4.], (9, 1)),
    }
    base = SimpleNamespace(
        layout=SimpleNamespace(unpack=lambda _: (qpos, qvel, history, True)),
        provider=SimpleNamespace(qids=np.arange(7, 15), vids=np.arange(6, 14)),
        measure=lambda _: measured,
    )
    result = canonical_measurement(base, np.zeros(111))
    np.testing.assert_array_equal(result["root_pos"], [1., 2., 3.])
    np.testing.assert_array_equal(result["root_com_lin_vel_w"], [9., 8., 7.])
    np.testing.assert_array_equal(result["joint_vel"], qvel[6:])
    assert result["first_reward_after_reset"]
    result["root_pos"][0] = -100.
    assert measured["body_pos_w"][0, 0] == 1.


def config(tmp_path):
    for name in ("motion.npz", "robot.usda", "profile.json"):
        (tmp_path/name).touch()
    return {"schema_version": 1, "source_root": str(tmp_path),
            "motion": str(tmp_path/"motion.npz"), "asset_path": str(tmp_path/"robot.usda"),
            "task_profile": str(tmp_path/"profile.json"), "device": "cuda:0"}


def test_config_sets_explicit_defaults_and_rejects_cpu(tmp_path):
    value = config(tmp_path)
    validated = validate_physx_config(value)
    assert validated["max_candidates"] == 64
    assert validated["environment_spacing_m"] == 20.
    with pytest.raises(ValueError, match="cuda:N"):
        validate_physx_config({**value, "device": "cpu"})


@pytest.mark.parametrize("field,value", [("max_candidates", True), ("timeout_s", float("nan")),
                                        ("environment_spacing_m", 1.), ("unknown", 1)])
def test_config_rejects_invalid_or_unknown_settings(tmp_path, field, value):
    with pytest.raises(ValueError):
        validate_physx_config({**config(tmp_path), field: value})


def test_repeatability_probe_handles_permutations_and_intervening_rollout():
    worker = object.__new__(PhysXBatchTrackingPredictor)
    worker.metadata = {}
    worker.mu_scenarios = (.7,)
    calls = []

    def evaluate(state, controls, phase, **kwargs):
        calls.append(controls.copy())
        values = controls.sum(axis=-1)[..., None]
        return {"candidate_costs": controls.sum(axis=(1, 2)),
                "trajectory": {name: values for name in
                    ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos")}}

    worker.evaluate_batch = evaluate
    controls = np.arange(48).reshape(2, 3, 8)/100.
    result = worker.probe_repeatability(np.zeros(111), controls, 12,
                                       previous_action=np.zeros(8), previous_vz=0.)
    assert result["passed"]
    assert len(calls) == 5
    np.testing.assert_array_equal(calls[0], calls[2])
    assert not np.array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(calls[3], controls[::-1])
    np.testing.assert_array_equal(calls[4][1:], controls[1:])
