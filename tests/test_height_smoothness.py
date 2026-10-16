import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import height_smoothness as smooth


def settings():
    return json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())["height_regularization"]


def fake_isaac(monkeypatch):
    module = ModuleType("isaaclab.managers")
    module.RewardTermCfg = SimpleNamespace

    class ManagerTermBase:
        def __init__(self, cfg, env):
            pass

    module.ManagerTermBase = ManagerTermBase
    monkeypatch.setitem(sys.modules, "isaaclab.managers", module)


def make_cfg():
    reward = lambda: SimpleNamespace(func=object(), weight=1., params={"std": .3})
    return SimpleNamespace(
        rewards=SimpleNamespace(motion_global_anchor_pos=reward(), motion_body_pos=reward(),
                                motion_body_lin_vel=reward(), motion_body_ori=reward(),
                                action_rate_l2=reward(), drift_wheel_vertical_velocity=reward()),
        terminations=SimpleNamespace(anchor_pos=object(), ee_body_pos=object(),
                                     anchor_ori=object(), motion_end=object()))


def test_planar_error_ignores_arbitrary_actual_and_reference_height():
    target = np.array([[1., 2., .75], [3., 4., .75]])
    actual = np.array([[1.1, 1.8, -12.], [2.8, 4.1, 100.]])
    np.testing.assert_allclose(smooth.planar_squared_error(target, actual), [.05, .05])
    target[:, 2] = [-100., 200.]
    np.testing.assert_allclose(smooth.planar_squared_error(target, actual), [.05, .05])


def test_acceleration_cost_allows_constant_vertical_travel_not_just_zero_speed():
    np.testing.assert_array_equal(smooth.acceleration_cost(
        np.array([-2., 0., 3.]), np.array([-2., 0., 3.]), .02, 5., np.exp), [0., 0., 0.])
    expected = 1. - np.exp(-1.)
    np.testing.assert_allclose(smooth.acceleration_cost(
        np.array([.1, -.1]), np.zeros(2), .02, 5., np.exp), [expected, expected])
    # Same physical acceleration at a different control clock, and no sign bias.
    assert smooth.acceleration_cost(.2, 0., .04, 5., np.exp) == pytest.approx(expected)
    assert 0. <= smooth.acceleration_cost(100., 0., .02, 5., np.exp) <= 1.


@pytest.mark.parametrize("key,value", [("weight", .1), ("weight", 0.), ("weight", True),
                                     ("acceleration_scale_m_s2", 0.),
                                     ("acceleration_scale_m_s2", float("nan")),
                                     ("mode", "constant_height")])
def test_invalid_smoothness_settings_fail_closed(key, value):
    with pytest.raises(ValueError):
        smooth.validate_settings({**settings(), key: value})


def test_installer_removes_both_z_gates_and_hidden_z_reward_but_keeps_stability(monkeypatch):
    fake_isaac(monkeypatch)
    monkeypatch.setattr(smooth, "vertical_acceleration_term", lambda: object)
    cfg = make_cfg()
    orientation, safety, action = cfg.rewards.motion_body_ori, cfg.terminations.anchor_ori, cfg.rewards.action_rate_l2
    contract = smooth.install_height_smoothness(cfg, settings())
    assert cfg.rewards.motion_global_anchor_pos.func is smooth.anchor_xy_tracking
    assert cfg.rewards.motion_body_pos.func is smooth.body_xy_tracking
    assert cfg.rewards.motion_body_lin_vel.func is smooth.body_xy_velocity_tracking
    assert cfg.rewards.drift_wheel_vertical_velocity is None
    assert cfg.terminations.anchor_pos is None and cfg.terminations.ee_body_pos is None
    assert cfg.rewards.motion_body_ori is orientation and cfg.terminations.anchor_ori is safety
    assert cfg.rewards.action_rate_l2 is action
    assert cfg.terminations.motion_end is not None
    assert cfg.rewards.height_vertical_acceleration.weight == -.1
    assert contract["absolute_height_reward"] is False
    assert contract["height_termination"] is False


def test_profiles_preserve_archived_recipe_and_source_validation_accepts_new_profile():
    old = json.loads((ROOT / "config/car_drift_beyondmimic_task.json").read_text())
    new = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    assert "height_regularization" not in old
    assert old["reward_weights"] == new["reward_weights"]
    spec = importlib.util.spec_from_file_location("height_test_drift", ROOT / "upstream/tron1-learning-from-human/training/tron1_drift.py")
    drift = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drift)
    assert drift.validate_task_profile(new) == new


def test_validation_cannot_restore_z_stops_or_relabel_old_checkpoint(monkeypatch):
    fake_isaac(monkeypatch)
    monkeypatch.setattr(smooth, "vertical_acceleration_term", lambda: object)
    task = {"height_regularization": settings()}
    with pytest.raises(ValueError, match="full-horizon"):
        smooth.require_height_free_validation(task, {}, False)
    with pytest.raises(ValueError, match="provenance"):
        smooth.require_height_free_validation(task, {}, True)
    contract = smooth.install_height_smoothness(make_cfg(), settings())
    smooth.require_height_free_validation(task, {"height_regularization": contract,
                                               "height_adapter_sha256": "recorded-source-hash"}, True)
    smooth.require_height_free_validation({}, {}, False)


def test_torch_managed_term_partial_reset_and_repeated_reads(monkeypatch):
    torch = pytest.importorskip("torch")
    fake_isaac(monkeypatch)
    robot = SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=torch.zeros(2, 3)))
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, common_step_counter=1,
                          scene={"robot": robot})
    term = smooth.vertical_acceleration_term()(None, env)
    # Initial/reset velocity itself must not be interpreted as acceleration.
    robot.data.root_lin_vel_w[:, 2] = torch.tensor([3., -2.])
    torch.testing.assert_close(term(env, 5.), torch.zeros(2))
    env.common_step_counter += 1
    robot.data.root_lin_vel_w[:, 2] += .1
    value = term(env, 5.).clone()
    torch.testing.assert_close(value, torch.full((2,), 1. - np.exp(-1.)))
    torch.testing.assert_close(term(env, 5.), value)  # No double history update.
    term.reset(torch.tensor([0]))
    env.common_step_counter += 1
    robot.data.root_lin_vel_w[:, 2] = torch.tensor([100., -1.8])
    value = term(env, 5.)
    assert value[0].item() == 0.
    assert value[1].item() == pytest.approx(1. - np.exp(-1.), abs=2e-6)
    term.reset()
    env.common_step_counter += 1
    torch.testing.assert_close(term(env, 5.), torch.zeros(2))
