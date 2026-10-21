"""Actual MuJoCo task parity checks; no synthetic experiment trajectories."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mujoco_ppo_env import (AdaptivePhaseSampler, MujocoPPOEnv, build_observations,
                           rotation_matrix)
from nmpc_tracking_objective import BODY_NAMES, HeightSmoothTrackingObjective

MODEL = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
MOTION = Path("/home/roy/tron1-learning-from-human/results/2026-10-14-car-drift/matched-friction-mu0p70/motion.npz")
CONTRACT = ROOT / "config/nominal_actuator_contract.json"
PROFILE = ROOT / "config/car_drift_height_smooth_task.json"


@pytest.fixture
def make_env():
    if not MODEL.is_file() or not MOTION.is_file():
        pytest.skip("Locked TRON1 model/reference assets are not downloaded")
    environments = []
    def make(**kwargs):
        args = dict(num_envs=2, training=False, dr_mode="none", seed=42)
        args.update(kwargs)
        result = MujocoPPOEnv(MODEL, MOTION, CONTRACT, PROFILE, **args)
        environments.append(result)
        return result
    yield make
    for environment in environments:
        environment.close()


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    from mujoco_ppo_native import build_ppo_native_library
    return build_ppo_native_library(tmp_path_factory.mktemp("ppo_native_env_parity"))


def test_actual_observation_matches_existing_deployment_actor_and_manual_critic(make_env):
    env = make_env()
    state, obs = env._state(), env.observations()
    assert obs["policy"].shape == (2, 51) and obs["critic"].shape == (2, 120)
    for index in range(2):
        measured = {"root_pos": state["body_pos_w"][index, 0],
                    "root_quat_wxyz": state["body_quat_w"][index, 0],
                    "root_com_lin_vel_w": state["body_lin_vel_w"][index, 0],
                    "root_ang_vel_w": state["body_ang_vel_w"][index, 0],
                    "joint_pos": state["joint_pos"][index], "joint_vel": state["joint_vel"][index]}
        expected = env.api.build_observation(measured, env.reference.sample(0),
                                             env.q0s[index], np.zeros(8))
        np.testing.assert_allclose(obs["policy"][index], expected, atol=1.e-7, rtol=0.)
        rotation = env.api.rotation(measured["root_quat_wxyz"])
        body_p = (rotation.T@(state["body_pos_w"][index]-measured["root_pos"]).T).T
        body_r = np.array([rotation.T@env.api.rotation(q) for q in state["body_quat_w"][index, :7]])
        critic = np.r_[expected, body_p.reshape(-1), body_r[..., :2].reshape(-1)]
        np.testing.assert_allclose(obs["critic"][index], critic, atol=1.e-7, rtol=0.)
    assert env.frames == 349 and env.max_episode_length == 351
    assert env.metadata()["actor_observation_corruption"] is False


def test_actual_reward_exact_scalar_objective_dt_and_held_reference(make_env):
    env = make_env()
    scalar = HeightSmoothTrackingObjective(env.profile, soft_joint_limits=env.soft_limits)
    actions = np.array([[.004, -.003, .001, -.002, .005, -.004, .1, .1],
                        [-.005, .002, .003, -.001, .002, -.003, .2, .2]], dtype=np.float32)
    pre, reference = env._state(), env._reference()
    held = []
    for index in range(2):
        held.append(scalar.hold_reference({key: reference[key][index] for key in
                                          ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")},
                                         {key: pre[key][index] for key in
                                          ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}))
    # Keep the exact physical last-four force samples used by the environment.
    original, captured = env.batch.step, {}
    def capture(action):
        result = original(action)
        captured.update(result)
        return result
    env.batch.step = capture
    _, rewards, terminated, truncated, extras = env.step(actions)
    state = env._state(captured)
    forces = captured.get("contact_substeps", captured.get("body_net_forces_w"))
    for index in range(2):
        current = {key: state[key][index] for key in
                   ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos")}
        current.update(body_net_forces_w=forces[index, -1], contact_force_history_w=forces[index, -3:])
        scored = scalar.evaluate(current, held[index], actions[index], np.zeros(8), None, reset_sample=True)
        assert rewards[index] == pytest.approx(.02*scored["total_reward"], abs=2.e-8)
        for key, value in scored["raw_terms"].items():
            assert extras["raw_reward_terms"][key][index] == pytest.approx(value, abs=1.e-12)
    assert not terminated.any() and not truncated.any()
    assert np.array_equal(extras["reference_frames_for_action"], [0, 0])
    assert np.array_equal(env.phases, [1, 1])
    assert np.array_equal(extras["raw_reward_terms"]["height_vertical_acceleration"], [0., 0.])
    env.step(actions)
    assert not env.first_reward.any()


def test_native_and_python_persistent_physics_match_on_actual_model(make_env, native_library):
    native = make_env(native_library=native_library, num_threads=2)
    python = make_env()
    actions = np.random.default_rng(1).uniform(-.003, .003, (8, 2, 8)).astype(np.float32)
    for action in actions:
        actual = native.step(action)
        expected = python.step(action)
        for key in ("policy", "critic"):
            np.testing.assert_allclose(actual[0][key], expected[0][key], rtol=0., atol=2.e-7)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0., atol=2.e-8)
        np.testing.assert_array_equal(actual[2], expected[2])
        np.testing.assert_array_equal(actual[3], expected[3])
        np.testing.assert_allclose(native.datas[0].qpos, python.datas[0].qpos, rtol=0., atol=1.e-12)
    assert native.datas[0].time == pytest.approx(.16)
    assert native.total_physical_steps == 64


def test_reset_timeout_flags_phase_and_smoothness_first_sample(make_env):
    env = make_env()
    # Select the actual last reference frame without inventing a trajectory.
    env.phases[:] = env.frames-1
    env.previous_actions[:] = .01
    env.previous_vz[:] = 4.
    env.first_reward[:] = False
    obs, _, terminated, truncated, extras = env.step(np.zeros((2, 8)))
    assert truncated.all()
    np.testing.assert_array_equal(extras["time_outs"], truncated)
    np.testing.assert_array_equal(extras["reference_frames_for_action"], [348, 348])
    assert extras["terminal_observation"]["policy"].shape == (2, 51)
    np.testing.assert_array_equal(env.phases, [0, 0])
    np.testing.assert_array_equal(env.episode_length_buf, [0, 0])
    np.testing.assert_array_equal(env.previous_actions, np.zeros((2, 8)))
    assert env.first_reward.all()
    assert all(data.time == 0. for data in env.datas)
    _, _, _, _, next_extras = env.step(np.zeros((2, 8)))
    np.testing.assert_array_equal(next_extras["raw_reward_terms"]["height_vertical_acceleration"], [0., 0.])


def test_no_absolute_xy_or_height_gate_and_projected_gravity_tilt(make_env):
    env = make_env()
    for data, model in zip(env.datas, env.models):
        data.qpos[:3] += [10., -10., 2.]
        mujoco.mj_forward(model, data)
    _, _, terminated, truncated, _ = env.step(np.zeros((2, 8)))
    assert not terminated.any() and not truncated.any()
    for data, model in zip(env.datas, env.models):
        data.qpos[3:7] = [0., 1., 0., 0.]
        mujoco.mj_forward(model, data)
    _, _, terminated, _, _ = env.step(np.zeros((2, 8)))
    assert terminated.all()


def test_common_dr_startup_draws_reset_com_velocity_and_default_offset(make_env, native_library):
    env = make_env(native_library=native_library, training=True, dr_mode="common_mu", num_envs=8)
    draw = env.robot_mu.copy()
    defaults = env.q0s.copy()
    assert np.ptp(draw) > .1 and np.max(np.abs(defaults[:, :6]-env.reference.q0[:6])) <= .01
    assert np.max(np.abs(env.com_offsets)) <= .05
    state, reference = env._state(), env._reference()
    np.testing.assert_allclose(state["body_lin_vel_w"][:, 0], reference["body_lin_vel_w"][:, 0], atol=1.e-12)
    np.testing.assert_allclose(state["body_ang_vel_w"][:, 0], reference["body_ang_vel_w"][:, 0], atol=1.e-12)
    np.testing.assert_allclose(state["joint_pos"], reference["joint_pos"], atol=1.e-12)
    obs = env.observations()["policy"]
    np.testing.assert_allclose(obs[:, 29:35], state["joint_pos"][:, :6]-defaults[:, :6], atol=1.e-7)
    env.reset()
    np.testing.assert_array_equal(env.robot_mu, draw)
    np.testing.assert_array_equal(env.q0s, defaults)
    metadata = env.metadata()
    assert metadata["dr"]["beyondmimic_full_material_recipe_equivalent"] is False
    assert metadata["dr"]["static_dynamic_distinction"] is False


def test_mjlab_recipe_wheel_priority_bias_and_unbiased_critic(make_env):
    env = make_env(training=True, dr_mode="mjlab_v1p4", num_envs=8)
    assert env.mu_range == (.3, 1.2)
    for i, model in enumerate(env.models):
        wheels = np.isin(model.geom_bodyid, env.bids[-2:])
        collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        np.testing.assert_allclose(model.geom_friction[wheels & collision, 0], env.robot_mu[i])
        np.testing.assert_array_equal(model.geom_priority[wheels & collision], 1)
        np.testing.assert_allclose(model.geom_friction[model.geom_bodyid == 0, 0], 1.)
        # The wheel's priority wins rather than max(ground_mu,wheel_mu).
        np.testing.assert_array_equal(model.geom_priority[(~wheels) & collision], 0)
    state, obs = env._state(), env.observations()
    actual = state["joint_pos"][:, :6]-env.reference.q0[:6]
    np.testing.assert_allclose(obs["policy"][:, 29:35], actual+env.encoder_bias[:, :6], atol=1.e-7)
    np.testing.assert_allclose(obs["critic"][:, 29:35], actual, atol=1.e-7)
    np.testing.assert_allclose(env.q0s[:, :6], env.reference.q0[:6]-env.encoder_bias[:, :6])
    assert env.metadata()["dr"]["push_velocity_frame"] == "root link world, additive"


def test_mjlab_push_adds_link_not_com_velocity(make_env):
    env = make_env(training=True, dr_mode="mjlab_v1p4")
    initial = [data.qvel.copy() for data in env.datas]
    env.push_timer[:] = -1.
    rng = np.random.default_rng()
    rng.bit_generator.state = env.rng.bit_generator.state
    from mujoco_ppo_env import PUSH_BOUNDS
    delta = rng.uniform(PUSH_BOUNDS[:, 0], PUSH_BOUNDS[:, 1], (env.num_envs, 6))
    env._apply_pushes()
    for i, data in enumerate(env.datas):
        np.testing.assert_allclose(data.qvel[:3]-initial[i][:3], delta[i, :3], atol=1.e-12)
    assert env.total_pushes == env.num_envs


def test_push_adds_com_velocity_after_reward_and_resets_episode_timer(make_env, native_library):
    env = make_env(native_library=native_library, training=True, dr_mode="common_mu")
    original = env.push_timer[1]
    env.push_timer[0] = -.1
    env._reset_indices(np.array([0]))
    assert 1. <= env.push_timer[0] <= 3.
    assert env.push_timer[1] == original
    # This is positive after decrement but below the installed 1e-6 trigger.
    env.push_timer[:] = [.0200005, 10.]
    original_step, captured = env.batch.step, {}
    def capture(action):
        result = original_step(action)
        captured.update(result)
        return result
    env.batch.step = capture
    _, _, _, _, extras = env.step(np.zeros((2, 8)))
    before_push, after_push = env._state(captured), env._state()
    linear = after_push["body_lin_vel_w"][:, 0]-before_push["body_lin_vel_w"][:, 0]
    angular = after_push["body_ang_vel_w"][:, 0]-before_push["body_ang_vel_w"][:, 0]
    assert np.linalg.norm(linear[0])+np.linalg.norm(angular[0]) > 0.
    assert np.all(np.abs(linear[0]) <= [.5, .5, .2])
    assert np.all(np.abs(angular[0]) <= [.52, .52, .78])
    np.testing.assert_allclose(linear[1], 0., atol=1.e-12)
    np.testing.assert_allclose(angular[1], 0., atol=1.e-12)
    assert extras["physical_counters"]["interval_pushes"] == 1
    # Smoothness records the pre-push reward state, matching reward-before-event.
    assert env.previous_vz[0] == before_push["body_lin_vel_w"][0, 0, 2]
    assert env.datas[0].time == pytest.approx(.02)


def test_done_reset_rearms_expired_push_before_postreward_event(make_env, native_library):
    env = make_env(native_library=native_library, training=True, dr_mode="common_mu")
    env.phases[:] = [env.frames-1, 0]
    env.push_timer[:] = [-.1, 2.]
    _, _, _, truncated, extras = env.step(np.zeros((2, 8)))
    assert truncated[0] and not truncated[1]
    assert .98 <= env.push_timer[0] <= 2.98
    assert env.push_timer[1] == pytest.approx(1.98)
    assert extras["physical_counters"]["interval_pushes"] == 0


def test_adaptive_sampler_exact_failure_histogram_ema_and_full_clip_component():
    sampler = AdaptivePhaseSampler(349, np.random.default_rng(7))
    assert sampler.bin_count == 35
    sampler.sample(2, old_phases=np.array([0, 200]), terminated=np.array([True, True]))
    expected = np.zeros(35)
    expected[[0, 200*35//349]] = 1.
    np.testing.assert_array_equal(sampler.current_failed, expected)
    sampler.update()
    np.testing.assert_allclose(sampler.failed, .001*expected)
    assert np.all(sampler.current_failed == 0.)
    phases = sampler.sample(10000)
    assert .19 < np.mean(phases == 0) < .23
    assert phases.min() == 0 and phases.max() < 348
    assert np.ptp(sampler.probabilities()) > 0.
    state = sampler.state_dict()
    expected = sampler.sample(10)
    sampler.load_state_dict(state)
    np.testing.assert_array_equal(sampler.sample(10), expected)


def test_invalid_actions_and_evaluation_randomization_fail_closed(make_env):
    with pytest.raises(ValueError):
        make_env(dr_mode="common_mu")
    env = make_env()
    for action in (np.zeros((1, 8)), np.full((2, 8), np.nan)):
        with pytest.raises(ValueError):
            env.step(action)
    env.close()
    with pytest.raises(RuntimeError):
        env.step(np.zeros((2, 8)))
