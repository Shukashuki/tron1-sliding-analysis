"""Exact objective arithmetic parity; bounded open-loop CPU traces only."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nmpc_batch_objective import (
    BatchHeightSmoothTrackingObjective, NativeBatchTrackingPredictor,
    NATIVE_WARNING_DIFFERENCE, reference_indices, select_reference_sequence,
)
from nmpc_tracking_objective import BODY_NAMES, HeightSmoothTrackingObjective


def profile():
    return json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())


def random_states(rng, batch=4, horizon=5):
    shape = (batch, horizon)
    q = rng.normal(size=(*shape, 9, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    history = rng.normal(scale=1.4, size=(*shape, 3, 9, 3))
    history[..., -1, -2:, :] *= 8.
    history[0, 0, -1, -1, :] = 0.
    limits = np.broadcast_to([-1.4, 1.1], (*shape, 8, 2)).copy()
    limits[..., :, 0] += rng.uniform(-.2, .2, (*shape, 8))
    limits[..., :, 1] += rng.uniform(-.2, .2, (*shape, 8))
    return {"body_names": BODY_NAMES,
            "body_pos_w": rng.normal(scale=.3, size=(*shape, 9, 3)),
            "body_quat_w": q,
            "body_lin_vel_w": rng.normal(size=(*shape, 9, 3)),
            "body_ang_vel_w": rng.normal(scale=.5, size=(*shape, 9, 3)),
            "body_net_forces_w": history[..., -1, :, :].copy(),
            "contact_force_history_w": history,
            "joint_pos": rng.normal(scale=2., size=(*shape, 8)),
            "soft_joint_pos_limits": limits,
            "soft_joint_limits_source": "isaac_runtime"}


def sample(state, candidate, stage):
    result = {}
    for name, value in state.items():
        if isinstance(value, np.ndarray):
            result[name] = value[candidate, stage]
        else:
            result[name] = value
    return result


def compare_to_scalar(batch_objective, pre, post, reference, actions, previous, vz, reset):
    scalar = batch_objective.scalar
    held = batch_objective.hold_reference(reference, pre)
    actual = batch_objective.evaluate(post, held, actions, previous, vz, reset_samples=reset)
    for candidate in range(actions.shape[0]):
        for stage in range(actions.shape[1]):
            expected = scalar.evaluate(sample(post, candidate, stage),
                scalar.hold_reference(sample(reference, candidate, stage), sample(pre, candidate, stage)),
                actions[candidate, stage], previous[candidate, stage], vz[candidate, stage],
                reset_sample=bool(reset[candidate, stage]))
            assert set(actual["raw_terms"]) == set(expected["raw_terms"])
            for key in ("raw_terms", "weighted_terms", "diagnostics"):
                for name in expected[key]:
                    assert actual[key][name][candidate, stage] == pytest.approx(expected[key][name], abs=3e-13, rel=3e-13)
            for key in ("stage_cost", "total_reward", "base_com_vz"):
                assert actual[key][candidate, stage] == pytest.approx(expected[key], abs=3e-13, rel=3e-13)
            assert actual["surrogate_flags"] == expected["surrogate_flags"]
    np.testing.assert_allclose(sum(actual["term_costs"].values()), actual["stage_cost"], rtol=1e-13, atol=1e-13)
    return actual


@pytest.mark.parametrize("seed", [19, 42, 812])
@pytest.mark.parametrize("extra_leg_rate", [False, True])
def test_all_terms_match_scalar_with_random_nontrivial_states_and_memory(seed, extra_leg_rate):
    rng = np.random.default_rng(seed)
    cfg = profile()
    if extra_leg_rate:
        cfg["leg_target_rate_weight"] = -.0001
    objective = BatchHeightSmoothTrackingObjective(cfg)
    pre, post, reference = [random_states(rng) for _ in range(3)]
    # Exercise sign ambiguity in both absolute and relative quaternion terms.
    post["body_quat_w"][::2] *= -1.
    reference["body_quat_w"][:, ::2] *= -1.
    actions, previous = rng.uniform(-1., 1., (2, 4, 5, 8))
    vz = rng.normal(size=(4, 5))
    reset = rng.random((4, 5)) < .4
    before = copy.deepcopy((pre, post, reference, actions, previous, vz, reset))
    actual = compare_to_scalar(objective, pre, post, reference, actions, previous, vz, reset)
    assert len(actual["raw_terms"]) == (16 if extra_leg_rate else 15)
    assert np.any(actual["raw_terms"]["action_rate_l2"] > 0.)
    assert np.any(actual["raw_terms"]["joint_limit"] > 0.)
    assert np.any(actual["raw_terms"]["undesired_contacts"] > 0.)
    assert np.any(actual["raw_terms"]["drift_tilt"] > 0.)
    for old, current in zip(before, (pre, post, reference, actions, previous, vz, reset)):
        if isinstance(old, dict):
            for name in old:
                if isinstance(old[name], np.ndarray):
                    np.testing.assert_array_equal(current[name], old[name])
        else:
            np.testing.assert_array_equal(current, old)


def directed_state(beta, yaw=.8, speed=2., batch=2, horizon=3):
    shape = (batch, horizon)
    state = random_states(np.random.default_rng(10), batch, horizon)
    state["body_pos_w"][:] = np.arange(9)[:, None]*[.03, .01, -.05]
    state["body_quat_w"][:] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    state["body_lin_vel_w"][:] = [speed*np.cos(yaw+beta), speed*np.sin(yaw+beta), .7]
    state["body_ang_vel_w"][:] = [0., 0., .3]
    state["joint_pos"][:] = 0.
    state["contact_force_history_w"][:] = 0.
    state["contact_force_history_w"][..., -2:, 2] = 100.
    state["body_net_forces_w"][:] = state["contact_force_history_w"][..., -1, :, :]
    return state


def evaluate_direct(objective, post, reference, pre=None, *, reset=False, vz=.7):
    shape = post["body_pos_w"].shape[:2]
    pre = post if pre is None else pre
    return objective.evaluate(post, objective.hold_reference(reference, pre),
        np.zeros((*shape, 8)), np.zeros((*shape, 8)), np.full(shape, vz), reset_samples=reset)


def test_large_sideslip_current_heading_and_pi_wrap_are_not_zero_targets():
    objective = BatchHeightSmoothTrackingObjective(profile())
    reference = directed_state(.65, yaw=1.2)
    matched = evaluate_direct(objective, reference, reference)
    assert np.all(matched["raw_terms"]["drift_sideslip"] == pytest.approx(1.))
    assert np.all(matched["raw_terms"]["motion_global_anchor_ori"] == pytest.approx(1.))
    zero_slip = directed_state(0., yaw=1.2)
    assert np.max(evaluate_direct(objective, zero_slip, reference)["raw_terms"]["drift_sideslip"]) < 1e-10
    wrong_heading = directed_state(.65, yaw=0.)
    assert np.max(evaluate_direct(objective, wrong_heading, reference)["raw_terms"]["motion_global_anchor_ori"]) < .001
    reference = directed_state(np.pi-.025, yaw=0.)
    post = directed_state(-np.pi+.025, yaw=0.)
    wrapped = evaluate_direct(objective, post, reference)
    np.testing.assert_allclose(wrapped["diagnostics"]["beta_error_rad"], .05, atol=1e-14)
    np.testing.assert_allclose(wrapped["raw_terms"]["drift_sideslip"], np.exp(-(.05/.12)**2))
    slow = directed_state(.65, speed=.25)
    assert not np.any(evaluate_direct(objective, slow, reference)["raw_terms"]["drift_sideslip"])
    slow_reference = directed_state(.65, speed=.3)
    slow_reference["body_lin_vel_w"][:] = [.3, 0., .7]
    assert not np.any(evaluate_direct(objective, post, slow_reference)["raw_terms"]["drift_sideslip"])


def test_z_translation_and_target_vz_do_not_change_cost_and_reset_skips_only_acceleration():
    objective = BatchHeightSmoothTrackingObjective(profile())
    reference, post = directed_state(.4), directed_state(.4)
    expected = evaluate_direct(objective, post, reference)
    post["body_pos_w"][..., 2] += 30.
    post["body_lin_vel_w"][..., 2] = 3.2
    reference["body_lin_vel_w"][..., 2] = -99.
    actual = evaluate_direct(objective, post, reference, vz=3.2)
    for name in actual["raw_terms"]:
        np.testing.assert_allclose(actual["raw_terms"][name], expected["raw_terms"][name])
    np.testing.assert_allclose(actual["stage_cost"], expected["stage_cost"])
    changed = evaluate_direct(objective, post, reference, vz=3.1)
    np.testing.assert_allclose(changed["raw_terms"]["height_vertical_acceleration"], 1.-np.exp(-1.))
    reset = evaluate_direct(objective, post, reference, reset=True, vz=3.1)
    np.testing.assert_array_equal(reset["raw_terms"]["height_vertical_acceleration"], 0.)
    for name in changed["raw_terms"]:
        if name != "height_vertical_acceleration":
            np.testing.assert_array_equal(reset["raw_terms"][name], changed["raw_terms"][name])


def test_pre_action_relative_targets_are_frozen_and_copied():
    objective = BatchHeightSmoothTrackingObjective(profile())
    pre = directed_state(.4, yaw=0.)
    reference = copy.deepcopy(pre)
    held = objective.hold_reference(reference, pre)
    post = copy.deepcopy(pre)
    post["body_pos_w"][..., 0] += .5
    shape = post["body_pos_w"].shape[:2]
    actual = objective.evaluate(post, held, np.zeros((*shape, 8)), np.zeros((*shape, 8)), np.full(shape, .7))
    reanchored = evaluate_direct(objective, post, reference)
    np.testing.assert_allclose(actual["raw_terms"]["motion_body_pos"], np.exp(-.5**2/.3**2))
    np.testing.assert_allclose(reanchored["raw_terms"]["motion_body_pos"], 1.)
    reference["body_pos_w"][:] = 999.
    assert np.max(held.body_pos_w) < 1.


def test_wheel_spin_phase_and_wheel_soft_limit_violations_are_masked():
    objective = BatchHeightSmoothTrackingObjective(profile())
    reference = directed_state(.4)
    post = copy.deepcopy(reference)
    post["body_quat_w"][..., -2:, :] = [0., 0., 1., 0.]
    post["body_ang_vel_w"][..., -2:, :] = [0., 100., 0.]
    post["joint_pos"][..., -2:] = [1.e8, -1.e8]
    actual = evaluate_direct(objective, post, reference)
    np.testing.assert_allclose(actual["raw_terms"]["motion_body_ori"], 1.)
    np.testing.assert_allclose(actual["raw_terms"]["motion_body_ang_vel"], 1.)
    np.testing.assert_array_equal(actual["raw_terms"]["joint_limit"], 0.)
    post["joint_pos"][..., 5] = post["soft_joint_pos_limits"][..., 5, 0]-.4
    np.testing.assert_allclose(evaluate_direct(objective, post, reference)["raw_terms"]["joint_limit"], .4)


def test_force_thresholds_and_last_three_history_exactly_match_scalar():
    objective = BatchHeightSmoothTrackingObjective(profile())
    post = directed_state(.4)
    # Exactly1N is not an undesired contact; a past sample just over1N is.
    post["contact_force_history_w"][..., 0, 1, 0] = 1.
    post["contact_force_history_w"][..., 0, 2, 0] = 1.+1.e-8
    post["body_net_forces_w"][..., -1, :] = [0., 0., 5.]
    actual = evaluate_direct(objective, post, post)
    np.testing.assert_array_equal(actual["raw_terms"]["undesired_contacts"], 1.)
    np.testing.assert_array_equal(actual["raw_terms"]["drift_grounded"], 0.)
    shape = post["body_pos_w"].shape[:2]
    compare_to_scalar(objective, post, post, post, np.zeros((*shape, 8)),
                      np.zeros((*shape, 8)), np.full(shape, .7), np.zeros(shape, bool))


def test_reference_selection_clamps_without_advancing_reward_phase_and_broadcasts():
    objective = BatchHeightSmoothTrackingObjective(profile())
    states = random_states(np.random.default_rng(17), batch=2, horizon=3)
    full = {"body_names": BODY_NAMES,
            **{name: states[name][0].copy() for name in
               ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}}
    selected, indices = select_reference_sequence(full, 2, 3)
    np.testing.assert_array_equal(indices, [2, 2, 2])
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_array_equal(selected[name], np.tile(full[name][2:3], (3, 1, 1)))
    held = objective.hold_reference(selected, states)
    assert held.body_pos_w.shape == (2, 3, 9, 3)
    np.testing.assert_array_equal(reference_indices(10**30, 3, 3), [2, 2, 2])
    for bad in (-1, True, .5):
        with pytest.raises(ValueError, match="start_index"):
            reference_indices(bad, 3, 3)
    with pytest.raises(ValueError, match="horizon_steps"):
        reference_indices(0, 0, 3)
    full["body_names"] = tuple(reversed(BODY_NAMES))
    with pytest.raises(ValueError, match="body order"):
        select_reference_sequence(full, 0, 3)


@pytest.mark.parametrize("problem", ["nan", "quaternion", "action", "previous_action", "limits", "reset", "previous_vz", "body_order"])
def test_invalid_input_fails_closed(problem):
    objective = BatchHeightSmoothTrackingObjective(profile())
    state = directed_state(.4)
    held = objective.hold_reference(state, state)
    shape = state["body_pos_w"].shape[:2]
    action, previous = np.zeros((*shape, 8)), np.zeros((*shape, 8))
    vz, reset = np.full(shape, .7), False
    if problem == "nan":
        state["body_lin_vel_w"][0, 0, 1, 1] = np.nan
    elif problem == "quaternion":
        state["body_quat_w"][0, 0, 0] = 0.
    elif problem == "action":
        action[0, 0, 0] = 1.1
    elif problem == "previous_action":
        previous[0, 0, 0] = -1.1
    elif problem == "limits":
        state["soft_joint_pos_limits"][0, 0, 0] = [1., -1.]
    elif problem == "reset":
        reset = np.zeros(shape)
    elif problem == "previous_vz":
        vz = None
    else:
        state["body_names"] = tuple(reversed(BODY_NAMES))
    with pytest.raises((ValueError, FloatingPointError)):
        objective.evaluate(state, held, action, previous, vz, reset_samples=reset)


def test_optional_surrogates_and_all_reset_missing_previous_vz_match_scalar_contract():
    objective = BatchHeightSmoothTrackingObjective(profile())
    post = directed_state(.4)
    del post["contact_force_history_w"]
    post["soft_joint_limits_source"] = "mujoco_ranges_times_0.95_surrogate"
    shape = post["body_pos_w"].shape[:2]
    actual = objective.evaluate(post, objective.hold_reference(post, post), np.zeros((*shape, 8)),
        np.zeros((*shape, 8)), None, reset_samples=True)
    assert actual["surrogate_flags"] == ["instantaneous_undesired_contacts_without_sensor_history",
                                           "mujoco_or_unspecified_soft_joint_limits"]
    np.testing.assert_array_equal(actual["raw_terms"]["height_vertical_acceleration"], 0.)
    assert objective.contract()["absolute_height_target"] is False
    assert objective.contract()["production_cem_wired"] is False


def fake_native_trace(pre, post, joint_indices):
    shape = post["body_pos_w"].shape[:2]
    trace = {"pre_"+name: pre[name].copy() for name in
             ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_net_forces_w"):
        trace[name] = np.repeat(post[name][..., None, :, :], 4, axis=-3)
    trace["qpos"] = np.zeros((*shape, 4, 15))
    trace["qpos"][..., joint_indices] = post["joint_pos"][..., None, :]
    trace["interval_time_s"] = np.broadcast_to([.005, .01, .015, .02], (*shape, 4)).copy()
    trace["contact_force_history_w"] = post["contact_force_history_w"].copy()
    return trace


def test_native_output_mapping_preserves_physical_forces_history_and_sequential_memory():
    rng = np.random.default_rng(49)
    objective = BatchHeightSmoothTrackingObjective(profile())
    pre, post, reference = [random_states(rng, 2, 4) for _ in range(3)]
    actions = rng.uniform(-.8, .8, (2, 4, 8))
    initial_action = rng.uniform(-.6, .6, (2, 8))
    initial_vz = np.array([.1, -.2])
    qids = np.array([7, 11, 8, 12, 9, 13, 10, 14])
    native = fake_native_trace(pre, post, qids)
    actual = objective.evaluate_native(native, reference, actions, initial_action, initial_vz,
        joint_qpos_indices=qids, soft_joint_pos_limits=post["soft_joint_pos_limits"],
        soft_joint_limits_source="isaac_runtime", initial_reset_samples=np.array([True, False]))
    previous = np.concatenate((initial_action[:, None], actions[:, :-1]), axis=1)
    previous_vz = np.concatenate((initial_vz[:, None], post["body_lin_vel_w"][:, :-1, 0, 2]), axis=1)
    reset = np.zeros((2, 4), bool); reset[0, 0] = True
    expected = compare_to_scalar(objective, pre, post, reference, actions, previous, previous_vz, reset)
    for name in expected["raw_terms"]:
        np.testing.assert_allclose(actual["raw_terms"][name], expected["raw_terms"][name], atol=3e-13)
    np.testing.assert_allclose(actual["candidate_costs"], expected["stage_cost"].sum(axis=1))
    assert actual["surrogate_flags"] == [NATIVE_WARNING_DIFFERENCE]
    assert actual["raw_terms"]["height_vertical_acceleration"][0, 0] == 0.
    assert np.all(actual["raw_terms"]["height_vertical_acceleration"][:, 1:] > 0.)
    # Do not replace final stored forces with a later zero-control solve.
    native["body_net_forces_w"][..., -1, :, :] = 0.
    with pytest.raises(ValueError, match="physical forces"):
        objective.evaluate_native(native, reference, actions, initial_action, initial_vz,
            joint_qpos_indices=qids, soft_joint_pos_limits=post["soft_joint_pos_limits"],
            soft_joint_limits_source="isaac_runtime")


@pytest.fixture(scope="module")
def licensed_native_setup(tmp_path_factory):
    mujoco = pytest.importorskip("mujoco")
    if mujoco.__version__ != "3.8.1" or sys.platform != "linux":
        pytest.skip("Optional native trace requires Linux MuJoCo3.8.1")
    raw = ROOT / "outputs/car_failure_sideslip_20261015/raw"
    model_path, contract_path = raw / "mujoco_model.xml", raw / "mu0p180/isaac/policy_contract.json"
    motion = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
    if not all(path.is_file() for path in (model_path, contract_path, motion)):
        pytest.skip("Licensed TRON1 model and locked motion are unavailable")
    from nmpc_native_batch import build_native_library
    from run_nmpc_tracking import MujocoTrackingPredictor, build_provider
    settings = json.loads((ROOT / "config/nmpc_height_smooth.json").read_text())
    provider = build_provider(model_path, motion, json.loads(contract_path.read_text()), settings, (.16, .7))
    predictor = MujocoTrackingPredictor(provider, model_path, profile(), (.16, .7))
    library = build_native_library(tmp_path_factory.mktemp("batch_objective_native"))
    return predictor, library


def test_licensed_native_adapter_matches_original_scalar_rollout_and_every_reward_term(licensed_native_setup):
    predictor, library = licensed_native_setup
    rng = np.random.default_rng(62)
    plans = rng.uniform(-.15, .15, (2, 3, 8))
    previous = np.linspace(-.12, .12, 8)
    history = rng.normal(scale=.5, size=(3, 9, 3))
    initial = predictor.layout.pack(predictor.provider.targets[0, :15],
        predictor.provider.targets[0, 15:], history, True)
    initial_vz = predictor.vertical_velocity(initial)
    with NativeBatchTrackingPredictor(predictor, library) as adapter:
        assert adapter.metadata["production_cem_wired"] is False
        assert adapter.metadata["whole_batch_warning_failure"] is True
        for mu in (.16, .7):
            actual = adapter.evaluate_batch(initial, plans, 0, mu=mu,
                previous_action=previous, previous_vz=initial_vz)
            for candidate, plan in enumerate(plans):
                state, old_action, old_vz = initial.copy(), previous.copy(), initial_vz
                expected_sum = 0.
                for stage, action in enumerate(plan):
                    before = state.copy()
                    state = predictor.transition(before, action, mu=mu)
                    expected = predictor.objective.evaluate(predictor.measure(state),
                        predictor.objective.hold_reference(predictor.reference_frame(stage), predictor.measure(before)),
                        action, old_action, old_vz, reset_sample=predictor.layout.unpack(before)[-1])
                    for key in ("raw_terms", "weighted_terms", "diagnostics"):
                        for name in expected[key]:
                            assert actual[key][name][candidate, stage] == pytest.approx(expected[key][name], abs=2e-9, rel=2e-9)
                    expected_sum += expected["stage_cost"]
                    old_action, old_vz = action.copy(), predictor.vertical_velocity(state)
                assert actual["candidate_costs"][candidate] == pytest.approx(expected_sum, abs=2e-9, rel=2e-9)
            assert actual["surrogate_flags"] == ["mujoco_or_unspecified_soft_joint_limits", NATIVE_WARNING_DIFFERENCE]
            np.testing.assert_array_equal(actual["reference_indices"], [0, 1, 2])
        with pytest.raises(ValueError, match="declared"):
            adapter.rollout_batch(initial, plans, 0, mu=.12, previous_action=previous, previous_vz=initial_vz)
        # One short repeated candidate also exercises terminal reference clamp;
        # this is an open-loop arithmetic test, not a closed-loop evaluation.
        last = predictor.reference.frames-1
        result = adapter.evaluate_batch(initial, plans[:1, :2], last, mu=.7,
            previous_action=previous, previous_vz=initial_vz)
        np.testing.assert_array_equal(result["reference_indices"], [last, last])
