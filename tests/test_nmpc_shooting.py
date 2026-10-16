from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from nmpc_shooting import CEMConfig, ShootingNMPC


class NonlinearPredictor:
    def __init__(self):
        self.mu = 999.0  # The controller must not overwrite predictor attributes.
        self.references = []
        self.previous_values = []

    def transition(self, state, action, *, mu):
        return state + np.array([mu * (0.2 * action[0] + 0.05 * action[0] ** 3), 0.1 * action[0]])

    def stage_cost(self, state, action, next_state, reference_index, *, previous_action, previous_vz):
        self.references.append(reference_index)
        self.previous_values.append((previous_action.copy(), previous_vz))
        return (next_state[0] - 1.0) ** 2 + 0.001 * (action @ action)

    def vertical_velocity(self, state):
        return state[1]

    def terminal_cost(self, state, reference_index):
        return 2.0 * (state[0] - 1.0) ** 2


def controller(predictor, **kwargs):
    config = CEMConfig(horizon_steps=8, control_blocks=4, candidates=64,
                       iterations=3, elite_count=8, initial_std=0.6, **kwargs)
    return ShootingNMPC(predictor, config, action_low=[-1.0], action_high=[1.0], mu_scenarios=(1.0,))


def test_nonlinear_cem_improves_seed_is_bounded_and_reproducible():
    first = controller(NonlinearPredictor())
    second = controller(NonlinearPredictor())
    a, info = first.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.0)
    b, other = second.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.last_plan, second.last_plan)
    assert 0 < a[0] <= 1.0
    assert info["selected_model_cost"] < info["initial_model_cost"]
    assert info["selected_model_cost"] == other["selected_model_cost"]
    assert info["predicted_cost_nonincreasing"] is True
    history = info["best_cost_by_iteration"]
    assert all(after <= before for before, after in zip(history, history[1:]))
    assert info["finite_candidates"] == info["candidate_evaluations"]
    assert info["failed_candidates"] == 0
    assert info["real_time_verified"] is False
    assert np.max(np.abs(first.last_plan)) <= 1
    json.dumps(info, allow_nan=False)


class RecordingRollout:
    def __init__(self, costs=None):
        self.mu = 17.0
        self.calls = []
        self.costs = costs

    def rollout(self, state, controls, reference_index, *, mu, previous_action, previous_vz):
        self.calls.append((mu, controls.copy(), reference_index, previous_action.copy(), previous_vz))
        if self.costs is not None:
            return self.costs[mu]
        return float(np.sum((controls - 0.4) ** 2))


@pytest.mark.parametrize("reduction, expected", [("worst", 9.0), ("mean", 5.0)])
def test_scenarios_share_entire_plan_and_do_not_mutate_predictor_mu(reduction, expected):
    p = RecordingRollout(costs={0.2: 1.0, 0.7: 9.0})
    c = ShootingNMPC(p, CEMConfig(horizon_steps=3, control_blocks=1, candidates=4,
                                 iterations=2, elite_count=2, scenario_reduction=reduction),
                     action_low=[-1.0], action_high=[1.0], mu_scenarios=(0.2, 0.7))
    _, info = c.action(np.zeros(2), 7, previous_action=np.array([0.3]), previous_vz=0.4)
    assert p.mu == 17.0
    assert info["selected_model_cost"] == expected
    assert info["selected_scenario_costs"] == [1.0, 9.0]
    assert info["scenario_evaluations"] == 2 * info["candidate_evaluations"]
    for low, high in zip(p.calls[::2], p.calls[1::2]):
        assert (low[0], high[0]) == (0.2, 0.7)
        np.testing.assert_array_equal(low[1], high[1])
        assert low[2] == high[2] == 7
        np.testing.assert_array_equal(low[3], [0.3])
        assert low[4] == 0.4
    assert info["robustness_certificate"] is False


def test_warm_start_shifts_previous_plan_and_repeats_final_action():
    p = RecordingRollout()
    c = controller(p)
    c.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.0)
    original = c.last_plan
    p.calls.clear()
    c.action(np.zeros(2), 1, previous_action=original[0], previous_vz=0.0)
    expected = np.concatenate((original[1:], original[-1:]))
    np.testing.assert_array_equal(p.calls[1][1], expected)
    assert c.last_plan is not original
    c.reset()
    assert c.last_plan is None
    a, _ = c.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.0)
    np.testing.assert_array_equal(a, original[0])


def test_stage_reference_is_held_and_history_advances_inside_rollout():
    p = NonlinearPredictor()
    c = controller(p)
    plan = np.full((8, 1), 0.5)
    c._rollout(np.zeros(2), plan, 11, mu=1.0, previous_action=np.array([-0.25]), previous_vz=-0.8)
    assert p.references == list(range(11, 19))
    np.testing.assert_array_equal(p.previous_values[0][0], [-0.25])
    assert p.previous_values[0][1] == -0.8
    np.testing.assert_array_equal(p.previous_values[1][0], [0.5])
    assert p.previous_values[1][1] == pytest.approx(0.05)


@pytest.mark.parametrize("fallback, expected", [("previous", 1.0), ("neutral", 0.0)])
def test_all_failed_rollouts_have_explicit_bounded_unlearned_fallback(fallback, expected):
    p = RecordingRollout(costs={0.2: np.inf, 0.7: 2.0})
    c = ShootingNMPC(p, CEMConfig(horizon_steps=3, control_blocks=1, candidates=4,
                                 iterations=2, elite_count=2, fallback_action=fallback),
                     action_low=[-1.0], action_high=[1.0], mu_scenarios=(0.2, 0.7))
    a, info = c.action(np.zeros(2), 0, previous_action=np.array([2.0]), previous_vz=0.0)
    np.testing.assert_array_equal(a, [expected])
    assert info["optimization_failed"] is True
    assert info["fallback_used"] is True
    assert info["learned_policy_fallback"] is False
    assert info["fallback_kind"] == fallback
    assert info["failed_candidates"] == info["candidate_evaluations"]
    assert info["finite_candidates"] == 0
    assert info["failed_scenarios"] == info["candidate_evaluations"]
    assert c.last_plan is None
    json.dumps(info, allow_nan=False)


def test_optional_physics_initial_candidate_and_failed_candidate_accounting():
    class WithFeedback(RecordingRollout):
        def initial_candidate(self, state, reference_index, *, previous_action, previous_vz, horizon_steps):
            return np.full((horizon_steps, 1), 0.4)

        def rollout(self, state, controls, reference_index, **kwargs):
            if np.any(controls < 0):
                raise FloatingPointError("infeasible prediction")
            return super().rollout(state, controls, reference_index, **kwargs)

    c = controller(WithFeedback())
    a, info = c.action(np.zeros(2), 0, previous_action=np.array([-0.5]), previous_vz=0.0)
    np.testing.assert_array_equal(a, [0.4])
    assert info["initial_candidate_source"] == "physics_only_initial_candidate"
    assert info["selected_model_cost"] == 0
    assert info["failed_candidates"] > 0
    assert info["finite_candidates"] > 0
    assert info["failed_candidates"] + info["finite_candidates"] == info["candidate_evaluations"]


def test_reference_seed_is_bounded_and_caller_arrays_are_not_modified():
    p = RecordingRollout()
    c = controller(p)
    state, previous = np.zeros(2), np.array([0.0])
    seed = np.full((8, 1), 3.0)
    c.action(state, 0, previous_action=previous, previous_vz=0.0, seed_plan=seed)
    np.testing.assert_array_equal(p.calls[1][1], np.ones((8, 1)))
    np.testing.assert_array_equal(seed, np.full((8, 1), 3.0))
    np.testing.assert_array_equal(state, np.zeros(2))
    np.testing.assert_array_equal(previous, [0.0])


def test_bad_initial_feedback_does_not_disable_independent_optimization():
    class BadFeedback(RecordingRollout):
        def initial_candidate(self, *args, **kwargs):
            raise FloatingPointError("feedback rollout failed")

    c = controller(BadFeedback())
    _, info = c.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.0)
    assert info["initial_candidate_generation_failed"] is True
    assert info["optimization_failed"] is False
    assert info["fallback_used"] is False


def test_absolute_plan_cem_update_does_not_double_count_displacement():
    # The optimum is not at a bound. Moving-base noise updates tend to overshoot
    # this optimum; absolute elite controls converge and preserve the incumbent.
    p = RecordingRollout()
    c = controller(p)
    _, info = c.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.0)
    assert info["selected_model_cost"] < 0.05
    assert np.mean(c.last_plan) == pytest.approx(0.4, abs=0.08)


@pytest.mark.parametrize("kwargs", [
    {"horizon_steps": 0}, {"control_blocks": 16}, {"candidates": 1},
    {"elite_count": 65}, {"smoothing": 1.0}, {"reference_convention": "next_state"},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CEMConfig(**kwargs)
