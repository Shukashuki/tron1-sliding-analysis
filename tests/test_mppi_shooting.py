"""Asset-free algorithm, density, action-clock and failure tests for MPPI."""
from dataclasses import FrozenInstanceError
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from mppi_shooting import MPPIConfig, MPPIController, _mppi_weights


class Rollout:
    def __init__(self, cost=None):
        self.calls = []
        self.mu = 999.
        self.cost = cost if cost is not None else lambda plan: np.sum((plan - .4)**2)

    def rollout(self, state, plan, index, *, mu, previous_action, previous_vz):
        self.calls.append((state.copy(), plan.copy(), index, mu, previous_action.copy(), previous_vz))
        return float(self.cost(plan))


class BatchRollout(Rollout):
    def rollout_batch(self, state, plans, index, *, mu, previous_action, previous_vz):
        self.calls.append((state.copy(), plans.copy(), index, mu, previous_action.copy(), previous_vz))
        return np.asarray([self.cost(plan) for plan in plans], dtype=float)


class BiasBatch(BatchRollout):
    def initial_candidate(self, state, index, *, previous_action, previous_vz, horizon_steps):
        return np.full((horizon_steps, len(previous_action)), .2)


def controller(predictor=None, **config):
    defaults = dict(horizon_steps=4, control_blocks=2, samples=7, iterations=2, noise_std=.4)
    defaults.update(config)
    return MPPIController(predictor or Rollout(), MPPIConfig(**defaults),
                          action_low=[-1.], action_high=[1.], mu_scenarios=(.7,))


def act(c, index=0, previous=None, **kwargs):
    return c.action(np.zeros(2), index, previous_action=np.zeros(1) if previous is None else previous,
                    previous_vz=.2, **kwargs)


class FixedNoise:
    def __init__(self, *draws):
        self.draws, self.calls = list(draws), 0

    def normal(self, *, size):
        value = np.asarray(self.draws[self.calls], dtype=float)
        self.calls += 1
        assert value.shape == size
        return value.copy()


def test_default_frozen_preregistered_config_and_no_mutable_noise_alias():
    config = MPPIConfig()
    assert (config.horizon_steps, config.control_blocks, config.samples, config.iterations) == (30, 10, 63, 3)
    assert config.temperature == 1. and config.seed == 42
    assert config.noise_std == (.04,)*6 + (.35,)*2
    with pytest.raises(FrozenInstanceError):
        config.temperature = .1
    noise = [.2, .3]
    copied = MPPIConfig(noise_std=noise)
    noise[0] = 999.
    assert copied.noise_std == (.2, .3)


@pytest.mark.parametrize("kwargs", [
    {"horizon_steps": 0}, {"horizon_steps": True}, {"control_blocks": 31},
    {"control_blocks": 1.5}, {"samples": 0}, {"samples": False}, {"iterations": -1},
    {"temperature": 0}, {"temperature": -1}, {"temperature": np.inf}, {"temperature": np.nan},
    {"temperature": True}, {"seed": -1}, {"seed": False}, {"seed": 1.5},
    {"noise_std": 0}, {"noise_std": -1}, {"noise_std": np.nan}, {"noise_std": True},
    {"noise_std": []}, {"noise_std": [[1.]]}, {"noise_std": 1e-300}, {"noise_std": 1e300},
    {"reference_convention": "post_action"}, {"fallback_action": "policy"},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        MPPIConfig(**kwargs)


@pytest.mark.parametrize("mus", [(), (.3, .7), (np.inf,), (-.1,), (True,)])
def test_only_singleton_fixed_nonnegative_mu(mus):
    with pytest.raises(ValueError):
        MPPIController(Rollout(), MPPIConfig(noise_std=.2), action_low=[-1.], action_high=[1.], mu_scenarios=mus)


@pytest.mark.parametrize("low,high", [([], []), ([0.], [0.]), ([1.], [-1.]), ([-np.inf], [1.]), ([-1.], [1., 2.])])
def test_bad_action_bounds(low, high):
    with pytest.raises(ValueError):
        MPPIController(Rollout(), MPPIConfig(noise_std=.2), action_low=low, action_high=high)


def test_noise_dimension_must_match_actuators():
    with pytest.raises(ValueError, match="one value per action"):
        MPPIController(Rollout(), MPPIConfig(), action_low=[-1.], action_high=[1.])


def test_exact_gaussian_density_ratio_and_block_not_held_horizon_multiplicity():
    mean = np.array([[.3, -.2], [.5, .1]])
    std = np.array([.4, .7])
    noise = np.array([[[.2, .4], [-.3, .1]], [[-.5, .3], [.6, -.2]], [[.1, -.8], [.3, .5]]])
    costs = np.array([2., 4., 1.])
    temperature = .6
    weights, stats = _mppi_weights(costs, mean, noise, std, temperature)
    z = mean[None] + noise
    logp = -.5 * np.sum((z / std)**2, axis=(1, 2))
    logq = -.5 * np.sum((noise / std)**2, axis=(1, 2))
    expected_log = -costs / temperature + logp - logq
    expected = np.exp(expected_log - expected_log.max())
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, atol=1e-15, rtol=1e-14)
    wrong = -costs / temperature - 3 * np.sum(mean * noise / std**2, axis=(1, 2))
    wrong = np.exp(wrong - wrong.max()); wrong /= wrong.sum()
    assert not np.allclose(weights, wrong)
    assert 1. <= stats["ess"] <= 3.


def test_clipped_decode_still_uses_raw_latent_density_and_noise():
    c = controller(horizon_steps=3, control_blocks=1, samples=2, iterations=1, noise_std=1.)
    baseline = np.full((3, 1), .9)
    mean, noise = np.array([[.4]]), np.array([[[2.]], [[-.1]]])
    plans, clip_fraction = c._decode(baseline, mean[None] + noise)
    np.testing.assert_array_equal(plans, np.ones((2, 3, 1)))
    assert clip_fraction == 1.
    weights, _ = _mppi_weights(np.ones(2), mean, noise, np.ones(1), 1.)
    expected = np.exp(-np.array([.8, -.04])); expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    assert weights[0] != weights[1]
    np.testing.assert_array_equal(noise, [[[2.]], [[-.1]]])


@pytest.mark.parametrize("temperature", [1e-308, 1e-100, 1., 1e100, 1e308])
def test_log_weight_stability_extreme_temperature(temperature):
    weights, stats = _mppi_weights(np.array([-1e100, 1e100, np.inf]), np.zeros((1, 1)),
                                  np.array([[[1.]], [[-1.]], [[.5]]]), np.ones(1), temperature)
    assert np.isfinite(weights).all() and np.isclose(weights.sum(), 1.)
    assert weights[2] == 0. and weights[0] >= weights[1]
    json.dumps(stats, allow_nan=False)


def test_invalid_cost_candidates_receive_zero_weight():
    weights, stats = _mppi_weights(np.array([np.nan, 0., np.inf, -np.inf]), np.zeros((1, 1)),
                                  np.ones((4, 1, 1)), np.ones(1), 1.)
    np.testing.assert_array_equal(weights, [0., 1., 0., 0.])
    assert stats["finite_population_candidates"] == 1
    assert _mppi_weights(np.array([np.inf]), np.zeros((1, 1)), np.zeros((1, 1, 1)), np.ones(1), 1.) == (None, None)


def test_final_weighted_mean_is_not_argmin_sample_or_initial_plan():
    predictor = BatchRollout(cost=lambda plan: np.sum(plan**2))
    c = controller(predictor, horizon_steps=1, control_blocks=1, samples=2, iterations=1,
                   noise_std=1., temperature=1e100)
    c.rng = FixedNoise([[[.2]], [[.8]]])
    action, info = act(c)
    np.testing.assert_allclose(action, [.5])
    assert info["initial_model_cost"] == 0.
    assert info["selected_model_cost"] == pytest.approx(.25)
    assert info["predicted_cost_nonincreasing"] is False
    assert action[0] != .2 and action[0] != .8
    population = predictor.calls[1][1]
    np.testing.assert_array_equal(population, [[[.2]], [[.8]]])
    assert len(predictor.calls) == 3  # Initial, all-IID population, mean check.


def test_iterative_mean_update_uses_importance_correction_not_plain_softmin():
    predictor = BatchRollout(cost=lambda plan: 0.)
    c = controller(predictor, horizon_steps=1, control_blocks=1, samples=2, iterations=2, noise_std=1.)
    c.rng = FixedNoise([[[.2]], [[.6]]], [[[.1]], [[.7]]])
    action, info = act(c)
    first_mean = .4
    second_weights = np.exp(-first_mean*np.array([.1, .7])); second_weights /= second_weights.sum()
    expected = first_mean + second_weights @ np.array([.1, .7])
    np.testing.assert_allclose(action, [expected])
    assert not np.isclose(action[0], .8)
    assert info["mppi_iterations"][0]["ess"] == pytest.approx(2.)
    assert info["mppi_iterations"][1]["ess"] < 2.


def test_every_population_entry_is_gaussian_not_forced_incumbent():
    p = BatchRollout(cost=lambda plan: 0.)
    c = controller(p, horizon_steps=2, control_blocks=1, samples=3, iterations=1, noise_std=1.)
    c.rng = FixedNoise([[[.1]], [[.2]], [[.3]]])
    act(c)
    np.testing.assert_array_equal(p.calls[1][1][:, 0, 0], [.1, .2, .3])


def test_scalar_and_batch_paths_produce_same_actions_costs_and_candidate_counts():
    scalar, batch = controller(Rollout()), controller(BatchRollout())
    previous = np.zeros(1)
    for index in range(3):
        a, info = act(scalar, index, previous)
        b, other = act(batch, index, previous)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(scalar.last_plan, batch.last_plan)
        for name in ("selected_model_cost", "initial_model_cost", "candidate_evaluations", "finite_candidates",
                     "failed_candidates", "scenario_evaluations", "failed_scenarios", "mppi_iterations"):
            assert info[name] == other[name], name
        assert info["batch_prediction_calls"] == 0
        assert other["batch_prediction_calls"] == (5 if index == 0 else 6)
        assert scalar.predictor.mu == batch.predictor.mu == 999.
        json.dumps(info, allow_nan=False)
        previous = a


def test_default_349_step_budget_is_68054_candidates_with_bias_and_full_warm_plan():
    p = BiasBatch(cost=lambda plan: 0.)
    c = MPPIController(p, MPPIConfig(), action_low=-np.ones(8), action_high=np.ones(8))
    counts, previous = 0, np.zeros(8)
    for index in range(349):
        action, info = c.action(np.zeros(2), index, previous_action=previous, previous_vz=0.)
        assert info["candidate_evaluations"] == (194 if index == 0 else 195)
        assert info["scenario_evaluations"] == info["candidate_evaluations"]
        counts += info["candidate_evaluations"]
        previous = action
        p.calls.clear()  # Test must not retain a whole episode of raw plans.
    assert counts == 68054
    assert counts * 30 * 4 == 8166480


def test_warm_full_plan_shift_uses_selected_mean_and_repeats_end():
    p = BatchRollout()
    c = controller(p)
    action, _ = act(c)
    full = c.last_plan
    p.calls.clear()
    _, info = act(c, 2, action)
    shifted = np.concatenate((full[2:], np.repeat(full[-1:], 2, axis=0)))
    np.testing.assert_array_equal(p.calls[1][1][0], shifted)
    assert info["warm_start_used"] is True
    assert p.calls[1][1].shape == (1, 4, 1)
    public_copy = c.last_plan
    public_copy[:] = 999.
    assert np.max(np.abs(c.last_plan)) <= 1.


def test_seed_validation_and_invalid_inputs_do_not_mutate_rng_or_warm_state():
    c = controller()
    action, _ = act(c)
    state = copy.deepcopy(c.rng.bit_generator.state)
    plan = c.last_plan
    invalids = [dict(state=np.array([np.nan])), dict(reference_index=True), dict(reference_index=-1),
                dict(previous_action=np.array([np.nan])), dict(previous_action=np.zeros(2)),
                dict(previous_vz=np.inf), dict(seed_plan=np.zeros((1, 1)))]
    for bad in invalids:
        kwargs = dict(state=np.zeros(2), reference_index=1, previous_action=action, previous_vz=0.)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            c.action(**kwargs)
        assert c.rng.bit_generator.state == state
        np.testing.assert_array_equal(c.last_plan, plan)


def test_reset_reproduces_rng_and_initial_lifecycle_without_predictor_mutation():
    c = controller()
    first, info = act(c)
    plan = c.last_plan
    act(c, 1, first)
    c.reset()
    assert c.last_plan is None
    second, other = act(c)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(plan, c.last_plan)
    assert info["mppi_iterations"] == other["mppi_iterations"]


def test_same_or_backwards_reference_index_does_not_shift_old_warm_plan():
    c = controller(BatchRollout())
    first, _ = act(c, 5)
    for index in (5, 2):
        _, info = act(c, index, first)
        assert not info["warm_start_used"]
        assert [item["source"] for item in info["initial_candidate_evaluations"]] == ["bounded_previous"]


@pytest.mark.parametrize("fallback", ["previous", "neutral"])
@pytest.mark.parametrize("batch", [False, True])
def test_all_failed_population_falls_back_and_clears_warm_plan(fallback, batch):
    cls = BatchRollout if batch else Rollout
    p = cls()
    c = controller(p, fallback_action=fallback)
    act(c)
    p.cost = lambda plan: np.inf
    action, info = act(c, 1, np.array([.3]))
    np.testing.assert_array_equal(action, [.3] if fallback == "previous" else [0.])
    assert c.last_plan is None and info["optimization_failed"]
    assert info["optimization_failure_reason"] == "all_population_predictions_failed"
    assert info["finite_candidates"] == 0
    assert info["failed_candidates"] == info["candidate_evaluations"] == 2 + 7
    assert info["prediction_failure_count"] == 0  # Returned inf, no exception.
    json.dumps(info, allow_nan=False)


def test_failed_mean_plan_check_does_not_apply_best_population_or_initial_plan():
    class FailingMean(BatchRollout):
        def rollout_batch(self, *args, **kwargs):
            result = super().rollout_batch(*args, **kwargs)
            return np.full_like(result, np.inf) if len(self.calls) == 3 else result
    p = FailingMean()
    c = controller(p, iterations=1)
    action, info = act(c, previous=np.array([.1]))
    np.testing.assert_array_equal(action, [.1])
    assert info["optimization_failure_reason"] == "updated_mean_prediction_failed"
    assert info["selected_model_cost"] is None and c.last_plan is None
    assert info["finite_candidates"] == 8 and info["failed_candidates"] == 1


@pytest.mark.parametrize("error_type", [RuntimeError, FloatingPointError, OverflowError])
def test_caught_runtime_prediction_errors_logged_without_retry(error_type):
    class Broken(BatchRollout):
        def rollout_batch(self, *args, **kwargs):
            self.calls.append(None)
            raise error_type("bad " + "界"*2000)
    p = Broken()
    c = controller(p)
    _, info = act(c)
    assert len(p.calls) == 2  # One initial + one population; no retry.
    assert info["prediction_failure_count"] == 2
    assert info["failed_batch_prediction_calls"] == info["batch_prediction_calls"] == 2
    record = info["prediction_failures"][0]
    assert record["exception_type"] == error_type.__name__
    assert record["message_truncated"] and len(record["message"].encode()) <= 2048
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("bad", ["shape", "value_error", "initial_shape"])
def test_programming_value_errors_propagate_without_numerical_fallback(bad):
    class Broken(BatchRollout):
        def rollout_batch(self, *args, **kwargs):
            if bad == "shape":
                return np.zeros((1, 1))
            if bad == "value_error":
                raise ValueError("programming error")
            return super().rollout_batch(*args, **kwargs)
        def initial_candidate(self, *args, **kwargs):
            return np.zeros((2, 9)) if bad == "initial_shape" else np.zeros((4, 1))
    with pytest.raises(ValueError):
        act(controller(Broken()))


def test_scalar_transition_stage_cost_reference_and_memory_conventions():
    class Transition:
        def __init__(self):
            self.stages = []
            self.terminals = []
        def transition(self, state, action, *, mu):
            return state + np.array([mu * action[0], .1])
        def stage_cost(self, state, action, next_state, reference_index, *, previous_action, previous_vz):
            self.stages.append((state.copy(), action.copy(), next_state.copy(), reference_index,
                                previous_action.copy(), previous_vz))
            return (next_state[0] - 1.)**2
        def vertical_velocity(self, state):
            return state[1]
        def terminal_cost(self, state, reference_index):
            self.terminals.append(reference_index)
            return state[0]**2
    p = Transition()
    c = controller(p, horizon_steps=2, control_blocks=1, samples=1, iterations=1)
    _, info = act(c, 7, np.array([.2]))
    assert info["candidate_evaluations"] == 3
    assert [item[3] for item in p.stages] == [7, 8]*3
    assert p.terminals == [9]*3
    for first, second in zip(p.stages[::2], p.stages[1::2]):
        np.testing.assert_array_equal(first[4], [.2])
        assert first[5] == .2
        np.testing.assert_array_equal(second[4], first[1])
        assert second[5] == .1


def test_contracts_disclose_implicit_prior_clamped_dynamics_and_no_best_sample():
    c = controller()
    algorithm, params = c.algorithm_contract(), c.parameterization_contract()
    assert algorithm["task_rollout_score_modified"] is False
    assert algorithm["covariance_fitting"] is algorithm["elite_selection"] is False
    assert algorithm["deterministic_initial_plans_in_weighted_population"] is False
    assert "KL" in algorithm["implicit_regularization"]
    assert params["latent_clip"] is params["noise_clip"] is False
    assert params["density_block_multiplicity"] == 1
    json.dumps(algorithm, allow_nan=False)
    json.dumps(params, allow_nan=False)
