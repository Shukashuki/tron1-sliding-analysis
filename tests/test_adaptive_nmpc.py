"""Synthetic arrays only: no simulator import, plant step, native worker, or asset."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from adaptive_contact_model import AdaptiveContactConfig
from adaptive_nmpc import AdaptiveShootingNMPC
from nmpc_shooting import CEMConfig, ShootingNMPC


def context(wheel=True, impact=False):
    return {"wheel_ground_contact_observed": wheel,
            "nonwheel_ground_contact_observed": impact, "duration_s": .02}


class FakePlanningPredictor:
    def __init__(self):
        self.calls = []

    def rollout(self, state, controls, start_index, *, mu, previous_action, previous_vz):
        self.calls.append((mu, state.copy(), controls.copy(), start_index))
        return float(np.sum((controls - (.2 + mu * .1)) ** 2))


class FakeIdentification:
    def __init__(self):
        self.feature_calls = []
        self.prediction_calls = []
        self.error_mu = None
        self.nonfinite_mu = None

    def features(self, packed):
        self.feature_calls.append(packed.copy())
        return packed[:3].copy()

    def transition_features(self, packed, action, *, mu):
        self.prediction_calls.append((mu, packed.copy(), action.copy()))
        if mu == self.error_mu:
            raise RuntimeError("synthetic identification failure")
        result = packed[:3] + mu * np.array([1., 0., 1.])
        if mu == self.nonfinite_mu:
            result[0] = np.nan
        return result


def make(*, hypotheses=(.3, .7, 1.6), initial=.7):
    config = AdaptiveContactConfig(mu_hypotheses=hypotheses, initial_mu=initial,
        minimum_evidence=1, minimum_spread=.01, maximum_fit_rms=100.)
    cem = CEMConfig(horizon_steps=4, control_blocks=2, candidates=8,
                    iterations=2, elite_count=2, seed=42, initial_std=.2)
    planning = FakePlanningPredictor()
    identification = FakeIdentification()
    controller = AdaptiveShootingNMPC(planning, cem, config, identification,
                                      action_low=[-1.], action_high=[1.])
    return controller, planning, identification


def applied(action):
    return action.astype(np.float32).astype(float)


def initialize(controller):
    return controller.action(np.zeros(4), 0, previous_action=np.zeros(1), previous_vz=0.)


def snapshot(controller):
    return {"rng": copy.deepcopy(controller.rng.bit_generator.state),
            "plan": controller.last_plan.copy(),
            "selector": copy.deepcopy(controller.selector.__dict__),
            "mu": controller.mu_scenarios,
            "previous": controller._adaptive_previous_state.copy(),
            "index": controller._adaptive_last_reference_index}


def assert_unchanged(controller, before):
    assert controller.rng.bit_generator.state == before["rng"]
    np.testing.assert_array_equal(controller.last_plan, before["plan"])
    # Selector's public selection plus copied internal finite window are pure
    # scalars/tuples/arrays; compare stable JSON projections where applicable.
    assert repr(controller.selector.__dict__) == repr(before["selector"])
    assert controller.mu_scenarios == before["mu"]
    np.testing.assert_array_equal(controller._adaptive_previous_state, before["previous"])
    assert controller._adaptive_last_reference_index == before["index"]


def test_initial_prior_performs_no_identification_and_bank_is_not_robust_scenarios():
    c, planning, identification = make()
    _, info = initialize(c)
    assert not identification.feature_calls and not identification.prediction_calls
    assert {call[0] for call in planning.calls} == {.7}
    assert info["prediction_mu_scenarios"] == [.7]
    a = info["adaptive_contact"]
    assert a["status"] == "initial_prior" and a["model_bank"] == [.3, .7, 1.6]
    assert a["identification_prediction_calls"] == a["identification_physics_substeps"] == 0
    assert a["source_interval_reference_index"] is None
    assert c.adaptation_contract()["prediction_models_per_solve"] == 1
    json.dumps(info, allow_nan=False)
    json.dumps(c.adaptation_contract(), allow_nan=False)


def test_update_uses_only_saved_measured_state_and_actual_previous_action():
    c, planning, identification = make()
    action, _ = initialize(c)
    planning.calls.clear()
    measured = np.array([1.6, 0., 1.6, 1.e12])
    previous = applied(action)
    _, info = c.action(measured, 1, previous_action=previous, previous_vz=37.,
                       transition_context=context())
    assert c.selector.selected_mu == 1.6
    assert {call[0] for call in planning.calls} == {1.6}
    assert len(identification.feature_calls) == 1
    assert len(identification.prediction_calls) == 3
    for mu, old, actual in identification.prediction_calls:
        assert mu in (.3, .7, 1.6)
        np.testing.assert_array_equal(old, np.zeros(4))
        np.testing.assert_array_equal(actual, previous)
    a = info["adaptive_contact"]
    assert a["selected_mu"] == 1.6 and a["source_interval_reference_index"] == 0
    assert a["identification_prediction_calls"] == 3
    assert a["identification_physics_substeps"] == 12
    assert a["transition_duration_s"] == .02 and a["estimation_compute_s"] >= 0
    assert info["warm_start_used"] is True
    # Mutating a caller-owned measurement cannot change saved causal memory.
    measured[:] = -999.
    np.testing.assert_array_equal(c._adaptive_previous_state, [1.6, 0., 1.6, 1.e12])


@pytest.mark.parametrize("wheel,impact,status", [(False, False, "no_contact"),
    (True, True, "nonwheel_contact"), (False, True, "nonwheel_contact")])
def test_skipped_context_requires_zero_identification_predictions(wheel, impact, status):
    c, _, identification = make()
    action, _ = initialize(c)
    _, info = c.action(np.ones(4), 1, previous_action=applied(action), previous_vz=0.,
                       transition_context=context(wheel, impact))
    assert identification.prediction_calls == []
    assert c.selector.selected_mu == .7
    assert info["adaptive_contact"]["status"] == status
    assert info["adaptive_contact"]["window_steps"] == 1
    assert info["adaptive_contact"]["identification_prediction_calls"] == 0


@pytest.mark.parametrize("problem", ["missing", "extra", "wheel_integer", "impact_float",
    "duration", "duration_nan", "duration_bool", "duration_array", "duration_text", "none"])
def test_invalid_context_fails_before_any_identification_or_controller_mutation(problem):
    c, planning, identification = make()
    action, _ = initialize(c)
    value = context()
    if problem == "missing": del value["duration_s"]
    if problem == "extra": value["plant_mu"] = .7
    if problem == "wheel_integer": value["wheel_ground_contact_observed"] = 1
    if problem == "impact_float": value["nonwheel_ground_contact_observed"] = 0.
    if problem == "duration": value["duration_s"] = .01
    if problem == "duration_nan": value["duration_s"] = np.nan
    if problem == "duration_bool": value["duration_s"] = True
    if problem == "duration_array": value["duration_s"] = np.array([.02])
    if problem == "duration_text": value["duration_s"] = ".02"
    if problem == "none": value = None
    before = snapshot(c); old_calls = len(planning.calls)
    with pytest.raises(ValueError):
        c.action(np.ones(4), 1, previous_action=applied(action), previous_vz=0.,
                 transition_context=value)
    assert not identification.feature_calls and not identification.prediction_calls
    assert len(planning.calls) == old_calls
    assert_unchanged(c, before)


@pytest.mark.parametrize("problem", ["nan_state", "wrong_state", "nan_action", "unrounded_action",
    "out_of_bounds_action", "vector_vz", "nan_vz", "seed_plan", "repeat", "gap", "bool_index"])
def test_invalid_solve_input_fails_before_any_identification_or_rng_change(problem):
    c, planning, identification = make()
    action, _ = initialize(c)
    x = np.ones(4); previous = applied(action); vz = 0.; seed = None; index = 1
    if problem == "nan_state": x[0] = np.nan
    if problem == "wrong_state": x = np.ones(3)
    if problem == "nan_action": previous[0] = np.nan
    if problem == "unrounded_action": previous[0] = .123456789
    if problem == "out_of_bounds_action": previous[0] = 2.
    if problem == "vector_vz": vz = np.array([0.])
    if problem == "nan_vz": vz = np.nan
    if problem == "seed_plan": seed = np.full((4, 1), np.nan)
    if problem == "repeat": index = 0
    if problem == "gap": index = 2
    if problem == "bool_index": index = True
    before = snapshot(c); old_calls = len(planning.calls)
    with pytest.raises(ValueError):
        c.action(x, index, previous_action=previous, previous_vz=vz, seed_plan=seed,
                 transition_context=context())
    assert not identification.feature_calls and not identification.prediction_calls
    assert len(planning.calls) == old_calls
    assert_unchanged(c, before)


@pytest.mark.parametrize("kind", ["runtime", "nonfinite", "observed_nonfinite", "observed_shape"])
def test_identification_failure_propagates_without_partial_bank_update_or_retry(kind):
    c, planning, identification = make()
    action, _ = initialize(c)
    if kind == "runtime": identification.error_mu = .7
    if kind == "nonfinite": identification.nonfinite_mu = .7
    if kind == "observed_nonfinite": identification.features = lambda x: [np.nan, 0., 0.]
    if kind == "observed_shape": identification.features = lambda x: [0., 0.]
    before = snapshot(c); old_calls = len(planning.calls)
    error = RuntimeError if kind == "runtime" else (ValueError if kind == "observed_shape" else FloatingPointError)
    with pytest.raises(error):
        c.action(np.ones(4), 1, previous_action=applied(action), previous_vz=0.,
                 transition_context=context())
    assert len(planning.calls) == old_calls
    assert len(identification.prediction_calls) == (2 if kind in ("runtime", "nonfinite") else 0)
    assert_unchanged(c, before)


def test_singleton_bank_preserves_baseline_action_full_plan_and_rng_exactly():
    c, _, _ = make(hypotheses=(.7,))
    baseline = ShootingNMPC(FakePlanningPredictor(), c.config,
                            action_low=[-1.], action_high=[1.], mu_scenarios=(.7,))
    previous = np.zeros(1)
    for k in range(5):
        x = np.array([k * .1, k * .2, k * .3, 1.e15 * k])
        actual, info = c.action(x, k, previous_action=previous, previous_vz=float(k),
                                transition_context=context() if k else None)
        expected, normal = baseline.action(x, k, previous_action=previous, previous_vz=float(k))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(c.last_plan, baseline.last_plan)
        assert c.rng.bit_generator.state == baseline.rng.bit_generator.state
        for key in ("candidate_evaluations", "scenario_evaluations", "selected_model_cost",
                    "selected_scenario_costs", "best_cost_by_iteration", "warm_start_used"):
            assert info[key] == normal[key]
        previous = applied(actual)


def test_reset_clears_adaptation_and_reproduces_initial_rng_and_plan():
    c, _, identification = make()
    first, _ = initialize(c)
    initial_plan = c.last_plan
    c.action(np.array([1.6, 0., 1.6, 0.]), 1, previous_action=applied(first),
             previous_vz=0., transition_context=context())
    assert c.selector.selected_mu == 1.6
    c.reset()
    assert c.selector.selected_mu == .7 and c.last_plan is None
    assert c._adaptive_previous_state is None and c._adaptive_last_reference_index is None
    old_calls = len(identification.prediction_calls)
    again, info = initialize(c)
    np.testing.assert_array_equal(again, first)
    np.testing.assert_array_equal(c.last_plan, initial_plan)
    assert len(identification.prediction_calls) == old_calls
    assert info["adaptive_contact"]["window_steps"] == 0


@pytest.mark.parametrize("index,ctx", [(1, None), (0, context())])
def test_initial_solve_requires_zero_index_and_no_preceding_context(index, ctx):
    c, planning, identification = make()
    with pytest.raises(ValueError):
        c.action(np.zeros(4), index, previous_action=np.zeros(1), previous_vz=0.,
                 transition_context=ctx)
    assert not planning.calls and not identification.feature_calls and not identification.prediction_calls


def test_full_length_synthetic_episode_has_only_348_causal_identification_updates():
    identification = FakeIdentification()
    config = AdaptiveContactConfig()
    c = AdaptiveShootingNMPC(FakePlanningPredictor(),
        CEMConfig(horizon_steps=1, control_blocks=1, candidates=2, iterations=1,
                  elite_count=1), config, identification, action_low=[-1.], action_high=[1.])
    previous = np.zeros(1)
    source_intervals = []
    for k in range(349):
        action, info = c.action(np.array([.7 * k, 0., .7 * k, float(k)]), k,
            previous_action=previous, previous_vz=0., transition_context=context() if k else None)
        source_intervals.append(info["adaptive_contact"]["source_interval_reference_index"])
        previous = applied(action)
    assert source_intervals == [None, *range(348)]
    assert len(identification.feature_calls) == 348
    assert len(identification.prediction_calls) == 348 * 3


def test_constructor_rejects_hypotheses_not_predeclared_by_predictor():
    planning = FakePlanningPredictor()
    planning.mu_scenarios = (.7,)
    with pytest.raises(ValueError, match="predeclared"):
        AdaptiveShootingNMPC(planning, CEMConfig(), AdaptiveContactConfig(),
                            FakeIdentification(), action_low=[-1.], action_high=[1.])
