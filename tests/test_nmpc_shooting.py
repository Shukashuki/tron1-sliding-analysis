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


@pytest.mark.parametrize("mode", ["absolute_blocks", "warm_start_residual_blocks"])
def test_warm_start_shifts_previous_plan_and_repeats_final_action(mode):
    p = RecordingRollout()
    c = controller(p,parameterization=mode)
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


class BatchRecordingRollout(RecordingRollout):
    def __init__(self, costs=None):
        super().__init__(costs)
        self.batch_calls = []

    def rollout_batch(self, state, plans, reference_index, **kwargs):
        self.batch_calls.append((kwargs["mu"], plans.copy(), reference_index,
                                 kwargs["previous_action"].copy(), kwargs["previous_vz"]))
        return np.array([self.rollout(state, plan, reference_index, **kwargs) for plan in plans])


def assert_same_decision_and_counts(scalar, batch, scalar_info, batch_info):
    np.testing.assert_array_equal(scalar.last_plan, batch.last_plan)
    for name in ("selected_model_cost", "selected_scenario_costs", "initial_model_cost",
                 "candidate_evaluations", "finite_candidates", "failed_candidates",
                 "scenario_evaluations", "failed_scenarios", "failed_scenarios_by_mu",
                 "best_cost_by_iteration", "initial_candidate_source", "warm_start_used",
                 "optimization_failed", "fallback_used", "fallback_kind"):
        assert scalar_info[name] == batch_info[name], name
    assert scalar_info["population_evaluation_backend"] == "scalar_candidate_loop"
    assert scalar_info["batch_prediction_calls"] == 0
    assert batch_info["population_evaluation_backend"] == "predictor_rollout_batch"


@pytest.mark.parametrize("reduction", ["worst", "mean"])
@pytest.mark.parametrize("mode", ["absolute_blocks", "warm_start_residual_blocks"])
def test_optional_batch_matches_seeded_scalar_plans_actions_counts_and_warm_start(reduction,mode):
    config = CEMConfig(horizon_steps=6, control_blocks=3, candidates=16,
                       iterations=3, elite_count=4, initial_std=.5,
                       scenario_reduction=reduction, seed=712,parameterization=mode)
    p, bp = RecordingRollout(), BatchRecordingRollout()
    scalar = ShootingNMPC(p, config, action_low=[-1.], action_high=[1.], mu_scenarios=(.2,.7))
    batch = ShootingNMPC(bp, config, action_low=[-1.], action_high=[1.], mu_scenarios=(.2,.7))
    previous = np.array([.1])
    for frame in (0, 1, 3):
        a, info = scalar.action(np.array([.4, -.2]), frame, previous_action=previous,
                                previous_vz=.3, seed_plan=np.full((6,1), .35))
        b, other = batch.action(np.array([.4, -.2]), frame, previous_action=previous,
                                previous_vz=.3, seed_plan=np.full((6,1), .35))
        np.testing.assert_array_equal(a, b)
        assert_same_decision_and_counts(scalar, batch, info, other)
        assert other["failed_batch_prediction_calls"] == 0
        assert other["batch_prediction_calls"] == 2*(2+config.iterations)
        previous = a.copy()
    # Entire candidate plans, initial phase and objective memory are held
    # identically across scenarios, including singleton initial/warm plans.
    for low, high in zip(bp.batch_calls[::2], bp.batch_calls[1::2]):
        assert (low[0], high[0]) == (.2, .7)
        np.testing.assert_array_equal(low[1], high[1])
        assert low[2] == high[2]
        np.testing.assert_array_equal(low[3], high[3])
        assert low[4] == high[4] == .3
    batch.reset();scalar.reset()
    a, info = scalar.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.)
    b, other = batch.action(np.zeros(2), 0, previous_action=np.zeros(1), previous_vz=0.)
    np.testing.assert_array_equal(a, b)
    assert_same_decision_and_counts(scalar, batch, info, other)


@pytest.mark.parametrize("mode", ["absolute_blocks", "warm_start_residual_blocks"])
def test_batch_per_candidate_nonfinite_scenario_costs_have_scalar_failure_accounting(mode):
    class FailingScalar(RecordingRollout):
        def rollout(self, state, plan, reference_index, **kwargs):
            if kwargs["mu"] == .2 and np.any(plan < 0.):
                return np.nan
            return RecordingRollout.rollout(self, state, plan, reference_index, **kwargs)

    class FailingBatch(BatchRecordingRollout):
        rollout = FailingScalar.rollout

    config = CEMConfig(horizon_steps=3, control_blocks=2, candidates=12,
                       iterations=2, elite_count=3, initial_std=.6,parameterization=mode)
    scalar = ShootingNMPC(FailingScalar(),config,action_low=[-1.],action_high=[1.],mu_scenarios=(.2,.7))
    batch = ShootingNMPC(FailingBatch(),config,action_low=[-1.],action_high=[1.],mu_scenarios=(.2,.7))
    kwargs = dict(previous_action=np.array([-.5]), previous_vz=.2)
    a, info = scalar.action(np.zeros(2), 5, **kwargs)
    b, other = batch.action(np.zeros(2), 5, **kwargs)
    np.testing.assert_array_equal(a,b)
    assert_same_decision_and_counts(scalar,batch,info,other)
    assert other["failed_scenarios_by_mu"][0] > 0
    assert other["failed_scenarios_by_mu"][1] == 0
    assert other["failed_batch_prediction_calls"] == 0


@pytest.mark.parametrize("fallback, expected", [("previous", .7), ("neutral", 0.)])
@pytest.mark.parametrize("mode", ["absolute_blocks", "warm_start_residual_blocks"])
def test_batch_runtime_warning_rejects_whole_scenario_and_has_explicit_fallback(fallback, expected,mode):
    class BatchWarning(BatchRecordingRollout):
        def rollout_batch(self, state, plans, reference_index, **kwargs):
            if kwargs["mu"] == .2:
                raise RuntimeError("Native warning: entire batch rejected")
            return np.ones(len(plans))

    config = CEMConfig(horizon_steps=3,control_blocks=1,candidates=4,
                       iterations=2,elite_count=2,fallback_action=fallback,parameterization=mode)
    c = ShootingNMPC(BatchWarning(),config,action_low=[-1.],action_high=[1.],mu_scenarios=(.2,.7))
    action, info = c.action(np.zeros(2),0,previous_action=np.array([.7]),previous_vz=0.)
    np.testing.assert_array_equal(action,[expected])
    assert info["optimization_failed"] and info["fallback_used"]
    assert info["failed_candidates"] == info["candidate_evaluations"] == 9
    assert info["failed_scenarios_by_mu"] == [9,0]
    assert info["batch_prediction_calls"] == 6
    assert info["failed_batch_prediction_calls"] == 3
    assert info["batch_runtime_failure_scope"] == "All candidates in the failed scenario call"
    assert c.last_plan is None
    json.dumps(info,allow_nan=False)


@pytest.mark.parametrize("problem", ["shape", "value_error"])
def test_batch_programming_errors_are_fatal_not_silent_fallback(problem):
    class BadBatch(BatchRecordingRollout):
        def rollout_batch(self,state,plans,reference_index,**kwargs):
            if problem == "value_error":
                raise ValueError("Deliberate malformed objective input")
            return np.zeros((len(plans),1))

    c = controller(BadBatch())
    with pytest.raises(ValueError,match="shape|malformed"):
        c.action(np.zeros(2),0,previous_action=np.zeros(1),previous_vz=0.)
    assert c.last_plan is None


@pytest.mark.parametrize("mode", ["absolute_blocks", "warm_start_residual_blocks"])
def test_batch_constant_cost_ties_keep_first_incumbent_and_stable_candidate_order(mode):
    costs={1.:2.}
    scalar, batch = controller(RecordingRollout(costs),parameterization=mode), controller(BatchRecordingRollout(costs),parameterization=mode)
    a, info = scalar.action(np.zeros(2),0,previous_action=np.array([.23]),previous_vz=.4,
                            seed_plan=np.full((8,1),.9))
    b, other = batch.action(np.zeros(2),0,previous_action=np.array([.23]),previous_vz=.4,
                           seed_plan=np.full((8,1),.9))
    np.testing.assert_array_equal(a,[.23]);np.testing.assert_array_equal(a,b)
    assert_same_decision_and_counts(scalar,batch,info,other)


def test_default_absolute_exact_prechange_plan_cost_and_random_stream_golden():
    c=controller(NonlinearPredictor())
    action,info=c.action(np.zeros(2),0,previous_action=np.zeros(1),previous_vz=0.)
    expected=np.array([.9555126940917484,.9555126940917484,.9750008602087729,.9750008602087729,
                       .11149400230942114,.11149400230942114,-.23077745882095135,-.23077745882095135])[:,None]
    np.testing.assert_array_equal(c.last_plan,expected)
    np.testing.assert_array_equal(action,expected[0])
    assert info["selected_model_cost"]==.9881224923456084
    assert info["initial_model_cost"]==10.
    assert info["best_cost_by_iteration"]==[2.799728496110986,1.2879915966675333,.9881224923456084]
    np.testing.assert_array_equal(c.rng.normal(size=4),[-1.2798409576400205,1.681816619300824,1.7289953408857035,1.359220600295612])
    assert info["action_parameterization"]=="absolute_blocks"


class PrescribedNormals:
    def __init__(self,*rows):
        self.rows=list(rows)
    def normal(self,*,size):
        return np.asarray(self.rows.pop(0),dtype=float).reshape(size)


def residual_controller(predictor,**kwargs):
    values=dict(horizon_steps=3,control_blocks=1,candidates=3,iterations=2,elite_count=1,
                initial_std=1.,min_std=1.,parameterization="warm_start_residual_blocks")
    values.update(kwargs)
    return ShootingNMPC(predictor,CEMConfig(**values),action_low=[-1.],action_high=[1.],mu_scenarios=(1.,))


def test_residual_fixed_nonconstant_baseline_encoded_elites_and_paired_incumbent_under_clip():
    baseline=np.array([-.8,0.,.8])[:,None]
    target=np.clip(baseline+.5,-1.,1.)
    class Seeded(BatchRecordingRollout):
        def initial_candidate(self,*args,**kwargs):
            return baseline.copy()
        def rollout(self,state,plan,reference_index,**kwargs):
            return float(np.sum((plan-target)**2))
    p=Seeded();c=residual_controller(p)
    c.rng=PrescribedNormals([0.,.5,-.5],[99.,0.,.2])
    action,info=c.action(np.zeros(2),0,previous_action=np.zeros(1),previous_vz=0.)
    populations=[call[1] for call in p.batch_calls if len(call[1])==3]
    # Zero delta preserves a full plan which cannot be represented by K=1
    # absolute blocks. Initial fallback and seed were scored before sampling.
    np.testing.assert_array_equal(populations[0][0],baseline)
    np.testing.assert_array_equal(populations[0][1],target)
    # Saturation means mean(decoded-baseline)=.4, NOT encoded delta=.5.
    # Both row0 and the zero-noise row1 must retain delta=.5 on iteration2.
    np.testing.assert_array_equal(populations[1][0],target)
    np.testing.assert_array_equal(populations[1][1],target)
    np.testing.assert_array_equal(c.last_plan,target)
    np.testing.assert_array_equal(action,target[0])
    np.testing.assert_array_equal(baseline,np.array([-.8,0.,.8])[:,None])
    assert info["residual_selected_parameter_linf"]==.5
    assert info["initial_baseline_block_projection_max_abs_error"]==.8
    assert info["residual_action_decode_clip_fraction"]>0.
    assert info["residual_baseline_fixed_per_solve"] is True


def test_residual_encoder_unequal_blocks_asymmetric_multiaction_bounds_and_finiteness():
    c=ShootingNMPC(RecordingRollout(),CEMConfig(horizon_steps=7,control_blocks=3,candidates=2,
        iterations=1,elite_count=1,parameterization="warm_start_residual_blocks"),
        action_low=[-2.,1.],action_high=[1.,4.],mu_scenarios=(1.,))
    baseline=np.column_stack((np.linspace(-2.,1.,7),np.linspace(4.,1.,7)))
    delta=np.array([[[100.,-100.],[.2,-.2],[-.3,.3]]])
    encoded=c._encode_residual(delta)
    plans,raw=c._decode_residual(baseline,encoded)
    np.testing.assert_array_equal(encoded[0,0],[3.,-3.])
    np.testing.assert_array_equal(plans,np.clip(baseline[None]+encoded[:,c.assignment],[-2.,1.],[1.,4.]))
    assert c.assignment.tolist()==[0,0,0,1,1,2,2]
    invalid=delta.copy();invalid[0,0,0]=np.inf
    with pytest.raises(FloatingPointError,match="before encoding clip"):
        c._encode_residual(invalid)
    with pytest.raises(ValueError,match="already-clipped"):
        c._decode_residual(baseline,delta)
    assert c.parameterization_contract()["baseline_copied_and_fixed_per_solve"] is True


@pytest.mark.parametrize("low,high", [(-1e308,1e308),(-1e16,1.)])
def test_residual_rejects_nonfinite_or_rounded_lossy_span_without_changing_absolute_constructor(low,high):
    base=dict(horizon_steps=3,control_blocks=1,candidates=2,iterations=1,elite_count=1)
    ShootingNMPC(RecordingRollout(),CEMConfig(**base),action_low=[low],action_high=[high],mu_scenarios=(1.,))
    with pytest.raises(ValueError,match="endpoint saturation"):
        ShootingNMPC(RecordingRollout(),CEMConfig(**base,parameterization="warm_start_residual_blocks"),
            action_low=[low],action_high=[high],mu_scenarios=(1.,))


def test_residual_checks_decoded_sum_and_elite_moments_before_any_final_clip():
    cfg=CEMConfig(horizon_steps=3,control_blocks=1,candidates=2,iterations=1,elite_count=1,
                  parameterization="warm_start_residual_blocks")
    c=ShootingNMPC(RecordingRollout(),cfg,action_low=[0.],action_high=[1e308],mu_scenarios=(1.,))
    with pytest.raises(FloatingPointError):
        c._decode_residual(np.full((3,1),1e308),np.full((1,1,1),1e308))
    c=residual_controller(RecordingRollout())
    with pytest.raises(FloatingPointError):
        c._residual_moments(np.array([[[1e200]],[[-1e200]]]),np.zeros((1,1)),np.ones((1,1)))


def test_residual_all_initial_fail_search_can_recover_and_all_search_fail_retains_initial():
    class InitialFail(BatchRecordingRollout):
        def rollout_batch(self,state,plans,reference_index,**kwargs):
            return np.array([np.inf if np.any(plan<0.) else float(np.sum((plan-.5)**2)) for plan in plans])
    c=residual_controller(InitialFail(),iterations=1)
    c.rng=PrescribedNormals([0.,1.,-1.])
    action,info=c.action(np.zeros(2),0,previous_action=np.array([-.5]),previous_vz=0.)
    np.testing.assert_array_equal(action,[.5])
    assert info["initial_candidate_source"]=="all_initial_candidates_failed"
    assert not info["optimization_failed"]
    class SearchFail(BatchRecordingRollout):
        def rollout_batch(self,state,plans,reference_index,**kwargs):
            if len(plans)>1:
                raise RuntimeError("whole population rejected")
            return np.array([1.])
    c=residual_controller(SearchFail())
    action,info=c.action(np.zeros(2),0,previous_action=np.array([.2]),previous_vz=0.)
    np.testing.assert_array_equal(action,[.2])
    assert not info["fallback_used"] and info["failed_batch_prediction_calls"]==2


def test_residual_repeated_backward_long_phase_shifts_and_reset_create_fresh_zero_baseline():
    class Seeded(BatchRecordingRollout):
        def initial_candidate(self,*args,**kwargs):
            return np.array([.2,.4,.6])[:,None]
    p=Seeded();c=residual_controller(p,iterations=1)
    for phase in (0,1,4,4,2):
        p.batch_calls.clear()
        action,info=c.action(np.zeros(2),phase,previous_action=np.zeros(1),previous_vz=0.)
        population=[x[1] for x in p.batch_calls if len(x[1])==3][0]
        initial_calls=[x[1][0] for x in p.batch_calls if len(x[1])==1]
        # Candidate0 is exactly one fully scored initial plan, not a residual
        # accidentally carried from a preceding solve with a different base.
        assert any(np.array_equal(population[0],seed) for seed in initial_calls)
    c.reset();assert c.last_plan is None
    with pytest.raises(ValueError,match="parameterization"):
        CEMConfig(parameterization="unknown")


def test_residual_total_failure_clears_existing_warm_plan_and_returns_bounded_unlearned_fallback():
    p=BatchRecordingRollout(costs={1.:1.});c=residual_controller(p,iterations=1)
    c.action(np.zeros(2),0,previous_action=np.array([.1]),previous_vz=0.)
    assert c.last_plan is not None
    p.costs={1.:np.inf}
    action,info=c.action(np.zeros(2),1,previous_action=np.array([.2]),previous_vz=0.)
    np.testing.assert_array_equal(action,[.2])
    assert c.last_plan is None and info["optimization_failed"] and info["fallback_used"]
    assert not info["learned_policy_fallback"]


def test_residual_profile_changes_only_parameterization_and_notes():
    root=Path(__file__).parents[1]
    base=json.loads((root/"config/nmpc_height_smooth.json").read_text())
    residual=json.loads((root/"config/nmpc_height_smooth_residual.json").read_text())
    assert residual["nmpc"].pop("parameterization")=="warm_start_residual_blocks"
    residual.pop("notes");base.pop("notes")
    assert residual==base
