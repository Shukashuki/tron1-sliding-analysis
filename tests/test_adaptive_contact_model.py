from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from adaptive_contact_model import AdaptiveContactConfig, AdaptiveContactSelector


PREDICTIONS = np.array([[0., 0., 0.], [.3, 0., 0.], [.6, 0., 0.]])


def observe(selector, predictions=PREDICTIONS, observed=None, **kwargs):
    return selector.observe(predictions, PREDICTIONS[0] if observed is None else observed,
                            contact_observed=kwargs.get("contact_observed", True),
                            nonwheel_contact=kwargs.get("nonwheel_contact", False))


def test_defaults_are_frozen_json_serializable_and_detached():
    config = AdaptiveContactConfig()
    assert asdict(config) == dict(mu_hypotheses=(.3, .7, 1.6), initial_mu=.7,
                                  feature_scales=(.05, .05, .5), window_size=10,
                                  minimum_evidence=5, minimum_spread=1., maximum_fit_rms=3.,
                                  switch_improvement_ratio=.8)
    with pytest.raises(FrozenInstanceError):
        config.initial_mu = .3
    contract = config.contract()
    json.dumps(contract, allow_nan=False)
    contract["mu_hypotheses"][0] = 999
    assert config.mu_hypotheses == (.3, .7, 1.6)
    assert contract["plant_coefficient_input"] is False
    assert contract["robustness_certificate"] is False


@pytest.mark.parametrize("updates", [
    {"mu_hypotheses": ()}, {"mu_hypotheses": (.7, .7)}, {"mu_hypotheses": (True, .7)},
    {"mu_hypotheses": (-.1, .7)}, {"mu_hypotheses": (.7, np.nan)},
    {"mu_hypotheses": "0.7"}, {"initial_mu": .6}, {"initial_mu": "0.7"},
    {"feature_scales": (.05, .1, .5)}, {"feature_scales": (.05, .05)},
    {"feature_scales": (.05, .05, 0)}, {"feature_scales": (.05, .05, np.inf)},
    {"feature_scales": (.05, .05, True)}, {"window_size": True}, {"window_size": 1.5},
    {"window_size": 0}, {"minimum_evidence": 0}, {"minimum_evidence": 11},
    {"minimum_spread": 0}, {"minimum_spread": np.nan}, {"maximum_fit_rms": -1},
    {"maximum_fit_rms": True}, {"switch_improvement_ratio": 0},
    {"switch_improvement_ratio": 1}, {"switch_improvement_ratio": np.inf},
])
def test_strict_configuration_rejection(updates):
    with pytest.raises(ValueError):
        AdaptiveContactConfig(**updates)


def test_list_configuration_is_copied_to_deeply_immutable_tuples():
    values = [.3, .7, 1.6]
    scales = [.05, .05, .5]
    config = AdaptiveContactConfig(mu_hypotheses=values, feature_scales=scales)
    values[0] = 900
    scales[0] = 800
    assert config.mu_hypotheses == (.3, .7, 1.6)
    assert config.feature_scales == (.05, .05, .5)


def test_requires_five_completed_informative_intervals_then_switches():
    selector = AdaptiveContactSelector(AdaptiveContactConfig())
    for index in range(4):
        info = observe(selector)
        assert info["status"] == "insufficient_evidence"
        assert selector.selected_mu == .7
        assert info["informative_steps"] == index + 1
    info = observe(selector)
    assert info["status"] == "switched"
    assert info["updated"] is True
    assert (info["previous_mu"], info["selected_mu"]) == (.7, .3)
    np.testing.assert_allclose(info["scores"], [0., np.sqrt(12.), np.sqrt(48.)])
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("kind", ["no_contact", "nonwheel_contact", "low_spread", "unsupported"])
def test_skipped_intervals_age_window_hold_selection_and_remove_stale_evidence(kind):
    selector = AdaptiveContactSelector(AdaptiveContactConfig())
    for _ in range(5):
        observe(selector)
    assert selector.selected_mu == .3
    for index in range(10):
        kwargs = {}
        predictions = PREDICTIONS
        measured = PREDICTIONS[0]
        if kind == "no_contact":
            kwargs["contact_observed"] = False
        elif kind == "nonwheel_contact":
            kwargs["nonwheel_contact"] = True
        elif kind == "low_spread":
            predictions = np.zeros((3, 3))
        else:
            measured = np.array([0., 20., 0.])
        info = observe(selector, predictions, measured, **kwargs)
        assert info["status"] == kind
        assert not info["updated"] and selector.selected_mu == .3
        assert info["window_steps"] == min(6+index, 10)
    assert info["informative_steps"] == 0 and info["scores"] is None
    info = observe(selector, observed=PREDICTIONS[2])
    assert info["status"] == "insufficient_evidence"
    assert info["informative_steps"] == 1 and selector.selected_mu == .3


def test_unsupported_current_observation_holds_despite_old_evidence():
    selector = AdaptiveContactSelector(AdaptiveContactConfig(minimum_evidence=2))
    observe(selector)
    info = observe(selector, observed=np.array([0., 20., 0.]))
    assert info["status"] == "unsupported" and selector.selected_mu == .7
    assert info["informative_steps"] == 1


def test_aggregate_fit_must_be_supported_even_if_each_step_has_a_good_model():
    selector = AdaptiveContactSelector(AdaptiveContactConfig(minimum_evidence=2))
    narrow = PREDICTIONS * .8
    observe(selector, narrow, narrow[0])
    info = observe(selector, narrow, narrow[-1])
    assert info["status"] == "held"  # middle model supports the mixed window
    # Use wider separation: every current interval is individually exact, but
    # no single hypothesis supports the mixed window. No edge-bank switch.
    selector.reset()
    wide = PREDICTIONS * 2
    observe(selector, wide, wide[0])
    info = observe(selector, wide, wide[-1])
    assert info["status"] == "unsupported" and selector.selected_mu == .7


def test_stable_ties_favor_current_and_hysteresis_rejects_small_gain():
    config = AdaptiveContactConfig(minimum_evidence=1)
    selector = AdaptiveContactSelector(config)
    tied = np.array([[.2, 0., 0.], [.2, 0., 0.], [.6, 0., 0.]])
    info = observe(selector, tied, np.zeros(3))
    assert info["status"] == "held" and selector.selected_mu == .7
    predictions = np.array([[.09, 0., 0.], [.1, 0., 0.], [.4, 0., 0.]])
    selector.reset()
    info = observe(selector, predictions, np.zeros(3))
    assert info["status"] == "held" and selector.selected_mu == .7


def test_tie_between_noncurrent_models_uses_bank_order_not_mu_sorting():
    selector = AdaptiveContactSelector(AdaptiveContactConfig(mu_hypotheses=(1.6, .7, .3), minimum_evidence=1))
    info = observe(selector, np.array([[0., 0., 0.], [.3, 0., 0.], [0., 0., 0.]]))
    assert info["selected_mu"] == 1.6


def test_rotation_invariance_and_common_velocity_offset_invariance():
    config = AdaptiveContactConfig(minimum_evidence=1)
    first = AdaptiveContactSelector(config)
    second = AdaptiveContactSelector(config)
    third = AdaptiveContactSelector(config)
    angle = .613
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    predictions = np.array([[.13, -.04, .2], [.34, .01, .4], [.6, .2, -.4]])
    measured = np.array([.14, -.05, .1])
    rotated = predictions.copy()
    rotated[:, :2] = predictions[:, :2] @ rotation.T
    rotated_measured = measured.copy()
    rotated_measured[:2] = measured[:2] @ rotation.T
    a = observe(first, predictions, measured)
    b = observe(second, rotated, rotated_measured)
    c = observe(third, predictions + [1., -2., .4], measured + [1., -2., .4])
    assert a["selected_mu"] == b["selected_mu"] == c["selected_mu"]
    np.testing.assert_allclose(a["scores"], b["scores"], atol=1e-14)
    np.testing.assert_allclose(a["scores"], c["scores"], atol=1e-14)
    assert a["prediction_spread_rms"] == pytest.approx(b["prediction_spread_rms"])


def test_single_model_has_no_identification_evidence():
    selector = AdaptiveContactSelector(AdaptiveContactConfig(mu_hypotheses=(.7,), minimum_evidence=1))
    info = observe(selector, np.zeros((1, 3)))
    assert info["status"] == "low_spread" and info["informative_steps"] == 0
    assert selector.selected_mu == .7


@pytest.mark.parametrize("predictions, observed, flags", [
    (np.zeros((2, 3)), np.zeros(3), {}),
    (np.zeros((3, 4)), np.zeros(3), {}),
    (np.zeros((3, 3)), np.zeros((1, 3)), {}),
    (np.full((3, 3), np.nan), np.zeros(3), {}),
    (np.zeros((3, 3)), np.array([np.inf, 0., 0.]), {}),
    (np.zeros((3, 3), dtype=complex), np.zeros(3), {}),
    (np.full((3, 3), "0"), np.zeros(3), {}),
    (np.ones((3, 3), dtype=bool), np.zeros(3), {}),
    (PREDICTIONS, np.zeros(3), {"contact_observed": 1}),
    (PREDICTIONS, np.zeros(3), {"nonwheel_contact": np.bool_(False)}),
    (np.full((3, 3), 1e308), np.full(3, -1e308), {}),
])
def test_invalid_observation_is_atomic(predictions, observed, flags):
    first = AdaptiveContactSelector(AdaptiveContactConfig())
    reference = AdaptiveContactSelector(AdaptiveContactConfig())
    for _ in range(4):
        observe(first)
        observe(reference)
    with pytest.raises(ValueError):
        observe(first, predictions, observed, **flags)
    assert observe(first) == observe(reference)


def test_input_arrays_are_never_retained_and_reset_clears_all_evidence():
    selector = AdaptiveContactSelector(AdaptiveContactConfig())
    predictions = PREDICTIONS.copy()
    measured = np.zeros(3)
    observe(selector, predictions, measured)
    predictions[:] = 900
    measured[:] = 800
    for _ in range(4):
        info = observe(selector)
    assert info["selected_mu"] == .3
    selector.reset()
    assert selector.selected_mu == .7
    info = observe(selector)
    assert info["informative_steps"] == 1 and info["window_steps"] == 1


def test_large_finite_features_do_not_overflow_score_squares():
    selector = AdaptiveContactSelector(AdaptiveContactConfig(feature_scales=(1., 1., 1.)))
    predictions = np.array([[1e180, 0., 0.], [0., 1e180, 0.], [0., 0., 1e180]])
    info = observe(selector, predictions, np.zeros(3))
    assert info["status"] == "unsupported"
    assert np.isfinite(info["prediction_spread_rms"])
    json.dumps(info, allow_nan=False)


def test_selector_rejects_untyped_configuration():
    with pytest.raises(TypeError):
        AdaptiveContactSelector({})
