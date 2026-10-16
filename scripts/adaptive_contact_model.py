"""Causal finite-bank contact-model selection, not physical-friction sensing.

The caller predicts an already completed, actually applied control interval
under every fixed hypothesis. This pure module sees only planar COM velocity
and world yaw-rate features. It cannot read a plant coefficient, future state,
reference error, contact forces, simulator, action planner, or learned policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np


FEATURE_NAMES = ("base_com_vx_world_m_s", "base_com_vy_world_m_s", "base_omega_z_world_rad_s")


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


@dataclass(frozen=True)
class AdaptiveContactConfig:
    mu_hypotheses: tuple[float, ...] = (.3, .7, 1.6)
    initial_mu: float = .7
    feature_scales: tuple[float, float, float] = (.05, .05, .5)
    window_size: int = 10
    minimum_evidence: int = 5
    minimum_spread: float = 1.0
    maximum_fit_rms: float = 3.0
    switch_improvement_ratio: float = .8

    def __post_init__(self):
        if not isinstance(self.mu_hypotheses, (tuple, list)) or not self.mu_hypotheses:
            raise ValueError("mu_hypotheses must be a nonempty tuple/list")
        bank = tuple(_real(value, "mu_hypotheses") for value in self.mu_hypotheses)
        if len(set(bank)) != len(bank):
            raise ValueError("mu_hypotheses must be unique")
        initial = _real(self.initial_mu, "initial_mu")
        if initial not in bank:
            raise ValueError("initial_mu must belong to mu_hypotheses")
        if not isinstance(self.feature_scales, (tuple, list)) or len(self.feature_scales) != 3:
            raise ValueError("feature_scales must contain three positive scales")
        scales = tuple(_real(value, "feature_scales", positive=True) for value in self.feature_scales)
        if scales[0] != scales[1]:
            raise ValueError("Equal XY feature scales are required for planar rotation invariance")
        for name in ("window_size", "minimum_evidence"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.minimum_evidence > self.window_size:
            raise ValueError("minimum_evidence cannot exceed window_size")
        object.__setattr__(self, "mu_hypotheses", bank)
        object.__setattr__(self, "initial_mu", initial)
        object.__setattr__(self, "feature_scales", scales)
        for name in ("minimum_spread", "maximum_fit_rms", "switch_improvement_ratio"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if self.switch_improvement_ratio >= 1:
            raise ValueError("switch_improvement_ratio must be strictly between zero and one")

    def contract(self):
        result = asdict(self)
        result["mu_hypotheses"] = list(self.mu_hypotheses)
        result["feature_scales"] = list(self.feature_scales)
        result.update(feature_names=list(FEATURE_NAMES),
                      feature_semantics="Completed-interval endpoint COM world vx/vy and world omega_z; equivalent increment residuals when every prediction starts at the same observed state.",
                      spread_definition="Maximum pairwise RMS of scaled hypothesis feature differences, across three channels.",
                      score_definition="RMS of scaled prediction-minus-observation residuals over informative transitions and three channels.",
                      window_clock="Every valid observed interval ages the bounded window; skipped or unsupported intervals append None.",
                      selected_mu_semantics="Effective prediction-contact coefficient, not a measured physical friction coefficient.",
                      robustness_certificate=False, posterior_confidence=False,
                      plant_coefficient_input=False, reference_error_input=False)
        return result


def _rms(values, axis=None):
    """Stable RMS without squaring large unscaled residuals."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values), axis=axis, keepdims=True)
    divisor = np.where(scale == 0, 1., scale)
    result = scale * np.sqrt(np.mean((values / divisor) ** 2, axis=axis, keepdims=True))
    return np.squeeze(result) if axis is None else np.squeeze(result, axis=axis)


class AdaptiveContactSelector:
    def __init__(self, config: AdaptiveContactConfig):
        if not isinstance(config, AdaptiveContactConfig):
            raise TypeError("config must be an AdaptiveContactConfig")
        self.config = config
        self.reset()

    @property
    def selected_mu(self):
        return self._selected_mu

    def reset(self):
        self._selected_mu = self.config.initial_mu
        self._window = deque(maxlen=self.config.window_size)

    def contract(self):
        return self.config.contract()

    def observe(self, predictions, observed, *, contact_observed: bool,
                nonwheel_contact: bool):
        """Validate atomically; then age one causal interval and possibly switch.

        Contact booleans must be actual interval evidence supplied by the caller,
        not inferred from FK/net force. They gate identification only, never task
        success. Unsupported data cannot become a confident edge-bank estimate.
        """
        if type(contact_observed) is not bool or type(nonwheel_contact) is not bool:
            raise ValueError("Contact evidence flags must be built-in booleans")
        try:
            predicted = np.asarray(predictions)
            actual = np.asarray(observed)
            if predicted.dtype.kind not in "iuf" or actual.dtype.kind not in "iuf":
                raise ValueError("Features must contain real numeric values")
            predicted = np.asarray(predicted, dtype=float)
            actual = np.asarray(actual, dtype=float)
        except (TypeError, OverflowError) as error:
            raise ValueError("Invalid numeric feature arrays") from error
        if (predicted.shape != (len(self.config.mu_hypotheses), 3) or actual.shape != (3,)
                or not np.isfinite(predicted).all() or not np.isfinite(actual).all()):
            raise ValueError("Expected finite predictions[N,3] and observed[3]")
        # Prepare EVERYTHING, including reductions, before changing state.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scales = np.asarray(self.config.feature_scales)
                residuals = (predicted - actual) / scales
                spread = 0.
                for index in range(len(predicted) - 1):
                    distances = _rms((predicted[index+1:] - predicted[index]) / scales, axis=1)
                    spread = max(spread, float(np.max(distances)))
                step_scores = _rms(residuals, axis=1)
                if nonwheel_contact:
                    status = "nonwheel_contact"
                elif not contact_observed:
                    status = "no_contact"
                elif spread < self.config.minimum_spread:
                    status = "low_spread"
                elif float(np.min(step_scores)) > self.config.maximum_fit_rms:
                    status = "unsupported"
                else:
                    status = None
                entry = residuals.copy() if status is None else None
                window = (list(self._window) + [entry])[-self.config.window_size:]
                evidence = [row for row in window if row is not None]
                scores = _rms(np.stack(evidence), axis=(0, 2)) if evidence else None
                if scores is not None and not np.isfinite(scores).all():
                    raise ValueError("Nonfinite aggregate model scores")
        except FloatingPointError as error:
            raise ValueError("Feature arithmetic exceeds the finite numeric contract") from error
        previous = self._selected_mu
        selected = previous
        if status is None:
            if len(evidence) < self.config.minimum_evidence:
                status = "insufficient_evidence"
            elif float(np.min(scores)) > self.config.maximum_fit_rms:
                status = "unsupported"
            else:
                current_index = self.config.mu_hypotheses.index(previous)
                minimum = float(np.min(scores))
                # Exact score ties favor the current model; other ties use bank
                # declaration order, never friction magnitude or random choice.
                best_index = current_index if scores[current_index] == minimum else int(np.argmin(scores))
                if (best_index != current_index
                        and scores[best_index] <= self.config.switch_improvement_ratio * scores[current_index]):
                    selected = self.config.mu_hypotheses[best_index]
                    status = "switched"
                else:
                    status = "held"
        result = {"status": status, "updated": selected != previous,
                  "selected_mu": float(selected), "previous_mu": float(previous),
                  "scores": None if scores is None else [float(value) for value in scores],
                  "prediction_spread_rms": float(spread), "window_steps": len(window),
                  "informative_steps": len(evidence)}
        self._window = deque(window, maxlen=self.config.window_size)
        self._selected_mu = selected
        return result
