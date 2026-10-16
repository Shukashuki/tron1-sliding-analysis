"""Causal multiple-model adaptive wrapper around the unchanged shooting NMPC.

An identification update uses the *preceding measured* control interval and
the actual float32-applied action supplied by the runner. A selected friction
hypothesis is an effective prediction-model index, not a sensor-validated
physical friction estimate. No reference, validation coefficient, reward, or
future plant state is supplied to the estimator.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np

from adaptive_contact_model import AdaptiveContactConfig, AdaptiveContactSelector
from nmpc_shooting import ShootingNMPC


CONTROL_DT_S = .02
IDENTIFICATION_SUBSTEPS = 4
CONTEXT_KEYS = frozenset(("wheel_ground_contact_observed",
                          "nonwheel_ground_contact_observed", "duration_s"))


def _features(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,): COM vx, vy, world angular wz")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Nonfinite {name}")
    return result.copy()


def _transition_context(value):
    if not isinstance(value, Mapping) or set(value) != CONTEXT_KEYS:
        raise ValueError("transition_context must contain exactly the two measured contact flags and duration_s")
    flags = []
    for key in ("wheel_ground_contact_observed", "nonwheel_ground_contact_observed"):
        flag = value[key]
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"transition_context {key} must be boolean")
        flags.append(bool(flag))
    duration = value["duration_s"]
    if (isinstance(duration, (bool, np.bool_))
            or not isinstance(duration, (int, float, np.integer, np.floating))
            or not np.isfinite(duration)
            or not np.isclose(duration, CONTROL_DT_S, atol=1e-12, rtol=0)):
        raise ValueError("transition_context duration_s must equal the actual 0.02 s control interval")
    return (*flags, float(duration))


class AdaptiveShootingNMPC(ShootingNMPC):
    """One selected prediction model per solve, with causal interval updates.

    ``identification_model`` provides ``features(packed) -> (3,)`` and
    ``transition_features(packed, actual_applied_action, *, mu) -> (3,)``.
    All bank models must already exist. The adapter must use isolated model
    state and exactly four 5 ms substeps per transition, never step the plant.

    Contact context describes all four samples of the preceding actual plant
    interval, not the current reset snapshot or a predicted contact. The
    runner includes this wrapper's entire call in its measured solve latency.
    """

    def __init__(self, predictor, cem, adaptation_config, identification_model,
                 *, action_low, action_high):
        if not isinstance(adaptation_config, AdaptiveContactConfig):
            raise TypeError("adaptation_config must be AdaptiveContactConfig")
        if not callable(getattr(identification_model, "features", None)) or not callable(
                getattr(identification_model, "transition_features", None)):
            raise TypeError("identification_model requires features and transition_features")
        self.adaptation_config = adaptation_config
        self.selector = AdaptiveContactSelector(adaptation_config)
        self.identification_model = identification_model
        self.model_bank = tuple(float(mu) for mu in adaptation_config.mu_hypotheses)
        declared = getattr(predictor, "mu_scenarios", None)
        if declared is not None and any(mu not in declared for mu in self.model_bank):
            raise ValueError("All adaptive hypotheses must belong to the predictor's predeclared model bank")
        super().__init__(predictor, cem, action_low=action_low, action_high=action_high,
                         mu_scenarios=(self.selector.selected_mu,))

    def reset(self):
        # ShootingNMPC.__init__ intentionally calls self.reset(). The selector
        # is allocated before that call, and reset never rebuilds a controller.
        super().reset()
        self.selector.reset()
        self.mu_scenarios = (float(self.selector.selected_mu),)
        self._adaptive_previous_state = None
        self._adaptive_last_reference_index = None

    def adaptation_contract(self):
        return {
            "kind": "causal_multiple_model_adaptive_prediction_selection",
            "selector": self.selector.contract(),
            "model_bank": list(self.model_bank),
            "prediction_models_per_solve": 1,
            "update_timing": "Before solve k using measured transition k-1 and its float32-applied action; no update at k=0.",
            "identification_model_substeps": IDENTIFICATION_SUBSTEPS,
            "identification_interval_s": CONTROL_DT_S,
            "selected_mu_is_physical_friction_estimate": False,
            "plant_mu_used_for_prediction": False,
            "future_plant_state_used": False,
            "reference_used_for_identification": False,
            "identification_failure_policy": "Propagate; no dropped model, candidate substitution, retry, or worker restart.",
            "robustness_certificate": False,
        }

    def _validated_inputs(self, state, reference_index, previous_action, previous_vz,
                          seed_plan, transition_context):
        x = np.asarray(state, dtype=float).copy()
        previous = np.asarray(previous_action, dtype=float).copy()
        if x.ndim != 1 or not x.size or not np.isfinite(x).all():
            raise ValueError("state must be a nonempty finite vector")
        if previous.shape != (self.nu,) or not np.isfinite(previous).all():
            raise ValueError("previous_action must be a finite action vector")
        if np.any(previous < self.low) or np.any(previous > self.high):
            raise ValueError("previous_action must be within actual action bounds")
        if not np.array_equal(previous, previous.astype(np.float32).astype(float)):
            raise ValueError("previous_action must be the actual float32-applied action")
        velocity = np.asarray(previous_vz, dtype=float)
        if velocity.shape != () or not np.isfinite(velocity):
            raise ValueError("previous_vz must be a finite scalar")
        if (isinstance(reference_index, (bool, np.bool_))
                or not isinstance(reference_index, (int, np.integer)) or reference_index < 0):
            raise ValueError("reference_index must be a nonnegative integer")
        index = int(reference_index)
        expected = 0 if self._adaptive_last_reference_index is None else self._adaptive_last_reference_index + 1
        if index != expected:
            raise ValueError("Adaptive updates require contiguous reference indices beginning at zero")
        if seed_plan is not None:
            self._plan(seed_plan)  # Fail before any identification prediction.
        if index == 0:
            if transition_context is not None:
                raise ValueError("The first solve has no preceding transition_context")
            context = None
        else:
            if self._adaptive_previous_state is None or x.shape != self._adaptive_previous_state.shape:
                raise ValueError("Measured state dimensions changed between adaptive solves")
            context = _transition_context(transition_context)
        return x, index, previous, float(velocity), context

    def action(self, state, reference_index: int, *, previous_action, previous_vz,
               seed_plan=None, transition_context=None):
        x, index, previous, previous_vz, context = self._validated_inputs(
            state, reference_index, previous_action, previous_vz, seed_plan, transition_context)
        started = time.perf_counter()
        calls = 0
        if index == 0:
            adaptation = {"status": "initial_prior", "updated": False,
                          "selected_mu": float(self.selector.selected_mu),
                          "previous_mu": float(self.selector.selected_mu),
                          "scores": None, "prediction_spread_rms": None,
                          "window_steps": 0, "informative_steps": 0}
        else:
            wheel_contact, nonwheel_contact, _ = context
            observed = _features(self.identification_model.features(x.copy()), "observed features")
            if wheel_contact and not nonwheel_contact:
                # Complete and validate the whole bank before mutating selector,
                # CEM scenarios, warm plan, RNG, or saved interval memory.
                predictions = []
                for mu in self.model_bank:
                    predicted = self.identification_model.transition_features(
                        self._adaptive_previous_state.copy(), previous.copy(), mu=mu)
                    predictions.append(_features(predicted, "predicted features"))
                    calls += 1
                predictions = np.stack(predictions)
            else:
                # A valid skipped interval ages the selector's finite window,
                # but requires no simulated identification transition.
                predictions = np.tile(observed, (len(self.model_bank), 1))
            adaptation = self.selector.observe(predictions, observed,
                contact_observed=wheel_contact, nonwheel_contact=nonwheel_contact)
        estimation_s = time.perf_counter() - started
        selected = float(self.selector.selected_mu)
        if selected not in self.model_bank:
            raise ValueError("Selector returned a model outside the predeclared bank")
        self.mu_scenarios = (selected,)
        action, info = super().action(x, index, previous_action=previous,
                                     previous_vz=previous_vz, seed_plan=seed_plan)
        self._adaptive_previous_state = x.copy()
        self._adaptive_last_reference_index = index
        info["adaptive_contact"] = {
            **adaptation,
            "model_bank": list(self.model_bank),
            "selected_mu": selected,
            "identification_prediction_calls": calls,
            "identification_physics_substeps": calls * IDENTIFICATION_SUBSTEPS,
            "estimation_compute_s": float(estimation_s),
            "source_interval_reference_index": index - 1 if index else None,
            "transition_duration_s": context[2] if context is not None else None,
            "prediction_models_per_solve": 1,
            "selected_mu_is_physical_friction_estimate": False,
            "timing_scope": "Identification and selector only; outer runner latency additionally includes unchanged CEM solve.",
        }
        return action, info
