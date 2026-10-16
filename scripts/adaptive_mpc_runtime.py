"""Explicit adaptive-MPC runner plumbing; no plant-friction access or tuning.

The identification model uses the same cold-reset, four-substep Python
transition as the existing audited predictor. Native CEM remains unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import json
from pathlib import Path

import numpy as np


def load_adaptation(path, explicit_scenarios=None):
    from adaptive_contact_model import AdaptiveContactConfig
    value = json.loads(Path(path).read_text())
    if (type(value) is not dict or set(value) != {"schema_version", "kind", "estimator"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "finite_contact_model_adaptive_mpc"):
        raise ValueError("Expected explicit schema-1 adaptive contact-model configuration")
    options = value["estimator"]
    if not isinstance(options, dict) or set(options) != {field.name for field in fields(AdaptiveContactConfig)}:
        raise ValueError("Adaptive configuration must explicitly declare every estimator field")
    options = dict(options)
    for key in ("mu_hypotheses", "feature_scales"):
        if not isinstance(options[key], (list, tuple)):
            raise ValueError("Adaptive bank and feature scales must be explicit arrays")
        options[key] = tuple(options[key])
    config = AdaptiveContactConfig(**options)
    if explicit_scenarios is not None and tuple(explicit_scenarios) != config.mu_hypotheses:
        raise ValueError("Explicit predictor scenarios conflict with the adaptive model bank")
    return config


class ScalarIdentificationModel:
    """Observe actual COM velocity; predict only the previous applied action."""

    def __init__(self, scalar_predictor):
        self.predictor = scalar_predictor
        if scalar_predictor.contract["physics_dt"] != .005 or scalar_predictor.contract["decimation"] != 4:
            raise ValueError("Identification requires the shared 5 ms / four-substep transition")

    def features(self, packed):
        measured = self.predictor.measure(packed)
        linear = np.asarray(measured["body_lin_vel_w"])
        angular = np.asarray(measured["body_ang_vel_w"])
        if any(value.shape != (9, 3) or value.dtype.kind not in "fiu" or not np.isfinite(value).all()
               for value in (linear, angular)):
            raise ValueError("Identification requires finite nine-body COM/angular velocity arrays")
        features = np.r_[linear[0, :2], angular[0, 2]]
        if features.shape != (3,) or not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite measured COM velocity / angular velocity")
        return features

    def transition_features(self, packed, action, *, mu):
        predicted = self.predictor.transition(packed, action, mu=mu)
        return self.features(predicted)


def transition_context(records):
    """Gate identification using exactly the preceding four ground samples.

    This is neither a task-success gate nor a net-force substitute for the
    existing ground-filtered contact recorder.
    """
    if len(records) != 4:
        raise ValueError("Adaptive context requires exactly four actual physics samples")
    times = np.array([row["time_s"] for row in records], dtype=float)
    wheel = np.array([row["wheel_ground_contact_count"] for row in records])
    nonwheel = np.array([row["nonwheel_ground_contact_count"] for row in records])
    applied = np.array([row["action_was_applied"] for row in records])
    if (times.shape != (4,) or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), .005, atol=1e-9, rtol=0)
            or wheel.shape != (4, 2) or nonwheel.shape != (4,)
            or wheel.dtype.kind not in "iu" or nonwheel.dtype.kind not in "iu"
            or np.any(wheel < 0) or np.any(nonwheel < 0)
            or applied.shape != (4,) or applied.dtype.kind != "b" or not applied.all()):
        raise ValueError("Invalid preceding measured ground-contact / clock evidence")
    return {"wheel_ground_contact_observed": bool(np.any(wheel > 0)),
            "nonwheel_ground_contact_observed": bool(np.any(nonwheel > 0)), "duration_s": .02}


def adaptation_summary(decisions):
    observations = [row["solver"]["adaptive_contact"] for row in decisions]
    elapsed = np.array([row["estimation_compute_s"] for row in observations], dtype=float)
    return {"decisions": len(observations),
            "status_counts": dict(Counter(row["status"] for row in observations)),
            "selected_model_counts": dict(Counter(str(row["selected_mu"]) for row in observations)),
            "model_switches": sum(bool(row["updated"]) for row in observations),
            "identification_prediction_calls": sum(row["identification_prediction_calls"] for row in observations),
            "identification_predicted_physics_substeps": sum(row["identification_physics_substeps"] for row in observations),
            "identification_compute_ms_median_p95_max": (1000 * np.quantile(elapsed, [.5, .95, 1.])).tolist(),
            "identification_backend": "existing_python_scalar_cold_reset_transition",
            "included_in_controller_compute_s": True, "physical_friction_estimate_claimed": False,
            "robustness_certificate": False}
