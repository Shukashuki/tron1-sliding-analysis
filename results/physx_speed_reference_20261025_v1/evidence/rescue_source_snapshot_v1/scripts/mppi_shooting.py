"""Fixed-covariance MPPI over unbounded Gaussian block residuals.

The physical decoder clamps actions, not the Gaussian latent samples.  The
importance ratio and weighted-mean update therefore use the original latent
noise.  A full initial-plan baseline is copied and held fixed throughout each
solve.  A Gaussian prior around that baseline adds an implicit KL regularizer;
the predictor's existing task rollout cost itself is not modified.

This is neither CEM nor best-sample random shooting: every finite Gaussian
sample participates and the FINAL updated-mean plan is deployed after a fresh
finite-cost rollout, even if its task cost exceeds an initial/sample cost.
No learned policy, online model selection, covariance fitting, terminal value,
real-time deadline, or robustness certificate is provided here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Literal

import numpy as np


def _native_diagnostics(value):
    """Allow only the same bounded, native-owned primitive telemetry as CEM."""
    if value is None:
        return None
    fields = {"event", "native_return_code", "child_exception_type", "ipc_exception_type",
              "parent_stop_initiated", "timeout_s", "mujoco_log_tail", "worker", "before_stop", "after_stop"}
    status_fields = {"alive", "closed", "pid", "exitcode"}

    def primitive(item):
        if item is None or type(item) in (bool, int):
            return item
        if type(item) is float and np.isfinite(item):
            return item
        if type(item) is str:
            return item.encode("utf-8", errors="replace")[:2048].decode("utf-8", errors="ignore")
        raise TypeError("Native diagnostics require finite built-in primitives")

    if type(value) is not dict or any(type(k) is not str or k not in fields for k in value):
        raise TypeError("Native diagnostics contain unsupported fields")
    result = {}
    for key, item in value.items():
        if key in ("worker", "before_stop", "after_stop"):
            if type(item) is not dict or any(type(k) is not str or k not in status_fields for k in item):
                raise TypeError("Native worker diagnostics contain unsupported fields")
            result[key] = {k: primitive(v) for k, v in item.items()}
        else:
            result[key] = primitive(item)
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > 8192:
        raise ValueError("Native diagnostic byte budget exceeded")
    return result


@dataclass(frozen=True)
class MPPIConfig:
    horizon_steps: int = 30
    control_blocks: int = 10
    samples: int = 63
    iterations: int = 3
    temperature: float = 1.0
    noise_std: float | tuple[float, ...] = (.04, .04, .04, .04, .04, .04, .35, .35)
    seed: int = 42
    reference_convention: str = "held_pre_action"
    fallback_action: Literal["previous", "neutral"] = "previous"

    def __post_init__(self):
        for name in ("horizon_steps", "control_blocks", "samples", "iterations"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.control_blocks > self.horizon_steps:
            raise ValueError("control_blocks cannot exceed horizon_steps")
        if (isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer))
                or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        if (isinstance(self.temperature, (bool, np.bool_)) or not np.isscalar(self.temperature)
                or not np.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("temperature must be a positive finite scalar")
        noise = np.asarray(self.noise_std)
        if noise.dtype.kind == "b" or noise.ndim > 1 or noise.size == 0:
            raise ValueError("noise_std must be a positive finite scalar or vector")
        noise = np.asarray(noise, dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            variance = noise * noise
        if (not np.isfinite(noise).all() or np.any(noise <= 0)
                or not np.isfinite(variance).all() or np.any(variance <= 0)):
            raise ValueError("noise_std requires finite, strictly positive representable variances")
        # A frozen config must not retain a caller-owned mutable ndarray/list.
        object.__setattr__(self, "noise_std", float(noise) if noise.ndim == 0 else tuple(noise.tolist()))
        object.__setattr__(self, "temperature", float(self.temperature))
        if self.reference_convention != "held_pre_action":
            raise ValueError("Only the held_pre_action reference convention is supported")
        if self.fallback_action not in ("previous", "neutral"):
            raise ValueError("fallback_action must be 'previous' or 'neutral'")


def _json_float(value):
    """Extended-precision diagnostic endpoints may exceed JSON float range."""
    value = np.longdouble(value)
    if not np.isfinite(value) or abs(value) > np.finfo(float).max:
        return None
    return float(value)


def _mppi_weights(costs, mean, noise, std, temperature):
    """Return stable self-normalized importance weights in LATENT block space.

    P=N(0,Sigma), Q=N(mean,Sigma), z=mean+noise.  The term
    -mean^T Sigma^-1 noise differs from log(P(z)/Q(z)) only by a
    candidate-independent constant.  No repeated-horizon multiplicity belongs
    in this ratio: each block is sampled exactly once.
    """
    costs = np.asarray(costs, dtype=float)
    noise = np.asarray(noise, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if (mean.ndim != 2 or noise.shape != (len(costs), *mean.shape)
            or costs.ndim != 1 or std.shape != (mean.shape[-1],)
            or not np.isfinite(noise).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Malformed MPPI importance-weight inputs")
    valid = np.isfinite(costs)
    if not np.any(valid):
        return None, None
    # Extended precision avoids overflow at otherwise valid tiny temperatures;
    # only negative shifted exponent arguments may underflow to zero.
    wide = np.longdouble
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        correction = np.sum((mean.astype(wide) / std.astype(wide)) *
                            (noise[valid].astype(wide) / std.astype(wide)), axis=(1, 2))
        log_weights = -costs[valid].astype(wide) / wide(temperature) - correction
        maximum = np.max(log_weights)
        shifted = log_weights - maximum
    with np.errstate(under="ignore", over="ignore"):
        nonzero = np.exp(np.asarray(shifted, dtype=float))
    normalizer = float(nonzero.sum())
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise FloatingPointError("MPPI weight normalizer must be finite and positive")
    weights = np.zeros(len(costs))
    weights[valid] = nonzero / normalizer
    stats = {"raw_cost_min": float(np.min(costs[valid])), "raw_cost_max": float(np.max(costs[valid])),
             "importance_correction_min": _json_float(np.min(correction)),
             "importance_correction_max": _json_float(np.max(correction)),
             "log_weight_max": _json_float(maximum),
             "shifted_weight_normalizer": normalizer,
             "ess": float(1. / np.sum(weights * weights)), "max_weight": float(weights.max()),
             "finite_population_candidates": int(valid.sum())}
    return weights, stats


class MPPIController:
    """Receding-horizon, fixed-single-model latent-residual MPPI controller."""

    def __init__(self, predictor, config: MPPIConfig, *, action_low, action_high, mu_scenarios=(.7,)):
        if not isinstance(config, MPPIConfig):
            raise TypeError("config must be an MPPIConfig")
        low, high = np.asarray(action_low, dtype=float), np.asarray(action_high, dtype=float)
        if (low.ndim != 1 or low.size == 0 or high.shape != low.shape
                or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high)):
            raise ValueError("action bounds must be finite matching vectors with low < high")
        raw_mus = tuple(mu_scenarios)
        if len(raw_mus) != 1 or isinstance(raw_mus[0], (bool, np.bool_)):
            raise ValueError("MPPI pilot requires exactly one fixed prediction friction")
        mus = tuple(float(mu) for mu in raw_mus)
        if not np.isfinite(mus).all() or mus[0] < 0:
            raise ValueError("Prediction friction must be finite and nonnegative")
        self.predictor, self.config = predictor, config
        self.low, self.high = low.copy(), high.copy()
        self.nu, self.mu_scenarios = len(low), mus
        std = np.asarray(config.noise_std, dtype=float)
        self.noise_std = np.full(self.nu, float(std)) if std.ndim == 0 else std.copy()
        if self.noise_std.shape != (self.nu,):
            raise ValueError("noise_std must be scalar or one value per action")
        self.assignment = np.arange(config.horizon_steps) * config.control_blocks // config.horizon_steps
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.config.seed)
        self._last_plan = None
        self._last_reference_index = None

    @property
    def last_plan(self):
        return None if self._last_plan is None else self._last_plan.copy()

    def parameterization_contract(self):
        return {"mode": "unbounded_gaussian_latent_residual_blocks",
                "baseline": "Winning bounded INITIAL full plan, copied and fixed per solve",
                "baseline_copied_and_fixed_per_solve": True,
                "decode": "clip(fixed_full_baseline + latent_blocks[assignment], action_low, action_high)",
                "latent_clip": False, "noise_clip": False,
                "density_and_update_coordinates": "Original unbounded Gaussian latent blocks, never clipped action deltas",
                "density_block_multiplicity": 1,
                "learned_baseline": False}

    def algorithm_contract(self):
        return {"optimizer": "fixed_covariance_latent_residual_mppi",
                "sampling_distribution": "Q=N(mean,diag(noise_std**2)); IID Gaussian blocks",
                "base_distribution": "P=N(0,diag(noise_std**2)) in per-solve baseline residual coordinates",
                "log_weights_up_to_constant": "-task_rollout_cost/temperature - sum_blocks(mean*raw_noise/noise_std**2)",
                "mean_update": "mean + sum_samples(normalized_weights * raw_noise)",
                "temperature": self.config.temperature, "noise_std": self.noise_std.tolist(),
                "temperature_adaptation": False, "covariance_fitting": False, "elite_selection": False,
                "deterministic_initial_plans_in_weighted_population": False,
                "deployed_plan": "FINAL weighted-mean plan, freshly scored; never best sample or best initial plan",
                "per_iteration_rollouts": {"iid_gaussian_samples": self.config.samples, "updated_mean_check": 1},
                "task_rollout_score_modified": False,
                "implicit_regularization": "Gaussian-base KL-to-initial-baseline prior; not identical overall optimization objective to CEM",
                "controller_added_terminal_value": False,
                "predictor_terminal_cost_hook_used_on_scalar_path": (
                    not callable(getattr(self.predictor, "rollout_batch", None))
                    and not callable(getattr(self.predictor, "rollout", None))
                    and callable(getattr(self.predictor, "terminal_cost", None))),
                "plant_friction_oracle": False,
                "robustness_certificate": False, "real_time_verified": False,
                "sources": ["https://arxiv.org/abs/1707.02342", "https://github.com/UM-ARM-Lab/pytorch_mppi"],
                "package_reimplementation_identity_claimed": False}

    def _plan(self, plan):
        value = np.asarray(plan, dtype=float)
        if value.shape != (self.config.horizon_steps, self.nu) or not np.isfinite(value).all():
            raise ValueError("A plan must be a finite (horizon_steps,action_dimension) array")
        return np.clip(value, self.low, self.high).copy()

    def _decode(self, baseline, latent):
        if latent.ndim != 3 or latent.shape[1:] != (self.config.control_blocks, self.nu):
            raise ValueError("Latent controls require (batch,control_blocks,action_dimension)")
        if not np.isfinite(latent).all():
            raise FloatingPointError("Nonfinite Gaussian latent controls before physical decode")
        with np.errstate(over="raise", invalid="raise"):
            proposed = baseline[None] + latent[:, self.assignment]
        return np.clip(proposed, self.low, self.high), float(np.mean((proposed < self.low) | (proposed > self.high)))

    def _rollout(self, state, plan, index, *, mu, previous_action, previous_vz):
        predictor = self.predictor
        if callable(getattr(predictor, "rollout", None)):
            return float(predictor.rollout(state.copy(), plan.copy(), index, mu=mu,
                previous_action=previous_action.copy(), previous_vz=previous_vz))
        x, previous, vz, cost = state.copy(), previous_action.copy(), previous_vz, 0.
        for step, action in enumerate(plan):
            before = x.copy()
            x = np.asarray(predictor.transition(before.copy(), action.copy(), mu=mu), dtype=float)
            if x.shape != state.shape or not np.isfinite(x).all():
                raise FloatingPointError("Nonfinite or mismatched predictor state")
            cost += float(predictor.stage_cost(before, action.copy(), x.copy(), index + step,
                previous_action=previous.copy(), previous_vz=vz))
            previous = action.copy()
            vz = float(predictor.vertical_velocity(x.copy()))
            if not np.isfinite(cost) or not np.isfinite(vz):
                raise FloatingPointError("Nonfinite predictor stage cost or vertical velocity")
        if callable(getattr(predictor, "terminal_cost", None)):
            cost += float(predictor.terminal_cost(x.copy(), index + len(plan)))
        return cost

    def action(self, state, reference_index: int, *, previous_action, previous_vz: float, seed_plan=None):
        started = time.perf_counter()
        x, previous = np.asarray(state, dtype=float).copy(), np.asarray(previous_action, dtype=float).copy()
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("state must be a nonempty finite vector")
        if previous.shape != (self.nu,) or not np.isfinite(previous).all():
            raise ValueError("previous_action must be a finite action vector")
        if not np.isscalar(previous_vz) or not np.isfinite(previous_vz):
            raise ValueError("previous_vz must be a finite scalar")
        if (isinstance(reference_index, (bool, np.bool_)) or not isinstance(reference_index, (int, np.integer))
                or reference_index < 0):
            raise ValueError("reference_index must be a nonnegative integer")
        reference_index, previous_vz = int(reference_index), float(previous_vz)
        # Validate even an unused supplied seed before touching RNG/warm state.
        seed = None if seed_plan is None else self._plan(seed_plan)
        counts = {"candidate_evaluations": 0, "finite_candidates": 0, "failed_candidates": 0,
                  "scenario_evaluations": 0, "failed_scenarios": 0,
                  "batch_prediction_calls": 0, "failed_batch_prediction_calls": 0}
        failures, failure_count = [], 0
        mu = self.mu_scenarios[0]
        batch = getattr(self.predictor, "rollout_batch", None)
        use_batch = callable(batch)

        def record_failure(error, number, *, evaluation_mode=None, candidate_index=None):
            nonlocal failure_count
            failure_count += 1
            if len(failures) >= 8:
                return
            try:
                raw = str(error).encode("utf-8", errors="replace")
                message = raw[:2048].decode("utf-8", errors="ignore")
                truncated = len(raw) > 2048
            except Exception:
                message, truncated = "<exception message unavailable>", False
            mode = evaluation_mode or ("batch" if use_batch else "scalar")
            record = {"exception_type": type(error).__name__[:128], "message": message,
                      "message_truncated": truncated, "evaluation_mode": mode,
                      "mu": mu, "reference_index": reference_index, "candidate_count": number,
                      "batch_call_index": counts["batch_prediction_calls"] if mode == "batch" else None,
                      "candidate_evaluation_index": counts["candidate_evaluations"] if candidate_index is None else candidate_index}
            try:
                record["native_diagnostics"] = _native_diagnostics(getattr(error, "diagnostics", None))
            except Exception as diagnostic_error:
                record["native_diagnostics"] = None
                record["diagnostic_capture_error"] = type(diagnostic_error).__name__
            failures.append(record)

        def score_many(plans):
            number = len(plans)
            counts["candidate_evaluations"] += number
            counts["scenario_evaluations"] += number
            if use_batch:
                counts["batch_prediction_calls"] += 1
                try:
                    values = np.asarray(batch(x.copy(), plans.copy(), reference_index, mu=mu,
                        previous_action=previous.copy(), previous_vz=previous_vz), dtype=float)
                except (FloatingPointError, OverflowError, RuntimeError) as error:
                    record_failure(error, number)
                    values = np.full(number, np.inf)
                    counts["failed_batch_prediction_calls"] += 1
                if values.shape != (number,):
                    raise ValueError(f"rollout_batch must return costs of shape {(number,)}, got {values.shape}")
            else:
                values = np.empty(number)
                for candidate, plan in enumerate(plans):
                    try:
                        values[candidate] = self._rollout(x, plan, reference_index, mu=mu,
                            previous_action=previous, previous_vz=previous_vz)
                    except (FloatingPointError, OverflowError, RuntimeError) as error:
                        record_failure(error, 1,
                            candidate_index=counts["candidate_evaluations"] - number + candidate + 1)
                        values[candidate] = np.inf
            bad = ~np.isfinite(values)
            counts["failed_candidates"] += int(bad.sum())
            counts["failed_scenarios"] += int(bad.sum())
            counts["finite_candidates"] += int((~bad).sum())
            return np.where(bad, np.inf, values)

        neutral = np.clip(np.zeros(self.nu), self.low, self.high)
        fallback = np.clip(previous, self.low, self.high) if self.config.fallback_action == "previous" else neutral
        repeated = np.tile(fallback, (self.config.horizon_steps, 1))
        initial = [("bounded_" + self.config.fallback_action, repeated)]
        warm = self._last_plan is not None and reference_index > self._last_reference_index
        if warm:
            shift = min(reference_index - self._last_reference_index, self.config.horizon_steps)
            shifted = np.concatenate((self._last_plan[shift:], np.repeat(self._last_plan[-1:], shift, axis=0)))
            initial.append(("shifted_previous_plan", shifted))
        elif seed is not None:
            initial.append(("explicit_physics_only_seed", seed))
        generation_failed = False
        if callable(getattr(self.predictor, "initial_candidate", None)):
            try:
                initial.append(("physics_only_initial_candidate", self._plan(self.predictor.initial_candidate(
                    x.copy(), reference_index, previous_action=previous.copy(),
                    previous_vz=previous_vz, horizon_steps=self.config.horizon_steps))))
            except (FloatingPointError, OverflowError, RuntimeError) as error:
                generation_failed = True
                record_failure(error, 0, evaluation_mode="initial_candidate")
        # Programming ValueErrors (including malformed generated plans) are not
        # numerical fallbacks. Optional multi-plan/donor hooks are not enabled.
        baseline, initial_cost, source = repeated.copy(), np.inf, "all_initial_candidates_failed"
        evaluations = []
        for name, plan in initial:
            cost = float(score_many(plan[None])[0])
            evaluations.append({"source": name, "model_cost": cost if np.isfinite(cost) else None,
                                "scenario_costs": [cost if np.isfinite(cost) else None], "generation_failed": False})
            if cost < initial_cost:
                baseline, initial_cost, source = plan.copy(), cost, name
        for evaluation in evaluations:
            evaluation["winning_initial_seed"] = evaluation["source"] == source
        mean = np.zeros((self.config.control_blocks, self.nu))
        history, selected_cost, selected_plan = [], np.inf, None
        failure_reason = None
        for iteration in range(self.config.iterations):
            noise = self.rng.normal(size=(self.config.samples, *mean.shape)) * self.noise_std
            latent = mean[None] + noise
            plans, clip_fraction = self._decode(baseline, latent)
            costs = score_many(plans)
            weights, stats = _mppi_weights(costs, mean, noise, self.noise_std, self.config.temperature)
            if weights is None:
                history.append({"iteration": iteration, "finite_population_candidates": 0,
                                "population_decode_clip_fraction": clip_fraction,
                                "updated_mean_model_cost": None, "failure": "all_population_predictions_failed"})
                failure_reason = "all_population_predictions_failed"
                break
            update = np.einsum("k,kbj->bj", weights, noise)
            mean = mean + update
            mean_plan, mean_clip = self._decode(baseline, mean[None])
            mean_cost = float(score_many(mean_plan)[0])
            history.append({"iteration": iteration, **stats, "update_l2": float(np.linalg.norm(update)),
                            "updated_mean_model_cost": mean_cost if np.isfinite(mean_cost) else None,
                            "population_decode_clip_fraction": clip_fraction,
                            "updated_mean_decode_clip_fraction": mean_clip})
            if not np.isfinite(mean_cost):
                failure_reason = "updated_mean_prediction_failed"
                break
            selected_plan, selected_cost = mean_plan[0].copy(), mean_cost
        failed = failure_reason is not None or selected_plan is None
        if failed:
            self._last_plan, self._last_reference_index = None, None
            action = fallback.copy()
            selected_cost = np.inf
        else:
            self._last_plan, self._last_reference_index = selected_plan.copy(), reference_index
            action = selected_plan[0].copy()
        elapsed = float(time.perf_counter() - started)
        info = {"optimizer": "fixed_covariance_latent_residual_mppi",
                "reference_convention": self.config.reference_convention,
                "prediction_mu_scenarios": list(self.mu_scenarios), "scenario_reduction": "singleton",
                "shared_full_control_sequence": True, "robustness_certificate": False,
                "optimization_failed": failed, "optimization_failure_reason": failure_reason,
                "fallback_used": failed, "fallback_kind": self.config.fallback_action if failed else None,
                "learned_policy_fallback": False, "initial_candidate_generation_failed": generation_failed,
                "initial_candidate_evaluations": evaluations, "initial_candidate_source": source,
                "initial_candidate_source_semantics": "Winning INITIAL full-plan baseline, not final deployed plan",
                "warm_start_used": warm, "initial_model_cost": initial_cost if np.isfinite(initial_cost) else None,
                "selected_model_cost": selected_cost if np.isfinite(selected_cost) else None,
                "selected_scenario_costs": [selected_cost] if np.isfinite(selected_cost) else None,
                "predicted_cost_nonincreasing": bool(selected_cost <= initial_cost) if np.isfinite(initial_cost) and not failed else None,
                "selected_plan_rule": "Final freshly scored weighted-mean plan, not best initial/sample plan",
                "mppi_iterations": history, "temperature": self.config.temperature,
                "noise_std": self.noise_std.tolist(), "horizon_steps": self.config.horizon_steps,
                "control_blocks": self.config.control_blocks, "samples": self.config.samples,
                "iterations_requested": self.config.iterations, "iterations_attempted": len(history),
                "iterations_completed": sum(item.get("updated_mean_model_cost") is not None for item in history),
                "action_parameterization": "unbounded_gaussian_latent_residual_blocks",
                "residual_baseline_fixed_per_solve": True, "implicit_gaussian_kl_prior": True,
                "compute_time_s": elapsed,
                "candidate_evaluations_per_s": counts["candidate_evaluations"] / elapsed if elapsed > 0 else None,
                "scenario_rollouts_per_s": counts["scenario_evaluations"] / elapsed if elapsed > 0 else None,
                "timing_scope": "controller_including_predictor_rollouts", "control_deadline_enforced": False,
                "real_time_verified": False,
                "population_evaluation_backend": "predictor_rollout_batch" if use_batch else "scalar_candidate_loop",
                "batch_runtime_failure_scope": "All candidates in the failed scenario call" if use_batch else None,
                "failed_scenarios_by_mu": [counts["failed_scenarios"]],
                "prediction_failure_count": failure_count, "prediction_failures": failures,
                "prediction_failures_truncated": failure_count > len(failures),
                "prediction_failure_semantics": "Caught prediction exceptions, not failed candidates; first 8 bounded records per solve. No retries/restarts. Nonfinite returned costs have separate counters.",
                **counts}
        return action, info
