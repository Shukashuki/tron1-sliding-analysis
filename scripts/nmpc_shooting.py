"""Bounded, derivative-free shooting NMPC with fixed friction scenarios.

This module has no learned-policy or TVLQR dependency. The predictor is isolated
from the validation plant and owns the complete action-to-physics transition.
The controller never changes predictor attributes to select a friction value.

Required predictor interface (unless ``rollout`` is supplied)::

    transition(state, action, *, mu) -> next_state
    stage_cost(state, action, next_state, reference_index,
               *, previous_action, previous_vz) -> float
    vertical_velocity(state) -> float

The stage reference is HELD at ``start_index + step`` while the action is
applied. Both pre- and post-action states are supplied so the objective can
freeze any robot-relative targets using the pre-action state, just as in
training. Optional ``terminal_cost(state, start_index + horizon)`` is added.

An optional ``rollout(state, full_controls, start_index, *, mu,
previous_action, previous_vz) -> float`` can replace that loop, but must include
the same held-reference convention and any terminal cost itself. Optional
``rollout_batch(state, controls[B,H,nu], start_index, *, mu,
previous_action, previous_vz) -> costs[B]`` evaluates independent candidate
plans in their original order. It is used for populations (and singleton
initial plans) only when explicitly supplied. A runtime failure of that call
rejects its entire candidate batch for that scenario; malformed shapes or
programming ValueErrors propagate instead of becoming silent fallbacks.
Optional
``initial_candidate(state, start_index, *, previous_action, previous_vz,
horizon_steps)`` supplies a deterministic physics-only full control plan.

Finite fixed scenarios are not a continuous-uncertainty robustness certificate.
CEM is not an SQP/RTI solver and its timing metadata is not a real-time claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import json
import time
from typing import Literal

import numpy as np


def _bounded_native_error_diagnostics(value):
    """Copy only bounded native-owned primitive fields, never user hooks."""
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
class CEMConfig:
    horizon_steps: int = 15
    control_blocks: int = 5
    candidates: int = 64
    iterations: int = 3
    elite_count: int = 8
    initial_std: float | tuple[float, ...] = 0.25
    min_std: float | tuple[float, ...] = 0.01
    smoothing: float = 0.0
    seed: int = 42
    scenario_reduction: Literal["worst", "mean"] = "worst"
    fallback_action: Literal["previous", "neutral"] = "previous"
    reference_convention: str = "held_pre_action"
    parameterization: Literal["absolute_blocks", "warm_start_residual_blocks"] = "absolute_blocks"

    def __post_init__(self):
        for name in ("horizon_steps", "control_blocks", "candidates", "iterations", "elite_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.control_blocks > self.horizon_steps:
            raise ValueError("control_blocks cannot exceed horizon_steps")
        if self.candidates < 2:
            raise ValueError("candidates must include an incumbent and at least one sample")
        if self.elite_count > self.candidates:
            raise ValueError("elite_count cannot exceed candidates")
        if not 0 <= self.smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        if self.scenario_reduction not in ("worst", "mean"):
            raise ValueError("scenario_reduction must be 'worst' or 'mean'")
        if self.fallback_action not in ("previous", "neutral"):
            raise ValueError("fallback_action must be 'previous' or 'neutral'")
        if self.reference_convention != "held_pre_action":
            raise ValueError("Only the held_pre_action reference convention is supported")
        if self.parameterization not in ("absolute_blocks", "warm_start_residual_blocks"):
            raise ValueError("Unknown CEM action parameterization")


class ShootingNMPC:
    """Receding-horizon CEM over bounded decoded control sequences.

    Default parameters are absolute blocks; optional residual blocks use a
    copied full-plan baseline held fixed for the duration of each solve.

    All friction scenarios evaluate exactly the same full candidate sequence.
    This is an open-loop scenario plan with feedback supplied by replanning,
    not an anticipative scenario tree or an oracle-friction controller.
    """

    def __init__(self, predictor, config: CEMConfig, *, action_low, action_high,
                 mu_scenarios: tuple[float, ...]):
        self.predictor = predictor
        self.config = config
        self.low = np.asarray(action_low, dtype=float).copy()
        self.high = np.asarray(action_high, dtype=float).copy()
        if (self.low.ndim != 1 or self.low.size == 0 or self.high.shape != self.low.shape
                or not np.isfinite(self.low).all() or not np.isfinite(self.high).all()
                or np.any(self.low >= self.high)):
            raise ValueError("action bounds must be finite, matching vectors with low < high")
        self.nu = self.low.size
        self.mu_scenarios = tuple(float(mu) for mu in mu_scenarios)
        if (not self.mu_scenarios or not np.isfinite(self.mu_scenarios).all()
                or any(mu < 0 for mu in self.mu_scenarios)):
            raise ValueError("mu_scenarios must be nonempty, finite and nonnegative")
        if len(set(self.mu_scenarios)) != len(self.mu_scenarios):
            raise ValueError("mu_scenarios must not contain duplicates")
        self.initial_std = self._std_vector(config.initial_std, "initial_std")
        self.min_std = self._std_vector(config.min_std, "min_std")
        self.assignment = np.arange(config.horizon_steps) * config.control_blocks // config.horizon_steps
        self.residual_span = None
        if config.parameterization == "warm_start_residual_blocks":
            with np.errstate(over="ignore", invalid="ignore"):
                span = self.high-self.low
                reach_high, reach_low = self.low+span, self.high-span
            if (not np.isfinite(span).all() or np.any(span <= 0)
                    or not np.isfinite(reach_high).all() or not np.isfinite(reach_low).all()
                    or np.any(reach_high < self.high) or np.any(reach_low > self.low)):
                raise ValueError("Residual encoding requires finite action span with exact endpoint saturation")
            self.residual_span = span
        self.reset()

    def parameterization_contract(self):
        residual = self.config.parameterization == "warm_start_residual_blocks"
        return {
            "mode": self.config.parameterization,
            "parameter_units": "Normalized absolute actions" if not residual else "Normalized action residuals",
            "baseline": ("Winning bounded initial FULL plan, or bounded fallback when all initial scores fail; fresh after each warm shift"
                         if residual else None),
            "baseline_copied_and_fixed_per_solve": residual,
            "decode": ("clip(fixed_full_baseline + block_residual[assignment], action_low, action_high)"
                       if residual else "clip(absolute_block_parameters, action_low, action_high)[assignment]"),
            "zero_residual": "Exactly represents the initial baseline, not a later improved incumbent" if residual else None,
            "incumbent": "Paired stored best encoded residual and decoded full plan" if residual else "Stored bounded full plan",
            "elite_moments": "Encoded clipped residual parameters; never re-compress decoded/clipped actions" if residual else "Block-compressed elite absolute actions",
            "residual_parameter_clip_low": (-self.residual_span).tolist() if residual else None,
            "residual_parameter_clip_high": self.residual_span.tolist() if residual else None,
            "finite_validation": ("Before residual encoding clip, before decoded action clip, and for elite moment arithmetic; invalid arithmetic raises"
                                  if residual else "Existing absolute-block behavior preserved"),
            "generic_bounds": ("New mode rejects nonfinite spans and rounded spans failing endpoint saturation; decoded sums must still be finite"
                               if residual else "Existing absolute bounds contract unchanged"),
            "learned_baseline": False,
        }

    def _encode_residual(self, value):
        value = np.asarray(value, dtype=float)
        if (self.residual_span is None or value.ndim != 3 or value.shape[0] < 1
                or value.shape[1:] != (self.config.control_blocks, self.nu)):
            raise ValueError("Residual parameters require (batch,control_blocks,action_dimension)")
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite residual proposal before encoding clip")
        return np.clip(value, -self.residual_span, self.residual_span)

    def _decode_residual(self, baseline, encoded):
        baseline = np.asarray(baseline, dtype=float)
        encoded = np.asarray(encoded, dtype=float)
        if (baseline.shape != (self.config.horizon_steps, self.nu)
                or not np.isfinite(baseline).all() or np.any(baseline < self.low) or np.any(baseline > self.high)):
            raise ValueError("Residual baseline must be a finite bounded FULL plan")
        checked = self._encode_residual(encoded)
        if not np.array_equal(checked, encoded):
            raise ValueError("Decode requires the stored already-clipped residual encoding")
        with np.errstate(over="raise", invalid="raise"):
            proposed = baseline[None] + encoded[:, self.assignment]
        if not np.isfinite(proposed).all():
            raise FloatingPointError("Nonfinite decoded residual sum before action clip")
        return np.clip(proposed, self.low, self.high), proposed

    def _residual_moments(self, elite, mean, std):
        if not np.isfinite(elite).all():
            raise FloatingPointError("Nonfinite encoded residual elite")
        with np.errstate(over="raise", invalid="raise"):
            smoothing = self.config.smoothing
            new_mean = smoothing*mean + (1-smoothing)*elite.mean(axis=0)
            new_std = np.maximum(smoothing*std + (1-smoothing)*elite.std(axis=0), self.min_std)
        if not np.isfinite(new_mean).all() or not np.isfinite(new_std).all():
            raise FloatingPointError("Nonfinite residual elite moments")
        return new_mean, new_std

    def _std_vector(self, value, name):
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(self.nu, float(values))
        if values.shape != (self.nu,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{name} must be positive and finite, scalar or one value per action")
        return values.copy()

    def reset(self):
        """Reset the warm plan and seeded random stream between episodes."""
        self.rng = np.random.default_rng(self.config.seed)
        self._last_plan = None
        self._last_reference_index = None

    @property
    def last_plan(self):
        return None if self._last_plan is None else self._last_plan.copy()

    def _plan(self, value):
        plan = np.asarray(value, dtype=float)
        if plan.shape != (self.config.horizon_steps, self.nu) or not np.isfinite(plan).all():
            raise ValueError("a control plan must be a finite (horizon_steps, action_dimension) array")
        return np.clip(plan, self.low, self.high).copy()

    def _compress(self, full_plan):
        return np.stack([full_plan[self.assignment == block].mean(axis=0)
                         for block in range(self.config.control_blocks)])

    def _rollout(self, state, plan, reference_index, *, mu, previous_action, previous_vz):
        p = self.predictor
        if callable(getattr(p, "rollout", None)):
            return float(p.rollout(state.copy(), plan.copy(), reference_index, mu=mu,
                                   previous_action=previous_action.copy(), previous_vz=previous_vz))
        x = state.copy()
        previous = previous_action.copy()
        vz = previous_vz
        cost = 0.0
        for step, action in enumerate(plan):
            before = x.copy()
            x = np.asarray(p.transition(before.copy(), action.copy(), mu=mu), dtype=float)
            if x.shape != state.shape or not np.isfinite(x).all():
                raise FloatingPointError("nonfinite or mismatched predictor state")
            cost += float(p.stage_cost(before, action.copy(), x.copy(), reference_index + step,
                                       previous_action=previous.copy(), previous_vz=vz))
            previous = action.copy()
            vz = float(p.vertical_velocity(x.copy()))
            if not np.isfinite(cost) or not np.isfinite(vz):
                raise FloatingPointError("nonfinite predictor stage cost or vertical velocity")
        if callable(getattr(p, "terminal_cost", None)):
            cost += float(p.terminal_cost(x.copy(), reference_index + len(plan)))
        return cost

    def action(self, state, reference_index: int, *, previous_action, previous_vz: float,
               seed_plan=None):
        started = time.perf_counter()
        x = np.asarray(state, dtype=float).copy()
        previous = np.asarray(previous_action, dtype=float).copy()
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("state must be a finite vector")
        if previous.shape != (self.nu,) or not np.isfinite(previous).all():
            raise ValueError("previous_action must be a finite action vector")
        if not np.isfinite(previous_vz):
            raise ValueError("previous_vz must be finite")
        if (isinstance(reference_index, bool) or not isinstance(reference_index, (int, np.integer))
                or reference_index < 0):
            raise ValueError("reference_index must be a nonnegative integer")
        reference_index = int(reference_index)
        previous_vz = float(previous_vz)
        counts = {"candidate_evaluations": 0, "finite_candidates": 0, "failed_candidates": 0,
                  "scenario_evaluations": 0, "failed_scenarios": 0}
        scenario_failures = [0] * len(self.mu_scenarios)
        batch_rollout = getattr(self.predictor, "rollout_batch", None)
        use_batch = callable(batch_rollout)
        batch_counts = {"batch_prediction_calls": 0, "failed_batch_prediction_calls": 0}
        prediction_failure_count = 0
        prediction_failures = []

        def record_prediction_failure(error, *, evaluation_mode, mu, candidate_count):
            # Telemetry only: retain the original exception, without changing
            # its catch policy, score, RNG, candidate order or worker lifetime.
            nonlocal prediction_failure_count
            prediction_failure_count += 1
            if len(prediction_failures) >= 8:
                return
            try:
                raw_message = str(error).encode("utf-8", errors="replace")
                message = raw_message[:2048].decode("utf-8", errors="ignore")
                message_truncated = len(raw_message) > 2048
            except Exception:
                message, message_truncated = "<exception message unavailable>", False
            record = {"exception_type": type(error).__name__[:128], "message": message,
                      "message_truncated": message_truncated,
                      "evaluation_mode": evaluation_mode, "mu": float(mu),
                      "reference_index": reference_index,
                      "candidate_count": int(candidate_count),
                      "batch_call_index": batch_counts["batch_prediction_calls"] if evaluation_mode == "batch" else None,
                      "candidate_evaluation_index": counts["candidate_evaluations"]}
            try:
                record["native_diagnostics"] = _bounded_native_error_diagnostics(
                    getattr(error, "diagnostics", None))
            except Exception as diagnostic_error:
                record["native_diagnostics"] = None
                record["diagnostic_capture_error"] = type(diagnostic_error).__name__
            prediction_failures.append(record)

        def score_many(plans):
            # This branch is opt-in. Batch axis is ALWAYS candidate order, not
            # scenario order; every scenario receives the same complete plans.
            number = len(plans)
            counts["candidate_evaluations"] += number
            scenario_costs = np.empty((number, len(self.mu_scenarios)))
            for scenario, mu in enumerate(self.mu_scenarios):
                counts["scenario_evaluations"] += number
                batch_counts["batch_prediction_calls"] += 1
                try:
                    values = np.asarray(batch_rollout(x.copy(), plans.copy(), reference_index,
                        mu=mu, previous_action=previous.copy(), previous_vz=previous_vz), dtype=float)
                except (FloatingPointError, OverflowError, RuntimeError) as error:
                    record_prediction_failure(error, evaluation_mode="batch", mu=mu,
                                              candidate_count=number)
                    values = np.full(number, np.inf)
                    batch_counts["failed_batch_prediction_calls"] += 1
                if values.shape != (number,):
                    raise ValueError(f"rollout_batch must return costs of shape {(number,)}, got {values.shape}")
                bad = ~np.isfinite(values)
                scenario_costs[:, scenario] = np.where(bad, np.inf, values)
                failed = int(np.sum(bad))
                counts["failed_scenarios"] += failed
                scenario_failures[scenario] += failed
            scores = np.full(number, np.inf)
            valid = np.all(np.isfinite(scenario_costs), axis=1)
            if np.any(valid):
                reduced = (np.max(scenario_costs[valid], axis=1)
                           if self.config.scenario_reduction == "worst"
                           else np.mean(scenario_costs[valid].astype(np.longdouble), axis=1))
                scores[valid] = reduced
            scores[~np.isfinite(scores)] = np.inf
            finite = int(np.sum(np.isfinite(scores)))
            counts["finite_candidates"] += finite
            counts["failed_candidates"] += number-finite
            return scores, scenario_costs

        def score(plan):
            if use_batch:
                scores, costs = score_many(plan[None])
                return float(scores[0]), costs[0].tolist()
            counts["candidate_evaluations"] += 1
            scenario_costs = []
            for scenario, mu in enumerate(self.mu_scenarios):
                counts["scenario_evaluations"] += 1
                try:
                    value = self._rollout(x, plan, reference_index, mu=mu,
                                          previous_action=previous, previous_vz=previous_vz)
                except (FloatingPointError, OverflowError, RuntimeError) as error:
                    record_prediction_failure(error, evaluation_mode="scalar", mu=mu,
                                              candidate_count=1)
                    value = np.inf
                if not np.isfinite(value):
                    value = np.inf
                    counts["failed_scenarios"] += 1
                    scenario_failures[scenario] += 1
                scenario_costs.append(float(value))
            if not np.isfinite(scenario_costs).all():
                counts["failed_candidates"] += 1
                return np.inf, scenario_costs
            reduced = (max(scenario_costs) if self.config.scenario_reduction == "worst"
                       else np.mean(np.asarray(scenario_costs, dtype=np.longdouble)))
            if not np.isfinite(reduced):
                counts["failed_candidates"] += 1
                return np.inf, scenario_costs
            counts["finite_candidates"] += 1
            return float(reduced), scenario_costs

        neutral = np.clip(np.zeros(self.nu), self.low, self.high)
        fallback = np.clip(previous, self.low, self.high) if self.config.fallback_action == "previous" else neutral
        repeated = np.tile(fallback, (self.config.horizon_steps, 1))
        initial = [("bounded_" + self.config.fallback_action, repeated)]
        warm_start_used = self._last_plan is not None and reference_index > self._last_reference_index
        if warm_start_used:
            shift = min(reference_index - self._last_reference_index, self.config.horizon_steps)
            shifted = np.concatenate((self._last_plan[shift:], np.repeat(self._last_plan[-1:], shift, axis=0)))
            initial.append(("shifted_previous_plan", shifted))
        elif seed_plan is not None:
            initial.append(("explicit_physics_only_seed", self._plan(seed_plan)))
        candidate_generation_failed = False
        if callable(getattr(self.predictor, "initial_candidate", None)):
            try:
                generated = self.predictor.initial_candidate(
                    x.copy(), reference_index, previous_action=previous.copy(),
                    previous_vz=previous_vz, horizon_steps=self.config.horizon_steps)
                initial.append(("physics_only_initial_candidate", self._plan(generated)))
            except (FloatingPointError, OverflowError, RuntimeError, ValueError):
                candidate_generation_failed = True

        # Optional ADDITIVE named candidates. Never replace the previous,
        # shifted-full-plan or existing bias-only candidate. Candidate plans
        # are evaluated from x under ALL scenarios below, not replayed.
        additional_hook = getattr(self.predictor, "initial_candidates", None)
        additional_hook_used = callable(additional_hook)
        additional_generation_s = None
        additional_generation_failures = []
        if additional_hook_used:
            generation_started = time.perf_counter()
            try:
                generated = additional_hook(x.copy(), reference_index,
                    previous_action=previous.copy(), previous_vz=previous_vz,
                    horizon_steps=self.config.horizon_steps)
                if not isinstance(generated, Mapping):
                    raise ValueError("initial_candidates must return an ordered mapping of names to full plans")
                reserved = {name for name, _ in initial} | {
                    "bounded_previous", "bounded_neutral", "shifted_previous_plan",
                    "explicit_physics_only_seed", "physics_only_initial_candidate",
                    "all_initial_candidates_failed", "initial_candidates_hook"}
                for name, value in generated.items():
                    try:
                        if not isinstance(name, str) or not name or name in reserved:
                            raise ValueError("Additional candidate name is empty, non-string or collides with an existing source")
                        plan = self._plan(value)
                        initial.append((name, plan))
                        reserved.add(name)
                    except (FloatingPointError, OverflowError, RuntimeError, ValueError) as error:
                        additional_generation_failures.append({"source": str(name), "error_type": type(error).__name__, "reason": str(error)})
            except (FloatingPointError, OverflowError, RuntimeError, ValueError) as error:
                additional_generation_failures.append({"source": "initial_candidates_hook", "error_type": type(error).__name__, "reason": str(error)})
            additional_generation_s = float(time.perf_counter()-generation_started)
            candidate_generation_failed = candidate_generation_failed or bool(additional_generation_failures)

        best_plan, best_cost, best_costs = repeated.copy(), np.inf, None
        initial_source = "all_initial_candidates_failed"
        initial_evaluations = []
        for source, plan in initial:
            cost, costs = score(plan)
            initial_evaluations.append({"source": source,
                "model_cost": float(cost) if np.isfinite(cost) else None,
                "scenario_costs": [float(value) if np.isfinite(value) else None for value in costs],
                "generation_failed": False})
            if cost < best_cost:
                best_plan, best_cost, best_costs = plan.copy(), cost, costs
                initial_source = source
        for evaluation in initial_evaluations:
            evaluation["winning_initial_seed"] = evaluation["source"] == initial_source
        initial_cost = best_cost
        residual_mode = self.config.parameterization == "warm_start_residual_blocks"
        # This baseline is copied ONCE. Never move it when a later sample wins:
        # that would add old residual means to an already moved best plan.
        baseline = best_plan.copy() if residual_mode else None
        best_parameters = np.zeros((self.config.control_blocks, self.nu)) if residual_mode else None
        mean = np.zeros_like(best_parameters) if residual_mode else self._compress(best_plan)
        with np.errstate(over="ignore", invalid="ignore"):
            projection = self._compress(best_plan)[self.assignment]
            projection_error = float(np.max(np.abs(projection-best_plan)))
        if not np.isfinite(projection_error):
            projection_error = None
        std = np.tile(np.maximum(self.initial_std, self.min_std), (self.config.control_blocks, 1))
        best_history = []
        parameter_clip_count = action_clip_count = parameter_count = action_count = 0
        for _ in range(self.config.iterations):
            # Moments live in the declared coordinates: absolute blocks or
            # encoded residuals about the SAME fixed per-solve baseline. Never
            # add a residual mean to a newly moved best-plan base.
            if residual_mode:
                with np.errstate(over="raise", invalid="raise"):
                    proposals = self.rng.normal(size=(self.config.candidates, *mean.shape))*std + mean
                if not np.isfinite(proposals).all():
                    raise FloatingPointError("Nonfinite residual proposal before encoding clip")
                proposals[0] = best_parameters
                parameter_clip_count += int(np.count_nonzero(np.abs(proposals) > self.residual_span))
                parameter_count += proposals.size
                parameters = self._encode_residual(proposals)
                plans, proposed_actions = self._decode_residual(baseline, parameters)
                if not np.array_equal(plans[0], best_plan):
                    raise FloatingPointError("Residual incumbent encoding does not reproduce its stored full plan")
                action_clip_count += int(np.count_nonzero((proposed_actions < self.low) | (proposed_actions > self.high)))
                action_count += proposed_actions.size
            else:
                blocks = np.clip(self.rng.normal(size=(self.config.candidates, *mean.shape)) * std + mean,
                                 self.low, self.high)
                plans = blocks[:, self.assignment].copy()
                plans[0] = best_plan
            if use_batch:
                scores, population_costs = score_many(plans)
                for candidate, plan in enumerate(plans):
                    value = float(scores[candidate])
                    if value < best_cost:
                        best_plan, best_cost, best_costs = plan.copy(), value, population_costs[candidate].tolist()
                        if residual_mode:
                            best_parameters = parameters[candidate].copy()
            else:
                scores = np.empty(self.config.candidates)
                for candidate, plan in enumerate(plans):
                    value, costs = score(plan)
                    scores[candidate] = value
                    if value < best_cost:
                        best_plan, best_cost, best_costs = plan.copy(), value, costs
                        if residual_mode:
                            best_parameters = parameters[candidate].copy()
            finite = np.flatnonzero(np.isfinite(scores))
            if finite.size:
                elite_indices = finite[np.argsort(scores[finite], kind="stable")[:self.config.elite_count]]
                if residual_mode:
                    mean, std = self._residual_moments(parameters[elite_indices], mean, std)
                else:
                    elite_blocks = np.stack([self._compress(plans[index]) for index in elite_indices])
                    smoothing = self.config.smoothing
                    mean = smoothing * mean + (1 - smoothing) * elite_blocks.mean(axis=0)
                    std = np.maximum(smoothing * std + (1 - smoothing) * elite_blocks.std(axis=0), self.min_std)
            best_history.append(float(best_cost) if np.isfinite(best_cost) else None)

        failed = not np.isfinite(best_cost)
        if failed:
            self._last_plan = None
            self._last_reference_index = None
            selected_action = fallback.copy()
        else:
            self._last_plan = best_plan.copy()
            self._last_reference_index = reference_index
            selected_action = best_plan[0].copy()
        model_plan_correction = None
        model_plans = [plan for name, plan in initial if name == "model_nominal_plan"]
        if model_plans:
            nominal = model_plans[0]
            action_delta = selected_action-nominal[0]
            model_plan_correction = {
                "winning_initial_seed_is_model_plan": initial_source == "model_nominal_plan",
                "final_action_equals_model_plan_first_action": bool(np.array_equal(selected_action, nominal[0])),
                "final_action_minus_model_plan_first_action_l2": float(np.linalg.norm(action_delta)),
                "final_action_minus_model_plan_first_action_linf": float(np.max(np.abs(action_delta))),
                "selected_full_plan_minus_model_plan_l2": float(np.linalg.norm(best_plan-nominal)) if not failed else None,
                "selected_full_plan_minus_model_plan_linf": float(np.max(np.abs(best_plan-nominal))) if not failed else None,
                "final_action_is_optimization_fallback": failed,
                "action_precision": "Raw selected CEM action before the runner's float32 conversion; applied actions are logged separately.",
                "units": "Normalized action units; a selected initial seed does not imply final action replay."}
        elapsed = time.perf_counter() - started
        info = {
            "optimizer": "bounded_cem_shooting",
            "reference_convention": self.config.reference_convention,
            "prediction_mu_scenarios": list(self.mu_scenarios),
            "scenario_reduction": self.config.scenario_reduction,
            "shared_full_control_sequence": True,
            "robustness_certificate": False,
            "optimization_failed": failed,
            "fallback_used": failed,
            "fallback_kind": self.config.fallback_action if failed else None,
            "learned_policy_fallback": False,
            "initial_candidate_generation_failed": candidate_generation_failed,
            "additional_initial_candidates_hook_used": additional_hook_used,
            "additional_initial_candidate_generation_s": additional_generation_s,
            "additional_initial_candidate_generation_failures": additional_generation_failures,
            "initial_candidate_evaluations": initial_evaluations,
            "initial_candidate_source": initial_source,
            "initial_candidate_source_semantics": "Winning INITIAL seed before CEM search, not the provenance of the final selected CEM action/plan.",
            "model_nominal_plan_online_correction": model_plan_correction,
            "warm_start_used": warm_start_used,
            "initial_model_cost": float(initial_cost) if np.isfinite(initial_cost) else None,
            "selected_model_cost": float(best_cost) if not failed else None,
            "selected_scenario_costs": best_costs,
            "predicted_cost_nonincreasing": bool(best_cost <= initial_cost) if np.isfinite(initial_cost) else None,
            "best_cost_by_iteration": best_history,
            "failed_scenarios_by_mu": scenario_failures,
            "horizon_steps": self.config.horizon_steps,
            "control_blocks": self.config.control_blocks,
            "compute_time_s": float(elapsed),
            "candidate_evaluations_per_s": counts["candidate_evaluations"] / elapsed if elapsed > 0 else None,
            "scenario_rollouts_per_s": counts["scenario_evaluations"] / elapsed if elapsed > 0 else None,
            "timing_scope": "controller_including_predictor_rollouts",
            "control_deadline_enforced": False,
            "real_time_verified": False,
            "action_parameterization": self.config.parameterization,
            "residual_baseline_fixed_per_solve": residual_mode,
            "initial_baseline_block_projection_max_abs_error": projection_error,
            "residual_parameter_clip_fraction": parameter_clip_count/parameter_count if residual_mode else None,
            "residual_action_decode_clip_fraction": action_clip_count/action_count if residual_mode else None,
            "residual_selected_parameter_linf": float(np.max(np.abs(best_parameters))) if residual_mode else None,
            "population_evaluation_backend": "predictor_rollout_batch" if use_batch else "scalar_candidate_loop",
            "batch_runtime_failure_scope": "All candidates in the failed scenario call" if use_batch else None,
            "prediction_failure_count": prediction_failure_count,
            "prediction_failures": prediction_failures,
            "prediction_failures_truncated": prediction_failure_count > len(prediction_failures),
            "prediction_failure_semantics": "Caught prediction exceptions, not failed candidates; first 8 records per solve. Message <=2048 UTF-8 bytes; native primitive diagnostics <=8192 JSON bytes. Nonfinite-cost and candidate/scenario counters retain separate meanings.",
            **batch_counts,
            **counts,
        }
        return selected_action, info
