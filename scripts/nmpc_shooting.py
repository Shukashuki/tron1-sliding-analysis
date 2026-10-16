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
``initial_candidate(state, start_index, *, previous_action, previous_vz,
horizon_steps)`` supplies a deterministic physics-only full control plan.

Finite fixed scenarios are not a continuous-uncertainty robustness certificate.
CEM is not an SQP/RTI solver and its timing metadata is not a real-time claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np


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


class ShootingNMPC:
    """Receding-horizon CEM over bounded absolute control sequences.

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
        self.reset()

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

        def score(plan):
            counts["candidate_evaluations"] += 1
            scenario_costs = []
            for scenario, mu in enumerate(self.mu_scenarios):
                counts["scenario_evaluations"] += 1
                try:
                    value = self._rollout(x, plan, reference_index, mu=mu,
                                          previous_action=previous, previous_vz=previous_vz)
                except (FloatingPointError, OverflowError, RuntimeError):
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

        best_plan, best_cost, best_costs = repeated.copy(), np.inf, None
        initial_source = "all_initial_candidates_failed"
        for source, plan in initial:
            cost, costs = score(plan)
            if cost < best_cost:
                best_plan, best_cost, best_costs = plan.copy(), cost, costs
                initial_source = source
        initial_cost = best_cost
        mean = self._compress(best_plan)
        std = np.tile(np.maximum(self.initial_std, self.min_std), (self.config.control_blocks, 1))
        best_history = []
        for _ in range(self.config.iterations):
            # CEM moments are ABSOLUTE bounded controls. Never add an old noise
            # mean to a newly moved best-plan base, which double-counts updates.
            blocks = np.clip(self.rng.normal(size=(self.config.candidates, *mean.shape)) * std + mean,
                             self.low, self.high)
            plans = blocks[:, self.assignment].copy()
            plans[0] = best_plan
            scores = np.empty(self.config.candidates)
            for candidate, plan in enumerate(plans):
                value, costs = score(plan)
                scores[candidate] = value
                if value < best_cost:
                    best_plan, best_cost, best_costs = plan.copy(), value, costs
            finite = np.flatnonzero(np.isfinite(scores))
            if finite.size:
                elite_indices = finite[np.argsort(scores[finite], kind="stable")[:self.config.elite_count]]
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
            "initial_candidate_source": initial_source,
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
            **counts,
        }
        return selected_action, info
