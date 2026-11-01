"""Bounded finite-difference SLSQP shooting on the shared TRON1 predictor.

This is a development baseline, not a real-time or smooth-contact claim.  It
optimizes block residuals around a fixed, physics-only full-plan seed.  Every
cost and finite-difference perturbation is a complete physical rollout; the
native batch predictor is optional and never supplies hidden derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SQPConfig:
    horizon_steps: int = 30
    control_blocks: int = 10
    max_iterations: int = 3
    finite_difference_step: float = 0.002
    seed: int = 42  # retained for an explicit, common experiment contract

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.horizon_steps, self.control_blocks, self.max_iterations)):
            raise ValueError("SQP counts must be positive integers")
        if self.control_blocks > self.horizon_steps:
            raise ValueError("Control blocks exceed horizon")
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0:
            raise ValueError("Finite difference step must be positive and finite")


class SQPController:
    """Receding-horizon SLSQP with batched forward-difference gradients."""

    def __init__(self, predictor, config: SQPConfig, *, action_low, action_high,
                 mu_scenarios=(0.7,)):
        if len(mu_scenarios) != 1 or not np.isfinite(mu_scenarios[0]) or mu_scenarios[0] < 0:
            raise ValueError("This SQP pilot uses one fixed prediction friction")
        self.predictor, self.config = predictor, config
        self.mu_scenarios = (float(mu_scenarios[0]),)
        self.low, self.high = np.asarray(action_low, float), np.asarray(action_high, float)
        if (self.low.ndim != 1 or self.low.shape != self.high.shape
                or not np.isfinite(self.low).all() or not np.isfinite(self.high).all()
                or np.any(self.low >= self.high)):
            raise ValueError("Invalid action bounds")
        self.nu = len(self.low)
        self.assignment = np.arange(config.horizon_steps) * config.control_blocks // config.horizon_steps
        self.reset()

    def reset(self):
        self._last_plan = None
        self._last_index = None

    def _score(self, state, plans, reference_index, previous_action, previous_vz):
        plans = np.asarray(plans, float)
        if plans.ndim != 3 or plans.shape[1:] != (self.config.horizon_steps, self.nu):
            raise ValueError("Expected batch of full control plans")
        mu = self.mu_scenarios[0]
        batch = getattr(self.predictor, "rollout_batch", None)
        if callable(batch):
            values = np.asarray(batch(state.copy(), plans.copy(), reference_index, mu=mu,
                previous_action=previous_action.copy(), previous_vz=previous_vz), float)
        else:
            values = np.asarray([self.predictor.rollout(state.copy(), p.copy(), reference_index,
                mu=mu, previous_action=previous_action.copy(), previous_vz=previous_vz)
                for p in plans], float)
        if values.shape != (len(plans),):
            raise ValueError("Predictor returned malformed cost batch")
        return values

    def action(self, state, reference_index: int, *, previous_action, previous_vz: float):
        started = time.perf_counter()
        state, previous_action = np.asarray(state, float), np.asarray(previous_action, float)
        if (state.ndim != 1 or not np.isfinite(state).all() or previous_action.shape != (self.nu,)
                or not np.isfinite(previous_action).all() or not np.isfinite(previous_vz)):
            raise ValueError("Invalid current physical state or previous action")
        if type(reference_index) is not int or reference_index < 0:
            raise ValueError("Invalid reference index")
        fallback = np.clip(previous_action, self.low, self.high)
        seeds = [("previous_action", np.tile(fallback, (self.config.horizon_steps, 1)))]
        if self._last_plan is not None and reference_index > self._last_index:
            shift = min(reference_index-self._last_index, self.config.horizon_steps)
            seeds.append(("shifted_previous_plan", np.concatenate((
                self._last_plan[shift:], np.repeat(self._last_plan[-1:], shift, axis=0)))))
        generator = getattr(self.predictor, "initial_candidate", None)
        if callable(generator):
            generated = np.asarray(generator(state.copy(), reference_index,
                previous_action=previous_action.copy(), previous_vz=previous_vz,
                horizon_steps=self.config.horizon_steps), float)
            if generated.shape != (self.config.horizon_steps, self.nu) or not np.isfinite(generated).all():
                raise ValueError("Malformed physics-only initial candidate")
            seeds.append(("physics_only_initial_candidate", np.clip(generated, self.low, self.high)))
        seed_plans = np.stack([plan for _, plan in seeds])
        seed_costs = self._score(state, seed_plans, reference_index, previous_action, previous_vz)
        finite_seed = np.where(np.isfinite(seed_costs), seed_costs, np.inf)
        seed_index = int(np.argmin(finite_seed))
        baseline = seed_plans[seed_index].copy()
        best_plan, best_cost = baseline.copy(), float(finite_seed[seed_index])
        evaluations = len(seed_plans)
        cache_x, cache_y = None, None

        def plans_for(vectors):
            z = np.asarray(vectors, float).reshape(-1, self.config.control_blocks, self.nu)
            return np.clip(baseline[None] + z[:, self.assignment], self.low, self.high)

        def value(vector):
            nonlocal cache_x, cache_y, evaluations, best_plan, best_cost
            vector = np.asarray(vector, float)
            if cache_x is not None and np.array_equal(vector, cache_x):
                return cache_y
            plan = plans_for(vector[None])[0]
            cost = float(self._score(state, plan[None], reference_index, previous_action, previous_vz)[0])
            evaluations += 1
            if np.isfinite(cost) and cost < best_cost:
                best_cost, best_plan = cost, plan.copy()
            cache_x, cache_y = vector.copy(), cost if np.isfinite(cost) else 1e9
            return cache_y

        def gradient(vector):
            nonlocal evaluations, best_plan, best_cost
            vector = np.asarray(vector, float)
            center = value(vector)
            step = self.config.finite_difference_step
            perturbed = np.repeat(vector[None], len(vector), axis=0)
            indices = np.arange(len(vector))
            # One-sided inward differences at a box boundary.
            signs = np.where(vector+step > 2.0, -1.0, 1.0)
            perturbed[indices, indices] += signs*step
            plans = plans_for(perturbed)
            costs = self._score(state, plans, reference_index, previous_action, previous_vz)
            evaluations += len(plans)
            finite = np.isfinite(costs)
            if np.any(finite):
                index = int(np.argmin(np.where(finite, costs, np.inf)))
                if costs[index] < best_cost:
                    best_cost, best_plan = float(costs[index]), plans[index].copy()
            costs = np.where(finite, costs, 1e9)
            derivative = (costs-center)/(signs*step)
            return np.clip(derivative, -1e6, 1e6)

        dimension = self.config.control_blocks*self.nu
        result = minimize(value, np.zeros(dimension), method="SLSQP", jac=gradient,
            bounds=[(-2.0, 2.0)]*dimension,
            options={"maxiter": self.config.max_iterations, "ftol": 1e-4, "disp": False})
        failed = not np.isfinite(best_cost)
        selected = fallback if failed else best_plan[0].copy()
        self._last_plan = None if failed else best_plan.copy()
        self._last_index = None if failed else reference_index
        return selected, {
            "optimizer": "scipy_slsqp_batched_finite_difference",
            "optimization_failed": failed,
            "slsqp_success": bool(result.success),
            "slsqp_status": int(result.status),
            "slsqp_message": str(result.message),
            "slsqp_iterations": int(result.nit),
            "candidate_evaluations": evaluations,
            "initial_seed_source": seeds[seed_index][0],
            "initial_cost": None if not np.isfinite(finite_seed[seed_index]) else float(finite_seed[seed_index]),
            "best_cost": None if failed else best_cost,
            "elapsed_s": time.perf_counter()-started,
            "prediction_mu_scenarios": list(self.mu_scenarios),
            "full_plan_warm_start": True,
            "finite_difference_step": self.config.finite_difference_step,
            "fallback_used": failed,
        }
