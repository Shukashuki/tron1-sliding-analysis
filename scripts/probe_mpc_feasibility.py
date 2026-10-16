"""Short offline NMPC/scenario-optimization probes on the actual TRON1 model.

Not a controller benchmark: no closed-loop rollout, no learned warm start,
no real-time or robust-feasibility claim. Neither model nor old costs are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
import scipy
from scipy.optimize import minimize

from classical_tracking import Predictor, ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yaw_error(actual, target):
    difference = actual - target
    return np.arctan2(np.sin(difference), np.cos(difference))


class ShortProblem:
    """Height-free diagnostic cost, NOT exact parity with the PPO reward."""
    def __init__(self, predictor, frame, horizon, blocks, mus):
        self.p = predictor
        self.frame, self.horizon, self.blocks = frame, horizon, blocks
        self.mus = mus
        self.assignment = np.minimum(np.arange(horizon) * blocks // horizon, blocks - 1)
        self.initial = predictor.targets[frame].copy()
        self.previous_action = predictor.nominal_action[frame].copy()
        self.initial_vz = self.measure_state(self.initial)["root_com_lin_vel_w"][2]
        self.calls = 0
        self.cached_controls = self.cached_costs = None

    def measure_state(self, state):
        p = self.p
        mujoco.mj_resetData(p.model, p.data)
        p.data.qpos[:], p.data.qvel[:] = state[:p.model.nq], state[p.model.nq:]
        mujoco.mj_forward(p.model, p.data)
        return p.measure.measure()

    def seed(self):
        indices = [min(self.frame + int(np.flatnonzero(self.assignment == b)[0]), self.p.n - 1)
                   for b in range(self.blocks)]
        return self.p.nominal_action[indices].reshape(-1)

    def costs(self, controls):
        if self.cached_controls is not None and np.array_equal(controls, self.cached_controls):
            return self.cached_costs
        p = self.p
        plan = np.asarray(controls).reshape(self.blocks, 8)[self.assignment]
        values = []
        original = p.model.geom_friction[:, 0].copy()
        try:
            for mu in self.mus:
                p.model.geom_friction[:, 0] = mu
                state, previous_vz, previous_action = self.initial.copy(), self.initial_vz, self.previous_action
                total = 0.
                for j, action in enumerate(plan):
                    state = p.transition(state, action)
                    actual = p.measure.measure()
                    target = p.reference.sample(min(self.frame + j + 1, p.n - 1))
                    R = p.api.rotation(actual["root_quat_wxyz"])
                    Rt = p.api.rotation(target["root_quat_wxyz"])
                    dyaw = yaw_error(np.arctan2(R[1, 0], R[0, 0]), np.arctan2(Rt[1, 0], Rt[0, 0]))
                    velocity = actual["root_com_lin_vel_w"]
                    speed_error = np.linalg.norm(velocity[:2]) - np.linalg.norm(target["root_com_lin_vel_w"][:2])
                    acceleration = (velocity[2] - previous_vz) / p.dt
                    total += (400. * np.sum((actual["root_pos"][:2] - target["root_pos"][:2])**2)
                              + 20. * speed_error**2 + 120. * dyaw**2
                              + 50. * (1. - R[2, 2])**2
                              + .1 * np.sum((action - previous_action)**2)
                              + .1 * (1. - np.exp(-(acceleration / 5.)**2)))
                    previous_vz, previous_action = velocity[2], action
                values.append(total)
        finally:
            p.model.geom_friction[:, 0] = original
        self.calls += 1
        self.cached_controls = np.asarray(controls).copy()
        self.cached_costs = np.asarray(values)
        if not np.isfinite(self.cached_costs).all():
            raise FloatingPointError("Nonfinite diagnostic horizon cost")
        return self.cached_costs

    def solve(self, maxiter):
        seed = self.seed()
        baseline = self.costs(seed).copy()
        bounds = [(-1., 1.)] * len(seed)
        start = time.perf_counter()
        if len(self.mus) == 1:
            result = minimize(lambda u: float(self.costs(u)[0]), seed, method="SLSQP",
                              bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-5})
            controls = result.x
            epigraph_violation = None
        else:
            # All scenarios share the ENTIRE control plan. No oracle branch
            # selection or perfect future identification of mu is assumed.
            result = minimize(lambda z: float(z[-1]), np.r_[seed, baseline.max()],
                              method="SLSQP", bounds=bounds + [(0., None)],
                              constraints=[{"type": "ineq", "fun": lambda z: z[-1] - self.costs(z[:-1])}],
                              options={"maxiter": maxiter, "ftol": 1e-5})
            controls = result.x[:-1]
            epigraph_violation = float(max(0., np.max(self.costs(controls) - result.x[-1])))
        elapsed = time.perf_counter() - start
        selected = self.costs(controls).copy()
        return {"frame": self.frame, "reference_time_s": self.frame * self.p.dt,
                "prediction_mus": self.mus, "horizon_steps": self.horizon,
                "horizon_s": self.horizon * self.p.dt, "control_blocks": self.blocks,
                "decision_variables": int(len(result.x)), "optimizer": "SciPy SLSQP",
                "maximum_iterations": maxiter,
                "success": bool(result.success), "status": int(result.status), "message": str(result.message),
                "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                "unique_plan_evaluations": self.calls, "elapsed_s": elapsed,
                "baseline_costs": baseline.tolist(), "selected_costs": selected.tolist(),
                "worst_cost_improved": bool(selected.max() < baseline.max()),
                "action_bound_violation": float(max(0., np.max(np.abs(controls)) - 1.)),
                "epigraph_constraint_violation": epigraph_violation,
                "first_action": controls.reshape(self.blocks, 8)[0].tolist()}


def derivative_probe(p, frame, epsilons=(1e-4, 1e-5, 1e-6)):
    x, u = p.targets[frame], p.nominal_action[frame]
    target = p.targets[min(frame + 1, p.n - 1)]
    outputs = []
    matrices = []
    for eps in epsilons:
        start = time.perf_counter()
        A, B = np.empty((2*p.nv, 2*p.nv)), np.empty((2*p.nv, 8))
        for j in range(2*p.nv):
            A[:, j] = (p.error(target, p.transition(p.perturb(x, j, eps), u))
                       - p.error(target, p.transition(p.perturb(x, j, -eps), u))) / (2*eps)
        for j in range(8):
            delta = np.zeros(8)
            delta[j] = eps
            B[:, j] = (p.error(target, p.transition(x, u + delta))
                       - p.error(target, p.transition(x, u - delta))) / (2*eps)
        matrices.append((A, B))
        outputs.append({"epsilon": eps, "elapsed_s": time.perf_counter() - start,
                        "A_frobenius_norm": float(np.linalg.norm(A)),
                        "B_frobenius_norm": float(np.linalg.norm(B))})
    middle_A, middle_B = matrices[1]
    for result, (A, B) in zip(outputs, matrices):
        result["A_relative_difference_vs_1e_5"] = float(np.linalg.norm(A-middle_A) / max(np.linalg.norm(middle_A), 1e-12))
        result["B_relative_difference_vs_1e_5"] = float(np.linalg.norm(B-middle_B) / max(np.linalg.norm(middle_B), 1e-12))
    return {"frame": frame, "central_differences_of": "complete normalized-action 20ms transition",
            "state_coordinates": "28D position tangent + velocity; not raw quaternion subtraction",
            "samples": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "motion", "contract", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 100])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserving previous diagnostic output")
    if not 1 <= args.blocks <= args.horizon or args.maxiter < 1:
        parser.error("Invalid short-probe settings")
    settings = json.loads((ROOT / "config/classical_tracking.json").read_text())
    contract = json.loads(args.contract.read_text())
    p = Predictor(args.model, args.motion, contract, settings)
    if any(not 0 <= frame < p.n for frame in args.frames):
        parser.error("Probe frame outside reference")
    result = {"kind": "offline_model_and_optimizer_feasibility_probe", "mujoco_version": mujoco.__version__,
              "scipy_version": scipy.__version__, "python_version": platform.python_version(),
              "platform": platform.platform(), "processor": platform.processor(),
              "probe_settings": {"frames": args.frames, "horizon_steps": args.horizon,
                                 "control_blocks": args.blocks, "maximum_iterations": args.maxiter},
              "model_sha256": digest(args.model), "motion_sha256": digest(args.motion),
              "contract_sha256": digest(args.contract), "probe_source_sha256": digest(Path(__file__)),
              "predictor_source_sha256": digest(ROOT / "scripts/classical_tracking.py"),
              "coordinates": {"nq": p.model.nq, "nv": p.model.nv, "nu": p.model.nu,
                              "stored_mechanical_state": p.model.nq+p.model.nv, "tangent_state": 2*p.model.nv},
              "absolute_height_cost": False, "reference_vz_cost": False,
              "diagnostic_cost_exact_ppo_parity": False, "learned_warm_start": False,
              "closed_loop_tracking_test": False, "real_time_test": False,
              "guaranteed_robust_control": False, "contact_parameters_modified": False,
              "scenario_note": "Shared full action sequence for mu=[0.3,0.7,1.6], fixed independently of validation mu; finite scenarios only",
              "constraints": "box-bounded normalized actions; no explicit hard path constraints beyond simulated contact/joint dynamics",
              "derivative_checks": [], "short_optimizations": []}
    for frame in args.frames:
        result["derivative_checks"].append(derivative_probe(p, frame))
        for mus in ([.7], [.3, .7, 1.6]):
            entry = ShortProblem(p, frame, args.horizon, args.blocks, mus).solve(args.maxiter)
            result["short_optimizations"].append(entry)
            print(json.dumps(entry), flush=True)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
