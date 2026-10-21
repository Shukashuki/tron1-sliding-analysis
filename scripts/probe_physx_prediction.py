"""Exercise the actual isolated PhysX predictor before a full MPC episode.

The probe checks zero-step state reconstruction, first-substep DC torque,
scalar/batch objective arithmetic, physical clocks and reset/slot sensitivity.
Only the candidate worker advances physics; no validation plant is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nmpc_tracking_objective import HeightSmoothTrackingObjective
from physx_batch_predictor import PhysXBatchTrackingPredictor, canonical_measurement
from run_nmpc_tracking import MujocoTrackingPredictor
from run_nmpc_tracking_isaac import SourceModelProvider


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def scalar_parity(result, objective, reference, controls, phase, previous, previous_vz, reset):
    errors = []
    pre, post = result["pre_trajectory"], result["trajectory"]
    for b in range(len(controls)):
        last_action, last_vz = previous.copy(), previous_vz
        for h in range(controls.shape[1]):
            before = {key: value[b, h] for key, value in pre.items()}
            after = {key: value[b, h] for key, value in post.items()}
            after["soft_joint_limits_source"] = "isaac_runtime"
            held = objective.hold_reference(reference(phase+h), before)
            scored = objective.evaluate(after, held, controls[b, h].astype(np.float32).astype(float),
                                        last_action, last_vz, reset_sample=reset and h == 0)
            errors.append(abs(scored["stage_cost"]-result["stage_cost"][b, h]))
            last_action = controls[b, h].astype(np.float32).astype(float)
            last_vz = scored["base_com_vz"]
    return max(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "model", "motion", "contract", "task-profile", "settings",
                "physx-prediction-config", "output-dir"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 100, 150])
    p.add_argument("--horizon-steps", type=int, default=5)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--mu", type=float, default=.7)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if a.candidates < 3 or a.horizon_steps < 1:
        p.error("Use at least three candidates and a positive horizon")
    a.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    status = {"kind": "actual_physx_prediction_probe", "status": "initializing",
              "physics_engine": "Isaac/PhysX", "validation_plant_created": False,
              "learned_policy_used": False, "nominal_experiment_eligible": False}
    target = a.output_dir/"report.json"
    def save():
        target.write_text(json.dumps(status, indent=2, default=serial, allow_nan=False)+"\n")
    save()
    try:
        settings = json.loads(a.settings.read_text())
        profile = json.loads(a.task_profile.read_text())
        provider = SourceModelProvider(a.source_root, a.model, a.motion,
                                      json.loads(a.contract.read_text()), settings, (a.mu,))
        base = MujocoTrackingPredictor(provider, a.model, profile, (a.mu,))
        cfg = json.loads(a.physx_prediction_config.read_text())
        for key, expected in (("motion", a.motion), ("task_profile", a.task_profile)):
            if Path(cfg[key]).resolve() != expected.resolve():
                raise ValueError("PhysX config differs from probe "+key)
        rng = np.random.default_rng(42)
        cases = []
        with PhysXBatchTrackingPredictor(base, cfg) as prediction:
            runtime = prediction.metadata["worker_runtime"]
            objective = HeightSmoothTrackingObjective(profile, dt=.02,
                soft_joint_limits=np.asarray(runtime["runtime_soft_joint_pos_limits"]))
            for frame in a.frames:
                if not 0 <= frame < provider.reference.frames:
                    raise ValueError("Probe frame outside reference")
                packed = base.layout.pack(provider.targets[frame, :15], provider.targets[frame, 15:],
                                          np.zeros(base.layout.history_shape), frame == 0)
                measurement = canonical_measurement(base, packed)
                previous = np.zeros(8)
                vz = float(measurement["root_com_lin_vel_w"][2])
                indices = np.minimum(frame+np.arange(a.horizon_steps), provider.reference.frames-1)
                seed = provider.nominal_action[indices]
                plans = np.clip(seed[None]+rng.normal(0., .08, (a.candidates, a.horizon_steps, 8)), -1., 1.)
                plans[0] = plans[-1] = seed
                result = prediction.diagnostic_rollout(packed, plans, frame, mu=a.mu,
                    previous_action=previous, previous_vz=vz)
                parity = scalar_parity(result, objective, base.reference_frame, plans,
                                      frame, previous, vz, frame == 0)
                expected = np.stack([provider.torque(plan[0].astype(np.float32).astype(float),
                                provider.targets[frame, :15], provider.targets[frame, 15:]) for plan in plans])
                torque_error = float(np.max(np.abs(expected-result["applied_joint_torque_nm"][:, 0, 0])))
                repeat = prediction.probe_repeatability(packed, plans, frame, previous_action=previous,
                    previous_vz=vz, cost_tolerance=1e-4, state_tolerance=.005)
                duplicate_error = float(abs(result["candidate_costs"][0]-result["candidate_costs"][-1]))
                clock_ok = result["native_physics_steps"] == 4*a.horizon_steps
                reset_ok = max(result["reset_audit"][key] for key in (
                    "root_link_position_max_abs_m", "root_com_velocity_max_abs_m_s",
                    "joint_position_max_abs_rad")) <= 5e-4
                passed = bool(parity < 1e-10 and torque_error < .02 and clock_ok and reset_ok
                              and repeat["passed"] and duplicate_error < 1e-4)
                cases.append({"reference_frame": frame, "horizon_steps": a.horizon_steps,
                    "candidate_count": a.candidates, "passed": passed,
                    "scalar_batch_stage_cost_max_abs_error": parity,
                    "first_substep_torque_max_abs_error_nm": torque_error,
                    "duplicate_candidate_cost_abs_error": duplicate_error,
                    "physical_clock_verified": clock_ok, "reset_audit": result["reset_audit"],
                    "repeatability": repeat, "candidate_costs": result["candidate_costs"]})
                arrays = {"controls": plans, "costs": result["candidate_costs"],
                          "stage_cost": result["stage_cost"],
                          "applied_joint_torque_nm": result["applied_joint_torque_nm"]}
                for role in ("trajectory", "pre_trajectory"):
                    arrays.update({role+"_"+key: value for key, value in result[role].items()})
                np.savez_compressed(a.output_dir/f"frame_{frame:03d}.npz", **arrays)
                print(json.dumps(cases[-1], default=serial, allow_nan=False), flush=True)
            status.update(worker=prediction.metadata, cases=cases,
                          nominal_experiment_eligible=all(case["passed"] for case in cases))
        status.update(status="passed" if status["nominal_experiment_eligible"] else "diagnostic_failed",
                      elapsed_s=time.perf_counter()-started,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save()
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_s=time.perf_counter()-started)
        save()
        raise
    print(json.dumps({key: status[key] for key in ("status", "nominal_experiment_eligible", "elapsed_s")}))
    return 0 if status["nominal_experiment_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
