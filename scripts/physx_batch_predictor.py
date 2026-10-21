"""Isolated Isaac/PhysX shooting predictor using the existing packed-state API.

The parent may own a MuJoCo or PhysX validation plant. A separate process owns
one vectorized PhysX scene, so evaluating candidates cannot advance that plant.
MuJoCo is used only by ``base.measure`` for coordinate conversion and by the
unchanged physics-only nominal-plan service; candidate transitions use PhysX.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from multiprocessing.connection import Listener
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value.copy()


def validate_physx_config(config):
    """Validate an explicit, serializable worker config without importing Isaac."""
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("PhysX prediction requires schema_version=1")
    required = {"source_root", "motion", "asset_path", "task_profile", "device"}
    allowed = required | {"schema_version", "python_executable", "worker_script", "max_candidates",
                          "seed", "timeout_s", "startup_timeout_s", "log_dir", "environment_spacing_m",
                          "beyondmimic_root"}
    if required - config.keys() or config.keys() - allowed:
        raise ValueError(f"Invalid PhysX config fields: missing={sorted(required-config.keys())}, "
                         f"unknown={sorted(config.keys()-allowed)}")
    out = copy.deepcopy(config)
    for name in ("source_root", "motion", "asset_path", "task_profile"):
        out[name] = str(Path(out[name]).expanduser().resolve())
        if not Path(out[name]).exists():
            raise FileNotFoundError(out[name])
    device = out["device"]
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("PhysX predictor requires an explicit cuda:N device")
    for name, default in (("max_candidates", 64), ("seed", 42)):
        value = out.get(name, default)
        if type(value) is not int or value < (1 if name == "max_candidates" else 0):
            raise ValueError(f"Invalid PhysX {name}")
        out[name] = value
    for name, default in (("timeout_s", 180.), ("startup_timeout_s", 240.), ("environment_spacing_m", 20.)):
        value = out.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (value == 0 and name != "environment_spacing_m")):
            raise ValueError(f"Invalid PhysX {name}")
        out[name] = float(value)
    if 0 < out["environment_spacing_m"] < 3.:
        raise ValueError("Prediction environments require zero or at least 3 m spacing")
    out["python_executable"] = str(Path(out.get("python_executable", sys.executable)).resolve())
    out["worker_script"] = str(Path(out.get("worker_script", Path(__file__).with_name("physx_batch_predictor_worker.py"))).resolve())
    for name in ("python_executable", "worker_script"):
        if not Path(out[name]).is_file():
            raise FileNotFoundError(out[name])
    if out.get("log_dir") is not None:
        out["log_dir"] = str(Path(out["log_dir"]).resolve())
    if out.get("beyondmimic_root") is not None:
        out["beyondmimic_root"] = str(Path(out["beyondmimic_root"]).resolve())
        if not (Path(out["beyondmimic_root"])/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py").is_file():
            raise FileNotFoundError("Missing configured BeyondMimic source")
    json.dumps(out, allow_nan=False)
    return out


def canonical_measurement(base, state):
    """Convert packed MuJoCo tangent velocities to original-link world COM data.

    The conversion is kinematic. No MuJoCo time step or control rollout occurs.
    """
    qpos, qvel, history, reset = base.layout.unpack(state)
    measured = base.measure(state)
    return {
        "root_pos": _finite(measured["body_pos_w"][0], (3,), "root_pos"),
        "root_quat_wxyz": _finite(measured["body_quat_w"][0], (4,), "root_quat_wxyz"),
        "root_com_lin_vel_w": _finite(measured["body_lin_vel_w"][0], (3,), "root_com_lin_vel_w"),
        "root_ang_vel_w": _finite(measured["body_ang_vel_w"][0], (3,), "root_ang_vel_w"),
        "joint_pos": _finite(qpos[base.provider.qids], (8,), "joint_pos"),
        "joint_vel": _finite(qvel[base.provider.vids], (8,), "joint_vel"),
        "body_pos_w": _finite(measured["body_pos_w"], (9, 3), "body_pos_w"),
        "body_quat_w": _finite(measured["body_quat_w"], (9, 4), "body_quat_w"),
        "body_lin_vel_w": _finite(measured["body_lin_vel_w"], (9, 3), "body_lin_vel_w"),
        "body_ang_vel_w": _finite(measured["body_ang_vel_w"], (9, 3), "body_ang_vel_w"),
        "contact_force_history_w": _finite(history, (3, 9, 3), "contact_force_history_w"),
        "first_reward_after_reset": bool(reset),
    }


class PhysXBatchTrackingPredictor:
    """ShootingNMPC adapter; same objective and seeds, native PhysX transitions.

    A fixed singleton friction scenario is intentional for the initial backend
    experiment. Each new run can choose a different fixed coefficient, including
    a declared plant-friction oracle, before starting its worker.
    """

    def __init__(self, scalar_predictor, config):
        self.base = scalar_predictor
        self.config = validate_physx_config(config)
        self.layout, self.provider = scalar_predictor.layout, scalar_predictor.provider
        self.objective = scalar_predictor.objective
        self.reference = scalar_predictor.reference
        self.motion_sha256 = hashlib.sha256(Path(self.config["motion"]).read_bytes()).hexdigest()
        if self.motion_sha256 != hashlib.sha256(Path(self.reference.path).read_bytes()).hexdigest():
            raise ValueError("PhysX worker motion differs from the base predictor reference archive")
        self.mu_scenarios = tuple(float(mu) for mu in scalar_predictor.mu_scenarios)
        if len(self.mu_scenarios) != 1 or not np.isfinite(self.mu_scenarios).all() or self.mu_scenarios[0] < 0:
            raise ValueError("PhysX v1 requires one predeclared finite nonnegative friction scenario")
        self._temporary = tempfile.TemporaryDirectory(prefix="tron1-physx-mpc-")
        self._listener = self._connection = self._process = self._log = None
        self._closed = False
        self._request_id = 0
        self.log_dir = Path(self.config.get("log_dir") or self._temporary.name)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"physx_predictor_worker_{os.getpid()}_{time.time_ns()}.log"
        self.metadata = {
            "kind": "isolated_isaac_physx_batch_tracking_predictor",
            "prediction_engine": "Isaac/PhysX", "population_batching": True,
            "prediction_mu_scenarios": list(self.mu_scenarios),
            "production_cem_wired": False, "learned_initialization": False,
            "nominal_seed_source": "unchanged base predictor physics-only bias-force reference plan",
            "mujoco_usage": "kinematic packed-state conversion and unchanged nominal seed only; no candidate transition",
            "objective": self.objective.contract(), "ipc_included_in_latency": True,
            "worker_log": str(self.log_path), "worker_config": copy.deepcopy(self.config),
            "motion_sha256": self.motion_sha256,
            "hidden_solver_state": "PhysX internal contact/solver caches are not serialized; repetition probe reported separately",
            "integration_state_semantics": "Reconstruct at candidate start, then integrate continuously over the full horizon; historical MuJoCo predictor instead cold-resets each control interval",
            "physics_dt_s": .005, "control_dt_s": .02,
            "worker_source_sha256": hashlib.sha256(Path(self.config["worker_script"]).read_bytes()).hexdigest(),
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        try:
            address = str(Path(self._temporary.name) / "worker.sock")
            authkey = os.urandom(32)
            self._listener = Listener(address, family="AF_UNIX", authkey=authkey)
            self._listener._listener._socket.settimeout(self.config["startup_timeout_s"])
            environment = os.environ.copy()
            environment["TRON1_PHYSX_WORKER_AUTH"] = authkey.hex()
            environment["PYTHONUNBUFFERED"] = "1"
            if self.config.get("beyondmimic_root"):
                environment["TRON1_BEYONDMIMIC_ROOT"] = self.config["beyondmimic_root"]
            self._log = self.log_path.open("x")
            command = [self.config["python_executable"], "-u", self.config["worker_script"],
                       "--socket", address]
            self._process = subprocess.Popen(command, env=environment, stdout=self._log,
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            self._connection = self._listener.accept()
            self._listener.close(); self._listener = None
            startup = {"kind": "initialize", "config": self.config,
                       "mu": self.mu_scenarios[0], "objective_contract": self.objective.contract(),
                       "expected_motion_sha256": self.motion_sha256,
                       "expected_soft_joint_limits": (None if self.objective.soft_joint_limits is None
                                                       else self.objective.soft_joint_limits.tolist())}
            extractor = getattr(self.base, "measure_extractor", None)
            startup["expected_leg_soft_limits"] = (None if extractor is None else
                                                   np.asarray(extractor.soft_limits)[:6].tolist())
            self._connection.send(startup)
            response = self._receive(self.config["startup_timeout_s"])
            if response.get("kind") != "ready":
                raise RuntimeError(f"PhysX worker startup failed: {response}")
            self.metadata["worker_runtime"] = response["metadata"]
            if self.objective.soft_joint_limits is None:
                self.objective.soft_joint_limits = np.asarray(
                    response["metadata"]["runtime_soft_joint_pos_limits"], dtype=float)
                self.metadata["plant_objective_limits_set_from_physx_runtime"] = True
            self.metadata["worker_pid"] = self._process.pid
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        if not self._connection.poll(timeout):
            status = None if self._process is None else self._process.poll()
            raise TimeoutError(f"PhysX predictor IPC timeout after {timeout}s, process_exit={status}; log={self.log_path}")
        response = self._connection.recv()
        if not isinstance(response, dict):
            raise RuntimeError("Malformed PhysX worker response")
        if response.get("kind") == "error":
            raise RuntimeError(f"PhysX worker {response.get('error_type')}: {response.get('message')}; log={self.log_path}")
        return response

    def evaluate_batch(self, state, full_controls, start_index, *, mu, previous_action, previous_vz,
                       return_trajectory=False):
        if self._closed:
            raise RuntimeError("PhysX predictor is closed")
        if float(mu) not in self.mu_scenarios:
            raise ValueError("Prediction friction was not declared before validation")
        plans = np.asarray(full_controls, dtype=float)
        if plans.ndim != 3 or plans.shape[-1] != 8 or min(plans.shape[:2]) < 1:
            raise ValueError("Candidate plans require shape (batch,horizon,8)")
        if plans.shape[0] > self.config["max_candidates"]:
            raise ValueError("Candidate count exceeds configured PhysX environment count")
        plans = _finite(plans, plans.shape, "full_controls")
        previous = _finite(previous_action, (8,), "previous_action")
        if max(np.max(np.abs(plans)), np.max(np.abs(previous))) > 1. + 1e-9:
            raise ValueError("Expected bounded normalized candidate actions")
        if isinstance(start_index, (bool, np.bool_)) or not isinstance(start_index, (int, np.integer)) or start_index < 0:
            raise ValueError("Reference index must be a nonnegative integer")
        if not np.isscalar(previous_vz) or not np.isfinite(previous_vz):
            raise ValueError("previous_vz must be a finite scalar")
        self._request_id += 1
        self._connection.send({"kind": "rollout", "request_id": self._request_id,
                               "measurement": canonical_measurement(self.base, state),
                               "controls": plans, "start_index": int(start_index),
                               "previous_action": previous, "previous_vz": float(previous_vz),
                               "return_trajectory": bool(return_trajectory)})
        result = self._receive(self.config["timeout_s"])
        if result.get("kind") != "rollout_result" or result.get("request_id") != self._request_id:
            raise RuntimeError("PhysX predictor response request mismatch")
        result["candidate_costs"] = _finite(result["candidate_costs"], (len(plans),), "candidate_costs")
        self.metadata["last_reset_audit"] = result["reset_audit"]
        self.metadata["completed_rollout_batches"] = result["completed_rollout_batches"]
        return result

    def rollout_batch(self, state, full_controls, start_index, *, mu, previous_action, previous_vz):
        return self.evaluate_batch(state, full_controls, start_index, mu=mu,
                                   previous_action=previous_action, previous_vz=previous_vz)["candidate_costs"]

    def diagnostic_rollout(self, state, full_controls, start_index, *, mu, previous_action, previous_vz):
        """Return native states, substep torques, costs and the zero-step reset audit."""
        return self.evaluate_batch(state, full_controls, start_index, mu=mu,
                                   previous_action=previous_action, previous_vz=previous_vz,
                                   return_trajectory=True)

    def rollout(self, state, full_controls, start_index, *, mu, previous_action, previous_vz):
        controls = np.asarray(full_controls, dtype=float)
        if controls.ndim != 2 or controls.shape[1] != 8:
            raise ValueError("Single plan requires shape (horizon,8)")
        return float(self.rollout_batch(state, controls[None], start_index, mu=mu,
                     previous_action=previous_action, previous_vz=previous_vz)[0])

    def initial_candidate(self, *args, **kwargs):
        return self.base.initial_candidate(*args, **kwargs)

    def state_contract(self):
        contract = self.layout.contract()
        contract["excluded_integration_state"] = (
            "PhysX internal contact/solver caches are not serialized at candidate start; "
            "native simulation then continues through all horizon intervals. Reset repeatability is empirical.")
        return contract

    def reference_frame(self, index):
        return self.base.reference_frame(index)

    def measure(self, state):
        return self.base.measure(state)

    def vertical_velocity(self, state):
        return self.base.vertical_velocity(state)

    def record_cem_wiring(self, controller):
        if controller.predictor is not self or not callable(getattr(controller.predictor, "rollout_batch", None)):
            raise AssertionError("CEM is not using this PhysX rollout_batch predictor")
        if controller.config.candidates > self.config["max_candidates"]:
            raise ValueError("CEM candidate count exceeds PhysX worker environments")
        self.metadata.update(production_cem_wired=True, runtime_cem_population_path="predictor.rollout_batch")

    def probe_repeatability(self, state, controls, start_index, *, previous_action, previous_vz,
                            cost_tolerance=1e-5, state_tolerance=2e-5):
        """Measure repeat and slot/order sensitivity; no equivalence claim by default."""
        controls = np.asarray(controls, dtype=float)
        kwargs = dict(mu=self.mu_scenarios[0], previous_action=previous_action,
                      previous_vz=previous_vz, return_trajectory=True)
        first = self.evaluate_batch(state, controls, start_index, **kwargs)
        intervening = np.clip(-controls + .03, -1., 1.)
        self.evaluate_batch(state, intervening, start_index, **kwargs)
        second = self.evaluate_batch(state, controls, start_index, **kwargs)
        permutation = np.arange(len(controls))[::-1]
        permuted = self.evaluate_batch(state, controls[permutation], start_index, **kwargs)
        isolated_controls = controls.copy()
        isolated_controls[0] = intervening[0]
        isolated = self.evaluate_batch(state, isolated_controls, start_index, **kwargs)
        back = np.argsort(permutation)
        cost_repeat = float(np.max(np.abs(first["candidate_costs"]-second["candidate_costs"])))
        cost_permuted = float(np.max(np.abs(first["candidate_costs"]-permuted["candidate_costs"][back])))
        isolation_cost = (float(np.max(np.abs(first["candidate_costs"][1:]-isolated["candidate_costs"][1:])))
                          if len(controls) > 1 else 0.)
        channels = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos")
        state_errors = {name: {"repeat_max_abs": float(np.max(np.abs(first["trajectory"][name]-second["trajectory"][name]))),
                               "permuted_max_abs": float(np.max(np.abs(first["trajectory"][name]-permuted["trajectory"][name][back]))),
                               "cross_candidate_max_abs": (float(np.max(np.abs(first["trajectory"][name][1:]-isolated["trajectory"][name][1:])))
                                                            if len(controls) > 1 else 0.)}
                        for name in channels}
        passed = (max(cost_repeat, cost_permuted, isolation_cost) <= cost_tolerance and
                  all(max(values.values()) <= state_tolerance for values in state_errors.values()))
        result = {"kind": "empirical_physx_reset_repeatability_probe", "passed": bool(passed),
                  "candidate_count": len(controls), "horizon_steps": controls.shape[1],
                  "sequence": "A / different B / A / reverse-slot A / change one candidate",
                  "cost_repeat_max_abs": cost_repeat, "cost_permuted_max_abs": cost_permuted,
                  "cross_candidate_cost_max_abs": isolation_cost,
                  "cost_tolerance": cost_tolerance, "state_tolerance": state_tolerance,
                  "state_errors": state_errors,
                  "scope": "tested states/plans only; internal contact caches are not serialized"}
        self.metadata["reset_repeatability_probe"] = result
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                self._connection.send({"kind": "close"})
            except (OSError, EOFError):
                pass
            self._connection.close(); self._connection = None
        if self._process is not None:
            try:
                self._process.wait(timeout=20.)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    self._process.kill(); self._process.wait(timeout=10.)
            self.metadata["worker_exitcode"] = self._process.returncode
        if self._listener is not None:
            self._listener.close(); self._listener = None
        if self._log is not None:
            self._log.close(); self._log = None
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
