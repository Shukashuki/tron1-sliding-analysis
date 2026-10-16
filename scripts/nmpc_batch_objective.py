"""Optional exact NumPy batch objective and isolated native-predictor adapter.

All reward arithmetic follows :mod:`nmpc_tracking_objective`; this is not a
reduced planar objective. Arrays use (candidate, horizon, ...). The command at
``start_index + h`` is clamped to the last reference frame and held against
the PRE-action body pose, then scored at the fourth physical substep. Link
positions are origins; linear velocities are original unmerged link-COM world
velocities. In particular, the last physical contact forces come from the
native substep history, never from a new zero-control ``mj_forward``.

No existing CEM or runner is changed or automatically selected by this file.
The optional native backend rejects an entire batch upon any MuJoCo warning;
that is explicitly stricter than the legacy predictor, not identical failure
semantics. Numerical parity on finite, warning-free traces is a separate claim
from cross-engine contact equivalence or real-time performance.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import math

import numpy as np

from nmpc_tracking_objective import (
    BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective,
    quaternion_angle, quaternion_inverse, quaternion_multiply, quaternion_rotate,
)


BODY_FIELDS = {"body_pos_w": 3, "body_quat_w": 4,
               "body_lin_vel_w": 3, "body_ang_vel_w": 3}
NATIVE_WARNING_DIFFERENCE = "native_mujoco_warning_rejects_whole_batch_stricter_than_legacy"


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return result


def _quaternions(value, shape, name):
    result = _finite(value, shape, name)
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    # Match the scalar objective's normalization tolerance and normalization.
    if not np.allclose(norms, 1., atol=1e-5):
        raise ValueError(f"{name} must contain normalized wxyz quaternions")
    return result / norms


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def reference_indices(start_index, horizon_steps, reference_frames):
    """Held command indices; no interpolation or extra phase advancement."""
    if (isinstance(start_index, (bool, np.bool_))
            or not isinstance(start_index, (int, np.integer)) or start_index < 0):
        raise ValueError("start_index must be a nonnegative integer")
    horizon_steps = _positive_integer(horizon_steps, "horizon_steps")
    reference_frames = _positive_integer(reference_frames, "reference_frames")
    # Avoid integer overflow for a caller's arbitrarily large phase index.
    first = min(int(start_index), reference_frames - 1)
    return np.minimum(first + np.arange(horizon_steps), reference_frames - 1)


def select_reference_sequence(reference: Mapping, start_index, horizon_steps,
                              *, body_names=BODY_NAMES):
    """Select a copied (H,bodies,...) command from a full ordered archive."""
    body_names = tuple(body_names)
    if reference.get("body_names") is not None and tuple(reference["body_names"]) != body_names:
        raise ValueError("Reference body order differs from objective body order")
    positions = np.asarray(reference["body_pos_w"])
    if positions.ndim != 3 or positions.shape[1:] != (len(body_names), 3):
        raise ValueError("Full reference positions must have shape (frames,bodies,3)")
    indices = reference_indices(start_index, horizon_steps, len(positions))
    selected = {"body_names": body_names}
    for name, width in BODY_FIELDS.items():
        values = _finite(reference[name], (len(positions), len(body_names), width), name)
        selected[name] = values[indices].copy()
    return selected, indices


@dataclass(frozen=True)
class HeldBatchReference:
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    relative_body_pos_w: np.ndarray
    relative_body_quat_w: np.ndarray


class BatchHeightSmoothTrackingObjective:
    """Vectorized scalar-objective arithmetic, with explicit (B,H) memory.

    ``evaluate`` accepts already aligned previous actions/COM vz/reset masks
    for every stage. ``evaluate_native`` instead assembles the sequential
    memory from one initial sample per candidate plus the returned rollout.
    The latter never compares post[h] with reference[h+1].
    """

    def __init__(self, task_profile, body_names=BODY_NAMES, dt=.02,
                 soft_joint_limits=None):
        self.scalar = HeightSmoothTrackingObjective(task_profile, body_names, dt, soft_joint_limits)
        self.body_names = self.scalar.body_names
        self.wheels, self.nonwheels = self.scalar.wheels, self.scalar.nonwheels
        self.dt = self.scalar.dt
        self.weights, self.stds = dict(self.scalar.weights), dict(self.scalar.stds)
        self.smooth = copy.deepcopy(self.scalar.smooth)
        self.reward_upper_bound = self.scalar.reward_upper_bound

    @classmethod
    def from_scalar(cls, objective):
        if not isinstance(objective, HeightSmoothTrackingObjective):
            raise TypeError("Expected HeightSmoothTrackingObjective")
        return cls(objective.profile, objective.body_names, objective.dt, objective.soft_joint_limits)

    def contract(self):
        result = self.scalar.contract()
        result.update(kind="numpy_vectorized_exact_height_smooth_tracking_objective",
                      array_axes="candidate, horizon, then body/history/vector axes",
                      scalar_objective_term_names=list(self.weights),
                      per_term_cost="dt * (max(weight,0) - weighted_term); sums to stage_cost",
                      reference_phase="reference[min(start_index+h,last_frame)] held BEFORE each action",
                      nonfinite_policy="Raise; never silently replace invalid samples with finite scores",
                      production_cem_wired=False)
        return result

    def _body_state(self, state, shape=None, *, reference=False):
        if state.get("body_names") is not None and tuple(state["body_names"]) != self.body_names:
            raise ValueError("State body order differs from objective body order")
        positions = np.asarray(state["body_pos_w"])
        n = len(self.body_names)
        if shape is None:
            if positions.ndim != 4 or positions.shape[-2:] != (n, 3) or min(positions.shape[:2]) < 1:
                raise ValueError("Batched positions require shape (batch,horizon,bodies,3)")
            shape = positions.shape[:2]
        result = []
        for name, width in BODY_FIELDS.items():
            value = np.asarray(state[name], dtype=float)
            expected = (*shape, n, width)
            if reference and value.shape in ((n, width), (shape[1], n, width)):
                value = np.broadcast_to(value, expected)
            validator = _quaternions if name == "body_quat_w" else _finite
            result.append(validator(value, expected, name))
        return tuple(result), shape

    def hold_reference(self, reference: Mapping, pre_action_state: Mapping):
        (robot_p, robot_q, _, _), shape = self._body_state(pre_action_state)
        (p, q, velocity, angular), _ = self._body_state(reference, shape, reference=True)
        difference = quaternion_multiply(robot_q[..., 0, :], quaternion_inverse(q[..., 0, :]))
        w, x, y, z = np.moveaxis(difference, -1, 0)
        yaw = np.arctan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
        zero = np.zeros(shape)
        delta_q = np.stack((np.cos(.5*yaw), zero, zero, np.sin(.5*yaw)), axis=-1)
        origin = robot_p[..., :1, :].copy()
        origin[..., 0, 2] = p[..., 0, 2]
        relative_p = origin + quaternion_rotate(delta_q[..., None, :], p-p[..., :1, :])
        relative_q = quaternion_multiply(delta_q[..., None, :], q)
        return HeldBatchReference(*(value.copy() for value in (p, q, velocity, angular, relative_p, relative_q)))

    def evaluate(self, state: Mapping, reference: HeldBatchReference, action,
                 previous_action, previous_base_com_vz, *, reset_samples=False):
        if not isinstance(reference, HeldBatchReference):
            raise TypeError("Use hold_reference with the PRE-action batch before evaluating")
        (p, q, velocity, angular), shape = self._body_state(state)
        n = len(self.body_names)
        # A held reference is public and may have been manually constructed.
        held = {}
        for name in BODY_FIELDS:
            width = BODY_FIELDS[name]
            validator = _quaternions if name == "body_quat_w" else _finite
            held[name] = validator(getattr(reference, name), (*shape, n, width), "held_"+name)
        relative_p = _finite(reference.relative_body_pos_w, (*shape, n, 3), "relative_body_pos_w")
        relative_q = _quaternions(reference.relative_body_quat_w, (*shape, n, 4), "relative_body_quat_w")
        target_p, target_q = held["body_pos_w"], held["body_quat_w"]
        target_v, target_w = held["body_lin_vel_w"], held["body_ang_vel_w"]
        action = _finite(action, (*shape, 8), "action")
        previous_action = _finite(previous_action, (*shape, 8), "previous_action")
        if max(np.max(np.abs(action)), np.max(np.abs(previous_action))) > 1.+1e-9:
            raise ValueError("Objective requires clipped, applied normalized actions")
        reset = np.asarray(reset_samples)
        if reset.dtype.kind != "b" or reset.shape not in ((), shape):
            raise ValueError("reset_samples must be boolean scalar or (batch,horizon) mask")
        reset = np.broadcast_to(reset, shape)
        if previous_base_com_vz is None:
            if not np.all(reset):
                raise ValueError("Non-reset reward requires previous_base_com_vz")
            previous_vz = np.zeros(shape)
        else:
            previous_vz = _finite(previous_base_com_vz, shape, "previous_base_com_vz")
        forces = _finite(state["body_net_forces_w"], (*shape, n, 3), "body_net_forces_w")
        history = state.get("contact_force_history_w")
        surrogate = []
        if history is None:
            history = forces[..., None, :, :]
            surrogate.append("instantaneous_undesired_contacts_without_sensor_history")
        else:
            history = _finite(history, (*shape, 3, n, 3), "contact_force_history_w")
        limits = self.scalar.soft_joint_limits
        if limits is None:
            limits = np.asarray(state["soft_joint_pos_limits"], dtype=float)
            if limits.shape == (8, 2):
                limits = np.broadcast_to(limits, (*shape, 8, 2))
            limits = _finite(limits, (*shape, 8, 2), "soft_joint_pos_limits")
            if np.any(limits[..., 0] > limits[..., 1]):
                raise ValueError("Soft joint lower limits exceed upper limits")
            if state.get("soft_joint_limits_source") != "isaac_runtime":
                surrogate.append("mujoco_or_unspecified_soft_joint_limits")
        joints = _finite(state["joint_pos"], (*shape, 8), "joint_pos")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            joint_error = np.maximum(limits[..., 0]-joints, 0.) + np.maximum(joints-limits[..., 1], 0.)
            body_error = np.mean(np.sum((relative_p[..., :2]-p[..., :2])**2, axis=-1), axis=-1)
            relative_angle = quaternion_angle(relative_q[..., self.nonwheels, :], q[..., self.nonwheels, :])
            body_v_error = np.mean(np.sum((target_v[..., :2]-velocity[..., :2])**2, axis=-1), axis=-1)
            body_w_error = np.mean(np.sum((target_w[..., self.nonwheels, :]-angular[..., self.nonwheels, :])**2, axis=-1), axis=-1)
            actual_body_v = quaternion_rotate(quaternion_inverse(q[..., 0, :]), velocity[..., 0, :])
            target_body_v = quaternion_rotate(quaternion_inverse(target_q[..., 0, :]), target_v[..., 0, :])
            beta = np.arctan2(actual_body_v[..., 1], actual_body_v[..., 0])
            target_beta = np.arctan2(target_body_v[..., 1], target_body_v[..., 0])
            beta_error = np.arctan2(np.sin(beta-target_beta), np.cos(beta-target_beta))
            speed = np.linalg.norm(velocity[..., 0, :2], axis=-1)
            target_speed = np.linalg.norm(target_v[..., 0, :2], axis=-1)
            gravity_b = quaternion_rotate(quaternion_inverse(q[..., 0, :]), np.array([0., 0., -1.]))
            delta_vz = np.zeros(shape)
            np.subtract(velocity[..., 0, 2], previous_vz, out=delta_vz, where=~reset)
            scaled_acceleration = delta_vz/(self.dt*self.smooth["acceleration_scale_m_s2"])
            acceleration_cost = -np.expm1(-scaled_acceleration**2)
            raw = {
                "motion_global_anchor_pos": np.exp(-np.sum((target_p[..., 0, :2]-p[..., 0, :2])**2, axis=-1)/self.stds["anchor_position_m"]**2),
                "motion_global_anchor_ori": np.exp(-quaternion_angle(target_q[..., 0, :], q[..., 0, :])**2/self.stds["anchor_orientation_rad"]**2),
                "motion_body_pos": np.exp(-body_error/self.stds["body_position_m"]**2),
                "motion_body_ori": np.exp(-np.mean(relative_angle**2, axis=-1)/self.stds["body_orientation_rad"]**2),
                "motion_body_lin_vel": np.exp(-body_v_error/self.stds["body_linear_velocity_m_s"]**2),
                "motion_body_ang_vel": np.exp(-body_w_error/self.stds["body_angular_velocity_rad_s"]**2),
                "action_rate_l2": np.sum((action-previous_action)**2, axis=-1),
                "joint_limit": np.sum(joint_error[..., :6], axis=-1),
                "undesired_contacts": np.sum(np.max(np.linalg.norm(history[..., self.nonwheels, :], axis=-1), axis=-2)>1., axis=-1).astype(float),
                "drift_sideslip": np.exp(-(beta_error/self.stds["sideslip_rad"])**2)*(target_speed>.3)*(speed>.25),
                "drift_speed": np.exp(-((speed-target_speed)/self.stds["speed_m_s"])**2),
                "drift_yaw_rate": np.exp(-((angular[..., 0, 2]-target_w[..., 0, 2])/self.stds["yaw_rate_rad_s"])**2),
                "drift_grounded": np.all(np.linalg.norm(forces[..., self.wheels, :], axis=-1)>5., axis=-1).astype(float),
                "drift_tilt": np.sum(gravity_b[..., :2]**2, axis=-1),
                "height_vertical_acceleration": acceleration_cost,
            }
            if "drift_leg_target_rate" in self.weights:
                raw["drift_leg_target_rate"] = np.sum(((action[..., :6]-previous_action[..., :6])/self.dt)**2, axis=-1)
            weighted = {name: self.weights[name]*value for name, value in raw.items()}
            term_costs = {name: self.dt*(max(self.weights[name], 0.)-value) for name, value in weighted.items()}
            reward = sum(weighted.values())
            stage_cost = self.dt*(self.reward_upper_bound-reward)
        diagnostics = {"beta_rad": beta, "target_beta_rad": target_beta,
                       "beta_error_rad": beta_error, "speed_m_s": speed,
                       "target_speed_m_s": target_speed}
        for name, value in {**raw, **weighted, **term_costs, **diagnostics,
                            "total_reward": reward, "stage_cost": stage_cost}.items():
            if value.shape != shape or not np.isfinite(value).all():
                raise FloatingPointError("Nonfinite or malformed batch objective output: " + name)
        return {"raw_terms": raw, "weighted_terms": weighted, "term_costs": term_costs,
                "total_reward": reward, "reward_upper_bound": self.reward_upper_bound,
                "stage_cost": stage_cost, "base_com_vz": velocity[..., 0, 2].copy(),
                "diagnostics": diagnostics, "surrogate_flags": surrogate}

    def evaluate_native(self, native: Mapping, reference: Mapping, actions,
                        initial_previous_action, initial_previous_base_com_vz,
                        *, joint_qpos_indices, soft_joint_pos_limits,
                        soft_joint_limits_source, initial_reset_samples=False):
        """Score NativeBatchTransition output; no physics or phase changes here.

        The reset flag is an initial-candidate property, suppressed at h=0
        only. Later stages use the preceding POST base-COM vz, including after
        a reset. Undesired contact history is the physical last-three samples.
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 3 or actions.shape[-1] != 8 or min(actions.shape[:2]) < 1:
            raise ValueError("Actions require shape (batch,horizon,8)")
        batch, horizon = shape = actions.shape[:2]
        actions = _finite(actions, (*shape, 8), "actions")
        previous = _finite(initial_previous_action, (batch, 8), "initial_previous_action")
        reset = np.asarray(initial_reset_samples)
        if reset.dtype.kind != "b" or reset.shape not in ((), (batch,)):
            raise ValueError("initial_reset_samples must be boolean scalar or (batch,) mask")
        reset = np.broadcast_to(reset, (batch,))
        if initial_previous_base_com_vz is None:
            if not np.all(reset):
                raise ValueError("Non-reset reward requires initial_previous_base_com_vz")
            initial_vz = np.zeros(batch)
        else:
            initial_vz = _finite(initial_previous_base_com_vz, (batch,), "initial_previous_base_com_vz")
        indices = np.asarray(joint_qpos_indices)
        if (indices.shape != (8,) or indices.dtype.kind not in "iu"
                or len(np.unique(indices)) != 8 or np.any(indices < 0) or np.any(indices >= 15)):
            raise ValueError("joint_qpos_indices must be eight distinct model qpos indices")
        n = len(self.body_names)
        pre = {"body_names": self.body_names}
        post = {"body_names": self.body_names}
        for name, width in BODY_FIELDS.items():
            pre[name] = _finite(native["pre_"+name], (*shape, n, width), "pre_"+name)
            post[name] = _finite(native[name], (*shape, 4, n, width), name)[..., -1, :, :]
        qpos = _finite(native["qpos"], (*shape, 4, 15), "qpos")
        history = _finite(native["contact_force_history_w"], (*shape, 3, n, 3), "contact_force_history_w")
        physical_forces = _finite(native["body_net_forces_w"], (*shape, 4, n, 3), "body_net_forces_w")
        if not np.array_equal(history[..., -1, :, :], physical_forces[..., -1, :, :]):
            raise ValueError("Final physical forces disagree with returned contact history")
        times = _finite(native["interval_time_s"], (*shape, 4), "interval_time_s")
        if not np.allclose(times, np.arange(1, 5)*(self.dt/4), rtol=0., atol=1e-12):
            raise ValueError("Native physical-substep clock differs from objective dt")
        post.update(joint_pos=qpos[..., -1, indices],
                    body_net_forces_w=history[..., -1, :, :],
                    contact_force_history_w=history,
                    soft_joint_pos_limits=soft_joint_pos_limits,
                    soft_joint_limits_source=soft_joint_limits_source)
        previous_actions = np.concatenate((previous[:, None, :], actions[:, :-1, :]), axis=1)
        previous_vz = np.concatenate((initial_vz[:, None], post["body_lin_vel_w"][:, :-1, 0, 2]), axis=1)
        reset_mask = np.zeros(shape, dtype=bool)
        reset_mask[:, 0] = reset
        result = self.evaluate(post, self.hold_reference(reference, pre), actions,
                               previous_actions, previous_vz, reset_samples=reset_mask)
        result["surrogate_flags"].append(NATIVE_WARNING_DIFFERENCE)
        result["candidate_costs"] = np.sum(result["stage_cost"], axis=1)
        if not np.isfinite(result["candidate_costs"]).all():
            raise FloatingPointError("Nonfinite candidate horizon cost")
        return result


class NativeBatchTrackingPredictor:
    """Opt-in adapter; independent fixed-mu workers, not a validation-mu oracle.

    ``scalar_predictor`` supplies the already audited model, named joint IDs,
    scalar objective, reference and bias-only nominal plan. This adapter never
    calls its old Q/R/TVLQR code or learned actions. Construction/build is never
    implicit in an existing runner. The caller must close owned workers.
    """

    def __init__(self, scalar_predictor, native_library, *, timeout_s=30.,
                 max_output_bytes=512*1024**2):
        from nmpc_native_batch import NativeBatchTransition
        self.base = scalar_predictor
        self.layout = scalar_predictor.layout
        self.objective = BatchHeightSmoothTrackingObjective.from_scalar(scalar_predictor.objective)
        self.mu_scenarios = tuple(scalar_predictor.mu_scenarios)
        if (not self.mu_scenarios or len(set(self.mu_scenarios)) != len(self.mu_scenarios)
                or not np.isfinite(self.mu_scenarios).all() or min(self.mu_scenarios) < 0):
            raise ValueError("Declare unique finite nonnegative prediction friction scenarios")
        if not math.isclose(self.objective.dt, .02, rel_tol=0., abs_tol=1e-15):
            raise ValueError("Native adapter requires unchanged 50Hz objective")
        self.full_reference = {"body_names": self.objective.body_names}
        for name in BODY_FIELDS:
            self.full_reference[name] = np.stack([frame[name] for frame in scalar_predictor.full_references])
        self.workers, self.qids, self.limits, self.limit_sources = {}, {}, {}, {}
        try:
            for mu in self.mu_scenarios:
                model = scalar_predictor.models[mu]
                self.qids[mu] = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
                extractor = scalar_predictor.extractors[mu]
                self.limits[mu] = extractor.soft_limits.copy()
                self.limit_sources[mu] = extractor.soft_limits_source
                self.workers[mu] = NativeBatchTransition(model, scalar_predictor.provider.q0,
                    scalar_predictor.contract, native_library, timeout_s=timeout_s,
                    max_output_bytes=max_output_bytes)
        except BaseException:
            self.close()
            raise
        self.metadata = {"kind": "optional_native_batch_exact_objective_adapter",
                         "prediction_mu_scenarios": list(self.mu_scenarios),
                         "plant_mu_used_for_prediction": False,
                         "production_cem_wired": False, "learned_initialization": False,
                         "terminal_cost": None, "objective": self.objective.contract(),
                         "warnings": NATIVE_WARNING_DIFFERENCE,
                         "whole_batch_warning_failure": True,
                         "workers": {str(mu): copy.deepcopy(worker.metadata) for mu, worker in self.workers.items()}}

    def evaluate_batch(self, state, full_controls, start_index, *, mu,
                       previous_action, previous_vz):
        if mu not in self.workers:
            raise ValueError("Prediction friction was not declared before validation")
        plans = np.asarray(full_controls, dtype=float)
        if plans.ndim != 3 or plans.shape[-1] != 8 or min(plans.shape[:2]) < 1:
            raise ValueError("Full candidate plans require shape (batch,horizon,8)")
        batch, horizon = plans.shape[:2]
        plans = _finite(plans, (batch, horizon, 8), "full_controls")
        previous = _finite(previous_action, (8,), "previous_action")
        if max(np.max(np.abs(plans)), np.max(np.abs(previous))) > 1.+1e-9:
            raise ValueError("Expected clipped applied normalized actions")
        if not np.isscalar(previous_vz) or not np.isfinite(previous_vz):
            raise ValueError("previous_vz must be finite scalar COM velocity")
        qpos, qvel, history, reset = self.layout.unpack(state)
        reference, indices = select_reference_sequence(self.full_reference, start_index, horizon,
                                                       body_names=self.objective.body_names)
        native = self.workers[mu].rollout(np.tile(qpos, (batch, 1)), np.tile(qvel, (batch, 1)), plans,
                                         initial_history=np.tile(history[None], (batch, 1, 1, 1)))
        result = self.objective.evaluate_native(native, reference, plans,
            np.tile(previous, (batch, 1)), np.full(batch, previous_vz),
            joint_qpos_indices=self.qids[mu], soft_joint_pos_limits=self.limits[mu],
            soft_joint_limits_source=self.limit_sources[mu], initial_reset_samples=reset)
        result["reference_indices"] = indices
        return result

    def rollout_batch(self, state, full_controls, start_index, *, mu,
                      previous_action, previous_vz):
        return self.evaluate_batch(state, full_controls, start_index, mu=mu,
            previous_action=previous_action, previous_vz=previous_vz)["candidate_costs"]

    def rollout(self, state, full_controls, start_index, *, mu,
                previous_action, previous_vz):
        plan = np.asarray(full_controls, dtype=float)
        if plan.ndim != 2 or plan.shape[-1] != 8:
            raise ValueError("Single candidate plan requires shape (horizon,8)")
        return float(self.rollout_batch(state, plan[None], start_index, mu=mu,
            previous_action=previous_action, previous_vz=previous_vz)[0])

    def initial_candidate(self, *args, **kwargs):
        return self.base.initial_candidate(*args, **kwargs)

    def record_cem_wiring(self, controller):
        """Record actual runtime attachment, not merely interface availability.

        Standalone transition/build manifests retain their historical false
        flag. The runner calls this only after constructing the shared CEM
        with THIS adapter as its predictor. No controller behavior is changed.
        """
        from nmpc_shooting import ShootingNMPC
        if not isinstance(controller, ShootingNMPC) or controller.predictor is not self:
            raise ValueError("CEM wiring audit requires the actual controller using this adapter")
        if not callable(getattr(controller.predictor, "rollout_batch", None)):
            raise ValueError("Actual controller predictor lacks the batch interface")
        self.metadata["production_cem_wired"] = True
        self.metadata["objective"]["production_cem_wired"] = True
        self.metadata["runtime_cem_population_path"] = "callable predictor.rollout_batch"
        self.metadata["worker_build_manifest_flag_semantics"] = "Nested worker production_cem_wired is historical build-time metadata, not this runtime attachment."

    def close(self):
        for worker in getattr(self, "workers", {}).values():
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
