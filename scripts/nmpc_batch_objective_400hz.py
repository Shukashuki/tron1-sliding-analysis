"""Separate 400 Hz native bridge for the existing 15-term tracking reward.

The scalar, NumPy and Torch reward formulas are unchanged. This adapter reads
all eight native 2.5 ms substeps, scores the final one against the command held
BEFORE the action interval, and retains the last three physical force samples.
It never relabels a four-substep trace as a 400 Hz result. Same mathematical
reward does not assert that MuJoCo and PhysX have identical contact forces.

Only candidate-horizon aggregation optionally discounts the unchanged stage
cost. No terminal value or learned-policy initialization is introduced.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import math

import numpy as np

from nmpc_batch_objective import (
    BODY_FIELDS, NATIVE_WARNING_DIFFERENCE,
    BatchHeightSmoothTrackingObjective as _OriginalBatchObjective,
    NativeBatchTrackingPredictor as _OriginalBatchPredictor, _finite,
)
from nmpc_tracking_objective import JOINT_NAMES


PHYSICS_DT = .0025
POLICY_DT = .02
SUBSTEPS = 8


def _discount(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value)) or not 0. < float(value) <= 1.):
        raise ValueError("discount_factor must be finite in (0,1]")
    return float(value)


class BatchHeightSmoothTrackingObjective400Hz(_OriginalBatchObjective):
    """Unmodified scalar stage reward, explicit eight-substep output contract."""

    def __init__(self, *args, discount_factor=1., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isclose(self.dt, POLICY_DT, rel_tol=0., abs_tol=1e-15):
            raise ValueError("400 Hz native scoring requires the unchanged 50 Hz reward clock")
        self.discount_factor = _discount(discount_factor)

    @classmethod
    def from_scalar(cls, objective, *, discount_factor=1.):
        from nmpc_tracking_objective import HeightSmoothTrackingObjective
        if not isinstance(objective, HeightSmoothTrackingObjective):
            raise TypeError("Expected HeightSmoothTrackingObjective")
        return cls(objective.profile, objective.body_names, objective.dt,
                   objective.soft_joint_limits, discount_factor=discount_factor)

    def contract(self):
        result = super().contract()
        result.update(kind="numpy_vectorized_exact_height_smooth_tracking_objective_400hz",
                      physics_dt_s=PHYSICS_DT, physical_substeps=SUBSTEPS,
                      stage_reward_formulas_changed=False,
                      native_contact_history="last three of eight physical 2.5 ms samples",
                      horizon_discount_factor=self.discount_factor,
                      horizon_cost="sum(discount_factor**h * stage_cost[h]); no terminal value")
        return result

    def evaluate_native(self, native: Mapping, reference: Mapping, actions,
                        initial_previous_action, initial_previous_base_com_vz,
                        *, joint_qpos_indices, soft_joint_pos_limits,
                        soft_joint_limits_source, initial_reset_samples=False):
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
        pre, post = ({"body_names": self.body_names} for _ in range(2))
        for name, width in BODY_FIELDS.items():
            pre[name] = _finite(native["pre_"+name], (*shape, n, width), "pre_"+name)
            post[name] = _finite(native[name], (*shape, SUBSTEPS, n, width), name)[..., -1, :, :]
        qpos = _finite(native["qpos"], (*shape, SUBSTEPS, 15), "qpos")
        history = _finite(native["contact_force_history_w"], (*shape, 3, n, 3), "contact_force_history_w")
        forces = _finite(native["body_net_forces_w"], (*shape, SUBSTEPS, n, 3), "body_net_forces_w")
        if not np.array_equal(history, forces[..., -3:, :, :]):
            raise ValueError("Last three physical forces disagree with returned contact history")
        times = _finite(native["interval_time_s"], (*shape, SUBSTEPS), "interval_time_s")
        if not np.allclose(times, np.arange(1, SUBSTEPS+1)*PHYSICS_DT, rtol=0., atol=1e-12):
            raise ValueError("Native physical-substep clock is not eight 2.5 ms samples")
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
        weights = self.discount_factor**np.arange(horizon)
        result["candidate_costs"] = np.sum(result["stage_cost"]*weights[None], axis=1)
        result["undiscounted_candidate_costs"] = np.sum(result["stage_cost"], axis=1)
        result["horizon_discount_weights"] = weights
        if not np.isfinite(result["candidate_costs"]).all():
            raise FloatingPointError("Nonfinite candidate horizon cost")
        return result


class NativeBatchTrackingPredictor400Hz(_OriginalBatchPredictor):
    """Existing CEM population API with the separately locked 400 Hz worker."""

    def __init__(self, scalar_predictor, native_library, *, timeout_s=30.,
                 max_output_bytes=512*1024**2, discount_factor=1.):
        from nmpc_native_batch_400hz import NativeBatchTransition
        self.base, self.layout = scalar_predictor, scalar_predictor.layout
        self.objective = BatchHeightSmoothTrackingObjective400Hz.from_scalar(
            scalar_predictor.objective, discount_factor=discount_factor)
        self.mu_scenarios = tuple(scalar_predictor.mu_scenarios)
        if (not self.mu_scenarios or len(set(self.mu_scenarios)) != len(self.mu_scenarios)
                or not np.isfinite(self.mu_scenarios).all() or min(self.mu_scenarios) < 0):
            raise ValueError("Declare unique finite nonnegative prediction friction scenarios")
        if (scalar_predictor.contract.get("physics_dt") != PHYSICS_DT
                or scalar_predictor.contract.get("decimation") != SUBSTEPS):
            raise ValueError("400 Hz adapter requires declared 2.5 ms x8 actuator contract")
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
        self.metadata = {
            "kind": "native_batch_exact_objective_adapter_400hz",
            "physics_dt_s": PHYSICS_DT, "control_dt_s": POLICY_DT,
            "physical_substeps": SUBSTEPS,
            "prediction_mu_scenarios": list(self.mu_scenarios),
            "plant_mu_used_for_prediction": False, "production_cem_wired": False,
            "learned_initialization": False, "terminal_cost": None,
            "horizon_discount_factor": self.objective.discount_factor,
            "objective": self.objective.contract(), "warnings": NATIVE_WARNING_DIFFERENCE,
            "whole_batch_warning_failure": True,
            "workers": {str(mu): copy.deepcopy(worker.metadata) for mu, worker in self.workers.items()},
        }


# Familiar names are intentionally local to this NEW module; frozen imports
# from nmpc_batch_objective continue selecting the original 200 Hz bridge.
BatchHeightSmoothTrackingObjective = BatchHeightSmoothTrackingObjective400Hz
NativeBatchTrackingPredictor = NativeBatchTrackingPredictor400Hz
