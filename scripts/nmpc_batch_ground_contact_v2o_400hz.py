"""O CEM objective: reuse K physical prediction, add the exact same command cost.

Stage0 uses an explicit actual pre-command load. Later stages use the previous
stage's final native substep; the first AFTER-command force is never substituted.
The existing native ABI3 rollout library and solver budget are unchanged.
"""
from __future__ import annotations

import numpy as np

import ground_contact_reward_v2o as definition
import nmpc_batch_ground_contact_v2k_400hz as parent


class BatchGroundContactTrackingObjective400Hz(parent.BatchGroundContactTrackingObjective400Hz):
    def __init__(self, *args, model_weight_n, ground_settings=None, discount_factor=.99, **kwargs):
        self.o_settings = definition.validate_settings(ground_settings)
        super().__init__(*args, model_weight_n=model_weight_n,
            ground_settings=definition.previous.DEFAULT_SETTINGS, discount_factor=discount_factor, **kwargs)
        self.initial_ground_Fz = None

    def set_initial_ground_Fz(self, force):
        force = np.asarray(force, dtype=float)
        if force.shape != (2,) or not np.isfinite(force).all() or np.any(force < -1e-4):
            raise ValueError("Require actual finite two-wheel pre-command NORMAL force")
        self.initial_ground_Fz = force.copy()

    def contract(self):
        result = super().contract()
        result.update(kind="numpy_ground_contact_reward_v2o_400hz",
            ground_contact_reward_v2=definition.formula_metadata(self.o_settings),
            initial_actual_pre_command_normal_force_required=True,
            subsequent_preceding_force="previous stage final of8 native physical samples",
            command_delta="current clipped leg actions minus prior clipped actions; interleaved L0/2/4 R1/3/5",
            wheel_torque_actions_in_new_cost=False)
        return result

    def evaluate_native(self, native, reference, actions, initial_previous_action,
                        initial_previous_base_com_vz, **kwargs):
        if self.initial_ground_Fz is None:
            raise ValueError("Explicit actual pre-command load is missing; cannot use a post-command sample")
        result = super().evaluate_native(native, reference, actions, initial_previous_action,
                                        initial_previous_base_com_vz, **kwargs)
        actions = np.asarray(actions, dtype=float)
        n, horizon = actions.shape[:2]
        preceding = np.empty((n, horizon, 2))
        preceding[:, 0] = self.initial_ground_Fz
        preceding[:, 1:] = native["ground_normal_force_w"][:, :-1, -1, :, 2]
        old = np.empty_like(actions)
        old[:, 0] = np.broadcast_to(np.asarray(initial_previous_action), (n, 8))
        old[:, 1:] = actions[:, :-1]
        commands = definition.command_numpy(preceding, actions, old)
        components = definition._replace(result["ground_contact_v2"], commands)
        result["raw_terms"]["drift_grounded"] = components["raw_reward"]
        result["weighted_terms"]["drift_grounded"] = .4*components["raw_reward"]
        result["term_costs"]["drift_grounded"] = self.dt*(.4-result["weighted_terms"]["drift_grounded"])
        result["total_reward"] = sum(result["weighted_terms"].values())
        result["stage_cost"] = self.dt*(self.reward_upper_bound-result["total_reward"])
        weights = self.discount_factor**np.arange(horizon)
        result["candidate_costs"] = np.sum(result["stage_cost"]*weights[None], axis=1)
        result["undiscounted_candidate_costs"] = np.sum(result["stage_cost"], axis=1)
        result["ground_contact_v2"] = components
        result["ground_contact_preceding_Fz_n"] = preceding
        if not np.isfinite(result["candidate_costs"]).all(): raise FloatingPointError("Nonfinite O horizon cost")
        return result


class NativeGroundContactTrackingPredictor400Hz(parent.NativeGroundContactTrackingPredictor400Hz):
    def __init__(self, scalar_predictor, native_library, *, ground_settings=None, discount_factor=.99, **kwargs):
        self.o_settings = definition.validate_settings(ground_settings)
        super().__init__(scalar_predictor, native_library,
            ground_settings=definition.previous.DEFAULT_SETTINGS, discount_factor=discount_factor, **kwargs)
        self.objective = BatchGroundContactTrackingObjective400Hz.from_scalar(scalar_predictor.objective,
            model_weight_n=self.objective.model_weight_n, ground_settings=self.o_settings, discount_factor=discount_factor)
        self.pre_command_frame = None
        self.metadata.update(kind="native_batch_ground_contact_reward_v2o_400hz",
            objective=self.objective.contract(), actual_pre_command_force_required=True)

    def set_preceding_ground_force(self, force_w, frame):
        value = np.asarray(force_w, dtype=float)
        if (value.shape != (2, 3) or not np.isfinite(value).all() or np.max(np.abs(value[:, :2])) > 1e-7
                or type(frame) is not int or frame < 0):
            raise ValueError("Require actual current NORMAL force[2,3] and explicit control frame")
        self.objective.set_initial_ground_Fz(value[:, 2])
        self.pre_command_frame = frame

    def evaluate_batch(self, state, full_controls, start_index, **kwargs):
        if self.pre_command_frame is None or start_index != self.pre_command_frame:
            raise ValueError("Missing or stale actual pre-command force for current control frame")
        return super().evaluate_batch(state, full_controls, start_index, **kwargs)


BatchHeightSmoothTrackingObjective = BatchGroundContactTrackingObjective400Hz
NativeBatchTrackingPredictor = NativeGroundContactTrackingPredictor400Hz
