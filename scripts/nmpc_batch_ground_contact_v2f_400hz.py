"""Opt-in V2f any-wheel ground-support CEM objective and isolated ABI3 worker.

Replaces ONLY drift_grounded using true floor-only normal forces at all eight
substeps. This is a NEW reward contract, not unchanged historical stage reward.
"""
from __future__ import annotations

import copy

import numpy as np

from nmpc_batch_objective import BODY_FIELDS, NativeBatchTrackingPredictor as _PopulationAPI
from nmpc_batch_objective_400hz import BatchHeightSmoothTrackingObjective400Hz as _LegacyObjective
from nmpc_native_ground_contact_400hz import NativeBatchTransition, model_weight_n
from nmpc_tracking_objective import JOINT_NAMES


class BatchGroundContactTrackingObjective400Hz(_LegacyObjective):
    def __init__(self, *args, model_weight_n, ground_settings=None, discount_factor=.99, **kwargs):
        from ground_contact_reward_v2f import validate_settings
        super().__init__(*args, discount_factor=discount_factor, **kwargs)
        if self.weights["drift_grounded"] != .4:
            raise ValueError("V2f preserves external drift_grounded weight+0.4")
        self.model_weight_n = float(model_weight_n)
        if not np.isfinite(self.model_weight_n) or self.model_weight_n <= 0:
            raise ValueError("V2f requires positive finite native model weight")
        self.ground_settings = validate_settings(ground_settings)

    @classmethod
    def from_scalar(cls, objective, *, model_weight_n, ground_settings=None, discount_factor=.99):
        return cls(objective.profile, objective.body_names, objective.dt, objective.soft_joint_limits,
                   model_weight_n=model_weight_n, ground_settings=ground_settings, discount_factor=discount_factor)

    def contract(self):
        from ground_contact_reward_v2f import formula_metadata
        result = super().contract()
        result.update(kind="numpy_ground_contact_reward_v2f_400hz", stage_reward_formulas_changed=True,
            replacement_term="drift_grounded", unchanged_other_reward_terms=14,
            ground_contact_reward_v2=formula_metadata(self.ground_settings), model_weight_n=self.model_weight_n,
            ground_readout="actual unique flat world floor plane; NORMAL force only; counts>=1e-6N",
            intended_support="At least one wheel continuously supporting; other wheel may lift; alternating wheels allowed",
            absolute_height_constraint=False, zero_sideslip_target=False)
        return result

    def evaluate_native(self, native, reference, actions, initial_previous_action,
                        initial_previous_base_com_vz, **kwargs):
        from ground_contact_reward_v2f import reward_numpy
        result = super().evaluate_native(native, reference, actions, initial_previous_action,
                                         initial_previous_base_com_vz, **kwargs)
        shape = np.asarray(actions).shape[:2]
        forces = np.asarray(native["ground_normal_force_w"], dtype=float)
        counts = np.asarray(native["ground_contact_count"])
        if (forces.shape != (*shape, 8, 2, 3) or not np.isfinite(forces).all()
                or counts.shape != (*shape, 8, 2) or counts.dtype.kind not in "ifu"
                or not np.isfinite(counts).all() or np.any(counts < 0)
                or not np.array_equal(counts, np.rint(counts))):
            raise ValueError("V2f requires finite8-substep floor force/count channels")
        if np.any(np.abs(forces[..., :2]) > 1e-7):
            raise ValueError("Flat-floor NORMAL-only force cannot contain tangential components")
        raw = result["raw_terms"]
        components = reward_numpy(forces[..., 2], self.model_weight_n,
            raw["motion_global_anchor_pos"], raw["drift_speed"], settings=self.ground_settings)
        raw["drift_grounded"] = components["raw_reward"]
        result["weighted_terms"]["drift_grounded"] = self.weights["drift_grounded"]*raw["drift_grounded"]
        result["term_costs"]["drift_grounded"] = self.dt*(self.weights["drift_grounded"]-result["weighted_terms"]["drift_grounded"])
        result["total_reward"] = sum(result["weighted_terms"].values())
        result["stage_cost"] = self.dt*(self.reward_upper_bound-result["total_reward"])
        weights = self.discount_factor**np.arange(shape[1])
        result["candidate_costs"] = np.sum(result["stage_cost"]*weights[None], axis=1)
        result["undiscounted_candidate_costs"] = np.sum(result["stage_cost"], axis=1)
        result["ground_contact_v2"] = components
        if not np.isfinite(result["candidate_costs"]).all():
            raise FloatingPointError("Nonfinite V2f horizon objective")
        return result


class NativeGroundContactTrackingPredictor400Hz(_PopulationAPI):
    """The existing population solver API, only selected via this new module."""
    def __init__(self, scalar_predictor, native_library, *, timeout_s=30.,
                 max_output_bytes=512*1024**2, ground_settings=None, discount_factor=.99):
        self.base, self.layout = scalar_predictor, scalar_predictor.layout
        self.mu_scenarios = tuple(scalar_predictor.mu_scenarios)
        if (not self.mu_scenarios or len(set(self.mu_scenarios)) != len(self.mu_scenarios)
                or not np.isfinite(self.mu_scenarios).all() or min(self.mu_scenarios) < 0):
            raise ValueError("Declare unique finite nonnegative prediction friction scenarios")
        if scalar_predictor.contract.get("physics_dt") != .0025 or scalar_predictor.contract.get("decimation") != 8:
            raise ValueError("Ground V2f requires2.5ms x8 physical clock")
        weights = [model_weight_n(scalar_predictor.models[mu]) for mu in self.mu_scenarios]
        if not np.allclose(weights, weights[0], rtol=0, atol=1e-12):
            raise ValueError("All friction hypotheses must have identical native weight")
        self.objective = BatchGroundContactTrackingObjective400Hz.from_scalar(scalar_predictor.objective,
            model_weight_n=weights[0], ground_settings=ground_settings, discount_factor=discount_factor)
        self.full_reference = {"body_names": self.objective.body_names}
        for name in BODY_FIELDS:
            self.full_reference[name] = np.stack([frame[name] for frame in scalar_predictor.full_references])
        self.workers, self.qids, self.limits, self.limit_sources = {}, {}, {}, {}
        try:
            for mu in self.mu_scenarios:
                model = scalar_predictor.models[mu]
                self.qids[mu] = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
                extractor = scalar_predictor.extractors[mu]
                self.limits[mu], self.limit_sources[mu] = extractor.soft_limits.copy(), extractor.soft_limits_source
                self.workers[mu] = NativeBatchTransition(model, scalar_predictor.provider.q0,
                    scalar_predictor.contract, native_library, timeout_s=timeout_s, max_output_bytes=max_output_bytes)
        except BaseException:
            self.close(); raise
        self.metadata = {"kind": "native_batch_ground_contact_reward_v2f_400hz", "native_abi": 3,
            "physics_dt_s": .0025, "control_dt_s": .02, "physical_substeps": 8,
            "prediction_mu_scenarios": list(self.mu_scenarios), "production_cem_wired": False,
            "learned_initialization": False, "terminal_cost": None,
            "horizon_discount_factor": discount_factor, "objective": self.objective.contract(),
            "workers": {str(mu): copy.deepcopy(worker.metadata) for mu, worker in self.workers.items()}}


BatchHeightSmoothTrackingObjective = BatchGroundContactTrackingObjective400Hz
NativeBatchTrackingPredictor = NativeGroundContactTrackingPredictor400Hz
