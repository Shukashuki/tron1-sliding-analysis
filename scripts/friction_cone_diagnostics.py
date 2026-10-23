"""Pure recorded-force diagnostics, NOT a reward or task acceptance rule.

Inputs are forces in N in an explicitly declared two-tangent basis. No impulse
conversion, material inference, missing-data imputation or simulator calls are
performed. L2 is a translational Coulomb-reference ellipse. The actual aligned
MuJoCo condim3 pyramidal boundary instead uses L1; condim4/6 full boundaries
cannot be inferred from these three translational force components.
"""
from __future__ import annotations

import numpy as np

BASIS_NAMES = ("contact_tangent_1_2", "wheel_longitudinal_lateral", "world_xy_flat_floor")
DATA_SCOPES = ("per_contact", "wheel_resultant_lower_bound")
ENGINE_BOUNDARIES = (None, "elliptic_condim3", "mujoco_pyramidal_condim3")


def _require(value, message):
    if not value: raise ValueError(message)


def _array(value, name):
    array = np.asarray(value)
    _require(array.dtype.kind in "fiu" and array.dtype.kind != "b", name+" must be real numeric data")
    return array.astype(np.float64, copy=True)


def _threshold(value, name, *, upper=None):
    _require(not isinstance(value, (bool, np.bool_)) and np.ndim(value) == 0,
             name+" must be a finite nonnegative scalar")
    result = float(value)
    _require(np.isfinite(result) and result >= 0 and (upper is None or result <= upper),
             name+" must be a finite nonnegative scalar")
    return result


def contact_diagnostics(fn, ft, mu, *, force_units, slip_m_s=None,
        tangent_basis="contact_tangent_1_2", wheel_lateral_slip_m_s=None,
        engine_boundary=None, condim=None, engine_tangents_aligned=False,
        data_scope="per_contact", normal_threshold_n=1e-6,
        saturation_threshold=.9, slip_threshold_m_s=.05):
    """Analyze arrays fn[...,C], ft[...,C,2], mu broadcastable to ft's shape.

    Optional slip_m_s has ft's shape; it means relative tangent velocity in the
    SAME basis, not wheel-lateral velocity by default. A separately measured
    wheel_lateral_slip_m_s must have fn's shape (NaN marks unavailable entries).
    force_units is deliberately required: only 'N' is accepted, never impulses.
    Engine-boundary claims require per-contact condim3 and caller-verified aligned
    engine tangents. Wheel resultants permit only a lower-bound L2 reference.

    Invalid/unloaded contacts yield NaN ratios with explicit masks, never rho=0.
    Missing required arrays return available=False. Optional missing/nonfinite
    slip never becomes static/rolling evidence. Near-limit/slip cutoffs are
    exploratory descriptive thresholds, not either task's frozen acceptance.
    """
    _require(force_units == "N", "Explicit force_units='N' required; impulses are not auto-converted")
    _require(tangent_basis in BASIS_NAMES, "Unknown explicitly declared tangent basis")
    _require(data_scope in DATA_SCOPES, "Unknown data scope")
    _require(engine_boundary in ENGINE_BOUNDARIES, "Only explicit translational condim3 engine boundaries are supported")
    _require(type(engine_tangents_aligned) is bool, "engine_tangents_aligned must be an explicit boolean")
    if engine_boundary is not None:
        _require(data_scope == "per_contact", "A resultant is not an actual per-contact engine-cone boundary")
        _require(type(condim) is int and condim == 3 and engine_tangents_aligned,
                 "Actual engine boundary requires condim3 and verified aligned engine tangents; not condim4/6")
    normal_threshold_n = _threshold(normal_threshold_n, "normal_threshold_n")
    saturation_threshold = _threshold(saturation_threshold, "saturation_threshold", upper=1.)
    slip_threshold_m_s = _threshold(slip_threshold_m_s, "slip_threshold_m_s")
    result = {"available": False, "diagnostic_only": True, "task_acceptance_changed": False,
        "force_units": "N", "tangent_basis": tangent_basis, "data_scope": data_scope,
        "actual_per_contact_measurement_claimed": data_scope == "per_contact",
        "input_scope_is_caller_declared": True, "source_provenance_verified_by_math_module": False,
        "resultant_reference_lower_bound_only": data_scope == "wheel_resultant_lower_bound",
        "coulomb_reference_boundary": "elliptic translational L2 norm(ft/mu)/fn",
        "engine_boundary": engine_boundary, "engine_condim": condim,
        "engine_tangents_aligned": engine_tangents_aligned,
        "thresholds": {"normal_force_n_strict": normal_threshold_n,
            "near_boundary_ratio_greater_equal": saturation_threshold,
            "relative_slip_speed_m_s_strict": slip_threshold_m_s},
        "thresholds_are_exploratory_not_task_acceptance": True,
        "wheel_support_screen_separate": "Summed upward wheel Fz>5N is a separate support screen, not this per-point load filter",
        "static_friction_state_inferred": False, "missing_slip_imputed_zero": False,
        "limitations": ["A near-boundary ratio does not itself prove sliding or loss of control.",
            "Nonzero relative slip may occur below a reference boundary; force and motion are independent diagnostics.",
            "Translational L2 is not the full condim4/6 torsional/rolling engine boundary.",
            "A wheel resultant can hide opposing point forces and cannot reveal the actual per-point boundary.",
            "Anisotropic mu components must be expressed in the same declared tangent basis as ft."]}
    missing = [name for name, value in (("fn", fn), ("ft", ft), ("mu", mu)) if value is None]
    if missing:
        result.update(reason="missing required force/material arrays", missing=missing)
        return result
    normal, tangent, friction = _array(fn, "fn"), _array(ft, "ft"), _array(mu, "mu")
    _require(normal.ndim >= 1 and tangent.shape == (*normal.shape, 2), "Expected fn[...,C], ft[...,C,2]")
    _require(friction.ndim >= 1 and friction.shape[-1] == 2, "Explicit two-component effective mu is required")
    try: friction = np.broadcast_to(friction, tangent.shape).copy()
    except ValueError as error: raise ValueError("Effective mu must broadcast to ft[...,C,2]") from error
    if normal.shape[-1] == 0:
        result.update(reason="no contact samples", contact_shape=list(normal.shape))
        return result
    normal_valid = np.isfinite(normal) & (normal >= 0)
    tangent_valid = np.isfinite(tangent).all(-1)
    friction_valid = np.isfinite(friction).all(-1) & (friction > 0).all(-1)
    loaded = normal_valid & (normal > normal_threshold_n)
    small_load = normal_valid & ~loaded
    valid = loaded & tangent_valid & friction_valid
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = tangent/friction
        l2 = np.hypot(scaled[..., 0], scaled[..., 1])/normal
        l1 = np.abs(scaled).sum(-1)/normal
    finite_ratio = np.isfinite(l2)
    l1_valid = valid & np.isfinite(l1)
    valid &= finite_ratio
    l2 = np.where(valid, l2, np.nan)
    engine = np.full(normal.shape, np.nan)
    if engine_boundary == "elliptic_condim3": engine = l2.copy()
    elif engine_boundary == "mujoco_pyramidal_condim3": engine = np.where(l1_valid, l1, np.nan)
    l1 = np.where(l1_valid, l1, np.nan) if engine_boundary == "mujoco_pyramidal_condim3" else np.full(normal.shape, np.nan)
    total_speed = np.full(normal.shape, np.nan)
    slip_valid = np.zeros(normal.shape, dtype=bool)
    tangent_slip = None
    if slip_m_s is not None:
        tangent_slip = _array(slip_m_s, "slip_m_s")
        _require(tangent_slip.shape == tangent.shape, "Relative tangent slip must have ft[...,C,2] shape")
        with np.errstate(over="ignore", invalid="ignore"):
            total_speed = np.hypot(tangent_slip[..., 0], tangent_slip[..., 1])
        slip_valid = loaded & np.isfinite(tangent_slip).all(-1) & np.isfinite(total_speed)
        total_speed = np.where(slip_valid, total_speed, np.nan)
    lateral = np.full(normal.shape, np.nan)
    lateral_source = "unavailable: contact tangent2 is not wheel-lateral"
    if wheel_lateral_slip_m_s is not None:
        measured = _array(wheel_lateral_slip_m_s, "wheel_lateral_slip_m_s")
        _require(measured.shape == normal.shape, "Explicit wheel-lateral slip must have fn[...,C] shape")
        lateral = np.abs(measured)
        lateral_source = "explicit caller-supplied wheel-lateral measurement"
    elif tangent_basis == "wheel_longitudinal_lateral" and tangent_slip is not None:
        lateral = np.abs(tangent_slip[..., 1])
        lateral_source = "tangent2 in explicitly declared wheel-longitudinal/lateral basis"
    lateral_valid = loaded & np.isfinite(lateral)
    lateral = np.where(lateral_valid, lateral, np.nan)
    near = valid & (l2 >= saturation_threshold)
    engine_valid = valid & np.isfinite(engine)
    engine_near = engine_valid & (engine >= saturation_threshold)
    sliding = slip_valid & (total_speed > slip_threshold_m_s)
    lateral_sliding = lateral_valid & (lateral > slip_threshold_m_s)
    result.update(available=True, reason=None, contact_shape=list(normal.shape),
        normal_force_n=normal, tangential_force_n=tangent, effective_mu=friction,
        normal_valid=normal_valid, tangent_force_valid=tangent_valid, effective_mu_valid=friction_valid,
        loaded=loaded, excluded_small_load=small_load, cone_valid_loaded=valid,
        invalid_ratio_numerics=loaded & tangent_valid & friction_valid & ~finite_ratio,
        coulomb_reference_rho_l2=l2, mujoco_aligned_condim3_rho_l1=l1, engine_rho=engine,
        engine_boundary_available=engine_valid,
        coulomb_reference_near_boundary=near, coulomb_reference_above_boundary=valid & (l2 > 1.),
        engine_near_boundary=engine_near, engine_above_boundary=engine_valid & (engine > 1.),
        slip_total_speed_m_s=total_speed, slip_total_valid=slip_valid, observed_total_slip=sliding,
        wheel_lateral_slip_abs_m_s=lateral, wheel_lateral_slip_valid=lateral_valid,
        wheel_lateral_slip_source=lateral_source, observed_wheel_lateral_slip=lateral_sliding,
        reference_near_boundary_with_total_slip=near & sliding,
        reference_near_boundary_low_measured_total_slip=near & slip_valid & ~sliding,
        reference_near_boundary_total_slip_unavailable=near & ~slip_valid,
        any_valid_loaded_contact=bool(valid.any()))
    return result


def _aggregate_scalar(values, mask, normal):
    weight = np.where(mask, normal, 0.)
    denominator = weight.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(denominator > 0, (np.where(mask, values, 0.)*weight).sum(-1)/denominator, np.nan)
    maximum = np.max(np.where(mask, values, -np.inf), axis=-1) if values.shape[-1] else np.full(values.shape[:-1], np.nan)
    maximum = np.where(mask.any(-1), maximum, np.nan)
    return maximum, mean


def aggregate_wheels(result, wheel_ids, *, wheel_names=("L", "R")):
    """Aggregate separate contacts per wheel, never norm the sum of Ft vectors.

    wheel_ids[C] assigns the last contact axis to exactly one named wheel.
    Max and Fn-weighted rho describe loaded VALID contacts; unloaded/invalid
    coverage is reported separately. No valid contact gives NaN, not zero rho.
    Leading batch/time dimensions are retained without temporal averaging.
    """
    _require(isinstance(result, dict), "Require contact_diagnostics result")
    if not result.get("available", False):
        return {"available": False, "reason": result.get("reason", "contact diagnostics unavailable"),
                "diagnostic_only": True, "task_acceptance_changed": False}
    names = tuple(wheel_names)
    _require(len(names) > 0 and len(names) == len(set(names)) and all(isinstance(v, str) for v in names), "Unique wheel names required")
    normal = result["normal_force_n"]
    identities = np.asarray(wheel_ids)
    _require(identities.shape == (normal.shape[-1],) and all(v in names for v in identities.tolist()), "wheel_ids[C] must name every contact")
    outputs = {}
    for name in names:
        select = identities == name
        fn = normal[..., select]
        valid = result["cone_valid_loaded"][..., select]
        loaded = result["loaded"][..., select]
        small = result["excluded_small_load"][..., select]
        rho = result["coulomb_reference_rho_l2"][..., select]
        peak, weighted = _aggregate_scalar(rho, valid, fn)
        engine = result["engine_rho"][..., select]
        engine_valid = result["engine_boundary_available"][..., select]
        engine_peak, engine_weighted = _aggregate_scalar(engine, engine_valid, fn)
        loaded_count, valid_count = loaded.sum(-1), valid.sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            coverage = np.where(loaded_count > 0, valid_count/loaded_count, np.nan)
        outputs[name] = {"contact_count": int(select.sum()), "loaded_contacts": loaded_count,
            "valid_cone_contacts": valid_count, "loaded_cone_count_coverage": coverage,
            "excluded_small_load_contacts": small.sum(-1),
            "excluded_small_load_normal_force_n": np.where(small, fn, 0.).sum(-1),
            "invalid_normal_contacts": (~result["normal_valid"][..., select]).sum(-1),
            "invalid_tangent_force_contacts": (~result["tangent_force_valid"][..., select]).sum(-1),
            "invalid_mu_contacts": (~result["effective_mu_valid"][..., select]).sum(-1),
            "valid_loaded_normal_force_n": np.where(valid, fn, 0.).sum(-1),
            "maximum_coulomb_reference_rho_l2": peak, "normal_load_weighted_coulomb_reference_rho_l2": weighted,
            "maximum_engine_rho": engine_peak, "normal_load_weighted_engine_rho": engine_weighted,
            "near_reference_boundary_contacts": result["coulomb_reference_near_boundary"][..., select].sum(-1),
            "near_engine_boundary_contacts": result["engine_near_boundary"][..., select].sum(-1),
            "observed_total_slip_contacts": result["observed_total_slip"][..., select].sum(-1),
            "observed_wheel_lateral_slip_contacts": result["observed_wheel_lateral_slip"][..., select].sum(-1)}
    return {"available": True, "diagnostic_only": True, "task_acceptance_changed": False,
        "data_scope": result["data_scope"], "thresholds": result["thresholds"],
        "vector_cancellation_used": False, "temporal_averaging_performed": False, "wheels": outputs}
