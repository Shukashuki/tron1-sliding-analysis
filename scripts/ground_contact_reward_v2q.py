"""Q retains K and penalizes positive wheel/plane gap only below any-wheel5N.

The20mm exponential length is reward shaping, never a support tolerance or an
absolute body-height target. One supporting wheel disables the new sample cost.
This module alone does not establish trained behavior or task completion.
"""
from __future__ import annotations
import copy
import hashlib
import math
from numbers import Real
from pathlib import Path
import numpy as np
import ground_contact_reward_v2k as previous

base = previous.base
BASE_F_DEFINITION_SHA256 = previous.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256 = previous.BASE_G_DEFINITION_SHA256
BASE_I_DEFINITION_SHA256 = previous.BASE_I_DEFINITION_SHA256
BASE_K_DEFINITION_SHA256 = 'dbea1148c00a394b990964f7e7b26d3a7e9bdbe75e43975b3e9e79e7e0ebe18f'
BASE_K_RUNTIME_SHA256 = '3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d'
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, 'candidate': 'ground_contact_v2q',
    'unsupported_geometry_gap_scale_m': .02, 'unsupported_geometry_gap_weight': 1.,
    'unsupported_geometry_gap_aggregation': 'max8_of_below5N_positive_minwheel_gap'}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_K_DEFINITION_SHA256:
        raise ValueError('Frozen K arithmetic differs')
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError('Require complete frozen Q settings')
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else: valid = isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_)) and math.isfinite(float(actual)) and actual == expected
        if not valid: raise ValueError('Frozen Q setting differs: '+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings), raw_reward_bounds=[-2., 1.],
        formula=result['formula']+'; missing=max_wheel(max(Fz,0))<=5N; '+
            'g=positive(min_wheel(authored_cylinder_plane_gap)); '+
            'Q_cost=max8(missing*(1-exp(-g/.02m))); raw_Q=raw_K-Q_cost',
        immutable_v2k_definition_sha256=BASE_K_DEFINITION_SHA256,
        only_causal_change='Add bounded positive authored-gap cost only in physically below5N samples',
        geometry_gap_scale_m=.02, geometry_gap_scale_is_acceptance_tolerance=False,
        native_point_count_used=False, root_height_target=False, bilateral_target=False,
        no_new_impact_cost=True, K_positive_bonus_and_load_margin_unchanged=True,
        other14_reward_terms_changed=False, continuous_contact_or_task_success_certified=False)
    return result


def verify_geometry_runtime(audit, *, expected_envs, expected_windows=None):
    """Startup/completion only; validate native provenance, never a reward hook."""
    windows = audit.get('steps_completed') if expected_windows is None else expected_windows
    if (type(windows) is not int or windows < 0
            or audit.get('geometry_physics_samples') != windows*8
            or audit.get('geometry_unique_reward_reads') != windows
            or audit.get('device_finite_geometry_health') is not True
            or audit.get('geometry_force_sampling_synchronous') is not True
            or audit.get('geometry_gap_scale_m') != .02
            or audit.get('geometry_gap_scale_is_acceptance_tolerance') is not False
            or audit.get('unsupported_geometry_cost_only_below5N') is not True
            or audit.get('immutable_v2k_definition_sha256') != BASE_K_DEFINITION_SHA256
            or audit.get('immutable_v2k_runtime_sha256') != BASE_K_RUNTIME_SHA256):
        raise ValueError('Q synchronous native geometry sampling/health differs')
    geometry = audit.get('authored_geometry_audit', {})
    required = dict(kind='readonly_authored_cylinder_gap', verified_environment_count=expected_envs,
        actual_wheels_per_environment=2, cpu_fixture_only=False, original_one_cylinder_per_wheel=True,
        no_geometry_or_physics_writes=True, distance_is_authored_geometry_not_native_manifold=True,
        all_environment_frames_checked=True)
    if any(geometry.get(k) != v for k,v in required.items()):
        raise ValueError('Q native all-environment authored geometry audit differs')
    offsets = geometry.get('native_wheel_offsets', {})
    for key, target in (('contact_m', .001), ('rest_m', 0.)):
        values = offsets.get(key)
        if (not isinstance(values, list) or len(values) != 2
                or not all(isinstance(v, Real) and math.isfinite(v) and abs(v-target) < 1e-8 for v in values)):
            raise ValueError('Q native original wheel offsets differ')
    specs = geometry.get('specs')
    if not isinstance(specs, list) or len(specs) != 2:
        raise ValueError('Q wheel geometry specs missing')
    for spec in specs:
        if (abs(spec.get('radius_m', 0.)-.127) > 1e-6
                or abs(spec.get('half_width_m', 0.)-.025) > 1e-6
                or np.asarray(spec.get('local_center_m')).shape != (3,)
                or np.asarray(spec.get('local_axis')).shape != (3,)
                or not np.isfinite(spec['local_center_m']).all()
                or not np.isfinite(spec['local_axis']).all()
                or abs(np.linalg.norm(spec['local_axis'])-1.) > 1e-6):
            raise ValueError('Q original finite-cylinder geometry differs')
    error = geometry.get('startup_CPU_Torch_max_gap_difference_m')
    if not isinstance(error, Real) or not math.isfinite(error) or not 0 <= error <= 2e-6:
        raise ValueError('Q independent geometry parity differs')
    ledger = geometry.get('source_sha256', {})
    for name in ('wheel_ground_geometry.py', 'wheel_ground_geometry_torch.py'):
        if ledger.get(name) != hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest():
            raise ValueError('Q geometry helper provenance differs: '+name)


def _replace(result, samples, minimum, *, torch_backend=False):
    cost = samples.amax(dim=-1) if torch_backend else samples.max(axis=-1)
    cost = cost+0.*result['raw_reward']
    result['K_raw_reward_before_geometry_cost'] = result['raw_reward']
    result['positive_minwheel_geometry_gap_m'] = minimum
    result['unsupported_geometry_gap_cost_samples'] = samples
    result['unsupported_geometry_gap_cost'] = cost
    result['raw_reward'] = result['raw_reward']-cost
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, geometry_gap_m, settings=None):
    validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    gap = np.asarray(geometry_gap_m)
    force = np.asarray(Fz_ground)
    if gap.dtype.kind not in 'fiu' or gap.shape != force.shape or not np.isfinite(gap).all():
        raise ValueError('Require finite authored geometry gaps matching force[...,8,2]')
    minimum = np.maximum(gap.astype(result['raw_reward'].dtype, copy=False).min(axis=-1), 0.)
    samples = (force.max(axis=-1) <= 5.)*(-np.expm1(-minimum/.02))
    return _replace(result, samples, minimum)


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, geometry_gap_m, settings=None, validate=True):
    import torch
    validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, validate=validate)
    force = torch.as_tensor(Fz_ground)
    gap = torch.as_tensor(geometry_gap_m, device=force.device)
    if gap.is_complex() or gap.dtype == torch.bool or gap.shape != force.shape:
        raise ValueError('Require real geometry gaps matching force[...,8,2]')
    gap = gap.to(dtype=force.dtype)
    good = torch.isfinite(gap).all(dim=(-1, -2))
    if validate and not bool(good.all()): raise ValueError('Require finite geometry gap')
    minimum = gap.amin(dim=-1).clamp_min(0.)
    samples = (force.amax(dim=-1) <= 5.).to(force.dtype)*(-torch.expm1(-minimum/.02))
    result = _replace(result, samples, minimum, torch_backend=True)
    if not validate:
        for name, value in result.items():
            mask = good[(...,)+(None,)*(value.ndim-good.ndim)]
            result[name] = torch.where(mask, value, torch.full_like(value, float('nan')))
    return result
