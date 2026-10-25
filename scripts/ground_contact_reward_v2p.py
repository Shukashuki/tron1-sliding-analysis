"""P: retain K and add an ungated, bounded peak-impact cost above existing3mg.

The8 actual native floor-normal samples are reused. No force-slew, velocity,
height, zero-slip, bilateral contact, actuator or observation target is added.
Arithmetic does not establish trained behavior or continuous support.
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
_FROZEN_SETTINGS = {**previous.DEFAULT_SETTINGS, 'candidate': 'ground_contact_v2p',
    'ungated_peak_impact_weight': 1., 'ungated_peak_impact_aggregation': 'max8_maxwheel'}
DEFAULT_SETTINGS = copy.deepcopy(_FROZEN_SETTINGS)


def verify_base():
    if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() != BASE_K_DEFINITION_SHA256:
        raise ValueError('Frozen K arithmetic differs')
    previous.verify_base()


verify_base()


def validate_settings(settings=None):
    value = _FROZEN_SETTINGS if settings is None else settings
    if not isinstance(value, dict) or value.keys() != _FROZEN_SETTINGS.keys():
        raise ValueError('Require complete frozen P settings')
    for name, expected in _FROZEN_SETTINGS.items():
        actual = value[name]
        if isinstance(expected, str): valid = isinstance(actual, str) and actual == expected
        elif isinstance(expected, int): valid = type(actual) is int and actual == expected
        else: valid = isinstance(actual, Real) and not isinstance(actual, (bool, np.bool_)) and math.isfinite(float(actual)) and actual == expected
        if not valid: raise ValueError('Frozen P setting differs: '+name)
    return copy.deepcopy(value)


def formula_metadata(settings=None):
    result = copy.deepcopy(previous.formula_metadata())
    result.update(settings=validate_settings(settings), raw_reward_bounds=[-2., 1.],
        formula=result['formula']+'; peak=max_substeps_wheels(impact_excess**2); '+
            'P_impact_cost=1-exp(-peak); raw_P=raw_K-P_impact_cost',
        immutable_v2k_definition_sha256=BASE_K_DEFINITION_SHA256,
        only_causal_change='Subtract bounded ungated peak-impact cost above existing3mg from K',
        ungated_peak_impact_weight=1., ungated_peak_impact_aggregation='max8_maxwheel',
        impact_threshold_mass_multiple=3., peak_impact_tracking_or_support_gated=False,
        original_positive_impact_gate_unchanged=True, K_support_margin_unchanged=True,
        support_gap_coefficient_current=1., other14_reward_terms_changed=False,
        force_slew_cost_added=False, no_new_force_or_velocity_inputs=True,
        continuous_contact_or_task_success_certified=False)
    return result


def _replace(result, peak, *, torch_backend=False):
    cost = -(-peak).expm1() if torch_backend else -np.expm1(-peak)
    # Preserve the parent's whole-window invalid-input poison.
    cost = cost+0.*result['raw_reward']
    result['K_raw_reward_before_peak_impact'] = result['raw_reward']
    result['ungated_peak_impact_squared_excess'] = peak
    result['ungated_peak_impact_cost'] = cost
    result['raw_reward'] = result['raw_reward']-cost
    return result


def reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None):
    validate_settings(settings)
    result = previous.reward_numpy(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed)
    return _replace(result, np.square(result['impact_excess']).max(axis=(-2, -1)))


def reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, *, settings=None, validate=True):
    validate_settings(settings)
    result = previous.reward_torch(Fz_ground, mg, anchor_xy, drift_speed, beta_quality, reference_speed, validate=validate)
    return _replace(result, result['impact_excess'].square().amax(dim=(-2, -1)), torch_backend=True)
