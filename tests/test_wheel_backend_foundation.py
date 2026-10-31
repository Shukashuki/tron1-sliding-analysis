"""Physical invariants that catch timing, contact-frame and fixture mistakes."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_wheel_backend_foundation import canonical_state, simulate, summarize

pytest.importorskip('mujoco')


def test_frictionless_floor_cannot_remove_lateral_momentum():
    trace, _, _ = simulate(canonical_state(), .0025, duration=.2, mu=0)
    np.testing.assert_allclose(trace['root_state_w'][:, 8], canonical_state()[8], atol=1e-12)
    assert np.max(np.abs(trace['interval_force_w_n'][:, :2])) < 1e-10


def test_same_interval_force_closes_translational_momentum():
    trace, _, _ = simulate(canonical_state(), .0025, duration=.2)
    metrics = summarize(trace)
    assert metrics['all_axes_momentum_residual_peak_n'] < 1e-9
    assert trace['root_state_w'][-1, 8] < trace['root_state_w'][0, 8]
    assert np.min(trace['interval_force_w_n'][:, 2]) >= -1e-10


def test_reflected_lateral_velocity_reflects_lateral_trajectory():
    initial = canonical_state()
    a, _, _ = simulate(initial, .0025, duration=.2)
    initial[8] *= -1
    b, _, _ = simulate(initial, .0025, duration=.2)
    np.testing.assert_allclose(a['root_state_w'][:, 1], -b['root_state_w'][:, 1], atol=1e-8)
    np.testing.assert_allclose(a['root_state_w'][:, 0], b['root_state_w'][:, 0], atol=1e-8)


def test_airborne_motion_is_not_counted_as_loaded_sliding():
    initial = canonical_state(); initial[2] = 1.
    trace, _, _ = simulate(initial, .0025, duration=.1)
    assert summarize(trace)['sliding_bottom_proxy']['intervals'] == 0
    assert summarize(trace)['normal_force_active_fraction'] == 0


def test_force_shift_breaks_momentum_balance():
    trace, _, _ = simulate(canonical_state(), .0025, duration=.2)
    aligned = summarize(trace)['vertical_momentum_residual_rms_n']
    trace['interval_force_w_n'] = np.roll(trace['interval_force_w_n'], 1, axis=0)
    assert summarize(trace)['vertical_momentum_residual_rms_n'] > max(1e-3, 1000*aligned)
