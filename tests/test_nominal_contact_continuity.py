import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_nominal_contact_continuity import interval_statistics, gap_metrics


def test_interval_weighting_and_endpoint_censoring():
    # Irregular intervals reveal sample-average errors; final true endpoint has
    # no extrapolated duration, but is still visible to the audit.
    result = interval_statistics([0., .1, .4, 1.], np.array([True, True, False, True]))
    assert result['duration_s'] == pytest.approx(.4)
    assert result['time_fraction'] == pytest.approx(.4)
    assert result['longest_interval_s'] == pytest.approx(.4)
    assert result['interval_count'] == 1
    assert result['final_endpoint_active']
    assert not result['all_sampled_intervals_active']


def test_load_and_deep_penetration_do_not_certify_contact():
    time = np.array([0., .1, .2, .3])
    gap = np.array([[0., 0.], [.002, 0.], [-.01, 0.], [0., 0.]])
    # Native records deliberately disagree with geometry: retain both, do not
    # fill geometric separation with a load mask or count penetration as valid.
    result = gap_metrics(time, gap, np.ones((4, 2)), np.ones((4, 2)))
    one_mm = result['tolerances']['0.001']
    assert one_mm['bilateral_envelope_near_ground']['time_fraction'] == pytest.approx(2/3)
    assert one_mm['bilateral_in_gap_band']['time_fraction'] == pytest.approx(1/3)
    assert one_mm['any_wheel_separated']['longest_interval_s'] == pytest.approx(.1)
    assert result['native_recorded_contact_eligibility']['both']['time_fraction'] == 1.
