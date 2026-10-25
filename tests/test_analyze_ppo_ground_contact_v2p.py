"""Do not let lower impact alone weaken the original nominal support task."""
from pathlib import Path
from unittest.mock import patch
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_ppo_ground_contact_v2p as audit


def test_real_P_protocol_keeps_every_existing_acceptance_gate_and_DR():
    p = audit.verify_protocol()
    k = audit.read(audit.K_PROTOCOL)
    assert p['nominal_acceptance_before_training'] == k['nominal_acceptance_before_training']
    assert p['training_randomization'] == k['training_randomization']


def test_relaxing_contact_or_speed_gate_is_rejected():
    original = audit.read
    for field, value in (('any_wheel_support_fraction_min', .9), ('core_speed_ratio_range', [.5, 1.1])):
        modified = copy.deepcopy(original(audit.P_PROTOCOL))
        modified['nominal_acceptance_before_training'][field] = value
        with patch.object(audit, 'read', side_effect=lambda p: modified if p == audit.P_PROTOCOL else original(p)):
            with pytest.raises(AssertionError): audit.verify_protocol()


def test_force_peak_does_not_label_initial_readout_as_controlled_impact():
    force = np.zeros((2793, 2, 3)); force[0, 1, 2] = 9000.; force[800, 0, 2] = 1200.
    trace = dict(time_s=np.arange(2793)*.0025, floor_normal_force=force)
    result = audit.force_diagnostics(trace)
    assert result['whole_recording_peak_time_s'] == 0.
    assert result['whole_recording_peak_n'] == 9000.
    assert result['policy_controlled_peak_n'] == result['core_peak_n'] == 1200.
    assert force[0, 1, 2] == 9000.


def test_incomplete_nominal_collection_is_rejected(tmp_path):
    import json
    (tmp_path/'collection_manifest.json').write_text(json.dumps(dict(kind='running_P_smoke_only', files={})))
    with pytest.raises(AssertionError): audit.verify_collection(tmp_path)
