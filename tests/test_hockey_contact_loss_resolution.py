"""Check that the candidate distinguishes a single missed physical sample."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_hockey_contact_loss_resolution import contact_scores


def test_any_missed_sample_has_negative_contact_subset_without_erasing_good_feedback():
    both = np.arange(8)[None, :] >= np.arange(9)[:, None]
    result = contact_scores(both, np.zeros(9),
                            dict(contact_fraction=2., contact_loss=-4., clearance=-2.))
    assert result['proposed_contact_score'][0] == result['current_contact_score'][0] == 2
    assert result['proposed_contact_score'][-1] == result['current_contact_score'][-1] == -4
    assert np.all(result['proposed_contact_score'][1:] < 0)
    assert np.all(np.diff(result['proposed_contact_score']) < 0)
    assert result['current_contact_score'][1] > 0
