"""Guard against omitting native tangential world-Z impulse on a curved wheel."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_physx_normal_compliance import vertical_impulse_balance


def test_tangential_vertical_force_participates_in_momentum_balance():
    mass, dt = 2., .0025
    normal = np.array([[0., 0., 0.], [0., 0., 100.]])
    tangent = np.array([[0., 0., 0.], [1., 0., .3]])
    state = np.zeros((2, 13))
    state[1, 9] = dt*(100.+.3-mass*9.81)/mass
    data = dict(root_state_w=state, ground_normal_force_w_n=normal,
        native_tangent_force_w_n=tangent, applied_extra_downward_load_n=np.zeros(1))
    np.testing.assert_allclose(vertical_impulse_balance(data, mass, dt), 0., atol=1e-12)
    data['native_tangent_force_w_n'] = np.zeros((2, 3))
    np.testing.assert_allclose(vertical_impulse_balance(data, mass, dt), .3, atol=1e-12)
