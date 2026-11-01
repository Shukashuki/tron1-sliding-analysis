"""Small deterministic checks for the finite-difference SQP controller."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from sqp_shooting import SQPConfig, SQPController


class QuadraticPredictor:
    def rollout_batch(self, state, plans, start_index, *, mu, previous_action, previous_vz):
        assert mu == .7
        return np.square(plans-.3).sum(axis=(1, 2))

    def initial_candidate(self, state, start_index, *, previous_action,
                          previous_vz, horizon_steps):
        return np.zeros((horizon_steps, 1))


def test_sqp_reaches_known_bounded_solution_and_warm_starts():
    controller = SQPController(QuadraticPredictor(),
        SQPConfig(horizon_steps=3, control_blocks=3, max_iterations=10),
        action_low=np.array([-1.]), action_high=np.array([1.]))
    action, info = controller.action(np.array([0.]), 0,
        previous_action=np.array([0.]), previous_vz=0.)
    assert action[0] == pytest.approx(.3, abs=.01)
    assert info["best_cost"] < 1e-4
    assert info["slsqp_success"]
    next_action, next_info = controller.action(np.array([0.]), 1,
        previous_action=action, previous_vz=0.)
    assert next_action[0] == pytest.approx(.3, abs=.01)
    assert next_info["initial_seed_source"] in ("previous_action", "shifted_previous_plan")


def test_sqp_rejects_malformed_friction_or_action_bounds():
    with pytest.raises(ValueError):
        SQPController(QuadraticPredictor(), SQPConfig(horizon_steps=3),
            action_low=np.array([1.]), action_high=np.array([-1.]))
    with pytest.raises(ValueError):
        SQPController(QuadraticPredictor(), SQPConfig(horizon_steps=3),
            action_low=np.array([-1.]), action_high=np.array([1.]),
            mu_scenarios=(.3, .7))
