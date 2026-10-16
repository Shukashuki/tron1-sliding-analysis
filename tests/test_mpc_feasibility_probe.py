from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_mpc_feasibility import ShortProblem, yaw_error


class FakePredictor:
    """Numerical fixture, not a replacement for the actual-model probe."""
    def __init__(self, height=.75, vz=0.):
        self.dt, self.n = .02, 20
        self.model = SimpleNamespace(geom_friction=np.array([[.7, 0., 0.]]))
        self.targets = np.tile([1., 0., height, 0., 0., vz], (self.n, 1))
        self.nominal_action = np.zeros((self.n, 8))
        self.state = self.targets[0].copy()
        self.api = SimpleNamespace(rotation=lambda q: np.eye(3))
        self.measure = SimpleNamespace(measure=self.measure_now)
        self.reference = SimpleNamespace(sample=lambda k: {
            "root_pos": np.array([0., 0., 999.]), "root_quat_wxyz": np.array([1., 0., 0., 0.]),
            "root_com_lin_vel_w": np.zeros(3)})
        self.applied = []

    def measure_now(self):
        return {"root_pos": self.state[:3].copy(), "root_com_lin_vel_w": self.state[3:].copy(),
                "root_quat_wxyz": np.array([1., 0., 0., 0.])}

    def transition(self, state, action):
        self.state = state.copy()
        self.state[:2] += action[:2] * self.dt
        self.state[2] += self.state[5] * self.dt
        self.applied.append((self.model.geom_friction[0, 0], action.copy()))
        return self.state.copy()


@pytest.fixture
def problem_factory(monkeypatch):
    def measure(self, state):
        self.p.state = state.copy()
        return self.p.measure_now()
    monkeypatch.setattr(ShortProblem, "measure_state", measure)
    return lambda p, mus: ShortProblem(p, 0, 5, 2, mus)


def test_yaw_wrap_does_not_penalize_full_revolutions():
    assert yaw_error(np.pi - .1, -np.pi + .1) == pytest.approx(-.2)
    assert yaw_error(2*np.pi, 0.) == pytest.approx(0.)


def test_probe_has_no_height_or_zero_vz_cost(problem_factory):
    a = problem_factory(FakePredictor(height=.75, vz=0.), [.7])
    b = problem_factory(FakePredictor(height=100., vz=3.), [.7])
    np.testing.assert_allclose(a.costs(a.seed()), b.costs(b.seed()))


def test_scenarios_share_controls_and_restore_predictor_mu(problem_factory):
    p = FakePredictor()
    problem = problem_factory(p, [.3, .7, 1.6])
    seed = problem.seed()
    problem.costs(seed)
    assert p.model.geom_friction[0, 0] == .7
    assert len(p.applied) == 15
    for j in range(5):
        np.testing.assert_array_equal(p.applied[j][1], p.applied[j+5][1])
        np.testing.assert_array_equal(p.applied[j][1], p.applied[j+10][1])
    problem.costs(seed.copy())
    assert len(p.applied) == 15  # Same optimizer query is cached.


@pytest.mark.parametrize("mus", [[.7], [.3, .7, 1.6]])
def test_short_box_constrained_optimization_improves_fixture_cost(problem_factory, mus):
    result = problem_factory(FakePredictor(), mus).solve(20)
    assert result["success"]
    assert result["worst_cost_improved"]
    assert result["action_bound_violation"] == 0.
    if len(mus) > 1:
        assert result["epigraph_constraint_violation"] < 1e-5


def test_recorded_actual_model_probe_has_provenance_and_is_not_a_benchmark():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "results/2026-10-15-mpc-feasibility.json").read_text())
    assert record["probe_source_sha256"] == hashlib.sha256((root / "scripts/probe_mpc_feasibility.py").read_bytes()).hexdigest()
    assert record["predictor_source_sha256"] == hashlib.sha256((root / "scripts/classical_tracking.py").read_bytes()).hexdigest()
    assert record["absolute_height_cost"] is False and record["reference_vz_cost"] is False
    assert record["closed_loop_tracking_test"] is False
    assert record["real_time_test"] is False
    assert record["guaranteed_robust_control"] is False
    assert record["diagnostic_cost_exact_ppo_parity"] is False
    assert len(record["short_optimizations"]) == 4
    for result in record["short_optimizations"]:
        assert result["success"] and result["worst_cost_improved"]
        assert result["action_bound_violation"] == 0.
        assert result["epigraph_constraint_violation"] is None or result["epigraph_constraint_violation"] < 1e-5
