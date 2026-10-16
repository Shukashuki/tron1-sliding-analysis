"""Pure configuration, measurement and accounting tests; no simulator imports."""
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import adaptive_mpc_runtime as runtime
from adaptive_contact_model import AdaptiveContactConfig


def payload():
    return {"schema_version": 1, "kind": "finite_contact_model_adaptive_mpc",
            "estimator": asdict(AdaptiveContactConfig())}


def load(monkeypatch, value, scenarios=None):
    monkeypatch.setattr(runtime.Path, "read_text", lambda self: json.dumps(value))
    return runtime.load_adaptation("explicit-config.json", scenarios)


def test_explicit_configuration_roundtrips_every_fixed_default(monkeypatch):
    config = load(monkeypatch, payload(), (.3, .7, 1.6))
    assert config == AdaptiveContactConfig()
    assert config.mu_hypotheses == (.3, .7, 1.6)
    assert config.initial_mu == .7


@pytest.mark.parametrize("field", list(asdict(AdaptiveContactConfig())))
def test_missing_any_estimator_field_is_not_silently_defaulted(monkeypatch, field):
    value = payload()
    del value["estimator"][field]
    with pytest.raises(ValueError, match="explicitly declare every estimator field"):
        load(monkeypatch, value)


@pytest.mark.parametrize("level,key", [("top", "plant_mu"), ("estimator", "plant_mu"),
                                       ("top", "implicit_defaults"), ("estimator", "gain")])
def test_extra_keys_and_plant_coefficient_are_not_estimator_inputs(monkeypatch, level, key):
    value = payload()
    (value if level == "top" else value["estimator"])[key] = .7
    with pytest.raises(ValueError):
        load(monkeypatch, value)


@pytest.mark.parametrize("scenarios", [(.7,), (.7, .3, 1.6), (.3, .7, 1.2)])
def test_conflicting_explicit_model_bank_fails(monkeypatch, scenarios):
    with pytest.raises(ValueError, match="conflict"):
        load(monkeypatch, payload(), scenarios)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("kind", "oracle_mpc"),
                                         ("schema_version", True),
                                         ("estimator", []), ("estimator", None)])
def test_wrong_configuration_schema_fails(monkeypatch, field, value):
    document = payload()
    document[field] = value
    with pytest.raises(ValueError):
        load(monkeypatch, document)


def test_configuration_top_level_must_be_a_mapping(monkeypatch):
    with pytest.raises(ValueError):
        load(monkeypatch, ["schema_version", "kind", "estimator"])


def records():
    return [{"time_s": 1.005 + index * .005,
             "wheel_ground_contact_count": np.array([0, 0], dtype=np.int64),
             "nonwheel_ground_contact_count": 0, "action_was_applied": True,
             # Deliberately irrelevant: no force/height/beta substitute.
             "body_net_forces_w": np.full((9, 3), 1e8), "root_height": 50., "beta": 2.}
            for index in range(4)]


def test_actual_any_wheel_contact_in_exact_four_substeps_is_sufficient():
    samples = records()
    samples[2]["wheel_ground_contact_count"] = np.array([0, 2])
    assert runtime.transition_context(samples) == {
        "wheel_ground_contact_observed": True,
        "nonwheel_ground_contact_observed": False, "duration_s": .02}
    samples[0]["nonwheel_ground_contact_count"] = 1
    assert runtime.transition_context(samples)["nonwheel_ground_contact_observed"] is True


def test_flight_is_valid_unavailable_identification_evidence_not_task_failure():
    context = runtime.transition_context(records())
    assert context == {"wheel_ground_contact_observed": False,
                       "nonwheel_ground_contact_observed": False, "duration_s": .02}
    assert not any("fail" in key or "height" in key or "beta" in key for key in context)


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_context_requires_exactly_four_actual_samples(count):
    samples = (records() + records())[:count]
    with pytest.raises(ValueError, match="exactly four"):
        runtime.transition_context(samples)


@pytest.mark.parametrize("field,value", [
    ("time_s", np.nan), ("time_s", 1.011),
    ("wheel_ground_contact_count", np.array([-1, 0])),
    ("wheel_ground_contact_count", np.array([1., 0.])),
    ("wheel_ground_contact_count", np.array([True, False])),
    ("wheel_ground_contact_count", np.array([1, 0, 0])),
    ("nonwheel_ground_contact_count", -1), ("nonwheel_ground_contact_count", 1.),
    ("action_was_applied", False), ("action_was_applied", 1),
])
def test_context_rejects_invalid_clock_contact_or_unapplied_samples(field, value):
    samples = records()
    for row in samples if field in ("wheel_ground_contact_count", "nonwheel_ground_contact_count") else samples[1:2]:
        row[field] = value
    with pytest.raises(ValueError):
        runtime.transition_context(samples)


@pytest.mark.parametrize("field", ["time_s", "action_was_applied"])
def test_context_requires_scalar_clock_and_applied_flags_per_sample(field):
    samples = records()
    for row in samples:
        row[field] = [row[field]]
    with pytest.raises(ValueError):
        runtime.transition_context(samples)


class FakeScalarPredictor:
    def __init__(self):
        self.contract = {"physics_dt": .005, "decimation": 4}
        self.measured = {"body_lin_vel_w": np.zeros((9, 3)),
                         "body_ang_vel_w": np.zeros((9, 3)),
                         "qvel": np.full(14, 999.), "root_origin_velocity_w": np.full(3, 888.)}
        self.measured["body_lin_vel_w"][0] = [1., -2., 3.]
        self.measured["body_ang_vel_w"][0] = [4., 5., -6.]
        self.calls = []
        self.predicted_state = np.arange(111, dtype=float)

    def measure(self, packed):
        self.calls.append(("measure", packed))
        return self.measured

    def transition(self, packed, action, *, mu):
        self.calls.append(("transition", packed, action, mu))
        return self.predicted_state


def test_features_use_measured_base_com_and_world_yaw_not_origin_or_generalized_velocity():
    predictor = FakeScalarPredictor()
    model = runtime.ScalarIdentificationModel(predictor)
    state = np.zeros(111)
    np.testing.assert_array_equal(model.features(state), [1., -2., -6.])
    assert predictor.calls == [("measure", state)]


@pytest.mark.parametrize("field,column", [("body_lin_vel_w", 0), ("body_lin_vel_w", 1),
                                          ("body_ang_vel_w", 2)])
def test_features_reject_nonfinite_measured_channels(field, column):
    predictor = FakeScalarPredictor()
    predictor.measured[field][0, column] = np.nan
    with pytest.raises(ValueError):
        runtime.ScalarIdentificationModel(predictor).features(np.zeros(111))


@pytest.mark.parametrize("field,shape", [("body_lin_vel_w", (9, 1)),
                                        ("body_lin_vel_w", (1, 3)),
                                        ("body_ang_vel_w", (9, 2)),
                                        ("body_ang_vel_w", (3,))])
def test_features_reject_wrong_measured_body_velocity_shape(field, shape):
    predictor = FakeScalarPredictor()
    predictor.measured[field] = np.zeros(shape)
    with pytest.raises(ValueError):
        runtime.ScalarIdentificationModel(predictor).features(np.zeros(111))


@pytest.mark.parametrize("changes", [{"physics_dt": .01}, {"decimation": 2}])
def test_identification_requires_same_four_step_transition_contract(changes):
    predictor = FakeScalarPredictor()
    predictor.contract.update(changes)
    with pytest.raises(ValueError, match="four-substep"):
        runtime.ScalarIdentificationModel(predictor)


def test_transition_features_forwards_all_eight_actual_float32_actions_and_declared_mu_once():
    predictor = FakeScalarPredictor()
    model = runtime.ScalarIdentificationModel(predictor)
    action = np.linspace(-.9, .9, 8).astype(np.float32).astype(float)
    before = action.copy()
    state = np.linspace(0., 1., 111)
    features = model.transition_features(state, action, mu=1.6)
    np.testing.assert_array_equal(features, [1., -2., -6.])
    assert len(predictor.calls) == 2
    call, measured = predictor.calls
    assert call[0] == "transition" and call[1] is state and call[2] is action and call[3] == 1.6
    assert measured[0] == "measure" and measured[1] is predictor.predicted_state
    np.testing.assert_array_equal(action, before)
    np.testing.assert_array_equal(action, action.astype(np.float32).astype(float))


def test_summary_accounts_actual_per_row_identification_steps_and_timing_without_renaming():
    rows = []
    for status, selected, updated, calls, steps, elapsed in [
            ("initial_prior", .7, False, 0, 0, .001),
            ("insufficient_evidence", .7, False, 3, 12, .002),
            ("switched", .3, True, 3, 12, .003),
            ("no_contact", .3, False, 0, 0, .004)]:
        rows.append({"solver": {"adaptive_contact": {
            "status": status, "selected_mu": selected, "updated": updated,
            "identification_prediction_calls": calls,
            "identification_physics_substeps": steps,
            "estimation_compute_s": elapsed}}})
    summary = runtime.adaptation_summary(rows)
    assert summary["decisions"] == 4 and summary["model_switches"] == 1
    assert summary["selected_model_counts"] == {"0.7": 2, "0.3": 2}
    assert summary["identification_prediction_calls"] == 6
    assert summary["identification_predicted_physics_substeps"] == 24
    assert "identification_physics_substeps" not in summary
    assert summary["status_counts"] == {"initial_prior": 1, "insufficient_evidence": 1,
                                         "switched": 1, "no_contact": 1}
    np.testing.assert_allclose(summary["identification_compute_ms_median_p95_max"], [2.5, 3.85, 4.])
    assert summary["included_in_controller_compute_s"] is True
    assert summary["physical_friction_estimate_claimed"] is False
    assert summary["robustness_certificate"] is False
    json.dumps(summary, allow_nan=False)


def test_runtime_source_does_not_import_simulator_or_learned_inference_dependencies():
    source = Path(runtime.__file__).read_text()
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    assert not set(imports).intersection({"mujoco", "torch", "isaacsim", "isaaclab", "pybullet"})
