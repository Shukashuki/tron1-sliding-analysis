"""H native-batch arithmetic parity, not a trained policy/controller result."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"tests"))
import nmpc_batch_ground_contact_v2g_400hz as parent
import nmpc_batch_ground_contact_v2h_400hz as candidate
import run_cem_ground_contact_v2g as parent_runner
import run_cem_ground_contact_v2h as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture, evaluate
from test_cem_ground_contact_v2g import objective


@pytest.mark.parametrize("gamma", (.8, .99, 1.))
def test_only_grounded_changes_with_exact_horizon_cost_delta(gamma):
    native, reference, profile, actions = fixture()
    native["body_lin_vel_w"][..., 0, 2] = np.array([.6, 0., -.45, .3, .45, -.6, 0., 0.])
    before, before_ref = copy.deepcopy(native), copy.deepcopy(reference)
    g = evaluate(parent.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
    h = evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        assert len(g[channel]) == len(h[channel]) == 15
        for term in g[channel]:
            if term != "drift_grounded": np.testing.assert_array_equal(g[channel][term], h[channel][term])
    for name in g["ground_contact_v2"].keys()-{"raw_reward"}:
        np.testing.assert_array_equal(g["ground_contact_v2"][name], h["ground_contact_v2"][name])
    loaded = native["ground_normal_force_w"][..., 2].max(-1)>5.
    cost = .5*np.mean(loaded*np.array([1., 0., .25, 0., .25, 1., 0., 0.]), axis=-1)
    np.testing.assert_allclose(h["ground_contact_v2"]["loaded_com_vz_penalty"], cost, atol=1e-16)
    expected = .02*.4*cost
    np.testing.assert_allclose(h["stage_cost"]-g["stage_cost"], expected, atol=4e-16, rtol=1e-12)
    np.testing.assert_allclose(h["candidate_costs"]-g["candidate_costs"],
        np.sum(expected*gamma**np.arange(5), axis=-1), atol=8e-16, rtol=1e-12)
    for name in native: np.testing.assert_array_equal(native[name], before[name])
    for name in reference: np.testing.assert_array_equal(reference[name], before_ref[name])


@pytest.mark.parametrize("missing", (0, 1, 3, 8))
@pytest.mark.parametrize("beta", (0., .37, 1.))
def test_actual_plant_uses_all8_velocity_and_preserves_other14_terms(missing, beta):
    legacy = objective(); legacy["raw_terms"]["drift_sideslip"] = beta
    before = copy.deepcopy(legacy)
    force = np.zeros((8, 2, 3)); force[:, 0, 2] = 100.; force[:missing] = 0.
    vz = np.array([.6, -.6, 0., .3, .45, -.45, 0., 0.])
    g = parent_runner.replace_actual_grounded(legacy, force, 100., reference_speed=1.)
    h = runner.replace_actual_grounded(legacy, force, 100., reference_speed=1., base_com_vz=vz)
    expected = .5*np.mean((force[..., 2].max(-1)>5.)*np.array([1., 1., 0., 0., .25, .25, 0., 0.]))
    assert legacy == before
    assert h["ground_contact_v2"]["loaded_com_vz_penalty"] == pytest.approx(expected)
    assert h["raw_terms"]["drift_grounded"] == pytest.approx(g["raw_terms"]["drift_grounded"]-expected)
    assert h["stage_cost"]-g["stage_cost"] == pytest.approx(.02*.4*expected, abs=3e-16)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        for term in g[channel]:
            if term != "drift_grounded": assert h[channel][term] == g[channel][term]


@pytest.mark.parametrize("velocity", (np.zeros(7), np.zeros((8, 1)),
    np.full(8, np.nan), np.full(8, np.inf), np.zeros(8, dtype=bool)))
def test_actual_reward_rejects_missing_bad_or_repeated_shape_velocity(velocity):
    force = np.zeros((8, 2, 3))
    with pytest.raises(ValueError):
        runner.replace_actual_grounded(objective(), force, 100., reference_speed=1., base_com_vz=velocity)


def test_explicit_all8_channel_is_required_and_no_new_native_build_or_policy_is_used():
    force = np.zeros((8, 2, 3))
    with pytest.raises(TypeError): runner.replace_actual_grounded(objective(), force, 100., reference_speed=1.)
    assert runner.sha256(ROOT/"config/ppo_ground_contact_v2h_400hz_v1.json") == runner.PROTOCOL_SHA256
    source = Path(runner.__file__).read_text()
    assert 'raw["body_lin_vel_w"][start+1:end+1, 0, 2]' in source
    assert '[row["body_lin_vel_w"][0, 2] for row in records[-8:]]' in source
    for forbidden in ("torch.load(", "torch.jit.load(", "build_native_library("):
        assert forbidden not in source
    assert '"ground_contact_reward_v2g.py"' in source
    assert '"PPO_actions_or_future_states_used_for_initial_plan": False' in source


def test_batch_contract_declares_com_velocity_not_contactpoint_slip():
    native, reference, profile, actions = fixture()
    objective_h = candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.)
    contract = objective_h.contract()
    assert contract["loaded_com_vz_all8_actual_substeps"] is True
    assert contract["velocity_end_state_repeated"] is False
    assert contract["unchanged_other_reward_terms"] == 14
    assert contract["absolute_height_constraint"] is False
