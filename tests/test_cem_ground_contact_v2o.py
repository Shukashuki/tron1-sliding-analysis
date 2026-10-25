"""Same O stage/horizon command cost and explicit before-command load memory."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"tests"))
import ground_contact_reward_v2o as definition
import nmpc_batch_ground_contact_v2k_400hz as parent
import nmpc_batch_ground_contact_v2o_400hz as candidate
import run_cem_ground_contact_v2k as parent_runner
import run_cem_ground_contact_v2o as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture, evaluate
from test_cem_ground_contact_v2m import objective


@pytest.mark.parametrize("discount", [.8, .99, 1.])
def test_other14_terms_unchanged_and_every_stage_uses_preceding_force(discount):
    native, reference, profile, actions = fixture()
    frozen = copy.deepcopy(native)
    k = parent.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=discount)
    o = candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=discount)
    with pytest.raises(ValueError, match="pre-command load"): evaluate(o, native, reference, actions)
    o.set_initial_ground_Fz([0., 100.])
    before = evaluate(k, native, reference, actions); after = evaluate(o, native, reference, actions)
    preceding = np.empty((3, 5, 2)); preceding[:, 0] = [0., 100.]
    preceding[:, 1:] = native["ground_normal_force_w"][:, :-1, -1, :, 2]
    previous = np.empty_like(actions); previous[:, 0] = .1; previous[:, 1:] = actions[:, :-1]
    penalty = definition.command_numpy(preceding, actions, previous)["stance_command_penalty"]
    np.testing.assert_array_equal(after["ground_contact_preceding_Fz_n"], preceding)
    np.testing.assert_array_equal(after["ground_contact_v2"]["stance_command_penalty"], penalty)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        assert len(after[channel]) == 15
        for name in before[channel]:
            if name != "drift_grounded": np.testing.assert_array_equal(after[channel][name], before[channel][name])
    np.testing.assert_allclose(after["stage_cost"]-before["stage_cost"], .02*.4*penalty, atol=4e-16, rtol=1e-12)
    np.testing.assert_allclose(after["candidate_costs"]-before["candidate_costs"],
        (.02*.4*penalty*discount**np.arange(5)).sum(-1), atol=7e-16, rtol=1e-12)
    for name in native: np.testing.assert_array_equal(native[name], frozen[name])
    o.set_initial_ground_Fz([0., 0.])
    no_prior = evaluate(o, native, reference, actions)
    np.testing.assert_array_equal(no_prior["ground_contact_v2"]["stance_command_penalty"][:, 0], np.zeros(3))


@pytest.mark.parametrize("prior", [[0., 0.], [100., 0.], [0., 100.], [100., 100.]])
def test_actual_stage_matches_core_formula_and_keeps_beta_reference(prior):
    legacy = objective(); old = np.zeros(8); action = np.zeros(8); action[[0, 2, 4]] = .1
    force = np.zeros((8, 2, 3)); force[:, 0, 2] = 100.
    before = np.zeros((2, 3)); before[:, 2] = prior
    k = parent_runner.replace_actual_grounded(legacy, force, 100., reference_speed=1.)
    o = runner.replace_actual_grounded(legacy, force, 100., reference_speed=1.,
        preceding_ground_force_w=before, action=action, previous_action=old)
    penalty = float(definition.command_numpy(prior, action, old)["stance_command_penalty"])
    assert o["raw_terms"]["drift_grounded"] == pytest.approx(k["raw_terms"]["drift_grounded"]-penalty)
    assert o["stage_cost"]-k["stage_cost"] == pytest.approx(.02*.4*penalty, abs=3e-16)
    assert o["ground_contact_command_provenance"]["actual_pre_command_ground_normal_force_w"] == before.tolist()
    for name in k["raw_terms"]:
        if name != "drift_grounded": assert o["raw_terms"][name] == k["raw_terms"][name]


def test_predictor_cannot_reuse_a_stale_actual_force_or_use_tangential_force():
    _, _, profile, _ = fixture()
    instance = candidate.NativeGroundContactTrackingPredictor400Hz.__new__(candidate.NativeGroundContactTrackingPredictor400Hz)
    instance.objective = candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100.)
    instance.pre_command_frame = None
    with pytest.raises(ValueError, match="pre-command force"): instance.evaluate_batch(None, None, 0)
    force = np.zeros((2, 3)); force[0, 2] = 100.
    instance.set_preceding_ground_force(force, 2)
    force[:] = 999.
    np.testing.assert_array_equal(instance.objective.initial_ground_Fz, [100., 0.])
    with patch.object(parent.NativeGroundContactTrackingPredictor400Hz, "evaluate_batch", return_value="test result"):
        assert instance.evaluate_batch(None, None, 2) == "test result"
        with pytest.raises(ValueError, match="stale"): instance.evaluate_batch(None, None, 3)
    with pytest.raises(ValueError): instance.set_preceding_ground_force(force, 3)
    assert runner.sha256(ROOT/"config/ppo_ground_contact_v2o_400hz_v1.json") == runner.PROTOCOL_SHA256
