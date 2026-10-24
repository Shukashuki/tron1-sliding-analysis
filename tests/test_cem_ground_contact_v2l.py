"""L CEM tests: same native ABI, exact other14 terms, added physical-force cost only."""
import copy
from pathlib import Path
import sys
import inspect

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"tests"))
import nmpc_batch_ground_contact_v2k_400hz as parent
import nmpc_batch_ground_contact_v2l_400hz as candidate
import run_cem_ground_contact_v2k as parent_runner
import run_cem_ground_contact_v2l as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture, evaluate
from test_cem_ground_contact_v2k import objective


@pytest.mark.parametrize("gamma", (.8, .99, 1.))
def test_only_grounded_changes_and_horizon_cost_addition_exact(gamma):
    native, reference, profile, actions = fixture()
    frozen = copy.deepcopy(native); ref_before = copy.deepcopy(reference)
    before = evaluate(parent.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=gamma),
                      native, reference, actions)
    after = evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile, model_weight_n=100., discount_factor=gamma),
                     native, reference, actions)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        assert len(before[channel]) == len(after[channel]) == 15
        for term in before[channel]:
            if term != "drift_grounded":
                np.testing.assert_array_equal(before[channel][term], after[channel][term])
    for key in before["ground_contact_v2"].keys()-{"raw_reward"}:
        np.testing.assert_array_equal(before["ground_contact_v2"][key], after["ground_contact_v2"][key])
    force=np.maximum(native["ground_normal_force_w"][...,2],0.).sum(-1)
    expected=np.minimum((np.diff(force,axis=-1)/100.)**2,1.).mean(-1)
    np.testing.assert_allclose(after["ground_contact_v2"]["normal_force_slew_cost"],expected)
    np.testing.assert_allclose(after["stage_cost"]-before["stage_cost"],.02*.4*expected,rtol=1e-12,atol=4e-16)
    np.testing.assert_allclose(after["candidate_costs"]-before["candidate_costs"],
                              (.02*.4*expected*gamma**np.arange(5)).sum(-1),rtol=1e-12,atol=6e-16)
    for key in native:np.testing.assert_array_equal(native[key],frozen[key])
    for key in reference:np.testing.assert_array_equal(reference[key],ref_before[key])


@pytest.mark.parametrize("missing",(0,1,3,8))
@pytest.mark.parametrize("beta,refspeed",((0.,1.),(.37,1.),(1.,1.),(0.,.3),(0.,0.)))
@pytest.mark.parametrize("load",(10.,25.,100.))
def test_actual_stage_same_beta_and_margin_with_only_slew_added(missing,beta,refspeed,load):
    legacy=objective();legacy["raw_terms"]["drift_sideslip"]=beta
    original=copy.deepcopy(legacy)
    force=np.zeros((8,2,3));force[:,0,2]=load;force[:missing]=0.
    before=parent_runner.replace_actual_grounded(legacy,force,100.,reference_speed=refspeed)
    after=runner.replace_actual_grounded(legacy,force,100.,reference_speed=refspeed)
    expected=np.minimum((np.diff(force[...,2].sum(-1))/100.)**2,1.).mean()
    for key in before["ground_contact_v2"].keys()-{"raw_reward"}:
        np.testing.assert_array_equal(before["ground_contact_v2"][key],after["ground_contact_v2"][key])
    assert after["ground_contact_v2"]["normal_force_slew_cost"]==pytest.approx(expected)
    assert after["raw_terms"]["drift_grounded"]==pytest.approx(before["raw_terms"]["drift_grounded"]-expected)
    for channel in ("raw_terms","weighted_terms","term_costs"):
        for term in before[channel]:
            if term!="drift_grounded":assert before[channel][term]==after[channel][term]
    assert after["stage_cost"]-before["stage_cost"]==pytest.approx(.02*.4*expected,abs=3e-16)
    assert legacy==original


def test_no_new_native_input_or_ABI_and_exact_protocol_pin():
    assert inspect.signature(candidate.BatchGroundContactTrackingObjective400Hz.evaluate_native) == inspect.signature(
        parent.BatchGroundContactTrackingObjective400Hz.evaluate_native)
    assert runner.sha256(ROOT/"config/ppo_ground_contact_v2l_400hz_v1.json")==runner.PROTOCOL_SHA256
    assert runner.formula_metadata()["normal_force_slew_difference_count"]==7

