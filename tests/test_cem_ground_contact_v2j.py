"""J PPO/CEM arithmetic equality fixtures, not native controller performance."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"tests"))
import nmpc_batch_ground_contact_v2i_400hz as parent
import nmpc_batch_ground_contact_v2j_400hz as candidate
import ground_contact_reward_v2j as definition
import run_cem_ground_contact_v2i as old_runner
import run_cem_ground_contact_v2j as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture,evaluate
from test_cem_ground_contact_v2i import objective


@pytest.mark.parametrize("gamma",(.8,.99,1.))
def test_only_grounded_changes_and_exact_horizon_difference(gamma):
    native,reference,profile,actions=fixture();before=copy.deepcopy(native)
    old=evaluate(parent.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.,discount_factor=gamma),native,reference,actions)
    new=evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.,discount_factor=gamma),native,reference,actions)
    for channel in ("raw_terms","weighted_terms","term_costs"):
        assert len(new[channel])==15
        for name in old[channel]:
            if name!="drift_grounded":np.testing.assert_array_equal(old[channel][name],new[channel][name])
    for name in old["ground_contact_v2"].keys()-{"raw_reward"}:
        np.testing.assert_array_equal(old["ground_contact_v2"][name],new["ground_contact_v2"][name])
    expected=.02*.4*new["ground_contact_v2"]["loaded_angular_increment_penalty"]
    np.testing.assert_allclose(new["stage_cost"]-old["stage_cost"],expected,atol=5e-16,rtol=1e-12)
    np.testing.assert_allclose(new["candidate_costs"]-old["candidate_costs"],(expected*gamma**np.arange(5)).sum(-1),atol=8e-16,rtol=1e-12)
    for name in native:np.testing.assert_array_equal(native[name],before[name])


def test_native_pre_state_is_true_ninth_sample_not_first_post_copy():
    native,reference,profile,actions=fixture();native["body_ang_vel_w"].fill(0.)
    native["pre_body_ang_vel_w"].fill(0.);native["pre_body_ang_vel_w"][...,0,:2]=100.
    result=evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.),native,reference,actions)
    c=result["ground_contact_v2"];np.testing.assert_array_equal(c["angular_velocity_increments"][...,0,:],np.full((3,5,2),-100.))
    np.testing.assert_array_equal(c["angular_velocity_increments"][...,1:,:],np.zeros((3,5,7,2)))
    del native["pre_body_ang_vel_w"]
    with pytest.raises(KeyError):evaluate(candidate.BatchGroundContactTrackingObjective400Hz(profile,model_weight_n=100.),native,reference,actions)


@pytest.mark.parametrize("missing",(0,1,8))
@pytest.mark.parametrize("beta,refspeed",((0.,1.),(.37,1.),(1.,1.),(0.,0.)))
def test_actual_plant_same_formula_for_start_and_eight_samples(missing,beta,refspeed):
    legacy=objective();legacy["raw_terms"]["drift_sideslip"]=beta;before=copy.deepcopy(legacy)
    force=np.zeros((8,2,3));force[:,0,2]=100.;force[:missing]=0.
    angular=np.arange(9)[:,None]*np.array([[.4,-.2]])
    old=old_runner.replace_actual_grounded(legacy,force,100.,reference_speed=refspeed)
    new=runner.replace_actual_grounded(legacy,force,100.,reference_speed=refspeed,angular_world_xy=angular)
    expected=definition.reward_numpy(force[...,2],100.,legacy["raw_terms"]["motion_global_anchor_pos"],legacy["raw_terms"]["drift_speed"],beta,refspeed,angular)
    assert new["raw_terms"]["drift_grounded"]==float(expected["raw_reward"])
    assert legacy==before
    assert new["stage_cost"]-old["stage_cost"]==pytest.approx(.02*.4*float(expected["loaded_angular_increment_penalty"]),abs=4e-16)
    for channel in ("raw_terms","weighted_terms","term_costs"):
        for key in old[channel]:
            if key!="drift_grounded":assert old[channel][key]==new[channel][key]


def test_actual_stage_no_implicit_or_missing_angular_start_and_no_new_build():
    force=np.zeros((8,2,3));force[:,0,2]=100.
    with pytest.raises(TypeError):runner.replace_actual_grounded(objective(),force,100.,reference_speed=1.)
    for angular in (np.zeros((8,2)),np.full((9,2),np.nan)):
        with pytest.raises(ValueError):runner.replace_actual_grounded(objective(),force,100.,reference_speed=1.,angular_world_xy=angular)
    source=Path(runner.__file__).read_text()
    assert 'raw["body_ang_vel_w"][start:end+1, 0, :2]' in source
    assert 'row["body_ang_vel_w"][0, :2] for row in records[-9:]' in source
    for bad in ("build_native_library(","torch.load(","torch.jit.load("):assert bad not in source
