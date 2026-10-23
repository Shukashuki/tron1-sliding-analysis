"""Native any-wheel continuity and conservative goal-completion reporting."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import analyze_cem_ground_contact_v2 as mod


def test_single_and_alternating_wheel_support_not_bilateral():
    t=np.arange(9)*.0025;force=np.zeros((9,2));force[:,0]=10
    first=mod.support_summary(t,force,(0.,.02))
    force[::2]=[0,10];second=mod.support_summary(t,force,(0.,.02))
    assert first==second and first["every_physical_sample_supported"] is True
    assert first["any_wheel_support_sample_fraction"]==1 and first["longest_no_support_observed_s"]==0


def test_strict_threshold_and_isolated_physical_gap():
    t=np.arange(9)*.0025;force=np.ones((9,2))*10;force[4]=5.
    result=mod.support_summary(t,force,(0.,.02))
    assert result["every_physical_sample_supported"] is False
    assert result["unsupported_physical_samples"]==1
    assert result["longest_no_support_observed_s"]==pytest.approx(.0025)
    assert result["no_support_gaps"][0]["onset_s"]==.01


def test_last_sample_is_not_extended_but_still_fails_every_sample():
    t=np.arange(9)*.0025;force=np.ones((9,2))*10;force[-1]=0
    result=mod.support_summary(t,force,(0.,.02))
    assert result["longest_no_support_observed_s"]==0
    assert result["every_physical_sample_supported"] is False
    assert result["no_support_gaps"][0]["right_window_censored"] is True


def fixture():
    windows={name:{"tracking":{"xy_rmse_m":.1,"xy_peak_m":.2,"speed_retention":1.,"beta_rmse_deg":5.},
        "world_angular_xy_common10_100":{"common_band_rms":.5}} for name in ("core","whole_reference_overlap")}
    value={"windows":windows,"any_wheel_ground_support":{"full_contact":{
        "every_physical_sample_supported":True,"any_wheel_support_sample_fraction":1.,"longest_no_support_observed_s":0.}}}
    baseline=copy.deepcopy(value);baseline["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]=1.
    return value,baseline,mod.read_json(mod.DEFAULT_PROTOCOL)


def test_numeric_pass_is_not_full_goal_completion():
    candidate,baseline,protocol=fixture();result=mod.nominal_screen(candidate,baseline,protocol)
    assert result["numeric_nominal_episode_pass"] is True
    assert result["overall_task_goal_complete_claimed"] is False
    assert result["core_HF_ratio_to_old_CEM"]==.5


@pytest.mark.parametrize("mutation",["whole_peak","core_speed","beta","contact","HF"])
def test_all_declared_nominal_checks_required(mutation):
    candidate,baseline,protocol=fixture()
    if mutation=="whole_peak":candidate["windows"]["whole_reference_overlap"]["tracking"]["xy_peak_m"]=.31
    if mutation=="core_speed":candidate["windows"]["core"]["tracking"]["speed_retention"]=.94
    if mutation=="beta":candidate["windows"]["core"]["tracking"]["beta_rmse_deg"]=11.
    if mutation=="contact":candidate["any_wheel_ground_support"]["full_contact"]["every_physical_sample_supported"]=False
    if mutation=="HF":candidate["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]=.71
    assert mod.nominal_screen(candidate,baseline,protocol)["numeric_nominal_episode_pass"] is False


def test_analysis_never_overwrites_existing_output(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path))
