"""Fresh fixed-recipe runner; actualplant and planner use the NEW contact term."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_cem_ground_contact_v2 as mod
from ground_contact_reward_v2 import reward_numpy


def objective():
    names=("motion_global_anchor_pos","drift_speed","drift_grounded")+tuple("unchanged"+str(i) for i in range(12))
    raw={name:.7 for name in names};raw["drift_grounded"]=0.
    weighted={name:.2*value for name,value in raw.items()};weighted["drift_grounded"]=0.
    return {"raw_terms":raw,"weighted_terms":weighted,"reward_upper_bound":12.,
            "total_reward":sum(weighted.values()),"stage_cost":0.,"base_com_vz":.3,"surrogate_flags":["legacy"]}


def test_actual_grounded_replaces_legacy_without_touching_other14_terms():
    old=objective();before=copy.deepcopy(old);forces=np.zeros((8,2,3));forces[:,0,2]=100.
    result=mod.replace_actual_grounded(old,forces,100.)
    assert old==before and len(result["raw_terms"])==15
    assert result["raw_terms"]["drift_grounded"]==pytest.approx(.49)
    assert result["weighted_terms"]["drift_grounded"]==pytest.approx(.4*.49)
    for name in old["raw_terms"]:
        if name!="drift_grounded":assert result["raw_terms"][name]==old["raw_terms"][name]
    assert result["total_reward"]==sum(result["weighted_terms"].values())
    assert result["stage_cost"]==.02*(12.-result["total_reward"])
    assert result["ground_contact_v2"]["support"]==1.
    assert result["grounded_reward_replaced_not_legacy"] is True


@pytest.mark.parametrize("bad",["shape","missing_step","tangential","nonfinite","negative"])
def test_actual_force_contract_cannot_silently_fall_back_to_net_force(bad):
    forces=np.zeros((8,2,3));forces[:,0,2]=100.
    if bad=="shape":forces=forces[...,2]
    if bad=="missing_step":forces=forces[:7]
    if bad=="tangential":forces[0,0,0]=2.
    if bad=="nonfinite":forces[0,0,2]=np.nan
    if bad=="negative":forces[0,0,2]=-1.
    with pytest.raises(ValueError):mod.replace_actual_grounded(objective(),forces,100.)


def test_cli_cannot_retune_mu_weights_or_optimizer():
    parser=mod.parser()
    args=parser.parse_args(["--native-library","a.so","--output-dir","new","--max-policy-steps","1"])
    assert args.max_policy_steps==1
    for extra in (["--mu",".1"],["--grounded-weight","1."],["--horizon-steps","5"]):
        with pytest.raises(SystemExit):parser.parse_args(["--native-library","a.so","--output-dir","new",*extra])


def test_existing_output_guard_precedes_any_inputs_or_native_work(tmp_path,monkeypatch):
    monkeypatch.setattr(mod.nominal,"inputs",lambda:pytest.fail("Do not inspect inputs"))
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path))


@pytest.mark.parametrize("maximum",[0,350,True,-1])
def test_invalid_smoke_length_refused_before_loading_inputs(maximum,tmp_path):
    with pytest.raises(ValueError,match="smoke length"):
        mod.run(SimpleNamespace(output_dir=tmp_path/"new",max_policy_steps=maximum))


def test_new_runner_never_loads_policy_or_future_PPO_actions():
    source=Path(mod.__file__).read_text()
    assert 'raw[name].copy() for name in ("initial_qpos", "initial_qvel")' in source
    assert "torch.load(" not in source and "torch.jit.load(" not in source
    assert '"PPO_actions_or_future_states_used_for_initial_plan": False' in source
    assert '"termination_or_reset_after_initialization": False' in source
    assert '"actual_reward_uses_ground_contact_v2": True' not in source # completion uses keyword, not static relabel
    assert "objective = replace_actual_grounded(legacy_objective, ground_window, mg)" in source


def test_actual_one_decision_cpu_smoke_and_exact_reward_audit(tmp_path):
    if not (mod.paired.DEFAULT_OFF/"manifest.json").is_file():
        pytest.skip("Optional local locked reference/model/retrievals unavailable")
    from nmpc_native_ground_contact_400hz import build_native_library
    library=build_native_library(tmp_path/"native")
    output=tmp_path/"smoke"
    report=mod.run(SimpleNamespace(native_library=library,output_dir=output,max_policy_steps=1))
    assert report["status"]=="smoke_completed" and report["smoke_only"] is True
    assert report["completed_policy_steps"]==1 and report["native_samples"]==9
    assert report["duration_s"]==pytest.approx(.02)
    assert report["solver_failures"]==0 and not any(report["plant_warning_counts"])
    assert report["objective_contract"]["stage_reward_formulas_changed"] is True
    assert report["native_backend"]["workers"]["0.7"]["abi_version"]==3
    assert report["actual_reward_audit"]["status"]=="verified_exact"
    assert report["PPO_V2_trained_policy_claimed"] is False
    assert report["actual_reward_uses_ground_contact_v2"] is True
    assert len(report["source_sha256"])>=15
    with np.load(output/"rollout.npz",allow_pickle=False) as archive:
        raw={name:archive[name].copy() for name in archive.files}
    assert raw["ground_normal_force_w"].shape==(9,2,3)
    assert raw["ground_normal_force_window_w"].shape==(1,8,2,3)
    assert raw["ground_contact_count_window"].shape==(1,8,2)
    decisions=json.loads((output/"solver_log.json").read_text())
    assert mod.recompute_actual_rewards(raw,report,decisions)==report["actual_reward_audit"]
    wrong=copy.deepcopy(decisions);wrong[0]["objective"]["raw_terms"]["drift_grounded"]+=.1
    with pytest.raises(ValueError,match="exactly reproduce"):mod.recompute_actual_rewards(raw,report,wrong)
    broken={**raw,"ground_normal_force_window_w":raw["ground_normal_force_window_w"].copy()}
    broken["ground_normal_force_window_w"][0,0,0,2]+=1.
    with pytest.raises(ValueError,match="windows differ"):mod.recompute_actual_rewards(broken,report,decisions)
