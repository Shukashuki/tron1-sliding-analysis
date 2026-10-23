"""PPO renderer cannot invent missing cases or inherit hardcoded CEM outcomes."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import render_ppo_ground_contact_v2 as mod


def trace():
    n=2793;force=np.zeros((n,2,3));force[:,0,2]=20.
    return {"position":np.zeros((n,3)),"contact_force":force,"contact_count":np.zeros((n,2)),
        "wheel_pos":np.zeros((n,2,3)),"body_heading_geometry_valid":np.ones(n,dtype=bool),
        "wheel_heading_geometry_valid":np.ones((n,2),dtype=bool),
        "signals":{"body_heading_rad":np.zeros(n),"course_rad":np.zeros(n),"speed_m_s":np.ones(n),
                   "wheel_rolling_plane_heading_rad":np.zeros((n,2))}}


def case(engine="physx",variant="new",fraction=.1234,gap=.045,passed=False):
    return {"engine":engine,"variant":variant,"analysis":{"any_wheel_ground_support":{"full_contact":{
        "any_wheel_support_sample_fraction":fraction,"longest_no_support_observed_s":gap}}},
        "new_nominal_screen":{"numeric_nominal_episode_pass":passed,"failed_checks":[] if passed else ["contact","gap"]}}


def test_exact_native_frames_and_eight_contact_samples_without_pointcount_gate():
    assert mod.native_indices().tolist()==list(range(0,2793,8))
    t=trace();window=mod.contact_window(t,16)
    assert window["indices"].tolist()==list(range(9,17)) and window["supported_samples"]==8
    assert (t["contact_count"]==0).all()
    t["contact_force"][12,:,2]=5.
    assert mod.contact_window(t,16)["supported_samples"]==7
    assert mod.contact_window(t,0)["samples"]==1


def test_wheel_colors_are_force_only_not_count_and_geometry_is_preserved():
    t=trace();glyphs=mod.heading_glyphs(t,10)
    left,right=glyphs[-2:]
    assert left["force_only_loaded"] is True and left["color"]==mod.COLORS["left"]
    assert right["force_only_loaded"] is False and right["color"]==mod.COLORS["unloaded"]
    t["contact_force"][10,1,2]=10;t["wheel_heading_geometry_valid"][10,1]=False
    right=mod.heading_glyphs(t,10)[-1]
    assert right["color"]==mod.COLORS["right"] and not right["valid"]
    t["body_heading_geometry_valid"][10]=False;t["signals"]["speed_m_s"][10]=.25
    g=mod.heading_glyphs(t,10)
    assert not g[0]["valid"] and g[1]["valid"]


def test_per_case_caption_uses_actual_data_and_reports_whole_window():
    text=mod.case_contact_caption(case(fraction=.1234,gap=.045))
    assert "12.34%" in text and "45.0ms" in text and "0.02-6.96s" in text
    other=mod.case_contact_caption(case(fraction=.9876,gap=.0025))
    assert "98.76%" in other and "2.5ms" in other and other!=text


def test_dynamic_outcome_caption_cannot_report_missing_engine_or_CEM_result():
    cases={"new_physx":case(),"baseline_physx":case(variant="baseline")}
    text=mod.outcome_caption(cases,["baseline_physx","new_physx"])
    assert "physx: FAIL 2" in text and "NOT DISPLAYED: mujoco" in text
    assert "70.47" not in text and "CEM" not in text and "NOT claimed" in text
    cases["new_mujoco"]=case("mujoco",passed=True)
    assert "mujoco: PASS numerical screen" in mod.outcome_caption(cases,list(cases))


def test_missing_engine_requires_explicit_subset_not_fake_four_panels():
    args=NS(physx_dir=Path("provided"),mujoco_dir=None,subset=None)
    with pytest.raises(ValueError,match="explicit --subset"):mod.selected_engines(args)
    args.subset="physx";assert mod.selected_engines(args)==("physx",)
    args.subset=None;args.mujoco_dir=Path("also_provided")
    assert mod.selected_engines(args)==("physx","mujoco")


def test_loader_delegates_to_both_strict_proof_and_pinned_baseline(monkeypatch):
    calls=[];reference={};fresh=trace();fresh["floor_normal_force"]=fresh["contact_force"].copy()
    def new(folder,engine,protocol):
        calls.append(("new",engine));return {"engine":engine},{"contract":{},"actor_sha256":"a"*64,"checkpoint_sha256":"b"*64}
    def old(engine,root,summary):calls.append(("old",engine));return {"engine":engine},{"contract":{}}
    monkeypatch.setattr(mod.audit,"load_completed",new);monkeypatch.setattr(mod.audit,"load_baseline",old)
    monkeypatch.setattr(mod.audit,"_compare_profiles",lambda *args:None)
    monkeypatch.setattr(mod.audit,"recorded_metrics",lambda raw,engine,ref:({},copy.deepcopy(fresh)))
    monkeypatch.setattr(mod.audit,"nominal_screen",lambda *args:{})
    args=NS(physx_dir="newp",mujoco_dir="newm",subset=None,protocol=mod.audit.DEFAULT_PROTOCOL,baseline_root="oldroot",baseline_summary="oldsummary")
    traces,cases,evidence,order=mod.load_cases(args,reference)
    assert calls==[("new","physx"),("old","physx"),("new","mujoco"),("old","mujoco")]
    assert order==["baseline_physx","new_physx","baseline_mujoco","new_mujoco"]
    assert len(traces)==len(cases)==len(evidence)==4


def test_bad_or_partial_formal_proof_is_not_rendered(tmp_path):
    args=NS(physx_dir=tmp_path,mujoco_dir=None,subset="physx",protocol=mod.audit.DEFAULT_PROTOCOL,
        baseline_root=mod.audit.BASELINE_ROOT,baseline_summary=mod.audit.BASELINE_SUMMARY)
    with pytest.raises(ValueError,match="Missing/indirect completed artifact"):mod.load_cases(args,{})


def test_existing_result_refused_before_any_read_or_graphics(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):mod.run(NS(output_dir=tmp_path))


def test_no_CEM_footer_constants_no_dynamics_or_actor_loading_and_named_wheel_legend():
    source=Path(mod.__file__).read_text()
    assert "70.47" not in source and "0.2575" not in source and "cem_footer_only" not in source
    assert "mj_step(" not in source and "torch.load(" not in source and "torch.jit.load(" not in source
    assert "LEFT red / RIGHT purple" in source
    assert '"camera_same_distance_all_cases_frames":True' in source
