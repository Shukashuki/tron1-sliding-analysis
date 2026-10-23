"""Synthetic-only E CEM render tests; no actual E results/graphics/physics."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import render_cem_ground_contact_v2e as mod
from nmpc_tracking_objective import JOINT_NAMES


def raw_report(key="v2e"):
    n=2793
    raw={"fixture_id":key,"qpos":np.zeros((n,15)),"qvel":np.zeros((n,14)),"actions":np.zeros((349,8)),
        "time_s":np.arange(n)*.0025,"physics_dt_s":np.asarray(.0025),"action_joint_names":np.asarray(JOINT_NAMES),
        "root_pos":np.zeros((n,3)),"root_quaternion_wxyz":np.tile([1.,0.,0.,0.],(n,1)),
        "base_com_velocity_w":np.zeros((n,3)),"base_angular_velocity_w":np.zeros((n,3)),
        "joint_pos":np.zeros((n,8)),"joint_vel":np.zeros((n,8)),"wheel_pos":np.zeros((n,2,3)),
        "wheel_ground_contact_count":np.zeros((n,2)),"wheel_ground_contact_force_w_n":np.zeros((n,2,3)),
        "wheel_lateral_contact_slip_m_s":np.zeros((n,2))}
    force=np.zeros((n,2,3));force[:,0,2]=20.
    if key=="original":raw["wheel_ground_contact_force_w_n"]=force
    else:raw["ground_normal_force_w"]=force
    report={"status":"completed","smoke_only":False,"truncated_smoke":False,"completed_policy_steps":349,
        "native_samples":2793,"condition":"beta_on","plant_mu":.7,"prediction_mu_scenarios":[.7],
        "prediction_friction_known":True,"motion_sha256":"b"*64,"model_sha256":"c"*64,
        "CEM_config":{"fixture":"SYNTHETIC_NOT_A_CONTROLLER_RESULT"},"discount_factor":.99,
        "plant_contact_material_audit":{"fixture":True},"prediction_contact_material_audit":{"fixture":True},"model_weight_n":100.}
    return raw,report


def metric(key,index):
    return {"fixture_id":key,"any_wheel_ground_support":{"full_contact":{
        "any_wheel_support_sample_fraction":.1234+.1*index,"longest_no_support_observed_s":.0125+.0025*index}}}


def screen(values,*args):
    return {"numeric_nominal_episode_pass":False,"failed_checks":["fixture_only_not_actual_"+values["fixture_id"]]}


def setup_fake(monkeypatch,*,optional=True):
    args=mod.parser().parse_args(["--analysis-sha256","a"*64,"--output-dir","never-created-by-tests"])
    args.analysis_summary=Path("SYNTHETIC_summary.json");args.reference=Path("SYNTHETIC_motion.npz");args.model=Path("SYNTHETIC_model.xml")
    keys=("original","v2a","v2b","v2c","v2d","v2e")
    raws,reports,values={}, {}, {}
    for index,key in enumerate(keys):raws[key],reports[key]=raw_report(key);values[key]=metric(key,index)
    hashes={"original":{name:mod.audit.BASELINE_PINS[name] for name in ("report.json","rollout.npz")},
        "v2a":dict(mod.audit.V2A_PINS),"v2b":dict(mod.audit.V2B_PINS),
        "v2c":dict(mod.audit.V2C_PINS),"v2d":dict(mod.audit.V2D_PINS),
        "v2e":{name:"e"*64 for name in ("report.json","rollout.npz","solver_log.json","experiment.json")}}
    rewards={key:{"fixture_reward_audit":key} for key in keys if key!="original"}
    prior={key:{"label":"SYNTHETIC_UNIT_FIXTURE_NOT_ACTUAL", "input_sha256":hashes[key],
        "reward_audit":rewards[key],"metrics":values[key]} for key in ("v2a","v2b","v2c","v2d")}
    summary={"kind":"CEM_ground_contact_V2e_ACTUAL_nominal_readonly_audit","status":"completed_analysis",
        "analyzer_sha256":mod.ANALYZER_SHA256,"shared_metric_source_sha256":mod.audit.METRICS_SOURCE_SHA256,
        "protocol_sha256":mod.audit.PROTOCOL_SHA256,"executed_source_sha256":copy.deepcopy(mod.audit.SOURCE_PINS),
        "actual_V2e":values["v2e"],"input_sha256":hashes["v2e"],"actual_E_reward_audit":rewards["v2e"],
        "nominal_screen":screen(values["v2e"]),"original_grounded_baseline":values["original"],
        "baseline_input_sha256":hashes["original"],"optional_prior_actual_V2a":copy.deepcopy(prior["v2a"]),
        "optional_prior_actual_V2c":copy.deepcopy(prior["v2c"]),"optional_prior_actual_V2b":copy.deepcopy(prior["v2b"]) if optional else None,
        "optional_prior_actual_V2d":copy.deepcopy(prior["v2d"]) if optional else None}
    actual_sha=mod.sha256
    def fake_sha(path):
        for p,digest in ((args.analysis_summary,"a"*64),(args.reference,"b"*64),(args.model,"c"*64)):
            if Path(path)==p:return digest
        return actual_sha(path)
    monkeypatch.setattr(mod,"sha256",fake_sha)
    monkeypatch.setattr(mod,"read_json",lambda path:copy.deepcopy(summary))
    monkeypatch.setattr(mod.audit,"load_protocol",lambda path:{"fixture":True})
    calls=[]
    def e_loader(folder,protocol):
        calls.append(("v2e",folder,protocol));return raws["v2e"],reports["v2e"],hashes["v2e"],rewards["v2e"]
    monkeypatch.setattr(mod.audit,"load_completed",e_loader)
    monkeypatch.setattr(mod.audit.base,"load_completed",lambda folder:(raws["v2a"],reports["v2a"],hashes["v2a"],rewards["v2a"]))
    monkeypatch.setattr(mod.audit.prior_c,"load_completed",lambda folder:(raws["v2c"],reports["v2c"],hashes["v2c"],rewards["v2c"]))
    monkeypatch.setattr(mod,"load_original_recording",lambda reference:(raws["original"],reports["original"],hashes["original"],values["original"],"f"*64))
    monkeypatch.setattr(mod.audit.base,"recorded_metrics",lambda raw,reference:copy.deepcopy(values[raw["fixture_id"]]))
    monkeypatch.setattr(mod.audit.base,"nominal_screen",screen)
    for key in ("v2b","v2d"):
        def prior_loader(folder,reference,e_raw,e_report,key=key):
            assert e_raw is raws["v2e"] and e_report is reports["v2e"]
            calls.append((key,folder));return copy.deepcopy(prior[key])
        monkeypatch.setattr(mod.audit,"load_"+key,prior_loader)
    return args,raws,reports,summary,prior,calls


def test_frozen_D_and_E_analyzer_pins_and_common_validator_exact_reuse():
    assert mod.sha256(mod.frozen_d.__file__)==mod.D_RENDERER_SHA256
    assert mod.sha256(mod.audit.__file__)==mod.ANALYZER_SHA256
    assert mod.sha256(mod.graphics.__file__)==mod.GRAPHICS_SHA256
    assert mod.validate_episode is mod.frozen_d.validate_episode
    for name in ("native_indices","contact_window","heading_glyphs","case_contact_caption","draw_map"):
        assert getattr(mod,name) is getattr(mod.graphics,name)


def test_actual_analysis_digest_required_not_invented_or_optional():
    with pytest.raises(SystemExit):mod.parser().parse_args(["--output-dir","fresh"])
    for value in (None,"pending","A"*64,"0"*63):
        with pytest.raises(ValueError,match="analysis-sha256"):mod.analysis_digest(value)
    assert mod.analysis_digest("a"*64)=="a"*64
    a=mod.parser().parse_args(["--analysis-sha256","a"*64,"--output-dir","fresh"])
    assert a.poster_frame==348 and "ground_contact_v2e_" in str(a.v2e_dir)
    assert "ground_contact_v2e_" in str(a.analysis_summary)


def test_synthetic_four_panel_E_identity_and_optional_all_six_numeric_cases(monkeypatch):
    args,raws,reports,summary,prior,calls=setup_fake(monkeypatch)
    traces,cases,evidence,order,binding=mod.load_cases(args,{})
    assert order==["original","v2a","v2c","v2e"] and len(traces)==len(cases)==4
    assert cases["v2e"]["title"]=="E CEM | gated mean8, deficit0"
    assert all("PPO" not in c["title"] and "CEM" in c["title"] for c in cases.values())
    assert cases["original"]["new_nominal_screen"] is None
    assert binding["numeric_case_order"]==["original","v2a","v2b","v2c","v2d","v2e"]
    assert binding["numeric_cases"]["v2b"]["displayed"] is False and binding["numeric_cases"]["v2d"]["displayed"] is False
    assert [call[0] for call in calls]==["v2e","v2b","v2d"]
    assert calls[0][1:]==(args.v2e_dir,args.protocol)
    assert binding["sha256"]==args.analysis_sha256 and binding["all_four_metrics_recomputed_and_bound"]
    assert not binding["episode_returns_compared_or_ranked"]


def test_absent_optional_B_D_are_not_fabricated_or_loaded(monkeypatch):
    args,_,_,_,_,calls=setup_fake(monkeypatch,optional=False)
    _,_,_,_,binding=mod.load_cases(args,{})
    assert binding["numeric_case_order"]==list(mod.ORDER)
    assert [call[0] for call in calls]==["v2e"]


def test_wrong_summary_sha_rejected_before_completed_episode_loader(monkeypatch):
    args,_,_,_,_,calls=setup_fake(monkeypatch);args.analysis_sha256="0"*64
    with pytest.raises(ValueError,match="analysis summary changed"):mod.load_cases(args,{})
    assert calls==[]


@pytest.mark.parametrize("what",["pending_summary","wrong_identity","wrong_ledger","wrong_E_metrics","wrong_E_reward","wrong_prior"])
def test_summary_binding_cannot_relabel_pending_or_prior_results(monkeypatch,what):
    args,_,_,summary,_,_=setup_fake(monkeypatch)
    if what=="pending_summary":summary["status"]="running"
    if what=="wrong_identity":summary["kind"]="CEM_ground_contact_V2d_ACTUAL_nominal_readonly_audit"
    if what=="wrong_ledger":summary["executed_source_sha256"]={}
    if what=="wrong_E_metrics":summary["actual_V2e"]={"forged":"not_actual_E"}
    if what=="wrong_E_reward":summary["actual_E_reward_audit"]={"forged":"D_reward"}
    if what=="wrong_prior":summary["optional_prior_actual_V2a"]["metrics"]={}
    with pytest.raises(ValueError):mod.load_cases(args,{})


def test_numeric_optional_data_must_be_independently_verified_not_blindly_copied(monkeypatch):
    args,_,_,summary,prior,_=setup_fake(monkeypatch)
    summary["optional_prior_actual_V2d"]["metrics"]={"forged":"not_the_strict_loaded_D"}
    with pytest.raises(ValueError,match="Numeric-only actual v2d differs"):mod.load_cases(args,{})


@pytest.mark.parametrize("field,value",[("status","running"),("smoke_only",True),("completed_policy_steps",348),("native_samples",2785)])
def test_full_actual_clock_pending_or_prefix_rejected(monkeypatch,field,value):
    args,_,reports,_,_,_=setup_fake(monkeypatch);reports["v2e"][field]=value
    with pytest.raises(ValueError,match="terminal349-step"):mod.load_cases(args,{})


def test_same_initial_state_budget_material_and_raw_force_shapes_required(monkeypatch):
    args,raws,reports,_,_,_=setup_fake(monkeypatch);raws["v2a"]["qpos"][0,0]=.1
    with pytest.raises(ValueError,match="initial state differs"):mod.load_cases(args,{})
    raws["v2a"]["qpos"][0,0]=0.;reports["v2c"]["CEM_config"]={"changed":True}
    with pytest.raises(ValueError,match="budget/discount"):mod.load_cases(args,{})
    reports["v2c"]["CEM_config"]=reports["v2e"]["CEM_config"];reports["v2c"]["model_weight_n"]=1.
    with pytest.raises(ValueError,match="material/normalization"):mod.load_cases(args,{})


def test_native_full_tail_force_only_eight_samples_ignore_pointcounts(monkeypatch):
    args,raws,_,_,_,_=setup_fake(monkeypatch)
    traces,_,_,_,_=mod.load_cases(args,{})
    t=traces["v2e"]
    assert len(t["time_s"])==2793 and t["time_s"][-1]==pytest.approx(6.98)
    assert mod.native_indices().tolist()==list(range(0,2793,8)) and mod.native_indices()[348]==2784
    assert not t["contact_count"].any()
    t["contact_force"][12,:,2]=5.
    w=mod.contact_window(t,16)
    assert w["indices"].tolist()==list(range(9,17)) and w["supported_samples"]==7


def test_E_caption_is_actual_per_case_not_D_contact_constant(monkeypatch):
    args,_,_,_,_,_=setup_fake(monkeypatch);_,cases,_,order,_=mod.load_cases(args,{})
    assert "62.34%" in mod.case_contact_caption(cases["v2e"])
    text=mod.outcome_caption(cases,order)
    assert "V2E: FAIL 1" in text and "B/D omitted" in text and "NOT claimed" in text
    assert "70.36" not in text and "PPO" not in text


def test_synthetic_canvas_E_identity_only_no_actual_graphics(monkeypatch):
    from PIL import Image,ImageDraw
    args,_,_,_,_,_=setup_fake(monkeypatch);_,cases,_,order,_=mod.load_cases(args,{})
    canvas=Image.new("RGB",(2560,850));calls=[]
    monkeypatch.setattr(mod.graphics,"render_frame",lambda *a:canvas)
    class Draw:
        def rectangle(self,box,**kwargs):calls.append(("rectangle",box))
        def text(self,point,text,**kwargs):calls.append(("text",text))
    monkeypatch.setattr(ImageDraw,"Draw",lambda image:Draw())
    assert mod.render_frame(NS(small=None),{},{},{},cases,order,{},None,348) is canvas
    text=[v for k,v in calls if k=="text"]
    assert "Original/A/C/E" in text[0] and "B/D excluded" in text[0] and "t=6.96s" in text[0]
    assert "V2E" in text[1] and "V2D" not in text[1] and "No policy result claimed" in text[2]


def test_no_overwrite_or_actual_runtime_geometry_changes(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):mod.run(NS(output_dir=tmp_path))
    source=Path(mod.__file__).read_text()
    for bad in ("mj_step(","torch.load(","torch.jit.load(","render_ground_contact_v2 as"):
        assert bad not in source
    for required in ('graphics.prepare_camera(', 'graphics.world_bounds(traces, reference)', 'graphics.build_qpos(',
                     'graphics.replay_fk_audit(', '"camera_same_distance_all_cases_frames": True',
                     '"camera_yaw_follow": False', '"camera_zoom": False', '"robot_mesh_alpha": 1.',
                     '"shared_world_bounds_all_panels": True', '"smoothing_interpolation_alignment_height_correction": False',
                     '"goal_completion_claimed": False', '"E_support_deficit_weight": 0.', '"numeric_comparison.json"'):
        assert required in source
