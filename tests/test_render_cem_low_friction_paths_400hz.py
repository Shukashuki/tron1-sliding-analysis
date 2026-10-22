"""A duplicated visual panel is never a second independent CEM episode."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import render_cem_low_friction_paths_400hz as mod


def fixture():
    t=np.arange(2793)*.0025; rt=np.arange(349)*.02
    arrays={"physics_dt_s":np.asarray(.0025),"reference_time_s":rt,
        "reference_position":np.column_stack((rt,rt*0.,rt*0.+.7)),
        "reference_velocity":np.tile([1.,0.,0.],(349,1)),
        "reference_quaternion":np.tile([1.,0.,0.,0.],(349,1))}
    rows=[];ids=[]
    for i,mu in enumerate(mod.UNIQUE_MU):
        key=mod.frozen.case_key(mod.GROUP,mu);ids.append(key)
        tracking={"xy_rmse_m":.1,"xy_peak_m":.3,"speed_retention":1.}
        rows.append({"id":key,"group":mod.GROUP,"mu":mu,"controller":"MPC-CEM","condition":"beta_on",
            "prediction_friction_known":True,"evaluated":True,"status":"completed_verified",
            "windows":{"core":{"tracking":tracking.copy()},"reference_overlap":{"tracking":tracking.copy()},
                       "full":{"maximum_tilt_deg":12.}}})
        prefix=key.replace("/","__")+"__"
        for name,shape in mod.native.SHAPES.items():arrays[prefix+name]=np.zeros(shape)
        arrays[prefix+"time_s"]=t.copy()
        arrays[prefix+"position"]=np.column_stack((t,t*0.+i*.2,t*0.+.7))
        arrays[prefix+"quaternion"][:,0]=1
        arrays[prefix+"wheel_quaternion_wxyz"][:,:,0]=1
        arrays[prefix+"com_velocity"][:,0]=1
        arrays[prefix+"wheel_center_velocity_w"][:,:,0]=1
    arrays["case_ids"]=np.asarray(ids)
    return {"rows":rows},arrays


def test_literal_order_retains_duplicate_object_not_duplicate_trial():
    summary,arrays=fixture();cases,_=mod.select_cases(summary,arrays)
    panels=mod.panel_list(cases,mod.LITERAL_MU)
    assert [p["mu"] for p in panels]==[.1,.05,.1,.15]
    assert panels[0] is panels[2] and len({p["id"] for p in panels})==3
    assert len(cases)==4 and all(p["trace"]["time_s"][-1]==6.98 for p in cases)


@pytest.mark.parametrize("field,value",[("status","running"),("status","smoke_completed"),
    ("evaluated",False),("condition","beta_off"),("controller","PPO"),("prediction_friction_known",False)])
def test_pending_and_controller_beta_oracle_identity_refused(field,value):
    summary,arrays=fixture();summary["rows"][0][field]=value
    with pytest.raises(ValueError,match="Pending or incorrect"):mod.select_cases(summary,arrays)


@pytest.mark.parametrize("mutation",["shape","clock","nan","duplicate"])
def test_native_arrays_and_unique_episode_identity(mutation):
    summary,arrays=fixture();prefix=arrays["case_ids"][0].replace("/","__")+"__"
    if mutation=="shape":arrays[prefix+"position"]=arrays[prefix+"position"][:-1]
    if mutation=="clock":arrays[prefix+"time_s"][1]=.1
    if mutation=="nan":arrays[prefix+"position"][1,0]=np.nan
    if mutation=="duplicate":arrays["case_ids"]=np.append(arrays["case_ids"],arrays["case_ids"][0])
    with pytest.raises(ValueError):mod.select_cases(summary,arrays)


def test_actual_renderings_shared_axes_full_paths_and_duplicate_caption_evidence(tmp_path):
    summary,arrays=fixture();cases,reference=mod.select_cases(summary,arrays)
    cases[0]["trace"]["position"][-1,:2]=[13.,-5.]
    limits=mod.native.shared_limits(cases,reference)
    first=mod.render(cases,reference,limits,tmp_path,"literal.png")
    second=mod.render(cases,reference,limits,tmp_path,"alternative.png",alternative=True)
    assert first["unique_episode_count"]==3 and second["unique_episode_count"]==4
    assert first["panels"][0]["repeated_same_recorded_episode"] and first["panels"][2]["repeated_same_recorded_episode"]
    assert second["alternative_interpretation_not_replacement"] is True
    panels=first["panels"]+second["panels"]
    for name in ("world_xlim_m","world_ylim_m","world_xticks_m","world_yticks_m","aspect"):
        assert all(p[name]==panels[0][name] for p in panels)
    assert all(p["native_samples_plotted"]==2793 and p["native_final_time_s"]==6.98 and p["core_samples_bold"]==1025 for p in panels)
    assert (tmp_path/"literal.png").stat().st_size>10000


def test_overwrite_refused_before_inputs(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(SimpleNamespace(summary="absent",compact="absent",output_dir=tmp_path))


def test_small_selected_public_replay_remains_sha_bound(tmp_path):
    summary,arrays=fixture();source=tmp_path/"summary.json";compact=tmp_path/"input.npz"
    np.savez_compressed(compact,**arrays)
    summary["artifact_sha256"]={"native_failure_signals.npz":mod.sha256(compact)}
    source.write_text(json.dumps(summary))
    output=tmp_path/"first";mod.run(SimpleNamespace(summary=source,compact=compact,output_dir=output))
    second=tmp_path/"replayed"
    replay=mod.run(SimpleNamespace(summary=output/"selected_summary.json",compact=output/"selected_native_signals.npz",output_dir=second))
    assert len(replay["selected_unique_case_ids"])==4
    assert (output/"cem_low_mu_literal_requested.png").read_bytes()==(second/"cem_low_mu_literal_requested.png").read_bytes()
