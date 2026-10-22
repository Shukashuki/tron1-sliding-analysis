"""Beta identities and complete late trajectories survive recorded rendering."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_ppo_cem_beta_paths_400hz as mod


def fixture():
    time=np.arange(2793)*.0025; rt=np.arange(349)*.02
    arrays={"physics_dt_s":np.asarray(.0025), "reference_time_s":rt,
        "reference_position":np.column_stack((rt,rt*0.,rt*0.+.7)),
        "reference_velocity":np.tile([1.,0.,0.],(349,1)),
        "reference_quaternion":np.tile([1.,0.,0.,0.],(349,1))}
    rows=[];ids=[]
    for i,(group,mu) in enumerate(mod.SELECTED):
        key=mod.frozen.case_key(group,mu);ids.append(key)
        tracking={"xy_rmse_m":.1+i*.001,"xy_peak_m":.2,"speed_retention":1.+i*.01,
                  "beta_rmse_deg":20.+i,"beta_valid_samples":1025}
        rows.append({"id":key,"group":group,"mu":mu,"evaluated":True,"status":"completed_verified",
            "condition":"beta_on" if group.endswith("beta_on") else "beta_off",
            "controller":"PPO" if group.startswith("ppo") else "MPC-CEM",
            "core_large_xy_screen":False,"overlap_large_xy_screen":False,
            "windows":{"core":{"tracking":tracking.copy()},"reference_overlap":{"tracking":tracking.copy()},
                       "full":{"maximum_tilt_deg":10.+i}}})
        prefix=key.replace("/","__")+"__"
        for name,shape in mod.native.SHAPES.items():arrays[prefix+name]=np.zeros(shape)
        arrays[prefix+"time_s"]=time.copy()
        arrays[prefix+"position"]=np.column_stack((time,time*0.+i*.2,time*0.+.7))
        arrays[prefix+"quaternion"][:,0]=1
        arrays[prefix+"wheel_quaternion_wxyz"][:,:,0]=1
        arrays[prefix+"com_velocity"][:,0]=1
        arrays[prefix+"wheel_center_velocity_w"][:,:,0]=1
    arrays["case_ids"]=np.asarray(ids)
    return {"rows":rows},arrays


def test_exact_eight_on_off_case_order_labels_and_native_arrays():
    summary,arrays=fixture();panels,reference=mod.select_panels(summary,arrays)
    assert [(p["group"],p["mu"]) for p in panels]==list(mod.SELECTED)
    assert [mod.label(p) for p in panels[:4]]==["PPO + beta","PPO - beta","MPC-CEM + beta","MPC-CEM - beta"]
    assert all(len(p["trace"]["time_s"])==2793 and p["trace"]["time_s"][-1]==6.98 for p in panels)
    assert all(np.array_equal(p["trace"]["position"],arrays[p["id"].replace("/","__")+"__position"]) for p in panels)


@pytest.mark.parametrize("field,value",[("status","running"),("status","smoke_completed"),("evaluated",False),
                                         ("condition","beta_off"),("controller","MPC-CEM")])
def test_pending_wrong_beta_or_controller_refused(field,value):
    summary,arrays=fixture();summary["rows"][0][field]=value
    with pytest.raises(ValueError,match="Pending case or incorrect"):mod.select_panels(summary,arrays)


@pytest.mark.parametrize("kind",["shape","clock","nan","missing","count","screen"])
def test_native_trace_contract_and_stored_screens(kind):
    summary,arrays=fixture();prefix=arrays["case_ids"][0].replace("/","__")+"__"
    if kind=="shape":arrays[prefix+"position"]=arrays[prefix+"position"][:-1]
    if kind=="clock":arrays[prefix+"time_s"][1]=.01
    if kind=="nan":arrays[prefix+"com_velocity"][2,0]=np.nan
    if kind=="missing":arrays["case_ids"]=arrays["case_ids"][1:]
    if kind=="count":arrays[prefix+"contact_count"][1,0]=.5
    if kind=="screen":summary["rows"][0]["core_large_xy_screen"]=True
    with pytest.raises(ValueError):mod.select_panels(summary,arrays)


def test_numeric_rows_are_exact_unrounded_summary_values_and_paired_differences():
    summary,arrays=fixture();panels,_=mod.select_panels(summary,arrays);rows=mod.numerical_rows(panels)
    assert rows[3]["core_xy_rmse_m"]==summary["rows"][3]["windows"]["core"]["tracking"]["xy_rmse_m"]
    assert rows[1]["whole_beta_rmse_deg"]==21.
    paired=mod.paired_comparisons(rows)
    assert len(paired)==4 and paired[0]["controller"]=="PPO" and paired[0]["mu"]==.7
    assert paired[0]["off_minus_on"]["core_xy_rmse_m"]==rows[1]["core_xy_rmse_m"]-rows[0]["core_xy_rmse_m"]
    assert paired[0]["off_minus_on"]["maximum_native_tilt_deg_diagnostic"]==1.


def test_three_actual_renderings_share_limits_ticks_and_preserve_late_paths(tmp_path):
    summary,arrays=fixture();panels,reference=mod.select_panels(summary,arrays)
    panels[1]["trace"]["position"][-1,:2]=[12.,-3.]
    limits=mod.native.shared_limits(panels,reference)
    selections=[panels[:4],panels[4:],[p for p in panels if p["group"].startswith("cem")]]
    outputs=[mod.render(p,reference,limits,tmp_path,f"{i}.png") for i,p in enumerate(selections)]
    all_panels=[p for result in outputs for p in result["panels"]]
    for name in ("world_xlim_m","world_ylim_m","world_xticks_m","world_yticks_m","aspect"):
        assert all(p[name]==all_panels[0][name] for p in all_panels)
    assert all(p["native_samples_plotted"]==2793 and p["native_final_time_s"]==6.98
               and p["core_samples_bold"]==1025 and not p["cropped_after4s"] for p in all_panels)
    assert outputs[0]["panels"][1]["label"]=="PPO - beta"
    assert all((tmp_path/f"{i}.png").stat().st_size>10000 for i in range(3))


def test_overwrite_refused_before_input_read(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(SimpleNamespace(summary="absent",compact="absent",output_dir=tmp_path))


def test_compact_sha_mismatch_refused_before_rendering(tmp_path):
    summary,arrays=fixture();summary["artifact_sha256"]={"native_failure_signals.npz":"0"*64}
    source=tmp_path/"summary.json";source.write_text(json.dumps(summary))
    compact=tmp_path/"compact.npz";np.savez_compressed(compact,**arrays)
    with pytest.raises(ValueError,match="Compact SHA"):
        mod.run(SimpleNamespace(summary=source,compact=compact,output_dir=tmp_path/"output"))
