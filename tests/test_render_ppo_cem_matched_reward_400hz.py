"""Offline matched-stage reward evidence, fixed camera, and native clocks."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import render_ppo_cem_matched_reward_400hz as mod


def valid_raw():
    actions=np.zeros((349,8))
    return {"time_s":np.arange(2793)*.0025,"physics_dt_s":np.array(.0025),"policy_dt_s":np.array(.02),
        "actions":actions,"reference_frame_for_action":np.r_[0,np.repeat(np.arange(349),8)],
        "held_action":np.zeros((2793,8)),"body_net_forces_w":np.zeros((2793,9,3)),
        "contact_force_history_w":np.zeros((2793,3,9,3))}


def test_four_horizontal_panels_exact_native_400hz_indices():
    assert mod.ORDER==("ppo_beta_on","cem_beta_on","ppo_beta_off","cem_beta_off")
    assert mod.CANVAS_SIZE==(2560,750)
    assert np.array_equal(mod.native_indices(),np.arange(350)*8)
    assert mod.native_indices()[-1]==2792
    mod.check_clock_and_actions(valid_raw())


def test_solver_log_list_schema_not_dict_only_metadata_loader(tmp_path):
    path=tmp_path/"solver_log.json"
    path.write_text('[{"frame":0}]')
    assert mod.read_decisions(path)==[{"frame":0}]
    path.write_text('{"frame":0}')
    with pytest.raises(ValueError,match="list-valued"):mod.read_decisions(path)


@pytest.mark.parametrize("kind",["clock","dt","actionbound","phase","held","history","force"])
def test_clock_action_phase_and_native_contact_history_fail_closed(kind):
    raw=valid_raw()
    if kind=="clock": raw["time_s"][20]+=1e-4
    elif kind=="dt": raw["physics_dt_s"]=np.array(.005)
    elif kind=="actionbound": raw["actions"][20,0]=2.
    elif kind=="phase": raw["reference_frame_for_action"][20]+=1
    elif kind=="held": raw["held_action"][20,0]=.1
    elif kind=="history": raw["contact_force_history_w"][20,0,0,0]=1.
    else: raw["body_net_forces_w"][20,0,0]=1.
    with pytest.raises(ValueError): mod.check_clock_and_actions(raw)


def test_fresh_output_checked_before_inputs_graphics_and_physics(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"load_comparison",lambda *a,**k:pytest.fail("Must not read inputs before fresh guard"))
    with pytest.raises(ValueError,match="fresh output"): mod.run(NS(output_dir=tmp_path))


def test_no_dynamics_actor_loading_or_faked_ppo_return():
    source=Path(mod.__file__).read_text()
    assert "mj_step(" not in source and "mj_forward(" not in source and "torch.jit.load(" not in source
    assert '"PPO_total_reward_recomputed": False' in source
    assert '"comparison_is_optimizer_only": False' in source
    assert '"height_or_airborne_failure_gate": False' in source
    assert '"lower_HF_alone_establishes_stability": False' in source
    assert 'lineterminator="\\n"' in source


def test_four_original_pose_panels_fixed_shared_camera_no_zoom(monkeypatch):
    from PIL import ImageFont
    n=2793; traces,poses,targets={},{},{}
    for column,label in enumerate(mod.ORDER):
        traces[label]={"time_s":np.arange(n)*.0025,"position":np.tile([column,0.,.75],(n,1)),
            "quaternion":np.tile([1.,0.,0.,0.],(n,1)),"com_velocity":np.tile([1.,0.,0.],(n,1)),
            "contact_count":np.ones((n,2)),"contact_force":np.tile([[0.,0.,10.],[0.,0.,10.]],(n,1,1)),
            "signals":{"body_heading_rad":np.zeros(n)}}
        poses[label]=np.column_stack((np.arange(n),np.full((n,14),column)))
        targets[label]=traces[label]["position"][mod.native_indices()]+[0.,0.,-.28]
    camera=NS(lookat=np.zeros(3),distance=0.)
    runtime=NS(model=object(),data=object(),option=object(),cameras=[camera],
        font=ImageFont.load_default(),small=ImageFont.load_default(),
        renderer=NS(update_scene=lambda *a,**k:None,render=lambda:np.zeros((360,640,3),np.uint8)))
    plan={"lookat_by_case":targets,"distance_m":1.8,"azimuth_deg":135.,"elevation_deg":-30.}
    seen,maps=[],[]
    monkeypatch.setattr(mod,"_pose",lambda model,data,q:seen.append((q[0],q[1],camera.lookat.copy(),camera.distance,camera.azimuth,camera.elevation)))
    monkeypatch.setattr(mod,"_map_panel",lambda draw,trace,reference,frame,*a:maps.append((len(trace["time_s"]),frame)))
    reference={"time_s":np.arange(349)*.02,"position":np.tile([0.,0.,.75],(349,1))}
    image=mod.render_frame(runtime,traces,poses,reference,{"world_bounds":[[-1.,-1.],[2.,1.]]},plan,250)
    assert image.size==(2560,750) and maps==[(350,250)]*4
    assert [row[:2] for row in seen]==[(2000,column) for column in range(4)]
    for row,label in zip(seen,mod.ORDER):
        assert np.array_equal(row[2],targets[label][250])
        assert row[3:]==(1.8,135.,-30.)


@pytest.fixture(scope="module")
def actual_ppo():
    required=[mod.paired.DEFAULT_OFF/"manifest.json",mod.paired.DEFAULT_ON/"manifest.json",mod.paired.DEFAULT_MODEL,mod.paired.DEFAULT_REFERENCE]
    if not all(path.is_file() for path in required):pytest.skip("Private licensed local evidence unavailable")
    return mod.paired.load_comparison()


def test_compact_reproduction_all_four_cases_uses_full_native_records(actual_ppo,tmp_path):
    traces,reference,evidence=actual_ppo
    four={"ppo_beta_on":traces["beta_on"],"cem_beta_on":traces["beta_on"],
        "ppo_beta_off":traces["beta_off"],"cem_beta_off":traces["beta_off"]}
    compact=mod.compact_arrays(four,reference)
    np.savez_compressed(tmp_path/"native.npz",**compact)
    with np.load(tmp_path/"native.npz",allow_pickle=False) as z: restored={name:z[name] for name in z.files}
    actual=mod.recompute_compact(restored)
    assert list(actual)==list(mod.ORDER)
    for name in mod.ORDER:
        assert actual[name]==mod.paired.trace_metrics(four[name],reference)
        assert len(restored[name+"_time_s"])==2793
        assert actual[name]["full"]["tracking"]["available"] is False
        assert actual[name]["reference_overlap"]["tracking"]["coverage_complete"] is True


def test_diagnostics_real_native_full_and_overlap_windows(actual_ppo,tmp_path):
    traces,reference,_=actual_ppo
    four={name:traces["beta_on" if name.endswith("beta_on") else "beta_off"] for name in mod.ORDER}
    destination=tmp_path/"diagnostics.png"
    mod.draw_diagnostics(four,reference,destination)
    assert destination.stat().st_size>10000


def test_actual_completed_cem_strict_loader_and_reward_reconstruction():
    experiment=mod.DEFAULT_CEM/"experiment.json"
    if not experiment.is_file() or mod.read_json(experiment).get("status")!="completed":
        pytest.skip("New CEM full episodes are not yet completed")
    traces,reference,summary=mod.load_comparison()
    assert list(traces)==list(mod.ORDER) and len(summary["rows"])==4
    assert summary["PPO_total_reward_recomputed"] is False
    assert summary["stage_reward_function_and_weights_matched"] is True
    assert summary["comparison_is_optimizer_only"] is False
    assert summary["same_native_plant_and_recorded_initial_state"] is True
    for row in summary["rows"]:
        assert len(traces[row["id"]]["time_s"])==2793
        assert row["windows"]["full"]["tracking"]["available"] is False
        if row["controller"]=="MPC-CEM":
            assert row["CEM_stage_reward_audit"]["all_349_stages_and_15_terms_recomputed_exactly"] is True
            assert len(row["compute_ms_median_p95_max"])==3
