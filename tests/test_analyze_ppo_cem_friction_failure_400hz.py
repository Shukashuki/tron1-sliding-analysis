"""Frozen-controller, partial-grid and honest failure-diagnostic regression."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_ppo_cem_friction_failure_400hz as mod
from controller_metrics import heading_signals


def native_fixture(offset=(0., 0.), fall_after=None):
    time = np.arange(2793)*.0025
    rt = np.arange(349)*.02
    quaternion = np.tile([1.,0.,0.,0.],(2793,1))
    if fall_after is not None:
        quaternion[time >= fall_after] = [2**-.5, 2**-.5, 0., 0.]
    velocity = np.tile([1.,0.,0.],(2793,1))
    wheel_q = np.tile([1.,0.,0.,0.],(2793,2,1))
    wheel_velocity = np.tile([1.,0.,0.],(2793,2,1))
    trace = {"time_s": time,"physics_dt_s": .0025,
        "position": np.column_stack((time+offset[0],np.full(2793,offset[1]),np.full(2793,.75))),
        "quaternion":quaternion,"com_velocity":velocity,"angular_velocity":np.zeros((2793,3)),
        "joint_pos":np.zeros((2793,8)),"joint_vel":np.zeros((2793,8)),
        "wheel_quaternion_wxyz":wheel_q,"wheel_center_velocity_w":wheel_velocity,
        "contact_count":np.ones((2793,2)),"contact_force":np.tile([[0.,0.,10.],[0.,0.,10.]],(2793,1,1)),
        "contact_slip":np.tile([.2,.3],(2793,1)),"wheel_pos":np.zeros((2793,2,3)),
        "signals":heading_signals(quaternion,velocity,wheel_q,wheel_velocity),
        "body_heading_geometry_valid":np.ones(2793,dtype=bool),
        "wheel_heading_geometry_valid":np.ones((2793,2),dtype=bool)}
    reference = {"time_s":rt,"position":np.column_stack((rt,rt*0.,rt*0.+.75)),
        "velocity":np.tile([1.,0.,0.],(349,1)),"quaternion":np.tile([1.,0.,0.,0.],(349,1))}
    return trace,reference


def test_locked17grid_fourconditions_partial_absent_is_not_a_pass():
    row = {"controller":"PPO","condition":"beta_on","mu":.15,"status":"running"}
    rows = mod.canonical_cases([row])
    assert len(rows)==68 and len(set(r["id"] for r in rows))==68
    assert len(mod.GRID)==17 and mod.GRID[0]==.01 and mod.GRID[-1]==2.
    assert sum(r["status"]=="missing" for r in rows)==67
    assert next(r for r in rows if r["group"]=="ppo_beta_on" and r["mu"]==.15)["status"]=="running"
    with pytest.raises(ValueError,match="Duplicate"):
        mod.canonical_cases([row,row])


@pytest.mark.parametrize("field,value",[("mu",.11),("mu",None),("status","success"),("controller","SAC"),("condition","zero_slip")])
def test_canonical_grid_rejects_different_protocol(field,value):
    row={"controller":"PPO","condition":"beta_on","mu":.15,"status":"running"};row[field]=value
    with pytest.raises(ValueError):mod.canonical_cases([row])


def test_partial_cell_is_never_opened_even_if_partial_file_exists(tmp_path):
    cell={"status":"running","directory":str(tmp_path)}
    with pytest.raises(ValueError,match="Only completed"):
        mod.read_completed(cell)


def test_completed_raw_and_report_sha_are_both_checked(tmp_path):
    report=tmp_path/"report.json";report.write_text('{"status":"completed"}')
    raw=tmp_path/"rollout.npz";np.savez(raw,time_s=np.arange(3))
    cell={"status":"completed","directory":str(tmp_path),"report_sha256":mod.sha256(report),"recording_sha256":mod.sha256(raw)}
    arrays,metadata,ledger=mod.read_completed(cell)
    assert np.array_equal(arrays["time_s"],np.arange(3)) and metadata["status"]=="completed"
    assert ledger["recording"]["sha256"]==cell["recording_sha256"]
    report.write_text('{"status":"failed"}')
    with pytest.raises(ValueError,match="SHA differs"):mod.read_completed(cell)


def test_read_completed_rejects_path_escape(tmp_path):
    outside=tmp_path/"outside.json";outside.write_text('{}')
    folder=tmp_path/"case";folder.mkdir()
    cell={"status":"completed","directory":str(folder),"report_file":"../outside.json"}
    with pytest.raises(ValueError,match="escaped"):mod.read_completed(cell)


def test_tangent_error_signed_coordinates_without_alignment():
    trace,reference=native_fixture((.4,-.2))
    result=mod.along_cross_error(trace["time_s"],trace["position"],reference,(1.44,4.))
    assert result["valid_samples"]==result["eligible_samples"]==1025
    assert result["along_track"]["signed_mean_m"]==pytest.approx(.4)
    assert result["cross_track"]["signed_mean_m"]==pytest.approx(-.2)
    assert result["cross_track"]["rms_m"]==pytest.approx(.2)
    assert result["time_alignment_or_closest_path_projection"] is False


def test_tangent_uses_reference_velocity_not_axis_or_actual_motion():
    trace,reference=native_fixture((.4,-.2))
    reference["velocity"][:]=[0.,1.,0.]
    result=mod.along_cross_error(trace["time_s"],trace["position"],reference,(1.44,4.))
    assert result["along_track"]["signed_mean_m"]==pytest.approx(-.2)
    assert result["cross_track"]["signed_mean_m"]==pytest.approx(-.4)


def test_low_reference_speed_has_no_fabricated_tangent_and_no_extrapolation():
    trace,reference=native_fixture()
    reference["velocity"][:]=[.3,0.,0.]
    result=mod.along_cross_error(trace["time_s"],trace["position"],reference,(1.44,4.))
    assert result["valid_samples"]==0 and result["cross_track"]["rms_m"] is None
    with pytest.raises(ValueError,match="cannot extrapolate"):
        mod.along_cross_error(trace["time_s"],trace["position"],reference,(0.,6.98))


def test_course_unwrap_does_not_bridge_invalid_zero_speed_gap():
    time=np.arange(6)*.1
    signals={"root_beta_valid":np.array([True,True,False,False,True,True]),
             "course_rad":np.radians([170.,-170.,0.,0.,-90.,-60.])}
    result=mod.contiguous_course_changes(time,signals,(0.,.5))
    assert result["net_course_change_deg"] is None
    assert result["valid_samples"]==4 and len(result["contiguous_segments"])==2
    assert [s["signed_course_change_deg"] for s in result["contiguous_segments"]]==pytest.approx([20.,30.])
    signals["root_beta_valid"][:]=True
    result=mod.contiguous_course_changes(time,signals,(0.,.5))
    assert result["complete_valid_window"] is True and result["net_course_change_deg"] is not None


def test_full_native_tracking_remains_unavailable_not_reference_extrapolated():
    trace,reference=native_fixture((.1,0.),fall_after=4.1)
    result=mod.failure_metrics(trace,reference)
    assert result["windows"]["core"]["tracking"]["xy_rmse_m"]==pytest.approx(.1)
    assert result["windows"]["reference_overlap"]["tracking"]["xy_rmse_m"]==pytest.approx(.1)
    assert result["windows"]["full"]["tracking"]["available"] is False
    assert result["windows"]["post_core"]["tracking"]["available"] is False
    assert result["windows"]["full"]["maximum_tilt_deg"]==pytest.approx(90.)
    assert result["core_large_xy_screen"] is False and result["overlap_large_xy_screen"] is False
    assert result["height_airborne_tilt_or_beta_failure_gate"] is False


def test_persistent_departure_has_onset_confirmation_and_censoring_no_endpoint_hold():
    trace,reference=native_fixture()
    trace["position"][trace["time_s"]>=4.2,1]=.5
    result=mod.failure_metrics(trace,reference)
    assert result["core_large_xy_screen"] is False and result["overlap_large_xy_screen"] is True
    event=result["failure_diagnostics"]["reference_overlap"]["persistent_xy_gt0p30m"]["first_persistent_event"]
    assert event["onset_s"]==pytest.approx(4.2) and event["confirmation_s"]==pytest.approx(4.3)
    assert event["end_s"]==pytest.approx(6.96) and event["right_window_censored"] is True


def test_ground_contact_metrics_require_each_wheel_count_and_positive_load():
    trace,reference=native_fixture()
    trace["contact_force"][:,0,2]=0.
    result=mod.failure_metrics(trace,reference)
    contact=result["failure_diagnostics"]["core"]["turn_and_ground_contact_heading_geometry"]
    assert contact["ground_contact_lateral_slip_m_s"]["left"] is None
    assert contact["ground_contact_lateral_slip_m_s"]["right"]==pytest.approx(.3)
    assert contact["ground_contact_heading_geometry"]["contact_fraction_left_right"]==[0.,1.]


@pytest.mark.parametrize("kind",["clock","dt","count"])
def test_no200hz_reset_splice_or_noninteger_contacts(kind):
    trace,_=native_fixture()
    if kind=="clock":trace["time_s"][5]+=.1
    elif kind=="dt":trace["physics_dt_s"]=.005
    else:trace["contact_count"][5,0]=.5
    with pytest.raises(ValueError):mod.clock_audit(trace)


def test_nominal_failure_is_not_newly_attributed_to_mu():
    trace,reference=native_fixture((.4,0.))
    nominal=mod.failure_metrics(trace,reference);nominal["evaluated"]=True
    trace["position"][:,1]=.3
    changed=mod.failure_metrics(trace,reference);changed["evaluated"]=True
    result=mod.baseline_differences(changed,nominal)
    assert result["nominal_core_large_xy_screen"] is True
    assert result["new_core_screen_exceedance_relative_to_nominal"] is False
    assert result["core"]["xy_rmse_change_m"]==pytest.approx(.1)
    assert result["friction_causality_or_statistical_significance_claimed"] is False
    assert mod.baseline_differences(changed,None)["available"] is False


def test_actual_course_change_is_compared_to_real_reference_course_without_alignment():
    trace,reference=native_fixture()
    reference["velocity"][:,0]=np.cos(reference["time_s"]*.2)
    reference["velocity"][:,1]=np.sin(reference["time_s"]*.2)
    result=mod.failure_metrics(trace,reference)
    core=result["failure_diagnostics"]["core"]
    assert core["course"]["net_course_change_deg"]==pytest.approx(0.)
    assert core["reference_course"]["net_course_change_deg"]==pytest.approx(np.degrees((4.-1.44)*.2))


def test_no_completed_cells_is_valid_partial_summary_not_a_success(tmp_path,monkeypatch):
    reference_path=tmp_path/"motion.npz";reference_path.write_bytes(b'reference fixture')
    model_path=tmp_path/"model.xml";model_path.write_bytes(b'model fixture')
    report=tmp_path/"report.json";report.write_text('{}')
    _,reference=native_fixture()
    baseline={"reference_sha256":mod.sha256(reference_path),"model_sha256":mod.sha256(model_path),
        "rows":[{"id":group,"directory":str(tmp_path)} for group in mod.ORDER],"CEM_input_ledger_sha256":{}}
    monkeypatch.setattr(mod.metrics,"load_fk",lambda *a:(None,{"isolated_FK_only":True}))
    monkeypatch.setattr(mod,"read_completed",lambda *a:pytest.fail("Partial cases must not read live output"))
    summary,traces=mod.analyze_cases([],reference_path,model_path,nominal_data=({},reference,baseline))
    assert summary["completed_verified_cells"]==0 and summary["remaining_or_unavailable_cells"]==68
    assert summary["grid_complete"] is False and traces=={}
    assert all(row["evaluated"] is False and row["core_large_xy_screen"] is None for row in summary["rows"])
    assert all(row["difference_from_own_nominal_mu0p7"]["available"] is False for row in summary["rows"])


def test_material_override_changes_only_sliding_mu_not_other_contact_law():
    baseline={"native_mapping_verified":True,"compiled_native_timestep_s":.0025,"explicit_contact_pair_count":0,
        "pair_coefficient":.7,"geom_friction":[[.7,.01,.001],[.7,.01,.001]],
        **{name:[] for name in ("wheel_geom_ids","colliding_geom_ids","colliding_geom_names","geom_priority",
            "geom_condim","geom_solref","geom_solimp","native_plant_readback")}}
    material=copy.deepcopy(baseline);material["pair_coefficient"]=.15
    for row in material["geom_friction"]:row[0]=.15
    mod.check_material(material,baseline,.15)
    material["geom_friction"][0][1]=.03
    with pytest.raises(ValueError,match="torsional/rolling"):mod.check_material(material,baseline,.15)


def report_fixture(controller="MPC-CEM"):
    material={"native_mapping_verified":True,"compiled_native_timestep_s":.0025,"explicit_contact_pair_count":0,
        "pair_coefficient":.7,"geom_friction":[[.7,.01,.001],[.7,.01,.001]],
        **{name:[] for name in ("wheel_geom_ids","colliding_geom_ids","colliding_geom_names","geom_priority",
            "geom_condim","geom_solref","geom_solimp","native_plant_readback")}}
    cell={"mu":.7,"condition":"beta_on","controller":controller,"recording_sha256":"recording"}
    profile={"contact_friction":.7,"action_rate_weight":-.5,"reward_weights":{"sideslip":2.,"grounded":.4}}
    report={"status":"completed","motion_sha256":"motion","model_sha256":"model"}
    if controller=="MPC-CEM":
        report.update(kind="400hz_friction_sweep_CEM_with_completed_PPO_stage_reward",condition="beta_on",
            plant_mu=.7,prediction_mu_scenarios=[.7],prediction_friction_known=True,physics_dt_s=.0025,
            policy_dt_s=.02,decimation=8,completed_policy_steps=349,native_samples=2793,truncated_smoke=False,
            task_profile=profile,CEM_config={"candidates":64},objective_contract={"fifteen_terms":True},
            discount_factor=.99,prediction_state_contract={"dimension":111},action_parameterization={"kind":"residual"},
            soft_joint_pos_limits=[[-1.,1.]],soft_joint_limits_source="baseline",rollout_sha256="recording",
            termination_or_reset_after_initialization=False,height_contact_tilt_failure_gate=False,
            PPO_actions_or_future_states_used_for_initial_plan=False,new_training_or_policy_inference=False,
            controller_tuned_on_this_episode=False,plant_contact_material_audit=material,
            prediction_contact_material_audit=copy.deepcopy(material))
    else:
        proof={"checkpoint_sha256":"frozen_checkpoint","normalized_actor_sha256":"frozen_actor",
            "training_backend":"physx","cohort_id":"400Hz","protocol_sha256":"protocol","generated_source_sha256":"evaluator",
            "status":"verified","smoke_only":False,"old_200hz_actor_used":False,"additional_training_updates":0,
            "native_states":2793,"policy_decisions":349,"recording_sha256":"recording"}
        report.update(contract={"task_profile":profile,"physics_dt":.0025,"decimation":8},
            backend_400hz_cohort_evaluation=proof,policy_steps=349,physics_steps=2792,hidden_resets=0,
            root_state_writes=1,completed_full_reference=True,contact_material_audit=material)
    return cell,report


@pytest.mark.parametrize("controller",["PPO","MPC-CEM"])
def test_report_contract_allows_declared_mu_only(controller):
    cell,baseline=report_fixture(controller)
    report=copy.deepcopy(baseline);cell["mu"]=1.8
    if controller=="PPO":
        report["contract"]["task_profile"]["contact_friction"]=1.8
        materials=[report["contact_material_audit"]]
    else:
        report["plant_mu"]=1.8;report["prediction_mu_scenarios"]=[1.8]
        report["task_profile"]["contact_friction"]=1.8
        materials=[report[name] for name in ("plant_contact_material_audit","prediction_contact_material_audit")]
    for material in materials:
        material["pair_coefficient"]=1.8
        for row in material["geom_friction"]:row[0]=1.8
    mod.check_report(cell,report,baseline,"motion","model")
    profile=report["contract"]["task_profile"] if controller=="PPO" else report["task_profile"]
    profile["action_rate_weight"]=-.1
    with pytest.raises(ValueError,match="reward|profile"):
        mod.check_report(cell,report,baseline,"motion","model")


@pytest.mark.parametrize("key,value",[("prediction_mu_scenarios",[.3]),("prediction_friction_known",False),
    ("completed_policy_steps",348),("controller_tuned_on_this_episode",True),("termination_or_reset_after_initialization",True)])
def test_cem_rejects_different_oracle_truncated_tuned_or_reset_variant(key,value):
    cell,baseline=report_fixture();report=copy.deepcopy(baseline);report[key]=value
    with pytest.raises(ValueError):mod.check_report(cell,report,baseline,"motion","model")


@pytest.mark.parametrize("key,value",[("checkpoint_sha256","different"),("training_backend","mjlab"),
    ("old_200hz_actor_used",True),("additional_training_updates",1),("native_states",1397)])
def test_ppo_rejects_different_actor_or_cohort_and_old_native_clock(key,value):
    cell,baseline=report_fixture("PPO");report=copy.deepcopy(baseline)
    report["backend_400hz_cohort_evaluation"][key]=value
    with pytest.raises(ValueError):mod.check_report(cell,report,baseline,"motion","model")


def test_live_manifest_snapshot_supports_runner_jobs_and_sha(tmp_path):
    path=tmp_path/"experiment.json";path.write_text('{"status":"running","jobs":[]}')
    values,proof=mod.manifest_snapshot(path)
    assert values==[] and proof["snapshot_sha256"]==mod.sha256(path)
    path.write_text('{"status":"completed","jobs":[{}]}')
    assert proof["runner_status_at_snapshot"]=="running" and proof["snapshot_sha256"]!=mod.sha256(path)


def test_fresh_analysis_guard_before_manifest_or_nominal_read(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(NS(output_dir=tmp_path))


def test_no_dynamics_or_policy_evaluation_implementation():
    source=Path(mod.__file__).read_text()
    assert "mj_step(" not in source and "mj_forward(" not in source and "torch.jit.load(" not in source
    assert "Historical200Hz grids are not pooled" in source
    assert '"statistical_success_probability_or_causal_effect_claimed": False' in source
    assert 'lineterminator="\\n"' in source


def test_partial_figures_four_horizontal_curves_gray_missing_and_shared_worldbounds(tmp_path):
    from PIL import Image
    rows=mod.canonical_cases([])
    for row in rows:
        row.update(evaluated=False,core_large_xy_screen=None,overlap_large_xy_screen=None)
    trace,reference=native_fixture()
    row=next(r for r in rows if r["group"]=="ppo_beta_on" and r["mu"]==.7)
    row.update(mod.failure_metrics(trace,reference),evaluated=True)
    summary={"rows":rows,"completed_verified_cells":1}
    mod.render_curves(summary,tmp_path)
    bounds=mod.render_topdown(summary,{row["id"]:trace},reference,tmp_path)
    assert Image.open(tmp_path/"friction_curves.png").size==(3200,832)
    assert (tmp_path/"xy_screen_matrix.png").is_file() and (tmp_path/"selected_topdown.png").is_file()
    assert bounds["poses_filtered_aligned_or_corrected"] is False
    compact=mod.metric_rows(summary)
    evaluated=next(r for r in compact if r["evaluated"])
    assert evaluated["core_speed_retention"]==pytest.approx(1.)
    assert all(r["core_large_xy_screen"] is None for r in compact if not r["evaluated"])


def test_compact_native_signals_recompute_without_model_mesh_weights_or_dynamics(tmp_path):
    trace,reference=native_fixture((.1,-.2))
    key=mod.case_key("ppo_beta_on",.7)
    destination=tmp_path/"native.npz"
    np.savez_compressed(destination,**mod.compact_arrays({key:trace},reference))
    with np.load(destination,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    assert arrays["ppo_beta_on__mu0p700__time_s"].shape==(2793,)
    actual=mod.recompute_compact(arrays)[key]
    assert actual==mod.failure_metrics(trace,reference)
    assert actual["windows"]["core"]["tracking"]["xy_rmse_m"]==pytest.approx(np.hypot(.1,.2))
