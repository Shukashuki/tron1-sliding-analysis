import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
pytest.importorskip("mujoco")
import mujoco

from run_nmpc_tracking import (
    PackedState,MujocoTrackingPredictor,build_provider,effective_settings,_metrics,selected_reference,parser,run,attach_model_nominal_plan,
)
from nmpc_tracking_objective import BODY_NAMES
from controller_references import get_reference,PRIMARY_REFERENCE_ID,SUPPLEMENT_REFERENCE_ID
from controller_trace import native_trace,WheelKinematics

ROOT=Path(__file__).parents[1]


def test_packed_state_keeps_physics_and_objective_memory_separate():
    layout=PackedState()
    q=np.arange(15)*.1;v=np.arange(14)*.2;h=np.arange(81).reshape(3,9,3)
    x=layout.pack(q,v,h,True)
    assert x.shape==(111,)
    qq,vv,hh,flag=layout.unpack(x)
    np.testing.assert_array_equal(qq,q);np.testing.assert_array_equal(vv,v)
    np.testing.assert_array_equal(hh,h);assert flag is True
    qq[:]=999
    np.testing.assert_array_equal(x[:15],q)
    with pytest.raises(ValueError,match="flag"):
        layout.unpack(np.r_[x[:-1],.5])
    assert layout.contract()["external_applied_forces"].startswith("Required zero")


def test_prediction_scenarios_never_default_to_plant_mu():
    args=parser().parse_args(["--model","m.xml","--motion","m.npz","--contract","c.json",
                              "--plant-mu","0.12","--output-dir","unused"])
    _,_,scenarios=effective_settings(args)
    assert scenarios==(.7,)
    args.prediction_mu_scenarios=[.16,.7]
    _,_,scenarios=effective_settings(args)
    assert scenarios==(.16,.7)
    args.prediction_mu_scenarios=[.7,.7]
    with pytest.raises(ValueError,match="unique"):
        effective_settings(args)


def test_truncated_metrics_do_not_claim_core_episode_result():
    ref=SimpleNamespace(frames=349,fps=50.,archive={
        "body_pos_w":np.zeros((349,9,3)),"body_lin_vel_w":np.ones((349,9,3))})
    trace={"time_s":np.array([0.,.02,.04]),"root_pos":np.zeros((3,3)),
           "base_com_velocity_w":np.ones((3,3))}
    metrics=_metrics(trace,ref)
    assert metrics["core_coverage_complete"] is False
    assert metrics["core_xy_rmse_m"] is None
    assert metrics["core_speed_mean_m_s"] is None
    assert trace["reference_xy_m"].shape==(3,2)
    assert metrics["reference_id"]==PRIMARY_REFERENCE_ID
    assert metrics["supplementary_event_metrics"]==[]


def test_selected_reference_rejects_wrong_motion_sha_before_simulation(tmp_path):
    motion=tmp_path/"wrong_motion.npz";motion.write_bytes(b"not the selected locked reference")
    # Missing reference_id on older Namespace/test fixtures retains the primary.
    with pytest.raises(ValueError,match="motion_npz_sha256.*drift_drl_traj4"):
        selected_reference(SimpleNamespace(motion=motion))
    with pytest.raises(ValueError,match="motion_npz_sha256.*doa_sa29_scale2"):
        selected_reference(SimpleNamespace(motion=motion,reference_id=SUPPLEMENT_REFERENCE_ID))
    with pytest.raises(ValueError,match="Unknown selected"):
        selected_reference(SimpleNamespace(motion=motion,reference_id="not_selected"))


def test_supplement_uses_demonstrated_core_and_keeps_event_windows_separate():
    spec=get_reference(SUPPLEMENT_REFERENCE_ID)
    ref=SimpleNamespace(frames=spec["frames"],fps=spec["fps"],archive={
        "body_pos_w":np.zeros((spec["frames"],9,3)),
        "body_lin_vel_w":np.tile([1.,0.,0.],(spec["frames"],9,1))})
    t=np.arange(spec["frames"])/spec["fps"]
    position=np.zeros((len(t),3));position[t>3.14+1e-9,0]=100.
    trace={"time_s":t,"root_pos":position,"base_com_velocity_w":np.tile([1.,0.,0.],(len(t),1))}
    metrics=_metrics(trace,ref,spec)
    assert metrics["reference_id"]==SUPPLEMENT_REFERENCE_ID
    assert metrics["role"]=="separate_supplement"
    assert metrics["stratum"]=="high_sideslip_s_avoidance"
    assert metrics["pooled_success_statistics"] is False
    assert metrics["core_window_s"]==[0.,3.14]
    assert metrics["core_xy_rmse_m"]==0.
    assert metrics["core_speed_retention_mean_ratio"]==1.
    assert metrics["observed_window_xy_rmse_m"]>0.
    events=metrics["supplementary_event_metrics"]
    assert [event["window_s"] for event in events]==[[.78,1.54],[1.88,3.04]]
    assert all(event["coverage_complete"] and event["xy_rmse_m"]==0. for event in events)
    assert all(event["descriptive_not_success_gate"] for event in events)


def test_supplement_partial_smoke_reports_only_fully_covered_event_metrics():
    spec=get_reference(SUPPLEMENT_REFERENCE_ID)
    ref=SimpleNamespace(frames=spec["frames"],fps=spec["fps"],archive={
        "body_pos_w":np.zeros((spec["frames"],9,3)),"body_lin_vel_w":np.ones((spec["frames"],9,3))})
    t=np.arange(81)/50.
    trace={"time_s":t,"root_pos":np.zeros((len(t),3)),"base_com_velocity_w":np.ones((len(t),3))}
    metrics=_metrics(trace,ref,spec)
    assert metrics["core_coverage_complete"] is False
    assert metrics["core_xy_rmse_m"] is None
    first,second=metrics["supplementary_event_metrics"]
    assert first["coverage_complete"] is True and first["xy_rmse_m"]==0.
    assert second["coverage_complete"] is False and second["xy_rmse_m"] is None


def test_runner_refuses_existing_output_without_reading_inputs(tmp_path):
    with pytest.raises(FileExistsError,match="Preserving"):
        run(SimpleNamespace(output_dir=tmp_path))


@pytest.fixture
def actual_predictor():
    base=ROOT/"outputs/car_failure_sideslip_20261015/raw"
    model=base/"mujoco_model.xml"
    motion=Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
    contract=base/"mu0p180/isaac/policy_contract.json"
    if not all(p.is_file() for p in (model,motion,contract)):
        pytest.skip("Licensed TRON1 model/local locked motion unavailable")
    settings=json.loads((ROOT/"config/nmpc_height_smooth.json").read_text())
    profile=json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text())
    provider=build_provider(model,motion,json.loads(contract.read_text()),settings,(.7,))
    assert selected_reference(SimpleNamespace(motion=motion))["reference_id"]==PRIMARY_REFERENCE_ID
    return MujocoTrackingPredictor(provider,model,profile,(.7,))


def test_actual_transition_preserves_substep_contact_history_and_reset_semantics(actual_predictor):
    p=actual_predictor
    # This is one isolated transition, not a closed-loop controller evaluation.
    x=p.layout.pack(p.provider.targets[0,:15],p.provider.targets[0,15:],
                    np.zeros(p.layout.history_shape),True)
    initial=x.copy();action=p.provider.nominal_action[0].copy()
    next_state=p.transition(x,action,mu=.7)
    np.testing.assert_array_equal(x,initial)
    assert p.layout.unpack(next_state)[-1] is False
    assert np.isfinite(next_state).all()
    assert np.any(p.layout.unpack(next_state)[2])
    pre,post=p.measure(x),p.measure(next_state)
    held=p.objective.hold_reference(p.reference_frame(0),pre)
    expected=p.objective.evaluate(post,held,action,np.zeros(8),0.,reset_sample=True)
    assert expected["raw_terms"]["height_vertical_acceleration"]==0.
    assert p.stage_cost(x,action,next_state,0,previous_action=np.zeros(8),previous_vz=0.)==pytest.approx(expected["stage_cost"])
    # A second prediction from the same packed state is deterministic and does
    # not inherit the previous candidate's hidden simulator warm state.
    np.testing.assert_allclose(p.transition(x,action,mu=.7),next_state,atol=1e-12)
    with pytest.raises(ValueError,match="declared"):
        p.transition(x,action,mu=.12)


def test_actual_bias_seed_never_uses_old_qr_or_tvlqr(actual_predictor):
    p=actual_predictor
    x=p.layout.pack(p.provider.targets[0,:15],p.provider.targets[0,15:],np.zeros(p.layout.history_shape))
    plan=p.initial_candidate(x,348,previous_action=np.zeros(8),previous_vz=0.,horizon_steps=3)
    assert plan.shape==(3,8)
    np.testing.assert_array_equal(plan,np.tile(p.provider.nominal_action[-1],(3,1)))
    assert np.max(np.abs(plan))<=1.
    for i,name in enumerate(BODY_NAMES):
        original=p.reference.body_names.index(name)
        for key in ("body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"):
            np.testing.assert_array_equal(p.reference_frame(0)[key][i],p.reference.archive[key][0,original])


def test_one_python_control_interval_records_native_contacts_clock_material_and_float32_action(actual_predictor,tmp_path):
    """One authorized CPU control interval only; no timing/quality benchmark."""
    base=ROOT/"outputs/car_failure_sideslip_20261015/raw"
    output=tmp_path/"one_interval_python"
    args=parser().parse_args([
        "--model",str(base/"mujoco_model.xml"),
        "--motion","/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz",
        "--contract",str(base/"mu0p180/isaac/policy_contract.json"),
        "--plant-mu","0.23","--output-dir",str(output),"--max-policy-steps","1",
        "--horizon-steps","1","--control-blocks","1","--candidates","2",
        "--iterations","1","--elite-count","1","--prediction-backend","python",
    ])
    report=run(args)
    saved=json.loads((output/"report.json").read_text())
    assert saved==json.loads(json.dumps(report))  # JSON represents tuple CEM stds as lists.
    assert report["status"]=="smoke_completed" and report["truncated_smoke"] is True
    assert report["completed_policy_steps"]==1 and report["duration_s"]==pytest.approx(.02)
    assert report["reference_id"]==PRIMARY_REFERENCE_ID
    assert report["core_coverage_complete"] is False and report["core_xy_rmse_m"] is None
    assert report["prediction_backend"]=="python" and report["native_prediction_backend"] is None
    assert report["action_parameterization"]["mode"]=="absolute_blocks"
    assert report["action_parameterization"]["baseline_copied_and_fixed_per_solve"] is False
    assert report["prediction_mu_scenarios"]==[.7]  # Never oracle-selected from plant .23.
    assert report["prediction_receives_plant_mu"] is False
    assert report["plant_material_readback"]["compiled_sliding_friction_range"]==[.23,.23]
    assert report["physics_dt_s"]==.005 and report["control_dt_s"]==.02
    assert report["contact_threshold_n"]==5.
    assert report["wheel_lateral_contact_slip_evidence_kind"]=="actual_ground_contact_points"
    assert report["contact_slip_definition"]==(
        "Per-wheel force-weighted contact-point lateral-speed RMS; >threshold normal force, ground-filtered. "
        "Zero with no active contact is unavailable, not rolling.")
    for field in ("learned_policy_used","tvlqr_used","old_qr_used","root_state_prescribed_after_reset",
                  "external_root_forces_used","pose_height_failure_gate","real_time_verified"):
        assert report[field] is False
    for name,value in (("horizon_steps",1),("control_blocks",1),("candidates",2),("iterations",1),("elite_count",1)):
        assert report["effective_cem_config"][name]==value
    assert report["rollout_sha256"]==hashlib.sha256((output/"rollout.npz").read_bytes()).hexdigest()
    assert report["solver_log_sha256"]==hashlib.sha256((output/"solver_log.json").read_bytes()).hexdigest()
    with np.load(output/"rollout.npz",allow_pickle=False) as archive:
        trace={name:archive[name].copy() for name in archive.files}
    canonical=native_trace(trace,"nmpc")  # Required measured ground fields must really be present.
    np.testing.assert_allclose(canonical["time_s"],np.arange(5)*.005,atol=1e-12,rtol=0)
    assert canonical["physics_dt_s"]==.005 and float(trace["policy_dt_s"])==.02
    assert canonical["joint_pos"].shape==(5,8) and canonical["joint_vel"].shape==(5,8)
    assert canonical["contact_count"].shape==(5,2) and canonical["contact_force"].shape==(5,2,3)
    assert np.all(canonical["contact_count"]>=0)
    np.testing.assert_array_equal(canonical["contact_count"],np.round(canonical["contact_count"]))
    assert np.all(canonical["contact_slip"]>=0)
    assert trace["nonwheel_ground_contact_count"].shape==(5,)
    assert trace["nonwheel_ground_contact_force_norm_n"].shape==(5,)
    assert np.all(trace["nonwheel_ground_contact_count"]>=0)
    assert np.all(trace["nonwheel_ground_contact_force_norm_n"]>=0)
    np.testing.assert_array_equal(trace["action_was_applied"],[False,True,True,True,True])
    np.testing.assert_array_equal(trace["reference_frame_for_action"],np.zeros(5,dtype=int))
    np.testing.assert_array_equal(trace["held_action"][0],np.zeros(8))
    np.testing.assert_array_equal(trace["held_action"][1:],np.tile(trace["actions"][0],(4,1)))
    assert np.isfinite(trace["actions"]).all() and np.max(np.abs(trace["actions"]))<=1.
    np.testing.assert_array_equal(trace["actions"],trace["actions"].astype(np.float32).astype(float))
    decisions=json.loads((output/"solver_log.json").read_text())
    assert len(decisions)==1 and decisions[0]["reference_frame_for_reward"]==0
    np.testing.assert_array_equal(trace["actions"][0],np.asarray(decisions[0]["raw_cem_action"]).astype(np.float32).astype(float))
    assert decisions[0]["solver"]["optimization_failed"] is False
    np.testing.assert_array_equal(trace["applied_joint_torque_nm"][0],np.zeros(8))
    for substep in range(1,5):
        expected=actual_predictor.provider.torque(trace["held_action"][substep],trace["qpos"][substep-1],trace["qvel"][substep-1])
        np.testing.assert_allclose(trace["applied_joint_torque_nm"][substep],expected,atol=1e-12,rtol=0)
    # Canonical measured velocities/poses must also pass offline no-step FK.
    audit=WheelKinematics(actual_predictor.models[.7]).reconstruct(canonical)["audit"]
    assert audit["maximum_wheel_link_position_error_m"]<1e-10
    assert audit["maximum_root_com_velocity_reconstruction_error"]<1e-10


def test_model_nominal_plan_default_off_does_not_import_loader_or_touch_predictor(monkeypatch):
    monkeypatch.setitem(sys.modules,"model_nominal_plan",None)
    target=SimpleNamespace()
    assert attach_model_nominal_plan(SimpleNamespace(),target,None,{})=={"enabled":False}
    assert not hasattr(target,"initial_candidates")
    args=parser().parse_args(["--model","m.xml","--motion","m.npz","--contract","c.json","--plant-mu",".7","--output-dir","unused"])
    assert args.model_nominal_plan is None


@pytest.mark.parametrize("model_arg", ["model","mujoco_model"])
def test_model_nominal_plan_attaches_only_additive_hook_with_exact_identity_arguments(tmp_path,monkeypatch,model_arg):
    paths={name:tmp_path/name for name in ("model","motion","contract","plan.json")}
    for name,path in paths.items():path.write_text(name)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata={"enabled":True,"artifact_sha256":digest(paths["plan.json"]),"contract":{"model_generated":True},"offline_generation_budget":{"candidate_evaluations":12}}
    calls=[]
    class Plan:
        def report_contract(self):return metadata.copy()
        def initial_candidates(self,state,index,**kwargs):return {"model_nominal_plan":np.zeros((kwargs["horizon_steps"],8))}
    def load(path,**kwargs):calls.append((path,kwargs));return Plan()
    monkeypatch.setitem(sys.modules,"model_nominal_plan",SimpleNamespace(load_model_nominal_plan=load))
    args=SimpleNamespace(model_nominal_plan=paths["plan.json"],motion=paths["motion"],contract=paths["contract"],**{model_arg:paths["model"]})
    objective={"kind":"height_smooth","absolute_height_target":False}
    scalar=SimpleNamespace(objective=SimpleNamespace(contract=lambda:objective),provider=SimpleNamespace(dt=.02))
    target=SimpleNamespace(initial_candidate=lambda *a,**k: "original bias candidate")
    result=attach_model_nominal_plan(args,target,scalar,{"reference_id":"primary","robot_reference_npz_sha256":"r"*64})
    assert result==metadata and target.initial_candidate()=="original bias candidate"
    assert set(target.initial_candidates(np.zeros(111),3,horizon_steps=30,previous_action=np.zeros(8),previous_vz=0.))=={"model_nominal_plan"}
    assert calls[0][1]["expected_motion_sha256"]==digest(paths["motion"])
    assert calls[0][1]["expected_model_sha256"]==digest(paths["model"])
    assert calls[0][1]["expected_actuator_contract_sha256"]==digest(paths["contract"])
    assert calls[0][1]["expected_robot_reference_sha256"]=="r"*64
    assert calls[0][1]["expected_objective_contract"]==objective
    assert calls[0][1]["control_dt_s"]==.02
    with pytest.raises(ValueError,match="overwrite"):
        attach_model_nominal_plan(args,target,scalar,{})


def test_model_nominal_plan_bad_provenance_is_fatal_before_hook_install(tmp_path,monkeypatch):
    artifact=tmp_path/"metadata.json";artifact.write_text("{}")
    bad=SimpleNamespace(report_contract=lambda:{"enabled":True,"artifact_sha256":"0"*64},initial_candidates=lambda *a,**k:{})
    monkeypatch.setitem(sys.modules,"model_nominal_plan",SimpleNamespace(load_model_nominal_plan=lambda *a,**k:bad))
    args=SimpleNamespace(model_nominal_plan=artifact,model=artifact,motion=artifact,contract=artifact)
    scalar=SimpleNamespace(objective=SimpleNamespace(contract=lambda:{}),provider=SimpleNamespace(dt=.02))
    target=SimpleNamespace()
    with pytest.raises(ValueError,match="inconsistent provenance"):
        attach_model_nominal_plan(args,target,scalar,{"reference_id":"primary","robot_reference_npz_sha256":"1"*64})
    assert not hasattr(target,"initial_candidates")
