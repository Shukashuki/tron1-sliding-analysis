"""Independent native-record audit of nonlinear-trained PPO and three plants.

No actor, checkpoint, simulator step, smoothing or pose correction is used.
The retained NumPy law independently recomputes the float32 GPU wrenches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as metrics
import collect_ppo_k_nonlinear_tire as collection
from nonlinear_tire_contact import tangent_wrench
from ppo_backend_400hz_contract import verify_recording

ROOT=Path(__file__).resolve().parents[1]
ORDER=("K_native",*collection.CASES)
BASELINE=ROOT/"results/physx_tron1_wheel_models_20261025_v1/evidence/matched_models_v1/authored__native_coulomb"
BASELINE_SHA="cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a"
INITIAL_SHA="7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"
REFERENCE_SHA="da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path,allow_pickle=False) as a:return {name:a[name].copy() for name in a.files}


def verify_training(folder):
    driver,status,proof=collection.verify_completed(folder)
    audit=read(folder/"formal_v1/nonlinear_tire_training_audit.json")
    manifest=read(folder/"formal_v1/manifest.json");contract=read(folder/"formal_v1/policy_contract.json")
    protocol=read(folder/"source_snapshot_v1/config/ppo_k_nonlinear_tire_400hz_v1.json")
    for key,name in (("manifest_sha256","manifest.json"),("policy_contract_sha256","policy_contract.json"),
            ("K_training_audit_sha256","ground_contact_v2k_training_audit.json"),
            ("startup_audit_sha256","nonlinear_tire_startup_audit.json")):
        assert audit[key]==sha(folder/"formal_v1"/name)
    assert audit["initial_policy_state_sha256"]==proof["initial_policy_state_sha256"]==INITIAL_SHA
    assert manifest["contract"]==contract and manifest["num_envs"]==512 and manifest["iterations"]==1000
    assert audit["model_audit"]==manifest["nonlinear_tire_training"]
    assert audit["model_audit"]["settings"]==protocol["wheel_contact_model"]
    assert contract["wheel_contact_model"]==protocol["wheel_contact_model"]
    assert contract["reference_file_sha256"]==REFERENCE_SHA
    assert contract["training_domain_randomization"]==protocol["training_randomization"]
    assert audit["protocol_sha256"]==sha(folder/"source_snapshot_v1/config/ppo_k_nonlinear_tire_400hz_v1.json")
    runtime=audit["model_audit"]["final_runtime"]
    assert runtime["physical_writes"]==192064 and runtime["environment_physical_writes"]==98336768
    assert runtime["num_envs"]==512 and not runtime["invalid_measurement"]
    assert runtime["contact_power_max_w"]<=.001 and runtime["capacity_excess_max_n"]<=.001
    assert runtime["native_material_readback"]["native_wheel_static_dynamic_mu"]==0.
    assert runtime["native_material_readback"]["external_mu_per_environment_shared_between_wheels"]
    assert audit["model_audit"]["from_scratch"] and audit["model_audit"]["other14_rewards_and_K_arithmetic_unchanged"]
    assert protocol["nominal_acceptance_before_training"]==read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")["nominal_acceptance_before_training"]
    for snapshot in ("source_snapshot_v1","evaluation_snapshot_v1"):
        source=read(folder/(snapshot+".manifest.json"))
        assert all(sha(folder/snapshot/x["path"])==x["sha256"] for x in source["files"])
    return status,proof,audit,protocol


def audit_tire_trace(tire,trace):
    """Recompute forces from measured kinematics, not reported reward/metrics."""
    t=np.asarray(tire["time_before_step_s"])
    np.testing.assert_allclose(t,trace["time_s"][:-1],rtol=0,atol=1e-12)
    assert t.shape==(2792,) and tire["history_valid"].shape==(2792,)
    assert not tire["history_valid"][0] and tire["history_valid"][1:].all()
    force=np.asarray(tire["applied_tangent_force_w_n"],dtype=float)
    moment=np.asarray(tire["applied_tangent_moment_COM_w_nm"],dtype=float)
    load=np.asarray(tire["preceding_normal_load_n"],dtype=float)
    assert force.shape==moment.shape==(2792,2,3) and load.shape==(2792,2)
    assert np.isfinite(force).all() and np.isfinite(moment).all() and np.isfinite(load).all() and (load>=0).all()
    expected=trace["floor_normal_force"][:-1,:,2].clip(min=0).copy();expected[0]=0
    np.testing.assert_allclose(load,expected,rtol=1e-6,atol=1e-4)
    np.testing.assert_allclose(tire["mu"],.7,rtol=0,atol=1e-7)
    q=tire["wheel_quaternion_wxyz"].astype(float)
    assert q.shape==(2792,2,4) and np.isfinite(q).all()
    np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1.,atol=1e-5,rtol=0)
    w,x,y,z=np.moveaxis(q,-1,0)
    axle=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
    active=load>0
    value=tangent_wrench(tire["COM_velocity_w_m_s"][active],tire["angular_velocity_w_rad_s"][active],
        (tire["preceding_contact_position_w_m"]-tire["COM_position_w_m"])[active],axle[active],
        [0,0,1],load[active],law="combined_slip_magic_formula")
    np.testing.assert_allclose(force[active],value["force_w_n"],rtol=2e-5,atol=.01)
    np.testing.assert_allclose(moment[active],value["torque_about_COM_w_nm"],rtol=2e-5,atol=.002)
    np.testing.assert_array_equal(force[~active],0.)
    np.testing.assert_array_equal(moment[~active],0.)
    assert np.max(value["contact_power_w"],initial=0.)<=1e-8
    assert np.max(tire["native_patch_count"])<128
    native=tire["native_friction_force_w_n"]
    assert native.shape==(2792,2,3) and np.isfinite(native).all()
    assert np.max(np.linalg.norm(native[1:],axis=-1))<1e-3
    # Utilization is aggregate, using the chosen flat-floor Z load. It does
    # not identify separate solver points or instantaneous friction cones.
    norm=np.linalg.norm(force,axis=-1);current=trace["floor_normal_force"][1:,:,2]
    core=(t>=1.44-1e-9)&(t<=4.+1e-9)
    loaded=(load>5)&core[:,None]
    utilization=np.divide(norm,.7*load,out=np.full_like(load,np.nan),where=load>0)
    carried=(norm>1e-3)&(current<=5)&core[:,None]
    angle=np.full((2792,2),np.nan);angle[active]=value["hub_side_angle_rad"]
    side_slip=np.full((2792,2),np.nan);side_slip[active]=value["lateral_contact_slip_m_s"]
    long_slip=np.full((2792,2),np.nan);long_slip[active]=value["longitudinal_contact_slip_m_s"]
    wheel=[]
    for side in range(2):
        valid=loaded[:,side]&np.isfinite(angle[:,side])
        v=np.abs(np.degrees(angle[valid,side]))
        wheel.append(dict(samples=int(valid.sum()),median_abs_hub_side_angle_deg=float(np.median(v)) if len(v) else None,
            p95_abs_hub_side_angle_deg=float(np.percentile(v,95)) if len(v) else None,
            max_abs_hub_side_angle_deg=float(v.max()) if len(v) else None,
            mean_abs_lateral_contact_slip_m_s=float(np.mean(np.abs(side_slip[loaded[:,side],side]))) if loaded[:,side].any() else None,
            mean_abs_longitudinal_contact_slip_m_s=float(np.mean(np.abs(long_slip[loaded[:,side],side]))) if loaded[:,side].any() else None,
            loaded_lateral_contact_slip_above0p05m_s_fraction=float(np.mean(np.abs(side_slip[loaded[:,side],side])>.05)) if loaded[:,side].any() else None))
    return dict(independent_NumPy_wrench_recomputation_passed=True,
        force_max_abs_difference_n=float(np.abs(force[active]-value["force_w_n"]).max(initial=0.)),
        moment_max_abs_difference_nm=float(np.abs(moment[active]-value["torque_about_COM_w_nm"]).max(initial=0.)),
        force_tolerance=dict(rtol=2e-5,atol_n=.01),moment_tolerance=dict(rtol=2e-5,atol_nm=.002),
        reset_first_application_zero=True,normal_load_delay_physical_steps=1,
        native_friction_after_first_buffer_max_n=float(np.linalg.norm(native[1:],axis=-1).max()),
        contact_energy_at_preceding_point_j=float(value["contact_power_w"].sum()*.0025),
        core_loaded_wheel_samples=int(loaded.sum()),
        core_used_load_utilization_at_least0p9_fraction=float(np.mean(utilization[loaded]>=.9)) if loaded.any() else None,
        core_wheel_applications_with_force_gt0p001N_and_next_Fz_le5N=int(carried.sum()),
        core_left_right_loaded_hub_side_angle=wheel,
        pointwise_friction_cone_or_sensor_sideslip_identified=False,
        angle_definition="atan2(hub lateral velocity, absolute hub longitudinal velocity); moving threshold0.2m/s; load>5N",
        instantaneous_law_equivalence_claimed=False)


def load_cases(folder,reference):
    folder=Path(folder);status,proof,audit,protocol=verify_training(folder)
    assert sha(BASELINE/"physics_trajectory.npz")==BASELINE_SHA
    old=read(BASELINE/"wheel_contact_model_status.json")
    assert old["normalized_actor_sha256"]=="fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912"
    old_raw=npz(BASELINE/"physics_trajectory.npz")
    original_physx=read(ROOT/"results/physx_tron1_wheel_models_20261025_v1/analysis.json")["original_baseline_metrics"]
    original_mj_raw,_=metrics.load_baseline("mujoco")
    original_mj,_=metrics.recorded_metrics(original_mj_raw,"mujoco",reference)
    old_values,old_trace=metrics.recorded_metrics(old_raw,"physx",reference)
    old_trace["contact_force"]=old_trace["floor_normal_force"].copy()
    rows=[dict(id="K_native",engine="physx",title="K native trained | native PhysX",training_law="native_coulomb",
        validation_law="native",actor_sha256=old["normalized_actor_sha256"],metrics=old_values,
        nominal_screen=metrics.nominal_screen(old_values,original_physx,protocol,"physx"),
        recording_sha256=BASELINE_SHA)]
    traces={"K_native":old_trace};raws={"K_native":old_raw}
    for child in status["cases"]:
        case=child["id"];path=folder/"nominal_v1"/case;meta=read(path/"nonlinear_tire_evaluation_status.json")
        engine=meta["validation_backend"];report=read(path/"report.json")
        name="rollout.npz" if engine=="mujoco" else "physics_trajectory.npz"
        verify_recording(path/name,"native CPU MuJoCo" if engine=="mujoco" else "PhysX")
        if engine=="physx":
            assert report["status"]=="evaluated" and report["deterministic_start_frame"]==0
            assert report["terminal_states_captured_before_auto_reset"]
            assert not report["domain_randomization_enabled"] and not report["observation_noise_enabled"]
            assert report["runtime_device_audit"]["status"]=="verified"
            assert report["policy_export"]["validation_wheel_contact_model"]==meta
        else:
            assert report["status"]=="completed" and report["completed_full_reference"]
            assert report["hidden_resets"]==0 and report["root_state_writes"]==1
            assert report["policy_steps"]==349 and report["physics_steps"]==2792
            assert report["contract"]["validation_wheel_contact_model"]==meta
        raw=npz(path/name);values,trace=metrics.recorded_metrics(raw,engine,reference)
        trace["contact_force"]=trace["floor_normal_force"].copy()
        for field in ("position","com_velocity","angular_velocity","joint_pos","joint_vel"):
            np.testing.assert_allclose(trace[field][0],old_trace[field][0],atol=2e-6,rtol=0)
        assert abs(np.dot(trace["quaternion"][0],old_trace["quaternion"][0]))>=1-1e-7
        row=dict(id=case,engine=engine,training_law="combined_slip_magic_formula",validation_law=meta["validation_force_law"],
            title={"physx_nonlinear":"Nonlinear trained | nonlinear PhysX","physx_native":"Nonlinear trained | native PhysX",
                   "mujoco_native":"Nonlinear trained | native MuJoCo"}[case],actor_sha256=proof["actor_sha256"],
            metrics=values,nominal_screen=metrics.nominal_screen(values,original_mj if engine=="mujoco" else original_physx,protocol,engine),
            recording_sha256=sha(path/name))
        if case=="physx_nonlinear":row["independent_tire_audit"]=audit_tire_trace(npz(path/"tire_trajectory.npz"),trace)
        rows.append(row);traces[case]=trace;raws[case]=raw
    return rows,traces,raws,proof,protocol


def run(a):
    assert not a.output_dir.exists()
    assert sha(a.reference)==REFERENCE_SHA,"Score the exact training reference; no reference substitution"
    reference=metrics.metrics.reference_data(a.reference)
    rows,traces,_,proof,protocol=load_cases(a.collection,reference)
    a.output_dir.mkdir(parents=True)
    result=dict(kind="K_nonlinear_trained_PPO_PhysX_law_and_MuJoCo_transfer",status="completed_analysis",
        rows=rows,analyzer_sha256=sha(__file__),reference_sha256=sha(a.reference),formal_actor_sha256=proof["actor_sha256"],
        initial_position_velocity_joint_state_comparison_atol=2e-6,
        initial_root_quaternion_absolute_dot_min=1-1e-7,
        original_nominal_acceptance=protocol["nominal_acceptance_before_training"],
        nonlinear_parameters_are_sensitivity_assumptions_not_identified=True,
        solver_only_effect_isolated=False,qualitative_review_complete=False,task_success_claimed=False)
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(22,4.5),layout="constrained")
    axes[0].plot(reference["position"][:,0],reference["position"][:,1],"--",c="gray",label="reference")
    for row in rows:
        trace=traces[row["id"]];time=trace["time_s"]
        axes[0].plot(trace["position"][:,0],trace["position"][:,1],label=row["title"])
        axes[1].plot(time,np.linalg.norm(trace["com_velocity"][:,:2],axis=-1))
        axes[2].plot(time,trace["floor_normal_force"][:,:,2].max(-1))
        up=metrics.quaternion_rotate(trace["quaternion"],np.array([0.,0.,1.]))
        axes[3].plot(time,np.degrees(np.arccos(np.clip(up[:,2],-1,1))))
    axes[0].set_aspect("equal",adjustable="datalim");axes[0].set_title("World XY [m]");axes[0].legend(fontsize=6)
    axes[1].set_title("COM XY speed [m/s]");axes[2].set_title("Maximum wheel ground Fz [N]")
    axes[2].axhline(5,c="gray",ls="--");axes[3].set_title("Body tilt [deg], diagnostic")
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle("Same K reward and training budget | mu=0.7 | model-specific learning and transfers")
    fig.savefig(a.output_dir/"comparison.png",dpi=150);plt.close(fig)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("collection","output-dir"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reference",type=Path,default=metrics.DEFAULT_REFERENCE)
    run(p.parse_args())
