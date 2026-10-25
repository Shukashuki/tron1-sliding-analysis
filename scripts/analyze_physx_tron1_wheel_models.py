"""Native metric and delayed-wrench audit of six frozen-K model transfers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as base
from nonlinear_tire_contact import tangent_wrench

ROOT=Path(__file__).resolve().parents[1]
ACTOR="fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912"
BASELINE="cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a"
NAMES=tuple(g+"__"+law for law in ("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula") for g in ("authored","slices9"))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path,allow_pickle=False) as a:return {n:a[n].copy() for n in a.files}


def load_cases(collection,reference,protocol,*,original_metrics=None):
    collection=Path(collection);driver=json.loads((collection/"matched_models_v1/status.json").read_text())
    assert driver["status"]=="six_matched_offset_PhysX_transfer_cases_completed_require_analysis"
    assert [c["id"] for c in driver["cases"]]==list(NAMES)
    source=json.loads((collection/"source_snapshot_v2.manifest.json").read_text())
    source_pins={Path(r["path"]).name:r["sha256"] for r in source["files"]}
    if original_metrics is None:
        original_raw,_=base.load_baseline("physx")
        original,_=base.recorded_metrics(original_raw,"physx",reference)
    else:original=original_metrics
    rows=[];traces={};raws={};contacts={}
    for child in driver["cases"]:
        name=child["id"];folder=collection/"matched_models_v1"/name
        assert child["status"]=="completed_verified" and child["returncode"]==0
        d=json.loads((folder/"wheel_contact_model_status.json").read_text());report=json.loads((folder/"report.json").read_text())
        assert report["wheel_contact_model_transfer"]==d
        assert sha(folder/"wheel_contact_model_status.json")==child["metadata_sha256"]
        assert d["status"]=="completed_requires_analysis" and d["kind"]=="frozen_K_PhysX_wheel_contact_model_transfer"
        assert d["normalized_actor_sha256"]==ACTOR and d["native_states"]==2793 and d["policy_decisions"]==349
        assert d["mu"]==.7 and d["physics_dt_s"]==.0025 and d["policy_dt_s"]==.02
        assert d["retraining_or_reward_action_reference_change"] is False
        assert d["native_audit"]["masses_COM_inertias_armature_damping_match_baseline"]
        assert report["status"]=="evaluated" and report["deterministic_start_frame"]==0
        assert report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
        assert report["terminal_states_captured_before_auto_reset"] is True
        assert all(source_pins[Path(p).name]==digest for p,digest in d["source_sha256"].items())
        for n in ("physics_trajectory","tire_trajectory"):
            assert sha(folder/(n+".npz"))==d[n+"_sha256"]==child[n+"_sha256"]
        for wheel in d["native_audit"]["wheels"].values():
            assert len(wheel["slots"])==(9 if d["geometry"]=="slices9" else 1)
            np.testing.assert_allclose(wheel["contact_offsets_m"],.001,rtol=0,atol=1e-8)
            np.testing.assert_allclose(wheel["rest_offsets_m"],0.,rtol=0,atol=1e-8)
        raw=npz(folder/"physics_trajectory.npz");tire=npz(folder/"tire_trajectory.npz")
        values,trace=base.recorded_metrics(raw,"physx",reference)
        trace["contact_force"]=trace["floor_normal_force"].copy()
        t=tire["time_before_step_s"]
        np.testing.assert_allclose(t,trace["time_s"][:-1],rtol=0,atol=1e-12)
        assert len(t)==2792 and np.max(tire["native_patch_count"])<128
        fn=tire["preceding_normal_load_n"]
        np.testing.assert_allclose(fn,trace["contact_force"][:-1,:,2],rtol=1e-6,atol=1e-4)
        active=fn>0.;law=d["law"]
        force=tire["applied_tangent_force_w_n"];moment=tire["applied_tangent_moment_COM_w_nm"]
        if law=="native_coulomb":
            np.testing.assert_array_equal(force,np.zeros((2792,2,3)))
            np.testing.assert_array_equal(moment,np.zeros((2792,2,3)))
            cost=0.
        else:
            assert law in ("external_coulomb_regularized","combined_slip_magic_formula")
            q=tire["wheel_quaternion_wxyz"];w,x,y,z=np.moveaxis(q,-1,0)
            axle=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
            recomputed=tangent_wrench(tire["COM_velocity_w_m_s"][active],tire["angular_velocity_w_rad_s"][active],
                (tire["preceding_contact_position_w_m"]-tire["COM_position_w_m"])[active],
                axle[active],[0,0,1],fn[active],law=law)
            np.testing.assert_allclose(force[active],recomputed["force_w_n"],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(moment[active],recomputed["torque_about_COM_w_nm"],rtol=1e-10,atol=1e-10)
            np.testing.assert_array_equal(force[~active],0.)
            assert np.max(recomputed["contact_power_w"])<=1e-8
            # Runtime explicitly preserves the first buffer sampled before the
            # material substitution. It belongs to initialization, not a step
            # solved with both native and external tangential friction.
            assert np.max(np.linalg.norm(tire["native_friction_force_w_n"][1:],axis=-1))<1e-3
            cost=float(recomputed["contact_power_w"].sum()*.0025)
        screen=base.nominal_screen(values,original,protocol,"physx")
        core_mask=(t>=1.44-1e-9)&(t<=4.+1e-9)
        carried=(np.linalg.norm(force,axis=-1)>1e-3)&(trace["contact_force"][1:,:,2]<=5.)&core_mask[:,None]
        row=dict(id=name,geometry=d["geometry"],law=law,metrics=values,nominal_screen=screen,
            contact=dict(native_patch_count_max=int(tire["native_patch_count"].max()),
                core_applied_force_wheel_samples_with_next_Fz_le5N=int(carried.sum()),
                external_contact_energy_using_preceding_point_j=cost,external_wrenches_recomputed_exact=True,
                first_native_friction_buffer_precedes_material_change=(law!="native_coulomb"),
                first_native_friction_buffer_w_n=tire["native_friction_force_w_n"][0].tolist(),
                actual_normal_vector_includes_XY=True,individual_point_cones_identified=False),
            input_sha256={n:sha(folder/n) for n in ("wheel_contact_model_status.json","report.json","physics_trajectory.npz","tire_trajectory.npz")})
        rows.append(row);traces[name]=trace;raws[name]=raw;contacts[name]=tire
    assert rows[0]["input_sha256"]["physics_trajectory.npz"]==BASELINE
    for name in NAMES[1:]:
        for key in ("position","quaternion","com_velocity","angular_velocity","joint_pos","joint_vel"):
            assert np.array_equal(traces[name][key][0],traces[NAMES[0]][key][0]),"Paired initial state differs"
    return rows,traces,raws,contacts,original


def run(a):
    assert not a.output_dir.exists()
    reference=base.metrics.reference_data(a.reference)
    protocol=json.loads(a.protocol.read_text())
    rows,traces,_,_,original=load_cases(a.collection,reference,protocol)
    a.output_dir.mkdir(parents=True)
    summary=dict(kind="six_matched_offset_PhysX_K_policy_model_transfers",status="completed_analysis",rows=rows,
        actor_sha256=ACTOR,reference_sha256=sha(a.reference),nominal_protocol_sha256=sha(a.protocol),
        analyzer_sha256=sha(__file__),baseline_byte_identical=True,masses_COM_inertias_actuation_unchanged=True,
        same_initial_mechanical_state=True,contact_offset_m=.001,rest_offset_m=0.,
        external_load_coupling_delay_physical_steps=1,retrained_for_each_model=False,
        original_baseline_metrics=original,visual_review_complete=False,task_success_claimed=False)
    (a.output_dir/"analysis.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True,layout="constrained")
    for ax,law in zip(axes,("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"),strict=True):
        ax.plot(reference["position"][:,0],reference["position"][:,1],ls="--",color="gray",label="reference")
        for geometry,color in (("authored","#2271b2"),("slices9","#d55e00")):
            tr=traces[geometry+"__"+law];ax.plot(tr["position"][:,0],tr["position"][:,1],color=color,label=geometry)
        ax.set(title=law,xlabel="World X (m)",ylabel="World Y (m)",aspect="equal");ax.legend();ax.grid(alpha=.2)
    fig.suptitle("Same frozen K policy, PhysX mu=.7 | matched1mm contact offset, original mass/COM/inertia")
    fig.savefig(a.output_dir/"comparison.png",dpi=150);plt.close(fig)
    print(json.dumps([{ "case":r["id"],"core":r["metrics"]["windows"]["core"]["tracking"],
        "failed":r["nominal_screen"]["failed_checks"]} for r in rows]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--collection",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    import run_cem_ground_contact_v2 as assets
    p.add_argument("--reference",type=Path,default=assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--protocol",type=Path,default=ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    a=p.parse_args()
    run(a)
