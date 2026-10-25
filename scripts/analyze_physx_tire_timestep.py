"""Full-task400/800Hz sensitivity from recorded PhysX states; same50Hz actor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_smoothness as smooth
from analyze_cem_ground_contact_v2 import support_summary
from controller_trace import native_trace


CASES=(("native_coulomb","tron1_native_coulomb_v2","tron1_native_coulomb_800hz_v1"),
       ("external_coulomb_regularized","tron1_external_coulomb_v2","tron1_external_coulomb_800hz_v1"),
       ("combined_slip_magic_formula","tron1_magic_formula_v1","tron1_magic_formula_800hz_v1"))


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    assert not a.output_dir.exists()
    reference=smooth._load_reference(a.motion);rows=[];convergence=[]
    for law,coarse,fine in CASES:
        traces=[]
        for hz,name,collection in ((400,coarse,a.coarse),(800,fine,a.fine)):
            folder=collection/name;status=json.loads((folder/"tire_study_status.json").read_text())
            assert status["status"]=="completed_requires_analysis" and status["law"]==law
            assert status["normalized_actor_sha256"]=="53c5a0f59428c5a5ba03092a19c269d64cdca0c9e96360d5d8abac195999dc69"
            for file in ("physics_trajectory.npz","tire_trajectory.npz"):assert sha(folder/file)==status[file+"_sha256"]
            trace=native_trace(load(folder/"physics_trajectory.npz"),"policy_isaac")
            tire=load(folder/"tire_trajectory.npz");t=trace["time_s"];dt=1/hz
            assert trace["physics_dt_s"]==dt and len(t)==round(6.98/dt)+1
            core=smooth._tracking(t,trace["position"],trace["com_velocity"],trace["quaternion"],reference,(1.44,4.))
            support=support_summary(t,trace["contact_force"][:,:,2],(.02,6.96))
            loaded=(tire["preceding_normal_load_n"]>5)&(t[:-1,None]>=1.44-1e-9)&(t[:-1,None]<=4.+1e-9)
            force=tire["native_friction_force_w_n"] if law=="native_coulomb" else tire["applied_tangent_force_w_n"]
            magnitude=np.linalg.norm(force[...,:2],axis=-1)
            ratio=magnitude[loaded]/(.7*tire["preceding_normal_load_n"][loaded])
            row={"law":law,"physics_hz":hz,"core":core,
                "support_fraction":support["any_wheel_support_sample_fraction"],
                "longest_support_gap_s":support["longest_no_support_observed_s"],
                "loaded_core_used_muFn_utilization_quantiles_50_95_100":np.quantile(ratio,[.5,.95,1.]).tolist(),
                "recording_sha256":status["physics_trajectory.npz_sha256"]}
            if law!="native_coulomb":
                nextload=trace["contact_force"][1:,:,2];selected=loaded&(nextload>5)
                ratios=magnitude[selected]/(.7*nextload[selected])
                row["loaded_core_next_actual_muFn_utilization_quantiles_50_95_100"]=np.quantile(ratios,[.5,.95,1.]).tolist()
            rows.append(row);traces.append(trace)
        c,f=traces
        assert len(f["time_s"][::2])==len(c["time_s"])
        select=(c["time_s"]>=1.44-1e-9)&(c["time_s"]<=4.+1e-9)
        difference=c["position"][select,:2]-f["position"][::2][select,:2]
        convergence.append({"law":law,"core_cross_timestep_XY_RMSE_m":float(np.sqrt(np.mean(np.sum(difference**2,axis=-1)))),
            "note":"Closed-loop sensitivity includes contact integration, coupling and divergent actor actions; not a pure integration-error bound."})
    a.output_dir.mkdir(parents=True)
    result={"kind":"full_TRON1_PhysX_step_refinement_diagnostic","rows":rows,"cross_timestep":convergence,
        "training_hz":400,"actor_hz":50,"retrained_for_800hz":False,
        "two_resolutions_do_not_establish_asymptotic_convergence":True,"analyzer_sha256":sha(__file__),
        "no_simulator_or_inference_or_FK":True,"task_success_claimed":False}
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+chr(10))
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("coarse","fine","motion","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
