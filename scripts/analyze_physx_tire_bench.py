"""Analyze completed native PhysX single-wheel traces; no simulator is invoked."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    assert not a.output_dir.exists()
    summary=json.loads((a.collection/"single_wheel_fixture_sweep_v2/status.json").read_text())
    assert summary["status"]=="all_six_single_wheel_cases_completed_require_analysis"
    cases={};rows=[]
    for case in summary["cases"]:
        folder=a.collection/"single_wheel_fixture_sweep_v2"/case["id"]
        status=json.loads((folder/"status.json").read_text())
        assert status["status"]=="completed" and case["status"]=="completed_verified"
        assert sha(folder/"native_trace.npz")==case["native_trace_sha256"]==status["native_trace_sha256"]
        with np.load(folder/"native_trace.npz",allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        t=trace["time_s"];q=trace["root_state_w"];dt=status["physics_dt_s"]
        assert len(t)==round(2/dt)+1 and np.allclose(t,np.arange(len(t))*dt,atol=1e-12,rtol=0)
        assert np.isfinite(q).all() and np.allclose(q[0,7:10],[np.cos(np.deg2rad(20)),np.sin(np.deg2rad(20)),0],atol=1e-6,rtol=0)
        force=trace["inferred_total_tangent_force_xy_n"]
        load=trace["ground_normal_force_w_n"][:-1,2]
        loaded=load>1.
        ratio=np.full(len(load),np.nan);ratio[loaded]=np.linalg.norm(force[loaded],axis=-1)/(.7*load[loaded])
        below=np.flatnonzero(np.abs(q[:,8])<=.05)
        rows.append({"id":case["id"],"law":status["law"],"dt_s":dt,
            "final_xy_m":q[-1,:2].tolist(),"final_velocity_xy_m_s":q[-1,7:9].tolist(),
            "time_to_abs_lateral_velocity_le_0p05_m_s_s":float(t[below[0]]) if len(below) else None,
            "normal_force_mean_n":float(trace["ground_normal_force_w_n"][:,2].mean()),
            "support_fraction_Fz_gt_1N":float((trace["ground_normal_force_w_n"][:,2]>1).mean()),
            "inferred_tangent_over_previous_muFn_p95":float(np.nanquantile(ratio,.95)),
            "peak_abs_external_force_n":float(np.linalg.norm(trace["applied_tangent_force_w_n"],axis=-1).max()),
            "state_trace_sha256":sha(folder/"native_trace.npz")})
        cases[case["id"]]=trace
    convergence=[]
    for law in ("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"):
        coarse=cases[law+"_400hz"]["root_state_w"];fine=cases[law+"_800hz"]["root_state_w"][::2]
        convergence.append({"law":law,"XY_RMSE_400_vs_800_m":float(np.sqrt(np.mean(np.sum((coarse[:,:2]-fine[:,:2])**2,axis=-1)))),
            "lateral_velocity_RMSE_400_vs_800_m_s":float(np.sqrt(np.mean((coarse[:,8]-fine[:,8])**2)))})
    a.output_dir.mkdir(parents=True)
    result={"kind":"single_wheel_PhysX_law_and_dt_diagnostic","status":"six_completed_traces_analyzed",
        "cases":rows,"dt_convergence":convergence,"analyzer_sha256":sha(__file__),
        "interpretation_limits":["Axle X/Z angular locks and low wheel-only load differ from a whole robot.",
            "External force laws have no static-adhesion branch and use one-step-delayed measured normal load.",
            "A difference from native friction is not evidence of real-tire accuracy.",
            "The Magic Formula coefficients are a sensitivity profile, not identified CARLA/TRON1 tire parameters.",
            "Ratios using previous normal load can exceed one during contact transients; this is not a same-step cone violation measurement."],
        "car_or_TRON1_task_result":False,"task_success_claimed":False}
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+chr(10))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(20,4.7),dpi=150)
    colors={"native_coulomb":"tab:blue","external_coulomb_regularized":"tab:orange","combined_slip_magic_formula":"tab:green"}
    for row in rows:
        trace=cases[row["id"]];q=trace["root_state_w"];t=trace["time_s"]
        kw={"color":colors[row["law"]],"ls":"-" if row["dt_s"]==.0025 else "--","label":row["id"]}
        axes[0].plot(q[:,0],q[:,1],**kw)
        axes[1].plot(t,q[:,8],**kw)
        axes[2].plot(t[:-1],trace["inferred_total_tangent_force_xy_n"][:,1],**kw)
        axes[3].plot(t,trace["ground_normal_force_w_n"][:,2],**kw)
    axes[0].set(title="Free translation under axle fixture",xlabel="X (m)",ylabel="Y (m)",aspect="equal")
    axes[1].set(title="Lateral velocity decay",xlabel="Time (s)",ylabel="Vy (m/s)")
    axes[2].set(title="Tangent force from m deltaVy / dt",xlabel="Time (s)",ylabel="Fy (N)")
    axes[3].set(title="Native normal contact",xlabel="Time (s)",ylabel="Fn (N)")
    axes[1].legend(fontsize=6)
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle("PhysX single-wheel diagnostic | solid400Hz / dashed800Hz | not a full drift task")
    fig.tight_layout();fig.savefig(a.output_dir/"comparison.png");plt.close(fig)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--collection",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    run(p.parse_args())
