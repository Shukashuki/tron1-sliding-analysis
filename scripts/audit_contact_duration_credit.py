"""Recompute K support-cost aliasing and prospective matched S/T costs offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import ground_contact_reward_v2k as k
import ground_contact_reward_v2s as s
import ground_contact_reward_v2t as t

ROOT=Path(__file__).resolve().parents[1]
COHORTS=[("heading2D","contact_taskspace_refinement_20261025_v2","taskspace"),
         ("axis3D","contact_wheel_axis_20261025_v1","axis"),
         ("camber_reference","contact_camber_reference_20261025_v1","camber")]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    import mujoco
    assert not a.output_dir.exists()
    model=mujoco.MjModel.from_xml_path(str(a.model));mg=float(model.body_mass.sum())*9.81
    rows=[];records={}
    for label,folder,prefix in COHORTS:
        root=ROOT/"results"/folder;old=json.loads((root/"analysis/analysis.json").read_text())
        for engine in ["physx","mujoco"]:
            row=next(r for r in old["rows"] if r["id"]==prefix+"_"+engine)
            path=root/"completed_evidence_v1/nominal_v1"/("K_"+engine)/("physics_trajectory.npz" if engine=="physx" else "rollout.npz")
            assert sha(path)==row["recording_sha256"]
            with np.load(path,allow_pickle=False) as data:
                force=np.asarray(data["wheel_ground_contact_force_w_n"],dtype=float)
                time=data["time_s"].copy()
            if engine=="physx":force=force[:,0]
            np.testing.assert_allclose(time,np.arange(2793)*.0025,rtol=0,atol=1e-10)
            assert force.shape==(2793,2,3) and np.isfinite(force).all()
            windows=force[1:,:,2].reshape(349,8,2)
            # Freeze positive score at0:these are ungated cost comparisons,not
            # reconstructed actual reward totals or counterfactual rollouts.
            args=(windows,mg,0.,0.,0.,1.)
            costs={name:module.reward_numpy(*args)["support_gap_penalty"] for name,module in [("K",k),("S",s),("T",t)]}
            bad=windows.max(-1)<=5.;count=bad.sum(-1)
            assert np.array_equal(costs["K"]==1.,bad.any(-1))
            multi=count>=2;fixed=windows.copy()
            for i in np.flatnonzero(multi):fixed[i,np.flatnonzero(bad[i])[0],0]=mg*.5
            repaired={name:module.reward_numpy(fixed,mg,0.,0.,0.,1.)["support_gap_penalty"] for name,module in [("K",k),("S",s),("T",t)]}
            np.testing.assert_allclose((costs["S"]-repaired["S"])[multi],.125,atol=1e-14,rtol=0)
            np.testing.assert_array_equal(costs["K"][multi],repaired["K"][multi])
            np.testing.assert_array_equal(costs["T"][multi],repaired["T"][multi])
            phase=np.arange(349)*.02;core=(phase>=1.44-1e-9)&(phase<4.-1e-9)
            selected=count[core]
            key=label+"_"+engine
            values=dict(id=key,actor_sha256=row["actor_sha256"],recording_sha256=sha(path),
                core_policy_windows=int(core.sum()),core_windows_with_any_unloaded_substep=int(np.sum(selected>0)),
                core_multi_unloaded_windows_with_zero_K_cost_gain_for_repairing_one_substep=int(np.sum(selected>=2)),
                core_gap_count_histogram_0_to8=np.bincount(selected,minlength=9).tolist(),
                core_cost_mean={name:float(value[core].mean()) for name,value in costs.items()},
                S_raw_cost_gain_for_one_substep_repaired_with_another_gap_remaining=.125,
                K_T_raw_cost_gain_for_same_repair=0.,other14_terms_or_positive_G_score_in_this_audit=False,
                actual_trajectory_or_training_changed=False)
            rows.append(values);records[key+"__unloaded_counts"]=count
            for name,value in costs.items():records[key+"__"+name+"_cost"]=value
    a.output_dir.mkdir(parents=True)
    result=dict(kind="offline_support_cost_temporal_credit_audit",rows=rows,source_sha256=sha(__file__),
        model_sha256=sha(a.model),weight_n=mg,source_arithmetic_sha256={name:sha(module.__file__) for name,module in [("K",k),("S",s),("T",t)]},
        core_window_starts_s=[1.44,3.98],core_physical_samples_s=[1.4425,4.],physics_dt_s=.0025,policy_dt_s=.02,
        S_T_penalty_bounds=[0,2],all_uniform_force_windows_match_between_S_T=True,
        physics_steps=0,policy_updates=0,trained_S_or_T_policy_claimed=False,task_success_claimed=False)
    (a.output_dir/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(a.output_dir/"costs.npz",**records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout="constrained")
    for row in rows:
        hist=np.asarray(row["core_gap_count_histogram_0_to8"])/row["core_policy_windows"]
        axes[0].plot(np.arange(9),hist,marker='o',label=row["id"])
    gap=np.arange(9)
    axes[1].plot(gap,np.where(gap>0,1.,0.),label="K:max8")
    axes[1].plot(gap,np.where(gap>0,2.,0.),label="T:2max8")
    axes[1].plot(gap,np.where(gap>0,1.,0.)+gap/8,label="S:max8+mean8")
    axes[0].set(title="Actual core window gap-count distribution",xlabel="Unloaded physical substeps per8",ylabel="Fraction of128 core windows")
    axes[1].set(title="Ideal fully-supported/zero-load mixtures",xlabel="Zero-load substeps per8",ylabel="Ungated cost")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Cost identifiability audit |no new physics/training or relaxed support gate")
    fig.savefig(a.output_dir/"comparison.png",dpi=150);plt.close(fig)
    print(json.dumps(rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    run(p.parse_args())
