"""Offline total wheel-normal-load audit; no new training reward or gate.

Investigates a possible physical-load regularizer without claiming it fixes
contact. Uses actual native forces and live audited robot mass, not root-vz
as a proxy for whole-body COM acceleration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_contact_taskspace_margin as analysis


def run(root):
    root = Path(root)
    out = root/"normal_load_diagnostic"
    assert not out.exists()
    reference = analysis.metrics.metrics.reference_data(root/"reference.npz")
    rows, traces, _, _ = analysis.load_cases(root/"completed_evidence_v1", reference)
    control = analysis.read(analysis.CONTROL/"formal_K/manifest.json")
    candidate = analysis.read(root/"completed_evidence_v1/formal_M/manifest.json")
    audits = [m["cohort"]["live_final"]["contact"] for m in (control, candidate)]
    assert audits[0]["mass_kg_range"] == audits[1]["mass_kg_range"]
    assert audits[0]["load_margin_gravity_w_m_s2"] == audits[1]["load_margin_gravity_w_m_s2"] == [0.,0.,-9.81]
    mass_range = audits[0]["mass_kg_range"]
    assert mass_range[0] == mass_range[1]
    mg = mass_range[0]*9.81
    results = {}
    for row in rows:
        t = traces[row["id"]]
        force = t["floor_normal_force"][:,:,2]
        assert np.isfinite(force).all() and force.min() >= -1e-4
        total = np.maximum(force,0.).sum(-1)/mg
        data = {}
        for label, (lo,hi) in {"core":(1.44,4.), "full_contact":(.02,6.96)}.items():
            select = (t["time_s"] >= lo-1e-12) & (t["time_s"] <= hi+1e-12)
            q = total[select]
            e = np.abs(q-1.)
            huber = np.where(e <= 1., .5*e*e, e-.5)
            support = (force[select].max(-1) > 5.).mean()
            expected = row["metrics"]["any_wheel_ground_support"][label]["any_wheel_support_sample_fraction"]
            assert float(support) == expected
            data[label] = dict(native_samples=int(select.sum()), mass_weight_n=mg,
                anywheel_support_fraction=float(support), mean_total_normal_over_mg=float(q.mean()),
                rms_total_normal_minus_mg_over_mg=float(np.sqrt(np.mean((q-1.)**2))),
                peak_total_normal_over_mg=float(q.max()), fraction_total_normal_above3mg=float((q>3.).mean()),
                total_normal_over_mg_quantiles={str(p):float(np.quantile(q,p)) for p in [.5,.9,.95,.99]},
                offline_unit_Huber_total_load_residual_mean=float(huber.mean()))
        results[row["id"]] = data
    out.mkdir()
    result = dict(kind="native_total_wheel_normal_load_offline_diagnostic", cases=results,
        source_sha256=analysis.sha(__file__), analysis_sha256=analysis.sha(root/"analysis/analysis.json"),
        mass_kg=mass_range[0], gravity_m_s2=9.81,
        Huber_definition="e=abs(sum_wheel_Fz/(mg)-1); h=.5e^2 if e<=1 else e-.5; mean over native samples",
        new_training_or_reward_or_gate_installed=False, absolute_height_target=False,
        root_acceleration_equated_to_whole_body_COM_acceleration=False,
        interpretation="A prospective load-consistency statistic only. Native outcomes and residuals do not establish causal repair or task success.",
        task_success_claimed=False)
    (out/"diagnosis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,4,figsize=(22,4.5),sharex=True,sharey=True,layout="constrained")
    maximum = max(np.maximum(traces[k]["floor_normal_force"][:,:,2],0.).sum(-1).max()/mg for k in analysis.ORDER)
    for ax,row in zip(axes,rows):
        t = traces[row["id"]]
        ax.plot(t["time_s"],np.maximum(t["floor_normal_force"][:,:,2],0.).sum(-1)/mg,lw=.65)
        ax.axhline(1.,color="black",ls="--",lw=.8,label="weight mg")
        ax.set(xlim=(1.44,4.),ylim=(0,maximum*1.03),title=row["title"],xlabel="Native time (s)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Sum of wheel-ground normal forces / mg")
    axes[0].legend(fontsize=7)
    fig.suptitle("Actual total normal load: same mass, scales and core interval; no new reward installed")
    fig.savefig(out/"normal_load.png",dpi=150);plt.close(fig)
    print(json.dumps({"output":str(out),"task_success_claimed":False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True)
    run(p.parse_args().result)
