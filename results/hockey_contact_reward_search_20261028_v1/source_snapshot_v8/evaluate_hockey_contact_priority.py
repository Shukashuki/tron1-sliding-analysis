"""Require continuous bilateral geometric contact before comparing task performance.

Consumes the independently aligned v3 contact evidence. Force distribution is
unconstrained: unequal loads do not fail this gate. No reward, simulation or
historical qualification is modified.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, segments, sha


def bilateral_gate(start, end, touching, unknown, window=(0., 2.76), dt=.0025):
    start, end = np.asarray(start), np.asarray(end)
    touching, unknown = np.asarray(touching, bool), np.asarray(unknown, bool)
    if start.ndim != 1 or end.shape != start.shape or touching.shape != (len(start), 2) or unknown.shape != touching.shape:
        raise ValueError("Expected aligned timestamps and two wheel contact states")
    if np.any(touching & unknown):
        raise ValueError("A wheel cannot be touching and unknown simultaneously")
    lo, hi = window
    if not 0 < dt < hi-lo or not np.isfinite([lo, hi, dt]).all():
        raise ValueError("Invalid task window or physics timestep")
    n = round((hi-lo)/dt)
    expected = lo + np.arange(n)*dt
    mask = (start >= lo-1e-9) & (start < hi-1e-9)
    terminal = np.flatnonzero(np.isclose(start, hi, atol=1e-9, rtol=0.))
    complete = (abs(n*dt-(hi-lo)) < 1e-9 and mask.sum() == n and len(terminal) == 1
                and np.isfinite(start).all() and np.isfinite(end).all()
                and np.allclose(start[mask], expected, atol=1e-9, rtol=0.)
                and np.allclose(end[mask], expected+dt, atol=1e-9, rtol=0.)
                and np.allclose(end[terminal], hi+dt, atol=1e-9, rtol=0.))
    result = dict(record_complete=bool(complete), gate_status="unverified",
                  eligible_for_task_comparison=False, secondary_metrics_scope="diagnostic_only",
                  window_s=[lo, hi], terminal_contact_checked=True,
                  primary_gate_only=True, full_task_success_claimed=False)
    if not complete:
        result["reason"] = "Missing, duplicate, or misaligned physical samples; continuity cannot be certified"
        return result
    checked = mask.copy(); checked[terminal] = True
    both = touching.all(-1)
    known_off = (~touching & ~unknown).any(-1)
    unverified = ~both & ~known_off

    def gaps(off):
        events = segments(end, mask & off, interval_samples=True)
        return dict(total_s=float((mask & off).sum()*dt), count=len(events),
                    longest_s=max((e["duration_s"] for e in events), default=0.), events=events)

    status = "fail" if known_off[checked].any() else ("unverified" if unverified[checked].any() else "pass")
    bad = np.flatnonzero(checked & ~both)
    result.update(gate_status=status, eligible_for_task_comparison=status == "pass",
                  secondary_metrics_scope="eligible_for_subsequent_task_gates" if status == "pass" else "diagnostic_only",
                  reason={"pass":"Both wheels touch at every recorded physical sample",
                          "fail":"At least one wheel has a confirmed contact interruption",
                          "unverified":"No confirmed interruption but contact evidence is unknown"}[status],
                  checked_intervals=n, both_touch_fraction=float(both[mask].mean()),
                  first_nonbilateral_contact_s=float(start[bad[0]]) if len(bad) else None,
                  bilateral_interruption=gaps(known_off), unknown_only=gaps(unverified),
                  terminal_touching=touching[terminal[0]].tolist(),
                  terminal_unknown=unknown[terminal[0]].tolist(), wheels=[])
    for side in (0, 1):
        result["wheels"].append(dict(side=("left", "right")[side],
            contact_fraction=float(touching[mask, side].mean()),
            interruption=gaps(~touching[:, side] & ~unknown[:, side]),
            unknown=gaps(unknown[:, side])))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/hockey_contact_priority_v4.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cfg = json.loads(a.config.read_text())
    if cfg["required_contact"] != "both_wheels_at_every_physical_sample" or cfg["initial_grace_s"] != 0 or cfg["allowed_contact_interruption_s"] != 0 or not cfg["check_terminal_contact"]:
        raise ValueError("This evaluator implements strict bilateral continuity including start and end")
    source = json.loads((a.audit/"report.json").read_text())
    assert source["kind"] == "hockey_contact_timing_audit_v3"
    assert source["contact_tolerance_m"] == cfg["contact_separation_tolerance_m"]
    assert sha(a.audit/"aligned_signals.npz") == source["aligned_signals_sha256"]
    archive = load(a.audit/"aligned_signals.npz")
    result = dict(kind=cfg["kind"], config=cfg, config_sha256=sha(a.config),
                  script_sha256=sha(__file__), audit_sha256=sha(a.audit/"report.json"),
                  aligned_signals_sha256=source["aligned_signals_sha256"],
                  scope="Four nominal episodes only, seed42, mu=0.7; primary contact gate, no secondary ranking for failed cases",
                  physics_steps=0, training_changed=False, perturbed_success_rates_recomputed=False,
                  sampling_limit="400 Hz contact evidence cannot certify continuity between samples",
                  cells=[])
    signals = []
    for cell in source["cells"]:
        name = cell["cell"]
        s = {k.split("__", 1)[1]: v for k,v in archive.items() if k.startswith(name+"__")}
        gate = bilateral_gate(s["time_start_s"], s["time_end_s"], s["touching"], s["contact_unknown"],
                              cfg["evaluation_window_s"], cfg["physics_dt_s"])
        result["cells"].append(dict(cell=name, source_provenance=cell["provenance"], **gate))
        signals.append(s)
    result["eligible_cells"] = [c["cell"] for c in result["cells"] if c["eligible_for_task_comparison"]]
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    rows = [dict(cell=c["cell"], gate=c["gate_status"], both_touch_percent=100*c.get("both_touch_fraction", 0.),
                 total_interruption_ms=1000*c["bilateral_interruption"]["total_s"] if c["record_complete"] else None,
                 longest_interruption_ms=1000*c["bilateral_interruption"]["longest_s"] if c["record_complete"] else None,
                 eligible=c["eligible_for_task_comparison"]) for c in result["cells"]]
    with (a.output/"summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1,4,figsize=(20,3.6),sharex=True,sharey=True,constrained_layout=True)
    labels = ["Earlier tracking 1x", "Hockey PPO 1x", "Hockey PPO 1.5x", "Hockey PPO 2x"]
    for ax,c,s,label in zip(axes,result["cells"],signals,labels):
        mask=(s["time_start_s"]>=-1e-9)&(s["time_end_s"]<=cfg["evaluation_window_s"][1]+1e-9)
        touch=s["touching"][mask]; unknown=s["contact_unknown"][mask]
        states=np.where(touch,1,np.where(unknown,2,0)).T
        both=np.where(touch.all(-1),1,np.where((~touch&~unknown).any(-1),0,2))
        ax.imshow(np.vstack((states,both)),aspect="auto",origin="upper",interpolation="nearest",
                  extent=[*cfg["evaluation_window_s"],2.5,-.5],vmin=0,vmax=2,
                  cmap=ListedColormap(["#d65353","#3e9b63","#d1a63a"]))
        ax.set(yticks=[0,1,2],yticklabels=["Left","Right","Both"],xlabel="Time [s]",
               title=f"{label} | {c['gate_status'].upper()}\nBoth touch: {100*c.get('both_touch_fraction',0):.2f}%")
        ax.axvline(.4,color="black",ls=":",lw=.7);ax.axvline(1.8,color="black",ls=":",lw=.7)
        if c["record_complete"]:
            ax.text(.5,-.34,f"Interruption: total {1000*c['bilateral_interruption']['total_s']:.1f} ms\nLongest {1000*c['bilateral_interruption']['longest_s']:.1f} ms",transform=ax.transAxes,ha="center")
    fig.suptitle("Primary gate: both wheels continuously touching | Green: touch; red: missing contact; amber: unknown\nNo force threshold, no equal-load constraint, no startup grace | Contact failures are diagnostic only")
    fig.savefig(a.output/"bilateral_contact_priority_4x1.png",dpi=160,bbox_inches="tight");plt.close(fig)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
