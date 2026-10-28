"""Diagnose recorded wheel-support interruptions without changing task gates.

Reconstruct native forces, verify archived signals and report every nominal
gap. Load-threshold sensitivity is descriptive, not a replacement evaluator.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, segments, sha, reconstruct_contacts, derive_signals


def read_support_case(cell):
    directory = cell / "eval-suite/nominal"
    run_path = directory / "report.json"
    run = json.loads(run_path.read_text())
    suite = json.loads((cell / "analyze/report.json").read_text())
    entry = next(row for row in suite["cases"] if row["id"] == "nominal")
    qpath = cell / "analyze/nominal.json"
    assert sha(qpath) == entry["report_sha256"]
    qualification = json.loads(qpath.read_text())
    assert sha(run_path) == qualification["run_report_sha256"]
    for name, digest in qualification["input_sha256"].items():
        assert sha(directory / name) == digest
    material = run["material_audit"]
    assert material["live_physx_materials_verified"] and material["ground_material_usd_verified"]
    assert np.allclose(material["wheel_mu_range"], [.7, .7], atol=1e-6, rtol=0)
    assert all(g["friction"] == 1 and g["combine"] == "multiply" for g in material["ground_colliders"])
    trace = load(directory / "final_rollout.npz")
    contact, audit = reconstruct_contacts(trace, load(directory / "native_contact_buffers.npz"), .7)
    signals = derive_signals(trace, contact)
    archived = load(cell / "analyze/nominal.signals.npz")
    # This analysis uses forces, speed and tilt; beta is not part of its evidence.
    fields = ("time_s", "normal_fz", "friction_force", "native_sample_valid", "body_speed_m_s", "root_tilt_deg")
    audit["archived_support_fields_verified"] = list(fields)
    for field in fields:
        np.testing.assert_allclose(signals[field], archived[field], atol=1e-10, rtol=1e-10)
    return dict(run=run, qualification=qualification, signals=signals, native_audit=audit,
                provenance=dict(run_report_sha256=sha(run_path), qualification_sha256=sha(qpath),
                                recorded_artifacts=qualification["input_sha256"]))


def support_state(force):
    return "+".join(side for side, f in zip(("L", "R"), force) if f > 5) or "none"


def analyze(cell, reference_dir):
    audited = read_support_case(cell)
    s = audited["signals"]
    trace = load(cell / "eval-suite/nominal/final_rollout.npz")
    reference = load(reference_dir / "motion.npz")
    command = json.loads((reference_dir / "report.json").read_text())
    assert sha(reference_dir / "motion.npz") == audited["run"]["motion_sha256"]
    t = s["time_s"]
    dt = float(np.median(np.diff(t)))
    end = (len(reference["joint_pos"]) - 1) / float(reference["fps"])
    valid = s["native_sample_valid"] & (t - dt >= .02 - 1e-9) & (t <= end + 1e-9)
    force = s["normal_fz"]
    unsupported = valid & ~(force > 5).any(-1)
    events = segments(t, unsupported, interval_samples=True)
    result_events = []
    for event in events:
        i, j = event["first_index"], event["last_index"]
        before, after = i - 1, j + 1
        gap_force = force[i:j + 1]
        has_after = after < len(t) and valid[after]
        post = valid & (t > t[j]) & (t <= t[j] + .02 + 1e-9)
        result_events.append(dict(event,
            phase=("approach" if t[j] <= command["brake_start_s"] else
                   "braking" if t[i] - dt >= command["brake_start_s"] - 1e-9 and
                   t[j] <= command["brake_end_s"] + 1e-9 else "other_or_cross_phase"),
            support_before=support_state(force[before]),
            support_after=support_state(force[after]) if has_after else None,
            force_before_N=force[before].tolist(),
            force_after_N=force[after].tolist() if has_after else None,
            force_max_during_N=gap_force.max(0).tolist(),
            zero_force_duration_s=float((gap_force.max(-1) <= 1e-6).sum() * dt),
            combined_capacity_max_during_N=float((.7 * gap_force.sum(-1)).max()),
            maximum_wheel_force_next_20ms_N=force[post].max(0).tolist() if post.any() else None,
            speed_mean_m_s=float(s["body_speed_m_s"][i:j+1].mean()),
            base_com_vertical_velocity_first_last_m_s=trace["root_com_vel"][[i, j], 2].tolist(),
            tilt_max_deg=float(s["root_tilt_deg"][i:j+1].max())))
    q = audited["qualification"]["qualification"]
    assert np.isclose(q["support_fraction"], 1 - unsupported.sum() / valid.sum())
    assert np.isclose(q["longest_support_gap_s"], max((e["duration_s"] for e in events), default=0))
    sensitivity = []
    for threshold in (1e-6, .1, 1., 2., 5., 10.):
        gap = valid & ~(force > threshold).any(-1)
        spans = segments(t, gap, interval_samples=True)
        sensitivity.append(dict(threshold_N=threshold, total_gap_s=float(gap.sum() * dt),
            longest_gap_s=max((e["duration_s"] for e in spans), default=0), continuous_support=not gap.any()))
    suite = json.loads((cell / "analyze/report.json").read_text())
    perturbations = []
    for entry in suite["cases"]:
        if entry["kind"] != "perturbed":
            continue
        path = cell / "analyze" / (entry["id"] + ".json")
        assert sha(path) == entry["report_sha256"]
        p = json.loads(path.read_text())["qualification"]
        perturbations.append(dict(id=entry["id"], support_fraction=p["support_fraction"],
            longest_support_gap_s=p["longest_support_gap_s"], failed_gates=p["failed_gates"]))
    assert len(perturbations) == 32
    return dict(cell=cell.name, provenance=audited["provenance"], native_audit=audited["native_audit"],
        reference_sha256=sha(reference_dir / "motion.npz"),
        support_fraction=q["support_fraction"], gap_count=len(events),
        total_gap_s=float(unsupported.sum() * dt), longest_gap_s=q["longest_support_gap_s"],
        zero_force_duration_s=float((unsupported & (force.max(-1) <= 1e-6)).sum() * dt),
        xy_rmse_m=q["xy_RMSE_m"], first_45deg_tilt_s=audited["qualification"]["first_45deg_tilt_s"],
        failed_gates=q["failed_gates"], events=result_events, threshold_sensitivity=sensitivity,
        perturbed_cases=perturbations,
        perturbed_support_failures=sum("anywheel_continuously_supported" in p["failed_gates"] for p in perturbations),
        perturbed_longest_gap_min_median_max_s=np.quantile([p["longest_support_gap_s"] for p in perturbations], [0, .5, 1]).tolist()), s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ["beyondmimic_s100_seed42", "hockey_s100_seed42", "hockey_s150_seed42", "hockey_s200_seed42"]
    cells, data = [], []
    for name in names:
        result, signals = analyze(args.runs / name, args.references / name.split("_")[1])
        cells.append(result)
        data.append(signals)
    report = dict(kind="hockey_native_support_failure_diagnosis_v1", source_sha256=sha(__file__),
        physics_hz=400, policy_hz=50, friction=.7,
        support_definition="At least one wheel has summed floor-normal force >5 N at every full 2.5 ms interval after 20 ms, through reference end.",
        original_success_gates_changed=False, physics_rerun=False,
        limitations=["Wheel-floor forces only; no all-body external force balance or geometric separation certificate.",
            "Forces represent each completed 2.5 ms step; first recontact load is not an instantaneous continuous-time peak.",
            "Zero wheel load eliminates wheel Coulomb friction capacity; no utilization ratio is defined at zero load.",
            "Recorded ordering identifies support handoff gaps, not a causal proof about individual actions.",
            "Nominal force sensitivity is not recomputed for the 32 perturbed episodes; those use frozen qualification reports."],
        cells=cells)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = ["cell", "start_s", "end_s", "duration_s", "phase", "support_before", "support_after",
               "zero_force_duration_s", "combined_capacity_max_during_N", "speed_mean_m_s", "tilt_max_deg"]
    with (args.output / "gap_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            for event in cell["events"]:
                writer.writerow({key: cell["cell"] if key == "cell" else event[key] for key in columns})
    np.savez_compressed(args.output / "plot_signals.npz", **{
        name + "__" + key: s[key] for name, s in zip(names, data)
        for key in ("time_s", "normal_fz", "root_tilt_deg", "native_sample_valid")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(23, 5), sharex=True, sharey=True, constrained_layout=True)
    titles = ["Earlier tracking, 1x", "Hockey PPO, 1x", "Hockey PPO, 1.5x", "Hockey PPO, 2x"]
    ymax = max(float(s["normal_fz"][(s["time_s"] >= .02) & (s["time_s"] <= 2.76)].max()) for s in data) * 1.15
    for ax, c, s, title in zip(axes, cells, data, titles):
        ax.axvspan(.4, 1.8, color="#e9e9e9", alpha=.6, label="Braking phase")
        for k, color in enumerate(("#2874b2", "#d77325")):
            ax.plot(s["time_s"], s["normal_fz"][:, k], color=color, lw=.85, label=("Left", "Right")[k])
        for n, event in enumerate(c["events"]):
            ax.axvspan(event["start_s"], event["end_s"], color="crimson", alpha=.30,
                       label="No wheel >5 N" if n == 0 else None)
        ax.axhline(5, color="black", lw=.8, ls="--", label="5 N threshold")
        ax.set(xlim=(.02, 2.76), ylim=(-1, ymax), yscale="symlog", xlabel="Time [s]",
               title=f"{title}\nGaps: {c['gap_count']}; total {1000*c['total_gap_s']:.1f} ms; max {1000*c['longest_gap_s']:.1f} ms")
        ax.grid(alpha=.18)
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("Native wheel floor-normal force [N], symlog scale")
    fig.suptitle("Recorded PhysX support: seed 42, mu=0.7 | Same axes; raw 400 Hz forces | One supporting wheel is sufficient")
    fig.savefig(args.output / "support_force_comparison_4x1.png", dpi=160)
    plt.close(fig)
    print(json.dumps([{key: c[key] for key in ("cell", "gap_count", "total_gap_s", "zero_force_duration_s", "perturbed_longest_gap_min_median_max_s")} for c in cells]))


if __name__ == "__main__":
    main()
