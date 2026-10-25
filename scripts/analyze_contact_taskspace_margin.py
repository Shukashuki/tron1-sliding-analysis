"""Matched task-space K/M comparison with unchanged complete task gates."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze_contact_taskspace_refinement as control
import car_taskspace_tracking as targets
import collect_contact_taskspace_margin as collection
from ppo_backend_400hz_contract import verify_recording

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("control_physx", "margin4_physx", "control_mujoco", "margin4_mujoco")
CONTROL = ROOT/"results/contact_taskspace_refinement_20261025_v2/completed_evidence_v1"
metrics, read, sha, npz = control.metrics, control.read, control.sha, control.npz


def load_cases(folder, reference):
    folder = Path(folder)
    source = collection.materialize((ROOT/"scripts/collect_contact_warm_refinement.py").read_text())
    ns = dict(__name__="_margin4_collected_admission", __file__=collection.__file__)
    exec(compile(source, "<margin4_collected_admission>", "exec"), ns)
    driver, status = ns["verify_completed"](folder)
    protocol = read(folder/"source_snapshot_v1/config/contact_taskspace_margin_400hz_v1.json")
    original = read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    for key in ("nominal_acceptance_before_training", "training_randomization"):
        assert protocol[key] == original[key]
    old_rows, old_traces, old_proofs, historical = control.load_cases(CONTROL, reference)
    old = read(CONTROL/"formal_K/manifest.json")
    new = read(folder/"formal_M/manifest.json")
    assert old["warm_start"] == new["warm_start"]
    assert old["initial_policy_state_sha256"] == new["initial_policy_state_sha256"]
    assert read(CONTROL/"formal_K/runner_config.json") == read(folder/"formal_M/runner_config.json")
    assert old["environment"]["reward_terms"] == new["environment"]["reward_terms"]
    assert old["iterations"] == new["iterations"] == 200 and old["num_envs"] == new["num_envs"] == 512
    before, after = old["contact_settings"], new["contact_settings"]
    assert {k for k in before if before[k] != after[k]} == {"candidate", "support_gap_weight"}
    assert before["support_gap_weight"] == 1. and after["support_gap_weight"] == 4.
    excluded = {"description", "ground_contact_reward_v2"}
    assert {k:v for k,v in old["task_profile"].items() if k not in excluded} == {k:v for k,v in new["task_profile"].items() if k not in excluded}
    assert new["task_profile"]["task_space_tracking"] == targets.settings()
    for key in ("live_startup", "live_final"):
        a = new["cohort"][key]["task_space_tracking"]
        b = old["cohort"][key]["task_space_tracking"]
        assert a["status"] == "live_target_selection_verified"
        assert a["configured"]["settings"] == b["configured"]["settings"] == targets.settings()
        assert a["active_tracking_params"] == b["active_tracking_params"]
        assert a["active_term_weights"] == b["active_term_weights"]
    for key in ("asset", "model", "motion"):
        assert old["inputs"][key]["sha256"] == new["inputs"][key]["sha256"]
    def target_hash(m):
        values = [v for k,v in m["cohort"]["source_sha256"].items() if Path(k).name == "car_taskspace_tracking.py"]
        assert len(values) == 1
        return values[0]
    assert target_hash(old) == target_hash(new) == sha(targets.__file__)
    proofs = dict(control=old_proofs["taskspace"], margin4=driver["proofs"]["formal_M"])
    assert proofs["control"]["cumulative_transitions"] == proofs["margin4"]["cumulative_transitions"] == 14745600
    rows, traces, original_metrics = [], {}, {}
    for engine in ("physx", "mujoco"):
        raw, _ = metrics.load_baseline(engine)
        original_metrics[engine], _ = metrics.recorded_metrics(raw, engine, reference)
        row = copy.deepcopy(next(r for r in old_rows if r["id"] == "taskspace_"+engine))
        row.update(id="control_"+engine, variant="control", title="Task-space K coefficient1 | "+engine)
        rows.append(row)
        traces[row["id"]] = old_traces["taskspace_"+engine]
    for case in status["cases"]:
        engine = case["backend"]
        f = folder/"nominal_v1"/case["id"]
        name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        verify_recording(f/name, "PhysX" if engine == "physx" else "native CPU MuJoCo")
        report = read(f/"report.json")
        meta = read(f/"warm_contact_evaluation_status.json")
        assert meta["kind"] == "taskspace_margin4_native_evaluation"
        assert meta["normalized_actor_sha256"] == proofs["margin4"]["actor_sha256"]
        if engine == "physx":
            assert report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
            assert report["terminal_states_captured_before_auto_reset"]
            assert not report["domain_randomization_enabled"] and not report["observation_noise_enabled"]
            assert report["runtime_device_audit"]["status"] == "verified"
        else:
            assert report["status"] == "completed" and report["completed_full_reference"]
            assert report["hidden_resets"] == 0 and report["root_state_writes"] == 1
            assert report["physics_steps"] == 2792 and report["policy_steps"] == 349
        values, trace = metrics.recorded_metrics(npz(f/name), engine, reference)
        trace["contact_force"] = trace["floor_normal_force"].copy()
        other = traces["control_"+engine]
        for key in ("position", "com_velocity", "angular_velocity", "joint_pos", "joint_vel"):
            np.testing.assert_allclose(trace[key][0], other[key][0], atol=2e-6, rtol=0)
        assert abs(np.dot(trace["quaternion"][0], other["quaternion"][0])) >= 1-1e-7
        row = dict(id="margin4_"+engine, variant="margin4", engine=engine,
                   title="Task-space M coefficient4 | "+engine, actor_sha256=meta["normalized_actor_sha256"],
                   recording_sha256=sha(f/name), metrics=values,
                   nominal_screen=metrics.nominal_screen(values, original_metrics[engine], protocol, engine))
        rows.append(row)
        traces[row["id"]] = trace
    rows.sort(key=lambda r: ORDER.index(r["id"]))
    assert [r["id"] for r in rows] == list(ORDER)
    return rows, traces, proofs, historical


def run(a):
    assert not a.output_dir.exists()
    reference = metrics.metrics.reference_data(a.reference)
    rows, traces, proofs, historical = load_cases(a.collection, reference)
    a.output_dir.mkdir(parents=True)
    result = dict(kind="matched_taskspace_K_M_margin_refinement_dual_engine", status="completed_analysis", rows=rows,
                  paired_training_proofs=proofs, historical_original_K=historical,
                  analyzer_sha256=sha(__file__), reference_sha256=sha(a.reference),
                  stage_transitions_per_arm=2457600, cumulative_transitions_per_resulting_policy=14745600,
                  same_K_initialization_optimizer_and_budget=True, all_original_gates_retained=True,
                  same_four_taskspace_tracking_definitions=True, all_reward_weights_unchanged=True,
                  only_contact_setting_differences=["candidate", "support_gap_weight"],
                  qualitative_review_complete=False, task_success_claimed=False)
    (a.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), layout="constrained")
    axes[0].plot(reference["position"][:,0], reference["position"][:,1], "--", color="gray", label="reference")
    for row in rows:
        t = traces[row["id"]]
        axes[0].plot(t["position"][:,0], t["position"][:,1], label=row["title"])
        axes[1].plot(t["time_s"], np.linalg.norm(t["com_velocity"][:,:2], axis=-1))
        axes[2].plot(t["time_s"], t["floor_normal_force"][:,:,2].max(-1))
        up = metrics.quaternion_rotate(t["quaternion"], np.array([0.,0.,1.]))
        axes[3].plot(t["time_s"], np.degrees(np.arccos(np.clip(up[:,2], -1, 1))))
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=6)
    for ax, title in zip(axes, ["World XY [m]", "COM XY speed [m/s]", "Max wheel floor Fz [N]", "Body tilt [deg]"]):
        ax.set_title(title);ax.grid(alpha=.2)
    axes[2].axhline(5, color="gray", ls="--")
    fig.suptitle("Same task-space targets | contact-margin coefficient1 vs4 | equal 2.46M refinement | mu=.7")
    fig.savefig(a.output_dir/"comparison.png", dpi=150);plt.close(fig)
    print(json.dumps({"rows":len(rows), "task_success_claimed":False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("collection", "reference", "output-dir"):
        p.add_argument("--"+n, type=Path, required=True)
    run(p.parse_args())
