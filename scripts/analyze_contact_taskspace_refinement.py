"""Compare matched warm-K control with task-space refinement in both engines."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze_contact_warm_refinement as warm
import car_taskspace_tracking as targets
import collect_contact_taskspace_refinement as collection
from ppo_backend_400hz_contract import verify_recording

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("control_physx", "taskspace_physx", "control_mujoco", "taskspace_mujoco")
CONTROL = ROOT/"results/contact_warm_refinement_20261025_v1/completed_evidence_v1"
metrics, read, sha, npz = warm.metrics, warm.read, warm.sha, warm.npz


def load_cases(folder, reference):
    folder = Path(folder)
    source = collection.materialize((ROOT/"scripts/collect_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_taskspace_collected_numerical_admission", __file__=collection.__file__)
    exec(compile(source, "<taskspace_collected_numerical_admission>", "exec"), namespace)
    driver, status = namespace["verify_completed"](folder)
    protocol = read(folder/"source_snapshot_v1/config/contact_taskspace_refinement_400hz_v1.json")
    original = read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    assert protocol["nominal_acceptance_before_training"] == original["nominal_acceptance_before_training"]
    assert protocol["training_randomization"] == original["training_randomization"]
    control_rows, control_traces, control_proofs, historical = warm.load_cases(CONTROL, reference)
    old = read(CONTROL/"formal_K/manifest.json")
    new = read(folder/"formal_K/manifest.json")
    assert old["warm_start"] == new["warm_start"]
    assert old["contact_settings"] == new["contact_settings"]
    assert old["initial_policy_state_sha256"] == new["initial_policy_state_sha256"]
    assert read(CONTROL/"formal_K/runner_config.json") == read(folder/"formal_K/runner_config.json")
    assert old["environment"]["reward_terms"] == new["environment"]["reward_terms"]
    assert old["iterations"] == new["iterations"] == 200 and old["num_envs"] == new["num_envs"] == 512
    excluded = {"description", "task_space_tracking"}
    assert {k:v for k,v in old["task_profile"].items() if k not in excluded} == {k:v for k,v in new["task_profile"].items() if k not in excluded}
    assert new["task_profile"]["task_space_tracking"] == targets.settings()
    for key in ("live_startup", "live_final"):
        a = new["cohort"][key]["task_space_tracking"]
        assert a["status"] == "live_target_selection_verified" and a["configured"]["settings"] == targets.settings()
    proofs = dict(control=control_proofs["formal_K"], taskspace=driver["proofs"]["formal_K"])
    assert proofs["control"]["cumulative_transitions"] == proofs["taskspace"]["cumulative_transitions"] == 14745600
    rows, traces = [], {}
    original_metrics = {}
    for engine in ["physx", "mujoco"]:
        raw, _ = metrics.load_baseline(engine)
        original_metrics[engine], _ = metrics.recorded_metrics(raw, engine, reference)
        old_row = copy.deepcopy(next(r for r in control_rows if r["id"] == "K_"+engine))
        old_row.update(id="control_"+engine, variant="control", title="Warm K control | "+engine)
        rows.append(old_row)
        traces[old_row["id"]] = control_traces["K_"+engine]
    for case in status["cases"]:
        engine = case["backend"]
        f = folder/"nominal_v1"/case["id"]
        name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        verify_recording(f/name, "PhysX" if engine == "physx" else "native CPU MuJoCo")
        report = read(f/"report.json")
        meta = read(f/"warm_contact_evaluation_status.json")
        assert meta["kind"] == "taskspace_contact_refinement_native_evaluation"
        assert meta["normalized_actor_sha256"] == proofs["taskspace"]["actor_sha256"]
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
        old_trace = traces["control_"+engine]
        for key in ("position", "com_velocity", "angular_velocity", "joint_pos", "joint_vel"):
            np.testing.assert_allclose(trace[key][0], old_trace[key][0], atol=2e-6, rtol=0)
        assert abs(np.dot(trace["quaternion"][0], old_trace["quaternion"][0])) >= 1-1e-7
        row = dict(id="taskspace_"+engine, variant="taskspace", engine=engine,
                   title="Task-space K refinement | "+engine, actor_sha256=meta["normalized_actor_sha256"],
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
    result = dict(kind="matched_warm_K_taskspace_refinement_dual_engine", status="completed_analysis", rows=rows,
                  paired_training_proofs=proofs, historical_original_K=historical,
                  analyzer_sha256=sha(__file__), reference_sha256=sha(a.reference),
                  stage_transitions_per_arm=2457600, cumulative_transitions_per_resulting_policy=14745600,
                  same_K_initialization_optimizer_and_budget=True, all_original_gates_retained=True,
                  four_tracking_terms_changed=list(targets.CHANGED), contact_formula_and_all_weights_unchanged=True,
                  qualitative_review_complete=False, task_success_claimed=False)
    (a.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), layout="constrained")
    axes[0].plot(reference["position"][:, 0], reference["position"][:, 1], "--", color="gray", label="reference")
    for row in rows:
        t = traces[row["id"]]
        time = t["time_s"]
        axes[0].plot(t["position"][:, 0], t["position"][:, 1], label=row["title"])
        axes[1].plot(time, np.linalg.norm(t["com_velocity"][:, :2], axis=-1))
        axes[2].plot(time, t["floor_normal_force"][:, :, 2].max(-1))
        up = metrics.quaternion_rotate(t["quaternion"], np.array([0., 0., 1.]))
        axes[3].plot(time, np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))))
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=6)
    for ax, title in zip(axes, ["World XY [m]", "COM XY speed [m/s]", "Max wheel floor Fz [N]", "Body tilt [deg]"]):
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[2].axhline(5, color="gray", ls="--")
    fig.suptitle("Warm K control / free leg-link task-space tracking | equal 2.46M refinement | mu=.7")
    fig.savefig(a.output_dir/"comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "task_success_claimed": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("collection", "reference", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    run(p.parse_args())
