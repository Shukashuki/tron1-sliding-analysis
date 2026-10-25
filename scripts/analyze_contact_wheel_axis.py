"""Matched warm refinement: 2D wheel-plane heading versus 3D wheel axes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze_contact_taskspace_refinement as control
import car_taskspace_tracking as old_targets
import car_wheel_axis_tracking as targets
import collect_contact_wheel_axis as collection
from ppo_backend_400hz_contract import verify_recording

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("control_physx", "axis_physx", "control_mujoco", "axis_mujoco")
CONTROL = ROOT/"results/contact_taskspace_refinement_20261025_v2/completed_evidence_v1"
metrics, read, sha, npz = control.metrics, control.read, control.sha, control.npz


def load_cases(folder, reference):
    folder = Path(folder)
    driver, status = collection.verify_completed(folder)
    protocol = read(folder/"source_snapshot_v1/config/contact_wheel_axis_400hz_v1.json")
    original = read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    for key in ("nominal_acceptance_before_training", "training_randomization"):
        assert protocol[key] == original[key]
    old_rows, old_traces, old_proofs, historical = control.load_cases(CONTROL, reference)
    before = read(CONTROL/"formal_K/manifest.json")
    after = read(folder/"formal_K/manifest.json")
    for key in ("warm_start", "initial_policy_state_sha256", "contact_settings"):
        assert before[key] == after[key]
    assert read(CONTROL/"formal_K/runner_config.json") == read(folder/"formal_K/runner_config.json")
    assert before["environment"]["reward_terms"] == after["environment"]["reward_terms"]
    assert before["iterations"] == after["iterations"] == 200
    assert before["num_envs"] == after["num_envs"] == 512
    excluded = {"description", "task_space_tracking"}
    assert {k:v for k,v in before["task_profile"].items() if k not in excluded} == {k:v for k,v in after["task_profile"].items() if k not in excluded}
    assert before["task_profile"]["task_space_tracking"] == old_targets.settings()
    assert after["task_profile"]["task_space_tracking"] == targets.settings()
    for key in ("live_startup", "live_final"):
        a = after["cohort"][key]["task_space_tracking"]
        b = before["cohort"][key]["task_space_tracking"]
        assert a["status"] == "live_target_selection_verified"
        assert a["configured"]["settings"] == targets.settings()
        assert a["active_tracking_params"] == b["active_tracking_params"]
        assert a["active_term_weights"] == b["active_term_weights"]
    for key in ("asset", "model", "motion"):
        assert before["inputs"][key]["sha256"] == after["inputs"][key]["sha256"]
    def target_hash(m, filename):
        values = [v for k,v in m["cohort"]["source_sha256"].items() if Path(k).name == filename]
        assert len(values) == 1
        return values[0]
    assert target_hash(before, "car_taskspace_tracking.py") == sha(old_targets.__file__)
    assert target_hash(after, "car_wheel_axis_tracking.py") == sha(targets.__file__)
    proofs = dict(control=old_proofs["taskspace"], axis=driver["proofs"]["formal_K"])
    assert proofs["control"]["cumulative_transitions"] == proofs["axis"]["cumulative_transitions"] == 14745600
    rows, traces, baselines = [], {}, {}
    for engine in ("physx", "mujoco"):
        raw, _ = metrics.load_baseline(engine)
        baselines[engine], _ = metrics.recorded_metrics(raw, engine, reference)
        row = copy.deepcopy(next(r for r in old_rows if r["id"] == "taskspace_"+engine))
        row.update(id="control_"+engine, variant="control", title="2D wheel heading | "+engine)
        rows.append(row)
        traces[row["id"]] = old_traces["taskspace_"+engine]
    for case in status["cases"]:
        engine = case["backend"]
        f = folder/"nominal_v1"/case["id"]
        name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        verify_recording(f/name, "PhysX" if engine == "physx" else "native CPU MuJoCo")
        report, meta = read(f/"report.json"), read(f/"warm_contact_evaluation_status.json")
        assert meta["kind"] == "wheel_axis_refinement_native_evaluation"
        assert meta["normalized_actor_sha256"] == proofs["axis"]["actor_sha256"]
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
        row = dict(id="axis_"+engine, variant="axis", engine=engine, title="3D wheel axis | "+engine,
                   actor_sha256=meta["normalized_actor_sha256"], recording_sha256=sha(f/name), metrics=values,
                   nominal_screen=metrics.nominal_screen(values, baselines[engine], protocol, engine))
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
    result = dict(kind="matched_2D_heading_3D_axis_refinement_dual_engine", status="completed_analysis", rows=rows,
                  paired_training_proofs=proofs, historical_original_K=historical,
                  analyzer_sha256=sha(__file__), reference_sha256=sha(a.reference),
                  stage_transitions_per_arm=2457600, cumulative_transitions_per_resulting_policy=14745600,
                  only_changed_reward_definition="motion_body_ori:2D_axial_heading_to_3D_axial_direction",
                  contact_formula_all_weights_and_other14_rewards_unchanged=True,
                  same_K_initialization_optimizer_and_budget=True, all_original_gates_retained=True,
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
        ax.set_title(title); ax.grid(alpha=.2)
    axes[2].axhline(5, color="gray", ls="--")
    fig.suptitle("2D heading /3D axle | matched K warm refinement | mu=.7")
    fig.savefig(a.output_dir/"comparison.png", dpi=150); plt.close(fig)
    print(json.dumps({r["id"]:r["nominal_screen"] for r in rows}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("collection", "reference", "output-dir"):
        p.add_argument("--"+n, type=Path, required=True)
    run(p.parse_args())
