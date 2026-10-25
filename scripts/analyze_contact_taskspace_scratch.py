"""Matched scratch K/M native analysis with original complete task criteria."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as metrics
import car_taskspace_tracking as targets
import collect_contact_taskspace_scratch as collection
from ppo_backend_400hz_contract import verify_recording

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("K_physx", "M_physx", "K_mujoco", "M_mujoco")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def npz(path):
    with np.load(path, allow_pickle=False) as a: return {k: a[k].copy() for k in a.files}


def slip_summary(trace):
    """Recorded force-weighted per-point lateral RMS, never fill flight with0.

    Count is the number of individual points above5N. Sum wheel Fz above5N
    can coexist with count0; its placeholder slip0 is not measured rolling.
    """
    time = trace["time_s"]; count = trace["contact_count"]; slip = trace["contact_slip"]
    assert np.isfinite(slip).all() and np.all(slip >= 0.)
    assert np.all(count >= 0) and np.array_equal(count, np.rint(count))
    assert np.all(slip[count == 0] == 0.)
    core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
    loaded = trace["floor_normal_force"][:,:,2] > 5.
    valid = core[:,None] & loaded & (count > 0)
    result = {}
    for i, name in enumerate(("left", "right")):
        values = slip[valid[:,i],i]
        result[name] = dict(core_loaded_wheel_samples=int((core & loaded[:,i]).sum()),
            core_identified_loaded_point_samples=int(valid[:,i].sum()),
            core_loaded_samples_without_identified_strong_point=int((core & loaded[:,i] & (count[:,i] == 0)).sum()),
            core_lateral_point_RMS_m_s_median=None if not len(values) else float(np.median(values)),
            core_lateral_point_RMS_m_s_p95=None if not len(values) else float(np.quantile(values, .95)),
            core_identified_loaded_fraction_RMS_over_0p1_m_s=None if not len(values) else float(np.mean(values > .1)))
    return dict(definition="Normal-force-weighted RMS of lateral material-point velocity over actual individual ground contacts with point normal force>5N",
        ground_stationary=True, signed_slip_or_individual_point_vectors_retained=False,
        zero_contact_placeholder_excluded=True, descriptive_threshold_m_s=.1,
        new_task_gate=False, wheels=result)


def body_integrity(raw, trace):
    time = trace["time_s"]
    force = np.asarray(raw["nonwheel_ground_contact_force_norm_n"], dtype=float).reshape(-1)
    assert force.shape == time.shape and np.isfinite(force).all() and np.all(force >= 0.)
    overlap = time <= 6.96+1e-9
    after_initial = overlap & (time > 1e-9)
    selected = after_initial & (force > 5.)
    up = metrics.quaternion_rotate(trace["quaternion"], np.array([0.,0.,1.]))
    tilt = np.degrees(np.arccos(up[:,2].clip(-1.,1.)))
    return dict(initial_nonwheel_force_norm_n=float(force[0]),
        first_postinitial_nonwheel_force_over5N_time_s=None if not selected.any() else float(time[np.flatnonzero(selected)[0]]),
        postinitial_nonwheel_force_over5N_samples=int(selected.sum()),
        peak_nonwheel_force_norm_n=float(force[overlap].max()),
        max_whole_overlap_tilt_deg=float(tilt[overlap].max()),
        root_height_range_m=[float(trace["position"][overlap,2].min()), float(trace["position"][overlap,2].max())],
        absolute_height_or_flight_gate_added=False, qualitative_review_still_required=True)


def verify_pair(folder):
    folder = Path(folder); driver, status = collection.verify_completed(folder)
    protocol = read(folder/"source_snapshot_v1/config/contact_taskspace_scratch_400hz_v1.json")
    baseline = read(ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")
    for key in ("algorithm", "smoke", "training_randomization", "nominal_acceptance_before_training", "validation"):
        assert protocol[key] == baseline[key], key
    manifests = {v: read(folder/("formal_"+v)/"manifest.json") for v in ("K", "M")}
    k, m = manifests["K"], manifests["M"]
    assert k["initial_policy_state_sha256"] == m["initial_policy_state_sha256"] == collection.INITIAL
    assert read(folder/"formal_K/runner_config.json") == read(folder/"formal_M/runner_config.json")
    for variant, manifest in manifests.items():
        assert manifest["iterations"] == 1000 and manifest["num_envs"] == 512 and manifest["seed"] == 42
        c = manifest["cohort"]
        assert c["from_scratch"] and not c["historical_checkpoint_used"] and c["initial_checkpoint"] is None
        assert c["normalizer_initial_count"] == 0 and c["normalizer_final_count"] == 12288000
        assert c["optimizer"] == protocol["optimizer"] == dict(learning_rate=.001, schedule="adaptive", clip_param=.2)
        assert manifest["algorithm_source_sha256"] == dict(
            runner="e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5",
            ppo="deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c")
        assert manifest["contact_settings"] == protocol["variant_contact_settings"][variant]
        for phase in ("live_startup", "live_final"):
            a = c[phase]["task_space_tracking"]; b = k["cohort"][phase]["task_space_tracking"]
            assert a["status"] == "live_target_selection_verified"
            assert a["configured"]["settings"] == targets.settings() == protocol["task_space_tracking"]
            assert a["active_tracking_params"] == b["active_tracking_params"] and a["active_term_weights"] == b["active_term_weights"]
        pins = [h for p, h in c["source_sha256"].items() if Path(p).name == "car_taskspace_tracking.py"]
        assert pins == [sha(targets.__file__)]
    assert k["environment"]["reward_terms"] == m["environment"]["reward_terms"] and len(k["environment"]["reward_terms"]) == 15
    exclude = {"ground_contact_reward_v2", "description"}
    assert {n:v for n,v in k["task_profile"].items() if n not in exclude} == {n:v for n,v in m["task_profile"].items() if n not in exclude}
    before, after = k["contact_settings"], m["contact_settings"]
    assert {n for n in before if before[n] != after[n]} == {"candidate", "support_gap_weight"}
    assert before["support_gap_weight"] == 1. and after["support_gap_weight"] == 4.
    for key in ("asset", "model", "motion"):
        assert k["inputs"][key]["sha256"] == m["inputs"][key]["sha256"]
    return driver, status, protocol


def load_cases(folder, reference):
    folder = Path(folder); driver, status, protocol = verify_pair(folder)
    originals = {}
    for engine in ("physx", "mujoco"):
        raw, _ = metrics.load_baseline(engine)
        originals[engine], _ = metrics.recorded_metrics(raw, engine, reference)
    rows, traces = [], {}
    for child in status["cases"]:
        key, engine, variant = child["id"], child["backend"], child["variant"]
        f = folder/"nominal_v1"/key
        name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        verify_recording(f/name, "PhysX" if engine == "physx" else "native CPU MuJoCo")
        report = read(f/"report.json"); material = report["contact_material_audit"]
        meta = read(f/"scratch_contact_evaluation_status.json")
        if engine == "physx":
            assert report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
            assert report["terminal_states_captured_before_auto_reset"] and not report["domain_randomization_enabled"]
            assert not report["observation_noise_enabled"] and report["runtime_device_audit"]["status"] == "verified"
            assert material["live_physx_materials_verified"] and material["ground_material_usd_verified"]
            assert not material["recipe"]["training"] and material["recipe"]["wheel_contact_friction"]["range"] == [.7, .7]
            assert material["encoder_bias_range_rad"] == [0., 0.]
        else:
            assert report["status"] == "completed" and report["completed_full_reference"]
            assert report["hidden_resets"] == 0 and report["root_state_writes"] == 1
            assert report["physics_steps"] == 2792 and report["policy_steps"] == 349
            assert material["native_mapping_verified"] and material["pair_coefficient"] == .7
            assert material["explicit_contact_pair_count"] == 0
            assert all(abs(v[0]-.7) < 1e-12 for v in material["geom_friction"])
            assert all(v == (1 if i in material["wheel_geom_ids"] else 0)
                       for i, v in zip(material["colliding_geom_ids"], material["geom_priority"]))
        raw = npz(f/name)
        values, trace = metrics.recorded_metrics(raw, engine, reference)
        trace["contact_force"] = trace["floor_normal_force"].copy()
        rows.append(dict(id=key, variant=variant, engine=engine, title=variant+" fresh task-space | "+engine,
            actor_sha256=meta["normalized_actor_sha256"], recording_sha256=sha(f/name), metrics=values,
            loaded_contact_sliding=slip_summary(trace),
            body_integrity=body_integrity(raw, trace),
            nominal_screen=metrics.nominal_screen(values, originals[engine], protocol, engine)))
        traces[key] = trace
    rows.sort(key=lambda r: ORDER.index(r["id"]))
    assert [r["id"] for r in rows] == list(ORDER)
    initial = traces[ORDER[0]]
    for key in ORDER[1:]:
        t = traces[key]
        for field in ("position", "com_velocity", "angular_velocity", "joint_pos", "joint_vel"):
            np.testing.assert_allclose(t[field][0], initial[field][0], atol=2e-6, rtol=0)
        assert abs(np.dot(t["quaternion"][0], initial["quaternion"][0])) > 1-1e-7
    return rows, traces, driver["proofs"], originals


def run(args):
    assert not args.output_dir.exists()
    reference = metrics.metrics.reference_data(args.reference)
    rows, traces, proofs, historical = load_cases(args.collection, reference)
    args.output_dir.mkdir(parents=True)
    result = dict(kind="paired_scratch_taskspace_K_M_contact_dual_engine", status="completed_analysis", rows=rows,
        paired_training_proofs=proofs, historical_original_PPO=historical, analyzer_sha256=sha(__file__),
        reference_sha256=sha(args.reference), transitions_per_arm=12288000, from_scratch=True,
        historical_checkpoint_used=False, all_original_gates_retained=True, qualitative_review_complete=False,
        force_weighted_contact_point_lateral_RMS_recorded=True,
        individual_point_vectors_or_signed_slip_recorded=False, task_success_claimed=False)
    (args.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
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
        axes[3].plot(t["time_s"], np.degrees(np.arccos(up[:,2].clip(-1.,1.))))
    axes[0].set_aspect("equal", adjustable="datalim"); axes[0].legend(fontsize=6)
    for ax, title in zip(axes, ("World XY [m]", "COM XY speed [m/s]", "Max wheel floor Fz [N]", "Body tilt [deg]")):
        ax.set_title(title); ax.grid(alpha=.2)
    axes[2].axhline(5, color="gray", ls="--")
    fig.suptitle("Fresh task-space K/M | contact-margin coefficient1 vs4 | matched12.288M transitions | mu=.7")
    fig.savefig(args.output_dir/"comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharex=True, sharey=True, layout="constrained")
    for ax, row in zip(axes, rows):
        t = traces[row["id"]]
        valid = (t["floor_normal_force"][:,:,2] > 5.) & (t["contact_count"] > 0)
        for i, name in enumerate(("Left", "Right")):
            ax.plot(t["time_s"], np.where(valid[:,i], t["contact_slip"][:,i], np.nan), label=name, alpha=.8)
        ax.axvspan(1.44, 4., color="gray", alpha=.12); ax.axhline(.1, color="gray", ls="--")
        ax.set(title=row["id"], xlabel="Time(s)", ylabel="Loaded point lateral RMS(m/s)"); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle("Actual per-point force-weighted slip RMS | gaps are unmeasured/unloaded,not zero rolling |0.1m/s descriptive only")
    fig.savefig(args.output_dir/"contact_slip.png", dpi=150); plt.close(fig)
    print(json.dumps({r["id"]: r["nominal_screen"]["failed_checks"] for r in rows}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("collection", "reference", "output-dir"): p.add_argument("--"+n, type=Path, required=True)
    run(p.parse_args())
