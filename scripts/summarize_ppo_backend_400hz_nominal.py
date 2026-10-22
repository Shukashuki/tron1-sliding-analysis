"""Summarize four retrieved nominal rollouts while the 17-mu matrix is partial.

This is offline FK/analysis only. Original remote verification proofs and copied
training metadata bind the records; no checkpoint or actor is downloaded/loaded.
It deliberately does not call the complete-68-cell matrix loader.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as analysis
from ppo_backend_400hz_contract import (DEFAULT_PROTOCOL, INPUT_PINS, read_json,
    require, sha256, verify_protocol, verify_recording, write_json)

ANALYZER_SHA256 = "2645c26150b4bbca5a5631beee3d1cf1178ee365bf0a11044d7e8c618220c0a1"
KIND = "verified_nominal_400hz_backend_partial_matrix_replay_inputs"
ORIGINS = ("physx", "mjlab")


def local_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts,
            "Payload path must stay relative to the retrieval root")
    path = root / relative
    require(path.is_file() and all(not part.is_symlink() for part in (path, *path.parents)),
            "Missing or indirect payload file: " + str(name))
    require(path.resolve().is_relative_to(root), "Payload escaped retrieval root")
    return path


def verify_files(root, manifest):
    ledger = {}
    require(isinstance(manifest.get("files"), list) and manifest["files"], "Missing payload SHA ledger")
    for item in manifest["files"]:
        name = item["path"]
        require(name not in ledger and name != "manifest.json", "Duplicate/self-referential payload ledger entry")
        path = local_file(root, name)
        require(path.stat().st_size == item["size"] and sha256(path) == item["sha256"],
                "Retrieved payload SHA/size differs: " + name)
        ledger[name] = item["sha256"]
    return ledger


def metadata_for(root, origin, manifest, ledger, protocol_path):
    paths = {name: local_file(root, value) for name, value in manifest["training_metadata"][origin].items()}
    required = {"completion_audit", "final_policy_contract", "run_status", "manifest", "startup_audit"}
    require(required <= paths.keys(), "Missing formal training metadata")
    for path in paths.values():
        require(str(path.relative_to(root)) in ledger, "Training metadata absent from SHA ledger")
    completion, contract, status = (read_json(paths[key]) for key in
                                  ("completion_audit", "final_policy_contract", "run_status"))
    training_manifest, startup = (read_json(paths[key]) for key in ("manifest", "startup_audit"))
    expected = {"status": "completed_verified", "run_kind": "formal", "backend": origin,
        "cohort_id": manifest["cohort_id"], "smoke_only": False, "trained_policy_claimed": True,
        "training_environment_steps": 12288000, "seed": 42, "protocol_sha256": sha256(protocol_path),
        "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "action_rate_weight": -.5, "grounded_reward_weight": .4,
        "source_files_unchanged": True, "additional_reward_terms": False, "historical_checkpoint_used": False}
    for key, value in expected.items():
        require(completion.get(key) == value, "Formal completion metadata differs: " + key)
    require(completion["final_policy_contract_sha256"] == sha256(paths["final_policy_contract"]),
            "Copied final contract differs from completed training")
    require(completion["manifest_sha256"] == sha256(paths["manifest"])
            and completion["startup_audit_sha256"] == sha256(paths["startup_audit"]),
            "Copied training manifest/startup differs from completion audit")
    require(status.get("status") == "training_completed" and status.get("backend") == origin
            and status.get("training_environment_steps") == 12288000, "Formal training status incomplete")
    for item in (contract, status):
        require(item.get("run_kind") == "formal" and item.get("smoke_only") is False
                and item.get("trained_policy_claimed") is True
                and item.get("checkpoint_sha256") == completion["checkpoint_sha256"]
                and item.get("exported_policy_sha256") == completion["exported_policy_sha256"],
                "Copied formal actor/checkpoint identity differs")
    expected_contract = {"contract_status": "trained_completed_normalized_actor", "policy_algorithm": "PPO",
        "policy_training_backend": origin, "physics_dt": .0025, "decimation": 8, "policy_fps": 50.,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8, "leg_kp": 500., "leg_kd": 10.,
        "actor_includes_observation_normalizer": True, "runtime_clock_and_interface_verified": True,
        "physics_clock_origin": "trained_at_native400Hz_not_evaluation_only_refinement",
        "reference_file_sha256": INPUT_PINS["motion"], "training_cohort_protocol_sha256": sha256(protocol_path)}
    for key, value in expected_contract.items():
        require(contract.get(key) == value, "Copied trained contract differs: " + key)
    profile = contract["trained_task_profile"]
    require(profile == contract["task_profile"] and profile["action_rate_weight"] == -.5
            and profile["reward_weights"]["grounded"] == .4
            and profile["height_regularization"]["mode"] == "vertical_acceleration_only",
            "Copied task profile differs from the frozen 400Hz baseline")
    require(completion["initial_policy_state_sha256"] == status["initial_policy_state_sha256"],
            "Training initialization identity differs")
    for item in (training_manifest, training_manifest["cohort"], startup):
        require(item.get("smoke_only") is False
                and item.get("trained_policy_claimed") is True
                and item.get("initial_policy_state_sha256") == completion["initial_policy_state_sha256"],
                "Copied startup/manifest run or network identity differs")
    require(startup["run_kind"] == training_manifest["cohort"]["run_kind"] == "formal",
            "Startup/cohort run is not formal")
    require(startup["runtime"]["status"] == "verified" and training_manifest["backend"] == origin,
            "Training startup/backend not verified")
    return {"paths": paths, "completion": completion, "contract": contract, "status": status}


def check_case(root, case, source_case, metadata, ledger, protocol_path):
    origin, engine = case["training_backend"], case["engine_tag"]
    backend = analysis.ENGINE_LABELS[engine]
    require(case["case_id"] == origin + "_to_" + engine and case["mu"] == .7
            and case["validation_backend"] == backend, "Nominal case labels differ")
    require(source_case.get("status") == "completed_verified" and source_case["directory"] == case["remote_directory"],
            "Original remote case is not completed or directory differs")
    proof = case["verification"]
    require(proof == source_case["verification"], "Copied proof differs from original queue snapshot")
    expected_proof = {"status": "verified", "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
        "physics_dt_s": .0025, "policy_dt_s": .02, "validation_backend": backend, "training_backend": origin,
        "smoke_only": False, "evaluation_contact_friction": .7,
        "checkpoint_sha256": metadata["completion"]["checkpoint_sha256"],
        "recording_sha256": case["recording_sha256"], "report_sha256": case["report_sha256"]}
    for key, value in expected_proof.items():
        require(proof.get(key) == value, "Remote validation proof differs: " + key)
    require(case["checkpoint_sha256"] == source_case["checkpoint_sha256"] == proof["checkpoint_sha256"]
            and case["actor_sha256"] == source_case["actor_sha256"] == metadata["completion"]["exported_policy_sha256"],
            "Nominal actor/checkpoint identity differs")
    raw_name = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
    require(case["raw_file"] == raw_name and case["report_file"] == "report.json", "Unexpected record filename")
    recording = local_file(root, str(Path(case["directory"]) / raw_name))
    report_file = local_file(root, str(Path(case["directory"]) / "report.json"))
    for path, digest in ((recording, case["recording_sha256"]), (report_file, case["report_sha256"])):
        require(ledger.get(str(path.relative_to(root))) == digest, "Case file absent from ledger or proof SHA differs")
    report = read_json(report_file)
    audit = report["backend_400hz_cohort_evaluation"]
    expected_audit = {"status": "verified", "training_backend": origin, "validation_backend": backend,
        "run_kind": "formal", "smoke_only": False, "trained_policy_claimed": True,
        "checkpoint_sha256": case["checkpoint_sha256"], "normalized_actor_sha256": case["actor_sha256"],
        "protocol_sha256": sha256(protocol_path), "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
        "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98, "old_200hz_actor_used": False,
        "additional_training_updates": 0, "normalizer_included_in_actor": True, "outer_normalizer": "Identity"}
    for key, value in expected_audit.items():
        require(audit.get(key) == value, "Nominal report cohort provenance differs: " + key)
    mapping = {"completion_audit": "cohort_completion_audit.json", "final_policy_contract": "final_policy_contract.json",
        "run_status": "run_status.json", "manifest": "manifest.json", "startup_audit": "cohort_startup_audit.json"}
    for local_name, original_name in mapping.items():
        require(audit["cohort_input_sha256"][original_name] == sha256(metadata["paths"][local_name]),
                "Report is not bound to copied completed metadata: " + original_name)
    wrapper = "eval_ppo_backend_400hz_isaac.py" if engine == "physx" else "eval_ppo_backend_400hz_mujoco.py"
    require(audit["wrapper_sha256"] == sha256(Path(__file__).with_name(wrapper)), "Evaluation adapter source differs")
    contract = report["policy_export" if engine == "physx" else "contract"]
    require(contract["physics_dt"] == .0025 and contract["decimation"] == 8
            and contract["task_profile"]["contact_friction"] == .7
            and contract["trained_task_profile"] == metadata["contract"]["trained_task_profile"],
            "Evaluation clock/profile differs from nominal declared deployment")
    if engine == "physx":
        require(report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
                and report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
                and report["physics_trajectory_sha256"] == case["recording_sha256"], "PhysX lifecycle differs")
    else:
        require(report["status"] == "completed" and report["completed_full_reference"] is True
                and report["hidden_resets"] == 0 and report["root_state_writes"] == 1
                and report["start_reference_frame"] == 0
                and report["domain_parameters_resampled_during_episode"] is False
                and report["policy_sha256"] == case["actor_sha256"], "Native MuJoCo lifecycle differs")
    local_proof = verify_recording(recording, backend)
    for key, value in local_proof.items():
        require(proof[key] == value, "Retrieved native recording differs from original remote proof: " + key)
    return recording, report


def load_nominal(input_root, reference, model, legacy_evaluator, protocol_path=DEFAULT_PROTOCOL):
    """Return tuple-keyed four native traces, reference values and a partial summary."""
    require(sha256(analysis.__file__) == ANALYZER_SHA256, "Frozen native metric analyzer source differs")
    root = Path(input_root).resolve()
    manifest_file = local_file(root, "manifest.json")
    manifest_digest = sha256(manifest_file)
    manifest = read_json(manifest_file)
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == KIND
            and manifest.get("nominal_only") is True and manifest.get("full_matrix_complete") is False
            and manifest.get("nominal_mu") == .7, "Not a nominal-only partial-matrix retrieval")
    ledger = verify_files(root, manifest)
    protocol = verify_protocol(protocol_path)
    require(manifest["cohort_id"] == protocol["cohort_id"] and manifest["reference_sha256"] == sha256(reference)
            and manifest["model_sha256"] == sha256(model), "Nominal cohort/reference/model differs")
    protocol_copy = local_file(root, "config/ppo_backend_400hz_comparison_v1.json")
    require(ledger.get(str(protocol_copy.relative_to(root))) == sha256(protocol_path), "Copied cohort protocol differs")
    snapshot_path = local_file(root, "matrix_status_snapshot.json")
    require(str(snapshot_path.relative_to(root)) in ledger, "Original queue status snapshot absent from SHA ledger")
    snapshot = read_json(snapshot_path)
    require(manifest["matrix_status"] == {key: snapshot[key] for key in ("status", "counts")},
            "Partial progress differs from original queue snapshot")
    counts = snapshot["counts"]
    require(counts["formal_expected"] == 68 and 4 <= counts["formal_completed"] < 68
            and counts["pipeline_smoke_expected"] == counts["pipeline_smoke_completed"] == 4
            and counts["runtime_failed"] == 0, "This summary requires a healthy partial 68-case matrix")
    cases = manifest["cases"]
    expected = {(origin, engine) for origin in ORIGINS for engine in analysis.ENGINE_LABELS}
    require(len(cases) == 4 and {(c["training_backend"], c["engine_tag"]) for c in cases} == expected,
            "Expected exactly four unique nominal train/evaluation pairs")
    originals = [c for c in snapshot["cases"] if c["phase"] == "formal" and c["mu"] == .7]
    require(len(originals) == 4 and {(c["training_backend"], c["engine_tag"]) for c in originals} == expected,
            "Original queue nominal inventory differs")
    originals = {(c["training_backend"], c["engine_tag"]): c for c in originals}
    metadata = {origin: metadata_for(root, origin, manifest, ledger, protocol_path) for origin in ORIGINS}
    require(len({m["completion"]["initial_policy_state_sha256"] for m in metadata.values()}) == 1
            and len({m["completion"]["checkpoint_sha256"] for m in metadata.values()}) == 2,
            "Paired initialization/checkpoint identities differ")
    verified = []
    for case in cases:
        key = (case["training_backend"], case["engine_tag"])
        path, report = check_case(root, case, originals[key], metadata[key[0]], ledger, protocol_path)
        verified.append((case, path, report))
    reference_values = analysis.reference_data(reference)
    kinematics, fk_audit = analysis.load_fk(model, legacy_evaluator)
    rows, nominal = [], {}
    for case, path, report in verified:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        trace, metrics = analysis.analyze_case(raw, case["engine_tag"], reference_values, kinematics)
        row = {"id": case["case_id"], "training_backend": case["training_backend"],
            "engine_tag": case["engine_tag"], "validation_backend": case["validation_backend"], "mu": .7,
            "directory": str((root / case["directory"]).resolve()), "checkpoint_sha256": case["checkpoint_sha256"],
            "actor_sha256": case["actor_sha256"], "input_hashes": {"report": case["report_sha256"], "recording": case["recording_sha256"]},
            "remote_verification": case["verification"],
            "material_mapping": analysis.material_summary(report["contact_material_audit"], case["engine_tag"]), **metrics}
        rows.append(row)
        nominal[(case["training_backend"], case["engine_tag"])] = trace
    require(sha256(manifest_file) == manifest_digest, "Retrieval manifest changed during offline analysis")
    verify_files(root, manifest)
    helper_names = ("analyze_ppo_smoothness.py", "analyze_ppo_contact_stability.py", "analyze_ppo_contact_timestep.py",
        "analyze_ppo_ground_support_failure.py", "controller_metrics.py", "controller_trace.py",
        "nmpc_tracking_objective.py", "height_smoothness.py", "ppo_backend_400hz_contract.py")
    return nominal, reference_values, {"schema_version": 1, "kind": "new400hz_nominal_four_partial_matrix_summary",
        "cohort_id": protocol["cohort_id"], "nominal_mu": .7, "nominal_only": True, "full_matrix_complete": False,
        "nominal_cells_completed": 4, "matrix_status": manifest["matrix_status"], "rows": rows,
        "observed_utc": manifest["observed_utc"], "sample_size_per_cell": 1,
        "physics_dt_s": .0025, "policy_dt_s": .02, "reference_real_last_time_s": 6.96,
        "retrieval_manifest_sha256": manifest_digest, "matrix_status_snapshot_sha256": sha256(snapshot_path),
        "reference_sha256": sha256(reference), "model_sha256": sha256(model), "protocol_sha256": sha256(protocol_path),
        "summarizer_sha256": sha256(__file__), "native_analyzer_sha256": ANALYZER_SHA256, "FK_audit": fk_audit,
        "analysis_helpers_sha256": {name: sha256(Path(__file__).with_name(name)) for name in helper_names},
        "remote_proofs_bound_to_local_payload": True, "checkpoint_or_actor_downloaded_or_loaded": False,
        "new_simulation": False, "new_policy_inference": False, "visual_reviewed": False,
        "limitations": ["Only four nominal mu=0.7 cells; the17-mu matrix is not yet complete.",
            "Original remote validation proofs are hash-bound to copied reports/records/training metadata; weights are not locally reverified.",
            "N=1 descriptive cell, not a statistical backend ranking or success probability.",
            "Reference ends6.96s; native episode ends6.98s. Full tracking is unavailable without extrapolation; overlap is separate.",
            "No alignment, root correction, smoothing or dynamics replay is applied.",
            "Intended nonzero beta is not a failure; only reference-relative error is reported.",
            "Height, tilt and zero measured ground load are not tracking failure gates.",
            "Native contact measurements and equal target mu do not imply equivalent contact laws or identical measurement operators."]}


def render_topview(nominal, reference, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    xy = np.concatenate([reference["position"][:, :2], *[t["position"][:, :2] for t in nominal.values()]])
    low, high = xy.min(axis=0) - .35, xy.max(axis=0) + .35
    colors = analysis.COLORS
    for i, origin in enumerate(ORIGINS):
        for j, engine in enumerate(analysis.ENGINE_LABELS):
            ax, trace = axes[i, j], nominal[(origin, engine)]
            ax.plot(*reference["position"][:, :2].T, "--", color=colors["reference"], lw=1.1)
            ax.plot(*trace["position"][:, :2].T, color=colors["path"], lw=1.)
            ax.scatter(*trace["position"][0, :2], marker="o", s=16, color=colors["path"])
            ax.scatter(*trace["position"][-1, :2], marker="x", s=25, color=colors["path"])
            body_valid, wheel_valid = analysis.heading_geometry_masks(trace)
            signals = trace["signals"]
            for stamp in (2., 3., 4.):
                k = int(round(stamp / .0025))
                root = trace["position"][k, :2]
                arrows = [(root, signals["body_heading_rad"][k], "body", body_valid[k]),
                    (root, signals["course_rad"][k], "velocity", signals["speed_m_s"][k] >= .25)]
                for wheel, color in enumerate(("left", "right")):
                    arrows.append((trace["wheel_pos"][k, wheel, :2], signals["wheel_rolling_plane_heading_rad"][k, wheel],
                                   color, wheel_valid[k, wheel]))
                for start, heading, color, valid in arrows:
                    if valid:
                        direction = .18 * np.array([np.cos(heading), np.sin(heading)])
                        axial = color in ("left", "right")
                        ax.annotate("", xy=start + direction * (.5 if axial else 1.),
                            xytext=start - direction * .5 if axial else start,
                            arrowprops={"arrowstyle": "<->" if axial else "->", "color": colors[color], "lw": 1.25})
                ax.text(*root, f" {stamp:g}s", fontsize=7)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World X (m)", ylabel="World Y (m)",
                   title=analysis.TRAIN_LABELS[origin] + " -> " + analysis.ENGINE_LABELS[engine])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
    handles = [Line2D([], [], color=colors[key], label=label) for key, label in
        (("reference", "Reference path"), ("path", "Measured whole turn (o start / x end)"), ("body", "Body +X"),
         ("velocity", "COM velocity"), ("left", "Left rolling plane (axial)"), ("right", "Right rolling plane (axial)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    count = summary["matrix_status"]["counts"]["formal_completed"]
    title = f"New400Hz PPO nominal mu=0.7: four cells complete | full matrix PARTIAL ({count}/68 observed)"
    fig.suptitle(title + "\nWhole6.98s paths; shared bounds; no alignment; fixed0.18m arrows at2/3/4s", fontsize=11)
    fig.tight_layout(rect=(0., .07, 1., .94))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"title": title, "shared_world_bounds": {"xlim": low[[0]].tolist() + high[[0]].tolist(),
        "ylim": low[[1]].tolist() + high[[1]].tolist()}, "arrow_length_m": .18, "arrow_timestamps_s": [2., 3., 4.],
        "wheel_arrow_kind": "opaque axial rolling plane from wheel local+Y axle cross world+Z; spin invariant",
        "contact_gates_arrow_opacity": False, "whole_native_path_s": [0., 6.98], "spatial_alignment": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("input-root", "reference", "model", "legacy-mujoco-evaluator", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--cohort-protocol", type=Path, default=DEFAULT_PROTOCOL)
    return result


def run(args):
    require(not args.output_dir.exists(), "Preserve previous nominal analysis; use a fresh output directory")
    nominal, reference, summary = load_nominal(args.input_root, args.reference, args.model,
                                            args.legacy_mujoco_evaluator, args.cohort_protocol)
    args.output_dir.mkdir(parents=True)
    summary["plot_settings"] = render_topview(nominal, reference, summary, args.output_dir / "nominal_2x2_topview.png")
    table, overview = analysis.compact_metrics(summary)
    overview.update(kind="new400hz_nominal_four_partial_matrix_overview", nominal_only=True,
        full_matrix_complete=False, nominal_cells_completed=4, matrix_status=summary["matrix_status"],
        interpretation="Four nominal N=1 cells only;17-mu matrix partial. No continuous feasible-range or stable-contact claim.")
    overview.pop("tested_mu_lists")
    write_json(args.output_dir / "overview.json", overview)
    with (args.output_dir / "metric_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    summary["artifacts"] = {name: sha256(args.output_dir / name) for name in
        ("overview.json", "metric_table.csv", "nominal_2x2_topview.png")}
    write_json(args.output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run(parser().parse_args())
