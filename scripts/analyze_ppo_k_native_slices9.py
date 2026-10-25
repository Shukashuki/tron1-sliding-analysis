"""Four verified PhysX arms: training geometry x validation geometry.

Reuses the two published original-K native-force recordings. New recordings
must come from the completed, independently initialized twelve-million-step
nine-slice training and its strict sequential validation, never the smoke.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_physx_tron1_wheel_models as previous
import analyze_ppo_ground_contact_v2 as metrics
import ppo_native_slices9_model as model
import verify_ppo_k_native_slices9 as verification

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("authored_to_authored", "authored_to_slices9", "slices9_to_slices9", "slices9_to_authored")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_new_training(collection, preflight):
    collection, preflight = Path(collection), Path(preflight)
    status = read(collection/"nominal_v1/status.json")
    require(status["status"] == "both_nominal_arms_completed_require_analysis", "Both verified nominal arms are required")
    proof = read(collection/"nominal_v1/formal_training_strict_verification.json")
    require(proof["kind"] == "strict_completed_native_slices9_training_verification"
        and proof["cohort_id"] == model.COHORT_ID and proof["model_id"] == model.MODEL_ID,
        "Wrong model-specific completed training identity")
    require(proof["formal_or_smoke"] == "formal" and proof["transitions"] == 12288000,
        "A smoke or incomplete training cannot enter the formal model comparison")
    require(proof["startup_and_final_all_environment_native_model_verified"]
        and proof["reward_and_DR_parent_K_verified"] and proof["native_environment_count"] == 512
        and proof["CPU_actor_output_shape"] == [1, 8], "Missing native-model or normalized export proof")
    require(status["formal_training_verification"] == proof, "Continuation and training verifier disagree")
    folder = collection/"formal_v1"
    audit = read(folder/"slices9_model_training_audit.json")
    manifest = read(folder/"manifest.json")
    contract = read(folder/"policy_contract.json")
    completion = read(folder/"cohort_completion_audit.json")
    training = read(folder/"ground_contact_v2k_training_audit.json")
    protocol = read(preflight/"source_snapshot_v1/config/ppo_k_native_slices9_400hz_v1.json")
    require(sha(folder/"slices9_model_training_audit.json") == proof["training_audit_sha256"], "Model training audit hash differs")
    require(audit["status"] == "completed_verified" and audit["model_id"] == model.MODEL_ID
        and audit["cohort_id"] == model.COHORT_ID and audit["training_environment_steps"] == 12288000
        and audit["smoke_only"] is False and audit["trained_policy_claimed"] is True,
        "Formal model training was not completed")
    require(audit["actor_sha256"] == proof["actor_sha256"] == training["exported_policy_sha256"]
        == completion["exported_policy_sha256"], "Normalized formal actor identity differs")
    require(audit["checkpoint_sha256"] == proof["checkpoint_sha256"] == training["checkpoint_sha256"]
        == completion["checkpoint_sha256"], "Formal checkpoint identity differs")
    for key, name in (("manifest_sha256", "manifest.json"), ("policy_contract_sha256", "policy_contract.json"),
                      ("K_training_audit_sha256", "ground_contact_v2k_training_audit.json"),
                      ("model_startup_audit_sha256", "slices9_model_startup_audit.json")):
        require(audit[key] == sha(folder/name), "Training metadata hash differs: "+name)
    require(audit["protocol_sha256"] == sha(preflight/"source_snapshot_v1/config/ppo_k_native_slices9_400hz_v1.json"),
        "Training model protocol differs")
    require(manifest["contract"] == contract and manifest["num_envs"] == 512 and manifest["iterations"] == 1000,
        "Actual runner configuration or contract differs")
    require(contract["training_domain_randomization"] == protocol["training_randomization"]
        and contract["wheel_contact_model"] == protocol["wheel_contact_model"], "Training DR or geometry differs")
    require(audit["initial_policy_state_sha256"] == verification.training.EXPECTED_INITIAL_POLICY_SHA256,
        "Initial policy differs from the matched original K initialization")
    require(audit["model_audit"] == manifest["wheel_contact_model_training"], "Native-model audit differs")
    for key in ("startup_native_audit", "final_native_audit"):
        native = audit["model_audit"][key]
        require(native["status"] == "all_environments_native_model_verified" and native["num_envs"] == 512
            and native["native_wheel_bodies_checked"] == 1024
            and native["per_environment_wheel_shape_counts"] == [[9, 9]]*512,
            "Not all formal training wheel bodies have native nine-slice geometry")
    require(audit["model_audit"]["from_scratch"] and audit["model_audit"]["other14_rewards_and_K_arithmetic_unchanged"],
        "Formal training changed reward or used a historical actor")
    return status, proof, audit, protocol


def verify_new_case(folder, child, proof, audit, geometry, preflight):
    folder = Path(folder)
    metadata = read(folder/"model_evaluation_status.json")
    report = read(folder/"report.json")
    require(child["id"] == geometry and child["status"] == "completed_verified" and child["returncode"] == 0,
        "Nominal child did not complete strict verification")
    require(child["metadata_sha256"] == sha(folder/"model_evaluation_status.json"), "Model metadata hash differs")
    require(metadata == report["model_specific_slices9_evaluation"]
        == report["policy_export"]["validation_wheel_contact_model"], "Deployed validation model metadata differs")
    require(metadata["kind"] == "model_specific_native_slices9_ppo_evaluation"
        and metadata["status"] == "completed_requires_analysis", "Wrong completed model evaluation")
    require(metadata["training_cohort_id"] == model.COHORT_ID and metadata["training_model_id"] == model.MODEL_ID
        and metadata["model_training_audit_sha256"] == proof["training_audit_sha256"], "Wrong trained policy model")
    require(metadata["normalized_actor_sha256"] == proof["actor_sha256"]
        and metadata["training_model"] == audit["model_audit"]["settings"], "Training actor or model settings differ")
    require(metadata["validation_geometry"] == geometry and metadata["validation_force_law"] == "native_coulomb"
        and metadata["validation_contact_offset_m"] == .001 and metadata["validation_rest_offset_m"] == 0.
        and metadata["mu"] == .7 and metadata["native_states"] == 2793 and metadata["policy_decisions"] == 349,
        "Nominal validation geometry, force law, offsets, friction or horizon differs")
    require(report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
        and report["domain_randomization_enabled"] is False and report["observation_noise_enabled"] is False
        and report["terminal_states_captured_before_auto_reset"] is True, "Nominal lifecycle differs")
    require(report["runtime_device_audit"]["status"] == "verified", "PhysX device was not verified")
    parent_proof = report["ground_contact_v2k_evaluation"]
    require(parent_proof["cohort_id"] == model.COHORT_ID and parent_proof["status"] == "verified"
        and parent_proof["normalized_actor_sha256"] == proof["actor_sha256"]
        and parent_proof["checkpoint_sha256"] == proof["checkpoint_sha256"]
        and parent_proof["native_states"] == 2793 and parent_proof["policy_decisions"] == 349,
        "Full-horizon normalized actor proof differs")
    source_files = {p.name: sha(p) for p in Path(preflight).rglob("*.py")}
    source_files["report.json"] = model.BASELINE_REPORT_SHA
    for remote_path, digest in metadata["source_sha256"].items():
        name = Path(remote_path).name
        if name not in source_files:
            local = ROOT/"scripts"/name
            require(local.is_file(), "Missing local audited model helper: "+name)
            source_files[name] = sha(local)
        require(source_files[name] == digest, "Executed model evaluation source differs: "+name)
    for name in ("physics_trajectory", "tire_trajectory"):
        require(sha(folder/(name+".npz")) == metadata[name+"_sha256"] == child[name+"_sha256"],
            "Native recording hash differs: "+name)
    require(parent_proof["recording_sha256"] == metadata["physics_trajectory_sha256"], "Parent recording proof differs")
    native = metadata["native_audit"]
    expected = 9 if geometry == "slices9" else 1
    require(native["masses_COM_inertias_armature_damping_match_authored"]
        and native["shapes_per_wheel"] == expected and native["total_native_shapes"] == 7+2*expected,
        "Validation shape count or original mass properties differ")
    for wheel in model.WHEELS:
        slots = native["shape_mapping"][wheel]
        require(len(slots) == expected, "Native wheel shape mapping differs")
        np.testing.assert_allclose(np.asarray(native["contact_offsets_m"])[:, slots], .001, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(native["rest_offsets_m"])[:, slots], 0., rtol=0, atol=1e-8)
    return metadata


def load_cases(collection, preflight, previous_result, reference):
    status, proof, audit, protocol = verify_new_training(collection, preflight)
    old_summary = read(Path(previous_result)/"analysis.json")
    require(old_summary["analyzer_sha256"] == sha(previous.__file__), "Published original-K analyzer differs")
    require(old_summary["reference_sha256"] == sha(reference["source_path"]), "Old and new motion references differ")
    old_rows, old_traces, old_raws, _, original = previous.load_cases(
        Path(previous_result)/"evidence", reference, protocol, original_metrics=old_summary["original_baseline_metrics"])
    require(proof["actor_sha256"] != previous.ACTOR, "New model training unexpectedly reuses the original actor")
    rows, traces, raws = [], {}, {}
    for geometry in ("authored", "slices9"):
        name = geometry+"__native_coulomb"
        old = next(row for row in old_rows if row["id"] == name)
        key = "authored_to_"+geometry
        rows.append(dict(id=key, training_geometry="authored", validation_geometry=geometry,
            actor_sha256=previous.ACTOR, source="previous_published_native_model_transfer",
            metrics=copy.deepcopy(old["metrics"]), nominal_screen=copy.deepcopy(old["nominal_screen"]),
            input_sha256=copy.deepcopy(old["input_sha256"])))
        traces[key], raws[key] = old_traces[name], old_raws[name]
    children = {child["id"]: child for child in status["cases"]}
    require(set(children) == {"slices9", "authored"}, "Exactly two new-model validation arms are required")
    for geometry in ("slices9", "authored"):
        folder = Path(collection)/"nominal_v1"/geometry
        metadata = verify_new_case(folder, children[geometry], proof, audit, geometry, preflight)
        require(sha(Path(collection)/"nominal_v1"/(geometry+"_verification.log")) == children[geometry]["verification_log_sha256"],
            "Strict child verification log differs")
        raw = previous.npz(folder/"physics_trajectory.npz")
        values, trace = metrics.recorded_metrics(raw, "physx", reference)
        trace["contact_force"] = trace["floor_normal_force"].copy()
        tire = previous.npz(folder/"tire_trajectory.npz")
        np.testing.assert_allclose(tire["time_before_step_s"], trace["time_s"][:-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(tire["preceding_normal_load_n"], trace["contact_force"][:-1, :, 2], rtol=1e-6, atol=1e-4)
        np.testing.assert_array_equal(tire["applied_tangent_force_w_n"], np.zeros((2792, 2, 3)))
        np.testing.assert_array_equal(tire["applied_tangent_moment_COM_w_nm"], np.zeros((2792, 2, 3)))
        require(np.max(tire["native_patch_count"]) < 128, "Native contact patch buffer may have saturated")
        key = "slices9_to_"+geometry
        rows.append(dict(id=key, training_geometry="slices9", validation_geometry=geometry,
            actor_sha256=proof["actor_sha256"], source="new_formal_model_specific_training",
            metrics=values, nominal_screen=metrics.nominal_screen(values, original, protocol, "physx"),
            input_sha256={name: sha(folder/name) for name in ("report.json", "model_evaluation_status.json",
                "physics_trajectory.npz", "tire_trajectory.npz")}, model_evaluation=metadata))
        traces[key], raws[key] = trace, raw
    require(tuple(row["id"] for row in rows) == ORDER, "Four-panel case order differs")
    for key in ORDER[1:]:
        for field in ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel"):
            np.testing.assert_array_equal(traces[key][field][0], traces[ORDER[0]][field][0],
                err_msg="Paired original initial state differs: "+field)
    return rows, traces, raws, original, proof


def run(args):
    require(not args.output_dir.exists(), "Preserve analysis: fresh output directory required")
    reference = metrics.metrics.reference_data(args.reference)
    reference["source_path"] = str(args.reference.resolve())
    rows, traces, _, original, proof = load_cases(args.collection, args.preflight, args.previous_result, reference)
    args.output_dir.mkdir(parents=True)
    result = dict(kind="PPO_training_geometry_by_validation_geometry_PhysX_nominal", status="completed_analysis",
        rows=rows, case_order=list(ORDER), formal_new_training=proof, reference_sha256=sha(args.reference),
        analyzer_sha256=sha(__file__), previous_result=str(args.previous_result.resolve()),
        preflight_root=str(args.preflight.resolve()), collection_root=str(args.collection.resolve()),
        original_baseline_metrics=original, all_initial_mechanical_states_equal=True,
        both_training_arms_same_initial_network_reward_DR_and_budget=True,
        both_validation_geometries_native_Coulomb_offsets_1mm_rest0=True,
        training_and_validation_backend="PhysX", cross_engine_result=False,
        visual_review_complete=False, task_success_claimed=False)
    (args.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True, layout="constrained")
    for axis, row in zip(axes, rows, strict=True):
        trace = traces[row["id"]]
        axis.plot(reference["position"][:, 0], reference["position"][:, 1], "--", color="gray", label="reference")
        axis.plot(trace["position"][:, 0], trace["position"][:, 1], color="#15765d", label="recorded PPO")
        axis.set(title=row["training_geometry"]+" trained -> "+row["validation_geometry"], xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("PhysX mu=.7 | matched K rewards, DR, seed and 12,288,000-transition budget | no pose alignment")
    figure.savefig(args.output_dir/"comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps([dict(id=row["id"], tracking=row["metrics"]["windows"]["core"]["tracking"],
        support=row["metrics"]["any_wheel_ground_support"]["full_contact"], failed=row["nominal_screen"]["failed_checks"]) for row in rows]))
    return result


if __name__ == "__main__":
    import run_cem_ground_contact_v2 as assets
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--preflight", type=Path, default=ROOT/"results/ppo_k_native_slices9_20261025_v1/evidence")
    p.add_argument("--previous-result", type=Path, default=ROOT/"results/physx_tron1_wheel_models_20261025_v1")
    p.add_argument("--reference", type=Path, default=assets.paired.DEFAULT_REFERENCE)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
