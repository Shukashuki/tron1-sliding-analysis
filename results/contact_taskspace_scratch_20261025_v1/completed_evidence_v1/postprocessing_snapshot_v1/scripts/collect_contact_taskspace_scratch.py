"""Collect both complete scratch policies and four native records, no weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

COHORT = "tron1_contact_taskspace_scratch_K_M_400hz_v1"
INITIAL = "7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"
CASES = ("K_physx", "K_mujoco", "M_physx", "M_mujoco")
PARENT_SHA = "5a3b65b4abf7ef35b220f3ea84937a6d46a94dd789231b7157f840bdab3ecb7d"


def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def verify_completed(root):
    root = Path(root); driver = read(root/"driver_status.json"); status = read(root/"nominal_v1/status.json")
    assert driver["status"] == "both_formal_verified_nominal_validation_pending"
    assert status["status"] == "four_native_nominal_episodes_completed_require_analysis_and_render"
    assert all(r["status"] == "completed_strictly_verified" and r["returncode"] == 0 for r in driver["children"])
    assert [r["id"] for r in status["cases"]] == list(CASES)
    for variant in ("K", "M"):
        folder = root/("formal_"+variant); proof = read(root/("formal_"+variant+"_verification.json"))
        assert proof == driver["proofs"]["formal_"+variant] == status["paired_formal_proofs"][variant]
        assert proof["cohort_id"] == COHORT and proof["variant"] == variant and proof["run_kind"] == "formal"
        assert proof["transitions"] == 12288000 and proof["initial_policy_state_sha256"] == INITIAL
        assert proof["CPU_actor_output_shape"] == [1, 8] and proof["from_scratch"] and not proof["historical_checkpoint_used"]
        assert sha((folder/"cohort_completion_audit.json").read_bytes()) == proof["completion_audit_sha256"]
        audit = read(folder/"cohort_completion_audit.json")
        assert audit["initial_policy_state_sha256"] == INITIAL and audit["initial_checkpoint"] is None
        assert audit["from_scratch"] and not audit["historical_checkpoint_used"]
        assert audit["training_environment_steps"] == 12288000
        assert audit["exported_policy_sha256"] == proof["actor_sha256"] and audit["checkpoint_sha256"] == proof["checkpoint_sha256"]
        for key, name in [("manifest", "manifest.json"), ("policy_contract", "policy_contract.json"),
                         ("pre_export_runtime_audit", "pre_export_runtime_audit.json"),
                         ("export_precision_audit", "export_precision_audit.json"), ("startup_audit", "cohort_startup_audit.json")]:
            assert audit[key+"_sha256"] == sha((folder/name).read_bytes())
        manifest = read(folder/"manifest.json"); c = audit["cohort"]
        assert c == manifest["cohort"] and c["normalizer_initial_count"] == 0 and c["normalizer_final_count"] == 12288000
        assert c["optimizer_initial_state_empty"] and c["initial_checkpoint"] is None
        for phase, count in [("live_startup", 8), ("live_final", 24008)]:
            contact = c[phase]["contact"]
            assert contact["steps_started"] == contact["steps_completed"] == contact["unique_reward_windows"] == count
            assert contact["physics_samples"] == count*8 and contact["sampled_environment_substeps"] == count*8*512
            assert contact["device_finite_normal_force_health"] and contact["device_finite_reward_health"]
            assert c[phase]["task_space_tracking"]["status"] == "live_target_selection_verified"
    for row in status["cases"]:
        assert row["status"] == "completed_require_independent_analysis" and row["returncode"] == 0
        folder = root/"nominal_v1"/row["id"]
        meta = read(folder/"scratch_contact_evaluation_status.json"); report = read(folder/"report.json")
        assert meta == report["scratch_contact_evaluation"]
        assert sha((folder/"scratch_contact_evaluation_status.json").read_bytes()) == row["metadata_sha256"]
        proof = driver["proofs"]["formal_"+row["variant"]]
        assert meta["kind"] == "fresh_taskspace_contact_native_evaluation" and meta["cohort_id"] == COHORT
        assert meta["reward_variant"] == row["variant"] and meta["validation_backend"] == row["backend"]
        assert meta["normalized_actor_sha256"] == proof["actor_sha256"]
        assert meta["training_completion_audit_sha256"] == proof["completion_audit_sha256"]
        assert meta["training_transitions"] == 12288000 and meta["normalizer_initial_count"] == 0
        assert meta["native_states"] == 2793 and meta["policy_decisions"] == 349 and meta["evaluation_mu"] == .7
        assert not meta["smoke_only"] and meta["from_scratch"] and not meta["historical_checkpoint_used"]
        name = "physics_trajectory.npz" if row["backend"] == "physx" else "rollout.npz"
        assert sha((folder/name).read_bytes()) == meta["recording_sha256"]
    return driver, status


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    old = "completed_paired_warm_contact_refinement_four_native_episodes"
    assert source.count(old) == 1
    return source.replace(old, "completed_fresh_taskspace_K_M_four_native_episodes", 1)


def collect(root, helper):
    source = materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_fresh_scratch_source_and_record_collection", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<fresh_scratch_source_and_record_collection>", "exec"), namespace)
    namespace["verify_completed"] = verify_completed
    return namespace["collect"](root, helper)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workspace", type=Path, required=True); p.add_argument("--idle-audit-source", type=Path, required=True)
    args = p.parse_args(); collect(args.workspace, args.idle_audit_source)
