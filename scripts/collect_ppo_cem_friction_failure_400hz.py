"""Create a fresh, immutable local snapshot manifest from completed/running sweeps.

Only completed, hash-verified recordings receive physical metrics downstream.
Mutable dispatcher status is not a source of fabricated completion or success.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import run_ppo_friction_failure_400hz as ppo
from ppo_backend_400hz_contract import read_json, require, sha256, write_json
from render_ppo_beta_physx_to_mujoco_400hz import DEFAULT_OFF, DEFAULT_ON
from run_cem_friction_failure_400hz import DEFAULT_NOMINAL
from summarize_ppo_backend_400hz_nominal import local_file, verify_files

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"outputs/ppo_cem_friction_failure_400hz_20261022_v1"


def stable_status(path):
    """A dispatcher may write between reads; never parse a mixed snapshot."""
    path = Path(path)
    for _ in range(5):
        before = path.read_bytes()
        try:
            value = json.loads(before)
        except json.JSONDecodeError:
            continue
        if before == path.read_bytes():
            require(isinstance(value, dict), "Expected object dispatcher status")
            return value, __import__("hashlib").sha256(before).hexdigest()
    raise RuntimeError("Dispatcher changed during snapshot; take a fresh snapshot later")


def ppo_cases(root):
    root = Path(root).resolve()
    manifest_file = local_file(root, "manifest.json")
    digest = sha256(manifest_file); manifest = read_json(manifest_file)
    require(manifest["schema_version"] == 1
        and manifest["kind"] == "verified_PhysX_trained_PPO400Hz_17mu_CPU_retrieval"
        and manifest["status"] == "completed"
        and manifest["counts"] == {"expected": 34, "completed": 34, "failed": 0, "reused_nominal": 2}
        and manifest["models_weights_or_credentials_included"] is False,
        "Require the genuine completed34-cell400Hz PPO CPU retrieval")
    ledger = verify_files(root, manifest)
    result = []
    require(len(manifest["cases"]) == 34 and len({(c["condition"], c["mu"]) for c in manifest["cases"]}) == 34,
            "Incomplete/duplicate PPO grid")
    require({(c["condition"], c["mu"]) for c in manifest["cases"]} ==
        {(label, mu) for label in ("beta_on", "beta_off") for mu in ppo.GRID}, "PPO17-point grid differs")
    for original in manifest["cases"]:
        case = copy.deepcopy(original); label = case["condition"]
        require(case["status"] == "completed_verified" and case["controller"] == "PPO"
            and case["training_backend"] == "physx" and case["validation_backend"] == "native CPU MuJoCo"
            and case["checkpoint_sha256"] == ppo.CHECKPOINTS[label] and case["actor_sha256"] == ppo.ACTORS[label],
            "Wrong completed400Hz actor/backend")
        if case["reused_nominal"]:
            require(case["mu"] == .7 and all(case[name] == value for name, value in ppo.NOMINAL_PINS[label].items()),
                    "Nominal PPO identity changed")
            folder = DEFAULT_ON/"cases/physx/mujoco" if label == "beta_on" else DEFAULT_OFF/"off"
        else:
            folder = (root/case["directory"]).resolve()
            require(folder.is_relative_to(root), "PPO input directory escaped retrieval")
            for name, key in (("rollout.npz", "recording_sha256"), ("report.json", "report_sha256")):
                relative = str((folder/name).relative_to(root))
                require(ledger.get(relative) == case[key] == sha256(folder/name), "PPO case/ledger hash binding differs")
        proof = case["verification"]
        require(proof["status"] == "verified" and proof["native_states"] == 2793 and proof["policy_decisions"] == 349
            and proof["evaluation_contact_friction"] == case["mu"] and proof["checkpoint_sha256"] == case["checkpoint_sha256"]
            and proof["recording_sha256"] == case["recording_sha256"] and proof["report_sha256"] == case["report_sha256"],
            "PPO strict remote verification proof differs")
        require(sha256(folder/"rollout.npz") == case["recording_sha256"] and sha256(folder/"report.json") == case["report_sha256"],
                "Completed PPO local source changed")
        case["directory"] = str(folder.resolve()); result.append(case)
    require(sha256(manifest_file) == digest, "PPO manifest changed during snapshot")
    return result, {"manifest_sha256": digest, "file_ledger": ledger}


def cem_cases(root):
    root = Path(root).resolve()
    metadata, digest = stable_status(root/"experiment.json")
    require(metadata["kind"] == "frozen400hz_matched_reward_CEM_friction_failure_validation"
        and metadata["full_grid_requested"] is True and metadata["mu_values_requested"] == list(ppo.GRID)
        and metadata["max_policy_steps"] is None and metadata["max_concurrent_local_CPU_episodes"] == 2
        and metadata["automatic_failed_job_retry"] is False, "Wrong or smoke CEM sweep scope")
    for name, value in metadata["source_sha256"].items():
        require(sha256(local_file(ROOT, name)) == value, "Frozen CEM source/configuration changed: "+name)
    jobs = metadata["jobs"]
    require(len(jobs) == 34 and {(j["condition"], j["mu"]) for j in jobs} ==
        {(label, mu) for label in ("beta_on", "beta_off") for mu in ppo.GRID}, "CEM grid incomplete/duplicate")
    result = []
    for original in jobs:
        case = copy.deepcopy(original)
        require(case["controller"] == "MPC-CEM", "Wrong controller in CEM sweep")
        if case["reused_nominal"]:
            require(case["mu"] == .7 and case["status"] == "reused_completed_nominal", "Mislabeled CEM nominal reuse")
            folder = DEFAULT_NOMINAL/case["condition"]
        else:
            require(Path(case["id"]).name == case["id"] and case["id"].startswith("cem_"), "Escaped CEM case identifier")
            folder = root/case["id"]
        case["directory_original"] = case["directory"]; case["directory"] = str(folder.resolve())
        if case["status"] in ("completed", "reused_completed_nominal"):
            require(case["native_samples"] == 2793 and case["completed_policy_steps"] == 349, "CEM case is not full-horizon")
            for name, key in (("rollout.npz", "recording_sha256"), ("report.json", "report_sha256"), ("solver_log.json", "solver_log_sha256")):
                require(sha256(folder/name) == case[key], "CEM completed input hash changed: "+name)
        elif case["status"].startswith("failed"):
            case["dispatcher_status"] = case["status"]; case["status"] = "failed"
        require(case["status"] in ("completed", "reused_completed_nominal", "pending", "running", "failed", "interrupted_preserved"),
                "Unknown CEM full-sweep status")
        result.append(case)
    return result, {"snapshot_sha256": digest, "dispatcher_status": metadata["status"],
        "source_sha256": metadata["source_sha256"], "completed_full_episode_count": sum(c["status"] in ("completed", "reused_completed_nominal") for c in result)}


def run(args):
    output = Path(args.output_file)
    require(not output.exists(), "Preserve existing snapshot; use a fresh manifest filename")
    first, ppo_proof = ppo_cases(args.ppo_root)
    second, cem_proof = cem_cases(args.cem_root)
    cells = first+second
    complete = sum(c["status"] in ("completed", "completed_verified", "reused_completed_nominal") for c in cells)
    result = {"schema_version": 1, "kind": "immutable_partial_PPO_CEM400Hz17mu_failure_manifest",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "cases": cells,
        "counts": {"expected": 68, "completed": complete, "pending_or_failed": 68-complete},
        "complete17x4_matrix": complete == 68, "new_simulation": False,
        "new_training": False, "PPO_provenance": ppo_proof, "CEM_provenance": cem_proof,
        "collector_sha256": sha256(__file__), "snapshot_does_not_auto_update": True}
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output, result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--ppo-root", type=Path, default=DEFAULT_ROOT/"ppo")
    result.add_argument("--cem-root", type=Path, default=DEFAULT_ROOT/"cem")
    result.add_argument("--output-file", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
