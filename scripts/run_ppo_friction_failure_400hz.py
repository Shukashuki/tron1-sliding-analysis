"""CPU-only17-point validation of the two completed PhysX-trained400Hz PPOs.

Run on the existing authorized host with its frozen source snapshots/assets.
No actor/checkpoint is downloaded, no training process is changed, and no GPU
is exposed. Nominal records are SHA-bound reuses, not fabricated new episodes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

GRID = (.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.)
ORDERED_GRID = (.15, .3, 1.2)+tuple(mu for mu in GRID if mu not in (.15, .3, 1.2, .7))
CHECKPOINTS = {"beta_on": "c508dc3ba50db81f56f538bf23c550aca12f3279ddd15749d60582a487d2d8c7",
    "beta_off": "fd9bb85554b7d25b31fbce57af63d33f73323e0530cf04990e95a7ac468f0688"}
ACTORS = {"beta_on": "ab3c486ef779deb0cff52bfb33415f5c3a779381740a2c12898506e4e80f08bc",
    "beta_off": "3e32748317244d4e1dc3f3df8fd6ec1052796fe1a44327ee7948c3601bbebdd5"}
SOURCE_PINS = {"scripts/eval_ppo_backend_400hz_mujoco.py": "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4",
    "scripts/eval_ppo_backend_400hz_beta_off_mujoco.py": "ab0fd9d4d8e0bffb11288a1f914bc36bb54ff5fee8513062ccd51f032e42fd16",
    "scripts/ppo_backend_400hz_contract.py": "b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e",
    "scripts/ppo_backend_400hz_beta_contract.py": "272d736abcabb04bb7ce0150857a211a8696cbc0e155e1d7bde25b3c635aac1b",
    "config/ppo_backend_400hz_comparison_v1.json": "c54cd530884cffa22811ca2232cb2588eba3582deefc91d4d9675e4fca7fe30f",
    "config/ppo_backend_400hz_beta_off_v1.json": "97a8a0b2d850bfc6da3903e85dbee5abcf6ed4e8c62769a31b21c00edfad7438"}
NOMINAL_PINS = {"beta_on": {"recording_sha256": "876a459afdcbcfff2282f59fc85b21da4adac7e2235875fceb38a4f0b70660ca",
    "report_sha256": "57b2a47935bde2f0db5eb1c3bd5f9dae1a19c26e11016f54f69c91e3dce5152d"},
    "beta_off": {"recording_sha256": "19c8781e4e94f012d017c9a52682126dc565188a3908084f3709f9ab29e72a60",
    "report_sha256": "09b93c9beabf4267b99a18fff5f8f99d2628bac66870fe0ac90a5f9e477dfc71"}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_status(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def source_ledger(snapshot):
    return {str(path.relative_to(snapshot)): sha256(path) for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts}


def layout(workspace):
    workspace = Path(workspace).resolve()
    on_root = workspace.with_name("tron1-ppo-backend-comparison-20261022")
    return {"snapshot": workspace/"source_snapshot_v4",
        "cohorts": {"beta_on": on_root/"pair_seed42_gpu4_v1/physx_formal", "beta_off": workspace/"pair_seed42_gpu6_v1/physx_formal"},
        "nominal": {"beta_on": on_root/"matrix_seed42_gpu4_v1/formal/physx/mujoco/mu0p700",
                    "beta_off": workspace/"single_physx_to_mujoco_beta_off_20261022_v1/evaluation"},
        "motion": workspace.with_name("tron1-drift-turn-worst-load-20261014")/"motion.npz",
        "model": workspace.with_name("tron1-sliding-analysis-20261014")/"mujoco_model.xml",
        "legacy": workspace.with_name("tron1-controller-comparison-20261015")/"controller_evaluation_source_v2/scripts/eval_tracking_mujoco.py"}


def cases(output, paths):
    result = []
    for mu in (.7,)+ORDERED_GRID:
        for label in ("beta_on", "beta_off"):
            reuse = mu == .7
            folder = paths["nominal"][label] if reuse else output/label/f"mu{mu:.3f}".replace(".", "p")/"evaluation"
            result.append({"case_id": f"ppo_{label}/mu{mu:.3f}".replace(".", "p"), "controller": "PPO", "condition": label,
                "mu": mu, "status": "pending", "directory": str(folder), "raw_file": "rollout.npz", "report_file": "report.json",
                "training_backend": "physx", "validation_backend": "native CPU MuJoCo", "reused_nominal": reuse,
                "checkpoint_sha256": CHECKPOINTS[label], "actor_sha256": ACTORS[label]})
    require(len(result) == 34 and len({c["case_id"] for c in result}) == 34, "Invalid17x2 plan")
    return result


def run(args):
    output = Path(args.output_dir).resolve(); paths = layout(args.workspace); snapshot = paths["snapshot"]
    require(not output.exists(), "Preserve existing evidence; fresh output directory required")
    require(isinstance(args.driver_sha256, str) and len(args.driver_sha256) == 64
            and set(args.driver_sha256) <= set("0123456789abcdef"), "Missing executed driver SHA binding")
    require(Path(args.python).resolve() == Path(sys.executable).resolve(), "Use the verified current Python interpreter")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU validation requires hidden CUDA devices")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(name) == "1", "Require one CPU thread per library: "+name)
    for label, directory in paths["cohorts"].items():
        require(directory.is_dir() and not output.is_relative_to(directory) and not directory.is_relative_to(output), "Evaluation/training overlap")
    require(snapshot.is_dir() and not output.is_relative_to(snapshot), "Evaluation/source overlap")
    frozen = source_ledger(snapshot)
    require(all(frozen.get(name) == digest for name, digest in SOURCE_PINS.items()), "Frozen400Hz evaluator source/protocol differs")
    sys.path.insert(0, str(snapshot/"scripts"))
    modules = {"beta_on": importlib.import_module("ppo_backend_400hz_contract"),
               "beta_off": importlib.import_module("ppo_backend_400hz_beta_contract")}
    protocols = {"beta_on": snapshot/"config/ppo_backend_400hz_comparison_v1.json", "beta_off": snapshot/"config/ppo_backend_400hz_beta_off_v1.json"}
    cohorts = {label: module.verify_cohort(paths["cohorts"][label], protocols[label]) for label, module in modules.items()}
    for label, cohort in cohorts.items():
        require(cohort["training_backend"] == "physx" and cohort["smoke_only"] is False
                and cohort["input_sha256"]["model_final.pt"] == CHECKPOINTS[label]
                and cohort["input_sha256"]["policy.pt"] == ACTORS[label], "Require exact completed nominal PhysX-trained actor")
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema_version": 1, "kind": "PhysX_trained_PPO400Hz_CPU_friction_failure_validation", "status": "running",
        "pid": os.getpid(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "driver_sha256": args.driver_sha256, "frozen_source_sha256": frozen, "friction_grid": list(GRID),
        "native_physics_dt_s": .0025, "policy_dt_s": .02, "cases": cases(output, paths),
        "new_training": False, "new_GPU_job": False, "original_training_or_queue_modified": False,
        "PPO_weights_downloaded": False, "automatic_retry": False, "concurrent_jobs": 1}
    path = output/"status.json"

    def save():
        status["counts"] = {"expected": 34, "completed": sum(c["status"] == "completed_verified" for c in status["cases"]),
            "failed": sum(c["status"] == "failed" for c in status["cases"]),
            "reused_nominal": sum(c["reused_nominal"] and c["status"] == "completed_verified" for c in status["cases"])}
        write_status(path, status)

    save()
    for case in status["cases"]:
        label = case["condition"]; module = modules[label]; folder = Path(case["directory"])
        try:
            require(source_ledger(snapshot) == frozen, "Frozen evaluator source changed")
            require(module.verify_cohort(paths["cohorts"][label], protocols[label])["input_sha256"] == cohorts[label]["input_sha256"], "Completed actor inputs changed")
            if not case["reused_nominal"]:
                script = "eval_ppo_backend_400hz_mujoco.py" if label == "beta_on" else "eval_ppo_backend_400hz_beta_off_mujoco.py"
                command = [str(args.python), "-u", str(snapshot/"scripts"/script), "--cohort-dir", str(paths["cohorts"][label]),
                    "--cohort-protocol", str(protocols[label]), "--legacy-evaluator", str(paths["legacy"]),
                    "--motion-file", str(paths["motion"]), "--model", str(paths["model"]), "--output-dir", str(folder),
                    "--evaluation-contact-friction", str(case["mu"])]
                folder.parent.mkdir(parents=True, exist_ok=False)
                case.update(status="running", command=command); save()
                with (folder.parent/"evaluation.log").open("x") as stream:
                    child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
                case["returncode"] = child.returncode
                require(child.returncode == 0, "Native CPU evaluation exited nonzero")
            proof = module.validate_output(folder, paths["cohorts"][label], protocols[label], validation_backend="native CPU MuJoCo")
            require(proof["checkpoint_sha256"] == CHECKPOINTS[label] and proof["evaluation_contact_friction"] == case["mu"]
                    and proof["native_states"] == 2793 and proof["policy_decisions"] == 349, "Full400Hz case identity differs")
            if case["reused_nominal"]:
                require(all(proof[name] == digest for name, digest in NOMINAL_PINS[label].items()), "Nominal reuse hashes differ")
            require(source_ledger(snapshot) == frozen, "Frozen source changed during native validation")
            case.update(status="completed_verified", verification=proof, recording_sha256=proof["recording_sha256"], report_sha256=proof["report_sha256"])
            save(); print(json.dumps({"case_id": case["case_id"], "status": case["status"], "counts": status["counts"]}), flush=True)
        except Exception as error:
            case.update(status="failed", error_type=type(error).__name__, error=str(error))
            save(); print(json.dumps({"case_id": case["case_id"], "status": "failed", "error": str(error)}), flush=True)
    status["status"] = "completed" if status["counts"]["completed"] == 34 else "completed_with_failures"
    status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
    return status


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--driver-sha256", required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
