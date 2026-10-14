"""Run frozen-policy matched-mu Isaac/MuJoCo evaluations and top-down XY grid.

The currently audited source evaluators support equal static/dynamic friction
in [0.01, 1.0]. Every coefficient gets one seed-42 environment per engine.
This entrypoint refuses incomplete training and never overwrites prior cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_name(mu: float) -> str:
    if not math.isfinite(mu) or not .01 <= mu <= 1.:
        raise ValueError("Current audited fixed-friction evaluators require 0.01 <= mu <= 1.0")
    return f"mu{mu:.3f}".replace(".", "p")


def launch(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run(args: argparse.Namespace) -> Path:
    source = args.source_root.resolve()
    target = args.output_dir.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Preserving nonempty validation output: {target}")
    required = [args.motion_file, args.reference_file, args.asset_path, args.mujoco_model,
                args.checkpoint, args.runner_config, args.task_profile,
                args.checkpoint.parent / "manifest.json", args.checkpoint.parent / "run_status.json",
                source / "scripts/eval_tracking_friction_isaac.py",
                source / "scripts/eval_tracking_mujoco.py"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    status = json.loads((args.checkpoint.parent / "run_status.json").read_text())
    training = json.loads((args.checkpoint.parent / "manifest.json").read_text())
    task = json.loads(args.task_profile.read_text())
    if (status.get("status") != "training_completed" or training.get("task_profile") != task
            or training.get("motion_sha256") != sha(args.motion_file)
            or training.get("asset_sha256") != sha(args.asset_path)
            or training.get("recipe", {}).get("friction_static") != [0.3, 1.6]
            or training.get("recipe", {}).get("friction_dynamic") != [0.3, 1.2]):
        raise ValueError("Checkpoint lacks completed BeyondMimic-style training provenance")
    values = list(args.mu)
    if not values or len({case_name(mu) for mu in values}) != len(values):
        raise ValueError("Friction grid must have unique, supported values")
    target.mkdir(parents=True, exist_ok=True)
    cases = []
    for mu in values:
        name = case_name(mu)
        case = target / name
        ia, mj = case / "isaac", case / "mujoco"
        print(f"VALIDATING {name}: Isaac then MuJoCo", flush=True)
        launch([sys.executable, str(source / "scripts/eval_tracking_friction_isaac.py"),
                "--motion-file", str(args.motion_file.resolve()),
                "--asset-path", str(args.asset_path.resolve()),
                "--checkpoint", str(args.checkpoint.resolve()),
                "--runner-config", str(args.runner_config.resolve()),
                "--task-profile", str(args.task_profile.resolve()),
                "--evaluation-contact-friction", str(mu), "--output-dir", str(ia),
                "--num-envs", "1", "--seed", "42", "--record-physics",
                "--verify-unmasked-device", "--device", args.device, "--headless"], source)
        launch([sys.executable, str(source / "scripts/eval_tracking_mujoco.py"),
                "--policy", str(ia / "actor_normalized.pt"),
                "--contract", str(ia / "policy_contract.json"),
                "--motion-file", str(args.motion_file.resolve()),
                "--model", str(args.mujoco_model.resolve()), "--output-dir", str(mj),
                "--no-visual-mesh", "--verify-isaac-trajectory", str(ia / "trajectory.npz")], source)
        cases.append({"case": name, "mu": mu,
                      "isaac_physics": str(ia / "physics_trajectory.npz"),
                      "isaac_report": str(ia / "report.json"),
                      "mujoco_rollout": str(mj / "rollout.npz"),
                      "mujoco_report": str(mj / "report.json")})
    manifest_path = target / "grid_manifest.json"
    manifest = {"schema_version": 1, "kind": "frozen_new_policy_fixed_mu_validation_grid",
                "policy_training_friction_randomization": True,
                "training_manifest": str((args.checkpoint.parent / "manifest.json").resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "reference_file": str(args.reference_file.resolve()),
                "motion_file": str(args.motion_file.resolve()),
                "source_evaluator_sha256": {
                    "isaac": sha(source / "scripts/eval_tracking_friction_isaac.py"),
                    "mujoco": sha(source / "scripts/eval_tracking_mujoco.py")},
                "cases": cases}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    launch([sys.executable, str(ROOT / "scripts/analyze_friction_grid.py"),
            "--manifest", str(manifest_path), "--output-dir", str(target / "analysis")], ROOT)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--asset-path", required=True, type=Path)
    parser.add_argument("--mujoco-model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--runner-config", required=True, type=Path)
    parser.add_argument("--task-profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mu", nargs="+", type=float, default=[.3, .45, .6, .7, .8, 1.0])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": "validated_and_rendered", "manifest": str(result)}))


if __name__ == "__main__":
    main()
