"""Resumable MuJoCo-only extension of the locked CEM factorial to 17 μ values."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/nmpc_beta_friction_17_mujoco_20261017_v1"
MOTION = ROOT.parent / "tron1-learning-from-human/results/2026-10-14-car-drift/matched-friction-mu0p70/motion.npz"
MODEL = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
LIBRARY = ROOT / "outputs/native_build_20261015_v1/libtron1_nmpc_batch.so"
SETTINGS = ROOT / "config/nmpc_height_smooth_residual.json"
BRANCHES = {
    "side_oracle": ("side", "oracle", ROOT / "config/car_drift_height_smooth_task.json",
                    ROOT / "outputs/car_failure_sideslip_20261015/raw/mu0p180/isaac/policy_contract.json"),
    "no_side_fixed": ("no_side", "fixed", ROOT / "config/car_drift_height_smooth_no_sideslip_task.json",
                      ROOT / "outputs/controller_primary_ppo_mu070_20261015_v1/mu0p700/isaac/policy_contract.json"),
    "no_side_oracle": ("no_side", "oracle", ROOT / "config/car_drift_height_smooth_no_sideslip_task.json",
                       ROOT / "outputs/controller_primary_ppo_mu070_20261015_v1/mu0p700/isaac/policy_contract.json"),
}
GRID = (.01, .03, .15, .45, .6, .8, 1., 1.2, 1.4, 1.8, 2.)


def case_name(mu: float) -> str:
    return f"mu{mu:.3f}".replace(".", "p")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run_case(branch: str, mu: float) -> dict:
    profile, mode, task, contract = BRANCHES[branch]
    output = OUT / profile / mode / case_name(mu)
    report_path = output / "report.json"
    if output.exists():
        if report_path.is_file() and json.loads(report_path.read_text()).get("status") == "completed":
            return {"branch": branch, "mu": mu, "status": "reused_completed",
                    "report_sha256": digest(report_path)}
        raise FileExistsError(f"Preserving incomplete/failed episode: {output}")
    logs = OUT / "dispatch"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{branch}_{case_name(mu)}.log"
    launch_path = logs / f"{branch}_{case_name(mu)}.launch.json"
    result_path = logs / f"{branch}_{case_name(mu)}.result.json"
    command = [sys.executable, str(ROOT / "scripts/run_nmpc_tracking.py"),
               "--model", str(MODEL), "--motion", str(MOTION), "--contract", str(contract),
               "--task-profile", str(task), "--settings", str(SETTINGS),
               "--plant-mu", str(mu), "--prediction-backend", "native",
               "--native-library", str(LIBRARY), "--horizon-steps", "30",
               "--control-blocks", "10", "--candidates", "64", "--iterations", "3",
               "--elite-count", "8", "--output-dir", str(output)]
    if mode == "oracle":
        command.append("--oracle-plant-friction")
    for path in (MOTION, MODEL, LIBRARY, task, contract, SETTINGS):
        if not path.is_file():
            raise FileNotFoundError(path)
    launch = {"branch": branch, "plant_mu": mu, "prediction_mu": mu if mode == "oracle" else .7,
              "command": command, "inputs_sha256": {str(path): digest(path) for path in
                                                 (MOTION, MODEL, LIBRARY, task, contract, SETTINGS)}}
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    with log_path.open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    result = {"branch": branch, "mu": mu, "status": report.get("status", "missing_report"),
              "returncode": process.returncode, "report_sha256": digest(report_path) if report else None,
              "log_sha256": digest(log_path)}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if process.returncode or report.get("status") != "completed":
        raise RuntimeError("MuJoCo 17-point case failed: " + json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--mu", nargs="+", type=float, default=list(GRID))
    args = parser.parse_args()
    if len(set(args.mu)) != len(args.mu) or any(mu not in GRID for mu in args.mu):
        parser.error("Use unique values from the eleven missing legacy-grid coefficients")
    for mu in args.mu:
        print(json.dumps(run_case(args.branch, mu)), flush=True)


if __name__ == "__main__":
    main()
