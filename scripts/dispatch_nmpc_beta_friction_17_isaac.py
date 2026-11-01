"""Guarded, resumable Isaac extension of the locked CEM factorial to 17 μ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time


JOB_ROOT = Path(__file__).resolve().parent
WORKSPACES = Path("/data/zengxiong/workspaces")
BASE = WORKSPACES / "tron1-controller-comparison-20261015/nmpc_oracle_friction_grid_snapshot_v1"
SOURCE = BASE / "source_overlay"
RUNNER = BASE / "scripts/run_nmpc_tracking_isaac.py"
MOTION = WORKSPACES / "tron1-drift-turn-worst-load-20261014/motion.npz"
ASSET = WORKSPACES / "tron1-beyondmimic-20261002/outputs/training-asset/WF_TRON1A.usda"
MODEL = WORKSPACES / "tron1-sliding-analysis-20261014/mujoco_model.xml"
CONTRACT = WORKSPACES / "tron1-controller-comparison-20261015/nmpc_isaac_bridge_snapshot_v1/config/nominal_actuator_contract.json"
LIBRARY = WORKSPACES / "tron1-controller-comparison-20261015/nmpc_isaac_native_snapshot_v2/build_v1/libtron1_nmpc_batch.so"
SETTINGS = BASE / "config/nmpc_height_smooth_residual.json"
BRANCHES = {
    "side_oracle": ("side", "oracle", BASE / "config/car_drift_height_smooth_task.json"),
    "no_side_fixed": ("no_side", "fixed", BASE / "config/car_drift_height_smooth_no_sideslip_task.json"),
    "no_side_oracle": ("no_side", "oracle", BASE / "config/car_drift_height_smooth_no_sideslip_task.json"),
}
GRID = (.01, .03, .15, .45, .6, .8, 1., 1.2, 1.4, 1.8, 2.)


def case_name(mu: float) -> str:
    return f"mu{mu:.3f}".replace(".", "p")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def gpu_guard(index: int) -> dict:
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    lines = [line.strip() for line in info.splitlines() if line.strip()]
    if not 0 <= index < len(lines):
        raise ValueError(f"GPU index {index} not found")
    uuid, memory, utilization = [part.strip() for part in lines[index].split(",")]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    attached = [line.strip() for line in apps.splitlines() if line.strip().startswith(uuid + ",")]
    audit = {"gpu_index": index, "gpu_uuid": uuid, "memory_mib": float(memory),
             "utilization_percent": float(utilization), "attached_compute_processes": attached,
             "checked_utc": datetime.now(timezone.utc).isoformat()}
    if audit["memory_mib"] >= 64 or audit["utilization_percent"] != 0 or attached:
        raise RuntimeError("Selected GPU is not fully idle: " + json.dumps(audit))
    return audit


def run_case(branch: str, mu: float, gpu: int) -> dict:
    profile, mode, task = BRANCHES[branch]
    output = JOB_ROOT / "isaac_17" / profile / mode / case_name(mu)
    report_path = output / "report.json"
    if output.exists():
        if report_path.is_file() and json.loads(report_path.read_text()).get("status") == "completed":
            return {"branch": branch, "mu": mu, "status": "reused_completed",
                    "report_sha256": digest(report_path)}
        raise FileExistsError(f"Preserving incomplete/failed episode: {output}")
    for path in (RUNNER, SOURCE / "scripts/eval_tracking_mujoco.py", MOTION, ASSET, MODEL,
                 CONTRACT, LIBRARY, SETTINGS, task):
        if not path.is_file():
            raise FileNotFoundError(path)
    for attempt in range(12):
        try:
            audit = gpu_guard(gpu)
            break
        except RuntimeError:
            if attempt == 11:
                raise
            time.sleep(5)
    logs = JOB_ROOT / "dispatch"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{branch}_{case_name(mu)}.log"
    launch_path = logs / f"{branch}_{case_name(mu)}.launch.json"
    result_path = logs / f"{branch}_{case_name(mu)}.result.json"
    command = [sys.executable, str(RUNNER), "--source-root", str(SOURCE),
               "--motion", str(MOTION), "--asset-path", str(ASSET),
               "--mujoco-model", str(MODEL), "--contract", str(CONTRACT),
               "--reference-id", "drift_drl_traj4", "--task-profile", str(task),
               "--settings", str(SETTINGS), "--plant-mu", str(mu),
               "--prediction-backend", "native", "--native-library", str(LIBRARY),
               "--num-envs", "1", "--seed", "42", "--horizon-steps", "30",
               "--control-blocks", "10", "--candidates", "64", "--iterations", "3",
               "--elite-count", "8", "--output-dir", str(output),
               "--device", f"cuda:{gpu}", "--headless", "--kit_args",
               f"--/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false "
               f"--/physics/cudaDevice={gpu} --/physics/sceneMultiGPUMode=0"]
    if mode == "oracle":
        command.append("--oracle-plant-friction")
    launch = {"branch": branch, "plant_mu": mu, "prediction_mu": mu if mode == "oracle" else .7,
              "gpu_guard": audit, "command": command,
              "inputs_sha256": {str(path): digest(path) for path in
                                (RUNNER, MOTION, ASSET, MODEL, CONTRACT, LIBRARY, SETTINGS, task)}}
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    with log_path.open("w") as log:
        process = subprocess.run(command, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, check=False)
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    result = {"branch": branch, "mu": mu, "status": report.get("status", "missing_report"),
              "returncode": process.returncode, "report_sha256": digest(report_path) if report else None,
              "log_sha256": digest(log_path)}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if process.returncode or report.get("status") != "completed":
        raise RuntimeError("Isaac 17-point case failed: " + json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--branch", choices=tuple(BRANCHES) + ("all",), required=True)
    parser.add_argument("--mu", nargs="+", type=float, default=list(GRID))
    args = parser.parse_args()
    if len(set(args.mu)) != len(args.mu) or any(mu not in GRID for mu in args.mu):
        parser.error("Use unique values from the eleven missing legacy-grid coefficients")
    branches = tuple(BRANCHES) if args.branch == "all" else (args.branch,)
    for branch in branches:
        for mu in args.mu:
            print(json.dumps(run_case(branch, mu, args.gpu_index)), flush=True)


if __name__ == "__main__":
    main()
