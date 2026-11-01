"""Sequential, guarded Isaac validation for the exact-friction CEM factorial.

Run on the previously authorized RTX host from the isolated oracle snapshot.
Each full-horizon episode is a new output directory; no existing run is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = Path("/data/zengxiong/workspaces")
SOURCE = ROOT / "source_overlay"
MOTION = WORKSPACES / "tron1-drift-turn-worst-load-20261014/motion.npz"
ASSET = WORKSPACES / "tron1-beyondmimic-20261002/outputs/training-asset/WF_TRON1A.usda"
MODEL = WORKSPACES / "tron1-sliding-analysis-20261014/mujoco_model.xml"
CONTRACT = WORKSPACES / "tron1-controller-comparison-20261015/nmpc_isaac_bridge_snapshot_v1/config/nominal_actuator_contract.json"
LIBRARY = WORKSPACES / "tron1-controller-comparison-20261015/nmpc_isaac_native_snapshot_v2/build_v1/libtron1_nmpc_batch.so"
PROFILES = {"side": ROOT / "config/car_drift_height_smooth_task.json",
            "no_side": ROOT / "config/car_drift_height_smooth_no_sideslip_task.json"}
GRID = (0.05, 0.10, 0.20, 0.30, 1.60)  # μ=0.70 has identical fixed/oracle predictor.


def case_key(mu: float) -> str:
    whole, fraction = divmod(round(mu * 1000), 1000)
    return f"mu{whole}p{fraction:03d}"


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
        raise RuntimeError("Selected GPU is not completely idle: " + json.dumps(audit))
    return audit


def run_case(profile: str, mu: float, gpu: int) -> dict:
    label = case_key(mu)
    output = ROOT / "isaac_oracle" / profile / label
    if output.exists():
        report = output / "report.json"
        if report.is_file() and json.loads(report.read_text()).get("status") == "completed":
            return {"profile": profile, "mu": mu, "status": "reused_completed", "report_sha256": digest(report)}
        raise FileExistsError(f"Preserving incomplete or failed run: {output}")
    # PhysX may leave a short utilization tail after its process exits.
    # Never launch on a non-idle GPU; also avoid treating that brief tail as
    # an experimental failure. A genuinely occupied GPU still stops the batch.
    for attempt in range(12):
        try:
            audit = gpu_guard(gpu)
            break
        except RuntimeError:
            if attempt == 11:
                raise
            time.sleep(5)
    logs = ROOT / "dispatch_oracle"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, str(ROOT / "scripts/run_nmpc_tracking_isaac.py"),
        "--source-root", str(SOURCE), "--motion", str(MOTION), "--asset-path", str(ASSET),
        "--mujoco-model", str(MODEL), "--contract", str(CONTRACT),
        "--reference-id", "drift_drl_traj4", "--task-profile", str(PROFILES[profile]),
        "--settings", str(ROOT / "config/nmpc_height_smooth_residual.json"),
        "--plant-mu", str(mu), "--oracle-plant-friction", "--prediction-backend", "native",
        "--native-library", str(LIBRARY), "--num-envs", "1", "--seed", "42",
        "--horizon-steps", "30", "--control-blocks", "10", "--candidates", "64",
        "--iterations", "3", "--elite-count", "8", "--output-dir", str(output),
        "--device", f"cuda:{gpu}", "--headless", "--kit_args",
        f"--/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false "
        f"--/physics/cudaDevice={gpu} --/physics/sceneMultiGPUMode=0"]
    path = logs / f"{profile}_{label}"
    launch = {"profile": profile, "plant_mu": mu, "prediction_mu": mu,
              "gpu_guard": audit, "command": command, "source_sha256": {
                  str(file): digest(file) for file in (ROOT / "scripts/run_nmpc_tracking.py",
                                                 ROOT / "scripts/run_nmpc_tracking_isaac.py",
                                                 PROFILES[profile])}}
    (path.with_suffix(".launch.json")).write_text(json.dumps(launch, indent=2) + "\n")
    with path.with_suffix(".log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    result = {"profile": profile, "mu": mu, "status": report.get("status", "missing_report"),
              "returncode": process.returncode, "report_sha256": digest(report_path) if report else None,
              "log_sha256": digest(path.with_suffix(".log"))}
    (path.with_suffix(".result.json")).write_text(json.dumps(result, indent=2) + "\n")
    if process.returncode != 0 or report.get("status") != "completed":
        raise RuntimeError("Isaac oracle run failed: " + json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--mu", nargs="+", type=float, default=list(GRID))
    args = parser.parse_args()
    if any(mu not in GRID for mu in args.mu) or len(set(args.mu)) != len(args.mu):
        parser.error("Use unique predeclared nonnominal grid values")
    for profile in args.profiles:
        for mu in args.mu:
            result = run_case(profile, mu, args.gpu_index)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
