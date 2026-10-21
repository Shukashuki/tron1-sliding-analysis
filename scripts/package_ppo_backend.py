"""Package only an allowlist of PPO adapter source, tests and configuration."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("train_ppo_backend", "mjlab_tron1_env", "smoke_mjlab_tron1",
           "preflight_mjlab_runtime", "mujoco_ppo_env", "mujoco_ppo_native",
           "physx_mjlab_recipe", "classical_tracking", "nmpc_tracking_objective",
           "nmpc_batch_objective", "height_smoothness", "package_ppo_backend", "run_ppo_backend_pair")
TESTS = ("test_train_ppo_backend", "test_mjlab_tron1_env", "test_physx_mjlab_recipe",
         "test_mjlab_runtime_preflight", "test_mujoco_ppo_env", "test_mujoco_ppo_native", "test_run_ppo_backend_pair")


def package(output):
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Never overwrite a snapshot")
    paths = ([ROOT/"scripts"/(s+".py") for s in SCRIPTS] +
             [ROOT/"tests"/(s+".py") for s in TESTS] +
             [ROOT/"config/nominal_actuator_contract.json", ROOT/"config/car_drift_height_smooth_task.json",
              ROOT/"config/ppo_mjlab_backend_comparison_v1.json",
              ROOT/"native/ppo_batch.cpp"] +
             sorted((ROOT/"upstream/tron1-learning-from-human/training").glob("*.py")) +
             [ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py",
              ROOT/"upstream/tron1-learning-from-human/scripts/eval_tracking.py"])
    paths.append(ROOT/"upstream/tron1-learning-from-human/scripts/train_tracking.py")
    # Include upstream license when supplied, without broad directory archiving.
    for name in ("LICENSE", "LICENCE", "LICENSE.md"):
        p = ROOT/"upstream/tron1-learning-from-human"/name
        if p.is_file():
            paths.append(p)
    entries = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        entries.append({"path": str(p.relative_to(ROOT)), "size": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {"schema_version": 1, "contents": "pure source/config/tests; no assets, weights, binaries or credentials",
                "files": entries, "total_files": len(entries),
                "upstream_mjlab_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT/"upstream/mjlab"), "rev-parse", "HEAD"], text=True).strip(),
                "upstream_mjlab_distribution": "clone public submodule separately at recorded commit"}
    rendered = (json.dumps(manifest, indent=2)+"\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for p, entry in zip(paths, entries):
            archive.add(p, arcname=entry["path"], recursive=False)
        info = tarfile.TarInfo("snapshot_manifest.json")
        info.size = len(rendered)
        archive.addfile(info, io.BytesIO(rendered))
    with manifest_path.open("xb") as stream:
        stream.write(rendered)
    return {"archive": str(output), "manifest": str(manifest_path),
            "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_files": len(entries), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))
