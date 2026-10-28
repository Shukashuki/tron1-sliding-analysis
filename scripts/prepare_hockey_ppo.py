"""Package the direct hockey PPO experiment; never include weights/assets/video."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.archive.exists():
        raise FileExistsError("Use a new snapshot/archive")
    files = {name: ROOT/"scripts"/name for name in (
        "train_hockey_ppo.py", "hockey_ppo_task.py", "dispatch_hockey_ppo.py",
        "physx_mjlab_recipe.py", "height_smoothness.py", "physx_reference_math.py")}
    reference = ROOT/"results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2"
    files.update({"protocol.json": ROOT/"config/hockey_ppo_learning_v1.json",
                  "motion.npz": reference/"motion.npz", "command_report.json": reference/"report.json"})
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    cfg = json.loads(files["protocol.json"].read_text())
    if sha(files["motion.npz"]) != cfg["reference_sha256"]:
        raise ValueError("Wrong reference")
    a.output.mkdir(parents=True)
    rows = []
    for name, source in files.items():
        target = a.output/name
        shutil.copyfile(source, target)
        rows.append(dict(path=name, bytes=target.stat().st_size, sha256=sha(target)))
    manifest = dict(kind="direct_hockey_ppo_snapshot_v1", files=rows, file_count=len(rows),
        total_bytes=sum(r["bytes"] for r in rows),
        scope="2 x 32-env/2-iteration smoke, then 2 x 512-env/1000-iteration PPO and 2 x single-env/139-step nominal validation; serial idle RTX only",
        excludes=["models", "weights", "credentials", "raw video", "SMPL assets"],
        reference="previously approved GMR posture plus designed stop path; unchanged v2 bytes")
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with tarfile.open(a.archive, "w:gz") as archive:
        for path in sorted(a.output.iterdir()):
            archive.add(path, arcname=path.name, recursive=False)
    print(json.dumps(dict(**manifest, archive=str(a.archive), archive_bytes=a.archive.stat().st_size,
                         archive_sha256=sha(a.archive)), indent=2))


if __name__ == "__main__":
    main()
