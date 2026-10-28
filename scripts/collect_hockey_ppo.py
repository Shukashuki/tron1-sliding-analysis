"""Retrieve completed hockey PPO evidence; put actors only in ignored outputs.

Uses the already authorized host and exact experiment directory. Does not
launch, retry, modify, or stop any remote simulation/training process.
"""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/data/zengxiong/workspaces/tron1-hockey-ppo-20261028-v1"
SSH = ["ssh", "-i", "/home/roy/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
       "zengxiong@14.103.233.29"]
PYTHON = "/data/zengxiong/.conda/envs/regrind/bin/python"


def remote_python(source):
    return subprocess.check_output(SSH+[shlex.join([PYTHON, "-c", source])])


def extract(data, output):
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            path = (output/member.name).resolve()
            if not member.isfile() or not path.is_relative_to(output.resolve()):
                raise ValueError("Unexpected archive entry")
        output.mkdir(parents=True)
        for member in members:
            path = output/member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--actors", type=Path, required=True)
    a = p.parse_args()
    if not a.actors.resolve().is_relative_to((ROOT/"outputs").resolve()):
        raise ValueError("Keep actor weights under ignored outputs")
    if a.output.exists() or a.actors.exists():
        raise FileExistsError("Use fresh destinations")
    status = json.loads(remote_python(f"from pathlib import Path; print((Path({REMOTE!r})/'comparison_v1/status.json').read_text())"))
    expected = {v+"_"+m for v in ("tracking", "support") for m in ("smoke", "formal", "eval")}
    if {r["id"] for r in status} != expected or any(r["status"] != "completed" or r["returncode"] for r in status):
        print(json.dumps([{k: r[k] for k in ("id", "status")} for r in status], indent=2))
        raise SystemExit("Comparison not complete; no evidence downloaded")
    evidence = ["launch.json", "comparison_v1/status.json"]
    actors = []
    for v in ("tracking", "support"):
        for m in ("smoke", "formal", "eval"):
            directory = f"comparison_v1/{v}_{m}/"
            evidence += [directory+"report.json", directory+"protocol.json"]
            if m == "eval":
                evidence += [directory+"final_rollout.npz", directory+"native_contact_buffers.npz"]
            else:
                evidence += [directory+"runner_config.json"]
        evidence.append(f"comparison_v1/{v}_formal.log")
        actors.append(f"comparison_v1/{v}_formal/actor_normalized.pt")
    def stream(files):
        source = ("from pathlib import Path\nimport sys,tarfile\n"
                  f"root=Path({REMOTE!r})\nfiles={files!r}\n"
                  "assert all((root/name).is_file() for name in files)\n"
                  "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:\n"
                  " for name in files: archive.add(root/name,arcname=name,recursive=False)\n")
        return remote_python(source)
    extract(stream(evidence), a.output)
    extract(stream(actors), a.actors)
    print(json.dumps(dict(evidence_files=len(evidence), ignored_actor_files=len(actors),
                         output=str(a.output), actors=str(a.actors))))


if __name__ == "__main__":
    main()
