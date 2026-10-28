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
    p.add_argument("--output", type=Path)
    p.add_argument("--actors", type=Path)
    p.add_argument("--status-only", action="store_true")
    a = p.parse_args()
    # The original queue was intentionally stopped for a sampler audit. Manual
    # continuations have their own launch identities; its old status.json alone
    # is no longer authoritative. Require each actual process to have ended,
    # plus the expected terminal report and artifact hashes. Unknown OS exit
    # codes remain unknown; a completed training artifact does not imply rc=0.
    probe = f'''from pathlib import Path
import hashlib,json
root=Path({REMOTE!r})
old={{r['id']:r for r in json.loads((root/'comparison_v1/status.json').read_text())}}
rows=[]
for variant in ('tracking','support'):
 for mode in ('smoke','formal','eval'):
  name=variant+'_'+mode
  launch=root/(name+'.launch.json')
  row=json.loads(launch.read_text()) if launch.exists() else old.get(name)
  report_path=root/'comparison_v1'/name/'report.json'
  if row is None:
   rows.append(dict(id=name,status='pending',live_process=None));continue
  proc=Path('/proc')/str(row['pid']);live=False
  if proc.exists():
   raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
   live=int(fields[19])==row['identity']['start_ticks'] and fields[0]!='Z'
  if not report_path.exists():
   rows.append(dict(id=name,status='running' if live else 'missing_terminal_report',live_process=live,pid=row['pid']));continue
  report=json.loads(report_path.read_text())
  expected='completed_requires_independent_qualification' if mode=='eval' else 'training_completed_not_task_success'
  valid=report.get('status')==expected
  if valid:
   files=report['artifacts'] if mode=='eval' else {{'actor_normalized.pt':report['actor_sha256'],'model_final.pt':report['checkpoint_sha256']}}
   valid=all(hashlib.sha256((report_path.parent/f).read_bytes()).hexdigest()==digest for f,digest in files.items())
   valid=valid and (report['steps']==139 if mode=='eval' else report['transitions']==(1536 if mode=='smoke' else 12288000))
  rows.append(dict(id=name,status='running' if live else 'completed' if valid else 'incomplete_or_failed',
     live_process=live,pid=row['pid'],recorded_exit_code=row.get('returncode'),
     completion_basis='terminal report and artifact hashes, original process no longer live'))
print(json.dumps(rows))
'''
    status = json.loads(remote_python(probe))
    if a.status_only:
        print(json.dumps(status, indent=2)); return
    if a.output is None or a.actors is None:
        p.error("--output and --actors are required for collection")
    if not a.actors.resolve().is_relative_to((ROOT/"outputs").resolve()):
        raise ValueError("Keep actor weights under ignored outputs")
    if a.output.exists() or a.actors.exists():
        raise FileExistsError("Use fresh destinations")
    expected = {v+"_"+m for v in ("tracking", "support") for m in ("smoke", "formal", "eval")}
    if {r["id"] for r in status} != expected or any(r["status"] != "completed" for r in status):
        print(json.dumps([{k: r[k] for k in ("id", "status")} for r in status], indent=2))
        raise SystemExit("Comparison not complete; no evidence downloaded")
    evidence = ["launch.json", "comparison_v1/status.json", "sampling_diagnostic_queue_stop.json",
                "tracking_eval.launch.json", "support_formal.launch.json", "support_eval.launch.json"]
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
    (a.output/"verified_completion_status.json").write_text(json.dumps(status, indent=2)+"\n")
    extract(stream(actors), a.actors)
    print(json.dumps(dict(evidence_files=len(evidence), ignored_actor_files=len(actors),
                         output=str(a.output), actors=str(a.actors))))


if __name__ == "__main__":
    main()
