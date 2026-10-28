"""Bounded, serial speed-study queue; select only a freshly audited idle RTX."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "prior-launch", "audit-helper", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--maximum-idle-wait-s", type=float, default=21600.)
    a = p.parse_args()
    if a.output.exists() or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Require fresh output and unmasked devices")
    if not 0 <= a.maximum_idle_wait_s <= 21600:
        raise ValueError("Idle wait must be bounded by six hours")
    if sha(a.audit_helper) != "6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866":
        raise ValueError("GPU audit helper changed")
    spec = importlib.util.spec_from_file_location("gpu_audit", a.audit_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.GPU_ORDER = (3, 5, 4, 0, 1, 2, 6, 7)
    read = lambda path: json.loads(Path(path).read_text())
    manifest = read(a.bundle/"manifest.json")
    for row in manifest["files"]:
        path = (a.bundle/row["path"]).resolve()
        if not path.is_relative_to(a.bundle.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Payload changed")
    plan = read(a.bundle/"plan.json")
    boundary = plan["kind"] == "hockey_friction_boundary_ablation_v1"
    budget = (2, 24576000, 66) if boundary else (18, 221184000, 594)
    if (plan["formal_training_runs"], plan["total_formal_transitions"], plan["total_evaluation_episodes"]) != budget:
        raise ValueError("Unreviewed study budget")
    if boundary:
        if len(plan['cells']) != 2:
            raise ValueError('Boundary comparison must have exactly two cells')
        protocols = [read(a.bundle/c['protocol']) for c in plan['cells']]
        if [c['friction_boundary']['weight'] for c in protocols] != [0., .5]:
            raise ValueError('Expected boundary-off and boundary-on weights')
        import copy
        compare = copy.deepcopy(protocols)
        for c in compare:
            c['friction_boundary'].pop('weight')
        if compare[0] != compare[1] or any(c['method'] != 'hockey' or c['speed_scale'] != 1.5 or c['seed'] != 42 for c in protocols):
            raise ValueError('Boundary arms must differ only in their reward weight')
    old = read(a.prior_launch)["argv"]
    get = lambda key: old[old.index("--"+key)+1]
    paths = {key: Path(get(key)) for key in ("asset", "tracking-root", "beyondmimic-root")}
    first_cfg = read(a.bundle/plan["cells"][0]["protocol"])
    if sha(paths["asset"]) != first_cfg["asset_sha256"] or sha(paths["tracking-root"]/"training/tron1_tracking.py") != first_cfg["tracking_sha256"]:
        raise ValueError("Existing runtime assets or tracking source changed")
    provenance = read(a.bundle/"upstream_provenance.json")
    for key, digest in provenance.items():
        if key.endswith("scripts/train_tracking.py"):
            continue  # Runner settings are recorded in each protocol, not imported remotely.
        prefix = ("upstream/whole_body_tracking/" if key.startswith("upstream/whole_body_tracking/")
                  else "upstream/tron1-learning-from-human/")
        location = paths["beyondmimic-root" if "whole_body_tracking" in prefix else "tracking-root"]/key[len(prefix):]
        if sha(location) != digest:
            raise ValueError("Historical upstream implementation changed: "+key)
    if importlib.metadata.version("rsl-rl-lib") != "3.0.1":
        raise ValueError("Wrong PPO runtime")
    if shutil.disk_usage(a.output.parent).free < 10*1024**3:
        raise ValueError("Need at least 10 GiB for immutable checkpoints and full trial recordings")
    a.output.mkdir(parents=True)
    status = dict(kind=plan["kind"], status="starting", completed=[], current=None,
                  idle_wait_s=0., manifest_sha256=sha(a.bundle/"manifest.json"))
    status_path = a.output/"status.json"
    def save():
        status["updated_unix_s"] = time.time(); write(status_path, status)
    save()
    try:
        # Obtain speed learnability results for seed42 before seed replications.
        cells = sorted(plan["cells"], key=lambda c: (c["seed"], c["speed_scale"], c["method"]))
        for cell in cells:
            cell_dir = a.output/cell["id"]
            cell_dir.mkdir()
            for mode in ("smoke", "formal", "eval-suite", "analyze"):
                stage = cell["id"]+"/"+mode
                status.update(status="preparing", current=dict(id=stage)); save()
                if mode != "analyze":
                    while True:
                        gpu = next((g for g in helper.query_gpus() if g["idle"]), None)
                        if gpu is not None:
                            break
                        if status["idle_wait_s"] >= a.maximum_idle_wait_s:
                            raise RuntimeError("Idle GPU wait budget exhausted; remaining cells were not started")
                        status["status"] = "waiting_for_completely_idle_RTX"; save()
                        start = time.monotonic()
                        time.sleep(min(60., a.maximum_idle_wait_s-status["idle_wait_s"]))
                        status["idle_wait_s"] += time.monotonic()-start
                    argv = [old[0], "-B", "-u", str(a.bundle/"train_hockey_ppo.py"),
                        "--protocol", str(a.bundle/cell["protocol"]), "--motion", str(a.bundle/cell["reference"]/"motion.npz"),
                        "--mode", mode, "--variant", "tracking", "--device", "cuda:"+str(gpu["physical_index"]),
                        "--output-dir", str(cell_dir/mode)]
                    for key, value in paths.items():
                        argv += ["--"+key, str(value)]
                    if mode == "eval-suite":
                        argv += ["--actor", str(cell_dir/"formal/actor_normalized.pt")]
                    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
                else:
                    gpu = None
                    argv = [old[0], "-B", "-u", str(a.bundle/"analyze_hockey_speed_suite.py"),
                        "--bundle", str(a.bundle), "--protocol", str(a.bundle/cell["protocol"]),
                        "--command-dir", str(a.bundle/cell["reference"]), "--training-dir", str(cell_dir/"formal"),
                        "--suite", str(cell_dir/"eval-suite"), "--actor", str(cell_dir/"formal/actor_normalized.pt"),
                        "--output", str(cell_dir/"analyze")]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
                with (cell_dir/(mode+".log")).open("xb") as log:
                    child = subprocess.Popen(argv, cwd=a.output, stdout=log, stderr=subprocess.STDOUT, env=env)
                row = dict(id=stage, pid=child.pid, identity=helper.process_snapshot(child.pid), argv=argv,
                           gpu_before_launch=gpu, started_unix_s=time.time())
                status.update(status="running", current=row); save()
                code = child.wait()
                row.update(returncode=code, finished_unix_s=time.time())
                if code:
                    raise RuntimeError(stage+" failed; no automatic retry or subsequent launches")
                report = read(cell_dir/mode/"report.json")
                if mode in ("smoke", "formal") and report.get("transitions") != (1536 if mode == "smoke" else 12288000):
                    raise ValueError("Training budget accounting mismatch")
                if mode == "eval-suite" and len(report.get("cases", [])) != 33:
                    raise ValueError("Incomplete evaluation suite")
                if mode == "analyze" and report.get("perturbation_trials") != 32:
                    raise ValueError("Incomplete independent success evaluation")
                row["report_sha256"] = sha(cell_dir/mode/"report.json")
                status["completed"].append(row); status["current"] = None; save()
        status["status"] = "completed_all_2_cells" if boundary else "completed_all_18_cells"; save()
    except BaseException as exc:
        status.update(status="stopped", error=f"{type(exc).__name__}: {exc}"); save()
        raise


if __name__ == "__main__":
    main()
