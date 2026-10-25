"""Collect only a completed nonlinear training and its three nominal episodes.

Source/config/audits/logs and explicitly named simulation-recording NPZ files
are allowed. Neural weights, model assets, credentials and native libraries
are excluded. Refuse live owned jobs or a partial/mixed-policy comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import time

COHORT = "tron1_ppo_k_nonlinear_tire_400hz_v1"
CASES = ("physx_nonlinear", "physx_native", "mujoco_native")


def sha(data): return hashlib.sha256(data).hexdigest()


def read(path): return json.loads(path.read_text())


def verify_completed(root):
    driver=read(root/"driver_status.json");status=read(root/"nominal_v1/status.json")
    assert driver["status"]=="formal_completed_verified_nominal_validation_pending"
    assert status["status"]=="three_nominal_episodes_completed_require_analysis_and_render"
    proof=read(root/"formal_v1_strict_verification.json")
    assert proof==driver["formal_proof"]==status["formal_training_proof"]
    assert proof["cohort_id"]==COHORT and proof["formal_or_smoke"]=="formal"
    assert proof["transitions"]==12288000 and proof["CPU_actor_output_shape"]==[1,8]
    assert len(status["cases"])==3 and [x["id"] for x in status["cases"]]==list(CASES)
    assert all(x["status"]=="completed_strictly_verified" and x["returncode"]==0 for x in driver["children"])
    audit=read(root/"formal_v1/nonlinear_tire_training_audit.json")
    assert sha((root/"formal_v1/nonlinear_tire_training_audit.json").read_bytes())==proof["training_audit_sha256"]
    assert audit["training_environment_steps"]==12288000 and not audit["smoke_only"]
    assert audit["actor_sha256"]==proof["actor_sha256"]
    for row in status["cases"]:
        assert row["returncode"]==0 and row["status"]=="completed_requires_independent_analysis"
        folder=root/"nominal_v1"/row["id"]
        metadata=read(folder/"nonlinear_tire_evaluation_status.json")
        report=read(folder/"report.json")
        assert sha((folder/"nonlinear_tire_evaluation_status.json").read_bytes())==row["metadata_sha256"]
        assert metadata==report["nonlinear_tire_model_evaluation"]
        assert metadata["normalized_actor_sha256"]==proof["actor_sha256"]
        assert metadata["model_training_audit_sha256"]==proof["training_audit_sha256"]
        assert metadata["native_states"]==2793 and metadata["policy_decisions"]==349 and metadata["mu"]==.7
        backend="mujoco" if row["id"]=="mujoco_native" else "physx"
        law="nonlinear" if row["id"]=="physx_nonlinear" else "native"
        assert (metadata["validation_backend"],metadata["validation_force_law"])==(backend,law)
        name="rollout.npz" if backend=="mujoco" else "physics_trajectory.npz"
        assert sha((folder/name).read_bytes())==metadata["recording_sha256"]
        if law=="nonlinear":
            assert sha((folder/"tire_trajectory.npz").read_bytes())==metadata["tire_trajectory_sha256"]
            assert metadata["runtime"]["physical_writes"]==2792
    return driver,status,proof


def collect(workspace,idle_audit_source,output_prefix="completed_evidence_v1"):
    root=Path(workspace).resolve()
    assert output_prefix.replace("_","").isalnum() and not output_prefix.startswith(".")
    driver,status,proof=verify_completed(root)
    helper=Path(idle_audit_source)
    assert sha(helper.read_bytes())=="6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
    spec=importlib.util.spec_from_file_location("_nonlinear_collection_process_audit",helper)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    launch_names=("driver_v1_launch.json","nominal_continuation_v1_launch.json")
    pids={read(root/name)["pid"] for name in launch_names}
    pids.update(c["pid"] for c in driver["children"])
    pids.update(c["pid"] for c in status["cases"])
    processes={str(pid):audit.process_snapshot(pid) for pid in sorted(pids)}
    assert all(p is None or p["state"]=="Z" for p in processes.values()),"Owned compute or continuation is still live"
    names=[*launch_names,"driver_status.json","formal_v1_strict_verification.json",
           "smoke_v1_strict_verification.json","formal_CPU_verification.log","nominal_v1/status.json"]
    names += [str(p.relative_to(root)) for p in sorted((root/"formal_v1").glob("*.json"))]
    for snapshot in ("source_snapshot_v1","evaluation_snapshot_v1"):
        names += [snapshot+".manifest.json"]
        manifest=read(root/(snapshot+".manifest.json"))
        for item in manifest["files"]:
            path=Path(item["path"])
            assert not path.is_absolute() and ".." not in path.parts and path.suffix in (".py",".json")
            assert sha((root/snapshot/path).read_bytes())==item["sha256"]
            names.append(snapshot+"/"+str(path))
    for case in CASES:
        folder=root/"nominal_v1"/case
        names += [str(p.relative_to(root)) for p in sorted(folder.rglob("*.json"))]
        name="rollout.npz" if case=="mujoco_native" else "physics_trajectory.npz"
        names += [str((folder/name).relative_to(root)),"nominal_v1/"+case+".log"]
        if case=="physx_nonlinear": names.append(str((folder/"tire_trajectory.npz").relative_to(root)))
    assert len(names)==len(set(names))
    payload={}
    for name in names:
        p=root/name
        assert p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink()
        assert p.suffix in (".json",".py",".npz",".log") and p.stat().st_size<=64*1024**2
        if p.suffix==".npz":assert p.name in ("rollout.npz","physics_trajectory.npz","tire_trajectory.npz")
        payload[name]=p.read_bytes()
    assert sum(map(len,payload.values()))<=256*1024**2
    assert read(root/"nominal_v1/status.json")==status and read(root/"driver_status.json")==driver
    payload["completion_process_audit.json"]=(json.dumps(dict(processes=processes,observed_unix_time=time.time(),
        all_owned_compute_and_continuation_terminal=True),indent=2)+"\n").encode()
    archive=root/(output_prefix+".tar.gz");manifest_path=root/(output_prefix+".manifest.json")
    assert not archive.exists() and not manifest_path.exists()
    with archive.open("xb") as stream:
        with tarfile.open(fileobj=stream,mode="w:gz") as tar:
            for name,data in payload.items():
                item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
                tar.addfile(item,io.BytesIO(data))
    result=dict(kind="completed_K_nonlinear_tire_training_three_nominal_episodes",archive=archive.name,
        archive_sha256=sha(archive.read_bytes()),archive_size_bytes=archive.stat().st_size,
        files=[dict(path=name,sha256=sha(data),bytes=len(data)) for name,data in payload.items()],
        actor_sha256=proof["actor_sha256"],collector_sha256=sha(Path(__file__).read_bytes()),
        models_weights_credentials_native_binaries_included=False,task_success_claimed=False)
    with manifest_path.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!="files"}))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--idle-audit-source",type=Path,required=True)
    p.add_argument("--output-prefix",default="completed_evidence_v1")
    a=p.parse_args();collect(a.workspace,a.idle_audit_source,a.output_prefix)
