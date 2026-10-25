"""Run six single-wheel PhysX diagnostics serially on freshly idle approved GPUs.

No installation, native compilation, policy training, retries, process signals,
or GPU reservation. Stops after a failed case. Uses the already audited full
NVML/PCI process query helper; no GPU allocation while waiting for a device.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+chr(10))
    os.replace(temporary,path)


def run(a):
    workspace=a.workspace.resolve();snapshot=workspace/"source_snapshot_v9"
    path=a.gpu_audit_module.resolve()
    assert sha(path)=="dee06f19fba115ff149703164bc35898d612b70e139e7874f038ad4440b25440"
    spec=importlib.util.spec_from_file_location("_tire_sweep_gpu_audit",path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.GPU_ORDER==(3,5,4)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    smoke=json.loads((workspace/"external_fixture_force_smoke_v1_launch.json").read_text())
    assert helper.process_snapshot(smoke["pid"]) is None,"Wait for exact smoke process exit"
    proof=json.loads((workspace/"external_fixture_force_smoke_v1/status.json").read_text())
    assert proof["status"]=="completed" and proof["native_states"]==41
    assert proof["axle_fixture_locked_world_rotation_bitmask_XZ"]==5
    assert .117 < proof["settled_state_w_before_velocity_initialization"][2] < .137
    assert proof["law"]=="external_coulomb_regularized" and proof["torch_current_cuda_device"] in (3,5,4)
    assert sha(workspace/"external_fixture_force_smoke_v1/native_trace.npz")==proof["native_trace_sha256"]
    previous=json.loads((workspace/"single_wheel_fixture_sweep_v1/status.json").read_text())
    assert previous["status"]=="failed_no_automatic_retry"
    assert all(helper.process_snapshot(case["pid"]) is None for case in previous["cases"])
    manifest=json.loads((workspace/"source_snapshot_v9.manifest.json").read_text())
    assert sha(workspace/"source_snapshot_v9.tar.gz")==smoke["source_archive_sha256"]
    assert sha(workspace/"source_snapshot_v9.manifest.json")==smoke["source_manifest_sha256"]
    assert all(sha(snapshot/x["path"])==x["sha256"] for x in manifest["files"])
    output=workspace/"single_wheel_fixture_sweep_v2";output.mkdir()
    status={"kind":"serial_PhysX_single_wheel_law_and_dt_bench","status":"initializing",
        "cases":[],"driver_sha256":sha(__file__),"source_archive_sha256":smoke["source_archive_sha256"],
        "new_native_compilation_or_installation":False,"automatic_retry":False,
        "other_process_signals":False,"GPU_reservation_claimed":False,
        "car_or_TRON1_task_result":False,"task_success_claimed":False}
    target=output/"status.json";write(target,status)
    deadline=time.monotonic()+a.deadline_seconds
    try:
        for dt,hz in ((.0025,400),(.00125,800)):
            for law in ("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"):
                name=f"{law}_{hz}hz";folder=output/name
                assert not folder.exists()
                while True:
                    assert time.monotonic()<deadline,"No approved idle GPU before deadline"
                    gpus=helper.query_gpus();idle=next((g for g in gpus if g["idle"]),None)
                    if idle is not None:break
                    status.update(status="waiting_for_fully_idle_approved_GPU",gpu_snapshot=gpus)
                    write(target,status);time.sleep(20)
                argv=[str(a.python),"-B","-u",str(snapshot/"scripts/probe_physx_tire_contact_fixture.py"),
                    "--law",law,"--dt",str(dt),"--duration","2","--device",f"cuda:{idle['physical_index']}",
                    "--output-dir",str(folder)]
                env=os.environ.copy();env["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
                with (output/(name+".log")).open("xb") as log:
                    child=subprocess.Popen(argv,cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT,shell=False)
                identity=helper.process_snapshot(child.pid)
                assert identity and identity["uid"]==os.getuid() and identity["argv"]==argv
                case={"id":name,"law":law,"dt_s":dt,"pid":child.pid,"argv":argv,
                    "process_identity_at_launch":identity,"gpu_idle_before_launch":idle,
                    "status":"running","started_unix_time":time.time()}
                status["cases"].append(case);status["status"]="running_single_case";write(target,status)
                result=child.wait()
                case.update(returncode=result,finished_unix_time=time.time())
                assert result==0,"PhysX case failed; preserve outputs and do not retry"
                report=json.loads((folder/"status.json").read_text())
                assert report["status"]=="completed" and report["law"]==law and report["physics_dt_s"]==dt
                assert report["physical_steps"]==round(2/dt) and report["native_states"]==round(2/dt)+1
                assert sha(folder/"native_trace.npz")==report["native_trace_sha256"]
                assert all(sha(p)==digest for p,digest in report["source_sha256"].items())
                case.update(status="completed_verified",report_sha256=sha(folder/"status.json"),
                    native_trace_sha256=report["native_trace_sha256"])
                write(target,status)
        status["status"]="all_six_single_wheel_cases_completed_require_analysis"
    except BaseException as error:
        status.update(status="failed_no_automatic_retry",error_type=type(error).__name__,error=str(error))
        write(target,status)
        raise
    write(target,status)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--gpu-audit-module",type=Path,required=True)
    p.add_argument("--python",type=Path,required=True)
    p.add_argument("--deadline-seconds",type=float,default=1800)
    run(p.parse_args())
