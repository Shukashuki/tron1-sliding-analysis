"""Serial native offset readbacks followed by matched-offset free-wheel tests."""
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


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temp,path)


def run(a):
    source=a.source_dir.resolve();m=read(a.source_manifest)
    assert len(m["files"])==5 and all(sha(source/r["path"])==r["sha256"] for r in m["files"])
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert sha(a.gpu_audit_module)=="6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
    spec=importlib.util.spec_from_file_location("_camber_gpu_audit",a.gpu_audit_module)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.GPU_ORDER==(3,5,4)
    protocol=read(source/"config/physx_wheel_camber_controls_v1.json")
    assert protocol["kind"]=="wheel_contact_offset_and_free_camber_controls_v1"
    a.output_dir.mkdir()
    status=dict(kind=protocol["kind"],status="starting",cases=[],source_manifest_sha256=sha(a.source_manifest),
        driver_sha256=sha(__file__),policy_training=False,task_success_claimed=False,
        automatic_retry=False,process_signals_sent=False,GPU_reservation_claimed=False)
    target=a.output_dir/"status.json";write(target,status)
    cases=[dict(id=f"auto_offset_slices{n}",slices=n,camber=0.,rotation="locked",duration=.1,explicit=False)
           for n in (1,9)]
    cases += [dict(id=f"free_camber{int(phi):+d}_slices{n}",slices=n,camber=phi,rotation="free",duration=2.,explicit=True)
              for phi in (0.,-15.,15.) for n in (1,9)]
    deadline=time.monotonic()+a.deadline_seconds
    try:
        for case in cases:
            assert all(sha(source/r["path"])==r["sha256"] for r in m["files"])
            if not case["explicit"] and a.offset_proof_root is not None:
                folder=a.offset_proof_root/case["id"]
                result=read(folder/"status.json")
                assert result["status"]=="completed" and result["physical_steps"]==40
                assert result["kind"]=="PhysX_single_wheel_offset_and_free_camber_controls"
                assert result["law"]=="native_coulomb" and result["episode_rotation"]=="locked"
                assert result["compound_geometry"]["slices"]==case["slices"]
                assert result["compound_geometry"]["offsets_native_readback_verified"]
                assert sha(folder/"native_trace.npz")==result["native_trace_sha256"]
                source_pins={Path(r["path"]).name:r["sha256"] for r in m["files"]}
                assert all(source_pins[Path(p).name]==digest and sha(p)==digest for p,digest in result["source_sha256"].items())
                status["cases"].append(dict(case,status="completed_verified",reused_prior_completed_measurement=True,
                    input_directory=str(folder.resolve()),report_sha256=sha(folder/"status.json"),
                    native_trace_sha256=result["native_trace_sha256"],
                    runtime_contact_offsets_m=result["compound_geometry"]["runtime_contact_offsets_m"],
                    runtime_rest_offsets_m=result["compound_geometry"]["runtime_rest_offsets_m"]))
                write(target,status)
                continue
            while True:
                assert time.monotonic()<deadline,"No fully idle approved GPU before deadline"
                gpus=helper.query_gpus();idle=next((g for g in gpus if g["idle"]),None)
                if idle is not None:break
                status.update(status="waiting_for_fully_idle_GPU",gpu_snapshot=gpus);write(target,status);time.sleep(10)
            folder=a.output_dir/case["id"]
            argv=[str(a.python),"-B","-u",str(source/"scripts/probe_physx_wheel_geometry_controls.py"),
                "--law","native_coulomb","--slices",str(case["slices"]),"--approximation","smooth",
                "--duration",str(case["duration"]),"--camber-deg",str(case["camber"]),
                "--episode-rotation",case["rotation"],"--device",f"cuda:{idle['physical_index']}","--output-dir",str(folder)]
            if case["explicit"]:
                assert len(status["cases"])>=2 and all(c["status"]=="completed_verified" for c in status["cases"][:2])
                argv += ["--contact-offset-m",str(protocol["explicit_contact_offset_m_after_readback"]),
                         "--rest-offset-m",str(protocol["explicit_rest_offset_m_after_readback"])]
            with (a.output_dir/(case["id"]+".log")).open("xb") as log:
                child=subprocess.Popen(argv,cwd=a.output_dir,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID"),shell=False)
            identity=helper.process_snapshot(child.pid)
            assert identity and identity["uid"]==os.getuid() and identity["argv"]==argv
            record=dict(case,argv=argv,pid=child.pid,process_identity_at_launch=identity,
                gpu_idle_before_launch=idle,status="running")
            status["cases"].append(record);status["status"]="running_single_case";write(target,status)
            while child.poll() is None:
                assert time.monotonic()<deadline,"Owned child preserved after deadline; no signals"
                time.sleep(1)
            record["returncode"]=child.returncode
            assert child.returncode==0 and not (folder/"failure.json").exists(),"Case failed; inspect preserved evidence"
            result=read(folder/"status.json")
            assert result["status"]=="completed" and result["physical_steps"]==round(case["duration"]/.0025)
            assert result["kind"]=="PhysX_single_wheel_offset_and_free_camber_controls"
            assert result["law"]=="native_coulomb" and result["initial_camber_deg"]==case["camber"]
            assert result["episode_rotation"]==case["rotation"]
            g=result["compound_geometry"]
            assert g["offsets_native_readback_verified"] and g["slices"]==case["slices"]
            if case["explicit"]:
                assert all(abs(v-protocol["explicit_contact_offset_m_after_readback"])<1e-8
                           for v in g["runtime_contact_offsets_m"][0]), "Explicit contact offset native readback differs"
                assert all(abs(v-protocol["explicit_rest_offset_m_after_readback"])<1e-8
                           for v in g["runtime_rest_offsets_m"][0]), "Native rest offset differs"
            assert sha(folder/"native_trace.npz")==result["native_trace_sha256"]
            assert all(sha(p)==digest for p,digest in result["source_sha256"].items())
            record.update(status="completed_verified",report_sha256=sha(folder/"status.json"),
                native_trace_sha256=result["native_trace_sha256"],runtime_contact_offsets_m=g["runtime_contact_offsets_m"],
                runtime_rest_offsets_m=g["runtime_rest_offsets_m"])
            write(target,status)
        status["status"]="all_eight_offset_and_free_camber_controls_completed_require_analysis"
    except BaseException as error:
        if status["cases"] and status["cases"][-1]["status"]=="running":status["cases"][-1]["status"]="failed_validation"
        status.update(status="failed_no_automatic_retry",error_type=type(error).__name__,error=str(error));write(target,status);raise
    write(target,status)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("source-dir","source-manifest","gpu-audit-module","python","output-dir"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--deadline-seconds",type=float,default=1200)
    p.add_argument("--offset-proof-root",type=Path)
    run(p.parse_args())
