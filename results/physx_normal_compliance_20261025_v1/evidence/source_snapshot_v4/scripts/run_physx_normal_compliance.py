"""Four predeclared normal-contact fixture cases, serial freshly idle GPUs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary=path.with_suffix(".tmp")
    with temporary.open("x") as stream:json.dump(data,stream,indent=2,allow_nan=False)
    os.replace(temporary,path)


def run(args):
    root=args.workspace.resolve();snapshot=args.snapshot.resolve()
    assert args.run_id.replace("_","").isalnum() and args.run_id.startswith("bench_")
    output=root/args.run_id;output.mkdir(exist_ok=False)
    status=dict(kind="serial_native_normal_compliance_bench",status="preflight",cases=[],task_success_claimed=False)
    write(output/"status.json",status)
    try:
        source=json.loads((root/(snapshot.name+".manifest.json")).read_text())
        ledger={str(snapshot/item["path"]):item["sha256"] for item in source["files"]}
        ledger[str(Path(__file__).resolve())]=sha(__file__)
        assert all(sha(path)==digest for path,digest in ledger.items())
        assert sha(args.idle_audit_source)=="6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
        spec=importlib.util.spec_from_file_location("_full_idle_audit",args.idle_audit_source)
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        for pid in (406653,408683):
            process=audit.process_snapshot(pid)
            assert process is None or process["state"]=="Z",("Prior owned job not terminal",process)
        protocol=snapshot/"config/physx_normal_compliance_v1.json"
        settings=json.loads(protocol.read_text())
        status.update(source_sha256=ledger,protocol_sha256=sha(protocol))
        deadline=time.monotonic()+1800.
        environment=os.environ.copy();environment.pop("CUDA_VISIBLE_DEVICES",None)
        environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",PYTHONDONTWRITEBYTECODE="1")
        for profile in settings["profiles"]:
            status["status"]="waiting_for_idle_GPU_for_"+profile;write(output/"status.json",status)
            while True:
                assert time.monotonic()<deadline,"Declared observation deadline reached; processes preserved"
                gpus=audit.query_gpus();selected=next((gpu for gpu in gpus if gpu["idle"]),None)
                if selected:break
                time.sleep(15.)
            assert all(sha(path)==digest for path,digest in ledger.items())
            command=[str(args.python),"-B","-u",str(snapshot/"scripts/probe_physx_normal_compliance.py"),
                "--profile",profile,"--protocol",str(protocol),"--reference-report",str(args.reference_report),
                "--device","cuda:"+str(selected["physical_index"]),"--output-dir",str(output/profile)]
            with (output/(profile+".log")).open("xb") as log:
                child=subprocess.Popen(command,cwd=snapshot,env=environment,stdout=log,stderr=subprocess.STDOUT)
                case=dict(id=profile,pid=child.pid,argv=command,status="running",gpu_audit=gpus,selected_gpu=selected,launched_unix_time=time.time())
                status["cases"].append(case);status["status"]="running_"+profile;write(output/"status.json",status)
                while child.poll() is None:
                    assert time.monotonic()<deadline,"Owned child observation deadline reached; process preserved"
                    time.sleep(2.)
            case["returncode"]=child.returncode
            assert child.returncode==0,("Bench failed; no retry",profile,child.returncode)
            failure=output/profile/"failure.json"
            if failure.is_file():
                case.update(status="failed",failure_sha256=sha(failure))
                raise AssertionError("Bench emitted failure.json; no retry: "+profile)
            proof=json.loads((output/profile/"status.json").read_text())
            assert proof["status"]=="completed_requires_analysis" and proof["main_physical_steps"]==1200
            assert proof["profile"]["id"]==profile and proof["peak_impulse_balance_error_n"]<.1
            case.update(status="completed_verified",proof_sha256=sha(output/profile/"status.json"),native_trace_sha256=sha(output/profile/"native_trace.npz"))
            write(output/"status.json",status)
        status["status"]="four_native_bench_cases_completed_require_analysis"
        write(output/"status.json",status)
    except BaseException as error:
        status.update(status="failed_no_retry_or_process_signal",error=repr(error))
        write(output/"status.json",status)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("workspace","snapshot","idle-audit-source","reference-report","python"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--run-id",default="bench_v1")
    run(p.parse_args())
