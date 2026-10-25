"""Serial three-law, two-mu PhysX car-surrogate diagnostics; no controller tuning."""
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


def run(a):
    assert sha(a.gpu_audit_module)=="dee06f19fba115ff149703164bc35898d612b70e139e7874f038ad4440b25440"
    spec=importlib.util.spec_from_file_location("_car_gpu_audit",a.gpu_audit_module)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.GPU_ORDER==(3,5,4) and "CUDA_VISIBLE_DEVICES" not in os.environ
    root=a.workspace;source=root/"source_snapshot_car_v2"
    assert sha(root/"source_snapshot_car_v2.manifest.json")=="0b0b044b89d2a491f8a6773751b6b8f6ca4caf5307b902b83a54af38619c445d"
    manifest=json.loads((root/"source_snapshot_car_v2.manifest.json").read_text())
    assert all(sha(source/x["path"])==x["sha256"] for x in manifest["files"])
    for name in ("car_native_smoke_v1","car_external_smoke_v1"):
        launch=json.loads((root/(name+"_launch.json")).read_text());assert helper.process_snapshot(launch["pid"]) is None
        proof=json.loads((root/name/"status.json").read_text())
        assert proof["status"]=="completed_requires_analysis" and proof["smoke_only"] and proof["native_states"]==41
        assert sha(root/name/"native_trace.npz")==proof["native_trace_sha256"]
    output=root/"car_tire_sweep_v1";output.mkdir()
    status={"status":"initializing","cases":[],"driver_sha256":sha(__file__),
        "car_kind":"explicitly_assumed_Audi_A2_size_FWD_PhysX_surrogate_not_original_CARLA_plant",
        "exact_source_vehicle_reproduced":False,"other_processes_stopped":False,"automatic_retry":False}
    def save():
        p=output/"status.tmp";p.write_text(json.dumps(status,indent=2)+chr(10));os.replace(p,output/"status.json")
    save();deadline=time.monotonic()+1800
    try:
        for mu in (.7,3.5):
            for law in ("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"):
                while True:
                    assert time.monotonic()<deadline,"No idle approved GPU before deadline"
                    gpus=helper.query_gpus();gpu=next((g for g in gpus if g["idle"]),None)
                    if gpu is not None:break
                    status.update(status="waiting_for_fully_idle_GPU",gpu_snapshot=gpus);save();time.sleep(20)
                name=law+"_mu"+str(mu).replace(".","p");folder=output/name
                args=[str(a.python),"-B","-u",str(source/"scripts/run_physx_car_tire_surrogate.py"),
                    "--config",str(source/"config/physx_car_surrogate_v1.json"),"--reference",str(source/"reference/car_source_reference_v1.npz"),
                    "--law",law,"--mu",str(mu),"--device",f"cuda:{gpu['physical_index']}","--output-dir",str(folder)]
                env=os.environ.copy();env["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
                with (output/(name+".log")).open("xb") as log:child=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,shell=False)
                identity=helper.process_snapshot(child.pid);assert identity and identity["uid"]==os.getuid() and identity["argv"]==args
                case={"id":name,"law":law,"mu":mu,"pid":child.pid,"process_identity_at_launch":identity,
                    "gpu_idle_before_launch":gpu,"status":"running"}
                status["cases"].append(case);status["status"]="running_single_case";save()
                result=child.wait();case["returncode"]=result;assert result==0
                proof=json.loads((folder/"status.json").read_text())
                assert proof["status"]=="completed_requires_analysis" and proof["smoke_only"] is False
                assert proof["native_states"]==993 and proof["physics_steps"]==992 and proof["law"]==law and proof["mu"]==mu
                assert sha(folder/"native_trace.npz")==proof["native_trace_sha256"]
                assert all(sha(p)==digest for p,digest in proof["source_sha256"].items())
                case.update(status="completed_verified",native_trace_sha256=proof["native_trace_sha256"],report_sha256=sha(folder/"status.json"));save()
        status["status"]="six_car_surrogate_cases_completed_require_analysis"
    except BaseException as e:
        status.update(status="failed_no_automatic_retry",error=str(e),error_type=type(e).__name__);save();raise
    save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","gpu-audit-module","python"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
