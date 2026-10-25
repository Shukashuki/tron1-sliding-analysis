"""Four fixed-policy normal-material transfers, serial fully idle RTX GPUs."""
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


def write(path,value):
    temporary=path.with_suffix(".tmp")
    with temporary.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
    os.replace(temporary,path)


def run(args):
    root=args.workspace.resolve();snapshot=args.snapshot.resolve()
    out=root/"turn_models_v1";out.mkdir(exist_ok=False)
    status=dict(kind="frozen_K_native_normal_contact_model_transfers",status="preflight",cases=[],task_success_claimed=False)
    write(out/"status.json",status)
    try:
        assert sha(args.idle_audit_source)=="6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
        spec=importlib.util.spec_from_file_location("_fresh_full_GPU_audit",args.idle_audit_source)
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        for pid in (431143,431147,432603,434242,434245,434578,434895,435236):
            observed=audit.process_snapshot(pid);assert observed is None or observed["state"]=="Z",("Prior owned job still active",observed)
        manifest=json.loads((root/(snapshot.name+".manifest.json")).read_text())
        ledger={str(snapshot/item["path"]):item["sha256"] for item in manifest["files"]}
        assert all(sha(path)==digest for path,digest in ledger.items())
        protocol=snapshot/"config/physx_normal_compliance_v1.json"
        settings=json.loads(protocol.read_text());deadline=time.monotonic()+1800.
        status.update(source_sha256=ledger,protocol_sha256=sha(protocol));write(out/"status.json",status)
        environment=os.environ.copy();environment.pop("CUDA_VISIBLE_DEVICES",None)
        environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",PYTHONDONTWRITEBYTECODE="1",
            TRON1_BEYONDMIMIC_ROOT="/data/zengxiong/workspaces/tron1-beyondmimic-20261002/third_party/whole_body_tracking")
        for profile in settings["profiles"]:
            status["status"]="waiting_for_fully_idle_GPU_for_"+profile;write(out/"status.json",status)
            while True:
                assert time.monotonic()<deadline,"Declared monitoring deadline reached; processes preserved"
                gpus=audit.query_gpus();selected=next((g for g in gpus if g["idle"]),None)
                if selected:break
                time.sleep(15.)
            assert all(sha(path)==digest for path,digest in ledger.items())
            command=[str(args.python),"-B","-u",str(snapshot/"scripts/eval_physx_normal_compliance.py"),
                "--contact-model-evaluator","/data/zengxiong/workspaces/tron1-wheel-model-transfer-20261025/source_snapshot_v2/scripts/eval_physx_wheel_contact_models.py",
                "--normal-profile",profile,"--normal-protocol",str(protocol),"--normal-reference-report",str(args.reference_report),"--bench-dir",str(args.bench_dir),
                "--base-snapshot","/data/zengxiong/workspaces/tron1-ground-contact-v2k-20261024/source_snapshot_v1",
                "--baseline-dir","/data/zengxiong/workspaces/tron1-ground-contact-v2k-20261024/physx_nominal_eval_v1",
                "--geometry","authored","--law","native_coulomb","--contact-offset-m",".001","--rest-offset-m","0",
                "--training-dir","/data/zengxiong/workspaces/tron1-ground-contact-v2k-20261024/physx_formal_v1",
                "--protocol","/data/zengxiong/workspaces/tron1-ground-contact-v2k-20261024/source_snapshot_v1/config/ppo_ground_contact_v2k_400hz_v1.json",
                "--legacy-evaluator","/data/zengxiong/workspaces/tron1-controller-comparison-20261015/controller_evaluation_source_v2/scripts/eval_tracking_friction_isaac.py",
                "--motion-file","/data/zengxiong/workspaces/tron1-drift-turn-worst-load-20261014/motion.npz",
                "--output-dir",str(out/profile),"--mu",".7","--backend","physx",
                "--asset-path","/data/zengxiong/workspaces/tron1-beyondmimic-20261002/outputs/training-asset/WF_TRON1A.usda",
                "--physical-gpu-index",str(selected["physical_index"])]
            with (out/(profile+".log")).open("xb") as log:
                child=subprocess.Popen(command,cwd=snapshot,env=environment,stdout=log,stderr=subprocess.STDOUT)
                case=dict(id=profile,pid=child.pid,argv=command,status="running",gpu_audit=gpus,selected_gpu=selected,launched_unix_time=time.time())
                status["cases"].append(case);status["status"]="running_"+profile;write(out/"status.json",status)
                while child.poll() is None:
                    assert time.monotonic()<deadline,"Owned child monitoring deadline reached; process preserved"
                    time.sleep(2.)
            case["returncode"]=child.returncode
            assert child.returncode==0,("Transfer failed; no retry",profile,child.returncode)
            proof=json.loads((out/profile/"normal_compliance_status.json").read_text())
            assert proof["kind"]=="frozen_K_PhysX_native_normal_contact_model_transfer"
            assert proof["normal_contact_model_intervention"]["profile"]["id"]==profile
            assert proof["native_states"]==2793 and proof["policy_decisions"]==349
            for name in ("physics_trajectory","tire_trajectory"):
                assert sha(out/profile/(name+".npz"))==proof[name+"_sha256"]
            case.update(status="completed_verified",proof_sha256=sha(out/profile/"normal_compliance_status.json"),
                physics_trajectory_sha256=proof["physics_trajectory_sha256"],tire_trajectory_sha256=proof["tire_trajectory_sha256"])
            write(out/"status.json",status)
        status["status"]="four_normal_contact_turn_cases_completed_require_analysis";write(out/"status.json",status)
    except BaseException as error:
        status.update(status="failed_no_retry_or_process_signal",error=repr(error));write(out/"status.json",status)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("workspace","snapshot","bench-dir","reference-report","idle-audit-source","python"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
