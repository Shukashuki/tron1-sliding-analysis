"""Six frozen-actor model rollouts, serial on freshly audited idle RTX GPUs."""
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


def write(path,value):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.replace(temp,path)


def run(a):
    source=a.source_dir.resolve();m=read(a.source_manifest)
    assert len(m["files"])==6 and all(sha(source/r["path"])==r["sha256"] for r in m["files"])
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert sha(a.gpu_audit_module)=="6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866"
    spec=importlib.util.spec_from_file_location("_wheel_transfer_gpu",a.gpu_audit_module)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.GPU_ORDER==(3,5,4)
    origin=read(a.native_audit_dir/"wheel_contact_model_status.json")
    assert origin["status"]=="completed_requires_analysis" and origin["geometry"]=="authored_auto"
    assert origin["read_only_baseline_byte_identical"]
    assert sha(a.native_audit_dir/"physics_trajectory.npz")==origin["physics_trajectory_sha256"]=="cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a"
    cpu=read(a.usd_preflight)
    assert cpu["kind"]=="CPU_USD_wheel_geometry_audit" and len(cpu["rows"])==4
    assert all(r["mass_COM_inertia_exact"] and r["transform_envelope_volume_verified"] for r in cpu["rows"])
    assert cpu["source_sha256"]["tron1_wheel_geometry.py"]==sha(source/"scripts/tron1_wheel_geometry.py")
    assert sha(a.asset)==cpu["asset_sha256"]
    protocol=read(source/"config/physx_tron1_wheel_models_v1.json")
    assert protocol["matched_robot_contact_offset_m"]==.001 and protocol["matched_robot_rest_offset_m"]==0.
    for wheel in origin["native_audit"]["wheels"].values():
        assert all(abs(v-.001)<1e-8 for v in wheel["contact_offsets_m"][0])
        assert all(abs(v)<1e-8 for v in wheel["rest_offsets_m"][0])
    a.output_dir.mkdir()
    status=dict(kind="frozen_K_matched_offset_PhysX_geometry_force_transfer",status="starting",cases=[],
        source_manifest_sha256=sha(a.source_manifest),native_offset_audit_sha256=sha(a.native_audit_dir/"wheel_contact_model_status.json"),
        CPU_USD_preflight_sha256=sha(a.usd_preflight),driver_sha256=sha(__file__),automatic_retry=False,
        other_process_signals=False,GPU_reservation_claimed=False,policy_training=False,task_success_claimed=False)
    target=a.output_dir/"status.json";write(target,status);deadline=time.monotonic()+a.deadline_seconds
    try:
        for law in protocol["force_levels"]:
            for geometry in protocol["geometry_levels"]:
                assert all(sha(source/r["path"])==r["sha256"] for r in m["files"])
                while True:
                    assert time.monotonic()<deadline,"No fully idle approved GPU before deadline"
                    gpus=helper.query_gpus();gpu=next((g for g in gpus if g["idle"]),None)
                    if gpu is not None:break
                    status.update(status="waiting_for_fully_idle_GPU",gpu_snapshot=gpus);write(target,status);time.sleep(10)
                name=geometry+"__"+law;folder=a.output_dir/name
                base=a.base_workspace
                argv=[str(a.python),"-B","-u",str(source/"scripts/eval_physx_wheel_contact_models.py"),
                    "--base-snapshot",str(base/"source_snapshot_v1"),"--baseline-dir",str(base/"physx_nominal_eval_v1"),
                    "--geometry",geometry,"--law",law,"--contact-offset-m",".001","--rest-offset-m","0",
                    "--training-dir",str(base/"physx_formal_v1"),"--protocol",str(base/"source_snapshot_v1/config/ppo_ground_contact_v2k_400hz_v1.json"),
                    "--legacy-evaluator",str(a.legacy_evaluator),"--motion-file",str(a.motion),"--output-dir",str(folder),
                    "--mu",".7","--backend","physx","--asset-path",str(a.asset),"--physical-gpu-index",str(gpu["physical_index"])]
                with (a.output_dir/(name+".log")).open("xb") as log:
                    child=subprocess.Popen(argv,cwd=a.output_dir,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",TRON1_BEYONDMIMIC_ROOT=str(a.beyondmimic_source)),shell=False)
                identity=helper.process_snapshot(child.pid)
                assert identity and identity["uid"]==os.getuid() and identity["argv"]==argv
                record=dict(id=name,geometry=geometry,law=law,pid=child.pid,argv=argv,process_identity_at_launch=identity,
                    gpu_idle_before_launch=gpu,status="running")
                status["cases"].append(record);status["status"]="running_single_case";write(target,status)
                while child.poll() is None:
                    assert time.monotonic()<deadline,"Owned child preserved, not signalled, after deadline"
                    time.sleep(1)
                record["returncode"]=child.returncode;assert child.returncode==0,"Case process failed; preserve evidence"
                d=read(folder/"wheel_contact_model_status.json")
                assert d["status"]=="completed_requires_analysis" and d["geometry"]==geometry and d["law"]==law
                assert d["normalized_actor_sha256"]==protocol["actor_sha256"] and d["native_states"]==2793
                assert d["native_audit"]["masses_COM_inertias_armature_damping_match_baseline"]
                assert sha(folder/"physics_trajectory.npz")==d["physics_trajectory_sha256"]
                assert sha(folder/"tire_trajectory.npz")==d["tire_trajectory_sha256"]
                assert all(sha(p)==digest for p,digest in d["source_sha256"].items())
                if geometry=="authored" and law=="native_coulomb":
                    assert d["read_only_baseline_byte_identical"],"Matched explicit offset altered the native baseline"
                record.update(status="completed_verified",metadata_sha256=sha(folder/"wheel_contact_model_status.json"),
                    physics_trajectory_sha256=d["physics_trajectory_sha256"],tire_trajectory_sha256=d["tire_trajectory_sha256"])
                write(target,status)
        status["status"]="six_matched_offset_PhysX_transfer_cases_completed_require_analysis"
    except BaseException as error:
        if status["cases"] and status["cases"][-1]["status"]=="running":status["cases"][-1]["status"]="failed_validation"
        status.update(status="failed_no_automatic_retry",error_type=type(error).__name__,error=str(error));write(target,status);raise
    write(target,status)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("source-dir","source-manifest","gpu-audit-module","native-audit-dir","usd-preflight","output-dir",
              "base-workspace","asset","legacy-evaluator","motion","beyondmimic-source","python"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--deadline-seconds",type=float,default=1800)
    run(p.parse_args())
