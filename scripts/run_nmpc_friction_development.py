"""Read-only reproduction planner for the unqualified MuJoCo NMPC mu grid.

This script never imports a simulator, launches commands, compiles, creates
outputs, or weakens the separate nominal-qualified validation gate. Commands
are printed only. Reused completed evidence is byte-bound, not proof of the
original simulator internals; no XY/posture/flight success filter is applied.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
KIND="fixed_parameter_mujoco_nmpc_friction_development"
GRID=[.01,.03,.05,.1,.15,.2,.3,.45,.6,.7,.8,1.,1.2,1.4,1.6,1.8,2.]
CONFIG_DIGEST="c30a83dffe39e167748d8305f02d43025dbb83af2d6539bd44b3de7ec4616a74"
SOURCES=("scripts/run_nmpc_tracking.py","scripts/nmpc_shooting.py","scripts/nmpc_tracking_objective.py",
    "scripts/nmpc_native_batch.py","scripts/nmpc_batch_objective.py","native/nmpc_batch.cpp",
    "scripts/controller_references.py","config/controller_references.json","scripts/height_smoothness.py",
    "scripts/classical_tracking.py","config/classical_tracking.json","config/nmpc_height_smooth_residual.json",
    "config/car_drift_height_smooth_task.json",
    "upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py",
    "upstream/tron1-learning-from-human/scripts/eval_tracking.py",
    "upstream/tron1-learning-from-human/training/tron1_domain_randomization.py",
    "upstream/tron1-learning-from-human/training/tron1_contact_slip.py",
    "upstream/tron1-learning-from-human/training/tron1_tracking.py",
    "upstream/tron1-learning-from-human/training/tron1_drift.py",
    "upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
    "upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py")


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def repository_file(root,relative):
    path=Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,"Expected repository-relative source path")
    result=(Path(root)/path).resolve()
    require(result.is_relative_to(Path(root).resolve()),"Source escapes repository")
    return result


def recorded_source(report,suffix):
    values=[value for name,value in report.get("source_sha256",{}).items()
            if name==suffix or name.endswith("/"+suffix)]
    require(len(values)==1,"Missing/ambiguous reused source: "+suffix)
    return values[0]


def npy_values(archive,key,shape):
    """Read only the small float64 clock vectors, with no NumPy import."""
    entry=archive.getinfo(key+".npy")
    require(entry.file_size<64*1024,"Unexpected oversized reused clock")
    raw=archive.read(entry)
    require(raw[:6]==b"\x93NUMPY" and raw[6:8] in (b"\x01\x00",b"\x02\x00",b"\x03\x00"),"Invalid NPY clock")
    size=2 if raw[6]==1 else 4;offset=8+size
    length=int.from_bytes(raw[8:offset],"little")
    header=ast.literal_eval(raw[offset:offset+length].decode("latin1"))
    require(header.get("shape")==shape and header.get("descr") in ("<f8","=f8")
            and header.get("fortran_order") is False,"Unexpected reused float64 clock schema")
    count=math.prod(shape);payload=raw[offset+length:]
    require(len(payload)==8*count,"Reused clock byte length mismatch")
    return struct.unpack("<"+"d"*count,payload)


def audit_reused_case(directory,mu,manifest,inputs,effective,settings,task,root):
    directory=Path(directory)
    hashes={name:sha(directory/name) for name in ("report.json","rollout.npz","solver_log.json")}
    report=json.loads((directory/"report.json").read_text())
    require(report.get("kind")=="full_body_height_smooth_cem_nmpc" and report.get("engine")=="MuJoCo"
            and report.get("status")=="completed","Reused case must be genuinely completed full-body MuJoCo NMPC evidence")
    require(report.get("plant_mu")==mu and report.get("prediction_mu_scenarios")==[.7]
            and report.get("effective_cem_config")==effective,"Reused plant/predictor/config differs")
    require(report.get("model_sha256")==manifest["input_sha256"]["model"]
            and report.get("motion_sha256")==manifest["input_sha256"]["motion"]
            and report.get("reference_id")==manifest["reference_id"]
            and report.get("reference_catalog_sha256")==sha(Path(root)/"config/controller_references.json"),"Reused model/reference differs")
    require(report.get("rollout_sha256")==hashes["rollout.npz"]
            and report.get("solver_log_sha256")==hashes["solver_log.json"],"Reused report-bound archive hashes differ")
    require(report.get("completed_policy_steps")==report.get("evaluated_policy_steps")
            ==report.get("total_reference_policy_steps")==349
            and math.isclose(report.get("duration_s",math.nan),6.98,abs_tol=1e-10,rel_tol=0)
            and report.get("physics_dt_s")==.005 and report.get("control_dt_s")==.02,"Reused episode is incomplete or clock changed")
    for field in ("truncated_smoke","prediction_receives_plant_mu","learned_policy_used","tvlqr_used",
                  "old_qr_used","root_state_prescribed_after_reset","external_root_forces_used","pose_height_failure_gate"):
        require(report.get(field) is False,"Unsupported reused controller declaration: "+field)
    require(report.get("model_generated_nominal_plan")=={"enabled":False}
            and report.get("prediction_backend")=="native","Reused controller is not the explicit native fixed-model baseline")
    require(report.get("plant_material_readback",{}).get("compiled_sliding_friction_range")==[mu,mu],"Reused compiled plant material differs")
    for relative,expected in manifest["source_sha256"].items():
        require(recorded_source(report,relative)==expected,"Reused source identity differs: "+relative)
    require(recorded_source(report,manifest["settings"])==sha(settings)
            and recorded_source(report,manifest["task_profile"])==sha(task)
            and manifest["input_sha256"]["actuator_contract"] in report["source_sha256"].values(),"Reused settings/task/actuator differs")
    native=report.get("native_prediction_backend",{})
    require(native.get("production_cem_wired") is True and set(native.get("workers",{}))=={"0.7"}
            and native["workers"]["0.7"].get("native_library_sha256")==manifest["input_sha256"]["native_library"],"Reused native worker identity differs")
    rows=json.loads((directory/"solver_log.json").read_text())
    require(isinstance(rows,list) and len(rows)==349 and all(row.get("frame")==k
            and row.get("reference_frame_for_reward")==k
            and math.isclose(row.get("time_s",math.nan),(k+1)*.02,abs_tol=1e-10,rel_tol=0)
            for k,row in enumerate(rows)),"Reused solver clock/phase is incomplete")
    with zipfile.ZipFile(directory/"rollout.npz") as archive:
        clock=npy_values(archive,"time_s",(1397,))
        require(all(math.isfinite(value) and math.isclose(value,k*.005,abs_tol=1e-10,rel_tol=0)
                    for k,value in enumerate(clock)),"Reused native physical clock differs")
    return {"plant_mu":mu,"directory":str(directory.resolve()),"input_sha256":hashes,
            "completed_policy_steps":349,"physical_samples":1397,"xy_success_required_for_reuse":False}


def build_plan(manifest_path,inputs,output_root,*,reused_cases=(),repo_root=ROOT,python=sys.executable):
    manifest=json.loads(Path(manifest_path).read_text());root=Path(repo_root).resolve()
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"]==1 and manifest.get("kind")==KIND
            and manifest.get("reference_id")=="drift_drl_traj4" and manifest.get("plant_mu")==GRID
            and manifest.get("reused_existing_plant_mu")==[.3,.7]
            and manifest.get("prediction_mu_scenarios")==[.7] and manifest.get("seed")==42,"Unexpected development plan schema/grid")
    require(manifest.get("evaluation_only") is True,"This planner is evaluation only")
    for key in ("controller_parameters_changed_between_coefficients","retuning_or_selection_after_friction_results",
        "qualified_nominal_pair_available","formal_controller_freeze_claimed","qualified_validation_driver_modified",
        "isaac_results_for_this_development_sweep","sim2sim_sweep_claimed","height_flight_or_heading_offset_failure_gate"):
        require(manifest.get(key) is False,"Unsafe/missing development scope: "+key)
    require(set(inputs)=={"model","motion","actuator_contract","native_library"},"Require four explicit runtime input roles")
    for role,path in inputs.items():
        require(sha(path)==manifest["input_sha256"][role],"Input byte identity differs: "+role)
    sources=manifest["source_sha256"]
    require(set(sources)==set(SOURCES),"Missing/unexpected fixed runtime source bindings")
    for relative,expected in sources.items():require(sha(repository_file(root,relative))==expected,"Current source identity differs: "+relative)
    settings=repository_file(root,manifest["settings"]);task=repository_file(root,manifest["task_profile"])
    declared=json.loads(settings.read_text());effective={**declared["nmpc"],**manifest["explicit_overrides"]}
    require(declared.get("schema_version")==1 and declared.get("task")=="car_drift_height_smooth"
            and declared.get("physics_dt_s")==.005 and declared.get("policy_dt_s")==.02
            and declared.get("prediction_mu_scenarios")==[.7],"Settings clocks/prediction model differ")
    require(digest(effective)==manifest.get("effective_cem_configuration_sha256")==CONFIG_DIGEST,"Effective CEM configuration digest differs")
    catalog=json.loads((root/"config/controller_references.json").read_text())
    spec=catalog["references"][manifest["reference_id"]]
    require(spec["motion_npz_sha256"]==manifest["input_sha256"]["motion"]
            and spec["robot_reference_npz_sha256"]==manifest["input_sha256"]["robot_reference"],"Reference catalog provenance differs")
    reused={}
    for value in reused_cases:
        coefficient,separator,directory=value.partition("=")
        require(separator and directory,"Use --reused-case MU=DIR")
        mu=float(coefficient)
        require(mu in manifest["reused_existing_plant_mu"] and mu not in reused,"Undeclared/duplicate reused coefficient")
        reused[mu]=audit_reused_case(directory,mu,manifest,inputs,effective,settings,task,root)
    commands=[]
    for mu in GRID:
        if mu in reused:continue
        output=Path(output_root).resolve()/("mu"+f"{mu:.3f}".replace(".","p"))
        require(not output.exists(),"Preserve existing planned output: "+str(output))
        argv=[str(python),str(root/"scripts/run_nmpc_tracking.py"),"--model",str(Path(inputs["model"]).resolve()),
            "--motion",str(Path(inputs["motion"]).resolve()),"--contract",str(Path(inputs["actuator_contract"]).resolve()),
            "--reference-id",manifest["reference_id"],"--settings",str(settings),"--task-profile",str(task),
            "--plant-mu",str(mu),"--prediction-mu-scenarios","0.7","--prediction-backend","native",
            "--native-library",str(Path(inputs["native_library"]).resolve()),"--output-dir",str(output)]
        for key,value in manifest["explicit_overrides"].items():argv.extend(("--"+key.replace("_","-"),str(value)))
        commands.append({"plant_mu":mu,"output_dir":str(output),"argv":argv,"environment":{"CUDA_VISIBLE_DEVICES":""},
                         "shell_preview":"CUDA_VISIBLE_DEVICES='' "+shlex.join(argv)})
    return {"kind":"read_only_unqualified_nmpc_friction_reproduction_plan","plan_only":True,"commands_executed":0,
        "manifest_sha256":sha(manifest_path),"effective_cem_configuration_sha256":CONFIG_DIGEST,
        "robot_reference_catalog_identity_verified":True,"robot_reference_file_bytes_read":False,
        "qualified_validation_gate_modified":False,"formal_controller_freeze_claimed":False,"sim2sim_claimed":False,
        "new_command_count":len(commands),"reused_cases":list(reused.values()),"commands":commands,
        "evidence_scope":"Byte-bound completed reports/logs and native clocks, not independent simulator proof or XY success certification; failed/fallen outcomes are not filtered."}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=ROOT/"config/nmpc_friction_development.json")
    for name in ("model","motion","contract","native-library","output-root"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reused-case",action="append",default=[],metavar="MU=DIR")
    return p


if __name__=="__main__":
    args=parser().parse_args()
    result=build_plan(args.manifest,{"model":args.model,"motion":args.motion,"actuator_contract":args.contract,
        "native_library":args.native_library},args.output_root,reused_cases=args.reused_case)
    print(json.dumps(result,indent=2,allow_nan=False))
