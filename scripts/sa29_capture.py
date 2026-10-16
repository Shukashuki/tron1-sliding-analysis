"""Prepare SA29 source or capture ONE explicit local evaluation process.

This new utility was NOT used by the historical experiment queue. It does not
train, choose/share a GPU, use SSH, queue/retry jobs, build native code, load a
model/actor, or synthesize receipts from old reports. run-engine really starts
the supplied frozen evaluator: use it only with your own execution authority.

prepare-source --source-root FRESH_UPSTREAM --output-dir NEW creates
original_source/ (the nine pinned originals), evaluation_source/ (the unchanged
materializer's twelve code files, original audit, and separately pinned terrain
reporting helper), and source_binding.json for the portable assembly plan.

run-engine requires an explicit JSON ARRAY of argv strings, matching output,
algorithm/engine/mu/device, and fresh receipt/log paths OUTSIDE that output.
Isaac additionally requires an unmasked, freshly idle selected RTX and explicit
--allow-auxiliary-contexts. MuJoCo masks CUDA and requires its original prior
Isaac receipt/log. Failed/incomplete executions retain truthful receipts/logs.
Interruption never signals foreign processes: an unclosed owned session is
recorded with live PIDs and operator_action_required, not a fabricated exit.

Receipts establish consistency of observed trusted execution, not independent
authenticity, GPU reservation/exclusivity, or physical tracking success.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import analyze_sa29_policy_friction_dynamics as policy
from analyze_controller_pair import read_json, sha256
from analyze_sa29_controller_comparison import ISAAC_ASSET_SHA256, TERMINAL_ERROR_PATTERN, audit_execution_case
from controller_references import ROOT, get_reference
import materialize_controller_source

NMPC_ENTRY_SHA256 = "494387154c15cba786f771356a38d1a51514821112e055cb50cad462d6eabb09"
POLICY_ROLES = {"isaac":{"isaac_trace":"physics_trajectory.npz","isaac_report":"report.json",
    "isaac_policy_trace":"trajectory.npz","isaac_actor":"actor_normalized.pt","isaac_contract":"policy_contract.json"},
    "mujoco":{"mujoco_trace":"rollout.npz","mujoco_report":"report.json"}}
NMPC_ROLES = {"report":"report.json","trace":"rollout.npz","solver":"solver_log.json"}
TRAIN_NAMES = {"manifest":"manifest.json","status":"run_status.json","checkpoint":"model_final.pt",
    "runner_config":"runner_config.json","policy_contract":"policy_contract.json","actor":"actor_normalized.pt"}


def require(condition, message):
    if not condition: raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path, expected=None):
    path=Path(path)
    require(path.is_file() and path.stat().st_size>0,"Missing original file: "+str(path))
    value=sha256(path)
    require(expected is None or value==expected,"Changed pinned file: "+path.name)
    return value


def binding(path, base):
    path,base=Path(path).resolve(),Path(base).resolve()
    require(path.is_relative_to(base),"Capture binding escapes its root")
    return {"path":str(path.relative_to(base)),"sha256":file_hash(path)}


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def fresh_path(path):
    raw=Path(path).absolute()
    require(not any(part.is_symlink() for part in (raw,*raw.parents)),"Preserve existing symlink, including dangling targets")
    require(not raw.exists(),"Preserve existing capture/preparation target")
    return raw.resolve()


def save_receipt(path, value):
    # Only this process's exclusively reserved receipt is updated.
    with Path(path).open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def prepare_source(source_root, output_dir):
    source,output=Path(source_root).resolve(),fresh_path(output_dir)
    require(source!=output and source not in output.parents and output not in source.parents,"Preparation must be separate from upstream")
    for name,digest in {**policy.BASE_SOURCE_SHA256,**policy.REPORTING_ONLY_DEPENDENCIES_SHA256}.items():file_hash(source/name,digest)
    output.mkdir(parents=True)
    original=output/"original_source"
    for name in policy.BASE_SOURCE_SHA256:
        target=original/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    generated=output/"evaluation_source"
    materialize_controller_source.materialize(original,generated)
    for name in policy.REPORTING_ONLY_DEPENDENCIES_SHA256:
        shutil.copyfile(source/name,generated/name)
    expected,_=policy.expected_evaluation_sources()
    source_spec={"root":"evaluation_source","audit":binding(generated/"wide_friction_source.json",output),
        "source_sha256":expected,"reporting_only_dependencies":dict(policy.REPORTING_ONLY_DEPENDENCIES_SHA256)}
    evidence=policy.audit_source(generated,source_spec["audit"],expected,output,
        reporting_only_dependencies=source_spec["reporting_only_dependencies"])
    write_new(output/"source_binding.json",source_spec)
    return {"status":"source_prepared_only","source_binding":str(output/"source_binding.json"),"source":evidence,
        "simulation_started":False,"training_source_modified":False}


def parse_argv(command, valued, switches):
    require(isinstance(command,list) and len(command)>=2 and all(isinstance(x,str) and x for x in command),"Command JSON must be an argv array")
    require(re.fullmatch(r"python(?:3(?:\.[0-9]+)?)?(?:\.exe)?",Path(command[0]).name) is not None,"Use an explicit Python interpreter")
    require(Path(command[0]).is_absolute() and os.access(command[0],os.X_OK),"Python interpreter must be an executable absolute path")
    operand=1
    while operand<len(command) and command[operand] in ("-u","-B"):operand+=1
    require(operand<len(command) and not command[operand].startswith("-"),"Require a direct script, never -c/-m or a launcher")
    options={};index=operand+1
    while index<len(command):
        flag=command[index]
        require(flag in valued or flag in switches,"Unsupported/truncation/adaptation/build option: "+flag)
        require(flag not in options,"Duplicate command option: "+flag)
        if flag in switches:options[flag]=True;index+=1
        else:
            require(index+1<len(command),"Missing command option value: "+flag)
            options[flag]=command[index+1];index+=2
    return Path(command[operand]),options


def canonical(value):
    path=Path(value)
    require(path.is_absolute() and str(path)==str(path.resolve()),"Command file paths must be absolute/canonical")
    return path


def command_contract(args, command):
    require(args.mu in policy.GRID and args.algorithm in ("PPO","SAC","NMPC"),"Undeclared SA29 algorithm/coefficient")
    require(args.engine in ("isaac","mujoco") and not (args.algorithm=="NMPC" and args.engine!="isaac"),"Only NMPC Isaac capture is supported")
    if args.algorithm=="NMPC":
        valued={"--source-root","--motion","--asset-path","--mujoco-model","--contract","--output-dir","--reference-id","--task-profile","--settings",
            "--plant-mu","--prediction-mu-scenarios","--num-envs","--seed","--contact-threshold-n","--horizon-steps","--control-blocks",
            "--candidates","--iterations","--elite-count","--prediction-backend","--native-library","--device","--kit_args"}
        switches={"--headless"};entry="run_nmpc_tracking_isaac.py"
    elif args.engine=="isaac":
        valued={"--motion-file","--asset-path","--checkpoint","--runner-config","--task-profile","--evaluation-contact-friction","--output-dir",
            "--num-envs","--seed","--contact-threshold-n","--device","--kit_args"}
        switches={"--record-physics","--verify-unmasked-device","--headless"};entry="eval_tracking_friction_isaac.py"
    else:
        valued={"--policy","--contract","--motion-file","--model","--output-dir","--verify-isaac-trajectory","--contact-threshold-n"}
        switches={"--no-visual-mesh"};entry="eval_tracking_mujoco.py"
    script,options=parse_argv(command,valued,switches)
    required=set(valued)|switches
    if args.engine=="mujoco":required-={"--contact-threshold-n"}
    require(required<=set(options),"Missing fixed explicit evaluator options")
    require(script.is_absolute() and script.name==entry and sum(Path(x).name==entry for x in command)==1,"Wrong direct frozen entrypoint")
    output=fresh_path(args.output_dir)
    require(canonical(options["--output-dir"])==output and not output.exists(),"Preserve existing or mismatched episode output")
    require(float(options.get("--contact-threshold-n",5.))==5.,"Measured contact threshold must remain5N")
    if args.engine=="isaac":
        require(options["--num-envs"]=="1" and options["--seed"]=="42" and options["--device"]==args.device,"Require one seed42 environment and explicit selected device")
        require(re.fullmatch(r"cuda:(0|[1-9][0-9]*)",args.device) is not None,"Use an explicit unmasked CUDA ordinal")
        index=int(args.device.split(":")[1]);kit=shlex.split(options["--kit_args"])
        require(len(kit)==4 and set(kit)=={f"--/renderer/activeGpu={index}","--/renderer/multiGpu/enabled=false",
            f"--/physics/cudaDevice={index}","--/physics/sceneMultiGPUMode=0"},"Require four matching explicit Kit settings")
    else:require(args.device=="cpu","MuJoCo capture is CPU-only")
    mu_flag="--plant-mu" if args.algorithm=="NMPC" else "--evaluation-contact-friction"
    if args.engine=="isaac":require(float(options[mu_flag])==args.mu,"Command/capture friction differs")
    if args.algorithm=="NMPC":
        fixed={"--reference-id":policy.REFERENCE_ID,"--prediction-mu-scenarios":"0.7","--horizon-steps":"30","--control-blocks":"10",
            "--candidates":"64","--iterations":"3","--elite-count":"8","--prediction-backend":"native"}
        require(all(options[k]==v for k,v in fixed.items()),"NMPC must keep the fixed singleton-prediction H30/B10/C64/I3/E8 recipe")
        require(canonical(options["--source-root"])==Path(args.source_root).resolve(),"Explicit original source root differs")
    else:require(script.resolve()==Path(args.source_root).resolve()/"scripts"/entry,"Entry must belong to supplied isolated source")
    return script.resolve(),options


def training_bindings(checkpoint, algorithm):
    directory=checkpoint.parent
    roles={"manifest","status","checkpoint","runner_config"}|({"policy_contract","actor"} if algorithm=="SAC" else set())
    paths={name:directory/TRAIN_NAMES[name] for name in roles}
    require(checkpoint==paths["checkpoint"],"Use the original final checkpoint filename")
    hashes={str(path):file_hash(path) for path in paths.values()}
    training,status,runner=[read_json(paths[name]) for name in ("manifest","status","runner_config")]
    contract=training.get("policy_contract") if algorithm=="PPO" else read_json(paths["policy_contract"])
    policy.audit_training_budget(algorithm,training,status,runner,contract,hashes[str(checkpoint)],hashes[str(paths["actor"])] if algorithm=="SAC" else None)
    return hashes


def preflight(args, command):
    script,options=command_contract(args,command);source=Path(args.source_root).resolve();hashes={}
    def add(path,expected=None):
        path=Path(path).resolve();hashes[str(path)]=file_hash(path,expected)
    declaration=read_json(ROOT/"config/nmpc_sa29_nominal_development.json")
    if args.algorithm=="NMPC":
        add(script,NMPC_ENTRY_SHA256);repo=script.parent.parent
        for name,digest in declaration["source_sha256"].items():add(repo/name,digest)
        for name,digest in policy.BASE_SOURCE_SHA256.items():add(source/name,digest)
        add(canonical(options["--contract"]),declaration["input_sha256"]["actuator_contract"])
        library=canonical(options["--native-library"]);add(library);add(library.with_suffix(".json"))
        motion=canonical(options["--motion"]);add(canonical(options["--mujoco-model"]),declaration["input_sha256"]["model"])
    else:
        expected,_=policy.expected_evaluation_sources()
        for name,digest in {**expected,**policy.REPORTING_ONLY_DEPENDENCIES_SHA256}.items():add(source/name,digest)
        audit=source/"wide_friction_source.json";add(audit)
        policy.audit_source(source,binding(audit,source),expected,source,reporting_only_dependencies=policy.REPORTING_ONLY_DEPENDENCIES_SHA256)
        motion=canonical(options["--motion-file"])
        if args.engine=="isaac":
            checkpoint=canonical(options["--checkpoint"])
            require(canonical(options["--runner-config"])==checkpoint.with_name("runner_config.json"),"Runner must be the original adjacent runner_config.json")
            hashes.update(training_bindings(checkpoint,args.algorithm))
            training=read_json(checkpoint.with_name("manifest.json"))
            add(canonical(options["--asset-path"]),training["asset_sha256"])
            add(canonical(options["--runner-config"]))
            require(canonical(options["--runner-config"]).parent==canonical(options["--checkpoint"]).parent,"Runner must belong to checkpoint")
        else:
            require(args.isaac_receipt is not None and args.isaac_log is not None,"MuJoCo needs original prior Isaac receipt/log")
            actor=canonical(options["--policy"]);contract=canonical(options["--contract"]);trajectory=canonical(options["--verify-isaac-trajectory"])
            require(actor.parent==contract.parent==trajectory.parent and actor.name=="actor_normalized.pt" and contract.name=="policy_contract.json"
                and trajectory.name=="trajectory.npz","Reuse original adjacent Isaac actor/contract/trajectory")
            files={name:actor.parent/filename for name,filename in POLICY_ROLES["isaac"].items()}
            common=Path(os.path.commonpath([str(Path(args.isaac_receipt).resolve().parent),str(Path(args.isaac_log).resolve().parent)]))
            evidence={"receipt":binding(args.isaac_receipt,common),"log":binding(args.isaac_log,common)}
            policy.audit_execution(evidence,args.algorithm,args.mu,"isaac",files,common)
            prior=read_json(args.isaac_receipt)
            require(prior.get("capture_complete") is prior.get("owned_processes_and_contexts_absent") is prior.get("source_bindings_unchanged") is True,"Prior Isaac execution is incomplete/unclosed")
            exported=read_json(contract)
            require(exported.get("terrain") is None and exported.get("policy_algorithm")==args.algorithm
                and exported.get("full_horizon_diagnostic") is True
                and exported.get("evaluation_contact_friction_override",{}).get("evaluation_contact_friction")==args.mu,"Prior actor/contract is not selected flat full-horizon fixed-mu episode")
            for path in (*files.values(),Path(args.isaac_receipt),Path(args.isaac_log)):add(path)
            add(canonical(options["--model"]),declaration["input_sha256"]["model"])
    add(motion,get_reference(policy.REFERENCE_ID)["motion_npz_sha256"])
    if args.engine=="isaac":
        add(canonical(options["--asset-path"]),ISAAC_ASSET_SHA256 if args.algorithm=="NMPC" else training["asset_sha256"])
        add(canonical(options["--task-profile"]),file_hash(ROOT/"config/car_drift_height_smooth_task.json"))
        if args.algorithm=="NMPC":add(canonical(options["--settings"]),file_hash(ROOT/"config/nmpc_height_smooth_residual.json"))
    bm_value=os.environ.get("TRON1_BEYONDMIMIC_ROOT")
    require(bm_value,"Set TRON1_BEYONDMIMIC_ROOT to the full pinned checkout, not a two-file stub")
    bm=Path(bm_value).resolve()
    require((bm/"source/whole_body_tracking/setup.py").is_file(),"Complete BeyondMimic package is missing")
    for name,digest in declaration["source_sha256"].items():
        prefix="upstream/whole_body_tracking/"
        if name.startswith(prefix):add(bm/name[len(prefix):],digest)
    for path in (bm/"source/whole_body_tracking").rglob("*"):
        if path.is_file() and path.suffix in (".py",".toml",".rst"):add(path)
    return options,hashes


def gpu_snapshot():
    """Read only; no torch/CUDA import and no GPU context is created."""
    if shutil.which("nvidia-smi") is None:
        require(not list(Path("/dev").glob("nvidia[0-9]*")),"GPU devices exist but their context observer is unavailable")
        return [],[]
    rows=csv.reader(subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines())
    apps=csv.reader(subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],text=True).splitlines())
    return ([{"index":int(r[0]),"uuid":r[1].strip(),"name":r[2].strip(),"memory_mib":float(r[3]),"utilization_percent":float(r[4])} for r in rows],
        [{"gpu_uuid":r[0].strip(),"pid":int(r[1]),"process_name":r[2].strip(),"memory_mib":r[3].strip()} for r in apps])


def idle_guard(device, snapshot):
    index=int(device.split(":")[1]);rows,apps=snapshot
    matches=[row for row in rows if row["index"]==index]
    require(len(matches)==1,"Selected GPU ordinal is unavailable")
    row=matches[0]
    require("RTX" in row["name"].upper() and row["memory_mib"]<64 and row["utilization_percent"]==0
        and not any(app["gpu_uuid"]==row["uuid"] for app in apps),"Selected RTX is not freshly idle; no launch or GPU substitution")
    return {**row,"observed_utc":now(),"attached_compute_processes":[],"scope":"Selected ordinal only; not a reservation or exclusivity guarantee"}


def process_snapshot(session, owned):
    """Track own session and previously observed descendants by PID/start time."""
    table={}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():continue
        try:
            fields=(directory/"stat").read_text().rsplit(") ",1)[1].split()
            table[int(directory.name)]={"ppid":int(fields[1]),"session":int(fields[3]),"start_ticks":int(fields[19])}
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    current={pid:value["start_ticks"] for pid,value in table.items() if value["session"]==session}
    live_owned={pid for pid,ticks in owned.items() if pid in table and table[pid]["start_ticks"]==ticks}
    frontier=live_owned|set(current)|{session}
    changed=True
    while changed:
        changed=False
        for pid,value in table.items():
            if value["ppid"] in frontier and pid not in frontier:
                frontier.add(pid);current[pid]=value["start_ticks"];changed=True
    for pid,ticks in owned.items():
        if pid in table and table[pid]["start_ticks"]==ticks:current[pid]=ticks
    return current


def observe_closure(session, owned, engine):
    current=process_snapshot(session,owned);owned.update(current)
    contexts=[app for app in gpu_snapshot()[1] if app["pid"] in owned]
    return current,contexts


def unchanged(hashes):
    try:return all(file_hash(path)==digest for path,digest in hashes.items())
    except (OSError,ValueError):return False


def run_engine(args):
    output=fresh_path(args.output_dir);receipt_path=fresh_path(args.receipt);log_path=fresh_path(args.log)
    command=json.loads(Path(args.command_json).read_text(encoding="utf-8"))
    options,hashes=preflight(args,command)
    require(receipt_path!=log_path and not receipt_path.is_relative_to(output) and not log_path.is_relative_to(output),"Receipt/log must be distinct and outside fresh evaluator output")
    require(not receipt_path.exists() and not log_path.exists(),"Preserve original terminal evidence")
    require(Path("/proc/self/stat").is_file(),"Owned-session observation requires Linux /proc")
    require(0<=args.closure_timeout_s<=30,"Closure observation must be bounded to30s")
    guard=None
    if args.engine=="isaac":
        require(args.allow_auxiliary_contexts,"Explicit --allow-auxiliary-contexts consent required; other visible GPUs may receive startup contexts")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") is None,"Isaac requires explicit unmasked ordinals, not CUDA_VISIBLE_DEVICES remapping")
        guard=idle_guard(args.device,gpu_snapshot())
    else:gpu_snapshot()  # Establish actual context-observer availability before CPU launch too.
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    if args.engine=="mujoco":env["CUDA_VISIBLE_DEVICES"]=""
    roles=NMPC_ROLES if args.algorithm=="NMPC" else POLICY_ROLES[args.engine]
    receipt={"schema_version":1,"kind":"sa29_nmpc_engine_terminal_receipt" if args.algorithm=="NMPC" else "sa29_policy_engine_terminal_receipt",
        "algorithm":args.algorithm,"engine":args.engine,"mu":args.mu,"reference_id":policy.REFERENCE_ID,"command":command,
        "requested_device":args.device,"terminal_status":"starting","exit_code":None,"started_utc":now(),"gpu_guard":guard,
        "auxiliary_contexts_approved":bool(args.allow_auxiliary_contexts),"source_input_sha256":hashes,"capture_complete":False,
        "capture_utility_sha256":sha256(Path(__file__)),"historical_queue_used_this_utility":False,"tracking_success_claimed":False}
    receipt_path.parent.mkdir(parents=True,exist_ok=True);log_path.parent.mkdir(parents=True,exist_ok=True)
    write_new(receipt_path,receipt)
    child=None;owned={};current={};contexts=[];code=None;supervisor_error=None;closure_error=None
    try:
        with log_path.open("xb") as stream:
            require(not output.exists(),"Output appeared after preflight; no evaluator launched")
            child=subprocess.Popen(command,cwd=Path(args.source_root).resolve(),env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            owned={child.pid:None}
            receipt.update(terminal_status="running",app_pid=child.pid,supervisor_pid=os.getpid());save_receipt(receipt_path,receipt)
            while True:
                current=process_snapshot(child.pid,owned);owned.update(current)
                try:code=child.wait(timeout=.5);break
                except subprocess.TimeoutExpired:pass
    except (Exception,KeyboardInterrupt) as error:
        supervisor_error=type(error).__name__+": "+str(error)
        if child is not None:
            try:code=child.poll()
            except Exception as error:supervisor_error+="; poll: "+type(error).__name__+": "+str(error)
    if child is not None:
        deadline=time.monotonic()+args.closure_timeout_s
        try:
            while True:
                current,contexts=observe_closure(child.pid,owned,args.engine)
                if not current and not contexts or time.monotonic()>=deadline:break
                time.sleep(.2)
        except (Exception,KeyboardInterrupt) as error:closure_error=type(error).__name__+": "+str(error)
    else:closure_error="No child started; no actual terminal exit exists"
    data=log_path.read_bytes() if log_path.is_file() else b"";decode_error=None
    try:text=data.decode("utf-8")
    except UnicodeDecodeError as error:text="";decode_error=str(error)
    markers=[m.group() for m in re.finditer(TERMINAL_ERROR_PATTERN,text,re.I)]
    artifacts={}
    for role,name in roles.items():
        path=output/name
        if path.is_file() and path.stat().st_size>0:artifacts[role]=sha256(path)
    closed=child is not None and code is not None and not current and not contexts and closure_error is None
    receipt.update(terminal_status="exited" if code is not None else "supervisor_interrupted_or_failed_unclosed",exit_code=code,
        finished_utc=now(),log_sha256=sha256(log_path) if log_path.is_file() else None,log_bytes=len(data),strict_utf8_decode_error=decode_error,
        log_failure_matches=markers,artifact_sha256=artifacts,source_bindings_unchanged=unchanged(hashes),
        owned_pids_observed=sorted(set(owned)|({child.pid} if child is not None else set())),
        owned_session_pids_remaining=sorted(set(current)|({child.pid} if child is not None and code is None else set())),
        owned_compute_contexts_remaining=contexts,owned_processes_and_contexts_absent=closed,
        supervisor_error=supervisor_error,closure_observation_error=closure_error,operator_action_required=not closed)
    candidate_complete=(type(code) is int and code==0 and supervisor_error is None and decode_error is None and not markers
        and set(artifacts)==set(roles) and closed and receipt["source_bindings_unchanged"])
    # Never expose completion before the frozen receipt/report contract check.
    receipt["capture_complete"]=False
    save_receipt(receipt_path,receipt)
    if candidate_complete:
        try:
            if args.algorithm=="NMPC":
                report=read_json(output/"report.json")
                audit_execution_case({"receipt":receipt_path,"log":log_path},args.mu,artifacts,report["device_audit"])
            else:
                base=Path(os.path.commonpath([str(receipt_path.parent),str(log_path.parent)]))
                evidence={"receipt":binding(receipt_path,base),"log":binding(log_path,base)}
                policy.audit_execution(evidence,args.algorithm,args.mu,args.engine,{role:output/name for role,name in roles.items()},base)
            receipt["capture_complete"]=True;save_receipt(receipt_path,receipt)
        except (Exception,KeyboardInterrupt) as error:
            receipt.update(capture_complete=False,post_capture_contract_error=type(error).__name__+": "+str(error));save_receipt(receipt_path,receipt)
    return receipt


def parser():
    result=argparse.ArgumentParser(description=__doc__);sub=result.add_subparsers(dest="subcommand",required=True)
    prepare=sub.add_parser("prepare-source")
    prepare.add_argument("--source-root",type=Path,required=True);prepare.add_argument("--output-dir",type=Path,required=True)
    run=sub.add_parser("run-engine",help="Actually execute ONE explicitly authorized local evaluator; not a replay/offline subcommand")
    run.add_argument("--algorithm",choices=("PPO","SAC","NMPC"),required=True);run.add_argument("--engine",choices=("isaac","mujoco"),required=True)
    run.add_argument("--mu",type=float,required=True);run.add_argument("--device",required=True)
    for name in ("source-root","output-dir","receipt","log","command-json"):run.add_argument("--"+name,type=Path,required=True)
    run.add_argument("--isaac-receipt",type=Path);run.add_argument("--isaac-log",type=Path)
    run.add_argument("--allow-auxiliary-contexts",action="store_true");run.add_argument("--closure-timeout-s",type=float,default=10.)
    return result


def main():
    args=parser().parse_args()
    if args.subcommand=="prepare-source":result=prepare_source(args.source_root,args.output_dir)
    else:result=run_engine(args)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return 0 if args.subcommand=="prepare-source" or result["capture_complete"] else 1


if __name__=="__main__":raise SystemExit(main())
