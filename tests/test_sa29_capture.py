"""Synthetic/mocked capture tests: NEVER start Isaac, MuJoCo, Torch or GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import sa29_capture as capture


def put(path,data=b"opaque original bytes"):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path


def options(tmp_path,algorithm="PPO",engine="isaac"):
    source=tmp_path/"source";source.mkdir(exist_ok=True)
    return argparse.Namespace(algorithm=algorithm,engine=engine,mu=.7,source_root=source,output_dir=tmp_path/"episode",
        receipt=tmp_path/"evidence/terminal.json",log=tmp_path/"evidence/full.log",device="cuda:2" if engine=="isaac" else "cpu",
        allow_auxiliary_contexts=True,closure_timeout_s=0.,isaac_receipt=None,isaac_log=None,command_json=tmp_path/"command.json")


def argv(args):
    p=lambda name:str((Path(args.source_root)/name).resolve())
    if args.algorithm=="NMPC":
        script=p("scripts/run_nmpc_tracking_isaac.py")
        data={"--source-root":str(args.source_root.resolve()),"--motion":p("motion.npz"),"--asset-path":p("robot.usda"),
            "--mujoco-model":p("model.xml"),"--contract":p("contract.json"),"--reference-id":"doa_sa29_scale2",
            "--task-profile":p("task.json"),"--settings":p("settings.json"),"--plant-mu":"0.7","--prediction-mu-scenarios":"0.7",
            "--horizon-steps":"30","--control-blocks":"10","--candidates":"64","--iterations":"3","--elite-count":"8",
            "--prediction-backend":"native","--native-library":p("existing.so")}
        switches=["--headless"]
    elif args.engine=="isaac":
        script=p("scripts/eval_tracking_friction_isaac.py")
        data={"--motion-file":p("motion.npz"),"--asset-path":p("robot.usda"),"--checkpoint":p("training/model_final.pt"),
            "--runner-config":p("training/runner_config.json"),"--task-profile":p("task.json"),"--evaluation-contact-friction":"0.7"}
        switches=["--headless","--record-physics","--verify-unmasked-device"]
    else:
        script=p("scripts/eval_tracking_mujoco.py")
        data={"--motion-file":p("motion.npz"),"--model":p("model.xml"),"--policy":p("isaac/actor_normalized.pt"),
            "--contract":p("isaac/policy_contract.json"),"--verify-isaac-trajectory":p("isaac/trajectory.npz")}
        switches=["--no-visual-mesh"]
    data.update({"--output-dir":str(args.output_dir.resolve()),"--contact-threshold-n":"5.0"})
    if args.engine=="isaac":
        data.update({"--num-envs":"1","--seed":"42","--device":"cuda:2",
            "--kit_args":"--/renderer/activeGpu=2 --/renderer/multiGpu/enabled=false --/physics/cudaDevice=2 --/physics/sceneMultiGPUMode=0"})
    return [sys.executable,"-u",script]+[token for pair in data.items() for token in pair]+switches


@pytest.mark.parametrize("algorithm,engine",[("PPO","isaac"),("SAC","isaac"),("PPO","mujoco"),("SAC","mujoco"),("NMPC","isaac")])
def test_known_fixed_command_contracts(tmp_path,algorithm,engine):
    args=options(tmp_path,algorithm,engine)
    script,parsed=capture.command_contract(args,argv(args))
    assert script.suffix==".py" and parsed["--output-dir"]==str(args.output_dir)


@pytest.mark.parametrize("extra",[["--max-policy-steps","12"],["--adaptive-contact-config","x"],["--model-nominal-plan","x"],
    ["--allow-smoke-policy"],["--build-native"],["--device","cuda:0"],["--num-envs=1"],["--unknown","x"]])
def test_unsupported_duplicate_truncated_or_build_options_reject(tmp_path,extra):
    args=options(tmp_path)
    with pytest.raises(ValueError):capture.command_contract(args,argv(args)+extra)


@pytest.mark.parametrize("flag,value",[("--seed","43"),("--num-envs","2"),("--evaluation-contact-friction","0.3"),
    ("--contact-threshold-n","0"),("--device","cuda:0"),("--kit_args","--/renderer/activeGpu=0")])
def test_command_cannot_contradict_capture(tmp_path,flag,value):
    args=options(tmp_path);command=argv(args);command[command.index(flag)+1]=value
    with pytest.raises(ValueError):capture.command_contract(args,command)


@pytest.mark.parametrize("prefix",[[sys.executable,"-c","pass"],[sys.executable,"-m","evil"],[sys.executable,"evil.py"]])
def test_decoy_entrypoint_and_interpreter_modes_reject(tmp_path,prefix):
    args=options(tmp_path)
    with pytest.raises(ValueError):capture.command_contract(args,prefix+argv(args)[2:])


@pytest.mark.parametrize("flag,value",[("--iterations","4"),("--horizon-steps","15"),("--prediction-backend","python"),("--prediction-mu-scenarios","0.3")])
def test_nmpc_fixed_knobs_cannot_change(tmp_path,flag,value):
    args=options(tmp_path,"NMPC");command=argv(args);command[command.index(flag)+1]=value
    with pytest.raises(ValueError):capture.command_contract(args,command)


def test_nmpc_multi_prediction_rejects(tmp_path):
    args=options(tmp_path,"NMPC");command=argv(args);command.insert(command.index("--prediction-mu-scenarios")+2,"0.8")
    with pytest.raises(ValueError):capture.command_contract(args,command)


def test_output_existing_rejected(tmp_path):
    args=options(tmp_path);args.output_dir.mkdir()
    with pytest.raises(ValueError,match="Preserve"):capture.command_contract(args,argv(args))


@pytest.fixture
def mocked_run(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    args=options(tmp_path)
    command=argv(args);args.command_json.write_text(json.dumps(command))
    frozen=put(tmp_path/"pinned_input",b"pinned")
    state={"code":0,"log":b"actual mocked evaluator log\n","omit":set(),"exception":None,"poll":None,"popen_error":None,
        "remaining":{},"contexts":[],"closure_error":None,"changed":False,"popen_calls":[],"guard_calls":0}
    monkeypatch.setattr(capture,"preflight",lambda a,c:({},{str(frozen):capture.sha256(frozen)}))
    def snapshot():
        state["guard_calls"]+=1
        return ([{"index":2,"uuid":"synthetic-gpu","name":"NVIDIA RTX synthetic","memory_mib":6.,"utilization_percent":0.}],[])
    monkeypatch.setattr(capture,"gpu_snapshot",snapshot)
    monkeypatch.setattr(capture,"process_snapshot",lambda session,owned:{777:1})
    def closure(session,owned,engine):
        if state["closure_error"]:raise RuntimeError(state["closure_error"])
        return state["remaining"],state["contexts"]
    monkeypatch.setattr(capture,"observe_closure",closure)
    monkeypatch.setattr(capture.time,"sleep",lambda _:None)
    monkeypatch.setattr(capture.policy,"audit_execution",lambda *a,**k:{})
    monkeypatch.setattr(capture,"audit_execution_case",lambda *a,**k:{})
    class Child:
        pid=777
        def wait(self,timeout):
            if state["changed"]:frozen.write_bytes(b"changed while child ran")
            if state["exception"]:raise state["exception"]
            return state["code"]
        def poll(self):return state["poll"]
    def popen(actual,**kwargs):
        state["popen_calls"].append((actual,kwargs))
        if state["popen_error"]:raise state["popen_error"]
        kwargs["stdout"].write(state["log"]);kwargs["stdout"].flush()
        args.output_dir.mkdir()
        roles=capture.NMPC_ROLES if args.algorithm=="NMPC" else capture.POLICY_ROLES[args.engine]
        for role,name in roles.items():
            if role not in state["omit"]:put(args.output_dir/name,b'{"device_audit":{}}' if name=="report.json" else b"opaque synthetic artifact")
        return Child()
    monkeypatch.setattr(capture.subprocess,"Popen",popen)
    return args,state


def test_actual_mocked_process_receipt_and_strict_log_hash(mocked_run):
    args,state=mocked_run;result=capture.run_engine(args)
    assert result["capture_complete"] and result["exit_code"]==0 and result["terminal_status"]=="exited"
    assert result["log_sha256"]==capture.sha256(args.log)
    assert set(result["artifact_sha256"])==set(capture.POLICY_ROLES["isaac"])
    assert result["command"]==json.loads(args.command_json.read_text())
    actual,kwargs=state["popen_calls"][0]
    assert kwargs["start_new_session"] and kwargs["stderr"]==subprocess.STDOUT
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"]=="1"
    assert result["owned_processes_and_contexts_absent"] and not result["tracking_success_claimed"]


@pytest.mark.parametrize("log",[b"Traceback (most recent call last):\nproblem",b"[ERROR] startup",b"CUDA out of memory",b"Segmentation fault",b"Fatal Python error",b"unhandled exception"])
def test_exit_zero_with_error_log_is_not_complete(mocked_run,log):
    args,state=mocked_run;state["log"]=log;result=capture.run_engine(args)
    assert result["exit_code"]==0 and not result["capture_complete"] and result["log_failure_matches"]


def test_nonzero_exit_preserves_genuine_partial_receipt(mocked_run):
    args,state=mocked_run;state.update(code=9,omit={"isaac_trace"})
    result=capture.run_engine(args)
    assert result["exit_code"]==9 and not result["capture_complete"] and "isaac_trace" not in result["artifact_sha256"]


def test_missing_artifact_rejects_even_clean_exit(mocked_run):
    args,state=mocked_run;state["omit"]={"isaac_actor"}
    assert not capture.run_engine(args)["capture_complete"]


def test_invalid_utf8_preserved_not_silently_replaced(mocked_run):
    args,state=mocked_run;state["log"]=b"invalid\xffUTF8"
    result=capture.run_engine(args)
    assert result["strict_utf8_decode_error"] and not result["capture_complete"]
    assert args.log.read_bytes()==state["log"]


def test_changed_input_or_source_rejects(mocked_run):
    args,state=mocked_run;state["changed"]=True
    result=capture.run_engine(args)
    assert not result["source_bindings_unchanged"] and not result["capture_complete"]


@pytest.mark.parametrize("kind",["process","context","observer_error"])
def test_owned_closure_must_be_observed(mocked_run,kind):
    args,state=mocked_run
    if kind=="process":state["remaining"]={778:2}
    elif kind=="context":state["contexts"]=[{"pid":777,"gpu_uuid":"synthetic-gpu"}]
    else:state["closure_error"]="context observer unavailable"
    result=capture.run_engine(args)
    assert not result["capture_complete"] and not result["owned_processes_and_contexts_absent"] and result["operator_action_required"]


def test_keyboard_interrupt_keeps_unclosed_live_pid_and_no_fake_exit(mocked_run):
    args,state=mocked_run;state.update(exception=KeyboardInterrupt(),poll=None,remaining={777:1})
    result=capture.run_engine(args)
    assert result["exit_code"] is None and result["terminal_status"]=="supervisor_interrupted_or_failed_unclosed"
    assert 777 in result["owned_session_pids_remaining"] and result["operator_action_required"] and not result["capture_complete"]
    assert "KeyboardInterrupt" in result["supervisor_error"]


def test_wait_exception_is_not_a_synthetic_success(mocked_run):
    args,state=mocked_run;state.update(exception=RuntimeError("supervisor failed"),poll=0)
    result=capture.run_engine(args)
    assert result["exit_code"]==0 and not result["capture_complete"] and result["supervisor_error"]


def test_popen_failure_preserves_no_actual_exit(mocked_run):
    args,state=mocked_run;state["popen_error"]=OSError("cannot start interpreter")
    result=capture.run_engine(args)
    assert result["exit_code"] is None and not result["capture_complete"] and result["artifact_sha256"]=={}


def test_existing_terminal_evidence_never_overwritten(mocked_run):
    args,state=mocked_run;put(args.receipt,b"keep original")
    with pytest.raises(ValueError,match="Preserve"):capture.run_engine(args)
    assert args.receipt.read_bytes()==b"keep original" and state["popen_calls"]==[]


def test_missing_auxiliary_consent_prevents_gpu_probe_and_launch(mocked_run):
    args,state=mocked_run;args.allow_auxiliary_contexts=False
    with pytest.raises(ValueError,match="consent"):capture.run_engine(args)
    assert not state["guard_calls"] and not state["popen_calls"] and not args.receipt.exists()


def test_masked_ordinal_rejects_before_launch(mocked_run,monkeypatch):
    args,state=mocked_run;monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2")
    with pytest.raises(ValueError,match="unmasked"):capture.run_engine(args)
    assert not state["popen_calls"]


@pytest.mark.parametrize("mutation",["busy_memory","busy_utilization","attached_foreign_process","not_RTX","wrong_ordinal"])
def test_selected_RTX_must_be_genuinely_idle_no_substitution(mutation):
    row={"index":2,"uuid":"selected","name":"NVIDIA RTX synthetic","memory_mib":6.,"utilization_percent":0.};apps=[]
    if mutation=="busy_memory":row["memory_mib"]=64.
    elif mutation=="busy_utilization":row["utilization_percent"]=1.
    elif mutation=="attached_foreign_process":apps=[{"gpu_uuid":"selected","pid":999}]
    elif mutation=="not_RTX":row["name"]="NVIDIA A800"
    else:row["index"]=3
    with pytest.raises(ValueError):capture.idle_guard("cuda:2",([row],apps))


def test_guard_ignores_other_cards_but_does_not_select_them():
    rows=[{"index":2,"uuid":"selected","name":"RTX synthetic","memory_mib":6.,"utilization_percent":0.},
        {"index":0,"uuid":"foreign","name":"RTX other","memory_mib":12000.,"utilization_percent":99.}]
    assert capture.idle_guard("cuda:2",(rows,[{"gpu_uuid":"foreign","pid":999}]))["index"]==2


def test_CPU_masks_cuda_and_observes_actual_closure(mocked_run,monkeypatch):
    args,state=mocked_run;args.engine="mujoco";args.device="cpu";args.allow_auxiliary_contexts=False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","foreign mapping")
    result=capture.run_engine(args)
    assert result["capture_complete"] and state["popen_calls"][0][1]["env"]["CUDA_VISIBLE_DEVICES"]==""
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="foreign mapping"


def test_CPU_context_observer_filters_only_owned_PIDs(monkeypatch):
    monkeypatch.setattr(capture,"process_snapshot",lambda session,owned:{})
    monkeypatch.setattr(capture,"gpu_snapshot",lambda:([],[{"pid":777},{"pid":999}]))
    current,contexts=capture.observe_closure(777,{777:1},"mujoco")
    assert current=={} and contexts==[{"pid":777}]


def test_post_capture_frozen_contract_failure_preserved(mocked_run,monkeypatch):
    args,state=mocked_run
    def reject(*a,**k):raise ValueError("wrong report argv")
    monkeypatch.setattr(capture.policy,"audit_execution",reject)
    result=capture.run_engine(args)
    assert not result["capture_complete"] and "wrong report argv" in result["post_capture_contract_error"]


def test_nmpc_uses_real_report_device_field(mocked_run,monkeypatch):
    args,state=mocked_run;args.algorithm="NMPC";seen=[]
    monkeypatch.setattr(capture,"audit_execution_case",lambda paths,mu,artifacts,device:seen.append(device))
    result=capture.run_engine(args)
    assert result["kind"]=="sa29_nmpc_engine_terminal_receipt" and result["capture_complete"] and seen==[{}]
    assert set(result["artifact_sha256"])=={"report","trace","solver"}


def test_source_preparation_copies_only_nine_plus_separate_helper(tmp_path,monkeypatch):
    source=tmp_path/"upstream";source.mkdir();original={}
    names=list(capture.policy.BASE_SOURCE_SHA256)
    for name in names:original[name]=capture.sha256(put(source/name,("synthetic "+name).encode()))
    helper="training/tron1_terrain.py";helper_digest=capture.sha256(put(source/helper,b"synthetic reporting helper"))
    put(source/"training/not_selected.py",b"must not be copied")
    expected={name:original[name] for name in names}
    for name in ("scripts/eval_tracking_friction_isaac.py","training/tron1_full_horizon_diagnostic.py","training/tron1_scripted_actor_runner.py"):
        expected[name]="unused mock"
    monkeypatch.setattr(capture.policy,"BASE_SOURCE_SHA256",original)
    monkeypatch.setattr(capture.policy,"REPORTING_ONLY_DEPENDENCIES_SHA256",{helper:helper_digest})
    audit_bytes=[]
    def materialize(original_root,output):
        for name in expected:
            put(output/name,(source/name).read_bytes() if (source/name).is_file() else b"synthetic generated source")
        audit={"source_hashes":original,"changed_output_sha256":{str(i):str(i) for i in range(5)}}
        data=json.dumps(audit).encode();put(output/"wide_friction_source.json",data);audit_bytes.append(data)
    monkeypatch.setattr(capture.materialize_controller_source,"materialize",materialize)
    monkeypatch.setattr(capture.policy,"expected_evaluation_sources",lambda:(expected,{}))
    def audit_source(root,audit,hashes,base,**kwargs):
        assert len(json.loads((root/"wide_friction_source.json").read_text())["source_hashes"])==9
        assert (root/helper).read_bytes()==b"synthetic reporting helper"
        assert kwargs["reporting_only_dependencies"]=={helper:helper_digest}
        return {"materialized_code_file_count":12,"runtime_code_file_count":13}
    monkeypatch.setattr(capture.policy,"audit_source",audit_source)
    output=tmp_path/"prepared";result=capture.prepare_source(source,output)
    assert result["status"]=="source_prepared_only"
    assert len([p for p in (output/"original_source").rglob("*") if p.is_file()])==9
    assert (output/"evaluation_source/wide_friction_source.json").read_bytes()==audit_bytes[0]
    assert not (output/"original_source"/helper).exists()
    assert not (output/"evaluation_source/training/not_selected.py").exists()
    assert json.loads((output/"source_binding.json").read_text())["reporting_only_dependencies"]=={helper:helper_digest}
    assert all(capture.sha256(source/name)==digest for name,digest in original.items())


def test_source_preparation_rejects_changed_pins_before_writing(tmp_path,monkeypatch):
    source=tmp_path/"upstream";source.mkdir();put(source/"bad.py")
    monkeypatch.setattr(capture.policy,"BASE_SOURCE_SHA256",{"bad.py":"0"*64})
    output=tmp_path/"prepared"
    with pytest.raises(ValueError,match="Changed"):capture.prepare_source(source,output)
    assert not output.exists()


def test_source_preparation_preserves_existing_output(tmp_path):
    source=tmp_path/"upstream";source.mkdir();output=tmp_path/"prepared";output.mkdir();put(output/"keep",b"keep")
    with pytest.raises(ValueError,match="Preserve"):capture.prepare_source(source,output)
    assert (output/"keep").read_bytes()==b"keep"


@pytest.mark.parametrize("role",["output_dir","receipt","log"])
def test_fresh_run_targets_reject_dangling_symlinks_without_launch(mocked_run,role):
    args,state=mocked_run;target=getattr(args,role)
    target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(target.parent/"missing-target")
    with pytest.raises(ValueError,match="symlink"):capture.run_engine(args)
    assert target.is_symlink() and not state["popen_calls"]


def test_preparation_dangling_symlink_is_preserved(tmp_path):
    source=tmp_path/"upstream";source.mkdir();output=tmp_path/"prepared";output.symlink_to(tmp_path/"missing")
    with pytest.raises(ValueError,match="symlink"):capture.prepare_source(source,output)
    assert output.is_symlink() and not (tmp_path/"missing").exists()


def test_fresh_target_symlink_ancestor_rejects(tmp_path):
    real=tmp_path/"real";real.mkdir();alias=tmp_path/"alias";alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):capture.fresh_path(alias/"new")


def test_preflight_rejects_alternate_adjacent_runner_config(tmp_path,monkeypatch):
    args=options(tmp_path);command=argv(args)
    command[command.index("--runner-config")+1]=str(args.source_root/"training/alternate.json")
    monkeypatch.setattr(capture,"file_hash",lambda path,expected=None:expected or "0"*64)
    monkeypatch.setattr(capture.policy,"expected_evaluation_sources",lambda:({},{}))
    monkeypatch.setattr(capture.policy,"audit_source",lambda *a,**k:{})
    def should_not_read_training(*a,**k):raise AssertionError("Wrong runner must fail before weight/budget work")
    monkeypatch.setattr(capture,"training_bindings",should_not_read_training)
    with pytest.raises(ValueError,match="original adjacent runner_config"):
        capture.preflight(args,command)


def test_nmpc_asset_is_hash_pinned_before_launch(tmp_path,monkeypatch):
    args=options(tmp_path,"NMPC");command=argv(args)
    def synthetic_hash(path,expected=None):
        if Path(path).name=="robot.usda":
            assert expected==capture.ISAAC_ASSET_SHA256
            raise ValueError("synthetic asset check reached")
        return expected or "0"*64
    monkeypatch.setattr(capture,"file_hash",synthetic_hash)
    with pytest.raises(ValueError,match="synthetic asset check"):
        capture.preflight(args,command)


def test_receipt_not_complete_until_frozen_postcheck_returns(mocked_run,monkeypatch):
    args,state=mocked_run
    def check(*a,**k):
        assert json.loads(args.receipt.read_text())["capture_complete"] is False
    monkeypatch.setattr(capture.policy,"audit_execution",check)
    assert capture.run_engine(args)["capture_complete"]


def test_session_observer_tracks_own_descendants_not_foreign_reused_pids(tmp_path,monkeypatch):
    proc=tmp_path/"proc";proc.mkdir()
    def process(pid,ppid,session,ticks):
        fields=["0"]*20;fields[0]="R";fields[1]=str(ppid);fields[3]=str(session);fields[19]=str(ticks)
        put(proc/str(pid)/"stat",(str(pid)+" (synthetic worker) "+" ".join(fields)).encode())
    process(777,1,777,10);process(778,777,777,11);process(779,778,779,12)
    process(999,1,999,20);process(880,1,880,99);process(881,880,880,100)
    original_path=capture.Path
    monkeypatch.setattr(capture,"Path",lambda value:proc if value=="/proc" else original_path(value))
    observed=capture.process_snapshot(777,{777:10,880:15})
    assert observed=={777:10,778:11,779:12}
