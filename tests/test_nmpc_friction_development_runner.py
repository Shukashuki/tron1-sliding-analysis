"""Pure read-only planner tests; no simulators, NumPy, assets or subprocesses."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_nmpc_friction_development as planner


@pytest.fixture
def case(tmp_path):
    repo=tmp_path/"repository";repo.mkdir()
    manifest=json.loads((ROOT/"config/nmpc_friction_development.json").read_text())
    for relative in planner.SOURCES:
        target=repo/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    inputs={}
    for role in ("model","motion","actuator_contract","native_library"):
        inputs[role]=tmp_path/(role+".fixture")
        inputs[role].write_bytes(("synthetic "+role).encode())
        manifest["input_sha256"][role]=planner.sha(inputs[role])
    catalog_path=repo/"config/controller_references.json"
    catalog=json.loads(catalog_path.read_text());spec=catalog["references"]["drift_drl_traj4"]
    spec["motion_npz_sha256"]=manifest["input_sha256"]["motion"]
    catalog_path.write_text(json.dumps(catalog))
    manifest["source_sha256"]={relative:planner.sha(repo/relative) for relative in planner.SOURCES}
    manifest_path=tmp_path/"manifest.json";manifest_path.write_text(json.dumps(manifest))
    return {"root":repo,"manifest":manifest,"path":manifest_path,"inputs":inputs,"output":tmp_path/"planned"}


def plan(case,**kwargs):
    return planner.build_plan(case["path"],case["inputs"],case["output"],repo_root=case["root"],**kwargs)


def write_manifest(case):case["path"].write_text(json.dumps(case["manifest"]))


def reused(case,mu):
    folder=case["root"].parent/("reused_"+str(mu));folder.mkdir()
    header=repr({"descr":"<f8","fortran_order":False,"shape":(1397,)}).encode()+b"\n"
    with zipfile.ZipFile(folder/"rollout.npz","w") as archive:
        archive.writestr("time_s.npy",b"\x93NUMPY\x01\x00"+len(header).to_bytes(2,"little")+header
            +struct.pack("<1397d",*(k*.005 for k in range(1397))))
    rows=[{"frame":k,"reference_frame_for_reward":k,"time_s":(k+1)*.02} for k in range(349)]
    (folder/"solver_log.json").write_text(json.dumps(rows))
    effective=json.loads((case["root"]/case["manifest"]["settings"]).read_text())["nmpc"]
    effective.update(case["manifest"]["explicit_overrides"])
    report={"kind":"full_body_height_smooth_cem_nmpc","engine":"MuJoCo","status":"completed",
        "plant_mu":mu,"prediction_mu_scenarios":[.7],"effective_cem_config":effective,
        "model_sha256":case["manifest"]["input_sha256"]["model"],"motion_sha256":case["manifest"]["input_sha256"]["motion"],
        "reference_id":"drift_drl_traj4","reference_catalog_sha256":planner.sha(case["root"]/"config/controller_references.json"),
        "completed_policy_steps":349,"evaluated_policy_steps":349,"total_reference_policy_steps":349,
        "duration_s":6.98,"physics_dt_s":.005,"control_dt_s":.02,
        "source_sha256":{"/archived/"+key:value for key,value in case["manifest"]["source_sha256"].items()},
        "model_generated_nominal_plan":{"enabled":False},"prediction_backend":"native",
        "plant_material_readback":{"compiled_sliding_friction_range":[mu,mu]},
        "native_prediction_backend":{"production_cem_wired":True,"workers":{"0.7":{
            "native_library_sha256":case["manifest"]["input_sha256"]["native_library"]}}},
        # Descriptive failures must not make matching evidence ineligible.
        "core_xy_rmse_m":4.,"core_xy_peak_m":5.,"max_tilt_deg":130.}
    report.update({field:False for field in ("truncated_smoke","prediction_receives_plant_mu","learned_policy_used",
        "tvlqr_used","old_qr_used","root_state_prescribed_after_reset","external_root_forces_used","pose_height_failure_gate")})
    report["source_sha256"]["/archive/policy_contract.json"]=case["manifest"]["input_sha256"]["actuator_contract"]
    seal(folder,report)
    return folder,report


def seal(folder,report):
    report["rollout_sha256"]=planner.sha(folder/"rollout.npz")
    report["solver_log_sha256"]=planner.sha(folder/"solver_log.json")
    (folder/"report.json").write_text(json.dumps(report))


def test_default_plan_has_seventeen_commands_no_execution_or_output_creation(case):
    before=set(case["root"].parent.rglob("*"));result=plan(case)
    assert result["plan_only"] is True and result["commands_executed"]==0
    assert result["new_command_count"]==17 and result["reused_cases"]==[]
    assert [row["plant_mu"] for row in result["commands"]]==planner.GRID
    assert set(case["root"].parent.rglob("*"))==before and not case["output"].exists()
    assert result["effective_cem_configuration_sha256"]==planner.CONFIG_DIGEST
    assert result["robot_reference_catalog_identity_verified"] is True
    assert result["robot_reference_file_bytes_read"] is False


def test_two_completed_reused_cases_produce_fifteen_commands_without_success_filter(case):
    low,_=reused(case,.3);nominal,_=reused(case,.7)
    result=plan(case,reused_cases=[f".3={low}",f".7={nominal}"])
    assert result["new_command_count"]==15
    assert {row["plant_mu"] for row in result["commands"]}==set(planner.GRID)-{.3,.7}
    for record in result["reused_cases"]:
        assert record["xy_success_required_for_reuse"] is False
        assert record["physical_samples"]==1397 and record["completed_policy_steps"]==349
        for name,digest in record["input_sha256"].items():assert planner.sha(Path(record["directory"])/name)==digest


def test_every_command_preserves_fixed_model_native_full_episode_and_locked_overrides(case):
    for row in plan(case)["commands"]:
        argv=row["argv"];options=dict(zip(argv[2::2],argv[3::2]))
        assert options["--prediction-mu-scenarios"]=="0.7"
        assert options["--prediction-backend"]=="native"
        assert options["--reference-id"]=="drift_drl_traj4"
        assert options["--horizon-steps"]=="30" and options["--control-blocks"]=="10"
        assert options["--candidates"]=="64" and options["--iterations"]=="3" and options["--elite-count"]=="8"
        assert not {"--max-policy-steps","--adaptive-contact-config","--model-nominal-plan"}&set(argv)
        assert row["environment"]=={"CUDA_VISIBLE_DEVICES":""}
        assert "CUDA_VISIBLE_DEVICES='' " in row["shell_preview"]


@pytest.mark.parametrize("role",["model","motion","actuator_contract","native_library"])
def test_any_runtime_input_byte_change_rejected(case,role):
    case["inputs"][role].write_bytes(b"changed")
    with pytest.raises(ValueError,match="Input byte identity"):plan(case)


@pytest.mark.parametrize("relative",["scripts/nmpc_shooting.py","config/nmpc_height_smooth_residual.json","config/car_drift_height_smooth_task.json"])
def test_runtime_or_configuration_source_byte_change_rejected(case,relative):
    (case["root"]/relative).write_bytes(b"changed")
    with pytest.raises(ValueError,match="Current source identity"):plan(case)


def test_effective_configuration_retuning_rejected_even_with_a_resealed_settings_file(case):
    settings=case["root"]/case["manifest"]["settings"]
    value=json.loads(settings.read_text());value["nmpc"]["initial_std"][0]=.02
    settings.write_text(json.dumps(value));case["manifest"]["source_sha256"][case["manifest"]["settings"]]=planner.sha(settings)
    write_manifest(case)
    with pytest.raises(ValueError,match="Effective CEM configuration"):plan(case)


@pytest.mark.parametrize("change",["unqualified","oracle","source_missing","unsafe_path"])
def test_scope_model_or_binding_changes_rejected(case,change):
    manifest=case["manifest"]
    if change=="unqualified":manifest["formal_controller_freeze_claimed"]=True
    if change=="oracle":manifest["prediction_mu_scenarios"]=[.3]
    if change=="source_missing":del manifest["source_sha256"]["config/car_drift_height_smooth_task.json"]
    if change=="unsafe_path":manifest["settings"]="../outside.json"
    write_manifest(case)
    with pytest.raises(ValueError):plan(case)


@pytest.mark.parametrize("change",["status","input","archive","config","phase","native","material"])
def test_reused_evidence_must_be_complete_and_bound_but_need_not_pass_xy(case,change):
    folder,report=reused(case,.7)
    if change=="status":report["status"]="running"
    if change=="input":report["model_sha256"]="f"*64
    if change=="archive":report["rollout_sha256"]="f"*64
    if change=="config":report["effective_cem_config"]["iterations"]=6
    if change=="phase":
        rows=json.loads((folder/"solver_log.json").read_text());rows[17]["reference_frame_for_reward"]=18
        (folder/"solver_log.json").write_text(json.dumps(rows));seal(folder,report)
    if change=="native":report["native_prediction_backend"]["workers"]["0.7"]["native_library_sha256"]="f"*64
    if change=="material":report["plant_material_readback"]["compiled_sliding_friction_range"]=[.3,.3]
    (folder/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):plan(case,reused_cases=[f".7={folder}"])


def test_existing_outputs_duplicate_reuse_and_nonpredeclared_reuse_are_preserved(case):
    folder,_=reused(case,.7)
    with pytest.raises(ValueError,match="duplicate"):plan(case,reused_cases=[f".7={folder}",f".7={folder}"])
    with pytest.raises(ValueError,match="Undeclared"):plan(case,reused_cases=[f".1={folder}"])
    output=case["output"]/"mu0p010";output.mkdir(parents=True)
    marker=output/"keep.txt";marker.write_text("preserve")
    with pytest.raises(ValueError,match="Preserve existing planned output"):plan(case)
    assert marker.read_text()=="preserve"


def test_planner_imports_only_standard_library_and_cli_has_no_execute_switch():
    tree=ast.parse(Path(planner.__file__).read_text())
    imports={node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names)
    assert imports<={"__future__","argparse","ast","hashlib","json","math","pathlib","shlex","struct","sys","zipfile"}
    destinations={action.dest for action in planner.parser()._actions}
    assert not {"execute","retry","resume","force","overwrite"}&destinations
