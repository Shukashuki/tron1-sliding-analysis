"""Pure arrays/fake planning pipelines only; no MuJoCo/Isaac imports or steps."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import reconstruct_nmpc_isaac_failed_prefix as probe


def fake_recording():
    actions = np.full((349,8),np.float32(.2),dtype=float)
    states = np.zeros((349,111));states[:,3]=1.;states[0,-1]=1.
    velocity = np.zeros((1397,3));velocity[:,2]=np.arange(1397)*.001
    trace = {"packed_controller_states":states,"actions":actions,"time_s":np.arange(1397)*.005,
             "simulation_step_counter":np.arange(1397)+10,"base_com_velocity_w":velocity,
             "body_net_forces_w":np.zeros((1397,9,3))}
    rows = []
    for frame in range(349):
        solver = {key:False for key in probe.EXACT_FIELDS}
        solver.update(initial_candidate_source="shifted_previous_plan",warm_start_used=frame>0,
                      action_parameterization="warm_start_residual_blocks",fallback_kind=None)
        solver.update({key:0 for key in probe.COUNT_FIELDS})
        solver["failed_scenarios_by_mu"] = [0]
        solver.update(initial_model_cost=3.,selected_model_cost=2.,selected_scenario_costs=[2.],
                      best_cost_by_iteration=[3.,2.5,2.3,2.2,2.1,2.])
        rows.append({"frame":frame,"reference_frame_for_reward":frame,"action":actions[frame].tolist(),
                     "raw_cem_action":[.2]*8,"solver":solver,
                     "actual_isaac_objective":{"base_com_vz":float(velocity[4*(frame+1),2])}})
    return trace,rows


def test_recorded_previous_actions_and_actual_vz_preserve_reset_inputs():
    trace,rows = fake_recording()
    states,actions = probe.validate_recorded_inputs(trace,rows)
    np.testing.assert_array_equal(states,trace["packed_controller_states"])
    np.testing.assert_array_equal(actions,trace["actions"])
    a,vz = probe.previous_inputs(0,trace,rows)
    np.testing.assert_array_equal(a,np.zeros(8));assert vz==trace["base_com_velocity_w"][0,2]
    for frame in (1,72,137):
        a,vz = probe.previous_inputs(frame,trace,rows)
        np.testing.assert_array_equal(a,trace["actions"][frame-1])
        assert vz==rows[frame-1]["actual_isaac_objective"]["base_com_vz"]
    with pytest.raises(ValueError,match="0..137"):
        probe.previous_inputs(138,trace,rows)


@pytest.mark.parametrize("problem",["nan","clock","counter","action","raw","vz","reset","history","quaternion","phase"])
def test_invalid_existing_recording_fails_closed_without_planning(problem):
    trace,rows = fake_recording()
    if problem=="nan":trace["packed_controller_states"][2,0]=np.nan
    if problem=="clock":trace["time_s"][3]+=.001
    if problem=="counter":trace["simulation_step_counter"][3]+=1
    if problem=="action":trace["actions"][0,0]=.123456789
    if problem=="raw":rows[1]["raw_cem_action"][0]=.8
    if problem=="vz":rows[1]["actual_isaac_objective"]["base_com_vz"]+=.1
    if problem=="reset":trace["packed_controller_states"][2,-1]=1
    if problem=="history":trace["packed_controller_states"][2,29]=1
    if problem=="quaternion":trace["packed_controller_states"][2,3]=0
    if problem=="phase":rows[2]["reference_frame_for_reward"]=3
    with pytest.raises(ValueError):
        probe.validate_recorded_inputs(trace,rows)


def test_decision_guard_exact_raw_float32_and_fixed_cost_tolerance():
    _,rows = fake_recording();expected=rows[0];action=np.array(expected["raw_cem_action"]);info=copy.deepcopy(expected["solver"])
    assert probe.decision_mismatches(action,info,expected)==[]
    changed=action.copy();changed[0]=np.nextafter(changed[0],np.inf)
    assert probe.decision_mismatches(changed,info,expected)==["raw_cem_action"]
    info["selected_model_cost"]+=1e-10
    assert probe.decision_mismatches(action,info,expected)==["selected_model_cost"]
    assert probe.decision_mismatches(action,info,expected,cost_atol=1e-9)==[]
    info["best_cost_by_iteration"][2]=None
    assert "best_cost_by_iteration" in probe.decision_mismatches(action,info,expected,cost_atol=1e-9)
    info["failed_batch_prediction_calls"]=1
    assert "failed_batch_prediction_calls" in probe.decision_mismatches(action,info,expected,cost_atol=1e-9)


def test_replay_loop_stops_before_any_decision_after_first_mismatch(tmp_path):
    trace,rows=fake_recording()
    class Controller:
        def __init__(self):self.rng=np.random.default_rng(42);self.frames=[];self.last_plan=np.zeros((30,8))
        def action(self,state,frame,*,previous_action,previous_vz):
            self.frames.append(frame)
            np.testing.assert_array_equal(state,trace["packed_controller_states"][frame])
            action=np.array(rows[frame]["raw_cem_action"])
            if frame==2:action[0]=.5
            return action,copy.deepcopy(rows[frame]["solver"])
    frames=[]
    observer=SimpleNamespace(begin=frames.append,total_candidates=0)
    controller=Controller()
    matched,mismatch=probe.replay_prefix(controller,observer,trace,rows,tmp_path)
    assert matched==[0,1] and mismatch["frame"]==2
    assert controller.frames==frames==[0,1,2]
    logged=[json.loads(line) for line in (tmp_path/"reconstructed_decisions.jsonl").read_text().splitlines()]
    assert len(logged)==3 and not logged[-1]["matched_available_archived_fields"]
    assert logged[0]["full_plan_sha256"]==probe.array_sha(np.zeros((30,8)))


def make_bundle(path, *, problem=None):
    expected={}
    with tarfile.open(path,"w:gz") as archive:
        for index,name in enumerate(probe.MEMBERS):
            data=("immutable "+name).encode()
            expected[name]=probe.hashlib.sha256(data).hexdigest()
            member=tarfile.TarInfo(name)
            if index==0 and problem=="traversal":member.name="../escape.py"
            if index==0 and problem=="symlink":member.type=tarfile.SYMTYPE;member.linkname="/tmp/elsewhere"
            member.size=len(data) if member.isfile() else 0
            archive.addfile(member,io.BytesIO(data) if member.isfile() else None)
    return expected


def test_source_preparation_is_exact_mechanical_copy_and_refuses_overwrite(tmp_path):
    archive=tmp_path/"source.tar.gz";expected=make_bundle(archive);dest=tmp_path/"snapshot"
    assert probe.extract_snapshot(archive,dest,expected,archive_sha256=probe.sha(archive))==expected
    assert {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}==set(probe.MEMBERS)
    assert all(probe.sha(dest/name)==digest for name,digest in expected.items())
    with pytest.raises(FileExistsError):
        probe.extract_snapshot(archive,dest,expected,archive_sha256=probe.sha(archive))


@pytest.mark.parametrize("problem",["traversal","symlink","hash"])
def test_source_preparation_rejects_unsafe_or_wrong_bytes_before_extract(tmp_path,problem):
    archive=tmp_path/"source.tar.gz";expected=make_bundle(archive,problem=problem);dest=tmp_path/"snapshot"
    if problem=="hash":expected[probe.MEMBERS[0]]="0"*64
    with pytest.raises(ValueError):
        probe.extract_snapshot(archive,dest,expected,archive_sha256=probe.sha(archive))
    assert not dest.exists()


class FakeBase:
    def __init__(self,phase=None):
        self.phase=phase;self.native_calls=0;self.batch_calls=0
        process=SimpleNamespace(is_alive=lambda:True,pid=123,exitcode=None)
        self.workers={.7:SimpleNamespace(rollout=self.native,_process=process,_closed=False)}
    def native(self,*args,**kwargs):
        self.native_calls+=1
        if self.phase=="native_transition":raise RuntimeError("Original native failure")
        return "unchanged result"
    def rollout_batch(self,state,plans,index,**kwargs):
        self.batch_calls+=1
        if self.phase=="cost_adapter_input":raise FloatingPointError("Original input failure")
        assert self.workers[.7].rollout(state,plans)=="unchanged result"
        if self.phase=="cost_adapter_objective":raise FloatingPointError("Original objective failure")
        return np.arange(len(plans),dtype=float)
    def initial_candidate(self,*args,**kwargs):return np.zeros((30,8))


@pytest.mark.parametrize("phase",["native_transition","cost_adapter_input","cost_adapter_objective"])
def test_external_observer_captures_first_error_phase_and_exact_inputs_without_retry(tmp_path,phase):
    base=FakeBase(phase);observer=probe.ObservedPredictor(base,tmp_path);observer.begin(137)
    state=np.arange(111,dtype=float);plans=np.zeros((4,30,8));previous=np.full(8,np.float32(.3))
    with pytest.raises((RuntimeError,FloatingPointError),match="Original"):
        observer.rollout_batch(state,plans,137,mu=.7,previous_action=previous,previous_vz=-.2)
    first=copy.deepcopy(observer.first_exception)
    assert first["phase"]==phase and first["frame"]==137 and first["frame_call_index"]==1
    assert first["worker"]["alive"] is True and first["worker"]["exitcode"] is None
    assert first["plan_sha256"]==probe.array_sha(plans)
    with np.load(tmp_path/"first_exception_inputs.npz",allow_pickle=False) as data:
        np.testing.assert_array_equal(data["state"],state)
        np.testing.assert_array_equal(data["plans"],plans)
        np.testing.assert_array_equal(data["previous_action"],previous)
        assert float(data["previous_vz"])==-.2
    with pytest.raises((RuntimeError,FloatingPointError)):
        observer.rollout_batch(state,plans,137,mu=.7,previous_action=previous,previous_vz=-.2)
    assert observer.first_exception==first and base.batch_calls==2
    assert base.native_calls==(0 if phase=="cost_adapter_input" else 2)
    assert len((tmp_path/"prediction_calls.jsonl").read_text().splitlines())==2


def test_observer_delegates_same_arguments_once_and_enforces_hard_budget_before_call(tmp_path):
    base=FakeBase();observer=probe.ObservedPredictor(base,tmp_path);observer.begin(0)
    state=np.zeros(111);plans=np.zeros((4,30,8));previous=np.zeros(8)
    actual=observer.rollout_batch(state,plans,0,mu=.7,previous_action=previous,previous_vz=0.)
    np.testing.assert_array_equal(actual,np.arange(4))
    assert base.batch_calls==base.native_calls==1 and observer.first_exception is None
    observer.total_candidates=probe.MAX_CANDIDATE_EVALUATIONS
    with pytest.raises(probe.ReconstructionStop,match="budget"):
        observer.rollout_batch(state,plans,0,mu=.7,previous_action=previous,previous_vz=0.)
    assert base.batch_calls==base.native_calls==1
    observer.begin(138)
    with pytest.raises(probe.ReconstructionStop,match="scope"):
        observer.rollout_batch(state,plans,138,mu=.7,previous_action=previous,previous_vz=0.)
    assert base.batch_calls==1


def test_module_and_prepare_cli_have_no_simulator_import_dependency():
    assert "import mujoco" not in Path(probe.__file__).read_text()
    assert probe.parser().parse_args(["--source-root","source","--model","model","--motion","motion",
        "--contract","contract","--output-dir","NEW","--prepare-only"]).prepare_only


@pytest.mark.parametrize("phase",["missing_dependency","native_constructor"])
def test_preflight_failure_persists_failed_initialization_and_zero_planning(monkeypatch,tmp_path,phase):
    trace,rows=fake_recording()
    status={"status":"prepared_no_planning","preparation_status":"prepared_no_planning",
            "planning_started":False,"planning_steps":0,"planning_mujoco_physics_executed":False,
            "source_sha256":{},"helper_sha256":{}}
    def fake_prepare(args):
        probe.dump(tmp_path/"preparation.json",status)
        return {},rows,trace,status
    monkeypatch.setattr(probe,"prepare",fake_prepare)
    constructed=[]
    def construct(*args):
        constructed.append(True)
        raise RuntimeError("Synthetic worker-construction failure")
    if phase=="missing_dependency":
        def initialize(*args):
            raise ModuleNotFoundError("No module named scipy")
        expected=ModuleNotFoundError
    else:
        library=tmp_path/"fake.so";library.write_bytes(b"unchanged fake native")
        library.with_suffix(".json").write_text("{}")
        def initialize(*args):
            return {"nmpc_batch_objective":SimpleNamespace(NativeBatchTrackingPredictor=construct)},object(),library
        expected=RuntimeError
    monkeypatch.setattr(probe,"initialize_predictor",initialize)
    with pytest.raises(expected):
        probe.reconstruct(SimpleNamespace(output_dir=tmp_path,prepare_only=False))
    saved=json.loads((tmp_path/"report.json").read_text())
    assert saved["status"]=="failed_initialization"
    assert saved["preparation_status"]=="prepared_no_planning"
    assert saved["planning_started"] is False and saved["planning_steps"]==0
    assert saved["planning_mujoco_physics_executed"] is False
    assert saved["matched_frames"]==[] and saved["initialization_completed"] is False
    assert constructed==([] if phase=="missing_dependency" else [True])
    assert not (tmp_path/"prediction_calls.jsonl").exists()
    assert not (tmp_path/"reconstructed_decisions.jsonl").exists()
    assert json.loads((tmp_path/"preparation.json").read_text())["status"]=="prepared_no_planning"
