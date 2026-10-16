"""MPPI runtime contracts using AST, synthetic data and mocked physics only."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
pytest.importorskip("mujoco")
import run_mppi_tracking as runtime
from controller_references import get_reference
from mppi_shooting import MPPIConfig, MPPIController
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES
from run_nmpc_tracking import PackedState


def payload():
    return {"schema_version":1,"task":"car_drift_height_smooth","policy_dt_s":.02,
            "physics_dt_s":.005,"prediction_mu_scenarios":[.7],"mppi":asdict(MPPIConfig())}


def load(monkeypatch,value):
    monkeypatch.setattr(runtime.Path,"read_text",lambda self:json.dumps(value))
    return runtime.effective_settings(SimpleNamespace(settings="explicit.json",plant_mu=.01))


def test_configuration_is_explicit_and_prediction_never_uses_plant_mu(monkeypatch):
    _,config,scenarios=load(monkeypatch,payload())
    assert config==MPPIConfig() and scenarios==(.7,)


@pytest.mark.parametrize("field",list(asdict(MPPIConfig())))
def test_every_controller_field_must_be_declared(monkeypatch,field):
    value=payload();del value["mppi"][field]
    with pytest.raises(ValueError,match="every MPPIConfig"):
        load(monkeypatch,value)


@pytest.mark.parametrize("scenarios",[[],[.3,.7],[.7,.7],[-.1],[True],[".7"],[float("nan")]])
def test_model_bank_or_invalid_prediction_coefficient_rejected(monkeypatch,scenarios):
    value=payload();value["prediction_mu_scenarios"]=scenarios
    with pytest.raises(ValueError,match="one explicit"):
        load(monkeypatch,value)


@pytest.mark.parametrize("field,value",[("schema_version",True),("schema_version",2),
    ("task","cem"),("policy_dt_s",.01),("physics_dt_s",.01),("mppi",[])])
def test_wrong_schema_or_clocks_rejected(monkeypatch,field,value):
    document=payload();document[field]=value
    with pytest.raises(ValueError):load(monkeypatch,document)


def test_extra_fields_and_plant_coefficient_are_not_controller_inputs(monkeypatch):
    value=payload();value["plant_mu"]=.7
    with pytest.raises(ValueError,match="schema fields"):load(monkeypatch,value)


def test_parser_has_no_hidden_cem_tuning_or_demonstration_inputs():
    p=runtime.parser();destinations={action.dest for action in p._actions}
    assert not destinations & {"elite_count","candidates","iterations","model_nominal_plan",
                               "adaptive_contact_config","design_dir","prediction_mu_scenarios"}
    args=p.parse_args(["--model","m.xml","--motion","m.npz","--contract","c.json",
                       "--plant-mu",".03","--output-dir","unused"])
    assert args.settings==ROOT/"config/mppi_height_smooth.json"
    assert args.prediction_backend=="python" and args.native_library is None


def tree(filename):return ast.parse((ROOT/"scripts"/filename).read_text())
def named(nodes,name):return next(node for node in nodes if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
def dump(node):return ast.dump(node,include_attributes=False)


def test_copied_physical_recorder_and_substep_loop_match_archived_cem_ast():
    original=named(tree("run_nmpc_tracking.py").body,"_run")
    new=named(tree("run_mppi_tracking.py").body,"_run")
    assert dump(named(original.body,"record"))==dump(named(new.body,"record"))
    original_loop=next(node for node in ast.walk(original) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=="k")
    new_loop=next(node for node in ast.walk(new) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=="k")
    # Actual actuator -> step/forward -> force guard -> history -> record -> clock.
    original_physics=next(node for node in original_loop.body if isinstance(node,ast.For))
    new_physics=next(node for node in new_loop.body if isinstance(node,ast.For))
    assert dump(original_physics)==dump(new_physics)
    # Shared packing, held pre-action reference, float32 command, reward memory.
    assert [dump(node) for node in original_loop.body[:2]]==[dump(node) for node in new_loop.body[:2]]
    targets={"action","held_action","held_frame","objective","previous","previous_vz"}
    def assignments(loop):
        return {tuple(target.id for target in node.targets if isinstance(target,ast.Name)):dump(node)
                for node in loop.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id in targets for target in node.targets)}
    assert assignments(original_loop)==assignments(new_loop)


def test_native_wiring_is_explicit_mppi_not_false_cem_claim():
    adapter=runtime.NativeMPPIPredictor.__new__(runtime.NativeMPPIPredictor)
    adapter.metadata={"production_cem_wired":False,"objective":{"production_cem_wired":False}}
    controller=MPPIController.__new__(MPPIController);controller.predictor=adapter
    adapter.record_mppi_wiring(controller)
    assert adapter.metadata["production_mppi_wired"] is True
    assert adapter.metadata["objective"]["production_mppi_wired"] is True
    assert adapter.metadata["production_cem_wired"] is False
    assert adapter.metadata["objective"]["production_cem_wired"] is False
    controller.predictor=object()
    with pytest.raises(ValueError,match="actual controller"):adapter.record_mppi_wiring(controller)
    with pytest.raises(ValueError,match="actual controller"):adapter.record_mppi_wiring(object())


@pytest.fixture
def fake_run(tmp_path,monkeypatch):
    """Exercise real recording/reporting, replacing every dynamics operation."""
    settings=tmp_path/"settings.json";settings.write_text(json.dumps(payload()))
    task=tmp_path/"task.json";task.write_text("{}")
    contract=tmp_path/"contract.json";contract.write_text("{}")
    model_path=tmp_path/"model.xml";model_path.write_text("mock model, never loaded")
    motion=tmp_path/"motion.npz";motion.write_bytes(b"mock motion, never loaded")
    spec=get_reference();calls=[];controllers=[]
    class Model:
        geom_friction=np.ones((3,3));opt=SimpleNamespace(timestep=.005);njnt=9
        def joint(self,index):return SimpleNamespace(name=JOINT_NAMES[index-1])
    class Data:
        def __init__(self,model):
            self.time=0.;self.qpos=np.r_[np.zeros(3),1.,np.zeros(11)];self.qvel=np.zeros(14)
            self.ctrl=np.zeros(8);self.qfrc_applied=np.zeros(14);self.xfrc_applied=np.zeros((9,6))
    class Physical:
        aids=np.arange(8);vids=np.arange(6,14)
        def __init__(self,model,data,reference):self.data=data
        def initialize(self):pass
        def ground_contact_slip(self,**kwargs):
            return {"wheel_ground_contact_count":np.array([1,1]),
                    "wheel_ground_contact_force_w":np.zeros((2,3)),
                    "wheel_ground_contact_lateral_speed_rms":np.zeros(2)}
        def nonwheel_ground_contacts(self):return 0,0.
    class Extractor:
        def __init__(self,model):pass
        def extract(self,data,**kwargs):
            positions=np.zeros((9,3));positions[0]=data.qpos[:3]
            return {"body_pos_w":positions,"body_quat_w":np.tile([1.,0,0,0],(9,1)),
                "body_lin_vel_w":np.zeros((9,3)),"body_ang_vel_w":np.zeros((9,3)),
                "joint_pos":data.qpos[7:].copy(),"body_net_forces_w":np.zeros((9,3))}
        def update_contact_history(self,data,history):return history,np.zeros((9,3))
    class Objective:
        def contract(self):return {"absolute_height_target":False}
        def hold_reference(self,reference,pre):calls.append(("hold",reference));return reference
        def evaluate(self,pre,held,action,previous,vz,**kwargs):
            calls.append(("objective",held,action.copy(),previous.copy(),kwargs["reset_sample"]))
            return {"stage_cost":1.,"base_com_vz":0.,"surrogate_flags":[]}
    reference=SimpleNamespace(frames=349,fps=50.,archive={"body_pos_w":np.zeros((349,9,3)),
        "body_lin_vel_w":np.ones((349,9,3))})
    api=SimpleNamespace(load_model=lambda *a,**k:(Model(),None),configure_model=lambda *a:None,MujocoState=Physical)
    provider=SimpleNamespace(api=api,contract={"decimation":4},reference=reference,dt=.02,
        targets=np.array([np.r_[np.zeros(3),1.,np.zeros(25)]]),torque=lambda action,q,v:2*action)
    class Predictor:
        def __init__(self,*args):self.layout=PackedState();self.objective=Objective()
        def reference_frame(self,index):return index
    class Controller:
        fail_frame=None
        def __init__(self,predictor,*args,**kwargs):
            self.predictor=predictor;self.mu_scenarios=kwargs["mu_scenarios"];controllers.append(self)
        def parameterization_contract(self):return {"mode":"latent_gaussian_blocks"}
        def algorithm_contract(self):return {"kind":"mppi"}
        def action(self,state,k,**kwargs):
            calls.append(("action",k,kwargs["previous_action"].copy(),state[-1]))
            if k==self.fail_frame:raise RuntimeError("injected solver error")
            return np.full(8,.123456789+k*.01),{"optimization_failed":False}
    def step(model,data):
        calls.append(("step",data.ctrl.copy()));data.time+=.005;data.qpos[0]+=.005
    monkeypatch.setattr(runtime,"selected_reference",lambda args:spec)
    monkeypatch.setattr(runtime,"build_provider",lambda *args:provider)
    monkeypatch.setattr(runtime,"MujocoTrackingPredictor",Predictor)
    monkeypatch.setattr(runtime,"MujocoObjectiveState",Extractor)
    monkeypatch.setattr(runtime,"MPPIController",Controller)
    monkeypatch.setattr(runtime,"source_files",lambda *args:[Path(runtime.__file__)])
    monkeypatch.setattr(runtime.mujoco,"MjData",Data)
    monkeypatch.setattr(runtime.mujoco,"mj_step",step)
    monkeypatch.setattr(runtime.mujoco,"mj_forward",lambda *args:None)
    args=runtime.parser().parse_args(["--model",str(model_path),"--motion",str(motion),"--contract",str(contract),
        "--settings",str(settings),"--task-profile",str(task),"--plant-mu",".03","--max-policy-steps","2",
        "--output-dir",str(tmp_path/"run")])
    return args,calls,Controller


def test_mock_episode_preserves_float32_actions_four_substeps_torque_and_clock(fake_run):
    args,calls,_=fake_run;report=runtime.run(args);out=args.output_dir
    assert report["status"]=="smoke_completed" and report["completed_policy_steps"]==2
    assert report["physics_samples"]==9 and report["duration_s"]==pytest.approx(.04)
    assert report["prediction_mu_scenarios"]==[.7] and report["plant_mu"]==.03
    assert report["plant_material_readback"]["compiled_sliding_friction_range"]==[.03,.03]
    assert report["core_coverage_complete"] is False and report["core_xy_rmse_m"] is None
    assert report["kind"]=="full_body_height_smooth_mppi" and "effective_cem_config" not in report
    assert report["optimization_equals_pure_task_cost_cem"] is False
    assert report["model_generated_nominal_plan"]=={"enabled":False}
    with np.load(out/"rollout.npz",allow_pickle=False) as raw:
        np.testing.assert_allclose(raw["time_s"],np.arange(9)*.005,atol=1e-12,rtol=0)
        assert raw["actions"].shape==(2,8)
        np.testing.assert_array_equal(raw["actions"],raw["actions"].astype(np.float32).astype(float))
        np.testing.assert_array_equal(raw["held_action"][1:],np.repeat(raw["actions"],4,axis=0))
        np.testing.assert_array_equal(raw["applied_joint_torque_nm"][1:],2*raw["held_action"][1:])
        np.testing.assert_array_equal(raw["reference_frame_for_action"][1:],np.repeat([0,1],4))
        np.testing.assert_array_equal(raw["action_was_applied"],np.r_[False,np.ones(8,dtype=bool)])
        np.testing.assert_array_equal(raw["action_joint_names"],JOINT_NAMES)
    rows=json.loads((out/"solver_log.json").read_text())
    assert all("raw_mppi_action" in row and "raw_cem_action" not in row for row in rows)
    actions=[call for call in calls if call[0]=="action"]
    np.testing.assert_array_equal(actions[1][2],rows[0]["action"])
    assert [call[-1] for call in actions]==[1.,0.]
    assert [call[0] for call in calls]==["hold","action",*("step",)*4,"objective","hold","action",*("step",)*4,"objective"]
    assert report["rollout_sha256"]==hashlib.sha256((out/"rollout.npz").read_bytes()).hexdigest()
    assert report["solver_log_sha256"]==hashlib.sha256((out/"solver_log.json").read_bytes()).hexdigest()


def test_mock_failure_preserves_failed_report_without_fabricating_completed_trace(fake_run):
    args,_,controller=fake_run;controller.fail_frame=1
    with pytest.raises(RuntimeError,match="injected solver"):runtime.run(args)
    report=json.loads((args.output_dir/"report.json").read_text())
    assert report["status"]=="failed" and report["completed_policy_steps"]==1
    assert report["error_type"]=="RuntimeError"
    assert not (args.output_dir/"rollout.npz").exists()


def test_mock_native_resources_close_on_wiring_failure(fake_run,monkeypatch):
    args,_,_=fake_run;args.prediction_backend="native";args.native_library=Path("prebuilt.so")
    events=[]
    class Adapter:
        metadata={}
        def __init__(self,*args):pass
        def __enter__(self):events.append("open");return self
        def __exit__(self,*args):events.append("close")
        def record_mppi_wiring(self,controller):raise RuntimeError("wiring failure")
    monkeypatch.setattr(runtime,"NativeMPPIPredictor",Adapter)
    with pytest.raises(RuntimeError,match="wiring failure"):runtime.run(args)
    assert events==["open","close"] and not args.output_dir.exists()


def test_existing_output_refused_before_reading_inputs(tmp_path):
    with pytest.raises(FileExistsError,match="Preserving"):
        runtime.run(SimpleNamespace(output_dir=tmp_path))


def test_native_library_requires_explicit_native_backend_before_loading_inputs(fake_run):
    args,_,_=fake_run;args.native_library=Path("prebuilt.so")
    with pytest.raises(ValueError,match="explicit opt-in"):runtime.run(args)
