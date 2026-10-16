"""CPU kinematic/configuration checks; not an actual Isaac runtime claim."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from run_nmpc_tracking_isaac import (
    SourceModelProvider,IsaacToMujocoState,configure_fixed_evaluation,
    SubstepRecorder,scene_update_observer,audit_actuators,
    add_prediction_backend_arguments,prediction_backend_options,build_prediction_backend,
)

ROOT=Path(__file__).parents[1]


def test_prediction_backend_is_python_by_default_without_native_import_or_compile():
    args=add_prediction_backend_arguments(argparse.ArgumentParser()).parse_args([])
    assert prediction_backend_options(args)==("python",None)
    original=object()
    assert build_prediction_backend(original,"python",None) is original
    # Older test/caller namespaces preserve the unchanged Python path.
    assert prediction_backend_options(NS())==("python",None)


def test_native_backend_requires_explicit_prebuilt_library_and_manifest(tmp_path):
    parser=add_prediction_backend_arguments(argparse.ArgumentParser())
    with pytest.raises(ValueError,match="explicit.*native-library"):
        prediction_backend_options(parser.parse_args(["--prediction-backend","native"]))
    library=tmp_path/"libtest.so"
    with pytest.raises(FileNotFoundError,match="existing library"):
        prediction_backend_options(NS(prediction_backend="native",native_library=library))
    assert not library.exists()  # No implicit compiler/build output.
    library.write_bytes(b"unit-test path only; never loaded")
    with pytest.raises(FileNotFoundError,match="manifest"):
        prediction_backend_options(NS(prediction_backend="native",native_library=library))
    library.with_suffix(".json").write_text("{}")
    assert prediction_backend_options(NS(prediction_backend="native",native_library=library))==("native",library.resolve())
    with pytest.raises(ValueError,match="only valid"):
        prediction_backend_options(NS(prediction_backend="python",native_library=library))


def test_native_backend_receives_actual_runtime_objective_and_retains_close_ownership(monkeypatch,tmp_path):
    calls=[]
    class FakeAdapter:
        def __init__(self,prediction,library):
            calls.append((prediction,library,prediction.objective))
            self.closed=False
        def close(self):
            self.closed=True
    monkeypatch.setitem(sys.modules,"nmpc_batch_objective",NS(NativeBatchTrackingPredictor=FakeAdapter))
    actual_limits=np.tile([-1.,1.],(8,1))
    prediction=NS(objective=NS(soft_joint_limits=actual_limits))
    library=tmp_path/"prebuilt.so"
    adapter=build_prediction_backend(prediction,"native",library)
    assert calls==[(prediction,library,prediction.objective)]
    assert calls[0][2].soft_joint_limits is actual_limits
    adapter.close()
    assert adapter.closed is True


def test_fixed_evaluation_removes_all_termination_types_without_mutating_misc_metadata():
    material=NS(static_friction=1.,dynamic_friction=1.,restitution=.5)
    term=lambda: NS(func=lambda e: True)
    cfg=NS(terminations=NS(time_out=term(),motion_end=term(),anchor_ori=term(),
                           none=None,metadata={"description":"not a term"},dict_term={"func":lambda e:True}),
           events=NS(physics_material=NS(func=lambda e:None,mode="startup",params={}),
                     legacy_joint_friction=term(),push=term(),default_joint_offset=term()),
           scene=NS(terrain=NS(physics_material=material)),sim=NS(physics_material=material))
    result=configure_fixed_evaluation(cfg,.18)
    for name in ("time_out","motion_end","anchor_ori","dict_term"):
        assert getattr(cfg.terminations,name) is None
    assert cfg.terminations.metadata=={"description":"not a term"}
    assert cfg.events.push is None and cfg.events.default_joint_offset is None
    assert cfg.events.legacy_joint_friction is not None
    assert cfg.events.physics_material.params["static_friction_range"]==(.18,.18)
    assert cfg.events.physics_material.params["restitution_range"]==(0.,0.)
    assert material.static_friction==material.dynamic_friction==.18
    assert material.restitution==0.
    assert result["all_termination_terms_removed"] is True


def test_substep_observer_updates_scene_before_reading_and_ignores_reset_callbacks():
    recorder=SubstepRecorder(10,.005,{"x":np.array([0.])})
    state={"counter":10,"refreshed":False,"reads":0}
    def original():
        state["refreshed"]=True
        return "original_result"
    def snapshot():
        assert state["refreshed"] is True
        state["reads"]+=1
        return {"x":np.array([state["counter"]])}
    update=scene_update_observer(original,lambda:state["counter"],recorder,snapshot)
    assert update()=="original_result"
    assert state["reads"]==0
    recorder.begin(10,np.ones(8)*.2,3)
    for counter in range(11,15):
        state.update(counter=counter,refreshed=False)
        update();update()  # duplicate same-counter forward/update is ignored
    recorder.finish(14,4)
    arrays=recorder.arrays()
    assert state["reads"]==4
    np.testing.assert_allclose(arrays["time_s"],[0.,.005,.01,.015,.02])
    np.testing.assert_array_equal(arrays["simulation_step_counter"],np.arange(10,15))
    np.testing.assert_array_equal(arrays["reference_frame_for_action"],[0,3,3,3,3])
    np.testing.assert_array_equal(arrays["action_was_applied"],[False,True,True,True,True])
    state["counter"]=14;update()
    assert state["reads"]==4


def test_recorder_refuses_counter_skips_or_missing_substeps():
    r=SubstepRecorder(0,.005,{"x":np.array([0.])})
    r.begin(0,np.zeros(8),0)
    with pytest.raises(RuntimeError,match="skipped"):
        r.observe(2,lambda:{"x":np.array([1.])})
    with pytest.raises(RuntimeError,match="missed"):
        r.finish(0,4)


def test_actuator_audit_checks_realized_envelope_and_cached_braking_speed():
    def motor(kp,kd,effort,speed):
        return NS(stiffness=np.array([kp]),damping=np.array([kd]),
                  effort_limit=np.array([effort]),velocity_limit=np.array([speed]),
                  _saturation_effort=np.array([effort]),_vel_at_effort_lim=np.array([2*speed]))
    robot=NS(actuators={"legs":motor(500.,10.,80.,15.),"wheels":motor(0.,0.,12.,100.)})
    assert audit_actuators(robot)["wheels"]["velocity_limit_rad_s"]==[100.,100.]
    robot.actuators["wheels"]._vel_at_effort_lim[:]=100.
    with pytest.raises(AssertionError,match="velocity_at_effort_limit"):
        audit_actuators(robot)


@pytest.fixture
def provider():
    pytest.importorskip("mujoco")
    model=ROOT/"outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
    motion=Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
    contract=ROOT/"outputs/car_failure_sideslip_20261015/raw/mu0p180/isaac/policy_contract.json"
    source=ROOT/"upstream/tron1-learning-from-human"
    if not all(path.is_file() for path in (model,motion,contract,source/"scripts/eval_tracking_mujoco.py")):
        pytest.skip("Licensed model/local locked motion/source absent")
    settings=json.loads((ROOT/"config/nmpc_height_smooth.json").read_text())
    return SourceModelProvider(source,model,motion,json.loads(contract.read_text()),settings,(.7,))


def test_measured_com_velocity_to_free_joint_coordinates_matches_forward_kinematics(provider):
    import mujoco
    from nmpc_tracking_objective import quaternion_multiply
    p=provider;m=p.model;d=mujoco.MjData(m)
    d.qpos[:]=p.targets[80,:m.nq]
    qx=np.array([np.cos(.13),np.sin(.13),0.,0.])
    qy=np.array([np.cos(-.09),0.,np.sin(-.09),0.])
    qz=np.array([np.cos(.35),0.,0.,np.sin(.35)])
    d.qpos[:3]=[1.1,-.4,.8];d.qpos[3:7]=quaternion_multiply(qz,quaternion_multiply(qy,qx))
    d.qvel[:]=np.random.default_rng(7).normal(0,.5,m.nv)
    mujoco.mj_forward(m,d)
    measured=p.api.MujocoState(m,d,p.reference)
    source=measured.measure();origin=np.array([2.,-3.,.1])
    source["root_pos"]+=origin
    history=np.arange(81,dtype=float).reshape(3,9,3)
    bridge=IsaacToMujocoState(p)
    packed,error=bridge.convert(source,history,environment_origin=origin,first_reward_after_reset=True)
    q,v,h,reset=bridge.layout.unpack(packed)
    np.testing.assert_allclose(q,d.qpos,atol=1e-12,rtol=0)
    np.testing.assert_allclose(v,d.qvel,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(h,history)
    assert reset is True and error<1e-12
    # The root-origin velocity is different when the off-axis COM rotates.
    jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,d.xpos[measured.base],measured.base)
    assert np.linalg.norm(source["root_com_lin_vel_w"]-jp@d.qvel)>.01
    assert not hasattr(p,"Q") and not hasattr(p,"R")


def test_converter_rejects_merged_com_model_and_bad_quaternion(provider):
    p=provider;bridge=IsaacToMujocoState(p)
    p.measure.initialize();measurement=p.measure.measure()
    measurement["root_quat_wxyz"]=np.array([2.,0.,0.,0.])
    with pytest.raises(ValueError,match="quaternion"):
        bridge.convert(measurement,np.zeros((3,9,3)))
    p.model.body_ipos[bridge.base,0]+=.01
    with pytest.raises(ValueError,match="COM"):
        IsaacToMujocoState(p)
