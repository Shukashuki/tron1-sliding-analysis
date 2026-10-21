"""Persistent native PPO integration parity; no policy or timing claims."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from classical_tracking import Predictor
from mujoco_ppo_native import (BODY_NAMES, JOINT_NAMES, NativePPOBatch,
                               build_ppo_native_library)
from nmpc_tracking_objective import MujocoObjectiveState


def synthetic_model():
    legs=[]
    for side,y in (("L",.14),("R",-.14)):
        legs.append(f'<body name="abad_{side}_Link" pos="0 {y} 0">'
                    f'<joint name="abad_{side}_Joint" axis="1 0 0"/>'
                    '<inertial pos=".01 .01 -.02" mass=".5" diaginertia=".005 .005 .005"/>'
                    f'<body name="hip_{side}_Link" pos="0 0 -.03">'
                    f'<joint name="hip_{side}_Joint" axis="0 1 0"/>'
                    '<inertial pos=".03 0 -.04" mass=".5" diaginertia=".005 .005 .005"/>'
                    f'<body name="knee_{side}_Link" pos="0 0 -.15">'
                    f'<joint name="knee_{side}_Joint" axis="0 1 0"/>'
                    '<inertial pos=".02 0 -.05" mass=".4" diaginertia=".004 .004 .004"/>'
                    f'<body name="wheel_{side}_Link" pos="0 0 -.15">'
                    f'<joint name="wheel_{side}_Joint" axis="0 1 0" limited="false"/>'
                    '<inertial pos=".01 0 0" mass=".2" diaginertia=".002 .002 .002"/>'
                    '<geom type="sphere" size=".08"/></body></body></body></body>')
    order=[f"{name}_{side}_Joint" for side in ("L","R") for name in ("abad","hip","knee","wheel")]
    motors=''.join(f'<motor name="{name}" joint="{name}" ctrllimited="true" '
                   f'ctrlrange="{-12 if name.startswith("wheel") else -80} '
                   f'{12 if name.startswith("wheel") else 80}"/>' for name in order)
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option timestep=".005" integrator="implicitfast" '
        'solver="Newton" iterations="100" tolerance="1e-10"/>'
        '<default><joint range="-2 2" damping="0" armature="0"/>'
        '<geom friction=".7 0 0" condim="3" solref=".01 1"/></default>'
        '<worldbody><geom type="plane" size="2 2 .1"/>'
        '<body name="base_Link" pos="0 0 .48"><freejoint/>'
        '<inertial pos=".05 0 -.02" mass="2" diaginertia=".05 .05 .05"/>'
        '<geom type="box" size=".1 .1 .04"/>'+''.join(legs)+
        '</body></worldbody><actuator>'+motors+'</actuator></mujoco>')


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    if mujoco.__version__ != "3.8.1":
        pytest.skip("Exact MuJoCo 3.8.1 required")
    return build_ppo_native_library(tmp_path_factory.mktemp("ppo_native"))


def setup_batch(library,models=None,batch=2):
    models=models or [synthetic_model() for _ in range(batch)]
    datas=[mujoco.MjData(m) for m in models]
    q0s=np.zeros((len(models),8))
    native=NativePPOBatch(models,datas,q0s,library,num_threads=2)
    qpos=np.stack([m.qpos0 for m in models]); qpos[:,2]-=.015
    qvel=np.zeros((len(models),14)); qvel[:,:3]=[1.,.3,-.1]
    qvel[:,3:6]=[.1,-.2,.3]
    native.reset(np.arange(len(models)),qpos,qvel)
    return native


def python_step(model,data,q0,action):
    qids=np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
    vids=np.array([int(model.joint(n).dofadr[0]) for n in JOINT_NAMES])
    aids=np.array([int(model.actuator(n).id) for n in JOINT_NAMES])
    helper=SimpleNamespace(qids=qids,vids=vids,q0=q0,
                           effort=np.r_[[80.]*6,[12.]*2],no_load=np.r_[[15.]*6,[100.]*2])
    extractor=MujocoObjectiveState(model)
    torques,contacts=[],[]
    action=np.clip(np.asarray(action,dtype=np.float32),-1.,1.).astype(np.float64)
    for _ in range(4):
        tau=Predictor.torque(helper,action,data.qpos,data.qvel)
        data.ctrl[aids]=tau
        mujoco.mj_step(model,data); mujoco.mj_forward(model,data)
        contacts.append(extractor.contact_forces(data))
        torques.append(tau)
    measured=extractor.extract(data)
    return {**{key:measured[key] for key in ("body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w")},
            "qpos":data.qpos.copy(),"qvel":data.qvel.copy(),"times":data.time,
            "contact_substeps":np.array(contacts),"applied_joint_torque_nm":np.array(torques)}


def assert_rows(actual,expected):
    for key in expected[0]:
        np.testing.assert_allclose(actual[key],np.array([item[key] for item in expected]),
                                   rtol=1e-10,atol=1e-9,err_msg=key)
    assert not actual["failed"].any()
    np.testing.assert_array_equal(actual["substeps_completed"],4)


def test_continuous_clock_torque_contact_body_parity_two_calls(native_library):
    native=setup_batch(native_library)
    native.q0s[:]=[.01,-.01,.02,-.02,.03,-.03,0.,0.]
    vids=[int(native.models[0].joint(n).dofadr[0]) for n in JOINT_NAMES]
    native.datas[0].qvel[vids]=[20.,-20.,8.,-8.,12.,-12.,150.,-150.]
    mujoco.mj_forward(native.models[0],native.datas[0])
    models=[copy.copy(m) for m in native.models]
    datas=[mujoco.MjData(m) for m in models]
    for original,model,data in zip(native.datas,models,datas):
        data.qpos[:],data.qvel[:]=original.qpos,original.qvel
        mujoco.mj_forward(model,data)
    actions=np.array([[.01234567891,-.2,.3,-.4,.5,-.6,1.5,-1.5],[-.05,.05,0.,0.,0.,0.,.2,.3]])
    for call in range(2):
        expected=[python_step(m,d,q0,a) for m,d,q0,a in zip(models,datas,native.q0s,actions)]
        actual=native.step(actions)
        assert_rows(actual,expected)
        np.testing.assert_allclose(actual["times"],(call+1)*.02,rtol=0,atol=1e-15)
    before=[d.qacc_warmstart.copy() for d in native.datas]
    measured=native.measure()
    np.testing.assert_allclose(measured["times"],.04,rtol=0,atol=1e-15)
    for d,old in zip(native.datas,before):
        np.testing.assert_array_equal(d.qacc_warmstart,old)
    assert native.metadata["persistent_integration"]


def test_selected_reset_preserves_other_env_clock_and_integration(native_library):
    native=setup_batch(native_library)
    actions=np.zeros((2,8)); native.step(actions)
    old=native.measure(); warmstart=native.datas[1].qacc_warmstart.copy()
    q=native.models[0].qpos0[None].copy(); v=np.zeros((1,14))
    now=native.reset([0],q,v)
    np.testing.assert_allclose(now["times"],[0.,.02])
    np.testing.assert_array_equal(now["qpos"][1],old["qpos"][1])
    np.testing.assert_array_equal(native.datas[1].qacc_warmstart,warmstart)
    np.testing.assert_allclose(native.step(actions)["times"],[.02,.04])


def test_different_models_com_friction_and_mass_are_isolated(native_library):
    a,b=synthetic_model(),synthetic_model()
    b.geom_friction[:,0]=.03
    b.body_ipos[int(b.body("base_Link").id)]+=[.02,-.01,.01]
    b.body_mass[int(b.body("base_Link").id)]*=1.2
    native=setup_batch(native_library,[a,b])
    clones=[copy.copy(m) for m in native.models]
    datas=[mujoco.MjData(m) for m in clones]
    for original,m,d in zip(native.datas,clones,datas):
        d.qpos[:],d.qvel[:]=original.qpos,original.qvel; mujoco.mj_forward(m,d)
    actions=np.zeros((2,8))
    expected=[python_step(m,d,q0,a) for m,d,q0,a in zip(clones,datas,native.q0s,actions)]
    assert_rows(native.step(actions),expected)
    assert a.geom_friction[0,0] == .7
    assert b.geom_friction[0,0] == .03
    assert not np.array_equal(native.measure()["body_lin_vel_w"][0],native.measure()["body_lin_vel_w"][1])


def test_external_forces_report_latched_bad_env_without_reset(native_library):
    native=setup_batch(native_library)
    native.datas[0].qfrc_applied[0]=1.
    failed=native.step(np.zeros((2,8)))
    np.testing.assert_array_equal(failed["error_codes"],[3,0])
    np.testing.assert_array_equal(failed["substeps_completed"],[0,4])
    np.testing.assert_allclose(failed["times"],[0.,.02])
    native.datas[0].qfrc_applied[0]=0.
    again=native.step(np.zeros((2,8)))
    np.testing.assert_array_equal(again["error_codes"],[3,0])
    np.testing.assert_allclose(again["times"],[0.,.04])
    native.reset([0],native.models[0].qpos0[None],np.zeros((1,14)))
    assert not native.step(np.zeros((2,8)))["failed"].any()


def test_warnings_are_reported_and_only_explicit_reset_clears(native_library):
    native=setup_batch(native_library)
    native.datas[1].warning[0].number=1
    native.datas[1].warning[0].lastinfo=37
    result=native.step(np.zeros((2,8)))
    assert result["error_codes"][1] == 1
    assert result["warning_counts"][1,0] == 1
    assert result["warning_lastinfo"][1,0] == 37
    assert result["times"][1] == 0.
    native.reset([1],native.models[1].qpos0[None],np.zeros((1,14)))
    assert not native.measure()["failed"].any()


def test_no_implicit_build_and_provenance_refuses_tamper(native_library,tmp_path):
    with pytest.raises(FileExistsError):
        build_ppo_native_library(native_library.parent)
    missing=tmp_path/"missing.so"
    with pytest.raises(FileNotFoundError):
        setup_batch(missing)
    assert not missing.exists()
    copied=tmp_path/native_library.name
    copied.write_bytes(native_library.read_bytes())
    metadata=json.loads(native_library.with_suffix(".json").read_text())
    metadata["source_sha256"]="0"*64
    copied.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError,match="provenance"):
        setup_batch(copied)


def test_reject_raw_pointers_shared_models_nonfinite_and_bad_reset(native_library):
    model=synthetic_model(); data=mujoco.MjData(model)
    with pytest.raises(TypeError,match="raw pointers"):
        NativePPOBatch([model._address],[data],np.zeros(8),native_library)
    with pytest.raises(ValueError,match="distinct"):
        NativePPOBatch([model,model],[data,mujoco.MjData(model)],np.zeros(8),native_library)
    with pytest.raises(ValueError,match="paired model"):
        NativePPOBatch([synthetic_model()],[data],np.zeros(8),native_library)
    native=setup_batch(native_library)
    with pytest.raises(ValueError,match="finite"):
        native.step(np.full((2,8),np.nan))
    with pytest.raises(ValueError,match="unique"):
        native.reset([0,0],np.zeros((2,15)),np.zeros((2,14)))
    with pytest.raises(ValueError,match="quaternions"):
        native.reset([0],np.zeros((1,15)),np.zeros((1,14)))
    np.testing.assert_allclose(native.measure()["times"],0.)


def test_licensed_tron1_model_two_control_intervals(native_library):
    path=ROOT/"outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
    if not path.is_file():
        pytest.skip("Licensed local TRON1 model unavailable")
    # Removing visual-only mesh dependencies leaves collision/inertia untouched.
    tree=ET.fromstring(path.read_text())
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == "mesh" or (child.tag == "geom" and "mesh" in child.attrib):
                parent.remove(child)
    tree.find("option").set("timestep",".005")
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding="unicode"))
    model.geom_friction[:,0]=.7
    native=setup_batch(native_library,[model])
    oracle_model=copy.copy(model); oracle=mujoco.MjData(oracle_model)
    oracle.qpos[:],oracle.qvel[:]=native.datas[0].qpos,native.datas[0].qvel
    mujoco.mj_forward(oracle_model,oracle)
    for action in (np.array([[0.,.01,.02,-.02,.03,-.03,.7,-.6]]),np.zeros((1,8))):
        expected=python_step(oracle_model,oracle,native.q0s[0],action[0])
        assert_rows(native.step(action),[expected])
