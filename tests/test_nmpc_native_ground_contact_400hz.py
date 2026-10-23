"""New ABI3 ground channels; all historical ABI2 physical outputs unchanged."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
mujoco = pytest.importorskip("mujoco")
import nmpc_native_ground_contact_400hz as mod
import nmpc_native_batch_400hz as legacy


def model_xml(extra="", floor=True):
    children, actuators = [], []
    for i, (body, joint) in enumerate(zip(mod.BODY_NAMES[1:], mod.JOINT_NAMES)):
        wheel = i >= 6
        position = f"0 {-.15 if i==6 else .15} -.4" if wheel else f"{.2+.1*i} 0 0"
        children.append(f'<body name="{body}" pos="{position}"><joint name="{joint}" axis="0 1 0" '
            f'limited="{"false" if wheel else "true"}" range="-2 2"/>'
            '<inertial pos="0 0 0" mass=".2" diaginertia=".002 .002 .002"/>'
            f'<geom type="sphere" size="{.1 if wheel else .01}"/></body>')
        effort = 12 if wheel else 80
        actuators.append(f'<motor name="{joint}" joint="{joint}" ctrllimited="true" ctrlrange="{-effort} {effort}"/>')
    return ('<mujoco><compiler angle="radian"/><option timestep=".0025" integrator="implicitfast" solver="Newton"/>'
        '<default><geom friction=".7 .0 .0" condim="3"/></default><worldbody>'
        + ('<geom name="actual_floor" type="plane" size="2 2 .1"/>' if floor else '') + extra +
        '<body name="base_Link" pos="0 0 .495"><freejoint/><inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>'
        '<geom type="sphere" size=".05"/>'+''.join(children)+'</body></worldbody><actuator>'+''.join(actuators)+'</actuator></mujoco>')


def model(extra="", floor=True):
    return mujoco.MjModel.from_xml_string(model_xml(extra, floor))


def contract():
    return {"actuator_model": "dc_motor", "physics_dt": .0025, "decimation": 8, **mod.ACTUATOR_SETTINGS}


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    parent=tmp_path_factory.mktemp("ground_native")
    return mod.build_native_library(parent/"v2"), legacy.build_native_library(parent/"legacy")


def test_floor_is_actual_unique_horizontal_plane_not_arbitrary_world_geom():
    wall='<geom name="world_wall" type="plane" size="2 2 .1" euler="0 1.5707963267948966 0" pos="2 0 0"/>'
    m=model(wall)
    assert m.geom(mod.resolve_floor_geom(m)).name=="actual_floor"
    with pytest.raises(ValueError,match="exactly one"):mod.resolve_floor_geom(model(floor=False))
    with pytest.raises(ValueError,match="exactly one"):
        mod.resolve_floor_geom(model('<geom name="second_floor" type="plane" size="2 2 .1" pos="0 0 -1"/>'))
    assert mod.model_weight_n(m)==pytest.approx(float(m.body_mass.sum())*9.81)


@pytest.mark.parametrize("width",[1691,1754,1756])
def test_old_or_wrong_stage_width_rejected(width):
    with pytest.raises(mod.NativeBatchError,match="1755"):
        mod._unpack(np.zeros((1,1,width)))


def test_append_layout_and_count_semantics():
    raw=np.zeros((2,3,1755));ground=raw[...,1691:].reshape(2,3,8,8)
    ground[...,:2]=[1,2];ground[...,2:]=[0,0,20,0,0,40]
    result=mod._unpack(raw)
    assert result["ground_contact_count"].shape==(2,3,8,2)
    np.testing.assert_array_equal(result["ground_normal_force_w"][...,2],np.broadcast_to([20,40],(2,3,8,2)))
    ground[0,0,0,0]=.5
    with pytest.raises(mod.NativeBatchError,match="counts"):mod._unpack(raw)


def test_ground_only_normal_readout_excludes_wall_and_selfcontact(monkeypatch):
    m=model('<geom name="wall" type="box" size=".01 2 2" pos="2 0 1"/>')
    floor=mod.resolve_floor_geom(m); wheel=[int(m.body(name).id) for name in mod.BODY_NAMES[-2:]]
    wheel_geoms=[int(np.flatnonzero(m.geom_bodyid==body)[0]) for body in wheel]
    wall=int(m.geom("wall").id)
    from types import SimpleNamespace
    contacts=[SimpleNamespace(geom1=floor,geom2=wheel_geoms[0],frame=np.eye(3)[[2,0,1]].reshape(-1)),
              SimpleNamespace(geom1=wheel_geoms[0],geom2=wall,frame=np.eye(3).reshape(-1)),
              SimpleNamespace(geom1=wheel_geoms[0],geom2=wheel_geoms[1],frame=np.eye(3).reshape(-1))]
    def contact_force(model,data,index,wrench):wrench[:]=[20.,300.,-200.,0.,0.,0.]
    monkeypatch.setattr(mujoco,"mj_contactForce",contact_force)
    actual=mod.ground_contact_readout(m,SimpleNamespace(contact=contacts))
    np.testing.assert_array_equal(actual["ground_contact_count"],[1,0])
    np.testing.assert_array_equal(actual["ground_normal_force_w"],[[0,0,20],[0,0,0]])
    def tiny(model,data,index,wrench):wrench[:]=[1e-6,0,0,0,0,0]
    monkeypatch.setattr(mujoco,"mj_contactForce",tiny)
    assert mod.ground_contact_readout(m,SimpleNamespace(contact=contacts))["ground_contact_count"][0]==1


def test_native_appends_ground_and_preserves_every_legacy_value(libraries):
    v2lib,oldlib=libraries;m=model();q=m.qpos0[None].copy();v=np.zeros((1,14));actions=np.zeros((1,2,8));history=np.zeros((1,3,9,3))
    with legacy.NativeBatchTransition(m,np.zeros(8),contract(),oldlib) as old:
        expected=old.rollout(q,v,actions,initial_history=history)
    with mod.NativeBatchTransition(m,np.zeros(8),contract(),v2lib) as new:
        actual=new.rollout(q,v,actions,initial_history=history)
        assert new.metadata["abi_version"]==3 and new.metadata["legacy_output_channels_unchanged"] is True
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
    assert np.any(actual["ground_contact_count"]>0)
    assert np.all(actual["ground_normal_force_w"][...,2]>=0)
    aids=np.array([m.actuator(name).id for name in mod.JOINT_NAMES]);d=mujoco.MjData(m)
    for h in range(2):
        for k in range(8):
            d.qpos[:]=actual["qpos"][0,h,k];d.qvel[:]=actual["qvel"][0,h,k]
            d.ctrl[aids]=actual["torques"][0,h,k];mujoco.mj_forward(m,d)
            oracle=mod.ground_contact_readout(m,d)
            np.testing.assert_array_equal(actual["ground_contact_count"][0,h,k],oracle["ground_contact_count"])
            np.testing.assert_allclose(actual["ground_normal_force_w"][0,h,k],oracle["ground_normal_force_w"],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual["interval_time_s"],np.broadcast_to(np.arange(1,9)*.0025,(1,2,8)),rtol=0,atol=1e-12)


def test_abi3_cannot_load_frozen_abi2_and_cannot_overwrite(libraries):
    v2lib,oldlib=libraries
    with pytest.raises(mod.NativeBatchError,match="provenance"):
        mod._validated_library(oldlib)
    with pytest.raises(FileExistsError):mod.build_native_library(v2lib.parent)
    metadata=json.loads(v2lib.with_suffix(".json").read_text())
    assert metadata["abi_version"]==3 and metadata["stage_width"]==1755
    assert hashlib.sha256((ROOT/"native/nmpc_batch_400hz.cpp").read_bytes()).hexdigest()=="00562a83506e36df01158112fca5c7512be14f819d11e6da0d0a94b3010c71e1"
