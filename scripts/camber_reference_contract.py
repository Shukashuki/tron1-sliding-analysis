"""Explicit new-reference admission for native camber tracking experiments.

Changes the declared motion,never the old cohort pins or source files. Loaded
native tensors are checked by name after the existing MotionLoader reorder.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

OLD_MOTION = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
MOTION = "0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9"
GENERATION = "ac825323e8104df81049bd3ad29d6daef99ffc6bb50bf9c9ed3895a2abba00a8"
VERIFICATION = "4ef2695324556d5a6da18e98c03379e3314597150a176a23a220bd031f2ce1a7"
DECLARATION = dict(id="fit_differential", motion_sha256=MOTION, original_motion_sha256=OLD_MOTION,
    generation_sha256=GENERATION, kinematic_verification_sha256=VERIFICATION,
    independent_signed_camber=True, targets_L_R=[1.1,.9], geometrical_support_is_measured_force=False,
    original_body_XY_speed_yaw_retained=True, new_height_or_slip_reward=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def validate_protocol(protocol):
    if protocol.get("kinematic_reference") != DECLARATION:
        raise ValueError("Require the exact independent-camber reference declaration")
    return dict(DECLARATION)


def bind_declared_contract(contract,descriptor):
    if contract.get("reference_file_sha256")!=OLD_MOTION or descriptor.get("motion_sha256")!=MOTION:
        raise ValueError("Unexpected source contract or reference binding")
    result=copy.deepcopy(contract)
    result["reference_file_sha256"]=MOTION
    result["kinematic_reference"]=copy.deepcopy(descriptor)
    return result


def validate_reference(path,protocol):
    import numpy as np
    path=Path(path).resolve();validate_protocol(protocol)
    if sha(path)!=MOTION:raise ValueError("Wrong camber motion bytes")
    generation=path.with_name("generation.json");verification=path.with_name("verification_v2.json")
    if sha(generation)!=GENERATION or sha(verification)!=VERIFICATION:
        raise ValueError("Wrong independent kinematic reference proof")
    g,v=read(generation),read(verification)
    row=next(r for r in g["rows"] if r["id"]=="fit_differential")
    assert row["reference_sha256"]==MOTION and g["reference_sha256"]==OLD_MOTION
    assert v["generation_sha256"]==GENERATION and v["frames_decoded"]==349
    assert not g["positive_normal_force_proven"] and not v["contact_force_or_dynamic_feasibility_verified"]
    with np.load(path,allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
    assert raw["joint_pos"].shape==raw["joint_vel"].shape==(349,8)
    assert raw["body_pos_w"].shape==(349,10,3) and raw["body_quat_w"].shape==(349,10,4)
    assert np.allclose(raw["time_s"],np.arange(349)*.02,atol=1e-12,rtol=0)
    for name in ["joint_pos","joint_vel","body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"]:
        assert np.isfinite(raw[name]).all(),name
    assert raw["wheel_geometrically_touching"].all()
    assert np.max(abs(raw["wheel_geometric_gap_m"]))<1e-7
    gamma=raw["wheel_camber_rad"];core=(raw["time_s"]>=1.44-1e-9)&(raw["time_s"]<=4.+1e-9)
    assert (abs(gamma[core])>.01).any() and np.max(abs(gamma[:,0]+gamma[:,1]))<1e-9
    return dict(**DECLARATION,reference_path=str(path),frames=349,fps=50,
        geometric_support_only=True,positive_contact_force_proven=False,
        peak_abs_camber_deg_L_R=np.rad2deg(abs(gamma)).max(0).tolist(),
        input_sha256={str(path):MOTION,str(generation):GENERATION,str(verification):VERIFICATION})


def live_audit(env,path,protocol):
    import numpy as np
    import torch
    metadata=validate_reference(path,protocol)
    command=env.command_manager.get_term("motion");robot=env.scene["robot"]
    with np.load(path,allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
    joints=raw["joint_names"].astype(str).tolist();bodies=raw["body_names"].astype(str).tolist()
    assert len(joints)==len(set(joints)) and len(bodies)==len(set(bodies))
    j=[joints.index(n) for n in robot.joint_names];b=[bodies.index(n) for n in robot.body_names]
    audit={}
    for key,attr,index in [("joint_pos","joint_pos",j),("joint_vel","joint_vel",j),
            ("body_pos_w","_body_pos_w",b),("body_quat_w","_body_quat_w",b),
            ("body_lin_vel_w","_body_lin_vel_w",b),("body_ang_vel_w","_body_ang_vel_w",b)]:
        actual=getattr(command.motion,attr).detach().cpu()
        expected=torch.as_tensor(raw[key][:,index],dtype=torch.float32)
        assert torch.equal(actual,expected),("Native motion tensor differs",key,tuple(actual.shape),tuple(expected.shape))
        audit[key]=dict(shape=list(actual.shape),bit_identical_after_float32_and_named_reorder=True)
    return dict(status="native_independent_camber_reference_tensors_verified",reference_sha256=MOTION,
        tensors=audit,body_names=list(robot.body_names),joint_names=list(robot.joint_names),
        extra_simulation_steps=0,geometric_contact_is_force=False)


def verify_result(result):
    p=result["protocol"];validate_protocol(p)
    assert result["manifest"]["inputs"]["motion"]["sha256"]==MOTION
    assert result["contract"]["reference_file_sha256"]==MOTION
    descriptor=result["manifest"]["kinematic_reference"]
    assert descriptor["motion_sha256"]==MOTION and descriptor["input_sha256"][descriptor["reference_path"]]==MOTION
    for path,digest in descriptor["input_sha256"].items():assert sha(path)==digest
    for name in ["live_startup","live_final"]:
        a=result["manifest"]["cohort"][name]["kinematic_reference"]
        assert a["status"]=="native_independent_camber_reference_tensors_verified" and a["reference_sha256"]==MOTION
        assert a["extra_simulation_steps"]==0 and not a["geometric_contact_is_force"]
        assert len(a["tensors"])==6 and all(x["bit_identical_after_float32_and_named_reorder"] for x in a["tensors"].values())
    result["independent_camber_reference_verified"]=True
    return result
