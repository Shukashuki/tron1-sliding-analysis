"""New-reference admission,loaded tensor mapping,and unchanged nominal gates."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import argparse
import json
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import camber_reference_contract as reference
import train_contact_camber_reference as training
import verify_contact_camber_reference as verification
import eval_contact_camber_reference as evaluation
import ppo_backend_400hz_contract as original_common
import run_contact_camber_reference as driver
import continue_contact_camber_reference as continuation


@pytest.fixture
def declared(tmp_path):
    r=ROOT/"results/camber_kinematic_reference_20261025_v1"
    for src,dst in [("independent/fit_differential.npz","fit_differential.npz"),("independent/generation.json","generation.json"),("verification_v2.json","verification_v2.json")]:
        shutil.copyfile(r/src,tmp_path/dst)
    return tmp_path/"fit_differential.npz",json.loads((ROOT/"config/contact_camber_reference_400hz_v1.json").read_text())


def test_declared_kinematic_proof_does_not_claim_dynamic_load(declared):
    path,protocol=declared
    result=reference.validate_reference(path,protocol)
    assert result["motion_sha256"]==reference.MOTION and not result["positive_contact_force_proven"]
    assert result["peak_abs_camber_deg_L_R"][0]>6.


def test_old_motion_and_changed_proof_are_rejected(declared):
    path,protocol=declared
    with pytest.raises(ValueError,match="motion bytes"):
        reference.validate_reference(ROOT/"results/contact_wheel_axis_20261025_v1/reference.npz",protocol)
    path.with_name("generation.json").write_text('{}')
    with pytest.raises(ValueError,match="reference proof"):
        reference.validate_reference(path,protocol)


def test_reference_declaration_cannot_silently_change_slip_targets(declared):
    _,protocol=declared
    protocol["kinematic_reference"]["targets_L_R"]=[.9,.9]
    with pytest.raises(ValueError,match="declaration"):reference.validate_protocol(protocol)


def test_declared_contract_tracks_new_motion_before_training_without_changing_motors(declared):
    path,protocol=declared
    descriptor=reference.validate_reference(path,protocol)
    original=json.loads((ROOT/"config/nominal_actuator_contract.json").read_text())
    before=copy.deepcopy(original)
    derived=reference.bind_declared_contract(original,descriptor)
    assert original==before and derived["reference_file_sha256"]==reference.MOTION
    assert {k:v for k,v in derived.items() if k not in {"reference_file_sha256","kinematic_reference"}}=={k:v for k,v in original.items() if k!="reference_file_sha256"}
    bad=copy.deepcopy(original);bad["reference_file_sha256"]="0"*64
    with pytest.raises(ValueError,match="source contract"):reference.bind_declared_contract(bad,descriptor)


def test_native_ingestion_is_named_bit_exact_and_detects_corruption(declared):
    path,protocol=declared
    with np.load(path,allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
    joints=raw["joint_names"].astype(str).tolist()[::-1]
    bodies=raw["body_names"].astype(str).tolist()[::-1]
    ji=[list(raw["joint_names"]).index(n) for n in joints];bi=[list(raw["body_names"]).index(n) for n in bodies]
    motion=NS()
    for k,attr,index in [("joint_pos","joint_pos",ji),("joint_vel","joint_vel",ji),
            ("body_pos_w","_body_pos_w",bi),("body_quat_w","_body_quat_w",bi),
            ("body_lin_vel_w","_body_lin_vel_w",bi),("body_ang_vel_w","_body_ang_vel_w",bi)]:
        setattr(motion,attr,torch.as_tensor(raw[k][:,index],dtype=torch.float32).clone())
    env=NS(command_manager=NS(get_term=lambda n:NS(motion=motion)),scene={"robot":NS(joint_names=joints,body_names=bodies)})
    assert len(reference.live_audit(env,path,protocol)["tensors"])==6
    motion.joint_pos[100,0]+=.01
    with pytest.raises(AssertionError,match="tensor differs"):reference.live_audit(env,path,protocol)


def test_explicit_common_pin_binding_does_not_modify_old_contract_module():
    pins=copy.deepcopy(original_common.INPUT_PINS)
    new,proof=evaluation.reference_common(original_common)
    assert original_common.INPUT_PINS==pins and pins["motion"]==reference.OLD_MOTION
    assert new.INPUT_PINS=={**pins,"motion":reference.MOTION}
    assert not proof["original_source_file_modified"] and proof["other_input_pins_unchanged"]


def test_explicit_idle_gpu0_fallback_does_not_admit_other_devices():
    p=argparse.ArgumentParser();p.add_argument('--physical-gpu-index',type=int,choices=(3,4,5))
    evaluation.allow_idle_gpu0(p)
    assert p.parse_args(['--physical-gpu-index','0']).physical_gpu_index==0
    with pytest.raises(SystemExit):p.parse_args(['--physical-gpu-index','1'])


def test_parent_admission_retains_original_gates_and_warm_budget(declared):
    _,protocol=declared
    source=training.materialize((ROOT/"scripts/train_contact_taskspace_refinement.py").read_text())
    ns=dict(__name__="_test_camber_hooks",__file__=str(Path(training.__file__).resolve()))
    exec(compile(source,"<test_camber_hooks>","exec"),ns)
    parent=training.parent_source(ns["materialize_parent"]((ROOT/"scripts/train_contact_warm_refinement.py").read_text()))
    hooks=dict(__name__="_test_warm_admission",__file__=str(ROOT/"scripts/train_contact_warm_refinement.py"),
               _taskspace_validate=lambda p:reference.validate_protocol(p))
    exec(compile(parent,"<test_warm_admission>","exec"),hooks)
    # The generated warm module has its own globals;exercise the actual import
    # there rather than assuming the outer adapter's callbacks propagate.
    assert hooks["_camber_reference"].validate_reference(*declared)["motion_sha256"]==reference.MOTION
    k=json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    assert hooks["validate_protocol"](protocol,k)["algorithm"]["transitions"]==2457600
    bad=copy.deepcopy(protocol);bad["nominal_acceptance_before_training"]["any_wheel_support_fraction_min"]=.9
    with pytest.raises(AssertionError):hooks["validate_protocol"](bad,k)


def test_all_materialized_entrypoints_compile():
    compile(verification.materialize((ROOT/"scripts/verify_contact_taskspace_refinement.py").read_text()),"<strict>","exec")
    compile(evaluation.materialize((ROOT/"scripts/eval_contact_warm_refinement.py").read_text()),"<native>","exec")
    compile(driver.materialize((ROOT/"scripts/run_contact_taskspace_refinement.py").read_text()),"<driver>","exec")
    compile(continuation.materialize((ROOT/"scripts/continue_contact_warm_refinement.py").read_text()),"<continuation>","exec")
