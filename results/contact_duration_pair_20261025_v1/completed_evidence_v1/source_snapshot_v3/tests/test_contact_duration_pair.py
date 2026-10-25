"""Strict S/T protocol admission and unchanged reference/optimizer/task gates."""
from pathlib import Path
import copy
import argparse
import json
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import train_contact_duration_pair as training
import verify_contact_duration_pair as verification
import eval_contact_duration_pair as evaluation
import run_contact_duration_pair as driver
import continue_contact_duration_pair as continuation
import camber_reference_contract as reference


def warm_namespace():
    source=training.materialize((ROOT/"scripts/train_contact_taskspace_refinement.py").read_text())
    ns=dict(__name__="_paired_cpu_target_module",__file__=str(Path(training.__file__).resolve()))
    exec(compile(source,"<paired_cpu_target_module>","exec"),ns)
    inner=training.paired_parent(ns["materialize_parent"]((ROOT/"scripts/train_contact_warm_refinement.py").read_text()))
    warm=dict(__name__="_paired_cpu_warm",__file__=str(ROOT/"scripts/train_contact_warm_refinement.py"),
        _taskspace_validate=reference.validate_protocol)
    exec(compile(inner,"<paired_cpu_warm>","exec"),warm)
    return warm


def test_both_variants_are_declared_and_other_task_requirements_preserved():
    warm=warm_namespace()
    p=json.loads((ROOT/"config/contact_duration_pair_400hz_v1.json").read_text())
    original=json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    result=warm["validate_protocol"](p,original)
    assert result["variants"]==["S","T"] and result["algorithm"]["transitions"]==2457600
    assert result["nominal_acceptance_before_training"]==original["nominal_acceptance_before_training"]
    assert result["training_randomization"]==original["training_randomization"]


def test_training_source_ledger_loads_both_actual_runtime_owners():
    sources={**training.sources(),**training.parent.parent.actual_sources()}
    names={Path(p).name for p in sources}
    assert {"ppo_ground_contact_v2k.py","ppo_ground_contact_v2s.py","ppo_ground_contact_v2t.py"}<=names
    for path,digest in sources.items():
        assert training.sha(path)==digest


@pytest.mark.parametrize("change",["support_gate","DR","optimizer","force_threshold"])
def test_relaxed_or_mixed_protocol_is_rejected(change):
    warm=warm_namespace()
    p=json.loads((ROOT/"config/contact_duration_pair_400hz_v1.json").read_text())
    original=json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    if change=="support_gate":p["nominal_acceptance_before_training"]["any_wheel_support_fraction_min"]=.9
    if change=="DR":p["training_randomization"]["sliding_friction"]["range"]=[.1,.7]
    if change=="optimizer":p["refinement_optimizer"]["learning_rate"]*=2
    if change=="force_threshold":p["variant_contact_settings"]["S"]["support_force_threshold_n"]=1.
    with pytest.raises((AssertionError,ValueError)):warm["validate_protocol"](p,original)


def test_all_strict_and_native_lifecycle_entrypoints_compile():
    verification.materialize((ROOT/"scripts/verify_contact_taskspace_refinement.py").read_text())
    verification.warm_source((ROOT/"scripts/verify_contact_warm_refinement.py").read_text())
    evaluation.materialize((ROOT/"scripts/eval_contact_warm_refinement.py").read_text())
    driver.materialize((ROOT/"scripts/run_contact_warm_refinement.py").read_text())
    continuation.materialize((ROOT/"scripts/continue_contact_warm_refinement.py").read_text())


def test_full_checkpoint_load_and_conservative_optimizer_are_inherited_unchanged():
    original=json.loads((ROOT/"config/contact_camber_reference_400hz_v1.json").read_text())
    paired=json.loads((ROOT/"config/contact_duration_pair_400hz_v1.json").read_text())
    for key in ["warm_start","refinement_optimizer","algorithm","smoke","kinematic_reference","task_space_tracking"]:
        assert paired[key]==original[key]


def test_native_device_parser_admits_host_devices_and_rejects_out_of_range():
    p=argparse.ArgumentParser();p.add_argument('--physical-gpu-index',type=int,choices=(3,4,5))
    evaluation.allow_all_fully_idle_devices(p)
    for i in range(8):assert p.parse_args(['--physical-gpu-index',str(i)]).physical_gpu_index==i
    with pytest.raises(SystemExit):p.parse_args(['--physical-gpu-index','8'])
