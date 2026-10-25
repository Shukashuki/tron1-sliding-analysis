"""Matched S/T temporal support shaping;fixed independent-camber motion."""
from __future__ import annotations
import importlib
from pathlib import Path
import sys

import train_contact_camber_reference as parent
import camber_reference_contract as reference

COHORT_ID="tron1_contact_duration_credit_pair_400hz_v1"
for _name in ("CHECKPOINT_SHA","ACTOR_SHA","WARM_STATE_SHA","RANDOM_STATE_SHA","BASE_TRANSITIONS","RUNNER_SHA","PPO_SHA","REFINEMENT","sha","read","write","state_sha","refinement_config"):
    globals()[_name]=getattr(parent,_name)


def paired_parent(source):
    source=parent.parent_source(source)
    for old,new in [("assert protocol['variants']==['K']","assert protocol['variants']==['S','T']"),
            ("for variant in ['K']:","for variant in ['S','T']:"),
            ("choices=['K']","choices=['S','T']"),
            ("K contact reward; same car task-space objectives but spin-invariant3D wheel-axis tracking replaces2D heading; explicit matched warm start.",
             "Matched S/T support-duration costs;fixed independent-camber reference and3D wheel-axis reward;explicit original K warm start.")]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    compile(source,"<paired_duration_warm_parent>","exec");return source


def materialize(source):
    assert sha(parent.__file__)=="d6a374cacbbcaf72b1098e2e2e471f6c9d9c6ad5023fc46f6ff1b660ae7b209d"
    source=parent.materialize(source)
    for old,new in [('COHORT_ID = "'+parent.COHORT_ID+'"','COHORT_ID = "'+COHORT_ID+'"'),
            ('here/"verify_contact_camber_reference.py"','here/"verify_contact_duration_pair.py"')]:
        assert source.count(old)==1;source=source.replace(old,new,1)
    return source


def sources():
    here=Path(__file__).resolve().parent
    paths=[Path(parent.__file__),here/"contact_duration_owner.py",here/"ground_contact_duration_cost.py",
        here/"verify_contact_duration_pair.py",here/"eval_contact_camber_reference.py",
        here.parent/"config/ppo_ground_contact_v2k_400hz_v1.json"]
    for name in ["ground_contact_reward_v2s","ground_contact_reward_v2t","ppo_ground_contact_v2s","ppo_ground_contact_v2t"]:
        paths.append(Path(importlib.import_module(name).__file__))
    return {**parent.sources(),**{str(p.resolve()):sha(p) for p in paths}}


def run(argv):
    here=Path(__file__).resolve().parent
    ns=dict(__name__="_paired_duration_refinement",__file__=str(Path(__file__).resolve()),
        _axis_actual_sources=parent.parent.actual_sources,_camber_sources=sources,
        _camber_validate_protocol=reference.validate_protocol,_camber_parent_source=paired_parent)
    exec(compile(materialize((here/"train_contact_taskspace_refinement.py").read_text()),"<paired_duration_refinement>","exec"),ns)
    ns["run"](argv)


if __name__=="__main__":run(sys.argv[1:])
