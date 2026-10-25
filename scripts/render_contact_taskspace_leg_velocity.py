"""Actual M/R poses, one fixed relative camera, shared world-XY scale."""
from __future__ import annotations

import argparse
from pathlib import Path

import render_contact_taskspace_margin as parent
from analyze_contact_taskspace_leg_velocity import sha

PARENT_SHA="2114320a4a88bdb10a6889af5e052a0b0e1f6f84c445346697c9735fa4022dfb"


def materialize(source):
    assert sha(parent.__file__)==PARENT_SHA
    source=parent.materialize(source)
    for old,new in [
        ("import analyze_contact_taskspace_margin as analysis","import analyze_contact_taskspace_leg_velocity as analysis"),
        ("matched_taskspace_K_M_margin_refinement_dual_engine","matched_taskspace_M_R_leg_velocity_dual_engine"),
        ("Task-space K / M margin4 | same K checkpoint +2.46M each","Task-space M / R actual leg velocity | same K checkpoint +2.46M each"),
        ("four_actual_taskspace_K_M_margin4_native_pose_replay","four_actual_taskspace_M_R_leg_velocity_native_pose_replay"),
        ("actor_sha256_margin4=training['margin4']","actor_sha256_leg_velocity=training['leg_velocity']"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<M_R_fixed_camera_replay>","exec");return source


def run(a):
    source=materialize((Path(__file__).parent/"render_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_M_R_fixed_camera_replay",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<M_R_fixed_camera_replay>","exec"),ns);return ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("collection","analysis","output-dir","reference","model","legacy-evaluator"):
        p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
