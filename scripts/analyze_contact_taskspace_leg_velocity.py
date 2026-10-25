"""Matched task-space M/R audit: actual joint-velocity regularization only."""
from __future__ import annotations

import argparse
from pathlib import Path

import analyze_contact_taskspace_margin as parent
from train_contact_taskspace_leg_velocity import velocity_audit

PARENT_SHA="ae6ed07b6020b10900ace9f248dcfb0251087dcaa1f120808eb53d63bf91d1bc"
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/"results/contact_taskspace_margin4_20261025_v1/completed_evidence_v1"
ORDER=("control_physx","leg_velocity_physx","control_mujoco","leg_velocity_mujoco")
metrics,read,sha,npz=parent.metrics,parent.read,parent.sha,parent.npz


def namespace():
    assert sha(parent.__file__)==PARENT_SHA
    source=Path(parent.__file__).read_text()
    pairs=[
        ("import analyze_contact_taskspace_refinement as control","import analyze_contact_taskspace_margin as control"),
        ("import collect_contact_taskspace_margin as collection","import collect_contact_taskspace_leg_velocity as collection"),
        ('ORDER = ("control_physx", "margin4_physx", "control_mujoco", "margin4_mujoco")',
         'ORDER = ("control_physx", "leg_velocity_physx", "control_mujoco", "leg_velocity_mujoco")'),
        ("results/contact_taskspace_refinement_20261025_v2/completed_evidence_v1","results/contact_taskspace_margin4_20261025_v1/completed_evidence_v1"),
        ("config/contact_taskspace_margin_400hz_v1.json","config/contact_taskspace_leg_velocity_400hz_v1.json"),
        ('old = read(CONTROL/"formal_K/manifest.json")','old = read(CONTROL/"formal_M/manifest.json")'),
        ('new = read(folder/"formal_M/manifest.json")','new = read(folder/"formal_R/manifest.json")'),
        ('read(CONTROL/"formal_K/runner_config.json") == read(folder/"formal_M/runner_config.json")',
         'read(CONTROL/"formal_M/runner_config.json") == read(folder/"formal_R/runner_config.json")'),
        ('assert {k for k in before if before[k] != after[k]} == {"candidate", "support_gap_weight"}',
         'assert {k for k in before if before[k] != after[k]} == {"candidate"}\n'
         '    assert set(after)-set(before) == {"leg_velocity_penalty_weight","leg_velocity_scale_rad_s","leg_velocity_sampling","leg_velocity_joint_names"}\n'
         '    assert after["leg_velocity_penalty_weight"] == .1 and after["leg_velocity_scale_rad_s"] == 1.'),
        ('assert before["support_gap_weight"] == 1. and after["support_gap_weight"] == 4.',
         'assert before["support_gap_weight"] == after["support_gap_weight"] == 4.'),
        ('        a = new["cohort"][key]["task_space_tracking"]',
         '        _R_velocity_audit(new["cohort"][key]["contact"])\n        a = new["cohort"][key]["task_space_tracking"]'),
        ('proofs["margin4"]','proofs["leg_velocity"]'),
        ('proofs = dict(control=old_proofs["taskspace"], margin4=driver["proofs"]["formal_M"])',
         'proofs = dict(control=old_proofs["margin4"], leg_velocity=driver["proofs"]["formal_R"])'),
        ('r["id"] == "taskspace_"+engine','r["id"] == "margin4_"+engine'),
        ('old_traces["taskspace_"+engine]','old_traces["margin4_"+engine]'),
        ('title="Task-space K coefficient1 | "+engine','title="Task-space M control | "+engine'),
        ('meta["kind"] == "taskspace_margin4_native_evaluation"','meta["kind"] == "taskspace_actual_leg_velocity_native_evaluation"'),
        ('row = dict(id="margin4_"+engine, variant="margin4", engine=engine,',
         'row = dict(id="leg_velocity_"+engine, variant="leg_velocity", engine=engine,'),
        ('title="Task-space M coefficient4 | "+engine','title="M + actual leg velocity | "+engine'),
        ('kind="matched_taskspace_K_M_margin_refinement_dual_engine"','kind="matched_taskspace_M_R_leg_velocity_dual_engine"'),
        ('only_contact_setting_differences=["candidate", "support_gap_weight"]',
         'only_contact_setting_differences=["candidate","leg_velocity_penalty_weight","leg_velocity_scale_rad_s","leg_velocity_sampling","leg_velocity_joint_names"]'),
        ('Same task-space targets | contact-margin coefficient1 vs4 | equal 2.46M refinement | mu=.7',
         'Same M contact | add actual leg-velocity cost | equal 2.46M refinement | mu=.7'),
    ]
    for old,new in pairs:
        assert old in source,old;source=source.replace(old,new)
    ns=dict(__name__="_matched_M_R_native_analysis",__file__=str(Path(__file__).resolve()),_R_velocity_audit=velocity_audit)
    exec(compile(source,"<matched_M_R_native_analysis>","exec"),ns)
    return ns


def load_cases(folder,reference):
    return namespace()["load_cases"](folder,reference)


def run(a):
    return namespace()["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("collection","reference","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
