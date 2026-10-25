"""Collect verified R training/velocity audits and native traces, omit weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import collect_contact_taskspace_margin as parent

PARENT_SHA="4b1b0ae03e5c93596be0b253a0811cbc44dbb358d0135c679a08f9c8069ca49c"


def materialize(source):
    assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()==PARENT_SHA
    source=parent.materialize(source)
    for old,new in [
        ("COHORT='tron1_contact_taskspace_margin4_refinement_400hz_v1'","COHORT='tron1_contact_taskspace_leg_velocity_400hz_v1'"),
        ("CASES=('M_physx','M_mujoco')","CASES=('R_physx','R_mujoco')"),
        ("['M']","['R']"),
        ("completed_taskspace_M_margin4_two_native_episodes","completed_taskspace_R_leg_velocity_two_native_episodes"),
        ("assert proof['cohort_id']==COHORT and proof['variant']==variant and proof['run_kind']=='formal'",
         "assert proof['cohort_id']==COHORT and proof['variant']==variant and proof['run_kind']=='formal' and proof['actual_leg_velocity_audit_verified']"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<R_completed_collection>","exec");return source


def collect(root,helper_path):
    source=materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_R_completed_collection",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<R_completed_collection>","exec"),ns);return ns["collect"](root,helper_path)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True);p.add_argument("--idle-audit-source",type=Path,required=True)
    a=p.parse_args();collect(a.workspace,a.idle_audit_source)
