"""Wait for exact R training, then run two serial native nominal episodes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import continue_contact_taskspace_margin as parent

PARENT_SHA="a1b78da1fd290038d2f6546e7fa100de4d10f1b6acafad1bdceeac0ff463b158"


def materialize(source):
    assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()==PARENT_SHA
    source=parent.materialize(source)
    for old,new in [
        ("taskspace_margin4_M_two_native_nominal_episodes","taskspace_actual_leg_velocity_R_two_native_nominal_episodes"),
        ("run_contact_taskspace_margin.py","run_contact_taskspace_leg_velocity.py"),
        ("['M']","['R']"),
        ("contact_taskspace_margin_launch_template.json","contact_taskspace_leg_velocity_launch_template.json"),
        ("eval_contact_taskspace_margin.py","eval_contact_taskspace_leg_velocity.py"),
        ("contact_taskspace_margin_400hz_v1.json","contact_taskspace_leg_velocity_400hz_v1.json"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<R_serial_native_continuation>","exec");return source


def run(a):
    source=materialize((Path(__file__).parent/"continue_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_R_serial_native_continuation",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<R_serial_native_continuation>","exec"),ns);ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","evaluation-snapshot","training-snapshot","idle-audit-source","python","legacy-physx","legacy-mujoco"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--expected-driver-pid",type=int,required=True);run(p.parse_args())
