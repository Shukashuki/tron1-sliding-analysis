"""Native R inference with strict physical-leg-velocity training admission."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

PARENT_SHA="13a5ebd7bc36ce54d86d21e2935d5c6fb4306d79102170b5997c8970a8f77d0a"


def main(argv):
    p=Path(__file__).parent/"eval_contact_warm_refinement.py"
    source=p.read_text();assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    for old,new in [
        ("import train_contact_warm_refinement as training","import train_contact_taskspace_leg_velocity as training"),
        ("import verify_contact_warm_refinement as verification","import verify_contact_taskspace_leg_velocity as verification"),
        ("kind='warm_contact_refinement_native_evaluation'","kind='taskspace_actual_leg_velocity_native_evaluation'"),
    ]:
        assert source.count(old)==1;source=source.replace(old,new,1)
    ns=dict(__name__="_R_actual_leg_velocity_native_evaluation",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<R_actual_leg_velocity_native_evaluation>","exec"),ns)
    return ns["main"](argv)


if __name__=="__main__":main(sys.argv[1:])
