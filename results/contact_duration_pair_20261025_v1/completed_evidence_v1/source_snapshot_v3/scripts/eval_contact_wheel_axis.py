"""Original native held-action evaluation with strict3D wheel-axis admission."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

PARENT_SHA = "13a5ebd7bc36ce54d86d21e2935d5c6fb4306d79102170b5997c8970a8f77d0a"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old,new in [
        ("import train_contact_warm_refinement as training","import train_contact_wheel_axis as training"),
        ("import verify_contact_warm_refinement as verification","import verify_contact_wheel_axis as verification"),
        ("kind='warm_contact_refinement_native_evaluation'","kind='wheel_axis_refinement_native_evaluation'"),
    ]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    compile(source,"<3D_axis_native_evaluation>","exec");return source


def main(argv=None):
    source=materialize((Path(__file__).parent/"eval_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_3D_axis_native_evaluation",__file__=str(Path(__file__).resolve()))
    previous=list(sys.path),sys.argv
    try:
        exec(compile(source,"<3D_axis_native_evaluation>","exec"),ns);return ns["main"](argv)
    finally:sys.path[:],sys.argv=previous


if __name__=="__main__":main(sys.argv[1:])
