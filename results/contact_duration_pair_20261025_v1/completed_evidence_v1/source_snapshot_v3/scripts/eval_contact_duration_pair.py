"""Native S/T nominal inference;no training reward owner or actor changes."""
from __future__ import annotations
from pathlib import Path
import sys

import eval_contact_camber_reference as parent
import camber_reference_contract as reference


def allow_all_fully_idle_devices(parser):
    parent.allow_idle_gpu0(parser)
    action=next(a for a in parser._actions if a.dest=="physical_gpu_index")
    assert tuple(action.choices)==(0,3,4,5)
    action.choices=tuple(range(8))
    return parser


def materialize(source):
    source=parent.materialize(source)
    for old,new in [("import train_contact_camber_reference as training","import train_contact_duration_pair as training"),
            ("import verify_contact_camber_reference as verification","import verify_contact_duration_pair as verification"),
            ("kind='camber_reference_native_evaluation'","kind='duration_credit_pair_native_evaluation'")]:
        assert source.count(old)==1;source=source.replace(old,new,1)
    return source


def main(argv=None):
    assert reference.sha(parent.__file__)=="a248d95f97be870490fd9cfb6c65cf3ab55945bc4321e1b2042ddb1d37cb54f3"
    ns=dict(__name__="_paired_duration_native_eval",__file__=str(Path(__file__).resolve()))
    exec(compile(Path(parent.__file__).read_text(),"<paired_duration_native_eval>","exec"),ns)
    ns["materialize"]=materialize
    ns["allow_idle_gpu0"]=allow_all_fully_idle_devices
    return ns["main"](sys.argv[1:] if argv is None else argv)


if __name__=="__main__":main()
