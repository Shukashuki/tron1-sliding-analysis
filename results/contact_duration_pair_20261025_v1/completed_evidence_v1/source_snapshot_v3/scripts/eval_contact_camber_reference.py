"""Native nominal evaluation with explicit camber motion admission and provenance."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import camber_reference_contract as reference

PARENT_SHA="13a5ebd7bc36ce54d86d21e2935d5c6fb4306d79102170b5997c8970a8f77d0a"


def allow_idle_gpu0(parser):
    action=next(a for a in parser._actions if a.dest=="physical_gpu_index")
    assert tuple(action.choices)==(3,4,5)
    action.choices=(0,3,4,5)
    return parser


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    for old,new in [
        ("import train_contact_warm_refinement as training","import train_contact_camber_reference as training"),
        ("import verify_contact_warm_refinement as verification","import verify_contact_camber_reference as verification"),
        ("kind='warm_contact_refinement_native_evaluation'","kind='camber_reference_native_evaluation'"),
    ]:
        assert source.count(old)==1;source=source.replace(old,new,1)
    return source


def reference_common(common):
    source=Path(common.__file__).read_text()
    digest=hashlib.sha256(source.encode()).hexdigest()
    assert digest=="b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e"
    assert source.count(reference.OLD_MOTION)==1
    source=source.replace(reference.OLD_MOTION,reference.MOTION,1)
    module=ModuleType("_camber_explicit_native_contract")
    module.__file__=common.__file__
    exec(compile(source,"<camber_explicit_native_contract>","exec"),module.__dict__)
    assert module.INPUT_PINS=={**common.INPUT_PINS,"motion":reference.MOTION}
    return module,dict(original_common_sha256=digest,materialized_common_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_motion_sha256=reference.OLD_MOTION,declared_motion_sha256=reference.MOTION,
        original_source_file_modified=False,other_input_pins_unchanged=True)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument("--motion-file",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True)
    args,_=p.parse_known_args(argv)
    descriptor=reference.validate_reference(args.motion_file,reference.read(args.protocol))
    source=materialize((Path(__file__).parent/"eval_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_camber_reference_native_evaluation",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<camber_reference_native_evaluation>","exec"),ns)
    original_load=ns["load_parent"]
    def load_parent(k_snapshot,base_snapshot):
        parent=original_load(k_snapshot,base_snapshot)
        common,binding=reference_common(parent.common)
        parent.prepare_plan.__globals__["common"]=common
        parent.common=common
        original_parser=parent.prepare_plan.__globals__["parser"]
        parent.prepare_plan.__globals__["parser"]=lambda:allow_idle_gpu0(original_parser())
        previous=parent.provenance
        def provenance(cohort):
            result=previous(cohort)
            result.update(kinematic_reference=descriptor,native_common_reference_binding=binding)
            return result
        parent.provenance=provenance
        parent.prepare_plan.__globals__["provenance"]=provenance
        return parent
    ns["load_parent"]=load_parent
    return ns["main"](argv)


if __name__=="__main__":main()
