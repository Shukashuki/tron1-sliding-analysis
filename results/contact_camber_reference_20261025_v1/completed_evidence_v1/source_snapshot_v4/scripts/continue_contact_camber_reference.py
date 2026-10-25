"""Wait on the exact camber-reference training handle,then two native episodes."""
from __future__ import annotations
import argparse
from pathlib import Path

import continue_contact_wheel_axis as parent


def materialize(source):
    source=parent.materialize(source)
    for old,new in [
        ("            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)",
         "            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)\n"
         "            assert helper.GPU_ORDER==(3,5,4)\n            helper.GPU_ORDER=(3,5,4,0)"),
        ("wheel_axis_K_two_native_nominal_episodes","camber_reference_K_two_native_nominal_episodes"),
        ("run_contact_wheel_axis.py","run_contact_camber_reference.py"),
        ("contact_wheel_axis_launch_template.json","contact_camber_reference_launch_template.json"),
        ("eval_contact_wheel_axis.py","eval_contact_camber_reference.py"),
        ("contact_wheel_axis_400hz_v1.json","contact_camber_reference_400hz_v1.json"),
        ("            option=lambda f:template[template.index(f)+1]",
         "            option=lambda f:template[template.index(f)+1].replace('__SNAPSHOT__',str(a.training_snapshot.resolve())).replace('__WORKSPACE__',str(root))"),
    ]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    compile(source,"<serial_camber_native_continuation>","exec");return source


def run(args):
    source=materialize((Path(__file__).parent/"continue_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_serial_camber_native_continuation",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<serial_camber_native_continuation>","exec"),ns);ns["run"](args)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","evaluation-snapshot","training-snapshot","idle-audit-source","python","legacy-physx","legacy-mujoco"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--expected-driver-pid",type=int,required=True);run(p.parse_args())
