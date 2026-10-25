"""One freshly idle GPU at a time:camber-reference smoke,CPU proof,refinement."""
from __future__ import annotations
import argparse
from pathlib import Path

import run_contact_wheel_axis as parent


def materialize(source):
    source=parent.materialize(source)
    for old,new in [
        ('            helper = importlib.util.module_from_spec(spec)\n            spec.loader.exec_module(helper)',
         '            helper = importlib.util.module_from_spec(spec)\n            spec.loader.exec_module(helper)\n'
         '            assert helper.GPU_ORDER==(3,5,4)\n            helper.GPU_ORDER=(3,5,4,0)'),
        ("K_3D_wheel_axis_warm_refinement","K_independent_camber_reference_refinement"),
        ("contact_wheel_axis_launch_template.json","contact_camber_reference_launch_template.json"),
        ("train_contact_wheel_axis.py","train_contact_camber_reference.py"),
        ("contact_wheel_axis_400hz_v1.json","contact_camber_reference_400hz_v1.json"),
        ("verify_contact_wheel_axis.py","verify_contact_camber_reference.py"),
        ('and proof["three_dimensional_axial_orientation_verified"]',
         'and proof["three_dimensional_axial_orientation_verified"] and proof["independent_camber_reference_verified"]'),
        ('                argv = template[:]',
         '                argv = [str(x).replace("__SNAPSHOT__",str(snapshot)).replace("__WORKSPACE__",str(root)) for x in template]'),
        ('            manifest = json.loads((root/(snapshot.name+".manifest.json")).read_text())',
         '            prior_axis=Path("/data/zengxiong/workspaces/tron1-contact-wheel-axis-20261025-v1")\n'
         '            prior_axis_pids=json.loads((prior_axis/"completion_process_audit.json").read_text())["processes"] if (prior_axis/"completion_process_audit.json").exists() else [973303,975853,973312,973700,973713,979242,980071,981075]\n'
         '            for pid in prior_axis_pids:\n'
         '                previous=helper.process_snapshot(int(pid))\n'
         '                assert previous is None or previous["state"]=="Z",("Previous axis job live",previous)\n'
         '            manifest = json.loads((root/(snapshot.name+".manifest.json")).read_text())'),
    ]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    compile(source,"<serial_camber_reference_refinement>","exec");return source


def run(args):
    source=materialize((Path(__file__).parent/"run_contact_taskspace_refinement.py").read_text())
    ns=dict(__name__="_serial_camber_reference_refinement",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<serial_camber_reference_refinement>","exec"),ns);ns["run"](args)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","snapshot","idle-audit-source","python"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--smoke-only",action="store_true");run(p.parse_args())
