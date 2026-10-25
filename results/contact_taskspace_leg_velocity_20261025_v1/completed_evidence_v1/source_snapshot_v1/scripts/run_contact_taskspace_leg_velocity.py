"""One fully-idle native smoke/proof, then matched-budget R refinement."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA="2a0271fce17e1b66f122549cb88a32ffcf89ecf79ef03dafdaa3343ea5b22008"


def run(a):
    p=Path(__file__).parent/"run_contact_taskspace_refinement.py"
    source=p.read_text();assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    for old,new in [
        ("K_taskspace_warm_refinement","R_actual_leg_velocity_warm_refinement"),
        ("contact_taskspace_refinement_launch_template.json","contact_taskspace_leg_velocity_launch_template.json"),
        ("train_contact_taskspace_refinement.py","train_contact_taskspace_leg_velocity.py"),
        ("contact_taskspace_refinement_400hz_v1.json","contact_taskspace_leg_velocity_400hz_v1.json"),
        ("verify_contact_taskspace_refinement.py","verify_contact_taskspace_leg_velocity.py"),
        ('label = mode+"_K"','label = mode+"_R"'),
        ('"--reward-variant": "K"','"--reward-variant": "R"'),
        ("for pid in [807979,","for pid in [881989,881991,882720,882792,886719,884819,887200,887817,807979,"),
        ('assert proof["status"] == "completed_verified" and proof["task_space_tracking_verified"]',
         'assert proof["status"] == "completed_verified" and proof["task_space_tracking_verified"] and proof["actual_leg_velocity_audit_verified"]'),
    ]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    ns=dict(__name__="_R_actual_leg_velocity_serial_driver",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<R_actual_leg_velocity_serial_driver>","exec"),ns);ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","snapshot","idle-audit-source","python"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--smoke-only",action="store_true");run(p.parse_args())
