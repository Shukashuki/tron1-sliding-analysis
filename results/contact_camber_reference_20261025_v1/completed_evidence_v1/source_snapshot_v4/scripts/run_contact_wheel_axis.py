"""One fully-idle native smoke/proof,then matched-budget3D wheel-axis refinement."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "2a0271fce17e1b66f122549cb88a32ffcf89ecf79ef03dafdaa3343ea5b22008"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("K_taskspace_warm_refinement", "K_3D_wheel_axis_warm_refinement"),
        ("contact_taskspace_refinement_launch_template.json", "contact_wheel_axis_launch_template.json"),
        ("train_contact_taskspace_refinement.py", "train_contact_wheel_axis.py"),
        ("contact_taskspace_refinement_400hz_v1.json", "contact_wheel_axis_400hz_v1.json"),
        ("verify_contact_taskspace_refinement.py", "verify_contact_wheel_axis.py"),
        ('assert proof["status"] == "completed_verified" and proof["task_space_tracking_verified"]',
         'assert proof["status"] == "completed_verified" and proof["task_space_tracking_verified"] and proof["three_dimensional_axial_orientation_verified"]'),
        ("            for pid in [807979,",
         "            prior=Path('/data/zengxiong/workspaces/tron1-contact-taskspace-scratch-20261025-v1')\n"
         "            prior_driver=json.loads((prior/'driver_status.json').read_text())\n"
         "            prior_native=json.loads((prior/'nominal_v1/status.json').read_text())\n"
         "            assert prior_driver['status']=='both_formal_verified_nominal_validation_pending'\n"
         "            assert prior_native['status']=='four_native_nominal_episodes_completed_require_analysis_and_render'\n"
         "            prior_pids=[json.loads((prior/n).read_text())['pid'] for n in ['driver_v1_launch.json','nominal_continuation_v1_launch.json']]\n"
         "            prior_pids += [r['pid'] for r in prior_driver['children']]+[r['pid'] for r in prior_native['cases']]\n"
         "            for pid in prior_pids:\n"
         "                process=helper.process_snapshot(pid)\n"
         "                assert process is None or process['state']=='Z',('Previous fresh-cohort process still live',process)\n"
         "            for pid in [807979,"),
    ]:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    compile(source, "<3D_axis_serial_training_driver>", "exec")
    return source


def run(args):
    source = materialize((Path(__file__).parent/"run_contact_taskspace_refinement.py").read_text())
    ns = dict(__name__="_3D_axis_serial_training_driver", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<3D_axis_serial_training_driver>", "exec"), ns)
    ns["run"](args)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","snapshot","idle-audit-source","python"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--smoke-only",action="store_true");run(p.parse_args())
