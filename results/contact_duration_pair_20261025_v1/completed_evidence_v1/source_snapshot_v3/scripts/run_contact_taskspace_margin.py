"""One fully-idle-GPU smoke/proof, then matched200-iteration M refinement."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "2a0271fce17e1b66f122549cb88a32ffcf89ecf79ef03dafdaa3343ea5b22008"


def run(a):
    p = Path(__file__).parent/"run_contact_taskspace_refinement.py"
    source = p.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("K_taskspace_warm_refinement", "M_taskspace_margin4_warm_refinement"),
        ("contact_taskspace_refinement_launch_template.json", "contact_taskspace_margin_launch_template.json"),
        ("train_contact_taskspace_refinement.py", "train_contact_taskspace_margin.py"),
        ("contact_taskspace_refinement_400hz_v1.json", "contact_taskspace_margin_400hz_v1.json"),
        ("verify_contact_taskspace_refinement.py", "verify_contact_taskspace_margin.py"),
        ('label = mode+"_K"', 'label = mode+"_M"'),
        ('"--reward-variant": "K"', '"--reward-variant": "M"'),
        ("for pid in [807979,", "for pid in [853798,853800,854367,854372,858246,855081,858323,859955,807979,"),
    ]:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    ns = dict(__name__="_taskspace_margin4_serial_driver", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_margin4_serial_driver>", "exec"), ns)
    ns["run"](a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("workspace", "snapshot", "idle-audit-source", "python"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    run(p.parse_args())
