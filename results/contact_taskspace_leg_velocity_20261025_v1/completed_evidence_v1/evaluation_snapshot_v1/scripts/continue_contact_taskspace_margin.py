"""Exact-driver wait and serial M PhysX/native-MuJoCo nominal validation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import continue_contact_taskspace_refinement as parent

PARENT_SHA = "621cf4b15962fbbdb31fdd0095a866cff14bdcc8fa1936c98af61a7813795d96"


def materialize(source):
    assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() == PARENT_SHA
    source = parent.materialize(source)
    for old, new in [
        ("taskspace_K_two_native_nominal_episodes", "taskspace_margin4_M_two_native_nominal_episodes"),
        ("run_contact_taskspace_refinement.py", "run_contact_taskspace_margin.py"),
        ("['K']", "['M']"),
        ("contact_taskspace_refinement_launch_template.json", "contact_taskspace_margin_launch_template.json"),
        ("eval_contact_taskspace_refinement.py", "eval_contact_taskspace_margin.py"),
        ("contact_taskspace_refinement_400hz_v1.json", "contact_taskspace_margin_400hz_v1.json"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<taskspace_margin4_serial_native_continuation>", "exec")
    return source


def run(a):
    source = materialize((Path(__file__).parent/"continue_contact_warm_refinement.py").read_text())
    ns = dict(__name__="_taskspace_margin4_serial_native_continuation", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_margin4_serial_native_continuation>", "exec"), ns)
    ns["run"](a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("workspace", "evaluation-snapshot", "training-snapshot", "idle-audit-source", "python", "legacy-physx", "legacy-mujoco"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--expected-driver-pid", type=int, required=True)
    run(p.parse_args())
