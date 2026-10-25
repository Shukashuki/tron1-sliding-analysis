"""Collect M task-space refinement and native records; omit weights/assets."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import collect_contact_taskspace_refinement as parent

PARENT_SHA = "04a79c7d3d6f60f320ab5257c5e7907eab93290fc1826ff075eb09bb903cdd65"


def materialize(source):
    assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() == PARENT_SHA
    source = parent.materialize(source)
    for old, new in [
        ("COHORT='tron1_contact_taskspace_refinement_400hz_v1'", "COHORT='tron1_contact_taskspace_margin4_refinement_400hz_v1'"),
        ("CASES=('K_physx','K_mujoco')", "CASES=('M_physx','M_mujoco')"),
        ("['K']", "['M']"),
        ("completed_taskspace_warm_contact_refinement_two_native_episodes", "completed_taskspace_M_margin4_two_native_episodes"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<taskspace_margin4_completed_collection>", "exec")
    return source


def collect(root, helper_path):
    source = materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text())
    ns = dict(__name__="_taskspace_margin4_completed_collection", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_margin4_completed_collection>", "exec"), ns)
    return ns["collect"](root, helper_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--idle-audit-source", type=Path, required=True)
    a = p.parse_args()
    collect(a.workspace, a.idle_audit_source)
