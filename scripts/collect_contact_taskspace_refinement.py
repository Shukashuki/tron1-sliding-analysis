"""Collect one verified task-space policy and its two nominal native traces."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "5a3b65b4abf7ef35b220f3ea84937a6d46a94dd789231b7157f840bdab3ecb7d"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("COHORT='tron1_contact_warm_refinement_400hz_v1'", "COHORT='tron1_contact_taskspace_refinement_400hz_v1'"),
        ("CASES=('K_physx','K_mujoco','M_physx','M_mujoco')", "CASES=('K_physx','K_mujoco')"),
        ("both_formal_arms_completed_verified_nominal_pending", "formal_completed_verified_nominal_pending"),
        ("four_native_nominal_episodes_completed_require_analysis_and_render", "two_native_nominal_episodes_completed_require_analysis_and_render"),
        ("['K','M']", "['K']"),
        ("completed_paired_warm_contact_refinement_four_native_episodes", "completed_taskspace_warm_contact_refinement_two_native_episodes"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<taskspace_completed_collection>", "exec")
    return source


def collect(root, helper_path):
    source = materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_taskspace_completed_collection", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_completed_collection>", "exec"), namespace)
    return namespace["collect"](root, helper_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--idle-audit-source", type=Path, required=True)
    a = p.parse_args()
    collect(a.workspace, a.idle_audit_source)
