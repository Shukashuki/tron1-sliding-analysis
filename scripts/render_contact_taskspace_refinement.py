"""Four-panel actual-pose replay for matched K control/task-space refinement."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "42e57a87b89fb8eb4d665babec2af20a4de627651692f57aca1a1b810ac976c2"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("import analyze_contact_warm_refinement as analysis", "import analyze_contact_taskspace_refinement as analysis"),
        ("paired_warm_K_M_contact_refinement_dual_engine", "matched_warm_K_taskspace_refinement_dual_engine"),
        ("Paired warm K/M | same K checkpoint +2.46M transitions each", "Warm K control / task-space | same K checkpoint +2.46M each"),
        ("Warm K/M native replay", "Control/task-space native replay"),
        ("four_actual_paired_warm_contact_refinement_native_pose_replay", "four_actual_warm_K_taskspace_native_pose_replay"),
        ("actor_sha256_K=training['formal_K']", "actor_sha256_control=training['control']"),
        ("actor_sha256_M=training['formal_M']", "actor_sha256_taskspace=training['taskspace']"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<taskspace_fixed_camera_native_replay>", "exec")
    return source


def run(a):
    source = materialize((Path(__file__).parent/"render_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_taskspace_fixed_camera_native_replay", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_fixed_camera_native_replay>", "exec"), namespace)
    return namespace["run"](a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("collection", "analysis", "output-dir", "reference", "model", "legacy-evaluator"):
        p.add_argument("--"+name, type=Path, required=True)
    run(p.parse_args())
