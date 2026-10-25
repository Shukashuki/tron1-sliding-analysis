"""Fixed-distance four-panel K/M task-space comparison from original poses."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import render_contact_taskspace_refinement as parent

PARENT_SHA = "6cf961071b709235ff20e13b8e3497911355d94eff9ffba5ef33048832772175"


def materialize(source):
    assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() == PARENT_SHA
    source = parent.materialize(source)
    for old, new in [
        ("import analyze_contact_taskspace_refinement as analysis", "import analyze_contact_taskspace_margin as analysis"),
        ("matched_warm_K_taskspace_refinement_dual_engine", "matched_taskspace_K_M_margin_refinement_dual_engine"),
        ("Warm K control / task-space | same K checkpoint +2.46M each", "Task-space K / M margin4 | same K checkpoint +2.46M each"),
        ("four_actual_warm_K_taskspace_native_pose_replay", "four_actual_taskspace_K_M_margin4_native_pose_replay"),
        ("actor_sha256_taskspace=training['taskspace']", "actor_sha256_margin4=training['margin4']"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<taskspace_K_M_fixed_camera_replay>", "exec")
    return source


def run(a):
    source = materialize((Path(__file__).parent/"render_contact_warm_refinement.py").read_text())
    ns = dict(__name__="_taskspace_K_M_fixed_camera_replay", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_K_M_fixed_camera_replay>", "exec"), ns)
    return ns["run"](a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("collection", "analysis", "output-dir", "reference", "model", "legacy-evaluator"):
        p.add_argument("--"+n, type=Path, required=True)
    run(p.parse_args())
