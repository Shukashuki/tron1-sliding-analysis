"""Four scratch-policy native episodes, fixed relative camera and opaque poses."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "42e57a87b89fb8eb4d665babec2af20a4de627651692f57aca1a1b810ac976c2"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    pairs = [
        ("import analyze_contact_warm_refinement as analysis", "import analyze_contact_taskspace_scratch as analysis"),
        ("paired_warm_K_M_contact_refinement_dual_engine", "paired_scratch_taskspace_K_M_contact_dual_engine"),
        ("four_actual_paired_warm_contact_refinement_native_pose_replay", "four_actual_fresh_taskspace_contact_native_pose_replay"),
        ("variant='warm_contact_refinement'", "variant='fresh_taskspace_contact'"),
        ("row['variant']+' warm refinement -> '+label", "row['variant']+' fresh task-space -> '+label"),
        ("Paired warm K/M | same K checkpoint +2.46M transitions each", "Fresh task-space K/M |12.288M transitions each"),
    ]
    for old, new in pairs:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    compile(source, "<scratch_K_M_native_pose_replay>", "exec")
    return source


def run(args):
    source = materialize((Path(__file__).parent/"render_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_scratch_K_M_native_pose_replay", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<scratch_K_M_native_pose_replay>", "exec"), namespace)
    namespace["run"](args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("collection", "analysis", "output-dir", "reference", "model", "legacy-evaluator"):
        p.add_argument("--"+n, type=Path, required=True)
    run(p.parse_args())
