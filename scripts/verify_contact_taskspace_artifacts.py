"""Recompute four native traces and decode the complete task-space video."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "4c3f99c7262b4ebc5080d44e937dbf333e72fe8b0bbd0bb86411e0f9c5ee4086"


def run(root):
    path = Path(__file__).parent/"verify_contact_warm_refinement_artifacts.py"
    source = path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("import analyze_contact_warm_refinement as analysis", "import analyze_contact_taskspace_refinement as analysis"),
        ("scripts/render_contact_warm_refinement.py", "scripts/render_contact_taskspace_refinement.py"),
        ("proofs[\"formal_M\"][\"actor_sha256\"]==provenance[\"actor_sha256_M\"]", "proofs[\"taskspace\"][\"actor_sha256\"]==provenance[\"actor_sha256_taskspace\"]"),
        ("proofs[\"formal_K\"][\"actor_sha256\"]==provenance[\"actor_sha256_K\"]", "proofs[\"control\"][\"actor_sha256\"]==provenance[\"actor_sha256_control\"]"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    namespace = dict(__name__="_taskspace_native_artifact_verifier", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_native_artifact_verifier>", "exec"), namespace)
    return namespace["run"](root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--result", type=Path, required=True)
    run(p.parse_args().result)
