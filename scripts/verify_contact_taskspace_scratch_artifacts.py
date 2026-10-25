"""Recompute four scratch-policy traces and verify all native replay frames."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "4c3f99c7262b4ebc5080d44e937dbf333e72fe8b0bbd0bb86411e0f9c5ee4086"


def run(root):
    source = (Path(__file__).parent/"verify_contact_warm_refinement_artifacts.py").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [("import analyze_contact_warm_refinement as analysis", "import analyze_contact_taskspace_scratch as analysis"),
                     ("scripts/render_contact_warm_refinement.py", "scripts/render_contact_taskspace_scratch.py")]:
        assert source.count(old) == 1
        source = source.replace(old, new, 1)
    ns = dict(__name__="_scratch_native_artifact_verifier", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<scratch_native_artifact_verifier>", "exec"), ns)
    return ns["run"](root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--result", type=Path, required=True); run(p.parse_args().result)
