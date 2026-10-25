"""Strict M warm-start/export verification plus unchanged live task-space targets."""
from __future__ import annotations

from pathlib import Path

import train_contact_taskspace_margin as training

PARENT_SHA = "07c43152ae4d32b87cdde7d97b4ecbb31fcb333b8884415ca4e192658e6e7f6e"


def namespace():
    p = Path(__file__).parent/"verify_contact_taskspace_refinement.py"
    assert training.sha(p) == PARENT_SHA
    source = p.read_text()
    for old, new in [
        ("import train_contact_taskspace_refinement as training", "import train_contact_taskspace_margin as training"),
        ('assert p["variants"] == ["K"] and result["variant"] == "K"', 'assert p["variants"] == ["M"] and result["variant"] == "M"'),
        ('variant="K", run_kind=', 'variant="M", run_kind='),
        ("strict_contact_taskspace_refinement_training_verification", "strict_contact_taskspace_margin4_training_verification"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    ns = dict(__name__="_taskspace_margin4_strict_verifier", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_margin4_strict_verifier>", "exec"), ns)
    return ns


def verify_training(directory, protocol_path, *, allow_smoke=False, cpu_inference=False):
    return namespace()["verify_training"](directory, protocol_path, allow_smoke=allow_smoke, cpu_inference=cpu_inference)


if __name__ == "__main__":
    namespace()["main"]()
