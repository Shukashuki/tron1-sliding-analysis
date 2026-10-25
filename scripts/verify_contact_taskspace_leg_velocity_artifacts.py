"""Independently recompute M/R metrics, verify cameras and decode350 frames."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA="4c3f99c7262b4ebc5080d44e937dbf333e72fe8b0bbd0bb86411e0f9c5ee4086"


def run(root):
    p=Path(__file__).parent/"verify_contact_warm_refinement_artifacts.py"
    source=p.read_text();assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    for old,new in [
        ("import analyze_contact_warm_refinement as analysis","import analyze_contact_taskspace_leg_velocity as analysis"),
        ("scripts/render_contact_warm_refinement.py","scripts/render_contact_taskspace_leg_velocity.py"),
        ('proofs["formal_M"]["actor_sha256"]==provenance["actor_sha256_M"]','proofs["leg_velocity"]["actor_sha256"]==provenance["actor_sha256_leg_velocity"]'),
        ('proofs["formal_K"]["actor_sha256"]==provenance["actor_sha256_K"]','proofs["control"]["actor_sha256"]==provenance["actor_sha256_control"]'),
    ]:
        assert old in source,old;source=source.replace(old,new)
    ns=dict(__name__="_M_R_native_artifact_verifier",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<M_R_native_artifact_verifier>","exec"),ns);return ns["run"](root)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument("--result",type=Path,required=True)
    run(p.parse_args().result)
