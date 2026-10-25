"""Fixed-relative4x1 native replay:old/new kinematic references,same3D reward."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="42e57a87b89fb8eb4d665babec2af20a4de627651692f57aca1a1b810ac976c2"
    for old,new in [
        ("import analyze_contact_warm_refinement as analysis","import analyze_contact_camber_reference as analysis"),
        ("paired_warm_K_M_contact_refinement_dual_engine","matched_original_independent_camber_reference_dual_engine"),
        ("Paired warm K/M | same K checkpoint +2.46M transitions each","Original /camber reference | same K checkpoint,3D reward,+2.46M each"),
        ("four_actual_paired_warm_contact_refinement_native_pose_replay","four_actual_original_camber_reference_native_pose_replay"),
        ("actor_sha256_K=training['formal_K']","actor_sha256_control=training['control']"),
        ("actor_sha256_M=training['formal_M']","actor_sha256_camber=training['camber']"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<camber_reference_native_pose_replay>","exec");return source


def run(a):
    ns=dict(__name__="_camber_reference_native_pose_replay",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"render_contact_warm_refinement.py").read_text()),"<camber_reference_native_pose_replay>","exec"),ns)
    return ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("collection","analysis","output-dir","reference","model","legacy-evaluator"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
