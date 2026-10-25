"""Four native S/T poses,opaque meshes and identical fixed relative cameras."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="42e57a87b89fb8eb4d665babec2af20a4de627651692f57aca1a1b810ac976c2"
    for old,new in [
        ("import analyze_contact_warm_refinement as analysis","import analyze_contact_duration_pair as analysis"),
        ("paired_warm_K_M_contact_refinement_dual_engine","matched_support_duration_S_T_dual_engine"),
        ("Paired warm K/M | same K checkpoint +2.46M transitions each","S:worst+mean /T:twice worst | same K checkpoint +2.46M each"),
        ("four_actual_paired_warm_contact_refinement_native_pose_replay","four_actual_matched_support_duration_native_pose_replay"),
        ("actor_sha256_K=training['formal_K']","actor_sha256_S=training['formal_S']"),
        ("actor_sha256_M=training['formal_M']","actor_sha256_T=training['formal_T']"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<duration_pair_native_replay>","exec");return source


def run(a):
    ns=dict(__name__="_duration_pair_native_replay",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"render_contact_warm_refinement.py").read_text()),"<duration_pair_native_replay>","exec"),ns)
    return ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("collection","analysis","output-dir","reference","model","legacy-evaluator"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
