"""Recorded S/T signed camber,loaded-point slip,and whole-body integrity."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="5df34dbb23c49fe1db4827ff153e88f43e6aceab245b7973842cdd9b43a46ba0"
    for old,new in [
        ("import analyze_contact_camber_reference as analysis","import analyze_contact_duration_pair as analysis"),
        ("completed_evidence_v1/source_snapshot_v4/references/fit_differential.npz","completed_evidence_v1/source_snapshot_v3/references/fit_differential.npz"),
        ('new["wheel_camber_rad"][phase] if row["variant"]=="camber" else np.zeros_like(gamma)','new["wheel_camber_rad"][phase]'),
        ('            base=analysis.CONTROL if row["variant"]=="control" else root/"completed_evidence_v1"','            base=root/"completed_evidence_v1"'),
        ('("K_"+row["engine"])','(row["variant"]+"_"+row["engine"])'),
    ]:
        assert source.count(old)==1,old;source=source.replace(old,new)
    compile(source,"<duration_pair_loaded_contact_diagnostic>","exec");return source


def run(a):
    ns=dict(__name__="_duration_pair_loaded_contact_diagnostic",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"diagnose_camber_tracking.py").read_text()),"<duration_pair_loaded_contact_diagnostic>","exec"),ns)
    return ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
    run(p.parse_args())
