"""Dense exact-support audit using the same stable separation certificate."""
from __future__ import annotations
import argparse
from pathlib import Path

import generate_camber_reference as base
from audit_hybrid_contact_support_dense import DenseSupportProblem
from hybrid_contact_reference_stable import _evaluate,MATERIALIZED_EVALUATE_SHA


class StableDenseSupportProblem(DenseSupportProblem):
    evaluate=_evaluate


def run(a):
    source=Path(__file__).with_name('audit_hybrid_contact_support_dense.py').read_text()
    assert base.sha(Path(__file__).with_name('audit_hybrid_contact_support_dense.py'))=='2a4e6723b996bc4fe292c5e353542075470aad2931b5a768133c1242da10d392'
    old='p=DenseSupportProblem(mujoco.MjModel';assert source.count(old)==1
    source=source.replace(old,'p=_StableDenseSupportProblem(mujoco.MjModel',1)
    old="support_planner_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference_support.py')),"
    assert source.count(old)==1
    source=source.replace(old,old+"\n        stable_separation_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference_stable.py')),\n        materialized_evaluate_sha256='"+MATERIALIZED_EVALUATE_SHA+"',",1)
    ns=dict(__name__='_stable_dense_audit',__file__=str(Path(__file__).resolve()),_StableDenseSupportProblem=StableDenseSupportProblem)
    exec(compile(source,'<stable_dense_audit>','exec'),ns);ns['run'](a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','fit','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
