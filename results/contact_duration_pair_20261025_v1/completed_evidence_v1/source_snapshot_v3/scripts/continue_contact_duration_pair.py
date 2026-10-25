"""Exact paired-driver wait,then four serial native S/T nominal episodes."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="f81f56eccf65c87f8f07d32792a95b2acfe74800a283f495f23b668ecaa1a08a"
    for old,new in [("paired_warm_K_M_four_native_nominal_episodes","paired_duration_S_T_four_native_nominal_episodes"),
            ("run_contact_warm_refinement.py","run_contact_duration_pair.py"),
            ("eval_contact_warm_refinement.py","eval_contact_duration_pair.py"),
            ("contact_warm_refinement_launch_template.json","contact_duration_pair_launch_template.json"),
            ("contact_warm_refinement_400hz_v1.json","contact_duration_pair_400hz_v1.json"),
            ("['K','M']","['S','T']"),
            ("            deadline=time.monotonic()+7200\n",""),
            ("                assert time.monotonic()<deadline,'Observation deadline; all compute is preserved'\n",""),
            ("            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)",
             "            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)\n"
             "            assert helper.GPU_ORDER==(3,5,4)\n            helper.GPU_ORDER=(3,5,4,0,1,2,6,7)"),
            ("            option=lambda f:template[template.index(f)+1]",
             "            option=lambda f:template[template.index(f)+1].replace('__SNAPSHOT__',str(a.training_snapshot.resolve())).replace('__WORKSPACE__',str(root))")]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<paired_duration_native_continuation>","exec");return source


def run(a):
    ns=dict(__name__="_paired_duration_native_continuation",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"continue_contact_warm_refinement.py").read_text()),"<paired_duration_native_continuation>","exec"),ns);ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","evaluation-snapshot","training-snapshot","idle-audit-source","python","legacy-physx","legacy-mujoco"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--expected-driver-pid",type=int,required=True);run(p.parse_args())
