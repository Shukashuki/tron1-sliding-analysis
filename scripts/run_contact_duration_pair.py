"""Serial native S/T smokes and matched-budget refinements on fully idle GPUs."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="1819390955a6a1a93f8c909463fe50f80fe553308f660099f6471b4ec59246ea"
    for old,new in [("paired_K_M_warm_contact_refinement","paired_S_T_duration_contact_refinement"),
            ("contact_warm_refinement_launch_template.json","contact_duration_pair_launch_template.json"),
            ("contact_warm_refinement_400hz_v1.json","contact_duration_pair_400hz_v1.json"),
            ("train_contact_warm_refinement.py","train_contact_duration_pair.py"),
            ("verify_contact_warm_refinement.py","verify_contact_duration_pair.py"),
            ("('smoke','K',32,2),('smoke','M',32,2)","('smoke','S',32,2),('smoke','T',32,2)"),
            ("('formal','K',512,200),('formal','M',512,200)","('formal','S',512,200),('formal','T',512,200)"),
            ("                bound();command=template[:]","                bound();command=[x.replace('__SNAPSHOT__',str(snapshot)).replace('__WORKSPACE__',str(root)) for x in template]"),
            ("            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)",
             "            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)\n"
             "            assert helper.GPU_ORDER==(3,5,4)\n            helper.GPU_ORDER=(3,5,4,0,1,2,6,7)\n"
             "            previous=Path('/data/zengxiong/workspaces/tron1-contact-camber-reference-20261025-v2')\n"
             "            prior=json.loads((previous/'driver_status.json').read_text());native=json.loads((previous/'nominal_v1/status.json').read_text())\n"
             "            assert prior['status']=='formal_completed_verified_nominal_pending' and native['status']=='two_native_nominal_episodes_completed_require_analysis_and_render'\n"
             "            pids=[json.loads((previous/n).read_text())['pid'] for n in ['driver_v1_launch.json','nominal_continuation_v1_launch.json']]\n"
             "            pids += [c['pid'] for c in prior['children']]+[c['pid'] for c in native['cases']]\n"
             "            for pid in pids:\n                p=helper.process_snapshot(pid)\n                assert p is None or p['state']=='Z',('Previous owned job active',p)"),
            ("assert proof['status']=='completed_verified' and proof['variant']==variant and proof['run_kind']==mode",
             "assert proof['status']=='completed_verified' and proof['variant']==variant and proof['run_kind']==mode and proof['matched_scale_duration_verified']")]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    compile(source,"<paired_duration_serial_driver>","exec");return source


def run(a):
    ns=dict(__name__="_paired_duration_serial_driver",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"run_contact_warm_refinement.py").read_text()),"<paired_duration_serial_driver>","exec"),ns);ns["run"](a)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("workspace","snapshot","idle-audit-source","python"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--smoke-only",action="store_true");run(p.parse_args())
