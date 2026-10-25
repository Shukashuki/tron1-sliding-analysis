"""Matched K/3D-axis warm refinement with an explicitly new camber reference."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys

import train_contact_wheel_axis as parent
import camber_reference_contract as reference

COHORT_ID="tron1_contact_camber_reference_400hz_v1"
for _name in ("CHECKPOINT_SHA","ACTOR_SHA","WARM_STATE_SHA","RANDOM_STATE_SHA","BASE_TRANSITIONS", "RUNNER_SHA","PPO_SHA","REFINEMENT","sha","read","write","state_sha","refinement_config"):
    globals()[_name]=getattr(parent,_name)


def parent_source(source):
    assert source.count("import sys\n")==1
    source=source.replace("import sys\n","import sys\nimport camber_reference_contract as _camber_reference\n",1)
    old="            for n,h in cohort.INPUT_PINS.items():assert sha(getattr(args,n))==h\n"
    new=("            current['camber_reference']=_camber_reference.validate_reference(args.motion,protocol)\n"
         "            ledger.update(current['camber_reference']['input_sha256'])\n"
         "            for n,h in {**cohort.INPUT_PINS,'motion':current['camber_reference']['motion_sha256']}.items():assert sha(getattr(args,n))==h\n")
    assert source.count(old)==1;source=source.replace(old,new,1)
    old="            manifest.update(warm_start=copy.deepcopy(current['warm_initialization']),reward_definition_version=version,\n"
    assert source.count(old)==1
    source=source.replace(old,"            manifest['kinematic_reference']=copy.deepcopy(current['camber_reference'])\n"+old,1)
    old="                task_space_tracking=_taskspace_audit(env,current['taskspace_configured']))\n"
    assert source.count(old)==1
    source=source.replace(old,"                task_space_tracking=_taskspace_audit(env,current['taskspace_configured']),\n"
        "                kinematic_reference=_camber_reference.live_audit(env,args.motion,protocol))\n",1)
    old="            out.mkdir(parents=True);write(out/'contract.json',contract);write(out/'task_profile.json',task)\n"
    assert source.count(old)==1
    source=source.replace(old,"            contract=_camber_reference.bind_declared_contract(contract,current['camber_reference'])\n"+old,1)
    return source


def materialize(source):
    assert sha(parent.__file__)=="94c5482ea0f89b002449c24368cc96db961b18c9f6c0c9583f5a872a11049fe6"
    source=parent.materialize(source)
    for old,new in [
        ('COHORT_ID = "'+parent.COHORT_ID+'"','COHORT_ID = "'+COHORT_ID+'"'),
        ('here/"verify_contact_wheel_axis.py"','here/"verify_contact_camber_reference.py"'),
        ('    return targets.settings(protocol["task_space_tracking"])',
         '    _camber_validate_protocol(protocol)\n    return targets.settings(protocol["task_space_tracking"])'),
        ('source = materialize_parent(Path(frozen.__file__).read_text())',
         'source = _camber_parent_source(materialize_parent(Path(frozen.__file__).read_text()))'),
        ('**_axis_actual_sources()', '**_axis_actual_sources(), **_camber_sources()'),
    ]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    compile(source,"<matched_camber_reference_refinement>","exec");return source


def sources():
    paths=[Path(parent.__file__),Path(reference.__file__)]
    return {str(p.resolve()):sha(p) for p in paths}


def run(argv):
    here=Path(__file__).resolve().parent
    ns=dict(__name__="_matched_camber_reference_refinement",__file__=str(Path(__file__).resolve()),
        _axis_actual_sources=parent.actual_sources,_camber_sources=sources,
        _camber_validate_protocol=reference.validate_protocol,_camber_validate_reference=reference.validate_reference,
        _camber_live_audit=reference.live_audit,_camber_parent_source=parent_source)
    exec(compile(materialize((here/"train_contact_taskspace_refinement.py").read_text()),"<matched_camber_reference_refinement>","exec"),ns)
    ns["run"](argv)


if __name__=="__main__":run(sys.argv[1:])
