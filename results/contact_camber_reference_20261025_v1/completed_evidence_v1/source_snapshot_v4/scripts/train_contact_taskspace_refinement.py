"""K warm-start ablation: release artificial leg-link targets, retain car task.

Reuse the pinned warm-start/count/export implementation with explicit target
hooks. Four tracking definitions change; all weights/contact/DR/motors stay.
The completed warm K arm supplies the same-checkpoint, same-budget control.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import car_taskspace_tracking as targets
import train_contact_warm_refinement as frozen

COHORT_ID = "tron1_contact_taskspace_refinement_400hz_v1"
PARENT_SHA = "aae7e6e8ff39fb340c03122bbaffcce6f09931f9956717605cad17eb367257ef"
for _name in ("CHECKPOINT_SHA", "ACTOR_SHA", "WARM_STATE_SHA", "RANDOM_STATE_SHA", "BASE_TRANSITIONS",
              "RUNNER_SHA", "PPO_SHA", "REFINEMENT", "sha", "read", "write", "state_sha", "refinement_config"):
    globals()[_name] = getattr(frozen, _name)


def materialize_parent(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    pairs = [
        ("COHORT_ID='tron1_contact_warm_refinement_400hz_v1'", "COHORT_ID='"+COHORT_ID+"'"),
        ("assert protocol['variants']==['K','M']", "assert protocol['variants']==['K']"),
        ("for variant in ['K','M']:", "for variant in ['K']:"),
        ("choices=['K','M']", "choices=['K']"),
        ("    return protocol\n", "    _taskspace_validate(protocol)\n    return protocol\n"),
        ("        plan=dict(protocol=protocol,", "        ledger.update(_taskspace_sources())\n        plan=dict(protocol=protocol,"),
        ("            contract.update(physics_dt=.0025,", "            task['task_space_tracking'] = _taskspace_validate(protocol)\n"
         "            task['description'] = 'K contact reward; free intermediate leg-link targets, retain base/wheel task-space tracking; explicit warm start.'\n"
         "            contract.update(physics_dt=.0025,"),
        ("            contact.configure_sensors(cfg.scene);cfg.rewards.drift_grounded=contact.reward_term(.4)\n",
         "            contact.configure_sensors(cfg.scene);cfg.rewards.drift_grounded=contact.reward_term(.4)\n"
         "            current['taskspace_configured'] = _taskspace_configure(cfg, protocol['task_space_tracking'])\n"),
        ("            return dict(contact=a,runtime=cohort.runtime_audit(env,args,manifest['environment'],current))\n",
         "            return dict(contact=a,runtime=cohort.runtime_audit(env,args,manifest['environment'],current),\n"
         "                task_space_tracking=_taskspace_audit(env,current['taskspace_configured']))\n"),
    ]
    for old, new in pairs:
        assert source.count(old) == 1, "Missing/ambiguous task-space anchor: " + old[:80]
        source = source.replace(old, new, 1)
    compile(source, "<taskspace_K_warm_parent>", "exec")
    return source


def validate_targets(protocol):
    return targets.settings(protocol["task_space_tracking"])


def source_ledger():
    here = Path(__file__).resolve().parent
    files = [Path(__file__), Path(targets.__file__), here/"verify_contact_taskspace_refinement.py"]
    return {str(p.resolve()): sha(p) for p in files}


def run(argv):
    namespace = dict(__name__="_taskspace_pinned_warm_parent", __file__=frozen.__file__,
                     _taskspace_validate=validate_targets, _taskspace_sources=source_ledger,
                     _taskspace_configure=targets.configure, _taskspace_audit=targets.runtime_audit)
    source = materialize_parent(Path(frozen.__file__).read_text())
    exec(compile(source, "<taskspace_K_warm_parent>", "exec"), namespace)
    namespace["run"](argv)


if __name__ == "__main__":
    run(sys.argv[1:])
