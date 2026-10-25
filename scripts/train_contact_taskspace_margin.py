"""Task-space M: same original K warm start; change only load-margin1 to4.

The completed task-space K arm is the matched-budget control. This is not a
resume of that resulting actor and does not change geometry, motors or gates.
"""
from __future__ import annotations

from pathlib import Path
import sys

import train_contact_taskspace_refinement as parent

COHORT_ID = "tron1_contact_taskspace_margin4_refinement_400hz_v1"
PARENT_SHA = "7ea8d431525484404646f7d9efb2e52786e40e6565b93c3b0a32a280c19c05fb"
for _name in ("CHECKPOINT_SHA", "ACTOR_SHA", "WARM_STATE_SHA", "RANDOM_STATE_SHA", "BASE_TRANSITIONS",
              "RUNNER_SHA", "PPO_SHA", "REFINEMENT", "sha", "read", "write", "state_sha", "refinement_config"):
    globals()[_name] = getattr(parent, _name)


def materialize(source):
    assert sha(parent.__file__) == PARENT_SHA
    source = parent.materialize_parent(source)
    for old, new in [
        ("COHORT_ID='"+parent.COHORT_ID+"'", "COHORT_ID='"+COHORT_ID+"'"),
        ("assert protocol['variants']==['K']", "assert protocol['variants']==['M']"),
        ("for variant in ['K']:", "for variant in ['M']:"),
        ("choices=['K']", "choices=['M']"),
        ("K contact reward; free intermediate leg-link targets", "M margin4 contact reward; free intermediate leg-link targets"),
    ]:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    compile(source, "<taskspace_M_margin4_warm_parent>", "exec")
    return source


def source_ledger():
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), Path(parent.__file__), Path(parent.targets.__file__),
             here/"verify_contact_taskspace_margin.py"]
    return {str(p.resolve()): sha(p) for p in paths}


def namespace():
    ns = dict(__name__="_taskspace_M_margin4_parent", __file__=parent.frozen.__file__,
              _taskspace_validate=parent.validate_targets, _taskspace_sources=source_ledger,
              _taskspace_configure=parent.targets.configure, _taskspace_audit=parent.targets.runtime_audit)
    source = materialize(Path(parent.frozen.__file__).read_text())
    exec(compile(source, "<taskspace_M_margin4_warm_parent>", "exec"), ns)
    return ns


def run(argv):
    namespace()["run"](argv)


if __name__ == "__main__":
    run(sys.argv[1:])
