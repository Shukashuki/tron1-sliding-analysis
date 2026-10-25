"""Same-checkpoint/budget K refinement: 3D axial wheel direction versus2D heading."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import train_contact_taskspace_refinement as parent

COHORT_ID = "tron1_contact_wheel_axis_refinement_400hz_v1"
PARENT_SHA = "7ea8d431525484404646f7d9efb2e52786e40e6565b93c3b0a32a280c19c05fb"
for _name in ("CHECKPOINT_SHA", "ACTOR_SHA", "WARM_STATE_SHA", "RANDOM_STATE_SHA", "BASE_TRANSITIONS",
              "RUNNER_SHA", "PPO_SHA", "REFINEMENT", "sha", "read", "write", "state_sha", "refinement_config"):
    globals()[_name] = getattr(parent, _name)


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("import car_taskspace_tracking as targets", "import car_wheel_axis_tracking as targets"),
        ('COHORT_ID = "tron1_contact_taskspace_refinement_400hz_v1"', 'COHORT_ID = "'+COHORT_ID+'"'),
        ('here/"verify_contact_taskspace_refinement.py"', 'here/"verify_contact_wheel_axis.py"'),
        ("K contact reward; free intermediate leg-link targets, retain base/wheel task-space tracking; explicit warm start.",
         "K contact reward; same car task-space objectives but spin-invariant3D wheel-axis tracking replaces2D heading; explicit matched warm start."),
        ('files = [Path(__file__), Path(targets.__file__), here/"verify_contact_wheel_axis.py"]',
         'files = [Path(__file__), Path(targets.__file__), here/"verify_contact_wheel_axis.py",\n'
         '             here/"car_taskspace_tracking.py", here/"train_contact_taskspace_refinement.py"]'),
        ('    return {str(p.resolve()): sha(p) for p in files}\n',
         '    return {**{str(p.resolve()): sha(p) for p in files}, **_axis_actual_sources()}\n'),
    ]:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    compile(source, "<3D_wheel_axis_matched_refinement>", "exec")
    return source


def actual_sources():
    names = ["ground_contact_reward_v2"+suffix for suffix in ("", "b", "c", "d", "f", "g", "i", "k")]
    names += ["ppo_ground_contact_v2k", "ground_contact_beta_contract"]
    paths = [Path(importlib.import_module(name).__file__).resolve() for name in names]
    return {str(path): sha(path) for path in paths}


def run(argv):
    namespace = dict(__name__="_3D_wheel_axis_matched_refinement", __file__=str(Path(__file__).resolve()),
                     _axis_actual_sources=actual_sources)
    exec(compile(materialize(Path(parent.__file__).read_text()), "<3D_wheel_axis_matched_refinement>", "exec"), namespace)
    namespace["run"](argv)


if __name__ == "__main__":
    run(sys.argv[1:])
