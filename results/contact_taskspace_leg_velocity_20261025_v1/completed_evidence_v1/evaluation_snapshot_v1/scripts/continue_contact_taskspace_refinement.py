"""Wait on the exact task-space driver, then evaluate both native engines."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "f81f56eccf65c87f8f07d32792a95b2acfe74800a283f495f23b668ecaa1a08a"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("paired_warm_K_M_four_native_nominal_episodes", "taskspace_K_two_native_nominal_episodes"),
        ("run_contact_warm_refinement.py", "run_contact_taskspace_refinement.py"),
        ("both_formal_arms_completed_verified_nominal_pending", "formal_completed_verified_nominal_pending"),
        ("['K','M']", "['K']"),
        ("contact_warm_refinement_launch_template.json", "contact_taskspace_refinement_launch_template.json"),
        ("eval_contact_warm_refinement.py", "eval_contact_taskspace_refinement.py"),
        ("contact_warm_refinement_400hz_v1.json", "contact_taskspace_refinement_400hz_v1.json"),
        ("four_native_nominal_episodes_completed_require_analysis_and_render", "two_native_nominal_episodes_completed_require_analysis_and_render"),
        ("waiting_for_exact_paired_training_driver", "waiting_for_exact_taskspace_training_driver"),
    ]:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, "<taskspace_serial_native_continuation>", "exec")
    return source


def run(a):
    source = materialize((Path(__file__).parent/"continue_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_taskspace_serial_native_continuation", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<taskspace_serial_native_continuation>", "exec"), namespace)
    namespace["run"](a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("workspace", "evaluation-snapshot", "training-snapshot", "idle-audit-source", "python", "legacy-physx", "legacy-mujoco"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--expected-driver-pid", type=int, required=True)
    run(p.parse_args())
