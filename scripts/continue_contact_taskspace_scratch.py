"""Wait for exact fresh K/M driver, then four serial native mu0.7 episodes.

No elapsed-time deadline cancels observation or compute. Source changes and
failed authoritative training evidence fail closed; no job is ever signalled.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA = "f81f56eccf65c87f8f07d32792a95b2acfe74800a283f495f23b668ecaa1a08a"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    pairs = [
        ("paired_warm_K_M_four_native_nominal_episodes", "paired_fresh_taskspace_K_M_four_native_nominal_episodes"),
        ("run_contact_warm_refinement.py", "run_contact_taskspace_scratch.py"),
        ("            deadline=time.monotonic()+7200\n", ""),
        ("                assert time.monotonic()<deadline,'Observation deadline; all compute is preserved'\n", ""),
        ("both_formal_arms_completed_verified_nominal_pending", "both_formal_verified_nominal_validation_pending"),
        ("assert proof['transitions']==2457600 and proof['cumulative_transitions']==14745600",
         "assert proof['transitions']==12288000 and proof['initial_policy_state_sha256']=='7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d'"),
        ("assert not proof['from_scratch'] and proof['historical_checkpoint_used']",
         "assert proof['from_scratch'] and not proof['historical_checkpoint_used']"),
        ("contact_warm_refinement_launch_template.json", "contact_taskspace_scratch_launch_template.json"),
        ("eval_contact_warm_refinement.py", "eval_contact_taskspace_scratch.py"),
        ("option('--k-snapshot')", "str(Path(option('--baseline-protocol')).parent.parent)"),
        ("contact_warm_refinement_400hz_v1.json", "contact_taskspace_scratch_400hz_v1.json"),
    ]
    for old, new in pairs:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    assert source.count("warm_contact_evaluation") == 3
    source = source.replace("warm_contact_evaluation", "scratch_contact_evaluation")
    compile(source, "<fresh_taskspace_exact_driver_native_continuation>", "exec")
    return source


def run(args):
    source = materialize((Path(__file__).parent/"continue_contact_warm_refinement.py").read_text())
    namespace = dict(__name__="_fresh_taskspace_native_continuation", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<fresh_taskspace_exact_driver_native_continuation>", "exec"), namespace)
    namespace["run"](args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("workspace", "evaluation-snapshot", "training-snapshot", "idle-audit-source", "python", "legacy-physx", "legacy-mujoco"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--expected-driver-pid", type=int, required=True)
    run(p.parse_args())
