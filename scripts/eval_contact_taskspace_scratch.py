"""Native held-action validation of independently verified fresh task-space PPO.

Reuse the frozen recording/reset/material lifecycle, with fresh K/M admission.
No training reward owner, actuator filter or extra physics step is installed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

PARENT_SHA = "13a5ebd7bc36ce54d86d21e2935d5c6fb4306d79102170b5997c8970a8f77d0a"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    pairs = [
        ("import train_contact_warm_refinement as training", "import train_contact_taskspace_scratch as training"),
        ("import verify_contact_warm_refinement as verification", "import verify_contact_taskspace_scratch as verification"),
        ("_refine_config=training.refinement_config", "_scratch_config=lambda config:config"),
        ('require(config == _refine_config(ppo_config(42, config["save_interval"])), "PPO refinement runner configuration differs")',
         'require(config == _scratch_config(ppo_config(42, config["save_interval"])), "Fresh PPO runner configuration differs")'),
        ("initial_policy_state_sha256=training.WARM_STATE_SHA,source_K_checkpoint_sha256=training.CHECKPOINT_SHA,\n"
         "            source_K_actor_sha256=training.ACTOR_SHA,from_scratch=False,historical_checkpoint_used=True,\n"
         "            additional_refinement_transitions=c['completion']['training_environment_steps'],\n"
         "            cumulative_training_transitions=c['completion']['cumulative_training_environment_steps'],",
         "initial_policy_state_sha256=training.RANDOM_STATE_SHA,from_scratch=True,historical_checkpoint_used=False,\n"
         "            initial_checkpoint=None,normalizer_initial_count=0,\n"
         "            training_transitions=c['completion']['training_environment_steps'],"),
        ("for f in [__file__,training.__file__,verification.__file__]}",
         "for f in [__file__,training.__file__,verification.__file__,_scratch_frozen_eval_source,\n"
         "        own.k_snapshot/'scripts/eval_ppo_ground_contact_v2k.py',\n"
         "        own.base_snapshot/'scripts/ppo_backend_400hz_contract.py',\n"
         "        own.base_snapshot/'scripts/eval_ppo_backend_400hz_mujoco.py',\n"
         "        own.base_snapshot/'scripts/eval_ppo_backend_400hz_isaac.py']}")
    ]
    for old, new in pairs:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    assert source.count("kind='warm_contact_refinement_native_evaluation'") == 1
    source = source.replace("kind='warm_contact_refinement_native_evaluation'", "kind='fresh_taskspace_contact_native_evaluation'")
    assert source.count("contact_warm_refinement_evaluation") == 1
    source = source.replace("contact_warm_refinement_evaluation", "contact_taskspace_scratch_evaluation")
    assert source.count("warm_contact_evaluation") == 3
    source = source.replace("warm_contact_evaluation", "scratch_contact_evaluation")
    compile(source, "<fresh_taskspace_native_evaluation>", "exec")
    return source


def main(argv=None):
    parent = Path(__file__).parent/"eval_contact_warm_refinement.py"
    generated = materialize(parent.read_text())
    namespace = dict(__name__="_fresh_taskspace_native_wrapper", __file__=str(Path(__file__).resolve()),
                     _scratch_frozen_eval_source=parent.resolve())
    previous = list(sys.path), sys.argv
    try:
        exec(compile(generated, "<fresh_taskspace_native_evaluation>", "exec"), namespace)
        return namespace["main"](argv)
    finally:
        sys.path[:], sys.argv = previous


if __name__ == "__main__":
    main(sys.argv[1:])
