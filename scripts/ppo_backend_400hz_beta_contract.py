"""Strict beta-OFF completion/evaluation contract, isolated from baseline files.

The immutable baseline verifier is SHA-checked and adapted only in memory to
the new trainer/protocol/evaluator names. Additional checks forbid relabeling
a beta-ON export, changing any other task value, or changing initialization.
"""
from __future__ import annotations

from pathlib import Path
import types

import train_ppo_backend_400hz_beta_off as trainer

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONTRACT_SHA256 = "b57b7444698e06f4d9faed6f246bb8a301b505f6c785bdbc2caafdb0b2b6f56e"
DEFAULT_PROTOCOL = trainer.DEFAULT_PROTOCOL
require, sha256 = trainer.require, trainer.sha256


def materialize_source(source):
    import hashlib
    require(hashlib.sha256(source.encode()).hexdigest() == BASELINE_CONTRACT_SHA256,
            "Immutable baseline400Hz contract SHA differs")
    replacements = [
        ('"config/ppo_backend_400hz_comparison_v1.json"', '"config/ppo_backend_400hz_beta_off_v1.json"', 1),
        ('TRAINER_SHA256 = "' + trainer.BASELINE_WRAPPER_SHA256 + '"',
         'TRAINER_SHA256 = "' + sha256(Path(trainer.__file__)) + '"', 1),
        ('"train_ppo_backend_400hz.py"', '"train_ppo_backend_400hz_beta_off.py"', 1),
        ('"scripts/train_ppo_backend_400hz.py"', '"scripts/train_ppo_backend_400hz_beta_off.py"', 1),
        ('from train_ppo_backend_400hz import validate_protocol',
         'from train_ppo_backend_400hz_beta_off import validate_protocol', 1),
        ('from train_ppo_backend_400hz import validate_runtime_fields',
         'from train_ppo_backend_400hz_beta_off import validate_runtime_fields', 1),
        ('"eval_ppo_backend_400hz_isaac.py"', '"eval_ppo_backend_400hz_beta_off_isaac.py"', 1),
        ('"eval_ppo_backend_400hz_mujoco.py"', '"eval_ppo_backend_400hz_beta_off_mujoco.py"', 1),
    ]
    for old, new, count in replacements:
        require(source.count(old) == count, "Frozen beta-off verifier anchor absent/ambiguous: " + old)
        source = source.replace(old, new, count)
    compile(source, "<isolated_beta_off_400hz_contract>", "exec")
    return source


def _load_frozen():
    source = ROOT / "scripts/ppo_backend_400hz_contract.py"
    module = types.ModuleType("_private_beta_off_400hz_contract")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(materialize_source(source.read_text()), "<isolated_beta_off_400hz_contract>", "exec"), module.__dict__)
    return module


_CORE = _load_frozen()
_VERIFY_COHORT, _PROVENANCE = _CORE.verify_cohort, _CORE.provenance


def verify_cohort(cohort_dir, protocol_path=DEFAULT_PROTOCOL, *, allow_smoke=False):
    cohort = _VERIFY_COHORT(cohort_dir, protocol_path, allow_smoke=allow_smoke)
    expected = trainer.ablation_identity()
    trainer.validate_task_profile(cohort["contract"]["trained_task_profile"], cohort["protocol"])
    for item in (cohort["contract"], cohort["completion"], cohort["manifest"], cohort["manifest"]["cohort"],
                 _CORE.read_json(cohort["paths"]["cohort_startup_audit.json"]),
                 _CORE.read_json(cohort["paths"]["run_status.json"])):
        require(item.get("beta_ablation") == expected, "Missing/different beta-off ablation identity")
    require(cohort["contract"].get("sideslip_reward_weight") == 0.
            and cohort["completion"].get("sideslip_reward_weight") == 0.
            and cohort["completion"].get("initial_policy_matches_beta_on") is True
            and cohort["completion"].get("initial_policy_state_sha256") == trainer.INITIAL_POLICY_SHA256,
            "A beta-on/differently initialized policy cannot be accepted as beta-off")
    require(cohort["completion"].get("declared_reward_term_count") == 15
            and cohort["completion"].get("listed_reward_term_count") == 15
            and cohort["completion"].get("nonzero_reward_term_count") == 14
            and cohort["completion"].get("sideslip_term_listed") is True
            and cohort["completion"].get("effective_sideslip_reward_enabled") is False,
            "Beta-off listed/nonzero/declared reward count differs")
    return cohort


def provenance(cohort):
    return {**_PROVENANCE(cohort), "beta_ablation": trainer.ablation_identity(), "sideslip_reward_weight": 0.,
        "initial_policy_matches_beta_on": True}


_CORE.verify_cohort, _CORE.provenance = verify_cohort, provenance
for _name in ("INPUT_PINS", "EVALUATION_DEPENDENCIES", "TRAINER_SHA256", "read_json", "write_json",
    "verify_protocol", "evaluation_contract", "prepare_inputs", "verify_legacy", "verify_recording",
    "validate_output", "StrictPPOExportRunner"):
    globals()[_name] = getattr(_CORE, _name)
