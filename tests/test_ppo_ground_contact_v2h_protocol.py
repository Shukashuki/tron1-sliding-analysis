"""H changes reward identity/design only; all frozen acceptance gates survive."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2h as definition
import continue_ground_contact_v2h_validation as continuation


def test_protocol_diff_is_only_new_reward_identity_settings_and_experiment_design():
    old = json.loads((ROOT/"config/ppo_ground_contact_v2g_400hz_v1.json").read_text())
    new = json.loads((ROOT/"config/ppo_ground_contact_v2h_400hz_v1.json").read_text())
    changed = {key for key in old.keys()|new.keys() if old.get(key)!=new.get(key)}
    assert changed == {"cohort_id", "reward_definition_version", "ground_contact_reward_v2", "experiment_design"}
    assert new["ground_contact_reward_v2"] == definition.validate_settings()
    assert new["nominal_acceptance_before_training"] == old["nominal_acceptance_before_training"]
    assert new["training_randomization"] == old["training_randomization"]
    assert new["algorithm"] == old["algorithm"]


def test_continuation_pins_actual_new_sources_and_protocol_before_launch():
    assert continuation.PROTOCOL_SHA256 == continuation.sha256(ROOT/"config/ppo_ground_contact_v2h_400hz_v1.json")
    for name, digest in continuation.DEPENDENCIES.items():
        assert continuation.sha256(ROOT/"scripts"/name) == digest, name
    assert {"ppo_ground_contact_v2h.py", "ppo_ground_contact_v2g.py", "ground_contact_reward_v2h.py",
            "ground_contact_reward_v2g.py", "ground_contact_velocity_contract.py"} <= continuation.DEPENDENCIES.keys()
