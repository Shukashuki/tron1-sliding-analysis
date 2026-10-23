"""C changes only identity plus the fixed ungated support-deficit coefficient."""
import ast
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2c as math


def test_protocol_changes_only_identity_and_deficit_not_acceptance_DR_or_physics():
    b = json.loads((ROOT/"config/ppo_ground_contact_v2b_400hz_v1.json").read_text())
    c = json.loads((ROOT/"config/ppo_ground_contact_v2c_400hz_v1.json").read_text())
    assert c["cohort_id"] == "tron1_ground_contact_v2c_any_wheel_400hz_v1"
    assert c["reward_definition_version"] == "ground_contact_v2c_any_wheel"
    assert c["ground_contact_reward_v2"] == math.validate_settings()
    assert c["nominal_acceptance_before_training"] == b["nominal_acceptance_before_training"]
    c["cohort_id"], c["reward_definition_version"] = b["cohort_id"], b["reward_definition_version"]
    c["ground_contact_reward_v2"]["candidate"] = "ground_contact_v2b"
    assert c["ground_contact_reward_v2"].pop("support_deficit_weight") == 1.
    assert c == b


def test_Ccollector_native_capture_reward_clock_exactly_unchanged_from_B():
    trees = [ast.parse((ROOT/"scripts"/name).read_text()) for name in ("ppo_ground_contact_v2b.py", "ppo_ground_contact_v2c.py")]
    classes = [next(node for node in tree.body if isinstance(node, ast.ClassDef)) for tree in trees]
    methods = [{node.name: ast.dump(node, include_attributes=False) for node in cls.body if isinstance(node, ast.FunctionDef)} for cls in classes]
    for name in ("_matrix", "step", "update", "_complete", "reward", "reset", "close", "_attached"):
        assert methods[0][name] == methods[1][name]
    functions = [{node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)} for tree in trees]
    for name in ("configure_sensors", "clock", "_restore"): assert functions[0][name] == functions[1][name]


def test_Ccollector_version_and_A_B_conflict_guards_and_audit_sources():
    source = (ROOT/"scripts/ppo_ground_contact_v2c.py").read_text()
    assert "import ground_contact_reward_v2c as definition" in source
    assert 'MARKER = "_tron1_ground_contact_v2c"' in source
    assert 'not hasattr(env, "_tron1_ground_contact_v2b")' in source
    assert 'not hasattr(env, "_tron1_ground_contact_v2")' in source
    assert '"immutable_v2a_definition_sha256": definition.BASE_DEFINITION_SHA256' in source
    assert '"immutable_v2b_definition_sha256": definition.BASE_B_DEFINITION_SHA256' in source
    assert '"reward_definition_version": "ground_contact_v2c_any_wheel"' in source
