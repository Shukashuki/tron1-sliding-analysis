"""L changes only the grounded definition and experiment identity, preserving K clocks and all original gates."""
import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2l as definition
import continue_ground_contact_v2l_validation as continuation


def test_exact_K_protocol_except_identity_design_and_force_slew_settings():
    old = json.loads((ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json").read_text())
    new = json.loads((ROOT/"config/ppo_ground_contact_v2l_400hz_v1.json").read_text())
    assert new["ground_contact_reward_v2"] == definition.validate_settings()
    assert new["cohort_id"] == continuation.COHORT_ID == "tron1_ground_contact_v2l_any_wheel_400hz_v1"
    assert new["reward_definition_version"] == "ground_contact_v2l_any_wheel"
    assert new["experiment_design"]["existing_beta_reward_weight"] == 2.
    assert new["experiment_design"]["beta_quality_std_rad"] == .12
    compare = copy.deepcopy(new)
    for key in ("cohort_id", "reward_definition_version", "experiment_design"):
        compare[key] = old[key]
    for key in ("normal_force_slew_weight", "normal_force_slew_mg_scale", "normal_force_slew_cap"):
        compare["ground_contact_reward_v2"].pop(key)
    compare["ground_contact_reward_v2"]["candidate"] = old["ground_contact_reward_v2"]["candidate"]
    assert compare == old


def test_capture_lifecycle_held_phase_and_existing_force_inputs_identical_to_K():
    trees = [ast.parse((ROOT/"scripts"/name).read_text()) for name in
             ("ppo_ground_contact_v2k.py", "ppo_ground_contact_v2l.py")]
    methods = [{n.name: ast.dump(n, include_attributes=False)
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for n in cls.body if isinstance(n, ast.FunctionDef)} for tree in trees]
    for name in ("_matrix", "update", "_complete", "reset", "close", "_attached", "step", "reward", "_phase"):
        assert methods[0][name] == methods[1][name], name
    funcs = [{n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
             for tree in trees]
    for name in ("configure_sensors", "clock", "_restore"):
        assert funcs[0][name] == funcs[1][name]


def test_continuation_pins_actual_L_and_immutable_K_sources():
    from ppo_backend_400hz_contract import sha256
    assert continuation.PROTOCOL_SHA256 == sha256(ROOT/"config/ppo_ground_contact_v2l_400hz_v1.json")
    assert {"ground_contact_reward_v2i.py", "ground_contact_reward_v2k.py", "ground_contact_reward_v2l.py",
            "ppo_ground_contact_v2l.py", "train_ppo_ground_contact_v2l.py",
            "ground_contact_beta_contract.py"} <= set(continuation.DEPENDENCIES)
    assert len(continuation.DEPENDENCIES) == 18
    for name, digest in continuation.DEPENDENCIES.items():
        assert sha256(ROOT/"scripts"/name) == digest

