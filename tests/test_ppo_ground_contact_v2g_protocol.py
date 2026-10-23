"""Only G identity/design and the positive-contact beta gate differ from F."""
import ast
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2g as definition
import continue_ground_contact_v2g_validation as continuation

def test_protocol_preserves_f_all_clocks_dr_acceptance_and_other_rewards():
    f = json.loads((ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json").read_text())
    g = json.loads((ROOT/"config/ppo_ground_contact_v2g_400hz_v1.json").read_text())
    assert g["ground_contact_reward_v2"] == definition.validate_settings()
    assert g["cohort_id"] == continuation.COHORT_ID == "tron1_ground_contact_v2g_any_wheel_400hz_v1"
    assert g["reward_definition_version"] == "ground_contact_v2g_any_wheel"
    assert g["experiment_design"]["existing_beta_reward_weight"] == 2.
    assert g["experiment_design"]["beta_quality_std_rad"] == .12
    value = copy.deepcopy(g)
    for key in ("cohort_id", "reward_definition_version", "experiment_design"):
        value[key] = f[key]
    value["ground_contact_reward_v2"].pop("beta_gate_low_reference_speed_m_s")
    value["ground_contact_reward_v2"]["candidate"] = f["ground_contact_reward_v2"]["candidate"]
    assert value == f

def test_native_force_capture_clock_and_reset_do_not_change():
    trees = [ast.parse((ROOT/"scripts"/name).read_text()) for name in
             ("ppo_ground_contact_v2f.py", "ppo_ground_contact_v2g.py")]
    methods = [{n.name: ast.dump(n, include_attributes=False)
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for n in cls.body if isinstance(n, ast.FunctionDef)} for tree in trees]
    for name in ("_matrix", "update", "_complete", "reset", "close", "_attached"):
        assert methods[0][name] == methods[1][name], name
    funcs = [{n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
             for tree in trees]
    for name in ("configure_sensors", "clock", "_restore"): assert funcs[0][name] == funcs[1][name]

def test_continuation_pins_actual_g_and_immutable_f_sources():
    from ppo_backend_400hz_contract import sha256
    assert continuation.PROTOCOL_SHA256 == sha256(ROOT/"config/ppo_ground_contact_v2g_400hz_v1.json")
    assert {"ground_contact_reward_v2f.py", "ground_contact_reward_v2g.py", "ground_contact_beta_contract.py"} <= set(continuation.DEPENDENCIES)
    for name, digest in continuation.DEPENDENCIES.items(): assert sha256(ROOT/"scripts"/name) == digest

