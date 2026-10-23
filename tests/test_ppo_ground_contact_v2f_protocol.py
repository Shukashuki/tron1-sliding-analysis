"""F branches from A: only the any-wheel gap cost changes task arithmetic."""
import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ground_contact_reward_v2f as definition
import continue_ground_contact_v2f_validation as continuation


def test_protocol_diff_from_A_is_only_identity_design_and_new_gap_settings():
    a = json.loads((ROOT/"config/ppo_ground_contact_v2_400hz_v1.json").read_text())
    f = json.loads((ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json").read_text())
    assert f["ground_contact_reward_v2"] == definition.validate_settings()
    assert f["cohort_id"] == "tron1_ground_contact_v2f_any_wheel_400hz_v1"
    assert f["reward_definition_version"] == "ground_contact_v2f_any_wheel"
    assert f["experiment_design"]["branch"] == "A, not E"
    comparable = copy.deepcopy(f)
    comparable.pop("experiment_design")
    comparable["ground_contact_reward_v2"].pop("support_gap_weight")
    comparable["ground_contact_reward_v2"].pop("support_force_threshold_n")
    comparable["ground_contact_reward_v2"]["candidate"] = a["ground_contact_reward_v2"]["candidate"]
    for field in ("cohort_id", "reward_definition_version"):
        comparable[field] = a[field]
    assert comparable == a


def test_native_sensor_capture_and_reward_lifecycle_are_exactly_A():
    trees = [ast.parse((ROOT/"scripts"/name).read_text())
             for name in ("ppo_ground_contact_v2.py", "ppo_ground_contact_v2f.py")]
    methods = [{method.name: ast.dump(method, include_attributes=False)
                for cls in tree.body if isinstance(cls, ast.ClassDef)
                for method in cls.body if isinstance(method, ast.FunctionDef)} for tree in trees]
    for name in ("_matrix", "step", "update", "_complete", "reward", "reset", "close", "_attached"):
        assert methods[0][name] == methods[1][name]
    funcs = [{fn.name: ast.dump(fn, include_attributes=False)
              for fn in tree.body if isinstance(fn, ast.FunctionDef)} for tree in trees]
    for name in ("configure_sensors", "clock", "_restore"):
        assert funcs[0][name] == funcs[1][name]


def test_continuation_pins_the_exact_F_sources_and_unchanged_acceptance():
    from ppo_backend_400hz_contract import sha256
    protocol = ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json"
    assert continuation.PROTOCOL_SHA256 == sha256(protocol)
    for name, digest in continuation.DEPENDENCIES.items():
        assert sha256(ROOT/"scripts"/name) == digest
    value = json.loads(protocol.read_text())["nominal_acceptance_before_training"]
    assert value["any_wheel_support_fraction_min"] == 1.
    assert value["longest_no_support_interval_s_max"] == 0.
    assert value["contact_force_threshold_n_strict"] == definition.DEFAULT_SETTINGS["support_force_threshold_n"]
    assert value["core_reference_beta_rmse_deg_max"] == 10.
    assert value["absolute_height_or_flight_gate"] is False
