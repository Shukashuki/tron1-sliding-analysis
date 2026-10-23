"""E changes only D's deficit weight, not task/DR/clock/acceptance."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import ground_contact_reward_v2e as math


def test_only_identity_and_deficit_weight_change_from_D():
    d=json.loads((ROOT/"config/ppo_ground_contact_v2d_400hz_v1.json").read_text());e=json.loads((ROOT/"config/ppo_ground_contact_v2e_400hz_v1.json").read_text())
    assert e["cohort_id"]=="tron1_ground_contact_v2e_any_wheel_400hz_v1"
    assert e["reward_definition_version"]=="ground_contact_v2e_any_wheel"
    assert e["ground_contact_reward_v2"]==math.validate_settings()
    e["cohort_id"],e["reward_definition_version"]=d["cohort_id"],d["reward_definition_version"]
    e["ground_contact_reward_v2"].update(candidate="ground_contact_v2d",support_deficit_weight=1)
    assert e==d


def test_exact_D_native_force_capture_reward_clock_and_reset_lifecycle():
    trees=[ast.parse((ROOT/"scripts"/n).read_text()) for n in ("ppo_ground_contact_v2d.py","ppo_ground_contact_v2e.py")]
    methods=[{f.name:ast.dump(f,include_attributes=False) for cls in t.body if isinstance(cls,ast.ClassDef) for f in cls.body if isinstance(f,ast.FunctionDef)} for t in trees]
    for n in ("_matrix","step","update","_complete","reward","reset","close","_attached"):assert methods[0][n]==methods[1][n]
    funcs=[{f.name:ast.dump(f,include_attributes=False) for f in t.body if isinstance(f,ast.FunctionDef)} for t in trees]
    for n in ("configure_sensors","clock","_restore"):assert funcs[0][n]==funcs[1][n]


def test_E_identity_all_ancestor_audit_guards_and_no_results():
    src=(ROOT/"scripts/ppo_ground_contact_v2e.py").read_text()
    assert "ground_contact_reward_v2e as definition" in src and "GroundContactV2E" in src
    for s in ("v2","v2b","v2c","v2d"):assert 'not hasattr(env, "_tron1_ground_contact_'+s+'")' in src
    for s in ("v2a","v2b","v2c","v2d"):assert "immutable_"+s+"_definition_sha256" in src
