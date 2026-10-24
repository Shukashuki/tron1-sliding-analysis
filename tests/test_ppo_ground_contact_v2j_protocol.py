"""J keeps I clocks, acceptance, DR, actions, observations and14 other terms."""
import copy,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ground_contact_reward_v2j as definition
import continue_ground_contact_v2j_validation as continuation


def test_exact_i_protocol_except_identity_design_and_new_contact_component():
    i=json.loads((ROOT/"config/ppo_ground_contact_v2i_400hz_v1.json").read_text())
    j=json.loads((ROOT/"config/ppo_ground_contact_v2j_400hz_v1.json").read_text())
    assert j["ground_contact_reward_v2"]==definition.validate_settings()
    assert j["cohort_id"]==continuation.COHORT_ID=="tron1_ground_contact_v2j_any_wheel_400hz_v1"
    assert j["experiment_design"]["existing_beta_reward_weight"]==2.
    v=copy.deepcopy(j)
    for key in ("cohort_id","reward_definition_version","experiment_design"):v[key]=i[key]
    for key in ("loaded_angular_increment_deadband_rad_s","loaded_angular_increment_scale_rad_s","loaded_angular_increment_cap","loaded_angular_increment_weight"):
        v["ground_contact_reward_v2"].pop(key)
    v["ground_contact_reward_v2"]["candidate"]=i["ground_contact_reward_v2"]["candidate"]
    assert v==i


def test_frozen_dependency_bytes_and_protocol():
    from ppo_backend_400hz_contract import sha256
    assert continuation.PROTOCOL_SHA256==sha256(ROOT/"config/ppo_ground_contact_v2j_400hz_v1.json")
    for name,digest in continuation.DEPENDENCIES.items():assert sha256(ROOT/"scripts"/name)==digest
