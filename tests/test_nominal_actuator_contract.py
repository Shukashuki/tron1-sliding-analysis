"""The published parameter-only input retains its historical byte identity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_actuator_contract_exact_original_bytes():
    raw = (ROOT/"config/nominal_actuator_contract.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "6e105cbd6efe7d35117a895fa097e8ec64b6ac822a9de07c00acfaef4d09642f"
    for name in ("nmpc_sa29_nominal_development.json", "nmpc_sa29_friction_development.json",
                 "nmpc_friction_development.json"):
        assert json.loads((ROOT/"config"/name).read_text())["input_sha256"]["actuator_contract"] == digest
    value = json.loads(raw)
    assert value["physics_dt"] == .005 and value["decimation"] == 4
    assert value["leg_kp"] == 500. and value["leg_kd"] == 10.
    assert value["wheel_torque_scale_nm"] == value["wheel_torque_limit_nm"] == 12.
    assert value["hardware_ready"] is False
    # Legacy export metadata is preserved, not recast as SA29's evaluation mu.
    assert value["task_profile"]["contact_friction"] == .18
    assert value["actor_file"] == "actor_normalized.pt"
    assert not any(token in raw for token in (b"/home/", b"/data/", b"PRIVATE KEY", b"zengxiong@"))
