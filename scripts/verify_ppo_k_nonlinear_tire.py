"""Strict completed K-reward/nonlinear-tire training and normalized export proof."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import train_ppo_k_nonlinear_tire as training


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_training(directory, protocol_path, k_snapshot, *, allow_smoke=False):
    directory, protocol_path = Path(directory).resolve(), Path(protocol_path).resolve()
    parent = Path(k_snapshot).resolve()/"scripts/eval_ppo_ground_contact_v2k.py"
    assert sha(parent) == "cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c"
    source = parent.read_text(); anchor = 'COHORT_ID = "'+training.OLD_ID+'"'
    assert source.count(anchor) == 1
    source = source.replace(anchor, 'COHORT_ID = "'+training.COHORT_ID+'"')
    namespace = dict(__name__="_K_nonlinear_tire_reward_verifier", __file__=str(parent))
    sys.path.append(str(parent.parent))
    exec(compile(source, "<K_nonlinear_tire_reward_verifier>", "exec"), namespace)
    cohort = namespace["verify_training"](directory, protocol_path, allow_smoke=allow_smoke)
    protocol = read(protocol_path); manifest = cohort["manifest"]
    audit_path = directory/"nonlinear_tire_training_audit.json"
    startup_path = directory/"nonlinear_tire_startup_audit.json"
    for p in (audit_path, startup_path):
        assert p.is_file() and not p.is_symlink()
    audit = read(audit_path); startup = read(startup_path); payload = manifest["nonlinear_tire_training"]
    assert audit["status"] == "completed_verified" and audit["kind"] == "model_specific_K_nonlinear_tire_PPO_training"
    assert audit["cohort_id"] == payload["cohort_id"] == training.COHORT_ID
    assert audit["model_id"] == protocol["wheel_contact_model"]["model_id"] == training.MODEL_ID
    assert audit["protocol_sha256"] == sha(protocol_path)
    assert audit["model_audit"] == payload and payload["settings"] == protocol["wheel_contact_model"]
    assert cohort["contract"]["wheel_contact_model"] == protocol["wheel_contact_model"]
    assert all(payload[k] == value for k, value in startup.items())
    assert payload["from_scratch"] and payload["other14_rewards_and_K_arithmetic_unchanged"]
    for key, path in (("K_training_audit_sha256", directory/"ground_contact_v2k_training_audit.json"),
            ("startup_audit_sha256", startup_path), ("manifest_sha256", directory/"manifest.json"),
            ("policy_contract_sha256", directory/"policy_contract.json"),
            ("actor_sha256", directory/"policy.pt"), ("checkpoint_sha256", directory/"model_final.pt")):
        assert audit[key] == sha(path)
    assert audit["smoke_only"] == cohort["smoke_only"] and audit["trained_policy_claimed"] == (not cohort["smoke_only"])
    assert audit["training_environment_steps"] == cohort["training_audit"]["training_environment_steps"]
    assert all(sha(path) == digest for path, digest in payload["source_sha256"].items())
    parent_source = Path(k_snapshot).resolve()/"scripts/train_ppo_ground_contact_v2k.py"
    assert hashlib.sha256(training.materialize_parent(parent_source.read_text()).encode()).hexdigest() == payload["materialized_parent_sha256"]
    n = manifest["num_envs"]; windows = 56 if cohort["smoke_only"] else 24008
    for stage, expected in (("startup_runtime", 64), ("final_runtime", windows*8)):
        d = payload[stage]
        assert d["num_envs"] == n and d["physical_writes"] == expected
        assert d["environment_physical_writes"] == n*expected
        assert d["force_law"] == "combined_slip_magic_formula" and not d["invalid_measurement"]
        assert d["normal_load_delay_physical_steps"] == 1 and d["physics_dt_s"] == .0025
        assert d["contact_power_max_w"] <= .001 and d["capacity_excess_max_n"] <= .001
        assert d["reset_first_physical_step_has_zero_external_force"]
        assert d["original_step_update_reward_owner_retained"] and d["root_joint_state_or_action_writes"] == 0
        assert d["per_physical_write_host_tensor_transfers"] == 0
        assert min(d["loaded_wheel_samples_L_R"]) > 0 and min(d["nonzero_force_samples_L_R"]) > 0
        assert d["native_material_readback"]["native_wheel_static_dynamic_mu"] == 0.
        assert d["native_material_readback"]["external_mu_per_environment_shared_between_wheels"]
        for path, digest in {**d["source_sha256"], **d["native_wrench_writer_source_sha256"]}.items():
            assert sha(path) == digest
    for stage in ("startup_geometry", "final_geometry"):
        d = payload[stage]
        assert d["physical_wheels_checked"] == 2*n and d["total_shapes_per_robot"] == 9
        assert d["joint_count"] == 8 and d["additional_DOF"] == 0
        assert d["original_mass_inertia_and_nonbase_COM_retained"] and d["base_COM_original_DR_only"]
        assert (d["contact_offset_m"],d["rest_offset_m"]) == (.001, 0.)
    cohort["model_training_audit"] = audit
    cohort["model_training_audit_sha256"] = sha(audit_path)
    return cohort


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("training-dir", "protocol", "k-snapshot"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--cpu-inference", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cohort = verify_training(a.training_dir, a.protocol, a.k_snapshot, allow_smoke=a.allow_smoke)
    shape = None
    if a.cpu_inference:
        import torch
        actor = torch.jit.load(str(cohort["actor"]), map_location="cpu")
        with torch.inference_mode(): value = actor(torch.zeros((1,51)))
        assert list(value.shape) == [1,8] and torch.isfinite(value).all()
        shape = list(value.shape)
    audit = cohort["model_training_audit"]
    proof = dict(kind="strict_completed_K_nonlinear_tire_training_verification",
        cohort_id=training.COHORT_ID, model_id=training.MODEL_ID,
        formal_or_smoke="smoke" if cohort["smoke_only"] else "formal",
        transitions=audit["training_environment_steps"], actor_sha256=audit["actor_sha256"],
        checkpoint_sha256=audit["checkpoint_sha256"],
        initial_policy_state_sha256=audit["initial_policy_state_sha256"],
        training_audit_sha256=cohort["model_training_audit_sha256"],
        native_environment_count=cohort["manifest"]["num_envs"],
        CPU_actor_output_shape=shape, task_success_claimed=False)
    with a.output.open("x") as f: json.dump(proof,f,indent=2,allow_nan=False)
    print(json.dumps(proof))


if __name__ == "__main__": main()
