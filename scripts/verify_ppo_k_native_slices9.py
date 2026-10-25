"""Strict completed training verification for the separate native-slices9 cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import ppo_native_slices9_model as model
import train_ppo_k_native_slices9 as training

EVALUATOR_SHA="cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def parent_evaluator(k_snapshot):
    path=Path(k_snapshot).resolve()/"scripts/eval_ppo_ground_contact_v2k.py"
    assert sha(path)==EVALUATOR_SHA
    source=path.read_text();anchor='COHORT_ID = "'+training.OLD_ID+'"'
    assert source.count(anchor)==1
    source=source.replace(anchor,'COHORT_ID = "'+model.COHORT_ID+'"',1)
    namespace=dict(__name__="_K_reward_slices9_cohort_evaluator",__file__=str(path))
    sys.path.append(str(path.parent))
    exec(compile(source,"<K_reward_slices9_cohort_evaluator>","exec"),namespace)
    return namespace


def verify_training(directory,protocol_path,k_snapshot,*,allow_smoke=False):
    directory,protocol_path=Path(directory).resolve(),Path(protocol_path).resolve()
    parent=parent_evaluator(k_snapshot)
    cohort=parent["verify_training"](directory,protocol_path,allow_smoke=allow_smoke)
    protocol=read(protocol_path);manifest=cohort["manifest"]
    assert protocol["cohort_id"]==model.COHORT_ID and protocol["wheel_contact_model"]["model_id"]==model.MODEL_ID
    p=directory/"slices9_model_training_audit.json";startup_path=directory/"slices9_model_startup_audit.json"
    assert p.is_file() and startup_path.is_file() and not p.is_symlink() and not startup_path.is_symlink()
    audit=read(p);startup=read(startup_path);payload=manifest["wheel_contact_model_training"]
    assert audit["kind"]=="model_specific_K_slices9_PPO_training" and audit["status"]=="completed_verified"
    assert audit["cohort_id"]==payload["cohort_id"]==model.COHORT_ID and audit["model_id"]==model.MODEL_ID
    assert audit["protocol_sha256"]==sha(protocol_path)
    assert audit["model_audit"]==payload and payload["settings"]==protocol["wheel_contact_model"]
    assert cohort["contract"]["wheel_contact_model"]==protocol["wheel_contact_model"]
    assert manifest["cohort"]["wheel_contact_model"]==protocol["wheel_contact_model"]
    assert all(payload[k]==v for k,v in startup.items())
    assert payload["from_scratch"] and payload["other14_rewards_and_K_arithmetic_unchanged"]
    assert audit["K_training_audit_sha256"]==sha(directory/"ground_contact_v2k_training_audit.json")
    assert audit["model_startup_audit_sha256"]==sha(startup_path)
    assert audit["manifest_sha256"]==sha(directory/"manifest.json")
    assert audit["policy_contract_sha256"]==sha(directory/"policy_contract.json")
    assert audit["actor_sha256"]==cohort["input_sha256"]["policy.pt"]
    assert audit["checkpoint_sha256"]==cohort["input_sha256"]["model_final.pt"]
    assert audit["smoke_only"]==cohort["smoke_only"] and audit["trained_policy_claimed"]==(not cohort["smoke_only"])
    assert audit["training_environment_steps"]==cohort["training_audit"]["training_environment_steps"]
    assert audit["initial_policy_state_sha256"]==training.EXPECTED_INITIAL_POLICY_SHA256
    parent_path=Path(k_snapshot).resolve()/"scripts/train_ppo_ground_contact_v2k.py"
    generated=training.materialize_parent(parent_path.read_text())
    assert hashlib.sha256(generated.encode()).hexdigest()==payload["materialized_parent_sha256"]
    assert all(sha(path)==digest for path,digest in payload["source_sha256"].items())
    n=manifest["num_envs"]
    for key in ("startup_native_audit","final_native_audit"):
        d=payload[key]
        assert d["status"]=="all_environments_native_model_verified" and d["model_id"]==model.MODEL_ID
        assert d["num_envs"]==n and d["native_wheel_bodies_checked"]==2*n
        assert d["per_environment_wheel_shape_counts"]==[[9,9]]*n
        assert d["total_shapes_per_robot"]==25 and d["geometry_applied_before_clone"] and d["prototype_spawn_calls"]==1
        assert d["contact_offset_m"]==.001 and d["rest_offset_m"]==0.
        assert d["all_body_masses_inertias_armatures_dampings_match_authored"]
        assert d["nonbase_COM_and_all_principal_frames_match_authored"] and d["base_COM_uses_original_DR_only"]
        assert d["authored_reference_report_sha256"]==model.BASELINE_REPORT_SHA
        assert d["extra_simulation_steps_or_rewards"] is False and d["original_asset_file_modified"] is False
        assert all(sha(path)==digest for path,digest in d["SDK_spawn_source_sha256"].items())
    cohort["model_training_audit"]=audit;cohort["model_training_audit_sha256"]=sha(p)
    return cohort


def run(a):
    cohort=verify_training(a.training_dir,a.protocol,a.k_snapshot,allow_smoke=a.allow_smoke)
    shape=None
    if a.cpu_inference:
        import torch
        actor=torch.jit.load(str(cohort["actor"]),map_location="cpu")
        with torch.inference_mode():value=actor(torch.zeros((1,51)))
        assert list(value.shape)==[1,8] and torch.isfinite(value).all()
        shape=list(value.shape)
    audit=cohort["model_training_audit"]
    proof=dict(kind="strict_completed_native_slices9_training_verification",cohort_id=model.COHORT_ID,
        model_id=model.MODEL_ID,formal_or_smoke="smoke" if cohort["smoke_only"] else "formal",
        transitions=audit["training_environment_steps"],actor_sha256=audit["actor_sha256"],checkpoint_sha256=audit["checkpoint_sha256"],
        initial_policy_state_sha256=audit["initial_policy_state_sha256"],
        training_audit_sha256=cohort["model_training_audit_sha256"],native_environment_count=cohort["manifest"]["num_envs"],
        startup_and_final_all_environment_native_model_verified=True,reward_and_DR_parent_K_verified=True,
        CPU_actor_output_shape=shape,verifier_sha256=sha(__file__),task_success_claimed=False)
    if a.output is not None:
        with a.output.open("x") as f:json.dump(proof,f,indent=2,allow_nan=False)
    print(json.dumps(proof))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("training-dir","protocol","k-snapshot"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--allow-smoke",action="store_true");p.add_argument("--cpu-inference",action="store_true")
    p.add_argument("--output",type=Path);run(p.parse_args())
