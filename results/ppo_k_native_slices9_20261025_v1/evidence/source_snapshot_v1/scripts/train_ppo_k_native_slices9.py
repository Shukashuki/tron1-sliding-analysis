"""Matched K-reward training on native nine-slice wheels, from scratch.

Counted in-memory hooks reuse the immutable K trainer and its exact arithmetic.
The new model has its own cohort/protocol and all-environment physical audit.
No installed SDK, parent source, reward weight or historical actor is edited.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import ppo_native_slices9_model as model

PARENT_SHA="65e6d41695219b676200f61d6d6941ca546fb8f1e95fbd6c60ca89347cf1d1e8"
OLD_ID="tron1_ground_contact_v2k_any_wheel_400hz_v1"
EXPECTED_INITIAL_POLICY_SHA256="7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize_parent(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA,"Immutable K trainer differs"
    pairs=[
        ('protocol["cohort_id"] == "'+OLD_ID+'"','protocol["cohort_id"] == "'+model.COHORT_ID+'"'),
        ('    own_ledger = source_ledger(snapshot)\n','    own_ledger = source_ledger(snapshot)\n    own_ledger.update(_model_source_ledger)\n'),
        ('        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)\n',
         '        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)\n        contract["wheel_contact_model"] = copy.deepcopy(protocol["wheel_contact_model"])\n'),
        ('        cfg.rewards.drift_grounded = contact.reward_term(.4)\n',
         '        cfg.rewards.drift_grounded = contact.reward_term(.4)\n        _model_configure(cfg, args, current)\n'),
        ('        cohort.before_learn(env, args, manifest, runner, current)\n',
         '        cohort.before_learn(env, args, manifest, runner, current)\n        _model_before_learn(env, args, manifest, current)\n'),
        ('        manifest["ground_contact_v2k"]["runtime_final"] = audit\n',
         '        manifest["ground_contact_v2k"]["runtime_final"] = audit\n        _model_before_complete(args, manifest, current)\n'),
        ('        write_json(args.output_dir/"ground_contact_v2k_training_audit.json", result)\n',
         '        write_json(args.output_dir/"ground_contact_v2k_training_audit.json", result)\n        _model_seal(args, manifest, report, current)\n')]
    for old,new in pairs:
        assert source.count(old)==1,"Missing/ambiguous model hook: "+old[:80]
        source=source.replace(old,new,1)
    compile(source,"<matched_K_slices9_parent>","exec")
    return source


def run(argv):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument("--k-snapshot",type=Path,required=True)
    p.add_argument("--authored-native-report",type=Path,required=True)
    own,inherited=p.parse_known_args(argv)
    parent=own.k_snapshot.resolve()/"scripts/train_ppo_ground_contact_v2k.py"
    protocol_path=Path(inherited[inherited.index("--cohort-protocol")+1]).resolve()
    protocol=json.loads(protocol_path.read_text())
    assert protocol["cohort_id"]==model.COHORT_ID and protocol["wheel_contact_model"]["model_id"]==model.MODEL_ID
    assert sha(own.authored_native_report)==model.BASELINE_REPORT_SHA
    source=materialize_parent(parent.read_text());generated_sha=hashlib.sha256(source.encode()).hexdigest()
    here=Path(__file__).resolve().parent
    ledger={str(path.resolve()):sha(path) for path in (Path(__file__),Path(model.__file__),
        here/"tron1_wheel_geometry.py",here/"verify_ppo_k_native_slices9.py",protocol_path,own.authored_native_report)}
    def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    def configure(cfg,args,current):
        assert args.backend=="physx" and cfg.scene.num_envs==args.num_envs
        context={};model.configure_spawn(cfg,context);current["slices9_model_context"]=context
    def before_learn(env,args,manifest,current):
        context=current["slices9_model_context"];observed=model.native_audit(env,context,own.authored_native_report)
        current["slices9_env"]=env
        payload=dict(cohort_id=model.COHORT_ID,settings=copy.deepcopy(protocol["wheel_contact_model"]),
            startup_native_audit=observed,materialized_parent_sha256=generated_sha,
            source_sha256=ledger,from_scratch=True,other14_rewards_and_K_arithmetic_unchanged=True)
        manifest["wheel_contact_model_training"]=payload
        manifest["cohort"]["wheel_contact_model"]=copy.deepcopy(protocol["wheel_contact_model"])
        write(args.output_dir/"slices9_model_startup_audit.json",payload)
    def before_complete(args,manifest,current):
        payload=manifest["wheel_contact_model_training"]
        payload["final_native_audit"]=model.native_audit(current["slices9_env"],current["slices9_model_context"],own.authored_native_report)
        assert all(sha(path)==digest for path,digest in ledger.items()),"Model training source/input changed"
    def seal(args,manifest,report,current):
        payload=manifest["wheel_contact_model_training"]
        smoke=current["run_kind"]=="smoke"
        audit=dict(kind="model_specific_K_slices9_PPO_training",status="completed_verified",
            cohort_id=model.COHORT_ID,model_id=model.MODEL_ID,protocol_sha256=sha(protocol_path),
            training_environment_steps=report["training_environment_steps"],smoke_only=smoke,
            trained_policy_claimed=not smoke,model_audit=payload,initial_policy_state_sha256=manifest["initial_policy_state_sha256"],
            K_training_audit_sha256=sha(args.output_dir/"ground_contact_v2k_training_audit.json"),
            model_startup_audit_sha256=sha(args.output_dir/"slices9_model_startup_audit.json"),
            manifest_sha256=sha(args.output_dir/"manifest.json"),policy_contract_sha256=sha(args.output_dir/"policy_contract.json"),
            actor_sha256=sha(args.output_dir/"policy.pt"),checkpoint_sha256=sha(args.output_dir/"model_final.pt"),
            task_success_claimed=False)
        write(args.output_dir/"slices9_model_training_audit.json",audit)
    namespace=dict(__name__="_immutable_K_trainer_model_variant",__file__=str(parent),
        _model_source_ledger=ledger,_model_configure=configure,_model_before_learn=before_learn,
        _model_before_complete=before_complete,_model_seal=seal)
    path,arguments=list(sys.path),sys.argv
    try:
        sys.path.append(str(parent.parent))
        exec(compile(source,"<matched_K_slices9_parent>","exec"),namespace)
        namespace["run"](inherited)
    finally:sys.path[:]=path;sys.argv=arguments


if __name__=="__main__":run(sys.argv[1:])
