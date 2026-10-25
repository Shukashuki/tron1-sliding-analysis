"""From-scratch K PPO trained with native eight-substep leg target ramps.

All fifteen reward terms, original plant, DR, network and budget remain K.
The new execution is declared in the policy contract and audited live before
export. Frozen-policy interpolation was a failed transfer, not this cohort.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

PARENT_SHA='65e6d41695219b676200f61d6d6941ca546fb8f1e95fbd6c60ca89347cf1d1e8'
OLD_ID='tron1_ground_contact_v2k_any_wheel_400hz_v1'
COHORT_ID='tron1_ppo_k_leg_ramp_400hz_v1'
EXECUTION=dict(mode='leg_position_linear_8substeps_reset_at_measured_q_v1',physical_applications_per_control=8,
    reset_start='measured native joint position',exact_final_endpoint=True,wheel_effort_interpolation=False,
    leg_kp=500.,leg_kd=10.,policy_dt_s=.02,physics_dt_s=.0025)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def materialize_generated(source):
    pairs=[('        runner.save(str(args.output_dir/"model_final.pt"))\n',
            '        _ramp_pre_export(args, manifest, _cohort_plan)\n        runner.save(str(args.output_dir/"model_final.pt"))\n'),
           ('            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})\n',
            '            expected = _ramp_export(runner.alg.policy, exported, probe, args, manifest, _cohort_plan)\n')]
    for old,new in pairs:
        assert source.count(old)==1,old
        source=source.replace(old,new,1)
    compile(source,'<K_leg_ramp_generated>','exec');return source


def materialize_parent(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    pairs=[('protocol["cohort_id"] == "'+OLD_ID+'"','protocol["cohort_id"] == "'+COHORT_ID+'"'),
        ('    own_ledger = source_ledger(snapshot)\n','    own_ledger = source_ledger(snapshot)\n    own_ledger.update(_ramp_sources)\n'),
        ('        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)\n',
         '        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)\n        contract["leg_command_execution"] = copy.deepcopy(protocol["leg_command_execution"])\n'),
        ('    generated = materialize_source(cohort)\n','    generated = _ramp_materialize(materialize_source(cohort))\n'),
        ('        current["contact_owner"] = owner\n','        current["contact_owner"] = owner\n        _ramp_install(env, owner, current)\n'),
        ('        cohort.before_learn(env, args, manifest, runner, current)\n',
         '        cohort.before_learn(env, args, manifest, runner, current)\n        _ramp_before_learn(env, args, manifest, current)\n'),
        ('        manifest["ground_contact_v2k"]["runtime_final"] = audit\n',
         '        manifest["ground_contact_v2k"]["runtime_final"] = audit\n        _ramp_before_complete(args, manifest, current)\n'),
        ('        write_json(args.output_dir/"ground_contact_v2k_training_audit.json", result)\n',
         '        write_json(args.output_dir/"ground_contact_v2k_training_audit.json", result)\n        _ramp_seal(args, manifest, report, current)\n'),
        ('        "_cohort_plan": plan, "_cohort_before_start": before_start,\n',
         '        "_cohort_plan": plan, "_cohort_before_start": before_start,\n        "_ramp_pre_export": _ramp_live_audit, "_ramp_export": _ramp_export_check,\n')]
    for old,new in pairs:
        assert source.count(old)==1,'Missing/ambiguous ramp hook: '+old[:90]
        source=source.replace(old,new,1)
    compile(source,'<K_leg_ramp_parent>','exec');return source


def run(argv):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--k-snapshot',type=Path,required=True)
    p.add_argument('--authored-native-report',type=Path,required=True)
    own,inherited=p.parse_known_args(argv)
    protocol_path=Path(inherited[inherited.index('--cohort-protocol')+1]).resolve()
    protocol=json.loads(protocol_path.read_text())
    assert protocol['cohort_id']==COHORT_ID and protocol['leg_command_execution']==EXECUTION
    assert sha(own.authored_native_report)=='29499d653e272b74aadb491be2f41c940b5232674ffb7e5f1c3a4dbc333d9130'
    parent=own.k_snapshot.resolve()/'scripts/train_ppo_ground_contact_v2k.py'
    source=materialize_parent(parent.read_text());here=Path(__file__).resolve().parent
    files=[Path(__file__),here/'ppo_batched_leg_ramp.py',here/'policy_export_precision.py',
        here/'verify_ppo_k_leg_ramp.py',here/'train_ppo_k_nonlinear_tire.py',protocol_path,own.authored_native_report]
    ledger={str(f.resolve()):sha(f) for f in files}
    def geometry(env):
        from train_ppo_k_nonlinear_tire import native_geometry_audit
        result=native_geometry_audit(env,own.authored_native_report)
        result.pop('model_id');result['geometry_model_id']='original_authored_rigid_cylinders'
        return result
    def install(env,owner,current):
        from ppo_batched_leg_ramp import BatchedLegRamp,MODE
        assert MODE==EXECUTION['mode']
        current['leg_ramp']=BatchedLegRamp(env,owner)
    def before_learn(env,args,manifest,current):
        runtime=current['leg_ramp'].audit()
        assert runtime['physical_applications']==current['contact_owner'].total_samples==64
        payload=dict(cohort_id=COHORT_ID,settings=copy.deepcopy(EXECUTION),startup_runtime=runtime,
            startup_geometry=geometry(env),materialized_parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
            source_sha256=ledger,from_scratch=True,all15_rewards_and_original_DR_unchanged=True)
        manifest['leg_command_training']=payload
        manifest['cohort']['leg_command_execution']=copy.deepcopy(EXECUTION)
        write(args.output_dir/'leg_ramp_startup_audit.json',payload)
    def before_complete(args,manifest,current):
        ramp=current['leg_ramp'];payload=manifest['leg_command_training']
        payload['final_runtime']=ramp.audit();payload['final_geometry']=geometry(ramp.env)
        assert all(sha(p)==h for p,h in ledger.items())
    def pre_export(args,manifest,current):
        value=dict(status='live_native_health_verified_before_export',leg_ramp=current['leg_ramp'].audit(),
            contact_owner=current['contact_owner'].audit(health=True))
        write(args.output_dir/'pre_export_runtime_audit.json',value)
    def export_check(policy,exported,probe,args,manifest,current):
        from policy_export_precision import verify_export
        expected,precision=verify_export(policy,exported,probe)
        manifest['leg_command_training']['export_precision']=precision
        write(args.output_dir/'export_precision_audit.json',precision)
        return expected
    def seal(args,manifest,report,current):
        smoke=current['run_kind']=='smoke'
        audit=dict(kind='K_PPO_trained_with_batched_leg_ramp',status='completed_verified',cohort_id=COHORT_ID,
            protocol_sha256=sha(protocol_path),training_environment_steps=report['training_environment_steps'],
            smoke_only=smoke,trained_policy_claimed=not smoke,execution_audit=manifest['leg_command_training'],
            initial_policy_state_sha256=manifest['initial_policy_state_sha256'],task_success_claimed=False)
        for key,name in [('K_training_audit','ground_contact_v2k_training_audit.json'),
                ('startup_audit','leg_ramp_startup_audit.json'),('manifest','manifest.json'),
                ('policy_contract','policy_contract.json'),('actor','policy.pt'),('checkpoint','model_final.pt'),
                ('pre_export_runtime_audit','pre_export_runtime_audit.json'),('export_precision_audit','export_precision_audit.json')]:
            audit[key+'_sha256']=sha(args.output_dir/name)
        write(args.output_dir/'leg_ramp_training_audit.json',audit)
    namespace=dict(__name__='_K_leg_ramp_parent',__file__=str(parent),_ramp_sources=ledger,
        _ramp_materialize=materialize_generated,_ramp_install=install,_ramp_before_learn=before_learn,
        _ramp_before_complete=before_complete,_ramp_seal=seal,_ramp_live_audit=pre_export,_ramp_export_check=export_check)
    previous=list(sys.path),sys.argv
    try:
        sys.path.append(str(parent.parent))
        exec(compile(source,'<K_leg_ramp_parent>','exec'),namespace);namespace['run'](inherited)
    finally:sys.path[:],sys.argv=previous


if __name__=='__main__':run(sys.argv[1:])
