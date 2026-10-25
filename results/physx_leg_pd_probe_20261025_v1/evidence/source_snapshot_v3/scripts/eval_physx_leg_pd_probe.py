"""Source-bound frozen K transfer to declared explicit leg-PD gains."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import physx_leg_pd_probe as probe

PARENT_SHA='32b690dac706929b4df582a1b758fb0cbebe242f9ca8c5646f6c064d2a2eec62'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_motor_recording(measured,actual):
    names=actual['joint_names'].tolist()
    indices=[names.index(n) for n in measured['leg_joint_names'].tolist()]
    assert np.asarray(actual['applied_joint_torque_nm']).shape==(2793,1,8)
    np.testing.assert_array_equal(measured['applied_effort_nm'],actual['applied_joint_torque_nm'][1:,0,indices])
    np.testing.assert_array_equal(measured['joint_pos_rad'],actual['joint_position_rad'][:-1,0,indices])
    np.testing.assert_array_equal(measured['joint_vel_rad_s'],actual['joint_velocity_rad_s'][:-1,0,indices])


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    for old,new in [
        ('        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))\n',
         '        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))\n        _gain_install(env,current,original)\n'),
        ('        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n',
         '        _gain_finish(metadata,report,contract,current)\n        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n')]:
        assert source.count(old)==1,'Missing/ambiguous gain diagnostic hook'
        source=source.replace(old,new,1)
    compile(source,'<K_explicit_leg_PD_probe>','exec')
    return source


def main(argv=None):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--contact-model-evaluator',type=Path,required=True)
    p.add_argument('--gain-profile',choices=probe.PROFILES,required=True)
    p.add_argument('--gain-protocol',type=Path,required=True)
    own,remaining=p.parse_known_args(argv)
    protocol=json.loads(own.gain_protocol.read_text())
    assert protocol['kind']=='frozen_K_PhysX_explicit_leg_PD_diagnostic'
    assert protocol['profiles']=={k:list(v) for k,v in probe.PROFILES.items()}
    assert protocol['mu']==.7 and protocol['physics_dt_s']==.0025 and protocol['policy_dt_s']==.02
    assert sha(own.contact_model_evaluator)==PARENT_SHA
    for flag,value in (('--geometry','authored'),('--law','native_coulomb'),('--contact-offset-m','.001'),('--rest-offset-m','0')):
        assert remaining.count(flag)==1 and remaining[remaining.index(flag)+1]==value
    sources={str(Path(v).resolve()):sha(v) for v in (__file__,probe.__file__,own.gain_protocol,own.contact_model_evaluator)}
    owners=[]

    def install(env,current,material_audit):
        assert not owners
        owner=probe.LegPDProbe(env,own.gain_profile);owners.append(owner)
        before=copy.deepcopy(material_audit['leg_pd_gain_readback'])
        assert before==dict(kp_range=[500.,500.],kd_range=[10.,10.])
        material_audit['leg_pd_gain_readback_before_intervention']=before
        material_audit['planned_policy_step_leg_pd_override']=dict(kp_range=[owner.kp]*2,kd_range=[owner.kd]*2)
        contract=current['contract']
        assert contract['leg_kp']==500. and contract['leg_kd']==10.
        contract.update(policy_training_leg_kp=500.,policy_training_leg_kd=10.,
            leg_kp=owner.kp,leg_kd=owner.kd,
            evaluation_explicit_leg_PD_override=dict(profile=own.gain_profile,training=[500.,10.],execution=[owner.kp,owner.kd],inference_only=True,activation='first_actual_policy_physics_action'))
        Path(current['contract_path']).write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')

    def finish(metadata,report,contract,current):
        assert len(owners)==1 and all(sha(p)==s for p,s in sources.items())
        output=current['output'];owner=owners[0]
        status=owner.save(output/'leg_pd_trajectory.npz')
        assert metadata['normalized_actor_sha256']==protocol['actor_sha256']
        assert (contract['leg_kp'],contract['leg_kd'])==probe.PROFILES[own.gain_profile]
        with np.load(output/'leg_pd_trajectory.npz',allow_pickle=False) as measured,np.load(output/'physics_trajectory.npz',allow_pickle=False) as actual:
            verify_motor_recording(measured,actual)
        if own.gain_profile=='baseline_500_10':
            assert metadata['physics_trajectory_sha256']==protocol['baseline_physics_trajectory_sha256']
        metadata.update(kind='frozen_K_PhysX_explicit_leg_PD_transfer',
            explicit_leg_PD_intervention=status,leg_pd_trajectory_sha256=sha(output/'leg_pd_trajectory.npz'),
            additional_training_updates=0,gain_protocol_sha256=sha(own.gain_protocol))
        metadata['source_sha256'].update(sources)
        report['explicit_leg_PD_intervention']=copy.deepcopy(status)
        report['contact_material_audit']['leg_pd_gain_readback']=owner.gain_readback()
        contract['explicit_leg_PD_intervention']=copy.deepcopy(status)
        (output/'leg_pd_status.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')

    namespace=dict(__name__='_frozen_K_PD_transfer',__file__=str(own.contact_model_evaluator.resolve()),_gain_install=install,_gain_finish=finish)
    old_argv,old_path=sys.argv,list(sys.path)
    try:
        sys.path.append(str(own.contact_model_evaluator.resolve().parent))
        sys.argv=[str(own.contact_model_evaluator),*remaining]
        exec(compile(materialize(own.contact_model_evaluator.read_text()),'<K_explicit_leg_PD_probe>','exec'),namespace)
        namespace['main']()
    finally:
        sys.argv=old_argv;sys.path[:]=old_path


if __name__=='__main__':main(sys.argv[1:])
