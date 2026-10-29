"""Qualify all 33 warm-started contact-screen perturbations independently.

Uses the unchanged evaluator from the immutable experiment snapshot. Actor
replay, recorded initial states, materials and every physical sample are
checked before counting success. Does not train or step any simulator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:raise ValueError(message)


def check_case_list(expected, recorded):
    ids=[case['id'] for case in expected]
    require(len(ids)==33 and len(set(ids))==33,'Require all 33 distinct declared cases')
    require(all(re.fullmatch(r'[a-zA-Z0-9_]+',name) for name in ids),'Unsafe case identifier')
    require(expected[0]['kind']=='nominal' and expected[0]['id']=='nominal','Nominal must be first')
    require(all(case['kind']=='perturbed' for case in expected[1:]),'Require 32 perturbations')
    require([case['id'] for case in recorded]==ids,'Missing, reordered or duplicate suite cases')


def check_reset(trace, reset, case, nominal_pose, nominal_velocity, nominal_joints, perturb):
    """Verify the physical recording, not just a self-reported reset error."""
    require(reset['case']==case,'Reset case differs from the declared trial')
    error=reset['maximum_initial_state_error']
    require(np.isfinite(error) and 0<=error<=5e-5,'Reset readback failed')
    require(reset['commands_transformed'] is False and reset['height_and_joint_pose_changed'] is False,
            'Reference or height/joint pose changed with the perturbation')
    expected_pose,expected_velocity=perturb(nominal_pose,nominal_velocity,case)
    pose=np.r_[trace['root_pos'][0]+trace['environment_origin_w'],trace['root_quat'][0]]
    velocity=np.r_[trace['root_com_vel'][0],trace['root_omega'][0]]
    for observed,expected in ((pose,expected_pose),(velocity,expected_velocity),
        (reset['initial_root_pose_w'],pose),(reset['initial_COM_velocity_w'],velocity),
        (trace['joint_pos'][0],nominal_joints)):
        observed=np.asarray(observed);expected=np.asarray(expected)
        require(observed.shape==expected.shape and np.isfinite(observed).all()
                and np.allclose(observed,expected,rtol=0,atol=5e-5),'Recorded initial state differs from perturbation')
    return dict(maximum_pose_error=float(np.max(abs(pose-expected_pose))),
        maximum_velocity_error=float(np.max(abs(velocity-expected_velocity))),
        maximum_joint_error=float(np.max(abs(trace['joint_pos'][0]-nominal_joints))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','protocol','training-dir','suite','actor','reference','screen-qualification','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),'Fresh qualification output required')
    read=lambda path:json.loads(path.read_text())
    manifest=read(a.bundle/'manifest.json')
    for row in manifest['files']:
        file=(a.bundle/row['path']).resolve()
        require(file.is_relative_to(a.bundle.resolve()) and sha(file)==row['sha256'],'Frozen snapshot changed')
    cfg=read(a.protocol);training=read(a.training_dir/'report.json');suite=read(a.suite/'report.json')
    screen=read(a.screen_qualification)
    require(a.protocol.resolve().is_relative_to(a.bundle.resolve()),'Protocol must belong to frozen snapshot')
    require(cfg['kind']=='hockey_contact_reward_search_v1','Require the warm-start contact study')
    require((cfg['physics_dt_s'],cfg['decimation'])==(.0025,8),'Physical clock changed')
    budget=cfg['formal']['num_envs']*cfg['formal']['iterations']*cfg['ppo']['steps_per_env']
    require(training['status']=='training_completed_not_task_success' and training['transitions']==budget,
            'Declared fine-tuning budget is incomplete')
    require(training['warm_start']['checkpoint_sha256']==cfg['warm_start']['checkpoint_sha256'],'Parent mismatch')
    require(suite['status']=='validation_suite_completed_requires_independent_qualification','Suite incomplete')
    require(screen['qualification']['task_success_claimed'] and all(screen['qualification']['gates'].values()),
            'Nominal screen did not pass before robustness validation')
    require(screen['source_manifest_sha256']==sha(a.bundle/'manifest.json')
            and screen['protocol_sha256']==sha(a.protocol),'Screen source mismatch')
    require(a.screen_qualification.name=='nominal_qualification.json','Require the original screening qualification')
    screen_dir=a.screen_qualification.parent/'nominal';screen_run=read(screen_dir/'report.json')
    require(sha(screen_dir/'report.json')==screen['run_report_sha256'],'Original screen report changed')
    for file,digest in screen_run['artifacts'].items():
        require(sha(screen_dir/file)==digest,'Original nominal recording changed')
    actor_sha=sha(a.actor)
    require(training['actor_sha256']==suite['actor_sha256']==screen['actor_sha256']==actor_sha,'Policy changed')
    expected_hashes=dict(source_sha256=sha(a.bundle/'train_hockey_contact_search.py'),
        task_source_sha256=sha(a.bundle/'hockey_ppo_task.py'),
        bilateral_source_sha256=sha(a.bundle/'hockey_bilateral_task.py'),
        reward_search_source_sha256=sha(a.bundle/'hockey_contact_search.py'),
        config_sha256=sha(a.protocol),motion_sha256=sha(a.reference/'motion.npz'),asset_sha256=cfg['asset_sha256'])
    if cfg.get('reward_refinement') is not None:
        expected_hashes.update(reward_refinement_source_sha256=sha(a.bundle/'hockey_contact_refinement.py'),
                              reward_refinement_math_sha256=sha(a.bundle/'hockey_contact_refinement_math.py'))
        for run in (training,suite):
            require(run.get('reward_refinement')==cfg['reward_refinement'],'Refinement declaration changed')
    for run in (training,suite):
        for key,digest in expected_hashes.items():require(run[key]==digest,'Source identity changed: '+key)
    require(suite['case_math_sha256']==sha(a.bundle/'hockey_speed_study.py'),'Perturbation math changed')
    check_case_list(cfg['evaluation_cases'],suite['cases'])
    sys.path.insert(0,str(a.bundle.resolve()))
    from analyze_hockey_bilateral import qualify
    from analyze_hockey_ppo import audit_observations
    from evaluate_hockey_physx import load
    from hockey_speed_study import perturbed_root,wilson_interval
    for module in ('analyze_hockey_bilateral','analyze_hockey_ppo','evaluate_hockey_physx','hockey_speed_study'):
        require(Path(sys.modules[module].__file__).resolve().parent==a.bundle.resolve(),'Evaluator import escaped frozen snapshot')
    import torch
    torch.set_num_threads(2);actor=torch.jit.load(str(a.actor),map_location='cpu').eval()
    reference=load(a.reference/'motion.npz');command=read(a.reference/'report.json')
    require(sha(a.reference/'motion.npz')==cfg['reference_sha256'],'Reference identity changed')
    # Anchor all perturbations to the already-passed screening state. A suite
    # cannot silently choose a different common reset and self-certify it.
    nominal=load(screen_dir/'final_rollout.npz')
    nominal_pose=np.r_[nominal['root_pos'][0]+nominal['environment_origin_w'],nominal['root_quat'][0]]
    nominal_velocity=np.r_[nominal['root_com_vel'][0],nominal['root_omega'][0]]
    rows=[];a.output.mkdir(parents=True)
    for case,entry in zip(cfg['evaluation_cases'],suite['cases']):
        d=a.suite/case['id'];run=read(d/'report.json')
        require(sha(d/'report.json')==entry['report_sha256'],'Case report changed')
        for key,digest in expected_hashes.items():require(run[key]==digest,'Case source mismatch: '+key)
        require(run['actor_sha256']==actor_sha and run['case_id']==case['id'],'Case/policy identity changed')
        require((run['mode'],run['worlds'],run['steps'],run['external_forces'])==('eval',1,139,0),
                'Require unchanged single-world full-horizon native validation')
        require(set(run['artifacts'])=={'case.json','final_rollout.npz','native_contact_buffers.npz'},'Missing case evidence')
        for file,digest in run['artifacts'].items():require(sha(d/file)==digest,'Case artifact changed')
        model=cfg['contact_model']['geometry'];geometry=run['geometry_audit']
        require(geometry['model']==model and geometry['per_wheel_shapes']==(9 if model=='slices9' else 1)
                and geometry['native_bodies_checked']==2 and geometry['external_forces']==0,'Wrong contact geometry')
        material=run['material_audit'];mu=cfg['evaluation_mu'];mus=mu*cfg['static_dynamic_friction_ratio']
        require(material['live_physx_materials_verified'] and material['ground_material_usd_verified'],'Unverified materials')
        np.testing.assert_allclose(material['wheel_mu_range'],[mu,mu],rtol=0,atol=1e-6)
        np.testing.assert_allclose(material.get('wheel_mu_static_range',material['wheel_mu_range']),[mus,mus],rtol=0,atol=1e-6)
        require(all(g['friction']==1 and g['combine']=='multiply' for g in material['ground_colliders']),'Ground material changed')
        trace=load(d/'final_rollout.npz');raw=load(d/'native_contact_buffers.npz')
        require(trace['root_pos'].shape==(1113,3) and trace['actions'].shape==(139,8),'Incomplete physical recording')
        reset=check_reset(trace,read(d/'case.json'),case,nominal_pose,nominal_velocity,nominal['joint_pos'][0],perturbed_root)
        obs=audit_observations(trace,reference,cfg)
        with torch.inference_mode():predicted=actor(torch.as_tensor(trace['observations'],dtype=torch.float32)).clamp(-1,1).numpy()
        error=float(np.max(abs(predicted-trace['actions'])))
        require(np.isfinite(error) and error<=5e-5,'Actor replay differs')
        q=qualify(trace,raw,reference,command,mu=mu,mu_static=mus)
        result=dict(case=case,qualification=q,observation_audit=obs,action_replay_max_error=error,
            initial_state_audit=reset,run_report_sha256=sha(d/'report.json'),input_sha256=run['artifacts'])
        out=a.output/(case['id']+'.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        rows.append(dict(id=case['id'],kind=case['kind'],success=q['task_success_claimed'],
            failed_gates=q['failed_gates'],contact_gate=q['primary_contact_gate']['gate_status'],
            both_contact_percent=100*q['primary_contact_gate']['both_touch_fraction'],xy_RMSE_m=q['xy_RMSE_m'],
            terminal_stop_hold_s=q['terminal_stop_hold_s'],report_sha256=sha(out)))
        print('QUALIFIED',case['id'],q['task_success_claimed'],flush=True)
    perturbed=rows[1:];successes=sum(row['success'] for row in perturbed)
    report=dict(kind='hockey_contact_screen_robustness_qualification_v1',nominal_success=rows[0]['success'],
        perturbation_trials=len(perturbed),perturbation_successes=successes,
        conditional_Wilson_95_interval=wilson_interval(successes,len(perturbed)),
        all_tested_cases_pass=all(row['success'] for row in rows),overall_goal_achieved=False,
        criterion_changed=False,contact_force_threshold=None,equal_load_target=False,physics_steps=0,
        early_termination_during_validation=False,analyzer_sha256=sha(__file__),actor_sha256=actor_sha,
        source_manifest_sha256=sha(a.bundle/'manifest.json'),protocol_sha256=sha(a.protocol),
        training_report_sha256=sha(a.training_dir/'report.json'),suite_report_sha256=sha(a.suite/'report.json'),
        screen_qualification_sha256=sha(a.screen_qualification),
        nominal_reset_anchor_sha256=screen_run['artifacts']['final_rollout.npz'],cases=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(nominal_success=rows[0]['success'],perturbed_successes=successes,trials=len(perturbed))))


if __name__=='__main__':main()
