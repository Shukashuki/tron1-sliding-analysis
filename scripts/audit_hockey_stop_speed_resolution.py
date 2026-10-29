"""Replay a proposed dense XYZ stop score on unchanged qualified recordings.

This compares numerical feedback on recorded states. It neither predicts the
learned policy nor establishes why a rollout failed. No physics is executed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_reward_signals import declared_scores, frozen_refinement_code
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import load, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    read=lambda path:json.loads(path.read_text())
    sources=read(args.cases_json);rows=[];arrays={}
    for number,source in enumerate(sources):
        bundle=(args.cases_json.parent/source['bundle']).resolve()
        evidence=(args.cases_json.parent/source['evidence']).resolve()
        cell,stage=source['case'].split(':')
        if stage!='nominal':raise ValueError('Qualified nominal recordings required')
        entry=next(c for c in read(bundle/'plan.json')['cells'] if c['id']==cell)
        protocol=bundle/entry['protocol'];cfg=read(protocol)
        run_path=evidence/source['run_name']/cell/stage
        qualifier=run_path.parent/'nominal_qualification.json';checked=read(qualifier)
        run=read(run_path/'report.json')
        if (checked['protocol_sha256']!=sha(protocol)
            or checked['source_manifest_sha256']!=sha(bundle/'manifest.json')
            or checked['run_report_sha256']!=sha(run_path/'report.json')):
            raise ValueError('Qualification identity mismatch')
        for item in read(bundle/'manifest.json')['files']:
            if sha(bundle/item['path'])!=item['sha256']:raise ValueError('Snapshot changed')
        for name,digest in run['artifacts'].items():
            if sha(run_path/name)!=digest:raise ValueError('Raw recording changed')
        trace=load(run_path/'final_rollout.npz');raw=load(run_path/'native_contact_buffers.npz')
        velocity=trace['root_com_vel'];dt=float(trace['physics_dt_s'])
        if velocity.shape!=(1113,3) or not np.isclose(dt,.0025,rtol=0,atol=1e-12):
            raise ValueError('Full 400 Hz recording required')
        contact=contact_kinematics(previous_state_trace(trace),raw,cfg['bilateral_sliding']['contact_tolerance_m'])
        touching=contact['touching'][1:].reshape(-1,8,2)
        _,_,gap=wheel_geometry(trace,'body_pos','body_quat',cfg['bilateral_sliding']['geometry'])
        q=trace['root_quat'];up=1-2*(q[:,1]**2+q[:,2]**2)
        angular=np.sum((np.diff(trace['root_omega'],axis=0)/dt)**2,axis=-1).reshape(-1,8)
        phase=np.arange(1,len(angular)+1)>=round(cfg['brake_end_s']/.02)
        inputs=(touching,gap[1:].reshape(-1,8,2),velocity,up,phase,angular,np.zeros(len(angular)))
        # Both forms use the same states, phase, contact rule, tilt and scale.
        # They are counterfactual on older rows; do not label them logged reward.
        exponential=copy.deepcopy(cfg)
        exponential['reward_refinement']=dict(kind='hockey_contact_refinement_v1',
            stop_term='bilateral_substeps_xyz',angular_term='pseudo_huber')
        dense=copy.deepcopy(exponential)
        dense['reward_refinement']['stop_term']='bilateral_substeps_xyz_huber'
        exp_score=declared_scores(exponential,*inputs)['stop']
        dense_score=declared_scores(dense,*inputs)['stop']
        if cfg.get('reward_refinement',{}).get('stop_term')=='bilateral_substeps_xyz':
            frozen=declared_scores(cfg,*inputs,reward_code=frozen_refinement_code(bundle))['stop']
            np.testing.assert_array_equal(exp_score,frozen)
        horizon=checked['qualification']['primary_contact_gate']['window_s'][1]
        time=np.arange(len(velocity))*dt
        terminal=(time>=horizon-.5-1e-9)&(time<=horizon+1e-9)
        ends=np.arange(1,len(angular)+1)*.02
        windows=(ends-.02>=horizon-.5-1e-9)&(ends<=horizon+1e-9)
        speed=np.linalg.norm(velocity,axis=-1)
        speed_score=np.exp(-(speed/cfg['reward_scales']['stop_speed_m_s'])**2)
        rows.append(dict(cell=cell,qualification_sha256=sha(qualifier),raw_sha256=run['artifacts'],
            source_manifest_sha256=sha(bundle/'manifest.json'),
            terminal_state_samples=int(terminal.sum()),terminal_control_windows=int(windows.sum()),
            speed_xyz_range_m_s=[float(speed[terminal].min()),float(speed[terminal].max())],
            speed_only_exponential_below_1e_minus_6_fraction=float((speed_score[terminal]<1e-6).mean()),
            speed_only_exponential_median=float(np.median(speed_score[terminal])),
            counterfactual_exponential_stop_integral=float(exp_score[windows].sum()),
            counterfactual_dense_stop_integral=float(dense_score[windows].sum()),
            task_success=checked['qualification']['task_success_claimed']))
        for key,value in dict(time_s=time,speed_xyz_m_s=speed,speed_only_exponential=speed_score,
                              control_end_s=ends,phase=phase,terminal_windows=windows,
                              counterfactual_exponential_stop=exp_score,counterfactual_dense_stop=dense_score).items():
            arrays[f'case{number}_{key}']=value
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'signals.npz',**arrays)
    report=dict(kind='hockey_stop_speed_resolution_audit_v1',script_sha256=sha(__file__),
        cases_selection_sha256=sha(args.cases_json),
        candidate_code_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('hockey_contact_refinement.py','hockey_contact_refinement_math.py','audit_hockey_contact_reward_signals.py')},
        physics_steps=0,training_changed=False,criterion_changed=False,logged_training_reward=False,
        learned_result_predicted=False,causal_failure_mechanism_proven=False,
        dense_cost_contact_gated=False,absolute_height_target=False,
        scope='Last 0.5 s of each frozen task; full substep windows and separately reported state samples',cases=rows,
        interpretation='A near-zero exponential has little numerical speed discrimination here. This is not a dynamics gradient or proof that changing the reward will improve PPO.')
    report['signals_sha256']=sha(args.output/'signals.npz')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
