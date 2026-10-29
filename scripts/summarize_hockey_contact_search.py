"""Summarize completed native-model screens; unfinished cells stay unverified."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from evaluate_hockey_physx import sha


def checked_row(run_root, cell, protocol, label):
    directory=run_root/cell['id']; qpath=directory/(label+'_qualification.json')
    if not qpath.exists():return None
    data=json.loads(qpath.read_text());run_path=directory/label/'report.json'
    run=json.loads(run_path.read_text())
    if data['run_report_sha256']!=sha(run_path) or data['protocol_sha256']!=sha(protocol):
        raise ValueError('Qualification no longer matches its run/protocol')
    if data['actor_sha256']!=run['actor_sha256'] or not data['observation_audit']['all_56_fields_reconstructed']:
        raise ValueError('Missing actor or observation identity')
    q=data['qualification']
    if q['task_success_claimed']!=all(q['gates'].values()):
        raise ValueError('Inconsistent success claim')
    raw_checked=True
    for name,digest in run['artifacts'].items():
        p=directory/label/name
        if p.exists():
            if sha(p)!=digest:raise ValueError('Downloaded raw recording changed')
        else:raw_checked=False
    cfg=json.loads(protocol.read_text())
    if label=='nominal':
        training=json.loads((directory/'formal/report.json').read_text())
        expected=cfg['formal']['num_envs']*cfg['formal']['iterations']*cfg['ppo']['steps_per_env']
        if training['status']!='training_completed_not_task_success' or training['transitions']!=expected:
            raise ValueError('Incomplete training budget')
        if training['actor_sha256']!=run['actor_sha256']:
            raise ValueError('Validation used a different policy')
        if training['warm_start']['checkpoint_sha256']!=cfg['warm_start']['checkpoint_sha256']:
            raise ValueError('Wrong parent initialization')
    gate=q['primary_contact_gate']
    return dict(cell=cell['id'],model=cfg['contact_model']['id'],
        reward='frozen_parent' if label=='parent_transfer' else cell['id'][len(cfg['contact_model']['id'])+1:],
        status='completed_nominal_qualification',mu_static=cfg['evaluation_mu']*cfg['static_dynamic_friction_ratio'],
        mu_dynamic=cfg['evaluation_mu'],new_training_transitions=0 if label=='parent_transfer' else expected,
        task_success=q['task_success_claimed'],contact_gate=gate['gate_status'],
        both_contact_percent=100*gate['both_touch_fraction'],interruption_ms=1000*gate['bilateral_interruption']['total_s'],
        longest_interruption_ms=1000*gate['bilateral_interruption']['longest_s'],
        xy_RMSE_m=q['xy_RMSE_m'],terminal_stop_hold_s=q['terminal_stop_hold_s'],
        longest_corroborated_slide_s=max(q['corroborated_slide_longest_s']),first_45deg_tilt_s=q['first_45deg_tilt_s'],
        body_angular_acceleration_RMS_rad_s2=q['body_angular_acceleration_RMS_rad_s2'],
        action_change_RMS=q['action_change_RMS'],
        failed_gates='; '.join(q['failed_gates']),diagnostic_only=not gate['eligible_for_task_comparison'],
        qualification_sha256=sha(qpath),raw_recording_hashes_locally_checked=raw_checked)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','evidence','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh summary required')
    plan=json.loads((a.bundle/'plan.json').read_text())
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    for row in manifest['files']:
        if sha(a.bundle/row['path'])!=row['sha256']:raise ValueError('Declared snapshot changed')
    run_root=a.evidence/'run_v2';state=json.loads((run_root/'status.json').read_text())
    if state['manifest_sha256']!=sha(a.bundle/'manifest.json'):raise ValueError('Wrong remote experiment')
    observation=json.loads((a.evidence/'latest_observation.json').read_text())
    rows=[];parents=[]
    for cell in plan['cells']:
        protocol=a.bundle/cell['protocol']
        row=checked_row(run_root,cell,protocol,'nominal')
        rows.append(row or dict(cell=cell['id'],status='not_yet_qualified',task_success=None))
        parent=checked_row(run_root,cell,protocol,'parent_transfer')
        if parent is not None:parents.append(parent)
    complete=[row for row in rows if row['status']=='completed_nominal_qualification']
    report=dict(kind='hockey_contact_search_screen_summary_v1',source_manifest_sha256=sha(a.bundle/'manifest.json'),
        observed_unix_s=observation['observed_unix_s'],observed_remote_status=state['status'],
        observed_current_process=observation['current_process'],planned_cells=len(rows),
        completed_nominal_screens=len(complete),nominal_passes=sum(row['task_success'] for row in complete),
        overall_goal_achieved=False,robustness_suite_complete=False,failed_contact_cases_are_diagnostic_only=True,
        smoothness_metrics_scope='Entire recorded 139-control-step rollout (2.78 s), unchanged qualification diagnostics',
        screening_budget_is_not_original_full_training_budget=True,parent_controls=parents,screens=rows)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if complete or parents:
        fields=list((complete+parents)[0])
        with (a.output/'completed_cases.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(parents+complete)
    print(json.dumps(dict(completed_nominal_screens=len(complete),nominal_passes=report['nominal_passes'],
                         parent_controls=len(parents),remote_status=state['status'])))


if __name__=='__main__':main()
