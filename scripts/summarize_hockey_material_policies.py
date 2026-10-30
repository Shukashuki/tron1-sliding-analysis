"""Compare three fixed-policy materials and the matched split-trained policy.

The first three rows isolate material changes with a fixed actor. Row four
tests the separately trained actor at row two's material. All are closed loop.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda p:json.loads(p.read_text())
    sources=read(a.selection)
    if len(sources)!=4:raise ValueError('Require three fixed-policy probes and one trained-policy result')
    rows=[];identities=[];states=[]
    for i,source in enumerate(sources):
        bundle=(a.selection.parent/source['bundle']).resolve()
        for file in read(bundle/'manifest.json')['files']:
            p=(bundle/file['path']).resolve()
            if not p.is_relative_to(bundle) or sha(p)!=file['sha256']:raise ValueError('Source snapshot changed')
        cell,stage=source['case'].split(':')
        if stage!='nominal':raise ValueError('Use completed nominal validations')
        plan=read(bundle/'plan.json')
        protocol=bundle/next(x['protocol'] for x in plan['cells'] if x['id']==cell)
        cfg=read(protocol)
        directory=(a.selection.parent/source['evidence']/source['run_name']/cell/stage).resolve()
        qualification_path=directory.parent/'nominal_qualification.json'
        qualified=read(qualification_path);run=read(directory/'report.json')
        for key,p in [('source_manifest_sha256',bundle/'manifest.json'),('protocol_sha256',protocol),('run_report_sha256',directory/'report.json')]:
            if qualified[key]!=sha(p):raise ValueError('Qualification identity changed: '+key)
        if qualified['actor_sha256']!=run['actor_sha256']:raise ValueError('Actor identity changed')
        for name,digest in run['artifacts'].items():
            if sha(directory/name)!=digest:raise ValueError('Recording changed')
        material=run['material_audit'];mu_d=cfg['evaluation_mu'];mu_s=mu_d*cfg['static_dynamic_friction_ratio']
        if not material['live_physx_materials_verified'] or not material['ground_material_usd_verified']:raise ValueError('Materials unverified')
        for name,value in [('wheel_mu_range',mu_d),('wheel_mu_static_range',mu_s)]:
            np.testing.assert_allclose(material[name],[value,value],rtol=0,atol=1e-6)
        tr=load(directory/'final_rollout.npz')
        states.append({key:tr[key][0] for key in ('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel')})
        q=qualified['qualification']
        if len(q['gates'])!=11 or q['task_success_claimed']!=all(q['gates'].values()):raise ValueError('Inconsistent gates')
        rows.append(dict(case=chr(65+i),label=source['label'],mu_static=mu_s,mu_dynamic=mu_d,
            success=q['task_success_claimed'],failed_gates=','.join(q['failed_gates']),
            xy_RMSE_m=q['xy_RMSE_m'],xy_peak_m=q['xy_peak_m'],
            bilateral_interruption_ms=1000*q['primary_contact_gate']['bilateral_interruption']['total_s'],
            entry_speed_m_s=q['entry_mean_speed_m_s'],stop_hold_s=q['terminal_stop_hold_s'],
            slide_left_s=q['corroborated_slide_total_s'][0],slide_right_s=q['corroborated_slide_total_s'][1],
            angular_acceleration_RMS_rad_s2=q['body_angular_acceleration_RMS_rad_s2'],
            action_change_RMS=q['action_change_RMS'],wheel_heading_RMSE_deg=q['touching_braking_wheel_heading_RMSE_deg']))
        identities.append(dict(actor_sha256=qualified['actor_sha256'],qualification_sha256=sha(qualification_path),
            source_manifest_sha256=sha(bundle/'manifest.json'),protocol_sha256=sha(protocol),
            run_report_sha256=sha(directory/'report.json'),raw_sha256=run['artifacts']))
    actors=[r['actor_sha256'] for r in identities]
    if len(set(actors[:3]))!=1 or actors[3]==actors[0]:raise ValueError('Wrong fixed/trained actor grouping')
    if [(r['mu_static'],r['mu_dynamic']) for r in rows]!=[(.7,.7),(.9,.7),(.9,.9),(.9,.7)]:raise ValueError('Material grouping changed')
    errors=[max(float(np.max(abs(state[key]-states[0][key]))) for key in state) for state in states]
    if max(errors)>5e-5:raise ValueError('Actual initial states differ')
    numeric=[k for k in rows[0] if k not in ('case','label','success','failed_gates','mu_static','mu_dynamic')]
    effects=[]
    for name,new,old in [('increase_static_friction_same_actor',1,0),('decrease_dynamic_friction_same_actor',1,2),('split_training_same_evaluation_material',3,1)]:
        effects.append(dict(comparison=name,new_case=rows[new]['case'],baseline_case=rows[old]['case'],
            metric_differences={key:rows[new][key]-rows[old][key] for key in numeric}))
    a.output.mkdir(parents=True)
    with (a.output/'cases.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(22,4.7),constrained_layout=True)
    x=np.arange(4);colors=['#467cad','#cf8737','#779c50','#9676b3']
    for ax,key,title,factor,unit in [(axs[0],'xy_RMSE_m','XY tracking',100,'RMSE [cm]'),
        (axs[1],'bilateral_interruption_ms','Bilateral contact',1,'Interruption [ms]'),
        (axs[3],'angular_acceleration_RMS_rad_s2','Body smoothness',1,'Angular-acceleration RMS [rad/s²]')]:
        values=[r[key]*factor for r in rows];bars=ax.bar(x,values,color=colors)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.set(title=title,ylabel=unit)
        ax.set_ylim(0,max(1,max(values)*1.2));ax.grid(axis='y',alpha=.2)
    axs[2].bar(x-.18,[r['slide_left_s'] for r in rows],width=.36,label='Left',color='#2779a5')
    axs[2].bar(x+.18,[r['slide_right_s'] for r in rows],width=.36,label='Right',color='#d37527')
    axs[2].set(title='Qualified lateral sliding',ylabel='Duration [s]');axs[2].legend();axs[2].grid(axis='y',alpha=.2)
    for ax in axs:ax.set_xticks(x,['A\n0.7/0.7','B\n0.9/0.7','C\n0.9/0.9','D\n0.9/0.7'])
    fig.suptitle('A–C: same v8 policy | D: split-friction trained policy | static/dynamic coefficients\nOne nominal case per condition; same initial state, contact geometry, reference and frozen gates')
    fig.savefig(a.output/'policy_material_4x1.png',dpi=140);plt.close(fig)
    report=dict(kind='hockey_material_policy_comparison_v1',source_sha256=sha(__file__),selection_sha256=sha(a.selection),
        initial_state_maximum_errors=errors,cases=rows,identities=identities,effects=effects,
        physics_steps=0,criteria_changed=False,
        limitations=['Nominal cases are not new success-rate estimates.',
            'Material probes are closed-loop responses with the same actor, not identical-action open-loop tests.',
            'The separately trained actor has a different training-material distribution; no multiple-seed causal generalization.',
            'No imposed terrain step and no recorded solver static/dynamic state flag.'],
        artifact_sha256={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
