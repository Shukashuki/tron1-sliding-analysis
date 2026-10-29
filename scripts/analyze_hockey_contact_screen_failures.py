"""Replay contact-state failures from completed screens without changing gates.

Counts geometric contact independently of load. Distinguishes single-wheel
support, complete contact loss and unknown geometry; none can pass the strict
bilateral gate. The summaries are diagnostic, never a new success ranking.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import derive_signals, load, segments, sha


def contact_states(touching, unknown):
    touching=np.asarray(touching,bool);unknown=np.asarray(unknown,bool)
    if touching.ndim!=2 or touching.shape[1]!=2 or unknown.shape!=touching.shape:
        raise ValueError('Expected two wheels at every sample')
    if (touching&unknown).any():raise ValueError('Touching cannot also be unknown')
    states=np.full(len(touching),4,dtype=np.uint8)
    known=~unknown.any(-1)
    states[known&touching.all(-1)]=0
    states[known&touching[:,0]&~touching[:,1]]=1
    states[known&~touching[:,0]&touching[:,1]]=2
    states[known&~touching.any(-1)]=3
    return states


def summarize_states(start, end, touching, unknown, horizon):
    start=np.asarray(start,float);end=np.asarray(end,float)
    if start.ndim!=1 or end.shape!=start.shape or len(start)!=len(touching):
        raise ValueError('Contact states and timestamps differ')
    dt=float(np.median(np.diff(end)))
    if not np.allclose(end-start,dt,rtol=0,atol=1e-9):raise ValueError('Incorrect physical clock')
    inside=(start>=-1e-9)&(start<horizon-1e-9)
    expected=np.arange(round(horizon/dt))*dt
    if inside.sum()!=len(expected) or not np.allclose(start[inside],expected,rtol=0,atol=1e-9):
        raise ValueError('Incomplete physical horizon')
    terminal=np.flatnonzero(np.isclose(start,horizon,atol=1e-9,rtol=0))
    if len(terminal)!=1:raise ValueError('Missing terminal contact sample')
    states=contact_states(touching,unknown)
    names=('both','left_only','right_only','neither','unknown')
    out={name:dict(duration_s=float((inside&(states==i)).sum()*dt),
                  events=segments(end,inside&(states==i),interval_samples=True)) for i,name in enumerate(names)}
    if not np.isclose(sum(row['duration_s'] for row in out.values()),horizon,atol=1e-9):
        raise ValueError('Contact partition does not cover the horizon')
    return dict(states=out,terminal_state=names[states[terminal[0]]],terminal_touching=touching[terminal[0]].tolist(),
                inside=inside,state_codes=states)


def terminal_stop_breakdown(time, touching, unknown, speed, tilt, horizon):
    """Replay the frozen last-0.5-second state gate, including its endpoint.

    Cause masks may overlap: sample counts are not additive durations. In
    particular, a one-sample interruption still breaks a continuous hold even
    though the state event has zero first-to-last duration.
    """
    time=np.asarray(time,float);speed=np.asarray(speed,float);tilt=np.asarray(tilt,float)
    touching=np.asarray(touching,bool);unknown=np.asarray(unknown,bool)
    contact_states(touching,unknown)
    if speed.shape!=time.shape or tilt.shape!=time.shape or len(touching)!=len(time):
        raise ValueError('Terminal signals have inconsistent lengths')
    window=(time>=horizon-.5-1e-9)&(time<=horizon+1e-9)&(time>=-1e-9)
    if not window.any() or not np.isfinite(speed[window]).all() or not np.isfinite(tilt[window]).all():
        raise ValueError('Missing terminal kinematics')
    terminal=np.flatnonzero(np.isclose(time,horizon,atol=1e-9,rtol=0))
    if len(terminal)!=1:raise ValueError('Missing terminal state')
    causes=dict(bilateral_contact_missing=~touching.all(-1),speed_above_0p1_m_s=speed>.1,
                tilt_above_30_deg=tilt>30,contact_unknown=unknown.any(-1))
    stopped=window&touching.all(-1)&(speed<=.1)&(tilt<=30)
    events=segments(time,stopped)
    end_index=terminal[0]
    return dict(window_s=[horizon-.5,horizon],state_samples=int(window.sum()),
        cause_sample_counts_are_not_additive=True,
        hold_s=max((e['duration_s'] for e in events),default=0.),hold_events=events,
        violations={name:dict(samples=int((window&mask).sum()),events=segments(time,window&mask))
                    for name,mask in causes.items()},
        speed_range_m_s=[float(speed[window].min()),float(speed[window].max())],
        maximum_tilt_deg=float(tilt[window].max()),
        terminal=dict(speed_m_s=float(speed[end_index]),tilt_deg=float(tilt[end_index]),
                      both_touching=bool(touching[end_index].all()),
                      stop_state=bool(stopped[end_index])))


def plot_group(directory, group, number):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig=plt.figure(figsize=(22,5),constrained_layout=True)
    grid=fig.add_gridspec(2,4,height_ratios=[3,1])
    for column in range(4):
        top=fig.add_subplot(grid[0,column]);bottom=fig.add_subplot(grid[1,column])
        if column>=len(group):
            top.text(.5,.5,'No completed case in this panel',ha='center',va='center',transform=top.transAxes)
            top.axis('off');bottom.axis('off');continue
        row,signals=group[column];t=signals['time_start_s'];inside=signals['inside']
        plot_mask=inside|np.isclose(t,row['horizon_s'],atol=1e-9,rtol=0)
        for side,color in enumerate(('#2677ad','#db772a')):
            top.plot(t[plot_mask],1000*signals['gap_pre_m'][plot_mask,side],color=color,lw=1,label=('Left','Right')[side])
        top.axhline(.1,color='black',ls='--',lw=.7,label='0.1 mm tolerance')
        for event in row['contact_states']['states']['neither']['events']:
            top.axvspan(event['start_s'],event['end_s'],color='crimson',alpha=.22)
        top.set(xlim=(0,row['horizon_s']),yscale='symlog',ylim=(-.2,30),ylabel='Wheel-floor clearance [mm]',
            title=f"{row['model']} / {row['policy']}\nBoth: {row['both_contact_percent']:.2f}% | Neither: {row['neither_contact_ms']:.1f} ms")
        top.grid(alpha=.2);top.legend(fontsize=7)
        touch=signals['touching'][inside];unknown=signals['unknown'][inside]
        codes=np.where(touch,1,np.where(unknown,2,0)).T
        both=np.where(touch.all(-1),1,np.where(unknown.any(-1),2,0))
        bottom.imshow(np.vstack((codes,both)),aspect='auto',interpolation='nearest',origin='upper',
            extent=[0,row['horizon_s'],2.5,-.5],vmin=0,vmax=2,cmap=ListedColormap(['#cd5353','#3b9867','#c7a246']))
        bottom.set(yticks=[0,1,2],yticklabels=['L','R','Both'],xlabel='Time [s]')
        bottom.text(.5,-.85,f"Terminal: {row['contact_states']['terminal_state']} | Hold: {row['terminal_stop_hold_s']:.4f} s",
            transform=bottom.transAxes,ha='center',fontsize=9)
    fig.suptitle('Full-horizon contact failure audit | Green: touch; red: missing; amber: unknown\n'
                 'Geometric contact, no load threshold | All failed-contact task metrics remain diagnostic')
    fig.savefig(directory/f'contact_states_{number:02d}_4x1.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','evidence','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh failure-audit output required')
    read=lambda path:json.loads(path.read_text())
    plan=read(a.bundle/'plan.json');reference=load(a.reference/'motion.npz')
    if sha(a.reference/'motion.npz')!=plan['reference_sha256']['motion.npz']:
        raise ValueError('Reference changed')
    horizon=(len(reference['joint_pos'])-1)/float(reference['fps'])
    rows=[];all_signals=[]
    for cell in plan['cells']:
        protocol=a.bundle/cell['protocol'];cfg=read(protocol)
        for label in ('parent_transfer','nominal'):
            parent=a.evidence/'run_v2'/cell['id'];qpath=parent/(label+'_qualification.json')
            if not qpath.exists():continue
            run_dir=parent/label;run=read(run_dir/'report.json');checked=read(qpath);q=checked['qualification']
            if checked['run_report_sha256']!=sha(run_dir/'report.json') or checked['protocol_sha256']!=sha(protocol):
                raise ValueError('Wrong qualification identity')
            for file,digest in run['artifacts'].items():
                if sha(run_dir/file)!=digest:raise ValueError('Downloaded recording hash mismatch')
            trace=load(run_dir/'final_rollout.npz');raw=load(run_dir/'native_contact_buffers.npz')
            c=contact_kinematics(previous_state_trace(trace),raw,.0001)
            te=np.arange(len(trace['root_pos']))*float(trace['physics_dt_s']);ts=te-float(trace['physics_dt_s'])
            partition=summarize_states(ts,te,c['touching'],c['contact_unknown'],horizon)
            inside=partition.pop('inside');codes=partition.pop('state_codes')
            fraction=partition['states']['both']['duration_s']/horizon
            if not np.isclose(fraction,q['primary_contact_gate']['both_touch_fraction'],atol=1e-12,rtol=0):
                raise ValueError('Independent contact replay differs from qualification')
            _,_,gap=wheel_geometry(trace,'body_pos','body_quat',cfg['bilateral_sliding']['geometry'])
            pre=np.maximum(np.arange(len(gap))-1,0);gap=gap[pre]
            kinematics=derive_signals(trace,{})
            terminal=terminal_stop_breakdown(ts,c['touching'],c['contact_unknown'],
                kinematics['body_speed_m_s'][pre],kinematics['root_tilt_deg'][pre],horizon)
            if not np.isclose(terminal['hold_s'],q['terminal_stop_hold_s'],atol=1e-12,rtol=0):
                raise ValueError('Independent terminal-stop replay differs from qualification')
            alignment=np.abs(gap-c['min_separation_m']);valid=inside[:,None]&np.isfinite(alignment)
            if not valid.any() or alignment[valid].max()>1e-6:
                raise ValueError('Authored clearance does not align with native pre-step contact geometry')
            policy='parent' if label=='parent_transfer' else cell['id'][len(cell['model'])+1:]
            row=dict(cell=cell['id'],model=cell['model'],policy=policy,horizon_s=horizon,
                both_contact_percent=100*fraction,
                one_wheel_only_ms=1000*(partition['states']['left_only']['duration_s']+partition['states']['right_only']['duration_s']),
                neither_contact_ms=1000*partition['states']['neither']['duration_s'],
                unknown_ms=1000*partition['states']['unknown']['duration_s'],
                maximum_gap_mm_per_wheel=(gap[inside].max(0)*1000).tolist(),
                terminal_gap_mm=gap[np.flatnonzero(np.isclose(ts,horizon,atol=1e-9,rtol=0))[0]].tolist(),
                terminal_stop_hold_s=q['terminal_stop_hold_s'],terminal_stop_breakdown=terminal,
                task_success=q['task_success_claimed'],
                failed_gates=q['failed_gates'],contact_states=partition,
                native_geometry_alignment_max_error_m=float(alignment[valid].max()),
                qualification_sha256=sha(qpath),raw_sha256=run['artifacts'])
            row['terminal_gap_mm']=[1000*v for v in row['terminal_gap_mm']]
            rows.append(row)
            all_signals.append(dict(time_start_s=ts,inside=inside,touching=c['touching'],unknown=c['contact_unknown'],
                gap_pre_m=gap,state_codes=codes))
    if not rows:raise ValueError('No completed recordings to analyze')
    a.output.mkdir(parents=True)
    for i in range(0,len(rows),4):plot_group(a.output,list(zip(rows,all_signals))[i:i+4],i//4)
    report=dict(kind='hockey_contact_screen_failure_modes_v2',script_sha256=sha(__file__),
        source_manifest_sha256=sha(a.bundle/'manifest.json'),physics_steps=0,criterion_changed=False,
        force_threshold_for_contact=None,missing_contact_is_not_proven_by_zero_load=True,
        single_seed_nominal_only=True,causal_mechanism_identified=False,cases=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    columns=['cell','model','policy','both_contact_percent','one_wheel_only_ms','neither_contact_ms','unknown_ms','terminal_stop_hold_s','task_success']
    with (a.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps([{k:row[k] for k in columns} for row in rows]))


if __name__=='__main__':main()
