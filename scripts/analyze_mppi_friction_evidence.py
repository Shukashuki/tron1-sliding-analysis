"""Same-point slip, native friction-pyramid and dissipation audit of fixed MPPI.

Continuous replay must exactly reproduce the archived plant before evidence
is accepted. This is evaluation only: no search, policy, model or reward change.
Force and velocity are paired at each actual contact, not separately maximized
over different points. A force-consistent sliding screen supplements, rather
than replaces, the measured kinematic slip label.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import mujoco
import generate_mppi_contact_reference as g
from screen_mppi_sliding import runs

FIELDS=('sample_index','wheel_index','native_contact_index','geom_first','geom_second',
    'point_w','contact_distance_m','contact_frame_world_to_contact','normal_on_wheel_w','wheel_lateral_axis_w',
    'material_point_velocity_w','force_tangent_w','velocity_tangent_w',
    'normal_velocity_m_s','Fn_n','Ft_contact_n','velocity_tangent_contact_m_s',
    'mu_tangent','signed_lateral_m_s','tangent_speed_m_s','rho_pyramid','rho_ellipse','native_cone_margin',
    'friction_power_w','dissipated_power_w','maximum_dissipative_power_w','dissipation_efficiency',
    'opposition_cosine','point_velocity_jacobian_error_m_s')


def friction_metrics(normal,force_t,velocity_t,mu):
    """condim3 native pyramid: |Ft1|/mu1+|Ft2|/mu2 <= Fn.

    Its maximum-dissipation power follows by maximizing a linear function
    over this diamond: Fn*max(mu1*|vt1|,mu2*|vt2|). Antiparallel force and
    velocity are NOT required by a pyramidal approximation.
    """
    f,v,mu=np.asarray(force_t),np.asarray(velocity_t),np.asarray(mu)
    if normal<=0 or np.any(mu<=0) or f.shape!=(2,) or v.shape!=(2,) or mu.shape!=(2,):raise ValueError('Invalid loaded contact')
    normalized=f/(normal*mu)
    power=float(f@v);available=float(normal*np.max(mu*np.abs(v)))
    denominator=float(np.linalg.norm(f)*np.linalg.norm(v))
    rho=float(np.sum(abs(normalized)))
    return dict(rho_pyramid=rho,rho_ellipse=float(np.linalg.norm(normalized)),native_cone_margin=1-rho,
        friction_power_w=power,dissipated_power_w=-power,maximum_dissipative_power_w=available,
        dissipation_efficiency=-power/available if available>1e-12 else float('nan'),
        opposition_cosine=-power/denominator if denominator>1e-12 else float('nan'))


def contact_points(model,data,sample_index):
    floor=model.geom('floor').id;wheels=[model.body('wheel_'+s+'_Link').id for s in ('L','R')]
    rows=[]
    for index,c in enumerate(data.contact):
        geoms=list(c.geom)
        if floor not in geoms:continue
        wheel_slot=1 if geoms[0]==floor else 0
        body=int(model.geom_bodyid[geoms[wheel_slot]])
        if body not in wheels:continue
        wrench=np.empty(6);mujoco.mj_contactForce(model,data,index,wrench)
        if wrench[0]<-1e-10:raise ValueError('Negative normal force')
        if wrench[0]<1e-6:continue
        if c.dim!=3 or model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL:raise ValueError('Require original condim3 pyramidal contact')
        frame=np.asarray(c.frame).reshape(3,3);sign=1. if wheel_slot==1 else -1.
        normal=sign*frame[0]
        np.testing.assert_allclose(normal,[0,0,1],atol=1e-8,rtol=0)
        twist=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,twist,0)
        point_velocity=twist[3:]+np.cross(twist[:3],c.pos-data.xipos[body])
        jp,jr=np.empty((3,model.nv)),np.empty((3,model.nv));mujoco.mj_jac(model,data,jp,jr,c.pos,body)
        jac_error=float(np.max(abs(point_velocity-jp@data.qvel)))
        if jac_error>1e-10:raise ValueError('Contact point velocity/Jacobian mismatch')
        # mj_contactForce acts on geom2. Convert to force on the wheel,
        # regardless of native geom ordering; velocity is wheel minus floor.
        ft=sign*wrench[1:3];vt=frame[1:]@point_velocity
        axis=data.xmat[body].reshape(3,3)[:,1].copy();axis-=normal*(normal@axis)
        if np.linalg.norm(axis)<1e-8:raise ValueError('Degenerate projected wheel axle')
        axis/=np.linalg.norm(axis)
        force_w=frame[1:].T@ft;velocity_w=point_velocity-normal*(point_velocity@normal)
        if not np.isclose(force_w@velocity_w,ft@vt,atol=1e-8,rtol=1e-12):raise ValueError('World/contact friction power mismatch')
        row=dict(sample_index=sample_index,wheel_index=wheels.index(body),native_contact_index=index,
            geom_first=geoms[0],geom_second=geoms[1],point_w=c.pos.copy(),contact_distance_m=float(c.dist),
            contact_frame_world_to_contact=frame.copy(),normal_on_wheel_w=normal,wheel_lateral_axis_w=axis,
            material_point_velocity_w=point_velocity,force_tangent_w=force_w,velocity_tangent_w=velocity_w,
            normal_velocity_m_s=float(point_velocity@normal),Fn_n=float(wrench[0]),Ft_contact_n=ft.copy(),
            velocity_tangent_contact_m_s=vt.copy(),mu_tangent=np.asarray(c.friction[:2]).copy(),
            signed_lateral_m_s=float(point_velocity@axis),tangent_speed_m_s=float(np.linalg.norm(vt)),
            point_velocity_jacobian_error_m_s=jac_error,**friction_metrics(float(wrench[0]),ft,vt,c.friction[:2]))
        rows.append(row)
    return rows


def matched_load_fraction(points,samples,cfg,*,slip_threshold=None,cone_threshold=None):
    """Only SAME-point conjunctions enter the force-supported slip numerator."""
    index=points['sample_index'].astype(int)*2+points['wheel_index'].astype(int)
    eligible=points['Fn_n']>cfg['point_Fn_n_strict'];weight=points['Fn_n']*eligible
    moving=abs(points['signed_lateral_m_s'])>(cfg['lateral_speed_m_s_strict'] if slip_threshold is None else slip_threshold)
    saturated=points['rho_pyramid']>=(cfg['pyramidal_utilization_min'] if cone_threshold is None else cone_threshold)
    dissipative=(points['dissipated_power_w']>0)&(points['dissipation_efficiency']>=cfg['maximum_dissipation_efficiency_min'])
    joint=eligible&moving&saturated&dissipative
    denominator=np.bincount(index,weights=weight,minlength=samples*2).reshape(samples,2)
    numerator=np.bincount(index,weights=weight*joint,minlength=samples*2).reshape(samples,2)
    return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0),denominator,joint


def aggregate(points,raw,cfg):
    n=len(raw['time_s']);index=points['sample_index'].astype(int)*2+points['wheel_index'].astype(int)
    eligible=points['Fn_n']>cfg['point_Fn_n_strict'];weight=points['Fn_n']*eligible
    fraction,denominator,joint=matched_load_fraction(points,n,cfg)
    out=dict(time_s=raw['time_s'],wheel_Fz_n=raw['ground_normal_force_w'][...,2],eligible_normal_load_n=denominator,
        matched_slip_cone_dissipation_load_fraction=fraction)
    def mean(value,mask=eligible):
        valid=mask&np.isfinite(value);w=points['Fn_n']*valid
        bottom=np.bincount(index,weights=w,minlength=n*2).reshape(n,2)
        top=np.bincount(index,weights=np.where(valid,value,0)*w,minlength=n*2).reshape(n,2)
        return np.divide(top,bottom,out=np.full_like(top,np.nan),where=bottom>0)
    out['lateral_rms_m_s']=np.sqrt(mean(points['signed_lateral_m_s']**2))
    out['rho_pyramid_load_weighted_mean']=mean(points['rho_pyramid'])
    out['rho_ellipse_load_weighted_mean']=mean(points['rho_ellipse'])
    active=eligible&(abs(points['signed_lateral_m_s'])>cfg['lateral_speed_m_s_strict'])
    out['sliding_point_dissipation_efficiency_load_weighted_mean']=mean(points['dissipation_efficiency'],active)
    out['friction_dissipated_power_w']=np.bincount(index,weights=np.where(eligible,points['dissipated_power_w'],0),minlength=n*2).reshape(n,2)
    return out,joint


def summaries(points,wheel,raw,cfg):
    time=raw['time_s'];low,high=cfg['core_interval_s'];core=(time>=low-1e-9)&(time<=high+1e-9)
    moving=np.linalg.norm(raw['base_com_velocity_w'][:,:2],axis=1)>=cfg['base_speed_m_s_min']
    t=time[core];rows=[]
    def quantile(value,mask):
        v=value[mask&np.isfinite(value)];return np.quantile(v,[.05,.5,.95]).tolist() if len(v) else None
    for side,name in enumerate(('L','R')):
        valid=core&moving&(wheel['wheel_Fz_n'][:,side]>cfg['wheel_support_Fz_n_strict'])&(wheel['eligible_normal_load_n'][:,side]>0)
        slip=valid&(wheel['lateral_rms_m_s'][:,side]>cfg['lateral_speed_m_s_strict'])
        joint=valid&(wheel['matched_slip_cone_dissipation_load_fraction'][:,side]>=cfg['matched_point_normal_load_fraction_min'])
        events=runs(t,joint[core]);persistent=[r for r in events if r['duration_s']>=cfg['continuous_event_duration_s_min']-1e-9]
        si=points['sample_index'].astype(int)
        eligible=(points['wheel_index']==side)&core[si]&moving[si]&(points['Fn_n']>cfg['point_Fn_n_strict'])
        active=eligible&(abs(points['signed_lateral_m_s'])>cfg['lateral_speed_m_s_strict'])
        near=points['rho_pyramid']>=cfg['pyramidal_utilization_min']
        dissipative=(points['dissipated_power_w']>0)&(points['dissipation_efficiency']>=cfg['maximum_dissipation_efficiency_min'])
        def fraction(mask,condition):
            total=float(np.sum(points['Fn_n'][mask]));return float(np.sum(points['Fn_n'][mask&condition])/total) if total>0 else None
        rows.append(dict(side=name,valid_supported_moving_wheel_samples=int(valid.sum()),
            velocity_only_sliding_wheel_samples=int(slip.sum()),same_point_force_corroborated_wheel_samples=int(joint.sum()),
            velocity_only_sliding_fraction=float(slip.sum()/valid.sum()) if valid.any() else None,
            joint_corroborated_fraction=float(joint.sum()/valid.sum()) if valid.any() else None,
            longest_joint_corroborated_s=max((r['duration_s'] for r in events),default=0.),persistent_events=persistent,
            core_loaded_point_samples=int(eligible.sum()),core_laterally_sliding_point_samples=int(active.sum()),
            sliding_point_normal_load_fraction_near_native_cone=fraction(active,near),
            sliding_point_normal_load_fraction_dissipative=fraction(active,dissipative),
            sliding_point_normal_load_fraction_near_cone_and_dissipative=fraction(active,near&dissipative),
            sliding_point_rho_pyramid_p05_p50_p95=quantile(points['rho_pyramid'],active),
            sliding_point_rho_ellipse_p05_p50_p95=quantile(points['rho_ellipse'],active),
            sliding_point_dissipation_efficiency_p05_p50_p95=quantile(points['dissipation_efficiency'],active)))
    sensitivity=[]
    for slip,cone,duration in itertools.product(cfg['sensitivity_lateral_speed_m_s'],cfg['sensitivity_cone_utilization'],cfg['sensitivity_duration_s']):
        fraction,denominator,_=matched_load_fraction(points,len(time),cfg,slip_threshold=slip,cone_threshold=cone)
        for side in (0,1):
            mask=core&moving&(wheel['wheel_Fz_n'][:,side]>cfg['wheel_support_Fz_n_strict'])&(denominator[:,side]>0)&(fraction[:,side]>=cfg['matched_point_normal_load_fraction_min'])
            events=runs(t,mask[core])
            sensitivity.append(dict(side=('L','R')[side],slip_m_s=slip,rho_min=cone,duration_min_s=duration,
                longest_joint_s=max((r['duration_s'] for r in events),default=0.),
                persistent_events=sum(r['duration_s']>=duration-1e-9 for r in events)))
    return rows,sensitivity


def replay(case,experiment,model_path,reference_path,material,cfg,output):
    raw_path=experiment/case['audit']/'evidence/rollout.npz';report_path=raw_path.with_name('report.json')
    report=json.loads(report_path.read_text());old=json.loads((experiment/case['audit']/'analysis.json').read_text())
    if g.sha(raw_path)!=report['rollout_sha256'] or g.sha(raw_path)!=old['rollout_sha256'] or g.sha(report_path)!=old['run_report_sha256']:raise ValueError('Fixed input evidence differs')
    with np.load(raw_path,allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
    with np.load(reference_path,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    if len(raw['time_s'])!=2793 or not np.allclose(raw['time_s'],np.arange(2793)*.0025,atol=1e-9,rtol=0):raise ValueError('Incomplete or discontinuous clock')
    m=g.load_model(model_path,material);d=mujoco.MjData(m)
    d.qpos[:]=material['initial_state']['qpos'];d.qvel[:]=material['initial_state']['qvel'];mujoco.mj_forward(m,d)
    q0=source['joint_pos'][0,[list(source['joint_names']).index(n) for n in g.JOINT_NAMES]]
    aids=[m.actuator(n).id for n in g.JOINT_NAMES];points=contact_points(m,d,0);errors=[];frame=0
    for action in raw['actions']:
        for sub in range(8):
            tau=g.torque(m,d,q0,action);d.ctrl[aids]=tau;mujoco.mj_step(m,d);mujoco.mj_forward(m,d);frame+=1
            if any(w.number for w in d.warning) or np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('Native warning or unexpected external force')
            errors.append([np.max(abs(d.qpos-raw['qpos'][frame])),np.max(abs(d.qvel-raw['qvel'][frame])),np.max(abs(g.ground_contact_readout(m,d)['ground_normal_force_w']-raw['ground_normal_force_w'][frame]))])
            points.extend(contact_points(m,d,frame))
    errors=np.asarray(errors)
    if np.max(errors[:,0])>1e-10 or np.max(errors[:,1])>1e-9 or np.max(errors[:,2])>1e-7:raise ValueError('Continuous replay failed; reject all friction evidence')
    p={k:np.asarray([row[k] for row in points]) for k in FIELDS};w,_=aggregate(p,raw,cfg)
    finite=np.isfinite(p['dissipation_efficiency'])
    if np.max(p['rho_pyramid'])>1+1e-8 or np.any(abs(p['dissipation_efficiency'][finite])>p['rho_pyramid'][finite]+1e-8):
        raise ValueError('Force-cone feasibility or its derived power bound is violated; do not publish corroboration')
    with np.load(experiment/case['audit']/'contact_audit.npz') as prior:
        mask=prior['loaded_point_count']>0
        slip_error=float(np.max(abs(w['lateral_rms_m_s'][mask]-prior['lateral_rms_m_s'][mask])))
        if slip_error>1e-10:raise ValueError('Replayed material-point slip differs from original audit')
    rows,sensitivity=summaries(p,w,raw,cfg)
    result=dict(id=case['id'],title=case['title'],point_rows=len(points),rows=rows,sensitivity=sensitivity,
        max_rho_pyramid=float(p['rho_pyramid'].max()),max_rho_ellipse=float(p['rho_ellipse'].max()),
        max_point_velocity_jacobian_error_m_s=float(p['point_velocity_jacobian_error_m_s'].max()),
        replay_max_error_qpos_qvel_Fz=errors.max(0).tolist(),prior_point_slip_max_error_m_s=slip_error,
        input_rollout_sha256=g.sha(raw_path),input_report_sha256=g.sha(report_path),prior_audit_sha256=g.sha(experiment/case['audit']/'analysis.json'),
        contact_friction_min=p['mu_tangent'].min(0).tolist(),contact_friction_max=p['mu_tangent'].max(0).tolist(),
        native_cone='pyramidal',native_condim=3,physical_samples=2793,root_actuation_added=False,
        actual_contact_force_timing='post-integration mj_forward refreshed solver force and point velocity at SAME state; not preceding-substep impulse',
        solver=dict(iterations=int(m.opt.iterations),tolerance=float(m.opt.tolerance),cone=int(m.opt.cone),integrator=int(m.opt.integrator),timestep=float(m.opt.timestep)),
        original_nominal_checks=old['checks'],nominal_task_success_claimed=False)
    output.mkdir();np.savez_compressed(output/'contact_points.npz',**p);np.savez_compressed(output/'wheel_evidence.npz',**w)
    result['point_data_sha256']=g.sha(output/'contact_points.npz');result['wheel_data_sha256']=g.sha(output/'wheel_evidence.npz')
    g.write_json(output/'analysis.json',result)
    return result,p,w


def plot_case(result,p,w,cfg,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(24,5.2),layout='constrained');colors=['#d62728','#9467bd']
    for i,color in enumerate(colors):
        axes[0].plot(w['time_s'],w['wheel_Fz_n'][:,i],color=color,lw=.65,label=('L','R')[i])
        axes[1].semilogy(w['time_s'],np.maximum(w['lateral_rms_m_s'][:,i],1e-6),color=color,lw=.65,label=('L','R')[i])
        axes[2].plot(w['time_s'],w['rho_pyramid_load_weighted_mean'][:,i],color=color,lw=.65,label=('L','R')[i])
    axes[0].axhline(5,color='black',ls=':');axes[0].set_ylabel('Actual upward force [N]');axes[0].set_title('Wheel-floor support')
    axes[1].axhline(cfg['lateral_speed_m_s_strict'],color='black',ls=':');axes[1].set_ylabel('Loaded lateral RMS [m/s]');axes[1].set_title('Unloaded / unobserved = gaps')
    axes[2].axhline(cfg['pyramidal_utilization_min'],color='black',ls=':');axes[2].axhline(1,color='gray',ls='--');axes[2].set_ylim(0,1.08);axes[2].set_ylabel('Normal-load weighted native rho');axes[2].set_title('Pyramid utilization; same contact samples')
    sample_time=w['time_s'][p['sample_index'].astype(int)]
    mask=(sample_time>=cfg['core_interval_s'][0]-1e-9)&(sample_time<=cfg['core_interval_s'][1]+1e-9)&(p['Fn_n']>cfg['point_Fn_n_strict'])
    # Each dot is one actual point, with paired force AND velocity.
    scatter=axes[3].scatter(p['rho_pyramid'][mask],np.maximum(abs(p['signed_lateral_m_s'][mask]),1e-6),
        c=np.clip(p['dissipation_efficiency'][mask],-1,1),s=7,alpha=.55,cmap='coolwarm_r',vmin=-1,vmax=1)
    axes[3].set_yscale('log');axes[3].set_xlim(0,1.04);axes[3].axvline(cfg['pyramidal_utilization_min'],color='black',ls=':')
    axes[3].axhline(cfg['lateral_speed_m_s_strict'],color='black',ls=':');axes[3].set_xlabel('Native pyramid rho (same point)');axes[3].set_ylabel('Absolute lateral point speed [m/s]');axes[3].set_title('Paired point evidence, turn core')
    fig.colorbar(scatter,ax=axes[3],label='D / maximum possible D')
    for ax in axes[:3]:
        ax.axvspan(*cfg['core_interval_s'],color='gray',alpha=.08);ax.set_xlabel('Time [s]');ax.legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(result['title']+' | fixed recorded actions; no new MPPI search | not a successful reference',fontsize=13)
    fig.savefig(path,dpi=145);plt.close(fig)


def run(a):
    if a.output_dir.exists():raise FileExistsError('Use a fresh evidence directory')
    cfg=json.loads(a.config.read_text());summary=json.loads((a.experiment/'summary.json').read_text())
    if cfg['kind']!='native_contact_slip_cone_dissipation_joint_audit_v1':raise ValueError('Wrong audit config')
    material_path=a.experiment/'inputs/original_native_material_report.json'
    if g.sha(a.model)!=g.MODEL_SHA or g.sha(a.reference)!=g.REFERENCE_SHA or g.sha(material_path)!=g.MATERIAL_REPORT_SHA:raise ValueError('Input asset/material hash mismatch')
    material=json.loads(material_path.read_text());a.output_dir.mkdir(parents=True);results=[]
    for case in summary['rows']:
        r,p,w=replay(case,a.experiment,a.model,a.reference,material,cfg,a.output_dir/case['id']);plot_case(r,p,w,cfg,a.output_dir/case['id']/'joint_evidence_4x1.png');results.append(r)
        print(json.dumps(dict(id=r['id'],rows=r['rows'],replay_errors=r['replay_max_error_qpos_qvel_Fz'])),flush=True)
    result=dict(kind=cfg['kind'],config=cfg,source_sha256=g.sha(__file__),generator_source_sha256=g.sha(g.__file__),
        event_helper_source_sha256=g.sha(g.ROOT/'scripts/screen_mppi_sliding.py'),config_sha256=g.sha(a.config),
        source_experiment_summary_sha256=g.sha(a.experiment/'summary.json'),model_sha256=g.sha(a.model),reference_sha256=g.sha(a.reference),
        material_sha256=g.sha(material_path),mujoco_version=mujoco.__version__,rows=results,
        same_contact_point_force_velocity_pairing=True,original_search_reward_policy_and_task_gates_unchanged=True,
        no_new_MPPI_or_policy_training=True,force_evidence_is_additional_not_definition_of_velocity_slip=True,
        theoretical_sources=['https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones',
            'https://github.com/google-deepmind/mujoco/blob/3.8.1/src/engine/engine_util_misc.c'],
        limitations=['Diagnostic thresholds are declared and swept, not a universal static/sliding law.',
            'Native soft-contact model may have finite slip inside the friction cone; lack of saturation does not erase measured motion.',
            'The maximum-dissipation normalization is a derived hard-cone reference, not an equality guaranteed by MuJoCo soft contact.',
            'Friction direction need not be antiparallel to velocity for pyramidal contact; use force power and the native support function.',
            'Summed wheel normal load and eligible point load differ; Fn<=5N points are retained but excluded from the primary estimator.',
            'Joint evidence requires at least half of eligible normal load at points that jointly pass all gates; different points are not mixed.',
            'No gaps or alternating wheels are stitched into continuous same-wheel events.',
            'This is a same-engine replay and model-consistency audit, not validation against physical tire sensors or PhysX.'],
        successful_drift_demonstration_claimed=False)
    g.write_json(a.output_dir/'summary.json',result)
    (a.output_dir/'report.html').write_text(html_report(result,a.experiment.name))


def html_report(result,experiment_name):
    import html
    table=''
    media=''
    for case in result['rows']:
        for side in case['rows']:
            def pct(v):return 'N/A' if v is None else f'{100*v:.1f}%'
            table+='<tr>'+''.join('<td>'+v+'</td>' for v in [html.escape(case['title']),side['side'],str(side['valid_supported_moving_wheel_samples']),
                pct(side['velocity_only_sliding_fraction']),pct(side['joint_corroborated_fraction']),f"{1000*side['longest_joint_corroborated_s']:.1f}",str(len(side['persistent_events'])),
                pct(side['sliding_point_normal_load_fraction_near_cone_and_dissipative'])])+'</tr>'
        media+=f'<h2>{html.escape(case["title"])}</h2><img src="{case["id"]}/joint_evidence_4x1.png"><p><a href="{case["id"]}/analysis.json">Analysis and threshold sensitivity</a> · <a href="{case["id"]}/contact_points.npz">All actual point samples</a> · <a href="{case["id"]}/wheel_evidence.npz">Wheel timelines</a></p>'
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint contact, slip and friction evidence</title><style>body{font:17px system-ui;max-width:1500px;margin:35px auto;padding:0 24px;line-height:1.6}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #ccc;padding:8px}img{width:100%}.note{color:#9c2828}pre{white-space:pre-wrap;background:#f0f2f4;padding:16px}</style>
<h1>Contact continuity + same-point slip + native friction-pyramid evidence</h1>
<p>Fixed MPPI actions, original rigid cylinders, μ=0.7 and400Hz native MuJoCo. This is a re-analysis of both failed candidates, not new controller development. Both continuous replays reproduce archived qpos,qvel and wheel-floor Fz exactly. Material-point velocities are independently checked against their MuJoCo point Jacobians.</p>
<p>Each point carries its own paired normal/tangential force, velocity, actual contact friction coefficients and geometry ordering. We do not combine a wheel's maximum cone utilization at one point with a large velocity at another point.</p>
<pre>Native condim3 pyramid: rho = |Ft1|/(mu1 Fn) + |Ft2|/(mu2 Fn) <= 1
Margin to native boundary = 1-rho
Dissipated friction power: D = -(Ft1 vt1 + Ft2 vt2)
Maximum power over this diamond: Dmax = Fn max(mu1 |vt1|, mu2 |vt2|)
eta = D/Dmax (undefined for zero tangential velocity)</pre>
<p>Dmax is our direct derivation from the documented pyramid edges. This is a hard-cone benchmark, not an equality assumed for MuJoCo soft contact. The circular L2 ratio is retained for comparison, but the actual pyramid has a different boundary. Antiparallel force and velocity are not required by a pyramid.</p>
<p><strong>Joint evidence:</strong> wheel summed Fz&gt;5N,base COM speed≥0.25m/s; eligible points have Fn&gt;5N. At the same points,|lateral velocity|&gt;0.05m/s,rho≥0.9,D&gt;0 and eta≥0.5. These matched points must account for at least50% of eligible wheel normal load. A sustained event requires the same wheel to satisfy this continuously for at least50ms. Contact-point identity may change during rolling; unsupported gaps and alternating wheels are never stitched together.</p>
<p>Velocity-based sliding remains a measured kinematic label even when the force corroboration is absent. We separately report near-boundary sliding,interior-cone sliding,and dissipation evidence rather than calling interior-cone motion impossible. Sensitivity spans velocity0.02/0.05/0.1m/s,rho0.8/0.9/0.95 and duration20/50/100ms; eta and matched-load fraction remain fixed.</p>
<table><tr><th>Attempt</th><th>Wheel</th><th>Valid supported moving samples</th><th>Velocity-only sliding</th><th>Joint corroborated</th><th>Longest joint [ms]</th><th>Events≥50ms</th><th>Sliding-point load near cone AND dissipative</th></tr>'''+table+'''</table><p>Wheel percentages use valid wheel-time samples. The last column is normal-load weighted over laterally sliding point samples; it is a different denominator. Statistics cover1.44–4.00s. All point arrays retain weak positive contacts; only the primary estimator excludes Fn≤5N.</p>
<p class="note">Neither fixed attempt is a successful stable-contact drift demonstration. Force-velocity consistency is additional evidence, not a replacement for full-body stability or proof of real tire behavior.</p>'''+media+f'''<p><a href="../{experiment_name}/report.html">Original trajectories and fixed-camera videos</a> · <a href="summary.json">Full audit</a> · <a href="https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones">MuJoCo primary documentation</a> · <a href="https://github.com/google-deepmind/mujoco/blob/3.8.1/src/engine/engine_util_misc.c">Pinned3.8.1 native cone conversion</a></p>
<pre>python scripts/analyze_mppi_friction_evidence.py --output-dir results/mppi_friction_evidence_reproduction</pre></html>'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--experiment',type=Path,default=g.ROOT/'results/mppi_contact_reference_20261025_v1')
    p.add_argument('--model',type=Path,default=g.ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml')
    p.add_argument('--reference',type=Path,default=g.ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz')
    p.add_argument('--config',type=Path,default=g.ROOT/'config/mppi_friction_evidence_v1.json')
    p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args())
