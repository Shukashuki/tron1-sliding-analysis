"""Report model-capability tests and audited open-source tire-model candidates."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import subprocess
import numpy as np
import generate_mppi_contact_reference as g


def run(a):
    if (a.result/'report.html').exists():raise FileExistsError('Preserve completed model report')
    d=json.loads((a.result/'analysis.json').read_text());rows=d['rows'];cfg=d['config'];cases=cfg['cases'];par=d['parameters']
    if d['source_sha256']!=g.sha(g.ROOT/'scripts/coupled_sliding_model_bench.py'):raise ValueError('Executed model source changed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    figures=[];loaded={}
    for row in rows:
        path=a.result/row['trace_file']
        if g.sha(path)!=row['trace_sha256']:raise ValueError('Model trace changed')
        with np.load(path) as f:loaded[(row['id'],row['dt_s'])]={k:f[k].copy() for k in f.files}
    fig,axes=plt.subplots(1,4,figsize=(23,5),layout='constrained')
    for j,case in enumerate(cases):
        ax=axes[j];right=ax.twinx()
        for dt,style in zip(cfg['physics_dt_s'],('-','--')):
            r=loaded[(case['id'],dt)];t=r['time_s']
            ax.plot(t,abs(r['slip_m_s'][:,0,1]),style,color='#d62728',label=f'L lateral /{round(1/dt)}Hz')
            ax.plot(t,abs(r['slip_m_s'][:,1,1]),style,color='#9467bd',alpha=.6,label=f'R lateral /{round(1/dt)}Hz')
            if dt==cfg['physics_dt_s'][0]:
                ax.plot(t,np.max(abs(r['slip_m_s'][...,0]),axis=1),':',color='#e49b21',label='Max longitudinal')
                right.plot(t,np.min(r['rho'],axis=1),color='#166b4e',lw=.9,label='Min wheel rho')
        ax.axhline(cfg['sliding_speed_threshold_m_s'],color='gray',ls=':');right.axhline(1,color='#166b4e',ls=':')
        ax.set_title(case['title'],fontsize=10);ax.set_xlabel('Time [s]');ax.set_ylabel('Post-impulse contact slip [m/s]')
        right.set_ylim(0,1.1);right.set_ylabel('Actual circular-cone utilization')
        ax.axvspan(2,4,color='gray',alpha=.07);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper left');right.legend(fontsize=7,loc='lower right')
    fig.suptitle('Model capability, not free TRON1 validation | fixed vertical support | same motor curve | force and slip solved together')
    fig.savefig(a.result/'saturation_and_slip_4x1.png',dpi=145);plt.close(fig);figures.append('saturation_and_slip_4x1.png')
    fig,axes=plt.subplots(1,4,figsize=(23,5.5),layout='constrained')
    for j,case in enumerate(cases):
        ax=axes[j];r=loaded[(case['id'],cfg['physics_dt_s'][0])];fn=r['normal_load_n']*case['mu'];force=r['force_t_n']/fn[...,None]
        angle=np.linspace(0,2*np.pi,401);ax.plot(np.cos(angle),np.sin(angle),color='black',lw=1.5)
        scatter=ax.scatter(force[...,0].ravel(),force[...,1].ravel(),c=np.maximum(abs(r['slip_m_s'][...,1]).ravel(),1e-4),
            cmap='viridis',norm=LogNorm(vmin=1e-4,vmax=5),s=6,alpha=.5)
        ax.set_aspect('equal');ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1);ax.grid(alpha=.2)
        ax.set_xlabel('Fx / (mu Fn)');ax.set_ylabel('Fy / (mu Fn)');ax.set_title(case['title'],fontsize=10)
    fig.colorbar(scatter,ax=axes.tolist(),label='Actual lateral constraint slip [m/s]',shrink=.8)
    fig.suptitle('Circular Coulomb boundary in the NEW coupled model | not the earlier MuJoCo pyramidal law')
    fig.savefig(a.result/'friction_sections_4x1.png',dpi=145);plt.close(fig);figures.append('friction_sections_4x1.png')
    fig,axes=plt.subplots(1,4,figsize=(23,5),layout='constrained')
    for j,case in enumerate(cases):
        ax=axes[j]
        for dt,style in zip(cfg['physics_dt_s'],('-','--')):
            r=loaded[(case['id'],dt)];ax.plot(*r['position_xy_m'].T,style,label=f'{round(1/dt)}Hz')
        r=loaded[(case['id'],cfg['physics_dt_s'][0])]
        for k in np.linspace(0,len(r['time_s'])-1,9).astype(int):
            xy=r['position_xy_m'][k];yaw=r['yaw_rad'][k];v=r['velocity'][k,:2];size=max(np.ptp(r['position_xy_m'],axis=0).max()*.04,.07)
            ax.arrow(*xy,size*np.cos(yaw),size*np.sin(yaw),width=size*.025,color='#1564a5',length_includes_head=True)
            ax.arrow(*xy,*(size*v/max(np.linalg.norm(v),1e-10)),width=size*.025,color='#248534',length_includes_head=True)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X [m]');ax.set_ylabel('World Y [m]');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(case['title'],fontsize=10)
    fig.suptitle('Planar fixture trajectories, not playback or retarget | blue body heading / green COM velocity | axes per case')
    fig.savefig(a.result/'planar_paths_4x1.png',dpi=145);plt.close(fig);figures.append('planar_paths_4x1.png')
    chrono=g.ROOT/'upstream/chrono'
    commit=subprocess.check_output(['git','-C',str(chrono),'rev-parse','HEAD'],text=True).strip()
    tire_paths={name:'src/chrono_vehicle/wheeled_vehicle/tire/'+name+'.cpp' for name in ('ChFialaTire','ChTMeasyTire','ChPac02Tire')}
    sources={name:dict(path='upstream/chrono/'+p,sha256=g.sha(chrono/p),url=f'https://github.com/projectchrono/chrono/blob/{commit}/{p}') for name,p in tire_paths.items()}
    source_audit=dict(kind='audited_model_options_not_installed_runtimes',Chrono_submodule_commit=commit,Chrono_sources=sources,
        existing_external_MF=dict(path='scripts/nonlinear_tire_contact.py',sha256=g.sha(g.ROOT/'scripts/nonlinear_tire_contact.py'),
            findings=['Steady dissipative combined-slip shape only.','No static adhesion,tread memory,camber thrust or identified real TRON1 tire coefficients.']),
        findings=[dict(model='Chrono Fiala',finding='Coupled longitudinal/lateral effects; documented zero-camber assumption. Not a camber-steering model.'),
            dict(model='Chrono basic TMeasy',finding='Audited file has low-speed Dahl bristle state and combined force curve. gamma enters overturning Mx; no explicit gamma-to-Fy term in this file. Do not infer commercial-version capability.'),
            dict(model='Chrono Pac02',finding='Has gamma-dependent tire forces. Current default camber clamp is3deg and is adjustable; the code says this avoids extrapolation beyond tire testing. Large-TRON1-camber use needs identified parameters and validation.'),
            dict(model='PhysX Vehicle default tire',finding='Documented combined-force normalization and camber stiffness. Separate Vehicle force model, not automatically part of an articulated rigid cylinder; not full Pacejka.')],
        official_sources=['https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/friction_contact.html',
            'https://api.projectchrono.org/wheeled_tire.html','https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Vehicles.html'],
        native_libraries_built_or_installed=False,Chrono_used_to_simulate_current_bench=False)
    g.write_json(a.result/'model_source_audit.json',source_audit)
    comparison=[]
    for case in cases:
        r1=next(r for r in rows if r['id']==case['id'] and r['dt_s']==.0025);r2=next(r for r in rows if r['id']==case['id'] and r['dt_s']==.00125)
        comparison.append(dict(id=case['id'],speed_mean_relative_difference=abs(r2['steady_speed_mean_m_s']-r1['steady_speed_mean_m_s'])/r1['steady_speed_mean_m_s'],
            yaw_rate_mean_abs_difference=abs(r2['steady_yaw_rate_mean_rad_s']-r1['steady_yaw_rate_mean_rad_s']),
            longest_joint_lateral_time_abs_difference_s=abs(r2['wheels'][0]['longest_joint_lateral_s']-r1['wheels'][0]['longest_joint_lateral_s']),
            two_steps_not_asymptotic_convergence_proof=True))
    g.write_json(a.result/'timestep_comparison.json',comparison)
    table=''
    for case in cases:
        r=next(r for r in rows if r['id']==case['id'] and r['dt_s']==.0025);slip=r['wheels'][0]
        table+='<tr>'+''.join('<td>'+value+'</td>' for value in [html.escape(case['title']),f"{r['nominal_demand_ratio']:.3f}",f"{r['steady_speed_mean_m_s']:.3f}",
            f"{r['steady_yaw_rate_mean_rad_s']:.3f} / {r['commanded_yaw_rate_rad_s']:.3f}",f"{r['steady_body_beta_deg_p50_p95'][0]:.1f}",
            f"{100*slip['steady_joint_lateral_fraction']:.0f}%",f"{slip['longest_joint_lateral_s']:.3f}",f"{100*slip['steady_joint_longitudinal_fraction']:.0f}%"] )+'</tr>'
    fiala=sources['ChFialaTire']['url'];tmeasy=sources['ChTMeasyTire']['url'];pac02=sources['ChPac02Tire']['url']
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Model-first saturation and sliding</title><style>body{font:17px system-ui;max-width:1550px;margin:35px auto;padding:0 24px;line-height:1.55}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:9px}pre{white-space:pre-wrap;background:#f1f3f4;padding:17px}.note{color:#9f2828}code{background:#f1f3f4}</style>
<h1>Model-first foundation: saturation and sliding from coupled dynamics</h1>
<p><strong>A minimum Coulomb stick/slip model already sustains saturation and lateral sliding for over2s under appropriate demand. A nonlinear empirical tire curve is not necessary to make those two conditions coexist.</strong> The result is a model-capability demonstration with a planar support fixture,not successful free-body TRON1 drift.</p>
<p>The prior MPPI audits show brief slip in one particular soft-contact model and task. They do not demonstrate that friction saturation and sustained sliding cannot coexist. The original50Hz reference has lateral COM acceleration demand at only5.8% /13.8% /16.3% of μg at median/p95/max for μ=0.7. This is a kinematic force-demand estimate,not measured tire force; it excludes leg actuation and variable contact loads. Large geometric β alone therefore does not establish an inertia-demanding friction-limit maneuver.</p>
<h2>Implemented coupled model</h2><p>Velocities are(vx_world,vy_world,yaw rate,omegaL,omegaR). Mass22.273kg,yaw inertia0.4035kgm²,wheel axial inertia0.009743kgm²,radius0.127m and track0.31114m come from the original TRON1 asset at its initial pose. Both motors use the original12Nm/100rad/s DC envelope. Fixed feedback gains command forward speed and yaw rate every20ms; torque clipping is recomputed each physical step. There is no prescribed tangent force,root velocity playback or forced lateral slip.</p>
<pre>u_free = u_k + h M^-1 S^T tau
A = Jt M^-1 Jt^T; b = Jt u_free
p* = argmin [ 0.5 p^T A p + b^T p ], subject to ||p_i|| <= mu Fn_i h
u_next = u_free + M^-1 Jt^T p*
v_slip,i = Jt_i u_next; Ft_i = p_i / h

Interior: v_slip = 0 and ||Ft|| <= mu Fn  (sticking / rolling)
Sliding: v_slip != 0 and Ft = -mu Fn v_slip/||v_slip||

M has BOTH body inertia and wheel-spin inertia; longitudinal and lateral force share one circle.</pre>
<p>The convex program is our fixed-normal tangential formulation of the documented <a href="https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/friction_contact.html">Coulomb stick/slip law</a>; it does not use the Siconos runtime and is not a general unknown-normal contact solver. A high-accuracy projected-KKT check controls the ADMM solve. Contact dissipated power and discrete kinetic-energy balance are checked independently.</p>
<p class="note">Vertical motion,roll,pitch and leg motion are removed by the fixture. Equal Fn=mg/2 is an assumption,not measured support. Camber is zero. This model does not test whether a free TRON1 can lean,keep contact,or sustain the same forces. Its circular hard-friction law also differs from the earlier MuJoCo soft pyramidal law.</p>
<h2>Four controlled capability tests</h2><p>Initial speed and nominal yaw-rate demand are extracted from the reference. The5x case scales both demands,as a time-compression proxy for a similar turning radius; it is not playback of the actual retargeted trajectory. Columns below use the2–4s late window; this is not a verified steady equilibrium. Original reference β is not a tracking target in this capability test.</p>
<table><tr><th>Case</th><th>Commanded v*r/(mu*g)</th><th>Actual COM speed [m/s]</th><th>Actual / commanded yaw rate [rad/s]</th><th>Median body β [deg]</th><th>Saturated lateral sliding,late window</th><th>Longest lateral joint [s]</th><th>Saturated longitudinal sliding,late window</th></tr>'''+table+'''</table>
<p>Both wheels have the same qualitative joint-sliding outcome. Pure longitudinal overdrive is not called drift. The high-speed case fails to achieve its requested yaw rate despite sustained saturation and slip; it proves model capability,not successful tracking. Each case runs at400 and800Hz. The results remain close under step halving; two steps alone do not establish asymptotic convergence.</p>
<img src="saturation_and_slip_4x1.png"><img src="friction_sections_4x1.png"><img src="planar_paths_4x1.png">
<h2>Which wheel/tire model should support the full-body study?</h2><p>The current external Magic Formula adapter is a steady dissipative combined-slip sensitivity curve. It has no static adhesion,tread memory,camber thrust or real TRON1 tire identification. Calling it nonlinear does not establish the needed physics. Relevant open-source implementations were read and pinned:</p>
<table><tr><th>Model</th><th>What it provides</th><th>Scope for independent wheel inclination</th></tr>
<tr><td>Coupled hard Coulomb graph,implemented here</td><td>Constitutive stick/slip and shared traction budget;body/wheel momentum</td><td>Capability baseline only;no camber thrust or tread elasticity</td></tr>
<tr><td><a href="'''+fiala+'''">Chrono Fiala</a></td><td>Brush-based handling law with combined-slip effects</td><td>Documented zero-camber assumption;not a direct choice for this camber-control task</td></tr>
<tr><td><a href="'''+tmeasy+'''">Chrono basic TMeasy</a></td><td>Low-speed Dahl bristle model and nonlinear combined-force curve</td><td>In the audited file,gamma enters overturning Mx;no explicit gamma-to-Fy term. Do not assume features of the commercial model.</td></tr>
<tr><td><a href="'''+pac02+'''">Chrono Pac02</a></td><td>Gamma-dependent lateral/longitudinal force terms</td><td>Current default gamma clamp3deg is adjustable,but highlights identification limits. Large-TRON1-camber extrapolation is not validated.</td></tr>
<tr><td><a href="https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Vehicles.html">PhysX Vehicle default tire</a></td><td>Combined force/load normalization with camber stiffness</td><td>Engineering force-law candidate;not automatically enabled for an articulated cylinder,not full Pacejka,and still needs coefficient calibration.</td></tr></table>
<p>Chrono is recorded as a source-reference submodule,not installed or used to simulate this bench. <a href="model_source_audit.json">Pinned source hashes and capability audit</a>. The <a href="https://api.projectchrono.org/wheeled_tire.html">official tire documentation</a> further distinguishes rigid,handling and deformable tire models.</p>
<h2>Next modeling layer before reference generation</h2><p>For independent signed camber,adopt a calibrated finite-compliance brush/contact-patch model,or a measured camber-aware tire law. Retain Coulomb saturation as the local traction constraint. Distinguish hub slip angle from actual tread-tip plastic sliding;normal load,contact footprint,rolling transport and wheel torque must affect the result. A generic camber force added to a cylinder without calibration is not sufficient.</p>
<pre>Whole-body: M(q) qdd + h(q,qd) = S^T tau + sum Jc(q)^T Fc
State: original q,qd (already includes wheel spins),plus tread/relaxation states for each wheel
Tangent displacement: relative base displacement = elastic tread deformation + plastic sliding
Plastic sliding branch: local tangent force reaches its identified traction bound
Contact pressure / normal load must come from the FULL contact/vertical model,not this fixture's mg/2.
CamberL and camberR are independent inputs through actual geometry and calibrated constitutive response.</pre>
<p>In an elastic tire,positive instantaneous wheel/contact power may release stored strain energy. The energy audit must include that storage instead of requiring all instantaneous power to be negative. Static peak friction and a post-peak dynamic plateau must also be kept distinct when normalizing saturation.</p>
<p>The full-body implementation must first pass below-limit adhesion/rolling,above-limit sustained lateral and combined slip,torque reversal,camber sweeps,load changes and loss/recovery of contact. It must avoid counting both native and external tangential forces. The existing one-step-delayed external-load coupling is a separate numerical assumption to validate,not a same-time contact solve.</p>
<p>Only after these tests should MPPI generate dynamically reachable speed,turning and independent-camber references. Any tread-memory state must propagate through the predictor and actual plant;zeroing it every20ms would change the physical model. The original task and controller comparison remain unpassed;this bench is a foundation for that work.</p>
<h2>Reproduction and evidence</h2><pre>git submodule update --init upstream/chrono   # source reference only;no build needed
python scripts/coupled_sliding_model_bench.py --output-dir results/coupled_model_reproduction
python scripts/report_coupled_sliding_model_bench.py --result results/coupled_model_reproduction</pre>
<p><a href="analysis.json">Eight fixed-parameter traces and model inputs</a> · <a href="timestep_comparison.json">Step comparison</a> · <a href="verification.json">Momentum,energy and KKT verification</a>. The existing pinned robot model is required only to extract parameters. No new engine,package,native build,GPU,policy training or MPPI search was used.</p></html>'''
    (a.result/'report.html').write_text(doc)
    manifest={p.name:g.sha(p) for p in a.result.iterdir() if p.is_file()}
    g.write_json(a.result/'report_provenance.json',dict(source_sha256=g.sha(__file__),analysis_sha256=g.sha(a.result/'analysis.json'),Chrono_submodule_commit=commit,artifacts=manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--result',type=Path,required=True);run(p.parse_args())
