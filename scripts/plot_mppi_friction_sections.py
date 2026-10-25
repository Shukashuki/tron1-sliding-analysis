"""Show actual same-point forces in the native normalized friction sections."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import generate_mppi_contact_reference as g
from analyze_mppi_friction_evidence import matched_load_fraction


def run(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    root=a.audit
    for name in ('friction_sections_4x1.png','index.html','sections_provenance.json'):
        if (root/name).exists():raise FileExistsError('Preserve completed figures')
    result=json.loads((root/'summary.json').read_text());cfg=result['config']
    fig,axes=plt.subplots(1,4,figsize=(23,6),layout='constrained');table='';input_hashes={};panel=0
    for case in result['rows']:
        point_path=root/case['id']/'contact_points.npz';wheel_path=root/case['id']/'wheel_evidence.npz'
        if g.sha(point_path)!=case['point_data_sha256'] or g.sha(wheel_path)!=case['wheel_data_sha256']:raise ValueError('Point data differs')
        input_hashes[case['id']]=dict(point_sha256=g.sha(point_path),wheel_sha256=g.sha(wheel_path))
        with np.load(point_path) as f:p={k:f[k].copy() for k in f.files}
        with np.load(wheel_path) as f:w={k:f[k].copy() for k in f.files}
        point_time=w['time_s'][p['sample_index'].astype(int)]
        _,_,joint=matched_load_fraction(p,len(w['time_s']),cfg)
        for side in (0,1):
            ax=axes[panel];panel+=1
            mask=(p['wheel_index']==side)&(p['Fn_n']>cfg['point_Fn_n_strict'])&(point_time>=1.44-1e-9)&(point_time<=4+1e-9)
            force=p['Ft_contact_n']/(p['mu_tangent']*p['Fn_n'][:,None])
            circle=np.linspace(0,2*np.pi,400);ax.plot(np.cos(circle),np.sin(circle),'--',color='#aaaaaa',label='Elliptic L2 boundary')
            diamond=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]])
            ax.plot(diamond[:,0],diamond[:,1],color='black',lw=1.8,label='Actual native pyramid')
            speed=np.maximum(abs(p['signed_lateral_m_s']),1e-3)
            scatter=ax.scatter(force[mask,0],force[mask,1],c=speed[mask],s=15,cmap='viridis',norm=LogNorm(vmin=1e-3,vmax=2),alpha=.65)
            accepted=mask&joint
            ax.scatter(force[accepted,0],force[accepted,1],facecolors='none',edgecolors='#d62728',s=27,lw=.65,label='Point passes joint gates')
            ax.set_aspect('equal');ax.set_xlim(-1.12,1.12);ax.set_ylim(-1.12,1.12);ax.grid(alpha=.2)
            ax.set_xlabel('Ft1 / (mu1 Fn)');ax.set_ylabel('Ft2 / (mu2 Fn)')
            name=('Left','Right')[side];ax.set_title(('MPPI' if case['id']=='independent_v1' else 'Smoothed MPPI')+' / '+name)
            row=case['rows'][side]
            ax.text(.02,.02,f"Same-point joint samples {row['same_point_force_corroborated_wheel_samples']}/{row['valid_supported_moving_wheel_samples']}\nLongest joint event {1000*row['longest_joint_corroborated_s']:.1f}ms",transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
            ax.legend(fontsize=7,loc='upper right')
            table+='<tr>'+''.join('<td>'+value+'</td>' for value in [html.escape(case['title']),name,
                f"{row['same_point_force_corroborated_wheel_samples']}/{row['valid_supported_moving_wheel_samples']}",
                f"{1000*row['longest_joint_corroborated_s']:.1f}",str(len(row['persistent_events']))])+'</tr>'
    fig.colorbar(scatter,ax=axes.tolist(),label='Absolute actual lateral contact-point speed [m/s]',shrink=.75)
    fig.suptitle('Native friction-pyramid cross-sections | turn core1.44-4.00s | Fn>5N | each point pairs its own force and velocity',fontsize=13)
    fig.savefig(root/'friction_sections_4x1.png',dpi=150);plt.close(fig)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Friction-cone corroboration of MPPI sliding</title><style>body{font:17px system-ui;max-width:1500px;margin:35px auto;padding:0 24px;line-height:1.6}img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}.note{color:#a32323}pre{background:#f3f4f5;padding:15px;white-space:pre-wrap}</style>
<h1>Does the measured sliding agree with the friction limit?</h1>
<p><strong>Some short slip episodes are supported by same-point force evidence. Neither MPPI attempt establishes sustained contact drift.</strong> Joint evidence requires continuous wheel support,measured lateral slip,proximity to the actual native friction boundary,and dissipative tangential force at the same points. Support continuity and full-body stability remain necessary for a successful demonstration.</p>
<img src="friction_sections_4x1.png">
<p>Black diamond: actual condim3 MuJoCo pyramidal friction boundary. Dashed circle: elliptic comparison only. Dots are actual loaded core contacts,color-coded by their own lateral speed; red rings pass the point-level joint gates. Axes use each contact's native tangent coordinates,not world XY. Point-level rings do not imply a sustained wheel-level event.</p>
<table><tr><th>Attempt</th><th>Wheel</th><th>Joint / valid supported moving wheel samples</th><th>Longest joint event [ms]</th><th>Events≥50ms</th></tr>'''+table+'''</table>
<p>Joint gates: Fn&gt;5N per eligible point,|vlateral|&gt;0.05m/s,native utilization≥0.9,friction dissipated power&gt;0,and dissipation efficiency≥0.5 of the maximum possible over the native diamond. Matched points must carry at least50% of eligible normal load; the wheel must have summed Fz&gt;5N and the base must be moving at≥0.25m/s. No contact gaps or alternating wheels are stitched into a continuous same-wheel event. The efficiency threshold is a diagnostic screen,not a MuJoCo contact-law equality.</p>
<pre>rho_native = |Ft1|/(mu1 Fn) + |Ft2|/(mu2 Fn)
Native cone margin = 1 - rho_native
D = -Ft dot vt
Dmax = Fn max(mu1 |vt1|,mu2 |vt2|)
eta = D / Dmax</pre>
<p>The forces and velocities are paired before aggregation. All2793 native states per attempt reproduce exactly; point velocities also match an independent Jacobian calculation. The source model and controllers are unchanged. Both original whole-body recordings lose posture.</p>
<p class="note">No event reaches50ms under the main screen. The declared sensitivity grid also finds no≥50ms event when varying slip thresholds0.02–0.1m/s and boundary thresholds0.8–0.95. This is a simulator-model consistency check,not physical tire-sensor validation. Soft-contact slip within the cone remains measured motion; it is reported separately from force-corroborated near-boundary slip.</p>
<p><a href="report.html">Force,velocity and support timelines;full method and thresholds</a> · <a href="summary.json">All numeric data and sensitivity results</a> · <a href="../mppi_contact_reference_20261025_v1/report.html">Original MPPI trajectories and videos</a> · <a href="https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones">MuJoCo model documentation</a></p>
<pre>python scripts/analyze_mppi_friction_evidence.py --output-dir results/mppi_friction_reproduction
python scripts/plot_mppi_friction_sections.py --audit results/mppi_friction_reproduction</pre></html>'''
    (root/'index.html').write_text(document)
    g.write_json(root/'sections_provenance.json',dict(kind='same_point_native_force_section_plot',source_sha256=g.sha(__file__),
        audit_summary_sha256=g.sha(root/'summary.json'),inputs=input_hashes,
        image_sha256=g.sha(root/'friction_sections_4x1.png'),index_sha256=g.sha(root/'index.html'),new_physics_or_controller_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--audit',type=Path,required=True);run(p.parse_args())
