"""Wheel-resultant Coulomb-cone diagnostics from existing audited PhysX data.

The cone is an analytical comparison, not a per-contact solver certificate.
No training, physics rerun, contact reassignment, or success-gate change.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_hockey_prefall_friction import read_case
from evaluate_hockey_physx import load, sha

MU = .7
API_SOURCE = 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_friction_data'
PATCH_SOURCE = 'https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#friction-and-restitution'


def cone_coordinates(longitudinal, lateral, normal, mu=MU, load_threshold=5.):
    fx, fy, fn = [np.asarray(v, dtype=float) for v in (longitudinal, lateral, normal)]
    if fx.shape != fy.shape or fx.shape != fn.shape or not all(np.isfinite(v).all() for v in (fx, fy, fn)):
        raise ValueError('Finite, matching force arrays required')
    if not np.isfinite(mu) or mu <= 0 or load_threshold < 0 or np.any(fn < -1e-6):
        raise ValueError('Invalid unilateral contact or friction coefficient')
    loaded = fn > load_threshold
    c = mu*fn
    u = np.full(fn.shape, np.nan); v = u.copy()
    np.divide(fx, c, out=u, where=loaded); np.divide(fy, c, out=v, where=loaded)
    rho = np.hypot(u, v)
    # A directional slice: keep Fn and longitudinal force fixed. Do not claim
    # all lateral controllability vanishes when increasing |Fy| is impossible.
    feasible_slice = loaded & (abs(fx) <= c+1e-4*np.maximum(c, 1.))
    bound = np.where(feasible_slice, np.sqrt(np.maximum(c*c-fx*fx, 0)), np.nan)
    return dict(loaded=loaded, capacity_N=c, u=u, v=v, rho=rho,
        cone_angle_deg=np.degrees(np.arctan(mu*rho)),
        radial_margin_N=np.where(loaded, c-np.hypot(fx, fy), np.nan),
        lateral_bound_fixed_longitudinal_N=bound,
        lateral_same_sign_headroom_N=bound-abs(fy),
        delta_lateral_min_N=-bound-fy, delta_lateral_max_N=bound-fy,
        outside_tolerance=loaded & (rho > 1.0001))


def window_mask(t, start, end):
    dt = float(np.median(np.diff(t)))
    return (t-dt >= start-1e-9) & (t <= end+1e-9) & (t > 0)


def quantiles(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(median=float(np.median(x)), p05=float(np.quantile(x,.05)),
                p95=float(np.quantile(x,.95)), maximum=float(np.max(x))) if len(x) else None


def summarize(data, mask):
    s, c = data['signals'], data['cone']
    dt = float(np.median(np.diff(s['time_s'])))
    rows = []
    for k, side in enumerate(('left', 'right')):
        m = mask & c['loaded'][:,k]
        row = dict(side=side, window_samples=int(mask.sum()), loaded_samples=int(m.sum()),
            unloaded_samples=int((mask & ~c['loaded'][:,k]).sum()),
            capacity_all_samples_N=quantiles(c['capacity_N'][mask,k]),
            abs_u_loaded=quantiles(abs(c['u'][m,k])), abs_v_loaded=quantiles(abs(c['v'][m,k])),
            near_limit_loaded_fraction=float(s['near_limit'][m,k].mean()) if m.any() else None,
            outside_tolerance_samples=int(c['outside_tolerance'][mask,k].sum()),
            lateral_slip_s=float(s['corroborated_lateral_slip'][mask,k].sum()*dt),
            tangential_slip_s=float(s['corroborated_tangent_slip'][mask,k].sum()*dt),
            anchor_sum_rho_loaded=quantiles(s['anchor_force_sum_utilization'][m,k]))
        for key in ('rho','cone_angle_deg','radial_margin_N','lateral_same_sign_headroom_N',
                    'delta_lateral_min_N','delta_lateral_max_N'):
            row[key+'_loaded'] = quantiles(c[key][m,k])
        rows.append(row)
    return rows


def contact_topology(cell, case_id, data, mask):
    raw = load(cell/'eval-suite'/case_id/'native_contact_buffers.npz')
    result = []
    for k, side in enumerate(('left','right')):
        m = mask[1:] & data['cone']['loaded'][1:,k]
        normal = raw[f'w{k}_normal_buffer_4'].reshape(-1)
        anchor = raw[f'w{k}_friction_buffer_2'].reshape(-1)
        pairs, counts = np.unique(np.column_stack((normal[m],anchor[m])), axis=0, return_counts=True)
        result.append(dict(side=side, record_count_histogram=[dict(normal_points=int(p[0]),
            friction_anchors=int(p[1]), physical_samples=int(n)) for p,n in zip(pairs,counts)],
            matching_counts_prove_force_correspondence=False,
            raw_patch_id_or_anchor_normal_allocation_available=False))
    return result


def sections(original, hockey, start, end, path, title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1,4,figsize=(21,5.1),constrained_layout=True)
    norm = Normalize(start,end)
    phi = np.linspace(0,2*np.pi,361)
    for ax, (data,k,label) in zip(axes, ((original,0,'Original / left'),(original,1,'Original / right'),
                                        (hockey,0,'Hockey / left'),(hockey,1,'Hockey / right'))):
        c = data['cone']; t=data['signals']['time_s']; mask=window_mask(t,start,end); m=mask&c['loaded'][:,k]
        ax.fill(np.cos(phi),np.sin(phi),color='#f1f1ed',zorder=0)
        ax.plot(np.cos(phi),np.sin(phi),color='black',lw=1,label='Ideal boundary: rho=1')
        ax.plot(.9*np.cos(phi),.9*np.sin(phi),color='#a7986b',lw=.8,ls='--',label='rho=0.9')
        points=np.column_stack((c['u'][:,k],c['v'][:,k]))
        adjacent=np.flatnonzero(m[:-1]&m[1:])
        if len(adjacent):
            lines=np.stack((points[adjacent],points[adjacent+1]),axis=1)
            artist=LineCollection(lines,cmap='viridis',norm=norm,linewidths=.65,alpha=.65)
            artist.set_array(t[adjacent+1]);ax.add_collection(artist)
        ax.scatter(points[m,0],points[m,1],c=t[m],cmap='viridis',norm=norm,s=13,zorder=3)
        if m.any():
            last=np.flatnonzero(m)[-1];ax.scatter(*points[last],marker='x',color='#bb3030',s=65,zorder=4)
        else:
            ax.text(0,0,'No loaded samples\nUtilization undefined',ha='center',va='center',fontsize=10)
        ax.text(.02,.02,f'Loaded: {m.sum()}/{mask.sum()} samples',transform=ax.transAxes,fontsize=8)
        ax.set(title=label,xlabel='u = longitudinal force / (mu Fn)',ylabel='v = lateral force / (mu Fn)',
               xlim=(-1.15,1.15),ylim=(-1.15,1.15),aspect='equal')
        ax.axhline(0,color='#bbb',lw=.5);ax.axvline(0,color='#bbb',lw=.5);ax.grid(alpha=.15)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axes,shrink=.75,label='Time [s]')
    fig.suptitle(title+'\nWheel-resultant analytical cone; no lines across lost contact; red x = last loaded sample')
    fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)


def cones_3d(data_by_case, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    selected=[]
    for case_id, data in data_by_case.items():
        for name in ('original','hockey'):
            d=data[name];s=d['signals'];c=d['cone'];end=data['end'];m=window_mask(s['time_s'],end-.1,end)&c['loaded'][:,1]
            selected.append((case_id,name,s,c,m,end))
    top=max(float(s['normal_fz'][m,1].max()) for _,_,s,_,m,_ in selected)*1.1
    phi=np.linspace(0,2*np.pi,49);height=np.linspace(0,top,12)
    phi,height=np.meshgrid(phi,height);x=MU*height*np.cos(phi);y=MU*height*np.sin(phi)
    fig=plt.figure(figsize=(24,6));axes=[]
    grid=fig.add_gridspec(1,4,left=.02,right=.92,bottom=.12,top=.82,wspace=.12)
    for i,(case_id,name,s,c,m,end) in enumerate(selected):
        ax=fig.add_subplot(grid[0,i],projection='3d');axes.append(ax)
        ax.set_proj_type('ortho');ax.grid(False)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.set_alpha(0.)
        ax.plot_surface(x,y,height,color='#b5c9db',alpha=.08,linewidth=0,shade=False)
        ax.plot_wireframe(x,y,height,rstride=3,cstride=6,color='#6e91ac',lw=.6,alpha=.5)
        phase=(s['time_s'][m]-(end-.1))/.1
        ax.scatter(s['longitudinal_force_N'][m,1],s['lateral_force_N'][m,1],s['normal_fz'][m,1],
                   c=phase,cmap='viridis',norm=Normalize(0,1),s=28,depthshade=False)
        ax.set(title=f'{name.title()} / {case_id} / right',xlabel='Longitudinal force [N]',
            ylabel='Lateral force [N]',zlabel='Normal force [N]',xlim=(-MU*top,MU*top),ylim=(-MU*top,MU*top),zlim=(0,top))
        ax.set_box_aspect((2*MU,2*MU,1));ax.view_init(elev=23,azim=-55)
        ax.tick_params(labelsize=7,pad=0)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.label.set_size(8)
    colorbar_axis=fig.add_axes([.94,.24,.009,.45])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='viridis'),cax=colorbar_axis,label='Progress through final pre-fall 100 ms')
    fig.suptitle('Force-space Coulomb cones, mu=0.7, half-angle 34.99 degrees | equal axes across all four panels\n'
                 'Same absolute time windows per case; left original wheel is unloaded; this is not a robot-space contact drawing')
    fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original-cell','comparison-cell','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import matplotlib
    matplotlib.use('Agg')
    result=dict(mu=MU,ideal_cone_half_angle_deg=float(np.degrees(np.arctan(MU))),
        coordinate_definition='u=F_long/(mu Fn); v=F_lat/(mu Fn); ideal admissibility u^2+v^2<=1, Fn>=0',
        load_gate_N=5.,numerical_outside_threshold_rho=1.0001,existing_near_limit_screen=[.9,1.05],
        pointwise_cone_certified=False,active_solver_friction_flags_verified=False,
        directional_headroom='At fixed Fn and F_long: B=sqrt((mu Fn)^2-F_long^2); delta F_lat in [-B-F_lat,B-F_lat]. Same-sign headroom B-|F_lat| is not all-direction controllability.',
        source_sha256=sha(__file__),capacity_helper_sha256=sha(Path(__file__).with_name('analyze_hockey_prefall_friction.py')),
        official_sources=[API_SOURCE,PATCH_SOURCE],physics_steps_performed=0,success_gates_changed=False,cases={})
    a.output.mkdir(parents=True);data_by_case={}
    for case_id in ('nominal','trial_022'):
        original,hockey=[read_case(p,case_id) for p in (a.original_cell,a.comparison_cell)]
        for key in ('seed','speed_scale','case_id','motion_sha256'):
            if original['run'][key]!=hockey['run'][key]:raise ValueError('Mismatched policy comparison')
        if (original['run']['method'],hockey['run']['method'],original['run']['seed'],original['run']['speed_scale'])!=('beyondmimic','hockey',42,1.):
            raise ValueError('Requires the completed original-speed seed-42 pair')
        if original['qualification']['case']!=hockey['qualification']['case']:raise ValueError('Unpaired initial state')
        for d in (original,hockey):
            s=d['signals'];d['cone']=cone_coordinates(s['longitudinal_force_N'],s['lateral_force_N'],s['normal_fz'])
            if not np.allclose(d['cone']['rho'][s['loaded']],s['rho'][s['loaded']],atol=1e-10):raise ValueError('Cone radius disagrees with native ratio')
        t=original['signals']['time_s'];dt=float(np.median(np.diff(t)));fall=original['qualification']['first_45deg_tilt_s'];end=fall-dt
        data_by_case[case_id]=dict(original=original,hockey=hockey,end=end)
        intervals=dict(braking=(.4,1.8),last_500ms=(end-.5,end),last_100ms=(end-.1,end),full_prefall=(.02,end))
        row=dict(first_45deg_tilt_s=fall,last_pre_fall_sample_s=end,windows={},original_provenance=original['provenance'],hockey_provenance=hockey['provenance'])
        for window,(start,stop) in intervals.items():
            m=window_mask(t,start,stop)
            row['windows'][window]=dict(interval_s=[float(start),float(stop)],samples=int(m.sum()),
                original=summarize(original,m),hockey=summarize(hockey,m))
            if window in ('braking','last_500ms','last_100ms'):
                sections(original,hockey,start,stop,a.output/f'{case_id}.{window}_cone_sections_4x1.png',
                         f'{case_id} | {window.replace("_"," ")} | x1 seed42 | mu=0.7 | {start:.4f}-{stop:.4f} s')
        m=window_mask(t,.02,end)
        row['record_topology']={name:contact_topology(cell,case_id,d,m) for name,cell,d in
            (('original',a.original_cell,original),('hockey',a.comparison_cell,hockey))}
        if row['windows']['last_100ms']['samples']!=40:raise ValueError('Wrong pre-fall sample window')
        result['cases'][case_id]=row
        fields=['policy','time_s','side','normal_N','capacity_N','longitudinal_N','lateral_N','loaded',
            'u','v','rho','cone_angle_deg','radial_margin_N','lateral_same_sign_headroom_N','delta_lateral_min_N','delta_lateral_max_N',
            'outside_tolerance','corroborated_lateral_slip','corroborated_tangent_slip']
        with (a.output/f'{case_id}.cone_samples.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for name,d in (('original',original),('hockey',hockey)):
                s=d['signals'];c=d['cone']
                for i in np.flatnonzero(m):
                    for k,side in enumerate(('left','right')):
                        r=dict(policy=name,time_s=float(t[i]),side=side,normal_N=float(s['normal_fz'][i,k]),
                            capacity_N=float(c['capacity_N'][i,k]),longitudinal_N=float(s['longitudinal_force_N'][i,k]),
                            lateral_N=float(s['lateral_force_N'][i,k]))
                        for key in fields[7:17]:r[key]=c[key][i,k].item()
                        for key in fields[17:]:r[key]=bool(s[key][i,k])
                        writer.writerow({k:('' if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items()})
    cones_3d(data_by_case,a.output/'right_wheel_force_space_cones_4x1.png')
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),cases=list(result['cases']))))


if __name__=='__main__':main()
