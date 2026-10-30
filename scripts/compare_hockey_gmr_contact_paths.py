"""Compare GMR geometry, the designed training reference and native PhysX contacts.

No retargeting, pose correction or physics is performed. The raw GMR panel
receives one declared XY/yaw display transform only. Airborne projections are
kept separate from geometric near-plane points and observed native contacts.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_hockey_physx import load,q_matrix,sha
from audit_hockey_reference_contact_gap import wheel_geometry
from wheel_ground_geometry import cylinder_support


TOL=.0001


def bottom_locus(center,axis,radius,half_width,tolerance=TOL):
    """Near-minimum bottom generatrix, avoiding arbitrary edge picks when upright."""
    if tolerance<0:raise ValueError('Nonnegative geometric tolerance required')
    answer=cylinder_support(center,axis,radius,half_width)
    az=axis[...,2];radial=np.array([0.,0.,1.])-az[...,None]*axis
    norm=np.linalg.norm(radial,axis=-1)
    if (norm<1e-8).any():raise ValueError('Vertical wheel axle has no unique bottom generatrix')
    radial_center=center-radius*radial/norm[...,None]
    extent=np.divide(tolerance,np.abs(az),out=np.full_like(az,np.inf),where=np.abs(az)>1e-12)
    lo=np.where(az<0,np.maximum(-half_width,half_width-extent),-half_width)
    hi=np.where(az>0,np.minimum(half_width,-half_width+extent),half_width)
    segment=np.stack((radial_center+lo[...,None]*axis,radial_center+hi[...,None]*axis),axis=-2)
    return answer['authored_lowest_point_w_m'],answer['authored_signed_gap_m'],segment


def point_segment_distance(points,segment):
    """Unsigned XY distance to the specified finite footprint segment."""
    a,b=np.asarray(segment,dtype=float);v=b-a;den=float(v@v)
    alpha=np.clip((np.asarray(points)-a)@v/den,0,1) if den>1e-20 else np.zeros(len(points))
    return np.linalg.norm(np.asarray(points)-(a+alpha[:,None]*v),axis=-1)


def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return None
    return dict(count=len(x),RMSE=float(np.sqrt(np.mean(x*x))),median=float(np.median(x)),maximum=float(x.max()))


def native_points_at(trace,raw,row,side):
    count=int(raw[f'w{side}_normal_buffer_4'][row].item());start=int(raw[f'w{side}_normal_buffer_5'][row].item())
    ids=np.arange(start,start+count)
    normal=raw[f'w{side}_normal_buffer_2'][row,ids].astype(float)
    gap=raw[f'w{side}_normal_buffer_3'][row,ids].reshape(-1).astype(float)
    normals=np.linalg.norm(normal,axis=-1)>1e-8;valid=normals&(gap<=TOL)
    points=raw[f'w{side}_normal_buffer_1'][row,ids[valid]].astype(float)-trace['environment_origin_w']
    bid=trace['body_names'].astype(str).tolist().index('wheel_'+('L','R')[side]+'_Link')
    velocity=trace['body_com_vel'][row,bid]+np.cross(trace['body_omega'][row,bid],points-trace['body_com_pos'][row,bid])
    unit=normal[valid]/np.linalg.norm(normal[valid],axis=-1,keepdims=True)
    velocity-=(velocity*unit).sum(-1,keepdims=True)*unit
    minimum_gap=float(gap[normals].min()) if normals.any() else None
    return points,gap[valid],velocity,minimum_gap


def geometric_case(label,t,center,axis,root,forward,radius,width):
    point,gap,segment=bottom_locus(center,axis,radius,width)
    return dict(label=label,time=t,center=center,axis=axis,root=root,forward=forward,
        point=point,gap=gap,segment=segment,touch=np.abs(gap)<=TOL,penetration=gap < -TOL,
        native=False,points=np.empty((0,10)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','gmr','model','geometry-config','selection','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda path:json.loads(path.read_text())
    meta=read(a.reference/'report.json');gmeta=read(a.gmr/'report.json');geom=read(a.geometry_config)['geometry']
    if sha(a.gmr/'retarget_trace.npz')!=meta['retarget_sha256']:raise ValueError('GMR lineage mismatch')
    if sha(a.reference/'motion.npz')!=meta['motion_sha256'] or sha(a.model)!=meta['model_sha256']:raise ValueError('Reference/model changed')
    ref=load(a.reference/'motion.npz');gmr=load(a.gmr/'retarget_trace.npz');rt=np.arange(len(ref['joint_pos']))/float(ref['fps'])
    names=ref['body_names'].astype(str).tolist();base=names.index('base_Link')
    center,axis,_=wheel_geometry(ref,'body_pos_w','body_quat_w',geom)
    reference=geometric_case('B  Training reference | 50 mm wheels',rt,center,axis,ref['body_pos_w'][:,base],q_matrix(ref['body_quat_w'][:,base])[...,:,0],geom['radius_m'],geom['half_width_m'])
    # Forward kinematics only, using the unchanged original GMR model.
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(a.model.resolve()));data=mujoco.MjData(model)
    gids=[model.geom('wheel_'+s+'_collision').id for s in ('L','R')]
    bids=[model.body('wheel_'+s+'_Link').id for s in ('L','R')]
    c=[];axes=[];body_p=[];body_q=[]
    for pose in gmr['qpos']:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        c.append(data.geom_xpos[gids].copy());axes.append(data.geom_xmat[gids].reshape(2,3,3)[...,:,2].copy())
        body_p.append(data.xpos[bids].copy());body_q.append(data.xquat[bids].copy())
    c=np.array(c);axes=np.array(axes);gforward=q_matrix(gmr['qpos'][:,3:7])[...,:,0]
    yaw=np.arctan2(gforward[0,1],gforward[0,0]);alignment=Rotation.from_euler('z',-yaw).as_matrix()
    offset=reference['root'][0,:2]-(alignment@gmr['qpos'][0,:3])[:2]
    def align(x,translate=False):
        value=np.asarray(x)@alignment.T
        if translate:value[...,:2]+=offset
        return value
    # No z translation and no scaling, no time warp and no floor snapping.
    widths=model.geom_size[gids,1];radii=model.geom_size[gids,0]
    np.testing.assert_allclose(widths,widths[0]);np.testing.assert_allclose(radii,radii[0])
    raw_case=geometric_case('A  Source GMR | original 10 mm wheels',gmr['time_s'],align(c,True),align(axes),align(gmr['qpos'][:,:3],True),align(gforward),float(radii[0]),float(widths[0]))
    authored_axis=q_matrix(np.array(body_q))[...,:,1]
    authored_center=np.array(body_p)+authored_axis*np.array(geom['center_y_in_link_m'])[None,:,None]
    _,authored_raw_gap,_=bottom_locus(authored_center,authored_axis,geom['radius_m'],geom['half_width_m'])
    cases=[raw_case,reference];summaries=[];recordings=[]
    sources=read(a.selection)
    if len(sources)!=2:raise ValueError('Two policies at the same material required')
    for index,source in enumerate(sources):
        bundle=(a.selection.parent/source['bundle']).resolve();cell,stage=source['case'].split(':')
        entry=next(row for row in read(bundle/'plan.json')['cells'] if row['id']==cell)
        protocol=bundle/entry['protocol'];cfg=read(protocol)
        directory=(a.selection.parent/source['evidence']/source['run_name']/cell/stage).resolve()
        run=read(directory/'report.json');qualified=read(directory.parent/'nominal_qualification.json');q=qualified['qualification']
        if (qualified['run_report_sha256']!=sha(directory/'report.json') or qualified['protocol_sha256']!=sha(protocol)
                or qualified['source_manifest_sha256']!=sha(bundle/'manifest.json') or run['motion_sha256']!=meta['motion_sha256']
                or run['asset_sha256']!=meta['physics_asset_sha256']):raise ValueError('Recording provenance differs')
        if not np.isclose(cfg['evaluation_mu'],.7) or not np.isclose(cfg['evaluation_mu']*cfg['static_dynamic_friction_ratio'],.9):raise ValueError('Use material 0.9/0.7')
        for file,digest in run['artifacts'].items():
            if sha(directory/file)!=digest:raise ValueError('Recording changed')
        tr=load(directory/'final_rollout.npz');raw=load(directory/'native_contact_buffers.npz');dt=float(tr['physics_dt_s'])
        indices=np.rint(rt/dt).astype(int);np.testing.assert_allclose(indices*dt,rt,atol=1e-10,rtol=0)
        full_center,full_axis,full_gap=wheel_geometry(tr,'body_pos','body_quat',geom)
        case=geometric_case(('C  Original PPO' if index==0 else 'D  Split-friction PPO')+' | 0.9/0.7',rt,full_center[indices],full_axis[indices],tr['root_pos'][indices],q_matrix(tr['root_quat'][indices])[...,:,0],geom['radius_m'],geom['half_width_m'])
        allpoints=[];touch=np.zeros((len(rt),2),bool);error=np.full((len(rt),2),np.nan);gap_error=[];centroid=np.full((len(rt),2,3),np.nan)
        for frame,row in enumerate(indices):
            if int(raw['physical_sample_indices'][row])!=row+1:raise ValueError('Native geometry is not aligned to pre-step pose')
            for side in (0,1):
                pts,gap,v,minimum_gap=native_points_at(tr,raw,row,side)
                if minimum_gap is not None:gap_error.append(abs(minimum_gap-full_gap[row,side]))
                touch[frame,side]=len(pts)>0
                if not len(pts):continue
                centroid[frame,side]=pts.mean(0)
                distance=point_segment_distance(pts[:,:2],reference['segment'][frame,side,:,:2]);error[frame,side]=np.sqrt(np.mean(distance**2))
                for k,point in enumerate(pts):allpoints.append([rt[frame],frame,side,*point,gap[k],*v[k]])
        if max(gap_error)>1e-5:raise ValueError('Authored support and native separation disagree')
        dense=[]
        for row in range(indices[-1]+1):
            for side in (0,1):
                pts,gap,v,_=native_points_at(tr,raw,row,side)
                for k,point in enumerate(pts):dense.append([row*dt,row,side,*point,gap[k],*v[k]])
        case.update(native=True,points=np.array(allpoints),dense_points=np.array(dense),touch=touch,centroid=centroid,footprint_error=error)
        cases.append(case);brake=(rt>=.4-1e-9)&(rt<=1.8+1e-9)
        wheels=[]
        for side in (0,1):
            overlap=touch[:,side]&reference['touch'][:,side];values=np.linalg.norm(case['center'][:,side,:2]-reference['center'][:,side,:2],axis=-1)
            wheels.append(dict(side=('L','R')[side],wheel_center_XY_error_m=stats(values),
                native_to_potential_reference_footprint_XY_distance_m=stats(error[:,side]),
                shared_reference_and_native_contact_frames=int(overlap.sum()),
                native_to_touching_reference_footprint_XY_distance_m=stats(error[overlap,side]),
                braking_potential_footprint_XY_distance_m=stats(error[brake,side]),
                native_contact_frames=int(touch[:,side].sum())))
        summaries.append(dict(label=case['label'],wheels=wheels,bilateral_contact_frames=int(touch.all(-1).sum()),
            native_geometry_alignment_max_error_m=max(gap_error),full_400Hz_bilateral_interruption_s=q['primary_contact_gate']['bilateral_interruption']['total_s'],
            full_task_success=q['task_success_claimed'],qualification_sha256=sha(directory.parent/'nominal_qualification.json')))
        recordings.append(dict(source=source,source_manifest_sha256=sha(bundle/'manifest.json'),protocol_sha256=sha(protocol),run_report_sha256=sha(directory/'report.json'),artifacts=run['artifacts']))
    a.output.mkdir(parents=True);arrays={}
    for i,case in enumerate(cases):
        for key,value in case.items():
            if isinstance(value,np.ndarray):arrays[f'case{i}_{key}']=value
    arrays['source_GMR_gap_under_authored_50mm_geometry']=authored_raw_gap
    np.savez_compressed(a.output/'signals.npz',**arrays)
    plot_and_render(cases,a.output,a.render)
    geometric=[]
    for case in cases[:2]:
        geometric.append(dict(label=case['label'],frames=len(case['time']),duration_s=float(case['time'][-1]),
            near_plane_frames_per_wheel=case['touch'].sum(0).tolist(),bilateral_near_plane_frames=int(case['touch'].all(-1).sum()),
            penetration_frames_per_wheel=case['penetration'].sum(0).tolist(),gap_range_mm=[(case['gap'].min(0)*1000).tolist(),(case['gap'].max(0)*1000).tolist()]))
    result=dict(kind='hockey_GMR_reference_native_contact_path_comparison_v1',script_sha256=sha(__file__),
        inputs=dict(reference=sha(a.reference/'motion.npz'),reference_report=sha(a.reference/'report.json'),gmr=sha(a.gmr/'retarget_trace.npz'),gmr_report=sha(a.gmr/'report.json'),model=sha(a.model),geometry=sha(a.geometry_config),selection=sha(a.selection)),
        dependencies={name:sha(Path(__file__).with_name(name)) for name in ('evaluate_hockey_physx.py','audit_hockey_reference_contact_gap.py','wheel_ground_geometry.py')},
        raw_GMR_display_transform=dict(yaw_rad=-float(yaw),translation_XY_m=offset.tolist(),translation_z_m=0.,scale=1.,time_warp=False),
        raw_GMR_geometry=dict(radius_m=float(radii[0]),width_m=float(widths[0]*2),collider_center_in_link_m=model.geom_pos[gids].tolist(),
            archived_source_already_has_one_wheel_height_projection=gmeta['per_frame_anywheel_support_height_projection']),
        raw_GMR_under_authored_geometry=dict(gap_range_mm=[(authored_raw_gap.min(0)*1000).tolist(),(authored_raw_gap.max(0)*1000).tolist()],
            near_plane_frames_per_wheel=(abs(authored_raw_gap)<=TOL).sum(0).tolist(),penetration_frames_per_wheel=(authored_raw_gap < -TOL).sum(0).tolist()),
        reference_lineage=dict(source_world_XY_used=meta['source_world_XY_used'],initial_posture_blend_s=meta['initial_posture_blend_s'],
            one_wheel_ground_projection=meta['one_wheel_ground_projection'],wheel_spin_reference=meta['wheel_spin_reference'],
            source_global_metric_trajectory_verified=bool(gmr['metric_world_trajectory_verified'])),
        geometric_cases=geometric,recorded_cases=summaries,recordings=recordings,contact_tolerance_m=TOL,
        comparison_clock_hz=50,reference_frames=len(rt),native_contact_clock_hz=400,native_footprint_background_hz=400,physics_steps=0,new_retargeting=False,
        footprint_metric='Per-frame RMS distance of all native normal-contact XY points to the reference bottom generatrix within 0.1 mm of its lowest height; frames weighted equally.',
        limitations=['Raw GMR XY is not the designed training path; its rigid display alignment does not make it ground truth.',
            'The source GMR model has 10 mm wheels; the designed reference and recordings use authored 50 mm cylinders.',
            'Airborne footprint XY is only a projection; conditional contact errors require the reference to be within +/-0.1 mm of the floor.',
            'Reference spin is a placeholder and reference forces do not exist; no measured reference sliding or friction saturation is claimed.',
            'Reference contact counts use 50 Hz frames, while native task gates inspect 400 Hz samples; counts are not interchangeable.',
            'A migrating geometric minimum or native manifold centroid is not a tracked material point and its derivative is not slip velocity.'],
        artifact_sha256={x.name:sha(x) for x in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(geometry=geometric,policies=summaries)))


def plot_and_render(cases,output,render):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    colors=('#2779a5','#d37527');allxy=np.concatenate([case['point'][...,:2].reshape(-1,2) for case in cases])
    bounds=[allxy.min(0)-.10,allxy.max(0)+.10]
    def setup():
        fig,axs=plt.subplots(1,4,figsize=(24,3.4),constrained_layout=True)
        for case,ax in zip(cases,axs):
            for side,color in enumerate(colors):
                point=case['point'][:,side]
                if case['native']:
                    rp=cases[1]['point'][:,side];ax.plot(rp[:,0],rp[:,1],color=color,ls=':',alpha=.35,lw=1)
                    pp=case['dense_points'][case['dense_points'][:,2]==side];ax.scatter(pp[:,3],pp[:,4],s=3,color=color,alpha=.5,label=('L','R')[side]+' native touch (400 Hz)')
                else:
                    ax.plot(point[:,0],point[:,1],ls=':',color=color,alpha=.3,label=('L','R')[side]+' potential footprint')
                    touch=case['touch'][:,side];ax.scatter(point[touch,0],point[touch,1],s=20,color=color,label=('L','R')[side]+' near plane')
                    below=case['penetration'][:,side];ax.scatter(point[below,0],point[below,1],s=12,marker='x',color='crimson',alpha=.5)
            ax.plot(case['root'][:,0],case['root'][:,1],color='black',lw=.8,alpha=.5,label='Body path')
            ax.set(title=case['label'],xlabel='World / aligned X [m]',ylabel='Y [m]',xlim=(bounds[0][0],bounds[1][0]),ylim=(bounds[0][1],bounds[1][1]),aspect='equal')
            ax.grid(alpha=.2);ax.legend(fontsize=6,loc='upper left')
        return fig,axs
    fig,axs=setup();fig.suptitle('Hockey stop: potential wheel footprints versus actual touching points | Shared XY scale\nDotted: geometric projection (may be airborne); colored dots: touching; red x: geometric penetration; raw GMR receives XY/yaw display alignment only')
    fig.savefig(output/'contact_paths_4x1.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(24,4.7),constrained_layout=True)
    for case,ax in zip(cases,axs):
        for side,color in enumerate(colors):ax.plot(case['time'],case['gap'][:,side]*1000,color=color,label=('L','R')[side])
        ax.axhspan(-.1,.1,color='green',alpha=.2);ax.axvspan(.4,1.8,color='gray',alpha=.07)
        ax.set(title=case['label'],xlim=(0,2.76),ylim=(-3,24),xlabel='Time [s]',ylabel='Signed cylinder-ground clearance [mm]');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Identical clearance scale | Geometry is unchanged | Raw GMR: 30 Hz; reference: 50 Hz; PhysX sampled here at 50 Hz\nNative contact acceptance comes from the recorded manifold; geometric reference near-plane tolerance is +/-0.1 mm')
    fig.savefig(output/'contact_clearance_4x1.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(24,4.5),constrained_layout=True)
    for k,case in enumerate(cases[2:]):
        for side,color in enumerate(colors):
            ax=axs[2*k+side];error=case['footprint_error'][:,side]*100
            ax.plot(case['time'],error,color=color,label='Potential footprint distance')
            mask=cases[1]['touch'][:,side]&case['touch'][:,side]
            ax.scatter(case['time'][mask],error[mask],s=12,color='black',label='Reference also near plane')
            ax.set(title=case['label']+' | '+('Left','Right')[side],xlabel='Time [s]',ylabel='Native points to reference footprint [cm]',xlim=(0,2.76));ax.grid(alpha=.2);ax.legend(fontsize=7)
    ymax=max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:ax.set_ylim(0,ymax)
    fig.suptitle('Time-aligned footprint distances at 50 Hz | No alignment, translation or time warp applied to either PhysX rollout\nColored line includes airborne reference projections; black dots identify geometrically comparable contact frames')
    fig.savefig(output/'contact_path_error_4x1.png',dpi=140);plt.close(fig)
    if not render:return
    import cv2,imageio_ffmpeg
    fig,axs=setup();fig.suptitle('GMR, training reference and PhysX contacts | Fixed world view | 4x slow\nBlack arrow: body heading; colored arrows: wheel rolling direction; red vectors: native contact material velocity (0.03 m per m/s)')
    artists=[]
    for case,ax in zip(cases,axs):
        body=FancyArrowPatch((0,0),(0,0),arrowstyle='-|>',mutation_scale=9,color='black',lw=2,animated=True);ax.add_patch(body)
        wheel=[]
        for color in colors:
            arrow=FancyArrowPatch((0,0),(0,0),arrowstyle='-|>',mutation_scale=8,color=color,lw=2,animated=True);ax.add_patch(arrow);wheel.append(arrow)
        dots=ax.scatter([],[],s=30,c='white',edgecolors='black',linewidths=.7,animated=True)
        slip=LineCollection([],colors='crimson',linewidths=1.8,animated=True);ax.add_collection(slip)
        stamp=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=7,animated=True,bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
        artists.append((body,wheel,slip,stamp,dots))
    fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox);width,height=fig.canvas.get_width_height();path=output/'contact_paths_4x1_4x.mp4'
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for frame,t in enumerate(cases[1]['time']):
            fig.canvas.restore_region(background)
            for case,ax,(body,wheel,slip,stamp,dots) in zip(cases,axs,artists):
                i=min(len(case['time'])-1,max(0,int(np.searchsorted(case['time'],t+1e-9,side='right')-1)))
                ended=t>case['time'][-1]+1e-8;root=case['root'][i,:2];f=case['forward'][i,:2];f=f/max(np.linalg.norm(f),1e-12)
                body.set_positions(root,root+.12*f)
                for side,line in enumerate(wheel):
                    pt=case['point'][i,side,:2];axis=case['axis'][i,side];heading=np.array([axis[1],-axis[0]]);heading/=max(np.linalg.norm(heading),1e-12)
                    # Original MJCF cylinder axial sign is arbitrary; orient it with the wheel-link rolling direction.
                    if case is cases[0]:heading=-heading
                    line.set_positions(pt,pt+.08*heading)
                pts=case['points'][case['points'][:,1]==i] if case['native'] else np.empty((0,10))
                dots.set_offsets(pts[:,3:5] if case['native'] else case['point'][i,case['touch'][i],:2])
                slip.set_segments(np.stack((pts[:,3:5],pts[:,3:5]+.03*pts[:,7:9]),axis=1))
                state='source ended; last pose held' if ended else ('native touch' if case['native'] else 'geometric near plane')+f" L/R={int(case['touch'][i,0])}/{int(case['touch'][i,1])}"
                stamp.set_text(f't={t:.2f}s | source sample={case["time"][i]:.3f}s\n{state}')
                for artist in (body,*wheel,slip,dots,stamp):ax.draw_artist(artist)
            pixels=np.asarray(fig.canvas.buffer_rgba())
            for _ in range(4):writer.stdin.write(pixels.tobytes())
            if frame in (20,50,80):cv2.imwrite(str(output/f'contact_paths_{t:.2f}s.png'),cv2.cvtColor(pixels,cv2.COLOR_RGBA2BGR))
    finally:writer.stdin.close();code=writer.wait();plt.close(fig)
    if code:raise RuntimeError('Encoder failed')
    cap=cv2.VideoCapture(str(path));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        if im.shape[:2]!=(height,width):raise ValueError('Invalid decoded size')
        count+=1
    fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if count!=len(cases[1]['time'])*4 or fps!=50:raise ValueError('Incomplete movie')
    (output/'render_provenance.json').write_text(json.dumps(dict(video=path.name,sha256=sha(path),frames=count,fps=fps,width=width,height=height,slowdown=4,frame_interpolation=False,
        raw_GMR_sampling='previous 30 Hz sample; held only for display after its 2.133 s source ends',physics_steps=0,static_background_contains_complete_trajectories=True),indent=2)+'\n')


if __name__=='__main__':main()
