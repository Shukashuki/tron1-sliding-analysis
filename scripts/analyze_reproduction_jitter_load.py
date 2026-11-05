"""Offline reference-relative jitter and normal-load audit of recorded PhysX PPO.

No training, simulator steps, time warping, or force reconstruction from root z.
Reference motion has no measured contact-force target. Required reference net
vertical force is only a whole-COM kinematic diagnostic at its 50 Hz bandwidth.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[1]
MASS_REPORT=ROOT/'results/hockey_contact_reward_search_20261028_v1/momentum_force_audit_v1/report.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def correlation(x,y):
    x,y=np.asarray(x),np.asarray(y)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and min(x.std(),y.std())>1e-10 else None


def spectrum(x,dt,lo=10.,hi=100.):
    """Linear detrend, Hann, channel-mean one-sided Parseval power."""
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x[:,None]
    if x.ndim!=2 or len(x)<8 or not np.isfinite(x).all() or dt<=0:raise ValueError('Invalid spectrum')
    n=len(x);design=np.c_[np.ones(n),np.arange(n)]
    y=x-design@np.linalg.lstsq(design,x,rcond=None)[0];w=np.hanning(n)
    power=np.mean(abs(np.fft.rfft(y*w[:,None],axis=0))**2,axis=1)/(n*np.sum(w*w))
    if n%2:power[1:]*=2
    else:power[1:-1]*=2
    power[0]=0;f=np.fft.rfftfreq(n,dt);band=(f>lo)&(f<hi)
    dominant=float(f[np.where(band)[0][np.argmax(power[band])]]) if band.any() and power[band].sum()>1e-18 else None
    return dict(rms=float(np.sqrt(power[band].sum())),dominant_hz=dominant),f,power


def interp(t,rt,x):
    x=np.asarray(x);flat=x.reshape(len(x),-1)
    return np.stack([np.interp(t,rt,c) for c in flat.T],axis=-1).reshape((len(t),)+x.shape[1:])


def vertical_balance(body_v,masses,fn,dt):
    if len(body_v)!=len(fn)+1:raise ValueError('Forces must span consecutive state samples')
    momentum=(body_v[:,:,2]*masses[None,:]).sum(-1)
    return np.diff(momentum)/dt-(fn.sum(-1)-masses.sum()*9.81)


def force_window_stats(force,dt,duration):
    width=round(duration/dt)
    if width<1 or not np.isclose(width*dt,duration) or len(force)%width:
        raise ValueError('Force integration must use complete equal-duration windows')
    values=np.asarray(force,dtype=float).reshape(-1,width).mean(-1)
    return dict(window_s=duration,mean_N=float(values.mean()),std_N=float(values.std()),
                peak_N=float(values.max()),maximum_impulse_Ns=float(values.max()*duration))


def default_cases():
    turn='/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz'
    hockey='results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2/motion.npz'
    direct='results/direct_brake_training_20261102_v1/source_snapshot_v1/motion.npz'
    rows=[]
    def add(id,label,path,group,reference,core):
        rows.append(dict(id=id,label=label,path=path,group=group,reference=reference,core=core))
    add('turn_beta_on','Turn: beta ON, 200 Hz','outputs/controller_primary_ppo_mu070_20261015_v1/mu0p700/isaac','turn',turn,[1.44,4.])
    add('turn_beta_off','Turn: beta OFF, 200 Hz','outputs/no_sideslip_ppo_mu0p7_validation_20261016_v1/isaac','turn',turn,[1.44,4.])
    for dt,hz in [('dt0p005',200),('dt0p0025',400)]:
        add('turn_ar05_'+str(hz),f'Turn: action-rate -0.5, {hz} Hz','outputs/ppo_contact_timestep_20261022_v1/'+dt,'turn',turn,[1.44,4.])
    base='results/hockey_contact_reward_search_20261028_v1/'
    add('hockey_dense','Hockey: dense contact',base+'publication_v2/run_v2/native_equal_dense/nominal','stop',hockey,[.4,1.8])
    add('hockey_v7','Hockey: continuity v7',base+'continuity_publication_v3/run_v7/native_equal_smooth_stop_xyz_huber_continuity/nominal','stop',hockey,[.4,1.8])
    base='results/hockey_static_dynamic_transition_20261029_v1/'
    for id,label,case in [('hockey_v8','Hockey v8: mu 0.7/0.7','probe_equal_070'),('hockey_v8_split','Hockey v8: mu 0.9/0.7','probe_split_090_070'),('hockey_v8_high','Hockey v8: mu 0.9/0.9','probe_equal_090')]:
        add(id,label,base+'probe_evidence/run_v10/'+case+'/nominal','stop',hockey,[.4,1.8])
    add('hockey_v9','Hockey v9: split trained',base+'continuation_evidence/run_v9/native_split_smooth_stop_xyz_huber_continuity/nominal','stop',hockey,[.4,1.8])
    add('direct','Direct braking','results/direct_brake_comparison_20261103_v1/evidence/evaluation/nominal','stop',direct,[.4,1.8])
    return rows


def contact_geometry(raw,n):
    """Geometry row k belongs to pre-step state k; forces act over [k,k+1]."""
    touch=np.zeros((n,2),bool);unknown=touch.copy();counts=np.zeros((n,2),int)
    for side in (0,1):
        for k in range(n):
            count=int(raw[f'w{side}_normal_buffer_4'][k].item());start=int(raw[f'w{side}_normal_buffer_5'][k].item())
            counts[k,side]=count
            if count<0 or start<0 or start+count>len(raw[f'w{side}_normal_buffer_3'][k]):raise ValueError('Invalid contact span')
            if not count:continue
            sl=slice(start,start+count)
            normal=raw[f'w{side}_normal_buffer_2'][k,sl]
            separation=raw[f'w{side}_normal_buffer_3'][k,sl].ravel()
            if not np.isfinite(normal).all() or not np.isfinite(separation).all():raise ValueError('Invalid contact geometry')
            good=np.linalg.norm(normal,axis=-1)>=1e-8
            touch[k,side]=bool((good&(separation<=.0001)).any())
            unknown[k,side]=bool((~good).any() and not touch[k,side])
    return touch,unknown,counts


def case_data(entry,mass_map):
    folder=ROOT/entry['path'];reference=ROOT/entry['reference'];report=read(folder/'report.json');ref=load(reference)
    reference_hash=sha(reference)
    if report.get('motion_sha256',report.get('motion_file_sha256'))!=reference_hash:raise ValueError('Reference differs')
    provenance=dict(report_sha256=sha(folder/'report.json'),reference_sha256=reference_hash)
    modern=(folder/'final_rollout.npz').exists()
    if modern:
        for name,digest in report['artifacts'].items():
            if sha(folder/name)!=digest:raise ValueError('Artifact differs: '+str(folder/name))
        tr=load(folder/'final_rollout.npz');raw=load(folder/'native_contact_buffers.npz')
        dt=float(tr['physics_dt_s']);t=np.arange(len(tr['root_pos']))*dt
        pos,quat,omega=tr['root_pos'],tr['root_quat'],tr['root_omega']
        body_names=tr['body_names'].astype(str).tolist();m=np.array([mass_map[s] for s in body_names])
        body_v=tr['body_com_vel'];fn=tr['normal_fz'][1:];ft=tr['friction_force_w'][1:]
        touch,unknown,counts=contact_geometry(raw,len(fn))
        actions=tr['actions'];joint_names=tr['joint_names'].astype(str).tolist()
        torque=tr.get('applied_joint_torque_nm')
        kind='geometric separation <=0.1 mm, valid normal; no force threshold'
        provenance.update(report['artifacts']);actor=report['actor_sha256']
    else:
        path=folder/'physics_trajectory.npz'
        if sha(path)!=report['physics_trajectory_sha256']:raise ValueError('Trajectory hash mismatch')
        tr=load(path);get=lambda k:tr[k][:,0]
        t=tr['time_s'];dt=float(tr['physics_dt_s'])
        if not tr['valid_mask'].all():raise ValueError('Invalid native records')
        pos,quat,omega=get('root_position_m'),get('root_quaternion_wxyz'),get('root_angular_velocity_w_rad_s')
        fn=get('wheel_ground_contact_force_w_n')[1:,:,2];ft=None
        counts=get('wheel_ground_contact_count')[1:];touch=counts>0;unknown=np.zeros_like(touch)
        body_v=None;m=None
        decimation=round(.02/dt);actions=get('actions_clipped')[1::decimation]
        joint_names=tr['joint_names'].astype(str).tolist();torque=get('applied_joint_torque_nm')
        kind='native manifold counts only; archived separation unavailable (not geometric continuity)'
        provenance['physics_trajectory_sha256']=sha(path);actor=report['checkpoint_sha256']
    if not np.allclose(np.diff(t),dt,atol=1e-10,rtol=0) or fn.min() < -1e-4:raise ValueError('Invalid clock/normal force')
    rt=np.arange(len(ref['joint_pos']))/float(np.asarray(ref['fps']).item());end=rt[-1]
    n=round(end/dt);t=t[:n+1];pos=pos[:n+1];quat=quat[:n+1];omega=omega[:n+1]
    fn=fn[:n];touch=touch[:n];unknown=unknown[:n];counts=counts[:n]
    if body_v is not None:body_v=body_v[:n+1];ft=ft[:n]
    bid=ref['body_names'].astype(str).tolist().index('base_Link')
    rp=interp(t,rt,ref['body_pos_w'][:,bid]);rw=interp(t,rt,ref['body_ang_vel_w'][:,bid])
    if t[0]<rt[0]-1e-10 or t[-1]>rt[-1]+1e-10:raise ValueError('Reference does not cover native states')
    rq=Slerp(rt,Rotation.from_quat(ref['body_quat_w'][:,bid],scalar_first=True))(np.clip(t,rt[0],rt[-1]))
    ori_error=(Rotation.from_quat(quat,scalar_first=True)*rq.inv()).as_rotvec()
    error=omega-rw
    return dict(t=t,dt=dt,pos=pos,quat=quat,omega=omega,rp=rp,rw=rw,error=error,ori_error=ori_error,
        fn=fn,touch=touch,unknown=unknown,counts=counts,body_v=body_v,masses=m,ft=ft,
        actions=actions,torque=torque,joint_names=joint_names,ref=ref,rt=rt,actor=actor,
        provenance=provenance,contact_kind=kind,modern=modern,report=report)


def analyze(entry,d,mass_map):
    dt=d['dt'];t=d['t'];lo,hi=entry['core'];mg=sum(mass_map.values())*9.81
    mask=(t>=lo-1e-9)&(t<=hi+1e-9)
    ti=(t[:-1]+t[1:])*.5;core=(ti>=lo-1e-9)&(ti<hi-1e-9)
    fn=d['fn'];total=fn.sum(-1);difference=fn[:,0]-fn[:,1]
    hf,f,power=spectrum(d['omega'][mask,:2],dt)
    shared,_,_=spectrum(d['error'][mask,:2],dt,10,20)
    own,_,_=spectrum(d['omega'][mask,:2],dt,10,20)
    target,_,_=spectrum(d['rw'][mask,:2],dt,10,20)
    beyond,_,_=spectrum(d['omega'][mask,:2],dt,25,100)
    loadhf,_,_=spectrum(total[core],dt)
    legs=[i for i,name in enumerate(d['joint_names']) if not name.startswith('wheel_')]
    at=np.arange(len(d['actions']))*.02;acore=(at>=lo-1e-9)&(at<=hi+1e-9)
    adelta=np.diff(d['actions'][acore][:,legs],axis=0)
    both=d['touch'].all(-1)&~d['unknown'].any(-1)
    alpha=np.diff(d['omega'],axis=0)/dt
    on_change=np.r_[False,(d['touch'][1:]!=d['touch'][:-1]).any(-1)]
    near_change=np.convolve(on_change.astype(int),np.ones(2*round(.02/dt)+1,dtype=int),mode='same')>0
    dforce=np.r_[0,np.diff(total)];energy=dforce[core]**2
    row=dict(**entry,actor_sha256=d['actor'],native_hz=1/dt,policy_hz=50,provenance=d['provenance'],
        core_native_states=int(mask.sum()),core_intervals=int(core.sum()),
        angular_xy_native_10_100_rms_rad_s=hf['rms'],angular_dominant_hz=hf['dominant_hz'],
        angular_xy_native_25_100_rms_rad_s=beyond['rms'],
        angular_xy_error_10_20_rms_rad_s=shared['rms'],angular_xy_actual_10_20_rms_rad_s=own['rms'],
        angular_xy_reference_10_20_rms_rad_s=target['rms'],
        orientation_error_xy_rms_deg=rms(np.degrees(d['ori_error'][mask,:2])),
        root_xy_tracking_rmse_m=float(np.sqrt(np.mean(np.sum((d['pos']-d['rp'])[mask,:2]**2,axis=-1)))),
        root_z_error_rms_mm=1000*rms((d['pos']-d['rp'])[mask,2]),
        root_z_error_std_mm=1000*float((d['pos']-d['rp'])[mask,2].std()),
        root_z_error_10_20_rms_mm=1000*spectrum((d['pos']-d['rp'])[mask,2],dt,10,20)[0]['rms'],
        leg_action_delta_rms=rms(adelta),leg_torque_10_100_rms_nm=None if d['torque'] is None else spectrum(d['torque'][:len(t)][mask][:,legs],dt)[0]['rms'],
        force=dict(total_mean_N=float(total[core].mean()),total_std_N=float(total[core].std()),
            total_p99_N=float(np.quantile(total[core],.99)),total_peak_N=float(total[core].max()),
            total_10_100_rms_N=loadhf['rms'],total_dominant_hz=loadhf['dominant_hz'],
            total_mean_over_mg=float(total[core].mean()/mg),peak_over_mg=float(total[core].max()/mg),
            mean_left_right_N=fn[core].mean(0).tolist(),std_left_right_N=fn[core].std(0).tolist(),
            wheel_load_correlation=correlation(fn[core,0],fn[core,1]),
            difference_std_N=float(difference[core].std()),
            equal_duration_windows={str(s):force_window_stats(total[core],dt,s) for s in (.005,.02)},
            corr_total_force_base_roll_acceleration=correlation(total[core],alpha[core,0]),
            corr_load_difference_base_roll_acceleration=correlation(difference[core],alpha[core,0])),
        contact=dict(definition=d['contact_kind'],core_both_fraction=float(both[core].mean()),
            core_missing_either_ms=float((~both[core]).sum()*dt*1000),
            full_missing_either_ms=float((~both).sum()*dt*1000),core_transition_count=int(on_change[core].sum()),
            core_zero_load_wheel_samples_while_touching=int((d['touch'][core]&(fn[core]<=1e-6)).sum()),
            force_difference_energy_near_contact_transitions_fraction=float((energy*near_change[core]).sum()/energy.sum()) if energy.sum()>0 else 0.,
            conditional_peak_force_both_touch_N=float(total[core&both].max()) if (core&both).any() else None),
        mass_basis='authored hash-locked masses; no new runtime readback',authored_mass_kg=mg/9.81)
    d['ti']=ti;d['core']=core;d['total']=total;d['difference']=difference;d['frequency']=f;d['angular_power']=power
    if d['body_v'] is not None:
        residual=vertical_balance(d['body_v'],d['masses'],fn,dt)
        az=(np.diff((d['body_v']*d['masses'][None,:,None]).sum(1),axis=0)/d['masses'].sum()/dt)[:,2]
        observed=az*d['masses'].sum()
        refm=np.array([mass_map[s] for s in d['ref']['body_names'].astype(str)])
        refvz=(d['ref']['body_lin_vel_w'][:,:,2]*refm[None,:]).sum(-1)/refm.sum()
        required=refm.sum()*(9.81+np.gradient(refvz,d['rt'],edge_order=2))
        required_native=interp(ti,d['rt'],required)
        row['momentum']=dict(vertical_balance_RMS_N=rms(residual[core]),
            shifted_force_RMS_N={str(k):rms((observed-(np.roll(total,k)-mg))[core]) for k in (-1,1)},
            total_force_vs_whole_COM_az_correlation=correlation(total[core],az[core]),
            required_reference_net_vertical_mean_N=float(required_native[core].mean()),
            required_reference_net_vertical_std_N=float(required_native[core].std()),
            required_reference_net_vertical_min_N=float(required_native[core].min()),
            required_reference_net_vertical_max_N=float(required_native[core].max()),
            reference_force_is_measured=False,reference_force_identifies_individual_wheels=False)
        d.update(az=az,balance_residual=residual,required_reference_N=required_native)
    else:row['momentum']=None
    return row


def render(rows,data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Four horizontal panels, with identical axes within each comparison.
    for name,ids in [('turn',['turn_beta_on','turn_beta_off','turn_ar05_200','turn_ar05_400']),
                     ('stop',['hockey_dense','hockey_v7','hockey_v8','direct'])]:
        fig,axes=plt.subplots(1,4,figsize=(23,5),layout='constrained')
        for index,id in enumerate(ids):
            d=data[id];r=next(v for v in rows if v['id']==id);color=plt.cm.tab10(index)
            mask=(d['t']>=r['core'][0])&(d['t']<=r['core'][1]);core=d['core']
            axes[0].plot(d['t'][mask],d['error'][mask,0],color=color,lw=.7,label=r['label'])
            axes[1].plot(d['ti'][core],d['total'][core],color=color,lw=.7)
            axes[2].semilogy(d['frequency'][1:],np.maximum(d['angular_power'][1:],1e-15),color=color)
            axes[3].scatter(r['force']['total_std_N'],r['angular_xy_native_10_100_rms_rad_s'],color=color,s=60)
            axes[3].annotate(id.replace('hockey_','').replace('turn_',''),(r['force']['total_std_N'],r['angular_xy_native_10_100_rms_rad_s']),xytext=(4,4),textcoords='offset points',fontsize=8)
        axes[0].set(xlabel='Native time [s]',ylabel='omega X - reference [rad/s]',title='Reference-relative roll-rate error')
        axes[1].axhline(rows[0]['authored_mass_kg']*9.81,color='black',ls='--',lw=.8)
        axes[1].set(xlabel='Interval midpoint [s]',ylabel='Left + right normal force [N]',title='Total normal load (dashed: mg)')
        axes[2].axvline(25,color='black',ls=':',lw=.8);axes[2].set(xlim=(0,100),xlabel='Frequency [Hz]',ylabel='Hann power per bin [rad2/s2]',title='Native body angular XY spectrum')
        axes[3].set(xlabel='Total normal-load standard deviation [N]',ylabel='Angular XY 10-100 Hz RMS [rad/s]',title='Load variation and body jitter')
        axes[0].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(('Car-derived turn' if name=='turn' else 'Stopping reproductions')+' | Native recorded physics; different recipes are descriptive comparisons')
        fig.savefig(output/(name+'_comparison_4x1.png'),dpi=145);plt.close(fig)
    ids=['hockey_v8','hockey_v8_split','hockey_v8_high','hockey_v9']
    fig,axes=plt.subplots(2,4,figsize=(23,7),sharex=True,layout='constrained')
    ymax=max(data[k]['total'][data[k]['core']].max() for k in ids)
    for j,id in enumerate(ids):
        d=data[id];r=next(v for v in rows if v['id']==id);mask=(d['t']>=.4)&(d['t']<=1.8);c=d['core']
        axes[0,j].plot(d['t'][mask],d['omega'][mask,0],lw=.8,label='PPO')
        axes[0,j].plot(d['t'][mask],d['rw'][mask,0],color='black',lw=1,label='Reference')
        axes[0,j].set_title(r['label'],fontsize=10)
        axes[1,j].plot(d['ti'][c],d['fn'][c,0],lw=.6,label='Left')
        axes[1,j].plot(d['ti'][c],d['fn'][c,1],lw=.6,label='Right')
        axes[1,j].plot(d['ti'][c],d['total'][c],lw=.7,color='black',alpha=.7,label='Total')
        axes[1,j].plot(d['ti'][c],d['required_reference_N'][c],lw=1,ls='--',color='purple',label='Reference required NET (inferred)')
        axes[1,j].set(xlim=(.4,1.8),ylim=(0,ymax*1.03),xlabel='Physical time [s]')
        for ax in axes[:,j]:ax.grid(alpha=.2)
    axes[0,0].set_ylabel('World roll rate [rad/s]');axes[1,0].set_ylabel('Normal force [N]')
    rollmax=max(max(abs(data[k]['omega'][(data[k]['t']>=.4)&(data[k]['t']<=1.8),0]).max(),
                    abs(data[k]['rw'][(data[k]['t']>=.4)&(data[k]['t']<=1.8),0]).max()) for k in ids)
    for ax in axes[0]:ax.set_ylim(-rollmax*1.05,rollmax*1.05)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=7)
    fig.suptitle('Continuous contact does not imply constant load | Same reference; v8 material probes keep the actor fixed')
    fig.savefig(output/'hockey_material_reference_load.png',dpi=145);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Refuse to overwrite an analysis')
    mass=read(MASS_REPORT);mass_map=mass['authored_body_masses_kg'];selection=default_cases();rows=[];data={}
    for entry in selection:
        d=case_data(entry,mass_map)
        if d['report'].get('asset_sha256',d['report'].get('asset_root_file_sha256'))!=mass['asset_sha256']:
            raise ValueError('Authored mass asset differs from the recording')
        row=analyze(entry,d,mass_map);rows.append(row);data[entry['id']]=d
    a.output.mkdir(parents=True)
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    saved={}
    for id,d in data.items():
        for key in ('t','ti','omega','rw','error','fn','total','difference','touch','unknown','core','frequency','angular_power','required_reference_N','balance_residual'):
            if key in d:saved[id+'_'+key]=d[key]
    np.savez_compressed(a.output/'signals.npz',**saved)
    render(rows,data,a.output)
    summary=dict(kind='recorded_reference_relative_jitter_normal_load_audit_v1',source_sha256=sha(__file__),
        mass_report_sha256=sha(MASS_REPORT),cases=rows,new_training=False,new_simulation=False,
        reference_sample_hz=50,reference_shared_spectral_band_hz=[10,20],native_comparison_band_hz=[10,100],
        native_above_reference_band_hz=[25,100],force_scope='Impulse divided by native timestep, over consecutive velocity states; geometric contact at interval start',
        reference_interpolation='Linear world angular velocity/position; quaternion SLERP. No phase optimization or time warping.',
        reference_force_scope='Inferred necessary whole-COM vertical resultant M(g+dvref_COM,z/dt) at 50 Hz; not measured or dynamically validated, no per-wheel allocation.',
        comparisons={'turn_beta_on_off':'Same nominal physical/evaluation contract, historical training setup/RNG confound.',
            'turn_ar05_200_400':'Same actor, reference and 50 Hz policy; physics and DC-PD update rates both change.',
            'hockey_dense_v7_v8':'Contact recipe and accumulated training budgets differ; descriptive progression.',
            'hockey_v8_materials':'Same actor/reference/initial state; only material coefficients differ in closed loop.',
            'hockey_v8_split_v9':'Same material/reference, matched lineage budgets; distinct trained policy.',
            'direct_hockey':'Different behavior references and training seeds; no isolated causal prior/beta conclusion.'},
        limitations=['Each entry is one archived nominal rollout, not independent training-seed evidence.',
            'A 50 Hz kinematic reference cannot certify absence of true reference motion above 25 Hz.',
            'Car contact manifold counts do not establish geometric touching; hockey geometry uses 0.1 mm tolerance.',
            'No measured reference normal forces exist; per-wheel reference loads are not identifiable from the supplied poses alone.',
            'Mean normal force, load distribution and force fluctuations are different statistics.',
            'Whole-system force balance supports the summed channel and timing, not individual contact patch force accuracy.',
            'Observed force-motion/action association does not establish a one-way causal direction.'],
        artifacts={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()})
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (a.output/'cases.csv').open('w') as f:
        fields=['id','angular_xy_native_10_100_rms_rad_s','angular_xy_error_10_20_rms_rad_s','leg_action_delta_rms','total_mean_N','total_std_N','total_peak_N','core_missing_either_ms']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:({**r,**r['force'],**r['contact']})[k] for k in fields})
    print((a.output/'cases.csv').read_text())


if __name__=='__main__':main()
