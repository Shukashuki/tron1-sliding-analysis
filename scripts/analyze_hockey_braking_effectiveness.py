"""Separate local friction utilization from net COM braking and yaw torque."""
import argparse
import json
from pathlib import Path
import numpy as np

from audit_hockey_contact_cone_relation import finite_stats
from evaluate_hockey_physx import load,sha


def planar_effects(force, velocity, mass, dt):
    force=np.asarray(force,dtype=float);velocity=np.asarray(velocity,dtype=float)
    if (force.ndim!=3 or force.shape[1:]!=(2,2) or velocity.shape!=(len(force)+1,2)
            or not np.isfinite(force).all() or not np.isfinite(velocity).all() or mass<=0 or dt<=0):
        raise ValueError('Require complete two-wheel planar forces and COM velocities')
    magnitude=np.linalg.norm(force,axis=-1);net=force.sum(1);netnorm=np.linalg.norm(net,axis=-1)
    cancellation=np.full(len(force),np.nan)
    valid=magnitude.sum(1)>1e-6
    cancellation[valid]=1-netnorm[valid]/magnitude.sum(1)[valid]
    speed=np.linalg.norm(velocity,axis=-1);direction=np.zeros_like(velocity[:-1])
    moving=speed[:-1]>.05;direction[moving]=velocity[:-1][moving]/speed[:-1][moving,None]
    braking=-(force*direction[:,None,:]).sum(-1)/mass;braking[~moving]=np.nan
    vmean=.5*(velocity[1:]+velocity[:-1])
    COM_work=(force*vmean[:,None,:]).sum(-1)*dt
    delta_KE=.5*mass*np.diff(speed**2)
    return dict(wheel_force_magnitude_N=magnitude,net_force_N=net,net_force_magnitude_N=netnorm,
                cancellation_fraction=cancellation,braking_acceleration_per_wheel_m_s2=braking,
                COM_xy_work_per_wheel_J=COM_work,COM_xy_kinetic_energy_change_J=delta_KE,
                COM_work_balance_residual_J=delta_KE-COM_work.sum(1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda p:json.loads(p.read_text());report=read(a.analysis/'report.json');sources=read(a.selection)
    if report['selection_sha256']!=sha(a.selection):raise ValueError('Selection changed')
    if sha(a.analysis/'signals.npz')!=report['artifact_sha256']['signals.npz']:raise ValueError('Signals changed')
    s=load(a.analysis/'signals.npz');rows=[];saved={};charts=[]
    for k,(source,prior) in enumerate(zip(sources,report['cases'])):
        directory=(a.selection.parent/source['evidence']/source['run_name']/source['case'].replace(':','/')).resolve()
        trace_path=directory/'final_rollout.npz'
        if sha(trace_path)!=prior['raw_sha256']['final_rollout.npz']:raise ValueError('Raw state changed')
        tr=load(trace_path);t=s[f'case{k}_time_start_s'];n=len(t);dt=.0025;mass=prior['authored_total_mass_kg']
        force=tr['friction_force_w'][1:n+1,:,:2]
        effects=planar_effects(force,s[f'case{k}_COM_velocity'][:,:2],mass,dt)
        anchors=s[f'case{k}_anchors'];tau=np.zeros((n,2));power=np.zeros((n,2));anchor_norm_sum=np.zeros((n,2))
        for side in (0,1):
            x=anchors[anchors[:,1]==side];indices=x[:,2].astype(int)
            r=x[:,4:7]-s[f'case{k}_COM_position'][indices]
            moment=np.cross(r,x[:,7:10])[:,2]
            np.add.at(tau[:,side],indices,moment);np.add.at(power[:,side],indices,x[:,10])
            np.add.at(anchor_norm_sum[:,side],indices,np.linalg.norm(x[:,7:9],axis=-1))
        within_wheel_cancellation=np.full((n,2),np.nan)
        valid=anchor_norm_sum>1e-6
        within_wheel_cancellation[valid]=1-effects['wheel_force_magnitude_N'][valid]/anchor_norm_sum[valid]
        effects.update(friction_yaw_torque_Nm=tau,anchor_pre_velocity_power_W=power,
                       within_wheel_anchor_cancellation_fraction=within_wheel_cancellation)
        phases=[]
        for label,start,end in [('braking',.4,1.8),('stop',1.8,2.76)]:
            mask=(t>=start-1e-9)&(t+dt<=end+1e-9)
            phases.append(dict(phase=label,cancellation=finite_stats(effects['cancellation_fraction'][mask]),
                within_wheel_anchor_cancellation=[finite_stats(within_wheel_cancellation[mask,side]) for side in (0,1)],
                net_force_magnitude_N=finite_stats(effects['net_force_magnitude_N'][mask]),
                wheel_force_magnitude_sum_N=finite_stats(effects['wheel_force_magnitude_N'][mask].sum(-1)),
                COM_xy_work_per_wheel_J=effects['COM_xy_work_per_wheel_J'][mask].sum(0).tolist(),
                COM_xy_kinetic_energy_change_J=float(effects['COM_xy_kinetic_energy_change_J'][mask].sum()),
                COM_work_balance_residual_J=float(effects['COM_work_balance_residual_J'][mask].sum()),
                anchor_power_proxy_integral_per_wheel_J=(power[mask].sum(0)*dt).tolist(),
                friction_yaw_angular_impulse_Nm_s=(tau[mask].sum(0)*dt).tolist(),
                friction_yaw_torque_RMS_Nm=float(np.sqrt(np.mean(tau[mask].sum(-1)**2))),
                moving_COM_braking_acceleration_per_wheel_m_s2=[finite_stats(effects['braking_acceleration_per_wheel_m_s2'][mask,side]) for side in (0,1)]))
        rows.append(dict(label=prior['label'],mu_static=prior['mu_static'],mu_dynamic=prior['mu_dynamic'],phases=phases,
            whole_task_COM_work_balance_max_interval_error_J=float(abs(effects['COM_work_balance_residual_J']).max())))
        for key,value in effects.items():saved[f'case{k}_{key}']=value
        charts.append((prior['label'],t,effects))
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'signals.npz',**saved)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for k,(label,t,e) in enumerate(charts):
        fig,axs=plt.subplots(1,4,figsize=(24,4.8),constrained_layout=True)
        for side,color in enumerate(('#2779a5','#d37527')):
            axs[0].plot(t,e['wheel_force_magnitude_N'][:,side],color=color,label=('L','R')[side])
            axs[2].plot(t,e['braking_acceleration_per_wheel_m_s2'][:,side],color=color,label=('L','R')[side])
            axs[3].plot(t,e['friction_yaw_torque_Nm'][:,side],color=color,label=('L','R')[side])
        axs[0].plot(t,e['net_force_magnitude_N'],color='black',label='Net')
        axs[1].plot(t,e['cancellation_fraction'],color='black')
        axs[2].plot(t,e['braking_acceleration_per_wheel_m_s2'].sum(-1),color='black',label='Net')
        axs[3].plot(t,e['friction_yaw_torque_Nm'].sum(-1),color='black',label='Net')
        for ax,title,y in zip(axs,['Wheel forces and resultant','Cancellation between wheels','COM braking contribution','Friction torque about COM'],['Magnitude [N]','1 - |sum F| / sum |F|','Signed deceleration [m/s²]','Yaw torque [Nm]']):
            ax.set(title=title,ylabel=y,xlabel='Time [s]',xlim=(.4,2.76));ax.axvline(1.8,color='gray',ls=':');ax.grid(alpha=.2)
        axs[1].set_ylim(-.02,1.02)
        for ax in (axs[0],axs[2],axs[3]):ax.legend(fontsize=8)
        fig.suptitle(label+' | Local capacity does not directly measure net braking\nCOM braking direction omitted below 0.05 m/s; yaw torque uses native friction anchors; no motor-work measurement')
        fig.savefig(a.output/f'case{k}_effectiveness_4x1.png',dpi=140);plt.close(fig)
    result=dict(kind='hockey_friction_braking_effectiveness_v1',source_sha256=sha(__file__),
        analysis_sha256=sha(a.analysis/'report.json'),selection_sha256=sha(a.selection),cases=rows,
        physics_steps=0,gates_changed=False,actual_friction_work_measured=False,
        interpretation=['Wheel-resultant cancellation and yaw torque distinguish contact saturation from net braking.',
            'COM work uses interval-mean whole-system velocity and matches the COM translational kinetic-energy change.',
            'COM work is not contact-slip dissipation; pre-velocity anchor power integrated over dt is only a proxy.',
            'Yaw torque is the friction contribution, not a complete angular-momentum audit or measured motor work.'],
        artifact_sha256={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
