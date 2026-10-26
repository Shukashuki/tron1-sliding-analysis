"""Model-first test of sustained saturation AND slip in coupled wheel dynamics.

Five velocities: world vx/vy,yaw rate,and two wheel spins. The hard Coulomb
stick/slip graph is solved simultaneously with body and wheel momentum at
each step, using a product of two friction DISKS. Normal loads are prescribed
by a planar support fixture, not a full-body contact solution. No tire curve
is fitted to a drift demonstration; no velocity or tangent force is prescribed.
Only initial speed and bounded wheel torque commands enter the simulation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import generate_mppi_contact_reference as g
from screen_mppi_sliding import runs


def disk_projection(value,radius):
    value=np.asarray(value,dtype=float).reshape(2,2)
    factor=np.minimum(1.,np.asarray(radius)/np.maximum(np.linalg.norm(value,axis=-1),1e-30))
    return (value*factor[:,None]).ravel()


class CoupledCoulomb:
    def __init__(self,mass,yaw_inertia,wheel_inertia,track,radius,*,tolerance=1e-9,max_iterations=800):
        self.mass,self.yaw_inertia,self.track,self.radius=mass,yaw_inertia,track,radius
        self.M=np.r_[mass,mass,yaw_inertia,wheel_inertia]
        if self.M.shape!=(5,) or np.any(self.M<=0):raise ValueError('Positive inertial parameters required')
        self.inv=1/self.M;self.tolerance=tolerance;self.max_iterations=max_iterations
        self.warm=np.zeros(4);self.dual=np.zeros(4)

    def jacobian(self,yaw):
        t=np.array([np.cos(yaw),np.sin(yaw)]);l=np.array([-t[1],t[0]])
        J=np.zeros((4,5))
        for i,y in enumerate((self.track/2,-self.track/2)):
            J[2*i,:2]=t;J[2*i,2]=-y;J[2*i,3+i]=-self.radius
            J[2*i+1,:2]=l
        return J

    def impulse(self,A,b,limits):
        """Convex tangential QP for FIXED normal impulses, solved by ADMM.

        min 1/2 p.T A p + b.T p, ||p_i||<=mu Fn_i h.
        KKT directly gives sticking interior or saturated dissipative sliding.
        This is not a general unknown-normal Coulomb/Signorini solver.
        """
        L=float(np.linalg.eigvalsh(A).max());rho=.4
        inverse=np.linalg.inv(A+rho*np.eye(4))
        z=disk_projection(self.warm,limits);dual=self.dual.copy()
        for iteration in range(1,self.max_iterations+1):
            p=inverse@(rho*(z-dual)-b)
            z=disk_projection(p+dual,limits);dual+=p-z
            gradient=A@z+b
            residual=L*float(np.max(abs(z-disk_projection(z-gradient/L,limits))))
            if residual<self.tolerance:break
        else:raise RuntimeError(f'Contact solve failed: projected KKT velocity residual={residual}')
        self.warm,self.dual=z.copy(),dual.copy()
        return z,iteration,residual

    def step(self,velocity,yaw,torques,mu,normal_load,dt):
        J=self.jacobian(yaw);A=(J*self.inv[None])@J.T
        force=np.r_[0.,0.,0.,torques];free=velocity+dt*self.inv*force
        limits=mu*np.asarray(normal_load)*dt
        p,it,residual=self.impulse(A,J@free,limits)
        after=free+self.inv*(J.T@p);slip=(J@after).reshape(2,2);F=p.reshape(2,2)/dt
        utilization=np.linalg.norm(p.reshape(2,2),axis=-1)/limits
        power=np.sum(F*slip,axis=-1)
        energy_free=.5*np.sum(self.M*free**2);energy_after=.5*np.sum(self.M*after**2)
        balance=energy_after-energy_free-float(p@(J@after))+.5*float(p@A@p)
        return after,dict(force_t_n=F,slip_m_s=slip,rho=utilization,friction_power_w=power,
            energy_loss_contact_step_j=energy_free-energy_after,energy_balance_error_j=balance,
            kkt_velocity_residual_m_s=residual,solver_iterations=it)


def parameters(model_path,reference_path,material_path):
    if g.sha(model_path)!=g.MODEL_SHA or g.sha(reference_path)!=g.REFERENCE_SHA or g.sha(material_path)!=g.MATERIAL_REPORT_SHA:
        raise ValueError('Require declared robot source assets')
    material=json.loads(material_path.read_text());m=g.load_model(model_path,material);d=mujoco.MjData(m)
    with np.load(reference_path,allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    d.qpos[:]=source['qpos_mujoco'][0];mujoco.mj_forward(m,d)
    mass=float(m.body_mass.sum());com=np.sum(d.xipos*m.body_mass[:,None],axis=0)/mass;inertia=np.zeros((3,3))
    for i in range(1,m.nbody):
        R=d.ximat[i].reshape(3,3);r=d.xipos[i]-com
        inertia+=R@np.diag(m.body_inertia[i])@R.T+m.body_mass[i]*((r@r)*np.eye(3)-np.outer(r,r))
    wheel_inertia=[]
    for side in ('L','R'):
        body=m.body('wheel_'+side+'_Link').id;R=d.ximat[body].reshape(3,3)
        wheel_inertia.append(float((R@np.diag(m.body_inertia[body])@R.T)[1,1]))
    geom=[m.geom('wheel_'+side+'_collision').id for side in ('L','R')]
    time_s=source['time_s'];base=list(source['body_names']).index('base_Link');v=source['body_lin_vel_w'][:,base,:2]
    speed=np.linalg.norm(v,axis=-1);course=np.unwrap(np.arctan2(v[:,1],v[:,0]));heading=np.unwrap(g.quaternion_heading(source['body_quat_w'][:,base]))
    mask=(time_s>=1.44-1e-9)&(time_s<=4+1e-9);acc=np.gradient(v,time_s,axis=0,edge_order=2)
    lateral=abs(v[:,0]*acc[:,1]-v[:,1]*acc[:,0])/np.maximum(speed,.25)
    profile=dict(mass_kg=mass,yaw_inertia_kg_m2=float(inertia[2,2]),wheel_inertia_kg_m2=wheel_inertia,
        track_m=float(d.geom_xpos[geom[0],1]-d.geom_xpos[geom[1],1]),wheel_radius_m=float(m.geom_size[geom[0],0]),
        nominal_speed_m_s=float(speed[mask].mean()),nominal_yaw_rate_rad_s=float(np.gradient(heading,time_s)[mask].mean()),
        reference_50Hz_core_samples=int(mask.sum()),reference_lateral_acceleration_m_s2_p50_p95_max=np.quantile(lateral[mask],[.5,.95,1]).tolist(),
        reference_lateral_demand_over_0p7g_p50_p95_max=np.quantile(lateral[mask]/(.7*9.81),[.5,.95,1]).tolist(),
        reference_derivative_method='gradient of original50Hz COM velocity; kinematic estimate, not measured tire force',
        model_sha256=g.sha(model_path),reference_sha256=g.sha(reference_path),material_sha256=g.sha(material_path))
    return profile


def motor_torque(request,omega):
    curve=np.clip(omega,-200.,200.)
    return np.clip(request,np.maximum(12*(-1-curve/100),-12),np.minimum(12*(1-curve/100),12))


def simulate(case,dt,cfg,par):
    model=CoupledCoulomb(par['mass_kg'],par['yaw_inertia_kg_m2'],par['wheel_inertia_kg_m2'],par['track_m'],par['wheel_radius_m'],
        tolerance=cfg['solver_velocity_tolerance_m_s'],max_iterations=cfg['solver_max_iterations'])
    initial_speed=case['speed_multiplier']*par['nominal_speed_m_s'];yaw_rate_target=case['yaw_multiplier']*par['nominal_yaw_rate_rad_s']
    velocity=np.r_[initial_speed,0.,0.,initial_speed/par['wheel_radius_m'],initial_speed/par['wheel_radius_m']]
    pos=np.zeros(2);yaw=0.;request=np.zeros(2);rows=[];count=round(cfg['duration_s']/dt);control=round(cfg['control_dt_s']/dt)
    normal=np.full(2,par['mass_kg']*9.81/2)
    for k in range(count):
        t=k*dt;co,si=np.cos(yaw),np.sin(yaw);R=np.array([[co,-si],[si,co]])
        body_v=R.T@velocity[:2]
        ramp=np.clip((t-cfg['turn_ramp_s'][0])/np.diff(cfg['turn_ramp_s'])[0],0,1)
        desired_rate=yaw_rate_target*ramp
        if k%control==0:
            if case['control']=='max_drive':request=np.full(2,12.)
            else:
                desired_ax=cfg['speed_feedback_gain_per_s']*(initial_speed-body_v[0])-velocity[2]*body_v[1]
                desired_rdot=cfg['yaw_rate_feedback_gain_per_s']*(desired_rate-velocity[2])
                total_fx=par['mass_kg']*desired_ax;moment=par['yaw_inertia_kg_m2']*desired_rdot;h=par['track_m']/2
                fx=np.array([(total_fx-moment/h)/2,(total_fx+moment/h)/2])
                request=par['wheel_radius_m']*fx+np.asarray(par['wheel_inertia_kg_m2'])*(desired_ax-np.array([h,-h])*desired_rdot)/par['wheel_radius_m']
        torque=motor_torque(request,velocity[3:])
        oldyaw=yaw;after,details=model.step(velocity,yaw,torque,case['mu'],normal,dt)
        pos=pos+dt*after[:2];yaw=yaw+dt*after[2]
        geometric_slip=(model.jacobian(yaw)@after).reshape(2,2)
        rows.append(dict(time_s=t+dt,position_xy_m=pos.copy(),yaw_rad=yaw,velocity=after.copy(),
            desired_yaw_rate_rad_s=desired_rate,requested_torque_nm=request.copy(),torque_nm=torque.copy(),normal_load_n=normal.copy(),
            constraint_heading_rad=oldyaw,geometric_end_pose_slip_m_s=geometric_slip,**details))
        velocity=after
    return {key:np.asarray([r[key] for r in rows]) for key in rows[0]}


def describe(raw,case,dt,cfg,par):
    mask=(raw['time_s']>=cfg['steady_interval_s'][0]-1e-9)&(raw['time_s']<=cfg['steady_interval_s'][1]+1e-9)
    speed=np.linalg.norm(raw['velocity'][:,:2],axis=-1)
    beta=g.angle(np.arctan2(raw['velocity'][:,1],raw['velocity'][:,0])-raw['yaw_rad'])
    saturated=raw['rho']>=1-cfg['constitutive_boundary_tolerance']
    threshold=cfg['sliding_speed_threshold_m_s'];side=abs(raw['slip_m_s'][...,1])>threshold;long=abs(raw['slip_m_s'][...,0])>threshold
    wheels=[]
    for i,name in enumerate(('L','R')):
        events=runs(raw['time_s'],saturated[:,i]&side[:,i])
        wheels.append(dict(side=name,steady_saturated_fraction=float(saturated[mask,i].mean()),
            steady_lateral_sliding_fraction=float(side[mask,i].mean()),steady_joint_lateral_fraction=float((saturated&side)[mask,i].mean()),
            steady_joint_longitudinal_fraction=float((saturated&long)[mask,i].mean()),
            longest_joint_lateral_s=max((e['duration_s'] for e in events),default=0.),
            steady_lateral_speed_abs_m_s_p50_p95=np.quantile(abs(raw['slip_m_s'][mask,i,1]),[.5,.95]).tolist(),
            steady_rho_p50_p95=np.quantile(raw['rho'][mask,i],[.5,.95]).tolist()))
    return dict(id=case['id'],title=case['title'],dt_s=dt,mu=case['mu'],steady_interval_s=cfg['steady_interval_s'],wheels=wheels,
        commanded_speed_m_s=case['speed_multiplier']*par['nominal_speed_m_s'],commanded_yaw_rate_rad_s=case['yaw_multiplier']*par['nominal_yaw_rate_rad_s'],
        nominal_demand_ratio=(case['speed_multiplier']*par['nominal_speed_m_s'])*(case['yaw_multiplier']*par['nominal_yaw_rate_rad_s'])/(case['mu']*9.81),
        steady_speed_mean_m_s=float(speed[mask].mean()),steady_yaw_rate_mean_rad_s=float(raw['velocity'][mask,2].mean()),
        steady_body_beta_deg_p50_p95=np.quantile(np.rad2deg(beta[mask]),[.5,.95]).tolist(),
        max_KKT_velocity_residual_m_s=float(raw['kkt_velocity_residual_m_s'].max()),
        max_positive_friction_power_w=float(max(0,raw['friction_power_w'].max())),
        maximum_rho=float(raw['rho'].max()),minimum_contact_energy_loss_j=float(raw['energy_loss_contact_step_j'].min()),
        max_abs_energy_balance_error_j=float(abs(raw['energy_balance_error_j']).max()),
        peak_solver_iterations=int(raw['solver_iterations'].max()),max_wheel_speed_rad_s=float(abs(raw['velocity'][:,3:]).max()),
        max_applied_torque_nm=float(abs(raw['torque_nm']).max()),normal_support_fraction=1.,
        late_window_motor_envelope_clip_fraction_per_wheel=np.mean(abs(raw['requested_torque_nm'][mask]-raw['torque_nm'][mask])>1e-8,axis=0).tolist(),
        late_window_is_not_a_verified_steady_equilibrium=True,
        frame_discretization='constraints solved at interval-start heading; momentum implicit; pose semi-implicit; geometric end-pose slip separately retained',
        full_TRON1_feasibility_or_task_success_claimed=False)


def run(a):
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    cfg=json.loads(a.config.read_text());par=parameters(a.model,a.reference,a.material_report)
    a.output_dir.mkdir(parents=True);started=time.monotonic();rows=[]
    for case in cfg['cases']:
        for dt in cfg['physics_dt_s']:
            raw=simulate(case,dt,cfg,par);row=describe(raw,case,dt,cfg,par)
            path=a.output_dir/(case['id']+f'__dt{dt:.5f}.npz');np.savez_compressed(path,**raw);row['trace_sha256']=g.sha(path);row['trace_file']=path.name;rows.append(row)
            print(json.dumps(row),flush=True)
    result=dict(kind=cfg['kind'],source_sha256=g.sha(__file__),config_sha256=g.sha(a.config),config=cfg,parameters=par,rows=rows,wall_s=time.monotonic()-started,
        no_tangent_force_or_slip_prescribed=True,only_wheel_torques_and_initial_state_applied=True,
        tire_or_full_body_model_calibrated=False,new_PPO_or_MPPI=False,new_physics_engine_or_package_installed=False,
        contact_law_source='https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/users_guide/problems_and_solvers/friction_contact.html',
        limitations=['This original five-velocity tangential fixture is not Siconos runtime or full unilateral contact.',
            'Normal loads are fixed equal mg/2 by a planar fixture; no roll,pitch,height,leg control,load transfer or loss of support is represented.',
            'The hard Coulomb graph uses identical static/dynamic mu and circular cones; it is a different model from the earlier MuJoCo soft pyramidal contact.',
            'No finite contact patch,tire deformation,camber thrust,relaxation or identified tire coefficients are present.',
            'This tests whether a coupled model can sustain saturation and sliding; it does not validate the exact turning task or a TRON1 controller.',
            'Changing time/space scaling changes inertia/friction demand even when the geometric reference beta is unchanged.'],
        full_TRON1_feasibility_or_task_success_claimed=False)
    g.write_json(a.output_dir/'analysis.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--config',type=Path,default=g.ROOT/'config/coupled_sliding_model_bench_v1.json')
    p.add_argument('--model',type=Path,default=g.ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml')
    p.add_argument('--reference',type=Path,default=g.ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz')
    p.add_argument('--material-report',type=Path,default=g.ROOT/'results/mppi_contact_reference_20261025_v1/inputs/original_native_material_report.json')
    p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args())
