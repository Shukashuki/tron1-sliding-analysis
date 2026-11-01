"""Bilateral geometric contact and useful sliding, with pre-step native timing.

Every action collects eight native contact buffers. Contact geometry/velocities
are paired with the state BEFORE their physics step; forces represent that
step's average. No force threshold or equal-load target defines touching.
"""
from __future__ import annotations

import math

from hockey_ppo_task import SupportWindow, WHEELS, SENSORS, axle, rewards as base_rewards
from hockey_friction_boundary import boundary_sample


def cylinder_gap(position, quaternion, cfg, side):
    """Locked authored finite cylinder; geometry shaping, not a contact certificate."""
    axis = axle(quaternion)
    center_z = position[:, 2] + axis[:, 2]*cfg['center_y_in_link_m'][side]
    height = cfg['radius_m']*(1-axis[:, 2].square()).clamp_min(0).sqrt() + cfg['half_width_m']*axis[:, 2].abs()
    return center_z-height-cfg['floor_z_m']


def geometric_sample(normal, com, velocity, angular, quaternion, tolerance, cap=128):
    import torch
    _, points, normals, distance, counts, starts = normal
    n = len(com)
    j = torch.arange(cap, device=com.device)[None]
    ix = (starts.reshape(n, 1)+j).long(); count=counts.reshape(n,1)
    mask=j<count
    fault=((count<0)|(count>cap)).reshape(n) | (mask&((ix<0)|(ix>=len(points)))).any(-1)
    if len(points):
        ids=ix.clamp(0,len(points)-1)
        p,normal,sep=points[ids],normals[ids],distance.reshape(-1)[ids]
    else:
        p=torch.zeros((n,cap,3),device=com.device,dtype=com.dtype)
        normal=torch.zeros_like(p);sep=torch.zeros((n,cap),device=com.device,dtype=com.dtype)
        mask=mask&False
    fault |= (mask & (~torch.isfinite(p).all(-1)|~torch.isfinite(normal).all(-1)|~torch.isfinite(sep))).any(-1)
    norm=normal.norm(dim=-1,keepdim=True)
    valid=mask&(norm[...,0]>=1e-8)
    normal=normal/norm.clamp_min(1e-8)
    touch=valid&(sep<=tolerance)
    native_gap=torch.where(valid,sep,torch.full_like(sep,float('inf'))).amin(-1)
    vp=velocity[:,None]+torch.cross(angular[:,None].expand_as(p),p-com[:,None],dim=-1)
    lateral= axle(quaternion)[:,None]
    lateral=lateral-(lateral*normal).sum(-1,keepdim=True)*normal
    length=lateral.norm(dim=-1,keepdim=True)
    fault |= (touch&(length[...,0]<1e-8)).any(-1)
    lateral=lateral/length.clamp_min(1e-8)
    lat=(vp*lateral).sum(-1)
    slip=(torch.where(touch,lat.square(),0.).sum(-1)/touch.sum(-1).clamp_min(1)).sqrt()
    return dict(touching=touch.any(-1), unknown=(~touch.any(-1))&(mask&~valid).any(-1),
                min_gap=native_gap, lateral_slip_rms=slip, fault=fault)


def objective(samples, mg, cfg):
    """Bounded reward components. Shape [world, substep, wheel]."""
    import torch
    contact=samples['touching'].bool()
    both=contact.all(-1)&(samples['post_gap']<=cfg['contact_tolerance_m']).all(-1)
    continuous=both.all(-1)
    loss=1-both.float()
    gap=(samples['post_gap']-cfg['contact_tolerance_m']).clamp_min(0)/cfg['gap_scale_m']
    gap=gap.clamp(0,1).amax(-1)
    f=cfg['friction'];fn=samples['normal_fz'].clamp_min(0)
    load=(fn/(mg[:,None,None]*f['load_full_fraction_mg'])).clamp(0,1)
    rho=samples['utilization_proxy']
    near=torch.exp(-((rho-1)/f['rho_width']).square())*(rho<=f['rho_upper_screen'])
    slip=((samples['lateral_slip_rms']-f['slip_threshold_m_s'])/f['slip_ramp_m_s']).clamp(0,1)
    power=((-samples['friction_anchor_power']-f['power_threshold_W'])/f['power_ramp_W']).clamp(0,1)
    brake=(-(samples['friction_force_w'][...,:2]*samples['root_velocity_xy'][:,:,None]).sum(-1)/f['braking_power_ramp_W']).clamp(0,1)
    per_wheel=contact*load*near*slip*power*brake*(samples['opposing_sliding_anchors']>0)
    task=torch.exp(-samples['xy_error_sq']/f['xy_scale_m']**2-samples['velocity_error_sq']/f['velocity_scale_m_s']**2)
    floor=math.cos(math.radians(f['zero_at_tilt_deg']))
    upright=((samples['up']-floor)/(1-floor)).clamp(0,1)
    per_step=per_wheel.amax(-1)*task*upright*samples['braking_phase']
    angular=(samples['angular_acceleration_sq']/cfg['angular_acceleration_scale_rad_s2']**2).clamp(0,1).mean(-1)
    return dict(continuous=continuous.float(), contact_loss=.5*(loss.amax(-1)+loss.mean(-1)),
                clearance=.5*(gap.amax(-1)+gap.mean(-1)), sliding=continuous*per_step.mean(-1),
                angular_acceleration=angular, both_contact_fraction=both.float().mean(-1))


class BilateralWindow(SupportWindow):
    def __init__(self, env):
        import torch
        super().__init__(env)
        self.cfg=env.cfg.hockey_protocol['bilateral_sliding']
        self.robot=env.scene['robot'];self.ids=[self.robot.body_names.index(n) for n in WHEELS]
        self.mu=env.mjlab_recipe_mu.to(device=env.device).clone()
        self.mu_static=getattr(env,'mjlab_recipe_mu_static',env.mjlab_recipe_mu).to(device=env.device).clone()
        if not bool(torch.isfinite(self.mu).all()&(self.mu>0).all()):raise ValueError('Invalid randomized friction')
        shape=(env.num_envs,8,2)
        self.values={k:torch.zeros(shape,device=env.device) for k in ('touching','post_gap','normal_fz','utilization_proxy',
            'lateral_slip_rms','friction_anchor_power','opposing_sliding_anchors')}
        self.values['friction_force_w']=torch.zeros((*shape,3),device=env.device)
        self.values['root_velocity_xy']=torch.zeros((env.num_envs,8,2),device=env.device)
        for key in ('xy_error_sq','velocity_error_sq','up','braking_phase','angular_acceleration_sq'):
            self.values[key]=torch.zeros((env.num_envs,8),device=env.device)
        self.fault=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.fault_reasons={key:torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
                            for key in ('geometry_buffer','force_buffer','alignment','nonfinite')}
        self.alignment_details=torch.zeros((env.num_envs,8,2,4),device=env.device)
        self.max_alignment_error=torch.zeros((),device=env.device)
        self.geometry_samples=torch.zeros((),device=env.device,dtype=torch.int64)
        self.total_contact=torch.zeros((),device=env.device);self.total_world_steps=0
        self.physical_samples=0;self.cache=None

    def capture(self):
        d=self.robot.data
        return {k:getattr(d,k).detach().clone() for k in ('body_pos_w','body_com_pos_w','body_com_lin_vel_w',
            'body_com_ang_vel_w','body_quat_w','root_pos_w','root_quat_w','root_com_lin_vel_w','root_com_ang_vel_w')}

    def step(self,*args,**kwargs):
        self.fault.zero_();self.cache=None
        for value in self.fault_reasons.values():value.zero_()
        for value in self.values.values():value.zero_()
        self.before=self.capture()
        self.phase=self.env.command_manager.get_term('motion').time_steps.clone()
        result=super().step(*args,**kwargs)
        if bool(self.fault.any()):
            import json
            bad=self.fault.nonzero().flatten()[:8]
            detail=dict(physical_samples=self.physical_samples,env_ids=bad.tolist(),
                reasons={key:value[bad].tolist() for key,value in self.fault_reasons.items()},
                alignment_columns=['native_gap','pre_gap','post_gap','absolute_error'],
                alignment=self.alignment_details[bad].tolist())
            raise RuntimeError('Invalid bilateral native contact/timing sample: '+json.dumps(detail))
        self.total_contact+=self.values['touching'].bool().all(-1).float().sum()
        self.total_world_steps+=self.env.num_envs*8
        return result

    def update(self,*args,**kwargs):
        import torch
        previous=self.samples
        result=super().update(*args,**kwargs)
        if self.samples==previous:return result
        k=self.samples-1;pre=self.before;post=self.capture()
        c=self.env.command_manager.get_term('motion');cfg=self.cfg
        for side,(sensor,b) in enumerate(zip(self.sensors,self.ids)):
            view=sensor.contact_physx_view
            if str(self.env.device).startswith('cuda'):torch.cuda.synchronize(self.env.device)
            normal=tuple(v.clone() for v in view.get_contact_data(.0025))
            if str(self.env.device).startswith('cuda'):torch.cuda.synchronize(self.env.device)
            friction=tuple(v.clone() for v in view.get_friction_data(.0025))
            if str(self.env.device).startswith('cuda'):torch.cuda.synchronize(self.env.device)
            com,vel,omega,q=[pre[key][:,b] for key in ('body_com_pos_w','body_com_lin_vel_w','body_com_ang_vel_w','body_quat_w')]
            geo=geometric_sample(normal,com,vel,omega,q,cfg['contact_tolerance_m'])
            f=boundary_sample(normal,friction,com,vel,omega,q,self.mu)
            self.fault |= geo['fault']|f['fault']
            self.fault_reasons['geometry_buffer'] |= geo['fault']
            self.fault_reasons['force_buffer'] |= f['fault']
            position=pre['body_pos_w'][:,b]-self.env.scene.env_origins
            pre_gap=cylinder_gap(position,q,cfg['geometry'],side)
            post_gap=cylinder_gap(post['body_pos_w'][:,b]-self.env.scene.env_origins,post['body_quat_w'][:,b],cfg['geometry'],side)
            measured=torch.isfinite(geo['min_gap'])
            error=torch.where(measured,(geo['min_gap']-pre_gap).abs(),0.)
            self.max_alignment_error=torch.maximum(self.max_alignment_error,error.max())
            self.geometry_samples+=measured.sum()
            self.fault |= error>cfg['geometry_alignment_max_error_m']
            self.fault_reasons['alignment'] |= error>cfg['geometry_alignment_max_error_m']
            self.alignment_details[:,k,side]=torch.stack((geo['min_gap'],pre_gap,post_gap,error),-1)
            self.values['touching'][:,k,side]=geo['touching']
            self.values['post_gap'][:,k,side]=post_gap
            for key in ('normal_fz','friction_anchor_power','friction_force_w','opposing_sliding_anchors'):
                self.values[key][:,k,side]=f[key]
            # Native aggregate force ratio with numerical zero-load safeguard only.
            self.values['utilization_proxy'][:,k,side]=torch.where(f['normal_fz']>1e-6,
                f['friction_force_w'][...,:2].norm(dim=-1)/(self.mu*f['normal_fz'].clamp_min(1e-6)),0.)
            self.values['lateral_slip_rms'][:,k,side]=geo['lateral_slip_rms']
        i0=self.phase.clamp(max=c.motion.time_step_total-1);i1=(i0+1).clamp(max=c.motion.time_step_total-1)
        alpha=k/8
        rp=torch.lerp(c.motion.body_pos_w[i0,0,:2],c.motion.body_pos_w[i1,0,:2],alpha)
        rv=torch.lerp(c.motion.body_lin_vel_w[i0,0,:2],c.motion.body_lin_vel_w[i1,0,:2],alpha)
        v=pre['root_com_lin_vel_w'][:,:2];q=pre['root_quat_w']
        self.values['root_velocity_xy'][:,k]=v
        self.values['xy_error_sq'][:,k]=(pre['root_pos_w'][:,:2]-self.env.scene.env_origins[:,:2]-rp).square().sum(-1)
        self.values['velocity_error_sq'][:,k]=(v-rv).square().sum(-1)
        self.values['up'][:,k]=1-2*(q[:,1].square()+q[:,2].square())
        phase=self.phase+(k+.5)/8;protocol=self.env.cfg.hockey_protocol
        self.values['braking_phase'][:,k]=(phase>=round(protocol['brake_start_s']/.02))&(phase<round(protocol['brake_end_s']/.02))
        self.values['angular_acceleration_sq'][:,k]=((post['root_com_ang_vel_w']-pre['root_com_ang_vel_w'])/.0025).square().sum(-1)
        for value in self.values.values():
            bad=~torch.isfinite(value[:,k]).reshape(self.env.num_envs,-1).all(-1)
            self.fault |= bad
            self.fault_reasons['nonfinite'] |= bad
        self.before=post;self.physical_samples+=1
        return result

    def components(self):
        if self.active and self.samples!=8:raise RuntimeError('Bilateral reward consumed before eight physics samples')
        if self.cache is None:self.cache=objective(self.values,self.mg,self.cfg)
        return self.cache

    def audit(self):
        return dict(native_physical_samples=self.physical_samples,geometry_samples=int(self.geometry_samples),
            native_vs_pre_geometry_max_error_m=float(self.max_alignment_error),
            bilateral_contact_fraction=float(self.total_contact/max(1,self.total_world_steps)),
            contact_is_force_independent=True, equal_load_target=False, pre_step_velocity=True,
            force_scope='Average over following physics step; anchor power is a proxy, not integrated work',
            mu_min=float(self.mu.min()),mu_max=float(self.mu.max()),
            mu_static_min=float(self.mu_static.min()),mu_static_max=float(self.mu_static.max()),
            sliding_reward_capacity_coefficient='dynamic_mu',
            static_dynamic_solver_state_observed=False,fault_seen=bool(self.fault.any()))


def reward_term(env,key):
    import torch
    window=getattr(env,'hockey_support_window',None)
    if not isinstance(window,BilateralWindow):return torch.zeros(env.num_envs,device=env.device)
    terms=window.components()
    if key in terms:return terms[key]
    # Existing positive terms retain their definitions and weights except the
    # declared leg-imitation relaxation. Whole-window bilateral eligibility is
    # additional; it cannot be offset by the other positive rewards.
    return base_rewards(env)[key]*terms['continuous']


def configure(cfg,protocol):
    from isaaclab.managers import RewardTermCfg
    spec=protocol['bilateral_sliding']
    for name in SENSORS:getattr(cfg.scene,name).track_contact_points=True
    for key in ('xy','velocity','heading','yaw_rate','leg_pose','wheel_axis','beta','stop_hold'):
        term=getattr(cfg.rewards,'hockey_'+key)
        term.func=reward_term;term.params={'key':key}
    for key,weight in spec['weights'].items():
        setattr(cfg.rewards,'bilateral_'+key,RewardTermCfg(func=reward_term,weight=weight,params={'key':key}))
