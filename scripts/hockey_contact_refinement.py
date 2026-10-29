"""Optional reward adapters; absent from and inactive in frozen v2 runs."""
from __future__ import annotations

from hockey_contact_search import DenseContactWindow,objective
from hockey_contact_refinement_math import bilateral_substep_stop,angular_pseudo_huber


def validate_settings(settings):
    expected={'kind','stop_term','angular_term'}
    if not isinstance(settings,dict) or set(settings)!=expected or settings['kind']!='hockey_contact_refinement_v1':
        raise ValueError('Require explicit contact-refinement settings')
    if settings['stop_term'] not in ('unchanged','bilateral_substeps','bilateral_substeps_xyz','bilateral_substeps_xyz_huber'):
        raise ValueError('Unknown stop-reward refinement')
    if settings['angular_term'] not in ('bounded_quadratic','pseudo_huber'):
        raise ValueError('Unknown angular-reward refinement')
    return dict(settings)


def refined_components(values, mg, cfg, protocol, phase):
    """Contact and force-corroborated slide terms retain their prior definitions."""
    settings=validate_settings(protocol['reward_refinement'])
    result=objective(values,mg,cfg)
    if settings['angular_term']=='pseudo_huber':
        result['angular_acceleration']=angular_pseudo_huber(values['angular_acceleration_sq'],
            cfg['angular_acceleration_scale_rad_s2'])
    if settings['stop_term'] in ('bilateral_substeps','bilateral_substeps_xyz','bilateral_substeps_xyz_huber'):
        xyz=settings['stop_term'] in ('bilateral_substeps_xyz','bilateral_substeps_xyz_huber')
        pre=values['root_velocity_xyz_pre'] if xyz else values['root_velocity_xy']
        post=values['root_velocity_xyz_post'] if xyz else values['root_velocity_xy_post']
        result['stop_hold_refined']=bilateral_substep_stop(values['touching'],values['post_gap'],
            pre,post,values['up'],values['up_post'],phase,velocity_dimensions=3 if xyz else 2,
            speed_form='separate_pseudo_huber' if settings['stop_term']=='bilateral_substeps_xyz_huber' else 'exponential',
            contact_tolerance_m=cfg['contact_tolerance_m'],speed_scale_m_s=protocol['reward_scales']['stop_speed_m_s'],
            tilt_scale_rad=protocol['reward_scales']['stop_tilt_rad'])
    return result


class RefinedContactWindow(DenseContactWindow):
    def __init__(self,env):
        import torch
        super().__init__(env)
        self.refinement=validate_settings(env.cfg.hockey_protocol['reward_refinement'])
        self.values['root_velocity_xy_post']=torch.zeros((env.num_envs,8,2),device=env.device)
        self.values['up_post']=torch.zeros((env.num_envs,8),device=env.device)
        if self.refinement['stop_term'] in ('bilateral_substeps_xyz','bilateral_substeps_xyz_huber'):
            for key in ('root_velocity_xyz_pre','root_velocity_xyz_post'):
                self.values[key]=torch.zeros((env.num_envs,8,3),device=env.device)

    def update(self,*args,**kwargs):
        import torch
        previous=self.samples
        xyz='root_velocity_xyz_pre' in self.values
        pre=getattr(self,'before',None) if xyz else None
        result=super().update(*args,**kwargs)
        if self.samples==previous:return result
        k=self.samples-1;post=self.before
        self.values['root_velocity_xy_post'][:,k]=post['root_com_lin_vel_w'][:,:2]
        q=post['root_quat_w'];self.values['up_post'][:,k]=1-2*(q[:,1].square()+q[:,2].square())
        keys=['root_velocity_xy_post','up_post']
        if xyz:
            if pre is None:raise RuntimeError('Missing pre-step state for three-dimensional stop reward')
            self.values['root_velocity_xyz_pre'][:,k]=pre['root_com_lin_vel_w']
            self.values['root_velocity_xyz_post'][:,k]=post['root_com_lin_vel_w']
            keys.extend(('root_velocity_xyz_pre','root_velocity_xyz_post'))
        for key in keys:
            bad=~torch.isfinite(self.values[key][:,k]).reshape(self.env.num_envs,-1).all(-1)
            self.fault|=bad;self.fault_reasons['nonfinite']|=bad
        return result

    def components(self):
        if self.active and self.samples!=8:raise RuntimeError('Refined reward consumed before eight physical substeps')
        if self.cache is None:
            command=self.env.command_manager.get_term('motion');protocol=self.env.cfg.hockey_protocol
            index=(self.phase+1).clamp(max=command.motion.time_step_total-1)
            hold=index>=round(protocol['brake_end_s']/self.env.step_dt)
            self.cache=refined_components(self.values,self.mg,self.cfg,protocol,hold)
        return self.cache


def stop_reward(env):
    import torch
    window=getattr(env,'hockey_support_window',None)
    if not isinstance(window,RefinedContactWindow):return torch.zeros(env.num_envs,device=env.device)
    return window.components()['stop_hold_refined']


def configure(env_cfg,protocol):
    settings=validate_settings(protocol['reward_refinement'])
    if protocol.get('phase_clock')!='integer_reference_frames':
        raise ValueError('Refinement preserves the integer reference clock')
    if settings['stop_term'] in ('bilateral_substeps','bilateral_substeps_xyz','bilateral_substeps_xyz_huber'):
        term=env_cfg.rewards.hockey_stop_hold
        term.func=stop_reward;term.params={}
    return settings
