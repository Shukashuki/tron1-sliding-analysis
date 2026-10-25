"""Validate observation-only trace ordering and the first-step coupling rule."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from eval_ppo_k_nonlinear_tire import TireTrace


def test_trace_keeps_force_that_native_writer_received_and_zero_reset_history(tmp_path):
    env=NS(_sim_step_counter=0)
    r=NS(owner=NS(inside_step=True),calls=0,last_counter=0,ids=[0,1],
         ready=torch.zeros(1,dtype=torch.bool),mu=torch.full((1,1),.7))
    pos=torch.zeros((1,2,3));pos[...,2]=.1
    vel=torch.zeros_like(pos);vel[...,0]=1
    q=torch.zeros((1,2,4));q[...,0]=1
    r.robot=NS(data=NS(body_com_pos_w=pos,body_com_lin_vel_w=vel,
        body_com_ang_vel_w=torch.zeros_like(pos),body_link_quat_w=q),
        _external_force_b=torch.zeros_like(pos),_external_torque_b=torch.zeros_like(pos))
    def fragments(dt):
        assert dt==.0025
        return torch.zeros((128,3)),None,torch.zeros((1,1),dtype=torch.int32),torch.zeros((1,1),dtype=torch.int32)
    f=torch.zeros((1,1,1,3));f[...,2]=100
    r.sensors=[NS(data=NS(force_matrix_w=f.clone(),contact_pos_w=torch.zeros_like(f)),
                  contact_physx_view=NS(get_friction_data=fragments)) for _ in range(2)]
    env.physics_dt=.0025
    written=[]
    def native_write():
        if env._sim_step_counter==r.last_counter:return
        r.robot._external_force_b.fill_(float(r.ready[0]))
        r.robot._external_torque_b.fill_(-2*float(r.ready[0]))
        written.append(r.robot._external_force_b.clone())
        r.calls+=1;r.last_counter=env._sim_step_counter;r.ready.fill_(True)
    env.scene=NS(write_data_to_sim=native_write)
    trace=TireTrace(env,r)
    # A reset/forward call at the same native counter has no trajectory row.
    env.scene.write_data_to_sim();assert not trace.rows
    for i in range(2792):
        env._sim_step_counter+=1;env.scene.write_data_to_sim()
    trace.save(tmp_path/'trace.npz')
    with np.load(tmp_path/'trace.npz') as data:
        np.testing.assert_array_equal(data['applied_tangent_force_w_n'],torch.stack(written).numpy()[:,0])
        assert data['history_valid'].shape==(2792,) and not data['history_valid'][0]
        assert data['history_valid'][1:].all()
        assert data['preceding_normal_load_n'][0].sum()==0
        np.testing.assert_array_equal(data['preceding_normal_load_n'][1:],np.full((2791,2),100.))
    trace.rows.pop()
    with pytest.raises(AssertionError):trace.save(tmp_path/'incomplete.npz')
