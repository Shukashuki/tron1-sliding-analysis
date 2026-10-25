"""Temporal/partial-reset and native-buffer fixtures, without claiming PhysX."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from physx_batched_tire import BatchedTireRuntime


def fixture():
    n = 3
    value = BatchedTireRuntime.__new__(BatchedTireRuntime)
    value.env = NS(_sim_step_counter=0)
    value.owner = NS(inside_step=True)
    value.n = n; value.nonlinear = True; value.ids = [1,3]
    value.body_ids = torch.tensor(value.ids)
    pos = torch.zeros((n,5,3)); pos[...,2] = .1
    vel = torch.zeros_like(pos); vel[...,0] = 1.; vel[...,1] = .5
    omega = torch.zeros_like(pos); omega[...,1] = 20.
    q = torch.zeros((n,5,4)); q[...,0] = 1.
    value.robot = NS(data=NS(body_com_pos_w=pos,body_com_lin_vel_w=vel,
        body_com_ang_vel_w=omega,body_link_quat_w=q),
        _external_force_b=torch.zeros_like(pos),_external_torque_b=torch.zeros_like(pos),
        _external_wrench_positions_b=torch.zeros_like(pos))
    f = torch.zeros((n,1,1,3)); f[...,2] = 100.
    point = torch.zeros_like(f)
    value.sensors = [NS(data=NS(force_matrix_w=f.clone(),contact_pos_w=point.clone())) for _ in range(2)]
    value.normal = torch.zeros((n,2,3)); value.normal[...,2] = 1.
    value.mu = torch.full((n,1),.7)
    value.ready = torch.zeros(n,dtype=torch.bool)
    value.fault = torch.zeros((n,2),dtype=torch.bool)
    for name in ('minimum_power','maximum_power','maximum_capacity_excess','maximum_force'):
        setattr(value,name,torch.zeros(()))
    value.loaded_count = torch.zeros(2,dtype=torch.long)
    value.applied_count = torch.zeros(2,dtype=torch.long)
    value.calls = value.reset_calls = value.reset_worlds = value.last_counter = 0
    value.writes = []
    value.original_write = lambda: value.writes.append(value.robot._external_force_b.clone())
    def reset(ids=None):
        ids = slice(None) if ids is None else ids
        for name in ('_external_force_b','_external_torque_b','_external_wrench_positions_b'):
            getattr(value.robot,name)[ids] = 0.
    value.original_reset = reset
    return value


def step(v):
    v.env._sim_step_counter += 1; v.write()


def test_reset_drops_only_reset_world_history_and_keeps_other_world_force():
    v = fixture(); step(v)
    assert torch.count_nonzero(v.writes[-1]) == 0
    step(v); previous = v.writes[-1]
    assert (previous[:,v.ids,0] > 0).all() and (previous[:,v.ids,1] < 0).all()
    v.reset(torch.tensor([1])); step(v)
    assert torch.equal(v.writes[-1][[0,2]], previous[[0,2]])
    assert torch.count_nonzero(v.writes[-1][1]) == 0
    step(v)
    assert torch.equal(v.writes[-1], previous)
    assert v.calls == 4 and v.reset_calls == 1 and v.reset_worlds == 1
    assert not v.fault.any()
    assert torch.count_nonzero(v.writes[-1][:,[0,2,4]]) == 0


def test_reset_forward_flush_is_not_an_additional_physical_application():
    v = fixture(); step(v); step(v)
    v.reset(); v.write()
    assert v.calls == 2 and not v.ready.any()
    step(v)
    assert v.calls == 3 and torch.count_nonzero(v.writes[-1]) == 0
    v.owner.inside_step = False; v.write()
    assert v.calls == 3


def test_skipped_native_step_rejected_and_loaded_nan_is_retained_as_fault():
    v = fixture(); v.env._sim_step_counter = 2
    with pytest.raises(AssertionError): v.write()
    v.env._sim_step_counter = 0; step(v)
    v.sensors[0].data.contact_pos_w[1] = float('nan'); step(v)
    assert v.fault[1,0] and torch.isfinite(v.writes[-1]).all()


def test_per_world_mu_scales_capacity_without_mutating_other_worlds():
    v = fixture(); v.mu = torch.tensor([[.3],[.6],[1.2]])
    step(v); step(v)
    F = v.writes[-1][:,v.ids]
    torch.testing.assert_close(F[1],2*F[0]);torch.testing.assert_close(F[2],4*F[0])
