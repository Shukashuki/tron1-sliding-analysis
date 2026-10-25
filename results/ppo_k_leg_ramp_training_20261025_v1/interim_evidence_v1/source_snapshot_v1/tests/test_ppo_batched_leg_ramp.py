from types import SimpleNamespace as NS
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ppo_batched_leg_ramp import BatchedLegRamp


class Term:
    def __init__(self,robot,ids,n):
        self._asset,self._joint_ids=robot,ids
        self.processed_actions=torch.zeros((n,len(ids)))
    def apply_actions(self):raise AssertionError('Original leg writer must not bypass ramp')
    def reset(self,ids=None):
        self.processed_actions[slice(None) if ids is None else ids]=0


def fixture(n=3):
    robot=NS(data=NS(joint_pos=torch.zeros(n,8),joint_pos_target=torch.zeros(n,8)))
    def write(command,*,joint_ids):robot.data.joint_pos_target[:,joint_ids]=command
    robot.set_joint_position_target=write
    leg=Term(robot,[0,2,4,1,3,5],n);wheel=Term(robot,[6,7],n)
    terms={'joint_pos':leg,'wheel_effort':wheel}
    env=NS(device='cpu',num_envs=n,physics_dt=.0025,step_dt=.02,cfg=NS(decimation=8),
           scene={'robot':robot},action_manager=NS(get_term=terms.__getitem__),_sim_step_counter=10)
    owner=NS(env=env,robot=robot,inside_step=False,samples=0,total_samples=0)
    ramp=BatchedLegRamp(env,owner,allow_cpu_fixture=True)
    return env,owner,ramp


def window(env,owner,ramp,target):
    owner.samples=0;owner.inside_step=True;ramp.term.processed_actions.copy_(target)
    for phase in range(8):
        env._sim_step_counter+=1;ramp.term.apply_actions()
        owner.samples+=1;owner.total_samples+=1
    owner.inside_step=False


def test_two_windows_and_wheels_and_raw_targets_unchanged():
    e,o,r=fixture();r.wheels.processed_actions.fill_(2.)
    target=torch.ones(3,6);window(e,o,r,target)
    assert torch.equal(r.window_targets,torch.arange(1,9)[None,:,None].expand(3,8,6)/8.)
    assert torch.equal(r.term.processed_actions,target)
    assert torch.equal(r.wheels.processed_actions,torch.full((3,2),2.))
    window(e,o,r,torch.full((3,6),-1.))
    assert torch.equal(r.start,target) and torch.equal(r.last_applied,-target)
    assert r.audit()['physical_applications']==16


def test_partial_and_full_reset_do_not_carry_previous_episode_commands():
    e,o,r=fixture();window(e,o,r,torch.ones(3,6))
    r.term.reset(torch.tensor([1]));e.scene['robot'].data.joint_pos[1]=.25
    window(e,o,r,torch.zeros(3,6))
    assert torch.equal(r.start[1],torch.full((6,),.25))
    assert torch.equal(r.start[[0,2]],torch.ones(2,6))
    r.term.reset();e.scene['robot'].data.joint_pos.fill_(-.3)
    window(e,o,r,torch.ones(3,6));assert torch.equal(r.start,torch.full((3,6),-.3))
    assert r.audit()['reset_world_events']==4


def test_duplicate_or_nonphysical_application_rejected():
    e,o,r=fixture()
    with pytest.raises(AssertionError):r.term.apply_actions()
    o.inside_step=True;e._sim_step_counter+=1;r.term.apply_actions()
    with pytest.raises(AssertionError):r.term.apply_actions()


def test_changed_held_target_is_latched_as_fault():
    e,o,r=fixture();o.inside_step=True
    for phase in range(8):
        o.samples=phase;e._sim_step_counter+=1
        r.term.processed_actions.fill_(float(phase>0));r.term.apply_actions();o.total_samples+=1
    o.inside_step=False
    with pytest.raises(AssertionError,match='held target'):r.audit()


def test_midwindow_reset_rejected():
    e,o,r=fixture();o.inside_step=True;o.samples=4
    with pytest.raises(AssertionError):r.term.reset(torch.tensor([0]))
