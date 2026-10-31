"""The SAC successor must not be the autoreset state or old reference phase."""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from matched_sac_terminal import make_gpu_terminal_class


class FakeGPUSequence:
    def __init__(self):
        self.phases=torch.tensor([2,9,4])
        self.frames=10
        self.state=torch.tensor([[10.],[20.],[30.]])
    def observations(self,state=None):
        value=self.state if state is None else state
        return {'policy':torch.cat((value,self.phases[:,None]),dim=1),
                'critic':torch.cat((value,self.phases[:,None]),dim=1)}
    def step(self,actions):
        state=self.state+1
        terminal=self.observations(state)
        terminated=torch.tensor([False,False,False])
        truncated=torch.tensor([True,True,False])
        self.phases=torch.tensor([0,0,5])
        # Includes an ordinary next-state event on the non-reset lane.
        self.state=torch.tensor([[1000.],[2000.],[32.]])
        return self.observations(),torch.zeros(3),terminated,truncated,{'terminal_observation':terminal}


def test_advanced_clamped_terminal_phase_and_actual_successor():
    env=make_gpu_terminal_class(FakeGPUSequence)()
    obs,_,_,_,extras=env.step(None)
    for group in ('policy','critic'):
        torch.testing.assert_close(extras['bootstrap_observation'][group],torch.tensor([[11.,3.],[21.,9.],[32.,5.]]))
        torch.testing.assert_close(obs[group],torch.tensor([[1000.,0.],[2000.,0.],[32.,5.]]))
    torch.testing.assert_close(extras['final_reference_frames'],torch.tensor([3,9]))
    assert not env._sac_capture


def test_capture_does_not_change_underlying_transition_or_reward_observation():
    reference=FakeGPUSequence()
    adapted=make_gpu_terminal_class(FakeGPUSequence)()
    expected=reference.step(None)
    actual=adapted.step(None)
    for key in expected[0]:
        torch.testing.assert_close(actual[0][key],expected[0][key])
        torch.testing.assert_close(actual[4]['terminal_observation'][key],expected[4]['terminal_observation'][key])
    torch.testing.assert_close(adapted.phases,reference.phases)
