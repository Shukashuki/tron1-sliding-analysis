"""Frozen car-PPO conditioning adapter; physical states/actions remain unfiltered.

Replacing reference fields changes the controller. The original hockey command
remains the evaluation target; this adapter alone is not posture reproduction.
"""
import numpy as np


MODES=('reference_dq','nominal_command')


class ConditioningAdapter:
    def __init__(self,actor,command,mode):
        import torch
        if mode not in MODES:raise ValueError('Unsupported conditioning mode')
        command=np.asarray(command,dtype=np.float32)
        if command.ndim!=2 or command.shape[1]!=14 or not np.isfinite(command).all():raise ValueError('Invalid command table')
        self.actor=actor;self.command=torch.from_numpy(command.copy());self.mode=mode;self.calls=0

    def eval(self):
        self.actor.eval();return self

    def __call__(self,obs):
        if obs.shape!=(1,51):raise ValueError('Adapter requires one environment and the frozen 51D interface')
        if self.calls>=len(self.command):raise ValueError('Unexpected repeated rollout or command overrun')
        sl=slice(6,12) if self.mode=='reference_dq' else slice(0,14)
        # The engine archives obs after actor inference. Mutating only these
        # documented reference fields ensures it saves the ACTUAL actor inputs.
        obs[:,sl]=self.command[self.calls,sl].to(device=obs.device,dtype=obs.dtype)
        self.calls+=1
        return self.actor(obs)
