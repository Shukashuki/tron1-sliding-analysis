"""SAC successor capture for the unchanged GPU task step/reset sequence."""


def make_gpu_terminal_class(base):
    import torch

    class GPUWithTerminal(base):
        def observations(self, state=None):
            result = super().observations(state)
            if (getattr(self, '_sac_capture', False) and self._sac_final is None
                    and state is not None):
                # First observation inside base.step is the pre-reset physical
                # successor. Its reference must advance as an ordinary next
                # transition would, clamped at the end of the reference.
                phases = self.phases.clone()
                try:
                    self.phases.copy_((phases+1).clamp(max=self.frames-1))
                    self._sac_final = {k:v.clone() for k,v in super().observations(state).items()}
                    self._sac_final_frames = self.phases.clone()
                finally:
                    self.phases.copy_(phases)
            return result

        def step(self, actions):
            self._sac_capture, self._sac_final = True, None
            try:
                obs, reward, terminated, truncated, extras = super().step(actions)
            finally:
                self._sac_capture = False
            if self._sac_final is None:
                raise RuntimeError('GPU task did not expose its physical successor before reset')
            done = terminated | truncated
            extras = dict(extras)
            extras['bootstrap_observation'] = {
                k:torch.where(done[:,None], self._sac_final[k], v) for k,v in obs.items()}
            extras['final_observation_env_ids'] = done.nonzero(as_tuple=False).squeeze(-1)
            extras['final_reference_frames'] = self._sac_final_frames[done]
            return obs, reward, terminated, truncated, extras

    return GPUWithTerminal
