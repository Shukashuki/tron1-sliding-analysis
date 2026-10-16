"""Capture SAC bootstrap observations before Isaac's automatic episode reset.

ManagerBasedRLEnv normally returns reset observations for done environments.
Those observations must never be used as the next state of a truncated SAC
transition. Only the motion command phase is temporarily advanced for capture;
the robot state, dynamics, reward and termination logic are not modified.
"""
from __future__ import annotations


def merge_bootstrap_observations(next_observation, env_ids, final_observation):
    if set(next_observation) != set(final_observation):
        raise ValueError("Final and ordinary observation groups differ")
    result = {}
    for name, value in next_observation.items():
        clone = value.clone() if hasattr(value, "clone") else value.copy()
        final = final_observation[name]
        if final.shape != (len(env_ids), *value.shape[1:]):
            raise ValueError(f"Final observation shape differs for {name}")
        clone[env_ids] = final
        result[name] = clone
    return result


def make_terminal_observation_env_class():
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    class TerminalObservationEnv(ManagerBasedRLEnv):
        def __init__(self, *args, **kwargs):
            self._capture_final_observation = False
            self._captured_final = None
            super().__init__(*args, **kwargs)
            # compute(update_history=False) is a valid final-state observation
            # only for the current memoryless, uncorrupted TRON1 observation.
            for group in vars(self.cfg.observations).values():
                if not hasattr(group, "enable_corruption"):
                    continue
                if group.enable_corruption or (getattr(group, "history_length", None) or 0) > 0:
                    raise ValueError("Terminal capture requires uncorrupted observations without history")
                for term in vars(group).values():
                    if not hasattr(term, "func"):
                        continue
                    if (getattr(term, "history_length", 0) or 0) > 0 or getattr(term, "modifiers", None):
                        raise ValueError("Terminal capture does not support history/stateful observation modifiers")

        def _reset_idx(self, env_ids):
            if self._capture_final_observation and len(env_ids):
                if self._captured_final is not None:
                    raise RuntimeError("Unexpected multiple reset calls in one control step")
                command = self.command_manager.get_term("motion")
                old_frames = command.time_steps[env_ids].clone()
                final_frames = (old_frames + 1).clamp(max=command.motion.time_step_total-1)
                try:
                    command.time_steps[env_ids] = final_frames
                    command._refresh_relative_targets()
                    observed = self.observation_manager.compute(update_history=False)
                    final = {name: value[env_ids].clone() for name, value in observed.items()}
                finally:
                    command.time_steps[env_ids] = old_frames
                    command._refresh_relative_targets()
                self._captured_final = (env_ids.clone(), final, final_frames.clone())
            super()._reset_idx(env_ids)

        def step(self, actions):
            self._captured_final = None
            self._capture_final_observation = True
            try:
                observation, reward, terminated, truncated, extras = super().step(actions)
            finally:
                self._capture_final_observation = False
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            extras = dict(extras)
            if len(done_ids):
                if self._captured_final is None:
                    raise RuntimeError("Done transition is missing its pre-reset observations")
                ids, final, frames = self._captured_final
                if not torch.equal(ids, done_ids):
                    raise RuntimeError("Captured final observations do not match done environments")
                bootstrap = merge_bootstrap_observations(observation, ids, final)
                extras["final_reference_frames"] = frames
            else:
                bootstrap = observation
                extras["final_reference_frames"] = torch.empty(0, device=self.device, dtype=torch.long)
            extras["bootstrap_observation"] = bootstrap
            extras["final_observation_env_ids"] = done_ids
            return observation, reward, terminated, truncated, extras

    return TerminalObservationEnv
