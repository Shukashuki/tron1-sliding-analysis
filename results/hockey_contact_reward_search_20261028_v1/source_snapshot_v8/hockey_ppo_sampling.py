"""Failure-only adaptive RSI for a finite-horizon hockey task.

Clip completion is a true PPO terminal but is NOT a failure for adaptive motion
sampling. The inherited BeyondMimic sampler uses the aggregate terminated flag,
which conflates these meanings after a finite-task clip-end term is added.
"""


def record_fall_bins(command, env_ids):
    import torch
    # Do not use termination_manager.terminated: that includes successful ends.
    failed = command._env.termination_manager.get_term("hockey_fall")[env_ids]
    if torch.any(failed):
        bins = torch.clamp((command.time_steps*command.bin_count)//max(command.motion.time_step_total, 1),
                           0, command.bin_count-1)
        command._current_bin_failed[:] = torch.bincount(bins[env_ids][failed], minlength=command.bin_count)


def motion_command_class():
    import math
    import torch
    from isaaclab.utils.math import sample_uniform
    from training.tron1_tracking import TronMotionCommand

    class HockeyMotionCommand(TronMotionCommand):
        def _adaptive_sampling(self, env_ids):
            if self.cfg.deterministic_start:
                return super()._adaptive_sampling(env_ids)
            record_fall_bins(self, env_ids)
            probabilities = self.bin_failed_count+self.cfg.adaptive_uniform_ratio/float(self.bin_count)
            probabilities = torch.nn.functional.pad(probabilities[None, None],
                (0, self.cfg.adaptive_kernel_size-1), mode="replicate")
            probabilities = torch.nn.functional.conv1d(probabilities, self.kernel.view(1, 1, -1)).view(-1)
            probabilities /= probabilities.sum()
            sampled = torch.multinomial(probabilities, len(env_ids), replacement=True)
            self.time_steps[env_ids] = ((sampled+sample_uniform(0., 1., (len(env_ids),), device=self.device))
                                       /self.bin_count*(self.motion.time_step_total-1)).long()
            entropy = -(probabilities*(probabilities+1e-12).log()).sum()
            maximum, index = probabilities.max(dim=0)
            self.metrics["sampling_entropy"][:] = entropy/math.log(self.bin_count)
            self.metrics["sampling_top1_prob"][:] = maximum
            self.metrics["sampling_top1_bin"][:] = index.float()/self.bin_count
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            full = torch.rand(len(ids), device=self.device) < self.cfg.start_fraction
            self.time_steps[ids[full]] = 0

    return HockeyMotionCommand
