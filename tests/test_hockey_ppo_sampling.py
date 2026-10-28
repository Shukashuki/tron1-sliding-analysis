"""Successful finite episodes must never increase failure-sampling density."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from hockey_ppo_sampling import record_fall_bins


def command(falls):
    manager = NS(terminated=torch.ones(4, dtype=torch.bool),
                 get_term=lambda name: torch.tensor(falls, dtype=torch.bool))
    return NS(_env=NS(termination_manager=manager), time_steps=torch.tensor([138, 138, 69, 138]),
              bin_count=14, motion=NS(time_step_total=139), _current_bin_failed=torch.zeros(14))


class SamplingTests(unittest.TestCase):
    def test_successful_clip_ends_are_not_failures(self):
        c = command([False]*4)
        record_fall_bins(c, torch.arange(4))
        self.assertTrue(torch.equal(c._current_bin_failed, torch.zeros(14)))

    def test_count_falls_at_terminal_and_midclip_without_successes(self):
        c = command([False, True, True, False])
        record_fall_bins(c, torch.arange(4))
        self.assertEqual(c._current_bin_failed.sum(), 2)
        self.assertEqual(c._current_bin_failed[13], 1)
        self.assertEqual(c._current_bin_failed[6], 1)

    def test_reset_subset_uses_matching_environment_ids(self):
        c = command([True, False, True, False])
        record_fall_bins(c, torch.tensor([1, 3]))
        self.assertEqual(c._current_bin_failed.sum(), 0)
        record_fall_bins(c, torch.tensor([2]))
        self.assertEqual(c._current_bin_failed.sum(), 1)
        self.assertEqual(c._current_bin_failed[6], 1)


if __name__ == "__main__":
    unittest.main()
