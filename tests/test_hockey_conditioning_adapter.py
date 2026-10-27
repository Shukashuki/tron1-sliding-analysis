"""Ensure reference adaptation never filters state or silently rewinds its clock."""
import sys
from pathlib import Path
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest('Run this contract test in the documented Torch environment')

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hockey_conditioning_adapter import ConditioningAdapter


class Spy:
    def eval(self):return self
    def __call__(self,x):self.seen=x.clone();return x[:,:8].clone()


class TestConditioningAdapter(unittest.TestCase):
    def test_only_named_reference_fields_change_and_recorded_input_is_actual(self):
        for mode,sl in [('reference_dq',slice(6,12)),('nominal_command',slice(0,14))]:
            actor=Spy();command=np.arange(42,dtype=np.float32).reshape(3,14)
            adapter=ConditioningAdapter(actor,command,mode).eval()
            for i in range(3):
                obs=torch.arange(51,dtype=torch.float32).reshape(1,51)+100
                expected=obs.clone();expected[:,sl]=torch.from_numpy(command[i,sl])
                action=adapter(obs)
                torch.testing.assert_close(obs,expected)
                torch.testing.assert_close(actor.seen,obs)
                torch.testing.assert_close(action,expected[:,:8])
            with self.assertRaisesRegex(ValueError,'overrun'):adapter(obs)

    def test_multiple_worlds_rejected_without_advancing_clock(self):
        adapter=ConditioningAdapter(Spy(),np.zeros((3,14)),'reference_dq')
        with self.assertRaisesRegex(ValueError,'one environment'):adapter(torch.zeros(2,51))
        self.assertEqual(adapter.calls,0)

    def test_bad_mode_or_nonfinite_conditioning_rejected(self):
        with self.assertRaises(ValueError):ConditioningAdapter(Spy(),np.zeros((3,14)),'freeze_state')
        with self.assertRaises(ValueError):ConditioningAdapter(Spy(),np.full((3,14),np.nan),'nominal_command')


if __name__=='__main__':unittest.main()
