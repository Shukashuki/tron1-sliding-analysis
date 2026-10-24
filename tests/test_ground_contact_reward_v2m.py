"""M arithmetic only; no training or closed-loop performance claim."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ground_contact_reward_v2m as mod


class GapWeightMTests(unittest.TestCase):
    def test_only_two_parent_outputs_change_and_exact_cost_increase(self):
        rng=np.random.default_rng(42)
        force=rng.uniform(0.,120.,(4,8,2))
        before=mod.previous.reward_numpy(force,200.,.8,.9,.7,1.)
        after=mod.reward_numpy(force,200.,.8,.9,.7,1.)
        for name in before.keys()-{"raw_reward","support_gap_penalty"}:
            np.testing.assert_array_equal(after[name],before[name])
        np.testing.assert_array_equal(after["K_support_gap_penalty"],before["support_gap_penalty"])
        np.testing.assert_array_equal(after["K_raw_reward_before_gap_weight_change"],before["raw_reward"])
        np.testing.assert_allclose(after["raw_reward"],before["raw_reward"]-3.*before["support_gap"])
        np.testing.assert_array_equal(after["support_gap_penalty"],4.*before["support_gap"])

    def test_full_load_single_double_and_alternating_support_unchanged(self):
        single=np.tile([200.,0.],(8,1));double=np.full((8,2),100.)
        alternating=single.copy();alternating[1::2]=[0.,200.]
        for force in [single,double,alternating]:
            with self.subTest(force=force.tolist()):
                value=mod.reward_numpy(force,200.,1.,1.,1.,1.)
                self.assertEqual(value["raw_reward"],1.)
                self.assertEqual(value["support_gap_penalty"],0.)

    def test_zero_load_and_one_missing_sample_use_fourfold_ungated_cost(self):
        for missing in [1,3,8]:
            force=np.tile([200.,0.],(8,1));force[:missing]=0.
            for beta in [0.,.7,1.]:
                with self.subTest(missing=missing,beta=beta):
                    value=mod.reward_numpy(force,200.,1.,1.,beta,1.)
                    self.assertEqual(value["support_gap_penalty"],4.)
                    self.assertEqual(value["raw_reward"],value["G_gated_support_bonus"]-4.)
        self.assertEqual(mod.reward_numpy(np.zeros((8,2)),200.,1.,1.,1.,1.)["raw_reward"],-4.)

    def test_force_mapping_threshold_and_external_weight_unchanged(self):
        for load in [0.,5.,5.0001,25.,100.,200.]:
            force=np.tile([load,0.],(8,1))
            value=mod.reward_numpy(force,200.,1.,1.,1.,1.)
            self.assertAlmostEqual(value["support_gap"],float(np.clip((100.-load)/95.,0.,1.)))
            self.assertAlmostEqual(value["support_gap_penalty"],4.*float(np.clip((100.-load)/95.,0.,1.)))
        self.assertEqual(mod.DEFAULT_SETTINGS["support_force_threshold_n"],5.)
        self.assertEqual(mod.DEFAULT_SETTINGS["external_reward_weight"],.4)
        metadata=mod.formula_metadata()
        self.assertEqual(metadata["raw_reward_bounds"],[-4.,1.])
        self.assertFalse(metadata["force_slew_cost_added"])
        self.assertFalse(metadata["continuous_contact_or_task_success_certified"])

    def test_real_torch_parity_float32_float64_and_hotpath_no_host_reads(self):
        import torch
        rng=np.random.default_rng(9)
        for dtype,td in [(np.float32,torch.float32),(np.float64,torch.float64)]:
            force=rng.uniform(0.,140.,(4,8,2)).astype(dtype)
            expected=mod.reward_numpy(force,200.,.8,.9,.7,1.)
            tensor=torch.tensor(force,dtype=td)
            with patch.object(Path,"read_bytes",side_effect=AssertionError("file read")), \
                 patch.object(torch.Tensor,"item",side_effect=AssertionError("host item")), \
                 patch.object(torch.Tensor,"__bool__",side_effect=AssertionError("host bool")):
                actual=mod.reward_torch(tensor,200.,.8,.9,.7,1.,validate=False)
            for name in expected:
                np.testing.assert_allclose(actual[name].numpy(),expected[name],rtol=3.e-6,atol=3.e-6)

    def test_invalid_window_poison_and_frozen_settings(self):
        import torch
        force=torch.full((8,2),100.,dtype=torch.float64);force[2,0]=float("nan")
        with self.assertRaises(ValueError):
            mod.reward_numpy(force.numpy(),200.,1.,1.,1.,1.)
        result=mod.reward_torch(force,200.,1.,1.,1.,1.,validate=False)
        self.assertTrue(torch.isnan(result["raw_reward"]))
        for key in mod.DEFAULT_SETTINGS:
            bad=copy.deepcopy(mod.DEFAULT_SETTINGS)
            bad[key]="changed" if isinstance(bad[key],str) else bad[key]+.1
            with self.subTest(key=key),self.assertRaises(ValueError):
                mod.validate_settings(bad)
        mod.verify_base()


if __name__=="__main__":
    unittest.main()

