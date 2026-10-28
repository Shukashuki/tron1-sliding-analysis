"""Control-only adversarial cases; no synthetic trajectory is called physics."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_control_replay import reconstruct_residual,replay_actions
from hockey_mppi_math import stop_basis


class ConstantActor:
    def __call__(self,x):
        return torch.tensor([3.,-4.,.1,.2,.3,.4,.5,.6])[None].repeat(len(x),1)


class ReplayContracts(unittest.TestCase):
    def test_independent_hermite_reconstruction_matches_torch_plan_and_full_clock(self):
        mean=np.random.default_rng(24).normal(size=(12,8))
        native=torch.einsum('tk,nka->tna',torch.tensor(stop_basis(139,12),dtype=torch.float32),
                            torch.tensor(mean[None],dtype=torch.float32)).numpy()[:,0]
        independent=reconstruct_residual(mean,139,139)
        np.testing.assert_allclose(native,independent,atol=1e-6,rtol=0)
        np.testing.assert_allclose(reconstruct_residual(mean,139,16),independent[:16],atol=1e-14)
        self.assertGreater(np.max(abs(reconstruct_residual(mean,16,16)-independent[:16])),.1)

    def test_replay_requires_bounding_the_prior_before_the_correction(self):
        x=np.zeros((3,51));residual=np.zeros((3,8));residual[:,:2]=[-.2,.2]
        recorded=np.tile([.8,-.8,.1,.2,.3,.4,.5,.6],(3,1))
        report=replay_actions(ConstantActor(),x,recorded,residual)
        self.assertLess(report['maximum_error'],1e-7)
        recorded[:,:2]=[1,-1]  # The old raw-prior decoder must be rejected.
        with self.assertRaisesRegex(ValueError,'replay mismatch'):
            replay_actions(ConstantActor(),x,recorded,residual)

    def test_one_changed_applied_action_is_rejected(self):
        x=np.zeros((139,51));recorded=np.tile([1,-1,.1,.2,.3,.4,.5,.6],(139,1))
        replay_actions(ConstantActor(),x,recorded)
        recorded[91,6]+=.01
        with self.assertRaisesRegex(ValueError,'replay mismatch'):
            replay_actions(ConstantActor(),x,recorded)

    def test_missing_or_invalid_plan_cannot_silently_become_zero(self):
        for mean,frames,steps in [(np.zeros((12,7)),139,139),(np.full((12,8),np.nan),139,139),
                                  (np.zeros((12,8)),139,140)]:
            with self.assertRaises(ValueError):reconstruct_residual(mean,frames,steps)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
