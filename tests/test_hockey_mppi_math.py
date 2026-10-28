import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hockey_mppi_math import stop_basis, decode_action, cost_terms


class HockeySearchContracts(unittest.TestCase):
    def fixture(self):
        q=torch.tensor([[1.,0,0,0]])
        state=dict(root_pos=torch.zeros(1,3), root_quat=q, root_velocity=torch.zeros(1,3),
                   wheel_quat=q[:,None].repeat(1,2,1), wheel_velocity=torch.zeros(1,2,3), leg_q=torch.zeros(1,6))
        target=dict(xy=torch.zeros(2),velocity=torch.zeros(2),leg_q=torch.zeros(6),
                    up=torch.tensor([0.,0,1]),yaw=torch.tensor(0.),wheel_heading=torch.zeros(2),
                    wheel_camber=torch.zeros(2),wheel_velocity=torch.zeros(2,2))
        contact=dict(normal_fz=torch.tensor([[100.,0]]),lateral_slip_rms=torch.zeros(1,2),
                     utilization_proxy=torch.zeros(1,2),friction_anchor_power=torch.zeros(1,2))
        return state,target,contact

    def test_residual_can_undo_saturated_actor_and_zero_matches_old_controller(self):
        raw=torch.tensor([[3.,-4.,.2]])
        torch.testing.assert_close(decode_action(raw,torch.zeros_like(raw)),raw.clamp(-1,1))
        torch.testing.assert_close(decode_action(raw,torch.tensor([[-.2,.2,0]])),torch.tensor([[.8,-.8,.2]]))

    def test_terminal_control_remains_available_and_initial_action_is_preserved(self):
        b=stop_basis(139,12)
        np.testing.assert_array_equal(b[0],0)
        self.assertEqual(b[-1,-1],1.)
        np.testing.assert_allclose(b[5:].sum(1),1,atol=1e-14)
        self.assertTrue(np.all(b>=0))

    def test_single_wheel_support_is_allowed_but_no_support_is_penalized(self):
        s,t,c=self.fixture();terms,_=cost_terms(s,t,c,1.)
        self.assertEqual(terms['support_gap'].item(),0)
        c['normal_fz'].zero_();terms,_=cost_terms(s,t,c,1.)
        self.assertEqual(terms['support_gap'].item(),1)

    def test_zero_speed_has_no_beta_penalty_and_wrong_velocity_direction_does(self):
        s,t,c=self.fixture();t['wheel_heading'][:]=1.2
        terms,_=cost_terms(s,t,c,2.5);self.assertEqual(terms['beta'].item(),0)
        t['velocity'][0]=1;s['root_velocity'][0,0]=-1
        terms,_=cost_terms(s,t,c,1.);self.assertGreater(terms['velocity'].item(),30)

    def test_root_height_and_wheel_spin_are_not_posture_errors(self):
        s,t,c=self.fixture();before,_=cost_terms(s,t,c,1.)
        s['root_pos'][:,2]=4.5
        s['wheel_quat'][:]=torch.tensor([np.cos(.7),0,np.sin(.7),0])
        after,_=cost_terms(s,t,c,1.)
        for k in before:torch.testing.assert_close(before[k],after[k])

    def test_slide_requires_load_saturation_material_slip_and_upright_body(self):
        s,t,c=self.fixture();s['wheel_velocity'][0,0,1]=1
        c['lateral_slip_rms'][0,0]=.3;c['utilization_proxy'][0,0]=1;c['friction_anchor_power'][0,0]=-3
        _,slide=cost_terms(s,t,c,1.);self.assertTrue(slide[0,0])
        for key,bad in [('normal_fz',0),('lateral_slip_rms',0),('utilization_proxy',.5),('friction_anchor_power',1)]:
            old=c[key].clone();c[key][0,0]=bad
            _,slide=cost_terms(s,t,c,1.);self.assertFalse(slide[0,0],key);c[key]=old
        s['root_quat'][0]=torch.tensor([np.cos(.5),np.sin(.5),0,0])
        _,slide=cost_terms(s,t,c,1.);self.assertFalse(slide[0,0])

    def test_terminal_motion_and_wrong_gmr_pose_remain_penalized(self):
        s,t,c=self.fixture();s['root_velocity'][0,0]=1;s['leg_q'][0,0]=.5
        early,_=cost_terms(s,t,c,1.);late,_=cost_terms(s,t,c,2.5)
        self.assertEqual(early['terminal_speed'].item(),0)
        self.assertGreater(late['terminal_speed'].item(),99)
        self.assertGreater(late['leg_pose'].item(),0)

    def test_population_costs_keep_worlds_independent(self):
        s,t,c=self.fixture()
        s={k:v.repeat((64,)+(1,)*(v.ndim-1)) for k,v in s.items()}
        c={k:v.repeat((64,)+(1,)*(v.ndim-1)) for k,v in c.items()}
        s['root_pos'][37,0]=.3
        terms,sliding=cost_terms(s,t,c,1.)
        self.assertEqual(terms['xy'].shape,(64,))
        self.assertEqual(sliding.shape,(64,2))
        self.assertEqual(terms['xy'][37].item(),4.)
        self.assertEqual(torch.count_nonzero(terms['xy']).item(),1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
