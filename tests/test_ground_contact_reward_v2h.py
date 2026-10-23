"""H arithmetic only; neither runtime samples nor a learned H controller exist."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ground_contact_reward_v2h as h
try: import torch
except ImportError: torch=None

def force():
    return np.full((8,2),50.,dtype=np.float64)

def evaluate(velocity, forces=None, quality=1., reference_speed=1.):
    return h.reward_numpy(force() if forces is None else forces,100.,1.,1.,quality,reference_speed,velocity)

class NumpyHTests(unittest.TestCase):
    def test_no_change_inside_deadband(self):
        for v in (0., .1, -.1, .3, -.3):
            with self.subTest(v=v):
                f=force();old=h.base.reward_numpy(f,100.,1.,1.,.7,1.)
                new=evaluate(np.full(8,v),f,.7)
                self.assertEqual(new["raw_reward"],old["raw_reward"])
                for key in old:np.testing.assert_array_equal(new[key],old[key])

    def test_exact_hinge_scale_and_cap(self):
        v=np.array([0.,.3,-.3,.45,-.45,.6,-.6,3.])
        value=evaluate(v)
        np.testing.assert_allclose(value["loaded_com_vz_sample_cost"],[0.,0.,0.,.25,.25,1.,1.,1.],atol=1e-15)
        self.assertAlmostEqual(value["loaded_com_vz_cost"],3.5/8)
        self.assertAlmostEqual(value["loaded_com_vz_penalty"],.5*3.5/8)
        self.assertAlmostEqual(value["raw_reward"],1.-.5*3.5/8)

    def test_no_extra_cost_when_unloaded_but_gap_cost_remains(self):
        value=evaluate(np.full(8,10.),np.zeros((8,2)),0.)
        self.assertEqual(value["loaded_com_vz_penalty"],0.)
        self.assertEqual(value["support_gap_penalty"],1.)
        self.assertEqual(value["raw_reward"],-1.)

    def test_any_wheel_support_and_strict_threshold(self):
        for loads,penalty in (([6.,0.],.5),([0.,6.],.5),([5.,5.],0.),([0.,0.],0.)):
            with self.subTest(loads=loads):
                f=np.broadcast_to(loads,(8,2))
                self.assertAlmostEqual(evaluate(np.full(8,1.),f)["loaded_com_vz_penalty"],penalty)

    def test_single_balanced_and_alternating_support_equivalent(self):
        f=force();single=f.copy();single[:,0]=100.;single[:,1]=0.
        alternating=single.copy();alternating[::2]=[0.,100.]
        values=[evaluate(np.full(8,.45),x)["raw_reward"] for x in (f,single,alternating)]
        np.testing.assert_array_equal(values,np.full(3,values[0]))

    def test_lifted_wheel_is_not_a_bilateral_constraint(self):
        f=np.tile([100.,0.],(8,1))
        self.assertEqual(evaluate(np.zeros(8),f)["raw_reward"],1.)

    def test_beta_positive_bonus_and_ungated_gap_are_unchanged(self):
        f=force();f[2]=0.
        g=h.base.reward_numpy(f,100.,.8,.9,.4,1.)
        new=h.reward_numpy(f,100.,.8,.9,.4,1.,np.full(8,.45))
        for key in g.keys()-{"raw_reward"}:np.testing.assert_array_equal(new[key],g[key])
        self.assertAlmostEqual(new["raw_reward"],g["raw_reward"]-new["loaded_com_vz_penalty"])

    def test_stop_neutral_beta_gate_stays_neutral(self):
        value=evaluate(np.zeros(8),quality=0.,reference_speed=0.)
        self.assertEqual(value["beta_contact_gate"],1.)
        self.assertEqual(value["raw_reward"],1.)

    def test_batched_horizons_use_each_eight_sample_window(self):
        f=np.full((2,3,8,2),50.)
        v=np.zeros((2,3,8));v[1,2]=.6
        result=h.reward_numpy(f,100.,1.,1.,1.,1.,v)
        expected=np.ones((2,3));expected[1,2]=.5
        np.testing.assert_allclose(result["raw_reward"],expected)

    def test_float32_scalar_window_keeps_native_arithmetic_dtype(self):
        result=evaluate(np.full(8,.45,dtype=np.float32),force().astype(np.float32))
        self.assertEqual(np.asarray(result["raw_reward"]).dtype,np.dtype(np.float32))
        self.assertEqual(np.asarray(result["loaded_com_vz_penalty"]).dtype,np.dtype(np.float32))

    def test_end_state_or_scalar_cannot_replace_eight_samples(self):
        for v in (0.,[0.],np.zeros(7),np.zeros((8,1))):
            with self.subTest(shape=np.shape(v)),self.assertRaises(ValueError):evaluate(v)

    def test_invalid_velocity_not_hidden_by_no_contact(self):
        for value in (np.nan,np.inf,-np.inf):
            v=np.zeros(8);v[2]=value
            with self.subTest(value=value),self.assertRaises(ValueError):evaluate(v,np.zeros((8,2)))

    def test_boolean_complex_or_nonnumeric_inputs_rejected(self):
        for v in (np.zeros(8,dtype=bool),np.zeros(8,dtype=complex),["bad"]*8):
            with self.assertRaises(ValueError):evaluate(v)
        for f in (np.ones((8,2),dtype=bool),np.ones((8,2),dtype=complex)):
            with self.assertRaises(ValueError):evaluate(np.zeros(8),f)

    def test_invalid_force_cannot_become_motion_credit(self):
        for value in (-1.,np.nan,np.inf):
            f=force();f[1,0]=value
            with self.assertRaises(ValueError):h.normal_motion_penalty_numpy(f,np.zeros(8))

    def test_settings_and_ancestry_are_frozen(self):
        h.verify_base()
        for key,value in (("candidate","wrong"),("loaded_com_vz_weight",0.),("loaded_com_vz_deadband_m_s",True)):
            settings=h.validate_settings();settings[key]=value
            with self.assertRaises(ValueError):h.validate_settings(settings)
        settings=h.validate_settings();settings.pop("loaded_com_vz_scale_m_s")
        with self.assertRaises(ValueError):h.validate_settings(settings)
        h.DEFAULT_SETTINGS["loaded_com_vz_weight"]=0.
        try:self.assertEqual(h.validate_settings()["loaded_com_vz_weight"],.5)
        finally:h.DEFAULT_SETTINGS["loaded_com_vz_weight"]=.5

    def test_inputs_not_mutated_and_metadata_has_no_success_claim(self):
        f=force();v=np.linspace(-1.,1.,8);before=(f.copy(),v.copy())
        evaluate(v,f)
        np.testing.assert_array_equal(f,before[0]);np.testing.assert_array_equal(v,before[1])
        meta=h.formula_metadata()
        self.assertFalse(meta["absolute_height_target_added"])
        self.assertFalse(meta["bilateral_support_required"])
        self.assertFalse(meta["contact_point_slip_or_friction_cone_measurement_claimed"])
        self.assertFalse(meta["runtime_velocity_samples_verified"])
        self.assertTrue(meta["arithmetic_alone_does_not_certify_runtime_or_task_success"])

@unittest.skipUnless(torch is not None,"Real Torch required; never count skipped Torch as runtime proof")
class TorchHTests(unittest.TestCase):
    def test_numpy_torch_parity_real_float32_and_float64(self):
        rng=np.random.default_rng(998)
        for dtype,tdtype in ((np.float32,torch.float32),(np.float64,torch.float64)):
            f=rng.uniform(0,100,(3,8,2)).astype(dtype);v=rng.uniform(-1,1,(3,8)).astype(dtype)
            f[0,0]=0.;quality=np.array([.2,.5,.9],dtype=dtype)
            expected=h.reward_numpy(f,100.,1.,1.,quality,1.,v)
            actual=h.reward_torch(torch.tensor(f,dtype=tdtype),100.,1.,1.,torch.tensor(quality,dtype=tdtype),1.,torch.tensor(v,dtype=tdtype))
            for key in expected:
                np.testing.assert_allclose(actual[key].detach().numpy(),expected[key],rtol=2e-6 if dtype==np.float32 else 1e-13,atol=2e-7 if dtype==np.float32 else 2e-15)

    def test_invalid_unloaded_velocity_poisons_entire_window_without_host_sync(self):
        f=torch.zeros((2,8,2));v=torch.zeros((2,8));v[0,3]=float("nan")
        actual=h.reward_torch(f,100.,1.,1.,1.,1.,v,validate=False)
        for value in actual.values():
            self.assertTrue(torch.isnan(value[0]).all());self.assertTrue(torch.isfinite(value[1]).all())
        with self.assertRaises(ValueError):h.reward_torch(f,100.,1.,1.,1.,1.,v)

    def test_autograd_of_cost_is_not_differentiable_physics_claim(self):
        f=torch.full((8,2),50.,dtype=torch.float64)
        v=torch.full((8,),.45,dtype=torch.float64,requires_grad=True)
        quality=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
        result=h.reward_torch(f,100.,1.,1.,quality,1.,v)
        result["raw_reward"].backward()
        self.assertTrue(torch.isfinite(v.grad).all());self.assertTrue((v.grad<0).all())
        torch.testing.assert_close(quality.grad,torch.tensor(1.,dtype=torch.float64))

    def test_hot_arithmetic_has_no_host_tensor_reads_or_filesystem(self):
        f=torch.full((8,2),50.);v=torch.full((8,),.45)
        def no(*args,**kwargs):raise AssertionError("No hot host read or filesystem")
        with patch.object(torch.Tensor,"cpu",no),patch.object(torch.Tensor,"numpy",no),patch.object(torch.Tensor,"item",no),patch.object(torch.Tensor,"__bool__",no),patch.object(Path,"read_bytes",no):
            result=h.reward_torch(f,100.,1.,1.,1.,1.,v,validate=False)
        torch.testing.assert_close(result["raw_reward"],torch.tensor(.875))

    def test_boolean_wrong_shape_and_complex_velocity_rejected(self):
        f=torch.full((8,2),50.)
        for v in (torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.complex64),torch.zeros(1)):
            with self.assertRaises(ValueError):h.reward_torch(f,100.,1.,1.,1.,1.,v)

if __name__=="__main__":unittest.main()

