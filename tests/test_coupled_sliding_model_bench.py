"""Momentum,contact constitutive graph and discrete energy checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from coupled_sliding_model_bench import CoupledCoulomb,motor_torque


@pytest.fixture
def model():return CoupledCoulomb(22.273,.4035,[.00974272,.00974272],.31114,.127)


def test_uncoupled_convex_problem_has_exact_disk_projection_solution(model):
    b=np.array([2.,3.,-.1,.2]);limits=np.array([.5,.3])
    p,_,residual=model.impulse(np.eye(4),b,limits)
    expected=-b.reshape(2,2);expected[0]*=.5/np.linalg.norm(expected[0])
    np.testing.assert_allclose(p,expected.ravel(),atol=2e-9,rtol=0);assert residual<1e-9


def test_rolling_without_drive_needs_no_friction_impulse(model):
    v=np.array([1.,0,0,1/.127,1/.127]);after,d=model.step(v,0.,[0.,0.],.7,[100.,100.],.0025)
    np.testing.assert_allclose(after,v,atol=1e-12,rtol=0)
    np.testing.assert_allclose(d['force_t_n'],0.,atol=1e-12)


def test_small_relative_velocity_can_stop_inside_static_capacity(model):
    v=np.array([1.,1e-4,0,1/.127,1/.127]);after,d=model.step(v,0.,[0.,0.],.7,[100.,100.],.0025)
    assert abs(after[1])<2e-8 and d['rho'].max()<.1


def test_loaded_sliding_has_saturated_opposing_friction_and_correct_deceleration(model):
    v=np.array([1.,2.,0,1/.127,1/.127]);dt=.0025;normal=model.mass*9.81/2
    for _ in range(100):
        after,d=model.step(v,0.,[0.,0.],.7,[normal,normal],dt)
        assert after[1]>0
        np.testing.assert_allclose(d['rho'],1.,atol=2e-8,rtol=0)
        assert np.all(d['friction_power_w']<0)
        assert after[1]==pytest.approx(v[1]-.7*9.81*dt,abs=2e-9)
        assert abs(d['energy_balance_error_j'])<1e-12
        assert d['energy_loss_contact_step_j']>=0
        v=after


def test_longitudinal_and_lateral_contact_forces_share_one_disk(model):
    v=np.array([1.,1.,0,20.,20.]);_,d=model.step(v,0.,[0.,0.],.7,[100.,100.],.0025)
    assert (d['force_t_n'][:,0]>0).all() and (d['force_t_n'][:,1]<0).all()
    np.testing.assert_allclose(np.linalg.norm(d['force_t_n'],axis=1),70.,atol=2e-6,rtol=0)
    efficiency=-d['friction_power_w']/(70*np.linalg.norm(d['slip_m_s'],axis=-1))
    np.testing.assert_allclose(efficiency,1.,atol=2e-7,rtol=0)


def test_body_and_wheel_momentum_balance_is_coupled(model):
    v=np.array([1.,.2,.3,10.,8.]);tau=np.array([2.,-1.]);dt=.0025;yaw=.7
    after,d=model.step(v,yaw,tau,.7,[100.,100.],dt)
    impulse=d['force_t_n'].ravel()*dt
    np.testing.assert_allclose(model.M*(after-v),dt*np.r_[0.,0.,0.,tau]+model.jacobian(yaw).T@impulse,atol=1e-14,rtol=0)


def test_differential_wheel_torque_generates_yaw_with_correct_sign(model):
    after,_=model.step(np.zeros(5),0.,[-1.,1.],.7,[100.,100.],.0025)
    assert after[2]>0


def test_native_motor_speed_curve_limits_drive_at_high_spin():
    np.testing.assert_allclose(motor_torque([12.,12.],[90.,0.]),[1.2,12.],atol=1e-12)
    assert motor_torque([-12.,12.],[120.,-120.])[0]<0
