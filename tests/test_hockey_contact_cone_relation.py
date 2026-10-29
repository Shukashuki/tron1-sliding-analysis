from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_hockey_contact_cone_relation import interval_contact, loss_windows
from wheel_friction_capacity import capacity_ratios


def test_loss_window_excludes_post_separation_and_catches_terminal_loss():
    touch=np.array([[1,1],[1,1],[1,0],[1,1],[0,1]],dtype=bool)
    events=loss_windows(touch,.0025,.005)
    assert events==[
        dict(side=0,state_index=4,preceding_interval=3,window_start=2,window_stop=4),
        dict(side=1,state_index=2,preceding_interval=1,window_start=0,window_stop=2)]
    np.testing.assert_array_equal(interval_contact(touch),[[1,1],[1,0],[1,0],[0,1]])


def test_capacity_drop_can_raise_ratio_while_friction_force_falls():
    fn=np.array([[100.,10.],[0.,10.]])
    ft=np.zeros((2,2,3));ft[0,:,0]=[35.,7.];ft[1,:,0]=[0.,7.]
    touch=np.array([[1,1],[1,0]],dtype=bool)
    ratios=capacity_ratios(fn,ft,touch,.9,.7)
    np.testing.assert_allclose(ratios['rho_dynamic'][0],[.5,1.])
    np.testing.assert_allclose(ratios['rho_static'][0],[35/90,7/9])
    assert np.isnan(ratios['rho_dynamic'][1]).all()
    # Contact at zero Fn remains contact: it is only the ratio that is undefined.
    assert touch[1,0]


def test_wheel_resultant_can_cancel_even_with_saturated_individual_anchors():
    ft=np.array([[7.,0.,0.],[-7.,0.,0.]])
    r=capacity_ratios(np.array([20.]),ft.sum(0)[None],np.array([True]),.7,.7)
    assert r['rho_dynamic'][0]==0
    assert np.linalg.norm(ft,axis=1).sum()/(.7*20)==1
