import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_wheel_model_factors import external_wrench


def test_pointwise_external_laws_are_dissipative_and_capacity_limited():
    state=np.array([0,0,.127,1,0,0,0,1,.34,0,0,1/.127,0.],dtype=float)
    data=dict(normal_points=np.array([[0,-.02,0],[0,.02,0]]),
              normal_normals=np.array([[0,0,1],[0,0,1]]),normal_scalars=np.array([3.,7.]))
    for law in ('external_coulomb_regularized','combined_slip_magic_formula'):
        force,torque,power=external_wrench(state,data,law,.7)
        assert np.linalg.norm(force)<=7.+1e-10
        assert abs(force[2])<1e-10
        assert power<0
        np.testing.assert_allclose(power,force@state[7:10]+torque@state[10:13],atol=1e-10)


def test_no_load_or_native_law_does_not_add_a_force():
    state=np.array([0,0,.127,1,0,0,0,1,.34,0,0,0,0.])
    empty=dict(normal_points=[],normal_normals=[],normal_scalars=[])
    for law in ('native_coulomb','external_coulomb_regularized','combined_slip_magic_formula'):
        force,torque,power=external_wrench(state,empty,law,.7)
        np.testing.assert_array_equal(force,np.zeros(3))
        np.testing.assert_array_equal(torque,np.zeros(3))
        assert power==0
