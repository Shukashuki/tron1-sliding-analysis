"""Physical sign/energy/frame tests; these do not claim a PhysX rollout."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nonlinear_tire_contact import TireParameters, tangent_wrench, wheel_frame

LAWS = ("external_coulomb_regularized", "combined_slip_magic_formula")


def wheel(v=(1., 0., 0.), omega=(0., 10., 0.), load=100., law=LAWS[1]):
    return tangent_wrench(v, omega, [0., 0., -.1], [0., 1., 0.], [0., 0., 1.], load, law=law)


@pytest.mark.parametrize("law", LAWS)
def test_pure_rolling_and_no_contact_have_no_tangent_wrench(law):
    for value in (wheel(law=law), wheel(v=(1., 2., 0.), load=0., law=law)):
        np.testing.assert_array_equal(value["force_w_n"], np.zeros(3))
        np.testing.assert_array_equal(value["torque_about_COM_w_nm"], np.zeros(3))


@pytest.mark.parametrize("law", LAWS)
def test_driven_wheel_pushes_body_forward_and_resists_spin(law):
    result = wheel(omega=(0., 20., 0.), law=law)
    assert result["force_w_n"][0] > 0
    assert result["torque_about_COM_w_nm"][1] < 0
    assert result["contact_power_w"] < 0


@pytest.mark.parametrize("law", LAWS)
def test_random_combined_slip_is_dissipative_bounded_and_has_no_normal_force(law):
    rng = np.random.default_rng(42)
    v, w, r = [rng.normal(size=(2000, 3)) for _ in range(3)]
    loads = rng.uniform(0, 1000, 2000)
    result = tangent_wrench(v, w, r, [0, 1, 0], [0, 0, 1], loads, law=law)
    assert np.all(result["contact_power_w"] <= 1e-10)
    assert np.all(np.linalg.norm(result["force_w_n"], axis=-1) <= .7 * loads + 1e-10)
    np.testing.assert_array_equal(result["normal_component_of_applied_tire_force_n"], np.zeros(2000))
    np.testing.assert_allclose(result["torque_about_COM_w_nm"], np.cross(r, result["force_w_n"]))


@pytest.mark.parametrize("law", LAWS)
def test_rigid_world_rotation_preserves_force_moment_and_power(law):
    theta = .73
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    args = [np.array(x) for x in ([1., .7, 0], [0, 14., 0], [0, 0, -.1], [0, 1., 0], [0, 0, 1.])]
    first = tangent_wrench(*args, 100, law=law)
    second = tangent_wrench(*(R @ x for x in args), 100, law=law)
    for key in ("force_w_n", "torque_about_COM_w_nm"):
        np.testing.assert_allclose(second[key], R @ first[key], atol=1e-12)
    assert second["contact_power_w"] == pytest.approx(first["contact_power_w"])


@pytest.mark.parametrize("law", LAWS)
def test_lateral_sliding_force_opposes_slip(law):
    positive, negative = wheel(v=(1., .5, 0.), law=law), wheel(v=(1., -.5, 0.), law=law)
    assert positive["force_w_n"][1] < 0 < negative["force_w_n"][1]
    np.testing.assert_allclose(positive["force_w_n"], -negative["force_w_n"])


@pytest.mark.parametrize("value", [0., -.7, float("nan"), float("inf")])
def test_invalid_coefficient_rejected(value):
    with pytest.raises(ValueError): TireParameters(mu=value)


def test_static_heading_is_undefined_and_degenerate_axis_rejected():
    assert np.isnan(wheel(v=(0, 0, 0), omega=(0, 0, 0))["hub_side_angle_rad"])
    with pytest.raises(ValueError): wheel_frame([0, 0, 1], [0, 0, 1])
    with pytest.raises(ValueError): wheel(load=-1)
