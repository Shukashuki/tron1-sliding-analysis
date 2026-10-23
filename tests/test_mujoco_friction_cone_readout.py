"""Real small CPU MuJoCo fixtures; the reader must never advance the model."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mujoco_friction_cone_readout as reader


def fixture(*, cone="pyramidal", condim=3, floor_first=True, moving_floor=False,
            z=.19, right_z=2., right_x=1.):
    floor = ('<body name="floor_body" pos="0 0 -.1">'
             + ('<joint name="floor_slide" type="slide" axis="1 0 0"/>' if moving_floor else '')
             + '<geom name="floor" type="box" size="2 2 .1" mass="1000"/></body>')
    left = f'<body name="left" pos="0 0 {z}"><freejoint name="left_free"/><geom name="left_geom" type="box" size=".2 .2 .2" mass="1"/></body>'
    right = f'<body name="right" pos="{right_x} 0 {right_z}"><freejoint name="right_free"/><geom name="right_geom" type="box" size=".2 .2 .2" mass="1"/></body>'
    bodies = floor+left+right if floor_first else left+right+floor
    xml = (f'<mujoco><compiler fusestatic="false"/><option timestep=".0025" gravity="0 0 -9.81" solver="Newton" '
           f'cone="{cone}" iterations="100" tolerance="1e-12"/>'
           f'<default><geom condim="{condim}" friction=".7 .02 .01"/></default><worldbody>{bodies}</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    floor_id = int(model.geom("floor").id)
    wheels = tuple(int(model.body(name).id) for name in ("left", "right"))
    return model, data, floor_id, wheels


def solve(model, data, *, linear=(1., 0., 0.), angular=(0., 0., 0.), floor_speed=0.):
    joint = model.joint("left_free")
    start = int(joint.dofadr[0])
    data.qvel[start:start+3] = linear
    data.qvel[start+3:start+6] = angular
    try:
        floor = model.joint("floor_slide")
    except KeyError:
        floor = None
    if floor is not None:
        data.qvel[int(floor.dofadr[0])] = floor_speed
    mujoco.mj_forward(model, data)  # Fixture preparation ONLY, never the reader.


def read(model, data, floor, wheels, axes=None):
    return reader.read_wheel_floor_contacts(model, data, floor_geom_id=floor,
        wheel_body_ids=wheels, wheel_lateral_axes_local=axes)


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_current_real_contact_wrench_frame_mu_and_torque_components(cone, condim):
    model, data, floor, wheels = fixture(cone=cone, condim=condim)
    if condim >= 4:
        # Deliberately large torque-friction lengths exercise real nonzero
        # torsional/rolling wrench components, not a production parameter.
        model.geom_friction[:, 1:] = [2., 1.]
    solve(model, data, linear=(0., 0., 0.) if condim >= 4 else (1., 0., 0.), angular=(.2, .3, .4))
    result = read(model, data, floor, wheels)
    count = result["metadata"]["contact_rows"]
    assert count > 0 and result["contact_frame_world"].shape == (count, 3, 3)
    assert np.all(result["condim"] == condim)
    assert np.all(result["fn_n"] >= 0.) and result["fn_n"].sum() > 0
    assert np.all(result["force_on_wheel_world_n"][:, 2] >= 0.)
    assert result["metadata"]["cone_type"] == cone
    assert result["metadata"]["force_units"] == "N"
    assert result["metadata"]["torque_units"] == "Nm"
    assert result["metadata"]["full_contact_wrench_cone_evaluated"] is False
    for row, index in enumerate(result["contact_index"]):
        wrench = np.empty(6)
        mujoco.mj_contactForce(model, data, int(index), wrench)
        contact = data.contact[int(index)]
        np.testing.assert_array_equal(result["wrench_on_geom2_contact_frame"][row], wrench)
        np.testing.assert_array_equal(result["friction_coefficients"][row], contact.friction)
        np.testing.assert_array_equal(result["ft_contact_frame_n"][row], wrench[1:3])
        basis = result["contact_frame_world"][row]
        sign = result["wheel_force_sign"][row]
        np.testing.assert_allclose(result["force_on_wheel_world_n"][row], sign*basis.T@wrench[:3])
        np.testing.assert_allclose(result["torque_on_wheel_at_contact_world_nm"][row], sign*basis.T@wrench[3:])
    if condim <= 3:
        np.testing.assert_array_equal(result["torque_on_geom2_contact_frame_nm"], 0.)
    else:
        assert np.any(np.abs(result["torque_on_geom2_contact_frame_nm"]) > 1.e-9)


def test_xml_body_order_exercises_both_geom_signs_with_force_on_wheel_upward():
    signs = []
    for floor_first in (True, False):
        model, data, floor, wheels = fixture(floor_first=floor_first)
        solve(model, data)
        result = read(model, data, floor, wheels)
        signs.extend(result["wheel_is_geom2"].tolist())
        assert np.all(result["force_on_wheel_world_n"][:, 2] >= 0.)
        assert result["force_on_wheel_world_n"][:, 2].sum() > 0.
        assert result["force_on_wheel_world_n"][:, 0].sum() < 0.
    assert {False, True} <= set(signs)


@pytest.mark.parametrize("floor_first", [True, False])
def test_moving_floor_and_angular_point_velocity_are_subtracted_in_engine_frame(floor_first):
    model, data, floor, wheels = fixture(floor_first=floor_first, moving_floor=True)
    linear = np.array([1.1, .2, 0.]); angular = np.array([0., 0., .7])
    solve(model, data, linear=linear, angular=angular, floor_speed=.4)
    result = read(model, data, floor, wheels)
    point = result["contact_position_world_m"]
    wheel_velocity = linear + np.cross(angular, point-data.xipos[wheels[0]])
    expected = wheel_velocity-np.array([.4, 0., 0.])
    np.testing.assert_allclose(result["wheel_relative_point_velocity_world_m_s"], expected, atol=1.e-12)
    assert np.ptp(expected[:, 0]) > .1  # Rotational point velocity, not COM-only.
    for row in range(len(point)):
        frame = result["contact_frame_world"][row]
        relative = result["geom2_point_velocity_world_m_s"][row]-result["geom1_point_velocity_world_m_s"][row]
        np.testing.assert_allclose(result["relative_point_velocity_geom2_minus_geom1_contact_frame_m_s"][row], frame@relative)
        np.testing.assert_allclose(result["relative_tangent_velocity_m_s"][row], (frame@relative)[1:])
        np.testing.assert_allclose(result["tangent_contact_power_w"][row],
            result["ft_contact_frame_n"][row]@result["relative_tangent_velocity_m_s"][row])


def test_explicit_lateral_axis_only_and_invalid_normal_axis_is_not_guessed():
    model, data, floor, wheels = fixture()
    solve(model, data, linear=(0., .5, 0.))
    omitted = read(model, data, floor, wheels)
    assert not omitted["wheel_lateral_axis_valid"].any()
    assert np.isnan(omitted["wheel_lateral_slip_m_s"]).all()
    axes = np.array([[0., 2., 0.], [0., 1., 0.]])
    original = axes.copy()
    result = read(model, data, floor, wheels, axes)
    np.testing.assert_array_equal(axes, original)
    assert result["wheel_lateral_axis_valid"].all()
    np.testing.assert_allclose(result["wheel_lateral_slip_m_s"], .5, atol=1.e-12)
    invalid = read(model, data, floor, wheels, [[0., 0., 1.], [0., 1., 0.]])
    assert not invalid["wheel_lateral_axis_valid"].any()
    assert np.isnan(invalid["wheel_lateral_axis_world"]).all()


def test_two_wheels_in_contact_exclude_self_contact_and_other_ground_geom():
    model, data, floor, wheels = fixture(right_z=.19, right_x=.25)
    solve(model, data)
    assert any(floor not in (int(c.geom1), int(c.geom2)) for c in data.contact)
    result = read(model, data, floor, wheels)
    assert set(result["wheel_index"]) == {0, 1}
    assert np.all((result["geom1_id"] == floor) | (result["geom2_id"] == floor))
    assert result["metadata"]["contact_rows"] < data.ncon


def test_no_contacts_have_consistent_empty_ragged_shapes():
    model, data, floor, wheels = fixture(z=2.)
    solve(model, data)
    result = read(model, data, floor, wheels)
    assert result["metadata"]["contact_rows"] == 0
    for name, shape in (("fn_n", (0,)), ("ft_contact_frame_n", (0, 2)),
                        ("friction_coefficients", (0, 5)), ("contact_frame_world", (0, 3, 3)),
                        ("wrench_on_geom2_contact_frame", (0, 6))):
        assert result[name].shape == shape


def test_world_floor_point_velocity_is_naturally_zero():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <body name="left" pos="0 0 .19"><freejoint name="left_free"/><geom type="box" size=".2 .2 .2"/></body>
      <body name="right" pos="1 0 2"><freejoint/><geom type="box" size=".2 .2 .2"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model); solve(model, data)
    floor = int(model.geom("floor").id)
    wheels = tuple(int(model.body(n).id) for n in ("left", "right"))
    result = read(model, data, floor, wheels)
    assert len(result["fn_n"]) > 0 and np.all(result["floor_body_id"] == 0)
    for i in range(len(result["fn_n"])):
        key = "geom1_point_velocity_world_m_s" if result["geom1_id"][i] == floor else "geom2_point_velocity_world_m_s"
        np.testing.assert_array_equal(result[key][i], 0.)


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_reader_arrays_feed_frozen_pure_condim3_math_without_frame_reinterpretation(cone):
    from friction_cone_diagnostics import contact_diagnostics
    model, data, floor, wheels = fixture(cone=cone); solve(model, data)
    result = read(model, data, floor, wheels)
    boundary = "elliptic_condim3" if cone == "elliptic" else "mujoco_pyramidal_condim3"
    values = contact_diagnostics(result["fn_n"], result["ft_contact_frame_n"],
        result["friction_coefficients"][:, :2], force_units="N",
        slip_m_s=result["relative_tangent_velocity_m_s"], condim=3,
        engine_boundary=boundary, engine_tangents_aligned=True)
    assert values["engine_rho"].shape == result["fn_n"].shape
    assert np.isfinite(values["engine_rho"][values["cone_valid_loaded"]]).all()


def test_inactive_detected_gap_contacts_are_not_labeled_solved_support():
    model, data, floor, wheels = fixture(z=.21)
    model.geom_margin[:] = .02
    model.geom_gap[:] = .02
    solve(model, data)
    assert data.ncon > 0 and all(c.efc_address == -1 for c in data.contact)
    result = read(model, data, floor, wheels)
    assert result["metadata"]["contact_rows"] == 0
    assert len(result["metadata"]["skipped_inactive_floor_wheel_contact_indices"]) == data.ncon


def test_reader_never_steps_forwards_resets_or_changes_any_observed_state():
    model, data, floor, wheels = fixture(condim=6)
    solve(model, data, angular=(.2, .3, .4))
    names = ("qpos", "qvel", "qacc", "qacc_warmstart", "act", "ctrl", "qfrc_applied",
             "xfrc_applied", "xpos", "xmat", "xipos", "efc_force")
    before = {name: getattr(data, name).copy() for name in names}
    time, ncon = data.time, data.ncon
    contact = [(c.frame.copy(), c.pos.copy(), c.friction.copy()) for c in data.contact]
    forbidden = ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_fwdPosition", "mj_resetData")
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in forbidden:
            stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError("Reader must not invoke " + name)))
        result = read(model, data, floor, wheels)
    for name in names: np.testing.assert_array_equal(getattr(data, name), before[name])
    assert data.time == time and data.ncon == ncon
    for c, old in zip(data.contact, contact):
        for value, expected in zip((c.frame, c.pos, c.friction), old): np.testing.assert_array_equal(value, expected)
    result["fn_n"][:] = -123.
    result["contact_frame_world"][:] = -123.
    again = read(model, data, floor, wheels)
    assert np.all(again["fn_n"] >= 0.)
    assert not np.all(again["contact_frame_world"] == -123.)


@pytest.mark.parametrize("kwargs", [
    {"floor_geom_id": True}, {"floor_geom_id": -1}, {"wheel_body_ids": (1, 1)},
    {"wheel_body_ids": (0, 1)}, {"wheel_lateral_axes_local": [[0., 0., 0.], [0., 1., 0.]]},
    {"wheel_lateral_axes_local": [[0., np.nan, 0.], [0., 1., 0.]]},
])
def test_invalid_explicit_ids_or_axes_fail_closed(kwargs):
    model, data, floor, wheels = fixture(); solve(model, data)
    args = {"floor_geom_id": floor, "wheel_body_ids": wheels, **kwargs}
    with pytest.raises(ValueError): reader.read_wheel_floor_contacts(model, data, **args)
