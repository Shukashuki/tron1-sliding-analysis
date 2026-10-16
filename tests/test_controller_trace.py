"""Asset-free schema checks and no-step FK round trips on native clocks."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import JOINT_NAMES, quaternion_multiply


SCHEMAS = {
    "policy_isaac": dict(position="root_position_m", quaternion="root_quaternion_wxyz",
                          com_velocity="root_linear_velocity_w_m_s", angular_velocity="root_angular_velocity_w_rad_s",
                          joint_pos="joint_position_rad", joint_vel="joint_velocity_rad_s", wheel_pos="wheel_position_m"),
    "policy_mujoco": dict(position="root_pos", quaternion="root_quat_wxyz", com_velocity="root_com_lin_vel_w",
                           angular_velocity="root_ang_vel_w", joint_pos="joint_pos", joint_vel="joint_vel", wheel_pos="wheel_pos"),
    "nmpc": dict(position="root_pos", quaternion="root_quaternion_wxyz", com_velocity="base_com_velocity_w",
                 angular_velocity="base_angular_velocity_w", joint_pos="joint_pos", joint_vel="joint_vel", wheel_pos="wheel_pos"),
}
CONTACTS = dict(contact_count="wheel_ground_contact_count", contact_force="wheel_ground_contact_force_w_n",
                contact_slip="wheel_lateral_contact_slip_m_s")


def canonical_data(n=5):
    time = np.arange(n)*.005
    return dict(time_s=time, physics_dt_s=.005, joint_names=JOINT_NAMES,
                position=np.column_stack((time, -time, np.ones(n))),
                quaternion=np.tile([1., 0., 0., 0.], (n, 1)),
                com_velocity=np.tile([1., -1., .2], (n, 1)),
                angular_velocity=np.tile([.1, .2, .3], (n, 1)),
                joint_pos=np.arange(n*8, dtype=float).reshape(n, 8)/100.,
                joint_vel=-np.arange(n*8, dtype=float).reshape(n, 8)/10.,
                wheel_pos=np.arange(n*6, dtype=float).reshape(n, 2, 3)/100.,
                contact_count=np.tile([1., 0.], (n, 1)),
                contact_force=np.tile([[0., 0., 20.], [0., 0., 0.]], (n, 1, 1)),
                contact_slip=np.tile([.4, 0.], (n, 1)))


def encoded_trace(kind, canonical=None, *, order=(6, 2, 4, 0, 7, 3, 1, 5)):
    canonical = canonical_data() if canonical is None else canonical
    data = {"time_s": canonical["time_s"].copy(), "physics_dt_s": np.array(canonical["physics_dt_s"])}
    names_key = "action_joint_names" if kind == "nmpc" else "joint_names"
    data[names_key] = np.array(JOINT_NAMES)[list(order)]
    for canonical_name, source in {**SCHEMAS[kind], **CONTACTS}.items():
        value = canonical[canonical_name].copy()
        if canonical_name in ("joint_pos", "joint_vel"):
            value = value[:, order]
        data[source] = value[:, None] if kind == "policy_isaac" else value
    if kind == "policy_isaac":
        data["valid_mask"] = np.ones((len(data["time_s"]), 1), dtype=bool)
    return data


@pytest.mark.parametrize("kind", list(SCHEMAS))
def test_all_explicit_native_schemas_reorder_named_joints_and_copy_data(kind):
    expected = canonical_data(); source = encoded_trace(kind, expected)
    before = copy.deepcopy(source)
    actual = native_trace(source, kind)
    assert actual["joint_names"] == JOINT_NAMES
    assert actual["physics_dt_s"] == .005
    for name in expected:
        if name not in ("joint_names", "physics_dt_s"):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    for name in source:
        np.testing.assert_array_equal(source[name], before[name])
    actual["joint_pos"][0, 0] = 999.
    np.testing.assert_array_equal(source[SCHEMAS[kind]["joint_pos"]], before[SCHEMAS[kind]["joint_pos"]])


def test_isaac_first_episode_prefix_drops_nonfinite_post_reset_samples_without_splicing():
    expected = canonical_data(); source = encoded_trace("policy_isaac", expected)
    source["valid_mask"][3:] = False
    for key in (*SCHEMAS["policy_isaac"].values(), *CONTACTS.values()):
        source[key][3:] = np.nan
    actual = native_trace(source, "policy_isaac")
    assert len(actual["time_s"]) == 3
    for key in ("position", "quaternion", "joint_pos", "joint_vel", "wheel_pos", "contact_count", "contact_slip"):
        np.testing.assert_array_equal(actual[key], expected[key][:3])


@pytest.mark.parametrize("mask", [
    [[True], [False], [True], [False], [False]],
    [[False], [True], [True], [True], [True]],
    [[True], [False], [False], [False], [False]],
    np.ones((5, 1), dtype=int), np.ones((5, 2), dtype=bool),
])
def test_isaac_rejects_reentry_empty_prefix_nonbool_or_multiple_environments(mask):
    data = encoded_trace("policy_isaac"); data["valid_mask"] = np.asarray(mask)
    with pytest.raises(ValueError, match="environment|prefix|episode"):
        native_trace(data, "policy_isaac")


@pytest.mark.parametrize("kind", list(SCHEMAS))
@pytest.mark.parametrize("missing", ["joint_vel", "contact_count", "contact_force", "contact_slip"])
def test_missing_physical_measurements_are_not_filled_with_kinematic_or_zero_values(kind, missing):
    data = encoded_trace(kind); key = {**SCHEMAS[kind], **CONTACTS}[missing]; del data[key]
    with pytest.raises(KeyError, match=key):
        native_trace(data, kind)


@pytest.mark.parametrize("kind", list(SCHEMAS))
@pytest.mark.parametrize("bad", ["duplicate_name", "unknown_name", "wrong_shape", "nonfinite", "nonunit_quaternion"])
def test_native_schema_rejects_wrong_coordinates_and_measurements(kind, bad):
    data = encoded_trace(kind)
    names_key = "action_joint_names" if kind == "nmpc" else "joint_names"
    if bad == "duplicate_name":
        data[names_key][0] = data[names_key][1]
    elif bad == "unknown_name":
        data[names_key][0] = "not_a_joint"
    elif bad == "wrong_shape":
        data[SCHEMAS[kind]["wheel_pos"]] = np.zeros((5, 3))
    elif bad == "nonfinite":
        data[SCHEMAS[kind]["com_velocity"]][0] = np.nan
    else:
        data[SCHEMAS[kind]["quaternion"]][0] *= 2.
    with pytest.raises(ValueError):
        native_trace(data, kind)


@pytest.mark.parametrize("bad", ["absent", "vector", "zero", "nan", "time_reset", "dropped_sample", "wrong_dt"])
def test_native_clock_must_have_explicit_scalar_interval_and_consecutive_samples(bad):
    data = encoded_trace("nmpc")
    if bad == "absent":
        del data["physics_dt_s"]
    elif bad == "vector":
        data["physics_dt_s"] = np.array([.005])
    elif bad == "zero":
        data["physics_dt_s"] = np.array(0.)
    elif bad == "nan":
        data["physics_dt_s"] = np.array(np.nan)
    elif bad == "time_reset":
        data["time_s"][2] = 0.
    elif bad == "dropped_sample":
        data["time_s"][2:] += .005
    else:
        data["physics_dt_s"] = np.array(.02)
    with pytest.raises(ValueError):
        native_trace(data, "nmpc")


def test_schema_selection_is_explicit_and_offset_physical_clock_is_preserved():
    data = encoded_trace("nmpc"); data["time_s"] += .005
    np.testing.assert_array_equal(native_trace(data, "nmpc")["time_s"], data["time_s"])
    with pytest.raises(ValueError, match="explicitly"):
        native_trace(data, "auto")


def synthetic_tron1_kinematic_model():
    mj = pytest.importorskip("mujoco")
    legs = []
    for side, lateral in (("L", .14), ("R", -.14)):
        chain = []
        for name, offset, axis in (("abad", f"0 {lateral} 0", "1 0 0"),
                                   ("hip", "0 0 -.12", "0 1 0"),
                                   ("knee", "0 0 -.15", "0 1 0"),
                                   ("wheel", "0 0 -.16", "0 1 0")):
            chain.append(f'<body name="{name}_{side}_Link" pos="{offset}">'
                         f'<joint name="{name}_{side}_Joint" axis="{axis}"/>'
                         '<inertial pos=".02 .01 -.03" mass=".3" diaginertia=".003 .004 .005"/>')
        legs.append(''.join(chain) + '</body>'*4)
    return mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="base_Link" pos="0 0 .8">'
        '<freejoint name="root"/>'
        '<inertial pos=".04576 .00014 -.16398" quat=".995004 0 .0998334 0" '
        'mass="8" diaginertia=".3 .4 .5"/>' + ''.join(legs) + '</body></worldbody></mujoco>')


def measured_kinematic_trace(model, samples=4):
    """Independent oracle: mj_objectVelocity at COM, shifted to wheel origin."""
    import mujoco as mj
    data = mj.MjData(model)
    qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    vids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
    base = int(model.body("base_Link").id)
    wheels = [int(model.body(name).id) for name in ("wheel_L_Link", "wheel_R_Link")]
    trace = canonical_data(samples)
    fields = ("position", "quaternion", "com_velocity", "angular_velocity", "joint_pos", "joint_vel", "wheel_pos")
    rows = {name: [] for name in fields}; wheel_vel = []; wheel_quat = []
    rng = np.random.default_rng(204)
    for i in range(samples):
        mj.mj_resetData(model, data)
        data.qpos[:3] = [.2*i, -.1*i, .9+.03*i]
        angles = [.15+.07*i, -.2+.04*i, .4-.06*i]
        quats = [np.r_[np.cos(a/2), np.sin(a/2)*np.eye(3)[j]] for j, a in enumerate(angles)]
        data.qpos[3:7] = quaternion_multiply(quats[2], quaternion_multiply(quats[1], quats[0]))
        data.qpos[qids] = rng.uniform(-.25, .25, 8)
        data.qpos[qids[-2:]] = [13.*i, -9.*i]
        data.qvel[:6] = [1.+.1*i, -.2, .1, .4, -.3, .6]
        data.qvel[vids] = rng.uniform(-2., 2., 8)
        data.qvel[vids[-2:]] = [18., -14.]
        mj.mj_fwdPosition(model, data); mj.mj_fwdVelocity(model, data)
        velocity = np.empty(6)
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, base, velocity, 0)
        values = dict(position=data.xpos[base], quaternion=data.xquat[base], com_velocity=velocity[3:],
                      angular_velocity=velocity[:3], joint_pos=data.qpos[qids], joint_vel=data.qvel[vids], wheel_pos=data.xpos[wheels])
        for name, value in values.items(): rows[name].append(value.copy())
        origin_vel = []
        for body in wheels:
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
            origin_vel.append(velocity[3:].copy() + np.cross(velocity[:3], data.xpos[body]-data.xipos[body]))
        wheel_vel.append(origin_vel); wheel_quat.append(data.xquat[wheels].copy())
    trace.update({name: np.asarray(values) for name, values in rows.items()})
    return trace, np.asarray(wheel_vel), np.asarray(wheel_quat), data


def assert_no_step_fk_round_trip(model, monkeypatch):
    import mujoco as mj
    trace, expected_velocity, expected_quat, measured_data = measured_kinematic_trace(model)
    before = copy.deepcopy(trace); qpos_before = measured_data.qpos.copy(); qvel_before = measured_data.qvel.copy()
    qpos0_before = model.qpos0.copy(); masses_before = model.body_mass.copy()
    def prohibited(*args, **kwargs):
        pytest.fail("Offline FK must not integrate or re-simulate contacts")
    monkeypatch.setattr(mj, "mj_step", prohibited)
    monkeypatch.setattr(mj, "mj_step1", prohibited)
    monkeypatch.setattr(mj, "mj_step2", prohibited)
    monkeypatch.setattr(mj, "mj_forward", prohibited)
    reconstructed = WheelKinematics(model).reconstruct(trace)
    np.testing.assert_allclose(reconstructed["wheel_center_velocity_w"], expected_velocity, atol=2e-12, rtol=1e-12)
    # Quaternion representations can flip sign without changing physical rotation.
    dot = np.sum(reconstructed["wheel_quaternion_wxyz"]*expected_quat, axis=-1)
    np.testing.assert_allclose(np.abs(dot), 1., atol=1e-12)
    audit = reconstructed["audit"]
    assert audit["dynamics_stepping"] is False and audit["measured_contact_replaced"] is False
    assert audit["maximum_wheel_link_position_error_m"] < 1e-12
    assert audit["maximum_root_com_velocity_reconstruction_error"] < 1e-12
    for name in trace: np.testing.assert_array_equal(trace[name], before[name])
    np.testing.assert_array_equal(measured_data.qpos, qpos_before)
    np.testing.assert_array_equal(measured_data.qvel, qvel_before)
    np.testing.assert_array_equal(model.qpos0, qpos0_before)
    np.testing.assert_array_equal(model.body_mass, masses_before)
    return trace, reconstructed


def test_asset_free_nontrivial_com_world_angular_velocity_and_wheel_origin_fk_parity(monkeypatch):
    assert_no_step_fk_round_trip(synthetic_tron1_kinematic_model(), monkeypatch)


def test_original_licensed_tron1_fk_round_trip_without_stepping(monkeypatch):
    model_path = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
    if not model_path.is_file():
        pytest.skip("Optional local licensed TRON1 model unavailable")
    pytest.importorskip("mujoco")
    from classical_tracking import load_pinned_evaluator
    model, _ = load_pinned_evaluator().load_model(model_path, no_visual_mesh=True)
    assert_no_step_fk_round_trip(model, monkeypatch)


def test_fk_wheel_spin_and_root_quaternion_sign_preserve_heading_geometry():
    model = synthetic_tron1_kinematic_model(); trace, _, _, _ = measured_kinematic_trace(model)
    fk = WheelKinematics(model); expected = fk.reconstruct(trace)
    changed = copy.deepcopy(trace)
    changed["quaternion"] *= -1.
    changed["joint_pos"][:, -2:] += [[123., -321.]]
    actual = fk.reconstruct(changed)
    for name in expected["signals"]:
        np.testing.assert_allclose(actual["signals"][name], expected["signals"][name], atol=1e-12)


def test_fk_rejects_merged_com_changed_wheel_geometry_wrong_order_and_bad_tolerance():
    model = synthetic_tron1_kinematic_model(); trace, _, _, _ = measured_kinematic_trace(model)
    changed_model = copy.copy(model)
    changed_model.body_ipos[int(model.body("base_Link").id)] = 0.
    with pytest.raises(ValueError, match="COM"):
        WheelKinematics(changed_model)
    fk = WheelKinematics(model)
    wrong = copy.deepcopy(trace); wrong["wheel_pos"][0, 0, 0] += .001
    with pytest.raises(ValueError, match="FK mismatch"):
        fk.reconstruct(wrong)
    assert fk.reconstruct(wrong, wheel_position_tolerance_m=.002)["audit"]["maximum_wheel_link_position_error_m"] == pytest.approx(.001)
    wrong = copy.deepcopy(trace); wrong["joint_names"] = JOINT_NAMES[::-1]
    with pytest.raises(ValueError, match="named physical action order"):
        fk.reconstruct(wrong)
    for tolerance in (0., -1., np.nan):
        with pytest.raises(ValueError, match="positive"):
            fk.reconstruct(trace, wheel_position_tolerance_m=tolerance)
