import importlib.util
import json
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("hdm05_gmr", ROOT / "scripts/hdm05_gmr.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
SETTINGS = json.loads((ROOT / "config/hdm05_gmr_experiment.json").read_text())


def test_rigid_yaw_transports_entire_landmark_arrangement():
    rotations = Rotation.from_euler("z", np.array([[0.], [90.]]), degrees=True).as_matrix()
    root = np.zeros((2, 3))
    initial = np.array([[1., 1., 0.], [1., -1., 0.]])
    ankles = np.einsum("tij,kj->tki", rotations, initial)
    robot_initial = np.array([[.2, .3, .1], [.2, -.3, .1]])
    robot_root, targets = mod.transport_targets(ankles, root, robot_initial, np.zeros(3), .5, rotations)
    np.testing.assert_allclose(robot_root, 0)
    np.testing.assert_allclose(targets[0], robot_initial)
    np.testing.assert_allclose(targets[1], robot_initial @ rotations[1].T)


def features():
    times = np.arange(6) * .1
    source_root = np.column_stack((np.zeros(6), times * 2, np.zeros(6)))
    robot_root = np.column_stack((times * 2, np.zeros(6), np.zeros(6)))
    ankle = np.zeros((6, 2, 3))
    ankle[:, 1, 2] = .3  # raised right foot cannot define a separate elevated floor
    toe = ankle + np.array([.1, 0, 0])
    return mod.heading_features(times, source_root, robot_root, ankle, toe,
                                np.zeros((6, 2)), np.zeros((6, 2)), SETTINGS)


def test_human_and_robot_courses_are_not_conflated():
    result = features()
    np.testing.assert_allclose(result["human_course_foot_offset_deg"], 90)
    np.testing.assert_allclose(result["robot_course_wheel_offset_deg"], 0)
    assert result["human_valid_common"][:, 0].all()
    assert not result["human_valid_common"][:, 1].any()
    assert not result["human_foot_velocity_valid"].any()  # stationary foot: undefined travel heading


def test_axial_backward_motion_and_event_censoring():
    np.testing.assert_allclose(mod.axial([180, -180, 90]), [0, 0, -90])
    times = np.arange(7) * .1
    mask = np.array([True, True, True, False, True, True, True])
    events = mod.events(times, mask, np.ones(7) * 45, SETTINGS)
    assert len(events) == 2
    assert events[0]["end_s"] < events[1]["start_s"]
    assert all(event["duration_s"] <= .201 for event in events)


def test_wheel_spin_does_not_become_a_yaw_or_contact_height():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
    <freejoint/>
    <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
    <body name="wheel_L_Link" pos="0 .2 0"><joint name="wheel_L_Joint" axis="0 1 0"/>
    <geom name="wheel_L_collision" type="cylinder" size=".127 .005" quat=".7071067811865476 .7071067811865476 0 0"/></body>
    <body name="wheel_R_Link" pos="0 -.2 0"><joint name="wheel_R_Joint" axis="0 1 0"/>
    <geom name="wheel_R_collision" type="cylinder" size=".127 .005" quat=".7071067811865476 .7071067811865476 0 0"/></body>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, .5, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    heading0, bottom0 = mod.rolling_geometry(model, data)
    np.testing.assert_allclose(heading0, 0)
    np.testing.assert_allclose(bottom0, .5 - .127)
    data.qpos[7:] = [1.2, -2.3]
    mujoco.mj_forward(model, data)
    heading1, bottom1 = mod.rolling_geometry(model, data)
    np.testing.assert_allclose(heading1, heading0, atol=1e-12)
    np.testing.assert_allclose(bottom1, bottom0, atol=1e-12)
    # A tilted cylinder's lowest point is not simply center_z - radius.
    data.qpos[3:7] = Rotation.from_euler("x", 30, degrees=True).as_quat(scalar_first=True)
    mujoco.mj_forward(model, data)
    _, bottom2 = mod.rolling_geometry(model, data)
    extent = .127 * np.cos(np.pi / 6) + .005 * np.sin(np.pi / 6)
    np.testing.assert_allclose(bottom2, [.6 - extent, .4 - extent], atol=1e-12)


def test_real_bvh_matches_pinned_vendor_euler_and_fk(tmp_path):
    GMR, params, extract, utils = mod.load_gmr(SETTINGS)
    text = '''HIERARCHY
ROOT hip
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT lfemur
 {
  OFFSET 1 0 0
  CHANNELS 3 Zrotation Yrotation Xrotation
  JOINT ltibia
  {
   OFFSET 0 -1 0
   CHANNELS 3 Zrotation Yrotation Xrotation
   JOINT lfoot
   {
    OFFSET 0 -1 0
    CHANNELS 3 Zrotation Yrotation Xrotation
   }
  }
 }
 JOINT rfemur
 {
  OFFSET -1 0 0
  CHANNELS 3 Zrotation Yrotation Xrotation
  JOINT rtibia
  {
   OFFSET 0 -1 0
   CHANNELS 3 Zrotation Yrotation Xrotation
   JOINT rfoot
   {
    OFFSET 0 -1 0
    CHANNELS 3 Zrotation Yrotation Xrotation
   }
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.01
'''
    frame = "0 0 0 20 30 40 " + "0 0 0 " * 6
    path = tmp_path / "test.bvh"
    path.write_text(text + frame + "\n" + frame + "\n")
    result = mod.read_source(path, extract, utils, SETTINGS)
    np.testing.assert_allclose(result["root_quat_wxyz"][0], [1, 0, 0, 0], atol=1e-14)
    assert result["bvh_channel_count"] == 24
