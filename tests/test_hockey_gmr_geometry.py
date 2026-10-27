"""Check physically known axle/rolling directions and left/right separation."""
import sys
from pathlib import Path

import numpy as np
import pytest
import mink
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retarget_hockey_gvhmr_gmr import wheel_headings_from_axles, cylinder_body_height, WorldPositionFrameTask


def test_independent_left_right_headings_and_time_axis():
    headings = np.array([[0, 90], [30, -45], [60, -30], [80, -10]])
    radians = np.radians(headings)
    axles = np.stack((-np.sin(radians), np.cos(radians), np.zeros_like(radians)), axis=-1)
    np.testing.assert_allclose(wheel_headings_from_axles(axles), headings, atol=1e-12)


def test_wheel_spin_does_not_change_rolling_plane_heading():
    yaw = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    spins = np.radians([0, 50, 160, 240])[:, None] * np.array([0, 1, 0])
    orientations = yaw @ Rotation.from_rotvec(spins).as_matrix()
    axles = orientations[..., :, 1]
    axles = np.repeat(axles[:, None], 2, axis=1)
    np.testing.assert_allclose(wheel_headings_from_axles(axles), 35., atol=1e-12)


def test_vertical_axle_does_not_fake_a_valid_heading():
    with pytest.raises(ValueError, match="vertical"):
        wheel_headings_from_axles(np.tile([0., 0., 1.], (3, 2, 1)))


def test_cambered_cylinder_height_agrees_with_independent_surface_sampling():
    # Use explicit points on both circular rims, independently of the support formula.
    angle = np.linspace(0, 2*np.pi, 100001)
    radius, half_width = .127, .005
    center = np.array([0., .008, 0.])
    rim = np.concatenate([np.stack((radius*np.cos(angle), np.full_like(angle, y),
                                    radius*np.sin(angle)), axis=-1)
                          for y in (-half_width, half_width)]) + center
    rotations = Rotation.from_euler("xyz", [[0,0,0],[25,45,80],[-55,10,-30]], degrees=True).as_matrix()
    bottom = np.array([0., .02, .09])
    height = cylinder_body_height(rotations, center, radius, half_width, bottom)
    for R, z, desired in zip(rotations, height, bottom):
        np.testing.assert_allclose((rim @ R.T)[:,2].min()+z, desired, atol=1e-9)


def test_world_position_task_jacobian_and_free_world_height():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    config = mink.Configuration(model)
    q = model.qpos0.copy()
    q[:3] = [.1,.2,.7]
    q[3:] = Rotation.from_euler("xyz",[25,-30,40],degrees=True).as_quat(scalar_first=True)
    config.update(q)
    task = WorldPositionFrameTask(frame_name="base",frame_type="body",position_cost=[1,1,0],orientation_cost=1)
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.identity(),np.array([.2,-.1,1.2])))
    jac = task.compute_jacobian(config)
    for k in range(model.nv):
        v = np.eye(model.nv)[k]
        qp,qm=q.copy(),q.copy()
        mujoco.mj_integratePos(model,qp,v,1e-6);mujoco.mj_integratePos(model,qm,v,-1e-6)
        config.update(qp);ep=task.compute_error(config)
        config.update(qm);em=task.compute_error(config)
        np.testing.assert_allclose((ep-em)/2e-6,jac[:,k],atol=1e-7)
    config.update(q);before=task.compute_error(config)*task.cost
    objective=task.compute_qp_objective(config)
    built=mink.build_ik(config,[task],.02,damping=0,limits=[])
    np.testing.assert_allclose(built.P,objective.H,atol=1e-12)
    np.testing.assert_allclose(built.q,objective.c,atol=1e-12)
    np.testing.assert_allclose(objective.H[2],0,atol=1e-12)
    np.testing.assert_allclose(objective.c[2],0,atol=1e-12)
    q[2] += 1.7;config.update(q)
    np.testing.assert_allclose(task.compute_error(config)*task.cost,before,atol=1e-12)
    shifted=task.compute_qp_objective(config)
    np.testing.assert_allclose(shifted.H,objective.H,atol=1e-12)
    np.testing.assert_allclose(shifted.c,objective.c,atol=1e-12)
