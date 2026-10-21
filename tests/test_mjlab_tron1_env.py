"""CPU mathematical bridge checks; CUDA integration requires explicit opt-in."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mjlab_tron1_env import (MJLAB_COMMIT, MjlabWarpBatch, cvel_to_com_velocities,
                            world_com_velocity_deltas)
from nmpc_tracking_objective import BODY_NAMES

MODEL = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
MOTION = Path("/home/roy/tron1-learning-from-human/results/2026-10-14-car-drift/matched-friction-mu0p70/motion.npz")
CONTRACT = ROOT / "config/nominal_actuator_contract.json"
PROFILE = ROOT / "config/car_drift_height_smooth_task.json"


def test_dependency_import_is_lazy_and_pin_is_explicit():
    assert MJLAB_COMMIT == "4360096b49c1bcc478668c11ce7d458db747e49d"
    # Mathematical utilities are usable without importing mjlab, Torch or Warp.
    assert MjlabWarpBatch.__module__ == "mjlab_tron1_env"


def test_cvel_conversion_at_each_original_link_com_matches_mujoco_jacobian():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.is_file():
        pytest.skip("Original unmerged model asset unavailable")
    from classical_tracking import load_pinned_evaluator
    api = load_pinned_evaluator()
    model, _ = api.load_model(MODEL, no_visual_mesh=True)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(500)
    data.qpos[3:7] = rng.normal(size=4)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)
    ids = np.array([model.body(name).id for name in BODY_NAMES])
    root = model.body("base_Link").id
    linear, angular = cvel_to_com_velocities(data.xipos[ids], data.subtree_com[root], data.cvel[ids])
    for row, body in enumerate(ids):
        jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jp, jr, int(body))
        np.testing.assert_allclose(linear[row], jp@data.qvel, atol=2.e-14, rtol=0.)
        np.testing.assert_allclose(angular[row], jr@data.qvel, atol=2.e-14, rtol=0.)


def test_world_com_increment_conversion_matches_free_joint_kinematics():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.is_file():
        pytest.skip("Original unmerged model asset unavailable")
    from classical_tracking import load_pinned_evaluator
    model, _ = load_pinned_evaluator().load_model(MODEL, no_visual_mesh=True)
    data = mujoco.MjData(model)
    base = model.body("base_Link").id
    rng = np.random.default_rng(12)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        data.qpos[3:7], data.qvel[:] = q, 0.
        model.body_ipos[base] = rng.uniform(-.05, .05, 3)
        mujoco.mj_setConst(model, data)
        # mj_setConst uses/restores its own reference configuration workspace.
        data.qpos[3:7] = q
        delta = rng.normal(size=(1, 6))
        data.qvel[:6] = world_com_velocity_deltas(q[None], model.body_ipos[base][None], delta)[0]
        mujoco.mj_forward(model, data)
        jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jp, jr, base)
        np.testing.assert_allclose(jp@data.qvel, delta[0, :3], atol=1.e-14, rtol=0.)
        np.testing.assert_allclose(jr@data.qvel, delta[0, 3:], atol=1.e-14, rtol=0.)


def test_invalid_push_inputs_rejected():
    with pytest.raises(ValueError):
        world_com_velocity_deltas([[1., 0., 0., 0.]], [[0., 0., 0.]], [[np.nan]*6])


def test_torch_dc_envelope_including_braking_above_no_load_matches_frozen_api():
    torch = pytest.importorskip("torch")
    from classical_tracking import load_pinned_evaluator
    api = load_pinned_evaluator()
    contract = api.read_contract(CONTRACT)
    rng = np.random.default_rng(91)
    count = 50
    actions, q, v = rng.uniform(-1., 1., (count, 8)), rng.normal(size=(count, 8)), rng.uniform(-300., 300., (count, 8))
    q0 = np.zeros((count, 8))
    plant = object.__new__(MjlabWarpBatch)
    plant.torch, plant.contract = torch, contract
    plant._qids = plant._vids = torch.arange(8)
    plant._q0s = torch.as_tensor(q0)
    plant._effort = torch.as_tensor([80.]*6+[12.]*2)
    plant._speed = torch.as_tensor([15.]*6+[100.]*2)
    plant._stall = plant._effort.clone()
    plant.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.as_tensor(q), qvel=torch.as_tensor(v)))
    actual = plant._motor_torques(torch.as_tensor(actions)).numpy()
    expected = np.array([api.action_torques(a, p, velocity, default, contract)
                         for a, p, velocity, default in zip(actions, q, v, q0)])
    np.testing.assert_allclose(actual, expected, atol=1.e-12, rtol=0.)


@pytest.mark.skipif(os.environ.get("TRON1_MJLAB_CUDA_TEST") != "1", reason="Explicit idle-GPU opt-in required")
def test_actual_mjlab_cuda_reset_step_contacts_and_continuity():
    pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    if not MODEL.is_file() or not MOTION.is_file():
        pytest.skip("Locked TRON1 model and reference unavailable")
    from mujoco_ppo_env import MujocoPPOEnv
    env = MujocoPPOEnv(MODEL, MOTION, CONTRACT, PROFILE, num_envs=2,
                       training=False, dr_mode="none", batch_factory=MjlabWarpBatch)
    try:
        measured = env.batch.measure()
        assert measured["qpos"].shape == (2, 15)
        physical = env.batch.step(np.zeros((2, 8), dtype=np.float32))
        assert physical["body_net_forces_w"].shape == (2, 4, 9, 3)
        np.testing.assert_allclose(physical["time_s"], .02, atol=1.e-7, rtol=0.)
        physical2 = env.batch.step(np.zeros((2, 8), dtype=np.float32))
        np.testing.assert_allclose(physical2["time_s"], .04, atol=1.e-7, rtol=0.)
        assert np.linalg.norm(physical2["body_net_forces_w"]) > 0.
        # Only the selected episode is reset; another world's clock persists.
        env.batch.reset([0], measured["qpos"][:1], measured["qvel"][:1])
        np.testing.assert_allclose(env.batch.measure()["time_s"], [0., .04], atol=1.e-7, rtol=0.)
    finally:
        env.close()
