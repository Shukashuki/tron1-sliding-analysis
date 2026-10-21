"""GPU bridge math, scheduling and actual Warp CPU-kernel tests.

These tests do not start a CUDA simulation or claim physical parity.  A separate
explicit idle-GPU smoke test must compare this bridge to the frozen GPU oracle.
"""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mjlab_tron1_gpu as gpu
from mjlab_tron1_env import MjlabWarpBatch, world_com_velocity_deltas


def test_module_import_remains_lazy_and_frozen_base_is_reused():
    code = ("import sys,json; sys.path.insert(0,'scripts'); import mjlab_tron1_gpu; "
            "print(json.dumps({name:name in sys.modules for name in ['torch','warp','mjlab']}))")
    output = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True)
    assert json.loads(output) == {"torch": False, "warp": False, "mjlab": False}
    assert issubclass(gpu.GpuMjlabBatch, MjlabWarpBatch)


def test_no_hidden_host_state_copy_or_dynamic_output_index_in_tensor_api():
    tree = ast.parse(Path(gpu.__file__).read_text())
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    host_calls = []
    for name in ("measure_tensors", "step_tensors", "reset_tensors", "apply_link_velocity_deltas",
                 "_body_contact_forces_into", "_record_physics_status", "_physics_scope",
                 "_raise_if_status_failed"):
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"cpu", "numpy", "nonzero", "tolist", "synchronize"}
                if node.func.attr == "item":
                    host_calls.append(name)
    assert host_calls == ["_raise_if_status_failed"]


def test_gpu_com_push_math_matches_frozen_numpy_for_distinct_world_com_offsets():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(425)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    com = rng.uniform(-.05, .05, (20, 3))
    delta = rng.normal(size=(20, 6))
    result = gpu.world_com_velocity_deltas_torch(
        torch.as_tensor(q), torch.as_tensor(com), torch.as_tensor(delta))
    np.testing.assert_allclose(result.numpy(), world_com_velocity_deltas(q, com, delta),
                               atol=2.e-15, rtol=0.)


def _fake_batch(torch):
    """CPU scheduling double, deliberately NOT a physical simulator."""
    plant = object.__new__(gpu.GpuMjlabBatch)
    plant.torch, plant.device, plant.num_envs = torch, "cpu", 2
    plant._closed, plant._failed = False, None
    plant._error_flags = torch.zeros(1, dtype=torch.int32)
    plant._contacts = torch.empty((2, 4, 9, 3))
    plant._torques = torch.empty((2, 4, 8))
    plant._times = torch.empty((2, 4))
    plant._reset_mask_tensor = torch.zeros(2, dtype=torch.bool)
    plant._reset_mask_wp = plant._reset_mask_tensor
    plant._aids = torch.arange(8)
    plant.base_id = 0
    data = SimpleNamespace(qpos=torch.zeros((2, 15)), qvel=torch.zeros((2, 14)),
                           ctrl=torch.zeros((2, 8)), time=torch.zeros(2))
    data.qpos[:, 3] = 1.
    model = SimpleNamespace(body_ipos=torch.tensor([[[.01, .02, .03]], [[-.02, .04, -.01]]]))
    calls = {"step": 0, "forward": 0, "health": 0, "readback": 0}

    def step():
        calls["step"] += 1
        data.time += .005

    def forward():
        calls["forward"] += 1

    def record():
        calls["health"] += 1

    def contact(substep):
        plant._contacts[:, substep] = substep + 1.

    def readback():
        calls["readback"] += 1
        gpu.GpuMjlabBatch._raise_if_status_failed(plant)

    def reset(_model, _data, *, reset):
        data.time.copy_(torch.where(reset, torch.zeros_like(data.time), data.time))
        data.qvel.copy_(torch.where(reset[:, None], torch.zeros_like(data.qvel), data.qvel))

    plant.sim = SimpleNamespace(data=data, model=model, wp_model=None, wp_data=None,
                                step=step, forward=forward)
    plant.mjwarp = SimpleNamespace(reset_data=reset)
    plant._physics_scope = nullcontext
    plant._record_physics_status = record
    plant._body_contact_forces_into = contact
    plant._raise_if_status_failed = readback
    plant._motor_torques = lambda action: action * 12.
    plant._measure_tensor = lambda: {"qpos": data.qpos, "qvel": data.qvel, "time_s": data.time}
    return plant, calls


def test_control_interval_preserves_four_step_four_forward_and_single_fault_read():
    torch = pytest.importorskip("torch")
    plant, calls = _fake_batch(torch)
    actions = torch.tensor([[2., -2., 0., .1, .2, .3, .4, .5]] * 2)
    result = plant.step_tensors(actions)
    assert calls == {"step": 4, "forward": 4, "health": 8, "readback": 1}
    torch.testing.assert_close(result["time_s"], torch.full((2,), .02))
    assert tuple(result["body_net_forces_w"].shape) == (2, 4, 9, 3)
    torch.testing.assert_close(result["applied_torque"][:, 0], actions.clamp(-1., 1.) * 12.)
    # Readout is deliberately zero-copy, not a silently retained CPU snapshot.
    assert result["qpos"].data_ptr() == plant.sim.data.qpos.data_ptr()
    plant.step_tensors(actions)
    assert calls["step"] == 8 and calls["forward"] == 8 and calls["readback"] == 2


def test_selected_mask_and_index_reset_preserve_other_clock_and_avoid_readback():
    torch = pytest.importorskip("torch")
    plant, calls = _fake_batch(torch)
    plant.sim.data.time[:] = .04
    q = plant.sim.data.qpos.clone()
    q[:, 0] = torch.tensor([4., 7.])
    v = torch.ones((2, 14))
    plant.reset_tensors(torch.tensor([True, False]), q, v)
    torch.testing.assert_close(plant.sim.data.time, torch.tensor([0., .04]))
    torch.testing.assert_close(plant.sim.data.qpos[:, 0], torch.tensor([4., 0.]))
    assert calls["forward"] == 1 and calls["readback"] == 0
    plant.reset_tensors(torch.tensor([1]), q[1:], v[1:])
    torch.testing.assert_close(plant.sim.data.qpos[:, 0], torch.tensor([4., 7.]))
    assert calls["forward"] == 2 and calls["readback"] == 0


def test_push_uses_link_velocity_not_double_com_conversion_and_has_no_host_readback():
    torch = pytest.importorskip("torch")
    plant, calls = _fake_batch(torch)
    mask = torch.tensor([True, False])
    plant.sim.data.qpos[0, 3:7] = torch.tensor([np.sqrt(.5), 0., 0., np.sqrt(.5)])
    delta = torch.tensor([[.5, -.2, .1, .3, .4, -.5], [.1, .2, .3, .4, .5, .6]])
    # World-link linear increment is unchanged despite nonzero COM.  Only the
    # angular increment rotates back through a +90-degree world Z orientation.
    expected = delta.clone()
    expected[0, 3:6] = torch.tensor([.4, -.3, -.5])
    plant.apply_link_velocity_deltas(mask, delta)
    torch.testing.assert_close(plant.sim.data.qvel[0, :6], expected[0])
    torch.testing.assert_close(plant.sim.data.qvel[1], torch.zeros(14))
    assert calls["forward"] == 1 and calls["readback"] == 0


def test_physical_failure_is_latched_and_cannot_be_cleared_by_episode_reset():
    torch = pytest.importorskip("torch")
    plant, _ = _fake_batch(torch)
    plant._error_flags[0] = 4
    with pytest.raises(FloatingPointError, match="constraint capacity"):
        plant.step_tensors(torch.zeros((2, 8)))
    plant._error_flags.zero_()
    with pytest.raises(FloatingPointError, match="Latched"):
        plant.reset_tensors(torch.ones(2, dtype=torch.bool), torch.zeros((2, 15)), torch.zeros((2, 14)))
    with pytest.raises(FloatingPointError, match="Latched"):
        plant.measure_tensors()


def test_invalid_actions_and_duplicate_indices_latch_device_faults():
    torch = pytest.importorskip("torch")
    plant, _ = _fake_batch(torch)
    action = torch.zeros((2, 8))
    action[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite action"):
        plant.step_tensors(action)
    plant, _ = _fake_batch(torch)
    q = plant.sim.data.qpos.clone()
    plant.reset_tensors(torch.tensor([0, 0]), q, torch.zeros((2, 14)))
    with pytest.raises(FloatingPointError, match="invalid reset"):
        plant.step_tensors(torch.zeros((2, 8)))


def test_real_fixed_capacity_warp_contact_kernel_on_cpu_includes_self_contacts():
    wp = pytest.importorskip("warp")
    contact_kernel, _ = gpu._device_kernels()
    # The two inactive capacity rows intentionally contain invalid IDs.  They
    # must never be read because contact_id>=nacon is checked on the device.
    worlds = np.array([0, 1, 0, 1, -999, -999], dtype=np.int32)
    geoms = np.array([[0, 1], [0, 2], [1, 2], [2, 0], [-99, -99], [-99, -99]], dtype=np.int32)
    forces = np.zeros((6, 6), dtype=np.float32)
    forces[:, :3] = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [1., 3., 5.], [0., 0., 0.], [0., 0., 0.]]
    output = wp.zeros((2, 4, 2), dtype=wp.vec3, device="cpu")
    flags = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(contact_kernel, dim=6, inputs=[
        wp.array([4], dtype=wp.int32, device="cpu"),
        wp.array(worlds, dtype=wp.int32, device="cpu"),
        wp.array(geoms, dtype=wp.vec2i, device="cpu"),
        wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
        wp.array([-1, 0, 1], dtype=wp.int32, device="cpu"),
        wp.array(forces, dtype=wp.spatial_vector, device="cpu"),
        2, 2, output, flags], device="cpu")
    expected = np.zeros((2, 4, 2, 3), dtype=np.float32)
    expected[0, 2, 0] = [-6., -6., -6.]
    expected[0, 2, 1] = [7., 8., 9.]
    expected[1, 2, 1] = [3., 2., 1.]
    np.testing.assert_array_equal(output.numpy(), expected)
    np.testing.assert_array_equal(flags.numpy(), [0])


def test_real_warp_health_kernel_latches_overflow_sparse_and_nonfinite_bits_on_cpu():
    wp = pytest.importorskip("warp")
    _, health_kernel = gpu._device_kernels()
    qpos = np.zeros((2, 15), dtype=np.float32)
    qpos[1, 0] = np.nan
    qvel = np.zeros((2, 14), dtype=np.float32)
    qacc = qvel.copy()
    qacc[0, 2] = np.inf
    addr = np.zeros((2, 4), dtype=np.int32)
    addr[0, 0] = 100
    flags = wp.zeros(1, dtype=wp.int32, device="cpu")
    args = [wp.array([9], dtype=wp.int32, device="cpu"),
            wp.array([8], dtype=wp.int32, device="cpu"),
            wp.array([1, 5], dtype=wp.int32, device="cpu"),
            wp.array(qpos, dtype=float, device="cpu"), wp.array(qvel, dtype=float, device="cpu"),
            wp.array(qacc, dtype=float, device="cpu"), wp.array(qvel, dtype=float, device="cpu"),
            wp.array(addr, dtype=wp.int32, device="cpu"),
            wp.zeros((2, 4), dtype=wp.int32, device="cpu"), 7, 4, 10, True, flags]
    wp.launch(health_kernel, dim=2, inputs=args, device="cpu")
    # Contact, broadphase, constraints, sparse nnz, qpos and qacc all contribute.
    np.testing.assert_array_equal(flags.numpy(), [1 | 2 | 4 | 8 | 16 | 64])
    # A later healthy kernel invocation may not clear an earlier physical fault.
    args[0] = wp.array([0], dtype=wp.int32, device="cpu")
    args[1] = wp.array([0], dtype=wp.int32, device="cpu")
    args[2] = wp.array([0, 0], dtype=wp.int32, device="cpu")
    args[3] = wp.zeros((2, 15), dtype=float, device="cpu")
    args[5] = wp.zeros((2, 14), dtype=float, device="cpu")
    wp.launch(health_kernel, dim=2, inputs=args, device="cpu")
    np.testing.assert_array_equal(flags.numpy(), [1 | 2 | 4 | 8 | 16 | 64])
