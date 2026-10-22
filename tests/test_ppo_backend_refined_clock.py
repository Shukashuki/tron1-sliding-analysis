"""CPU-only clock/dispatch checks; these do not establish CUDA physics parity."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mujoco_ppo_native import _model_indices, ppo_clock_from_contract
from mujoco_ppo_env import MujocoPPOEnv
from mjlab_tron1_env import MjlabWarpBatch

MODEL = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
MOTION = Path("/home/roy/tron1-learning-from-human/results/2026-10-14-car-drift/matched-friction-mu0p70/motion.npz")
PROFILE = ROOT / "config/car_drift_height_smooth_task.json"


@pytest.mark.parametrize("dt,n", [(.005, 4), (.0025, 8)])
def test_declared_clock_preserves_fifty_hz(dt, n):
    assert ppo_clock_from_contract({"physics_dt": dt, "decimation": n,
                                   "policy_fps": 50., "control_dt_s": .02,
                                   "physics_dt_s": dt}) == (dt, n, .02)
    assert ppo_clock_from_contract({}) == (.005, 4, .02)


@pytest.mark.parametrize("patch", [
    {"physics_dt": True}, {"physics_dt": float("nan")}, {"physics_dt": "0.005"},
    {"physics_dt": .00125, "decimation": 16}, {"physics_dt": .0025},
    {"decimation": 8}, {"decimation": 4.}, {"decimation": True},
    {"physics_dt_s": .0025}, {"control_dt_s": .01},
    {"policy_dt_s": .01}, {"policy_fps": 100.}, {"policy_fps": False},
])
def test_wrong_or_ambiguous_clock_fails_closed(patch):
    with pytest.raises(ValueError, match="PPO clock"):
        ppo_clock_from_contract({"physics_dt": .005, "decimation": 4, **patch})


@pytest.fixture
def make_env(tmp_path):
    if not MODEL.is_file() or not MOTION.is_file():
        pytest.skip("Locked model/reference assets are not downloaded")
    environments = []

    def make(dt=.0025, n=8, **kwargs):
        contract = json.loads((ROOT / "config/nominal_actuator_contract.json").read_text())
        contract.update(physics_dt=dt, decimation=n)
        path = tmp_path / f"contract_{len(environments)}.json"
        path.write_text(json.dumps(contract))
        profile = json.loads(PROFILE.read_text())
        profile["action_rate_weight"] = -.5
        result = MujocoPPOEnv(MODEL, MOTION, path, profile, num_envs=2,
                             training=False, dr_mode="none", **kwargs)
        environments.append(result)
        return result

    yield make
    for environment in environments:
        environment.close()


@pytest.mark.parametrize("dt,n", [(.005, 4), (.0025, 8)])
def test_python_physics_clock_actual_model_and_substep_shapes(make_env, dt, n):
    env = make_env(dt, n)
    assert (env.physics_dt, env.decimation, env.step_dt) == (dt, n, .02)
    assert all(model.opt.timestep == dt for model in env.models)
    assert env.objective.weights["action_rate_l2"] == -.5
    captured, original_step = {}, env.batch.step

    def step(action):
        physical = original_step(action)
        captured.update(physical)
        return physical

    env.batch.step = step
    action = np.zeros((2, 8), dtype=np.float32)
    action[:, :6] = env.reference.archive["joint_pos"][0, env.reference.order[:6]] - env.q0s[:, :6]
    _, reward, _, _, extra = env.step(action)
    assert reward.shape == (2,) and np.isfinite(reward).all()
    assert captured["body_net_forces_w"].shape == (2, n, 9, 3)
    assert captured["applied_torque"].shape == (2, n, 8)
    np.testing.assert_allclose(captured["substep_time_s"],
                               np.tile(np.arange(1, n + 1) * dt, (2, 1)), atol=1.e-14)
    np.testing.assert_array_equal(env.previous_actions, action)
    np.testing.assert_array_equal(extra["reference_frames_for_action"], [0, 0])
    assert env.total_transitions == 2 and env.total_physical_steps == 2 * n
    assert env.metadata()["physics_dt_s"] == dt
    assert env.metadata()["decimation"] == n
    assert env.metadata()["policy_dt_s"] == .02


def test_model_validator_requires_explicit_refined_clock_and_native_abi_stays_200hz(make_env):
    env = make_env()
    model, data = env.models[0], env.datas[0]
    with pytest.raises(ValueError, match="clock/integrator"):
        _model_indices(model, data)
    assert _model_indices(model, data, physics_dt=.0025).shape == (33,)
    with pytest.raises(ValueError, match=r"Native C\+\+ PPO batch ABI"):
        make_env(native_library="/not/loaded/or/built.so")


@pytest.mark.parametrize("dt,n", [(.005, 4), (.0025, 8)])
def test_host_warp_dispatch_pairs_each_step_with_forward_and_real_time(dt, n):
    """Scheduling double only: no Torch, Warp, CUDA or contact simulation."""
    plant = object.__new__(MjlabWarpBatch)
    plant.physics_dt, plant.decimation, plant.step_dt = dt, n, .02
    plant.num_envs, plant.device, plant.stream = 2, "cpu_fixture", None
    plant._closed, plant._failed, plant._aids = False, None, np.arange(8)
    plant.torch = SimpleNamespace(
        cuda=SimpleNamespace(stream=lambda _: nullcontext()),
        as_tensor=lambda value, **_: np.array(value, copy=True),
        stack=lambda values, dim: np.stack(values, axis=dim), float32=np.float32)
    plant.wp = SimpleNamespace(ScopedDevice=lambda _: nullcontext())
    data = SimpleNamespace(ctrl=np.zeros((2, 8)), time=np.zeros(2))
    events = []

    # Real bridge uses Tensor.clone(); this double supplies the same operation.
    class Clock(np.ndarray):
        def clone(self):
            return self.copy()

    data.time = data.time.view(Clock)

    def step():
        events.append("step")
        data.time[:] += dt

    def forward():
        events.append("forward")

    plant.sim = SimpleNamespace(data=data, wp_device="cpu_fixture", step=step, forward=forward)
    plant._motor_torques = lambda action: events.append("motor") or action.copy()
    plant._validate_physics_status = lambda: events.append("health")
    plant._body_contact_forces = lambda: np.zeros((2, 9, 3))
    plant._measure_tensor = lambda: {"time_s": data.time.copy()}
    plant._numpy = lambda value: value
    result = plant.step(np.zeros((2, 8)))
    assert events == ["motor", "step", "health", "forward", "health"] * n
    assert result["body_net_forces_w"].shape == (2, n, 9, 3)
    assert result["applied_torque"].shape == (2, n, 8)
    np.testing.assert_allclose(result["substep_time_s"],
                               np.tile(np.arange(1, n + 1) * dt, (2, 1)), atol=1.e-14)


@pytest.mark.parametrize("dt,n", [(.005, 4), (.0025, 8)])
def test_gpu_dispatch_cpu_double_preserves_guards_and_dynamic_windows(dt, n):
    torch = pytest.importorskip("torch")
    from mjlab_tron1_gpu import GpuMjlabBatch
    plant = object.__new__(GpuMjlabBatch)
    plant.torch, plant.device, plant.num_envs = torch, "cpu", 2
    plant.physics_dt, plant.decimation, plant.step_dt = dt, n, .02
    plant._closed, plant._failed = False, None
    plant._error_flags = torch.zeros(1, dtype=torch.int32)
    plant._contacts = torch.zeros((2, n, 9, 3))
    plant._torques, plant._times = torch.empty((2, n, 8)), torch.empty((2, n))
    plant._aids = torch.arange(8)
    data = SimpleNamespace(ctrl=torch.zeros((2, 8)), time=torch.zeros(2))
    events = []

    def step():
        events.append("step")
        data.time.add_(dt)

    def contact(index):
        plant._contacts[:, index] = index + 1

    plant.sim = SimpleNamespace(data=data, step=step, forward=lambda: events.append("forward"))
    plant._physics_scope = nullcontext
    plant._motor_torques = lambda action: events.append("motor") or action * 12.
    plant._record_physics_status = lambda: events.append("health")
    plant._body_contact_forces_into = contact
    plant._measure_tensor = lambda: {"time_s": data.time}
    result = plant.step_tensors(torch.zeros((2, 8)))
    assert events == ["motor", "step", "health", "forward", "health"] * n
    assert result["body_net_forces_w"].shape == (2, n, 9, 3)
    assert result["applied_torque"].shape == (2, n, 8)
    torch.testing.assert_close(result["substep_time_s"],
                               (torch.arange(1, n + 1) * dt).repeat(2, 1))
    bad = torch.zeros((2, 8))
    bad[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="nonfinite action"):
        plant.step_tensors(bad)
    with pytest.raises(FloatingPointError, match="Latched"):
        plant.step_tensors(torch.zeros((2, 8)))


@pytest.mark.parametrize("dt,n", [(.005, 4), (.0025, 8)])
def test_gpu_buffer_allocation_uses_declared_decimation_with_cpu_fixture(monkeypatch, dt, n):
    torch = pytest.importorskip("torch")
    import mjlab_tron1_gpu as module

    class TorchFixture:
        cuda = SimpleNamespace(Event=lambda: object())

        def __getattr__(self, name):
            return getattr(torch, name)

    def base_init(self, *args, **kwargs):
        self.physics_dt, self.decimation, self.step_dt = dt, n, .02
        self.num_envs, self.device = 2, "cpu"
        self.torch = TorchFixture()
        self.wp = SimpleNamespace(
            get_stream=lambda _: object(), from_torch=lambda value, **_: value,
            zeros=lambda shape, **_: torch.zeros(shape, dtype=torch.int32),
            vec3=object(), int32=object(), bool=object())
        self.sim = SimpleNamespace(wp_device="cpu", wp_model=SimpleNamespace(is_sparse=False))
        self.bids = list(range(9))
        self._geom_bodyid, self._body_to_tracked = torch.arange(9), torch.arange(9)
        self.metadata = {}

    monkeypatch.setattr(MjlabWarpBatch, "__init__", base_init)
    monkeypatch.setattr(module.GpuMjlabBatch, "_physics_scope", lambda _: nullcontext())
    monkeypatch.setattr(module, "_device_kernels", lambda: (object(), object()))
    plant = module.GpuMjlabBatch(None, None, None, None, None)
    assert plant._contacts.shape == (2, n, 9, 3)
    assert plant._torques.shape == (2, n, 8)
    assert plant._times.shape == (2, n)
    assert f"{n} pairs" in plant.metadata["physics_semantics"]


def test_device_task_inherits_declared_clock_and_reward_without_gpu_allocation(make_env, monkeypatch):
    torch = pytest.importorskip("torch")
    import mjlab_gpu_ppo_env as module
    import mujoco_ppo_env as host
    boot = make_env()
    monkeypatch.setattr(host, "MujocoPPOEnv", lambda *a, **kw: boot)

    class CpuTensorFixture:
        # Constructor wiring only; actual GPU startup and physics are not run.
        def __getattr__(self, key):
            return getattr(torch, key)

        @staticmethod
        def device(_):
            return torch.device("cpu")

    monkeypatch.setattr(module, "torch", CpuTensorFixture())
    env = module.GpuTron1PPOEnv("unused", "unused", "unused", "unused", device="cuda:3")
    assert (env.physics_dt, env.decimation, env.step_dt) == (.0025, 8, .02)
    assert env.objective.weights["action_rate_l2"] == -.5
    env.batch.metadata = {"physics_dt_s": .0025, "decimation": 8}
    assert "8 step + 8 forward" in env.metadata()["physics_forward_schedule"]
    assert env.metadata()["physics_dt_s"] == .0025
    assert env.metadata()["decimation"] == 8
