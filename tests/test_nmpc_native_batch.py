"""Bounded CPU parity/fault-containment tests; not a controller timing benchmark."""
import copy
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from classical_tracking import Predictor, load_pinned_evaluator
from nmpc_tracking_objective import MujocoObjectiveState
from nmpc_native_batch import (NativeBatchTransition, NativeBatchError, NativeWorkerError,
                               ACTUATOR_SETTINGS, BODY_NAMES, JOINT_NAMES, build_native_library)


class DiagnosticFakeProcess:
    """No subprocess, native library, model creation or physics in these tests."""
    def __init__(self, *, alive=True, exitcode=None):
        self.pid, self.exitcode, self.alive = 12345, exitcode, alive
        self.join_calls, self.terminate_calls, self.kill_calls = [], 0, 0
    def is_alive(self):
        return self.alive
    def join(self, timeout):
        self.join_calls.append(timeout)
    def terminate(self):
        self.terminate_calls += 1
        self.alive, self.exitcode = False, -15
    def kill(self):
        self.kill_calls += 1
        self.alive, self.exitcode = False, -9


class DiagnosticFakePipe:
    def __init__(self, response=None, *, poll=True, receive_error=None, send_error=None):
        self.response, self.poll_result = response, poll
        self.receive_error, self.send_error = receive_error, send_error
        self.sent, self.closed = [], False
    def poll(self, timeout):
        return self.poll_result
    def recv(self):
        if self.receive_error is not None:
            raise self.receive_error
        return self.response
    def send(self, message):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(message)
    def close(self):
        self.closed = True


@pytest.fixture
def diagnostic_fake_native(tmp_path):
    instances = []
    def make(pipe, *, alive=True, exitcode=None):
        native = object.__new__(NativeBatchTransition)
        native._lock, native._closed = threading.Lock(), False
        native._process = DiagnosticFakeProcess(alive=alive, exitcode=exitcode)
        native._pipe, native.timeout_s = pipe, 30.
        native._workdir = SimpleNamespace(name=str(tmp_path), cleanup=lambda: None)
        instances.append(native)
        return native
    yield make
    # No real resources exist; prevent destructor requests outside assertions.
    for native in instances:
        native._closed = True


def test_diagnostic_exception_optional_metadata_preserves_runtimeerror_args_and_str():
    detail = {"worker": {"alive": True}}
    error = NativeBatchError("same", 7, diagnostics=detail)
    assert error.args == ("same", 7) and str(error) == str(RuntimeError("same", 7))
    detail["worker"]["alive"] = False
    assert error.diagnostics == {"worker": {"alive": True}}
    assert str(NativeWorkerError("old text")) == "old text"
    assert str(NativeBatchError()) == ""


@pytest.mark.parametrize("bad", [object(), {"unknown": True}, {"event": np.array([1.])},
                                  {"worker": {"alive": {"deep": True}}}, {"timeout_s": float("nan")}])
def test_diagnostic_malformed_constructor_metadata_never_replaces_original_error(bad):
    error = NativeBatchError("Keep original", diagnostics=bad)
    assert str(error) == "Keep original" and error.diagnostics is None


def test_diagnostic_constructor_never_invokes_deepcopy_hooks_and_bounds_text():
    class BadCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("Not a primitive")
    error = NativeWorkerError("Original", diagnostics=BadCopy())
    assert str(error) == "Original" and error.diagnostics is None
    detail = NativeBatchError("Original", diagnostics={"event": "é"*10000}).diagnostics
    assert len(detail["event"].encode("utf-8")) <= 2048
    assert len(json.dumps(detail).encode("utf-8")) <= 8192


@pytest.mark.parametrize("event,extra,message", [
    ("native_c_return", {"native_return_code": 1}, "MuJoCo warning during native physics step"),
    ("child_exception", {"child_exception_type": "ValueError"}, "ValueError: Native worker plan shape mismatch"),
    ("child_outer_exception", {"child_exception_type": "ValueError"}, "ValueError: Unknown native worker request"),
])
def test_diagnostic_negative_child_reply_keeps_message_and_live_worker(diagnostic_fake_native, event, extra, message):
    pipe = DiagnosticFakePipe({"ok": False, "error": message, "error_kind": event, **extra})
    native = diagnostic_fake_native(pipe)
    with pytest.raises(NativeBatchError) as caught:
        native._request({"command": "rollout"})
    assert type(caught.value) is NativeBatchError and str(caught.value) == message
    diagnostic = caught.value.diagnostics
    assert diagnostic["event"] == event
    assert diagnostic["worker"] == {"alive": True, "closed": False, "pid": 12345, "exitcode": None}
    for key, value in extra.items():
        assert diagnostic[key] == value
    assert diagnostic["parent_stop_initiated"] is False
    assert native.worker_alive and native._process.terminate_calls == 0
    assert native._process.join_calls == [] and not pipe.closed
    json.dumps(diagnostic, allow_nan=False)


def test_diagnostic_eof_retains_pre_stop_exitcode_log_and_terminal_no_restart(diagnostic_fake_native, tmp_path):
    details = "fatal native detail"
    (tmp_path/"MUJOCO_LOG.TXT").write_text(details)
    pipe = DiagnosticFakePipe(receive_error=EOFError("pipe EOF"))
    native = diagnostic_fake_native(pipe, alive=False, exitcode=-11)
    with pytest.raises(NativeWorkerError) as caught:
        native._receive()
    assert str(caught.value) == "Native worker exited (code=-11); "+details
    assert isinstance(caught.value.__cause__, EOFError)
    diagnostic = caught.value.diagnostics
    assert diagnostic["event"] == "receive_ipc_error"
    assert diagnostic["before_stop"] == {"alive": False, "closed": False, "pid": 12345, "exitcode": -11}
    assert diagnostic["after_stop"] == {"alive": False, "closed": True, "pid": 12345, "exitcode": -11}
    assert diagnostic["mujoco_log_tail"] == details
    assert native._process.terminate_calls == 0
    (tmp_path/"MUJOCO_LOG.TXT").unlink()
    assert diagnostic["mujoco_log_tail"] == details
    with pytest.raises(NativeWorkerError, match="closed or terminal") as repeated:
        native._request({"command": "rollout"})
    assert repeated.value.diagnostics["event"] == "request_worker_unavailable"
    assert repeated.value.diagnostics["worker"]["exitcode"] == -11
    assert pipe.sent == []


def test_diagnostic_timeout_distinguishes_parent_stop_from_prior_worker_exit(diagnostic_fake_native):
    pipe = DiagnosticFakePipe(poll=False)
    native = diagnostic_fake_native(pipe)
    with pytest.raises(NativeWorkerError) as caught:
        native._receive()
    assert str(caught.value) == "Native worker timeout; isolated worker stopped, no automatic restart"
    diagnostic = caught.value.diagnostics
    assert diagnostic["event"] == "receive_timeout" and diagnostic["timeout_s"] == 30.
    assert diagnostic["before_stop"]["exitcode"] is None and diagnostic["before_stop"]["alive"] is True
    assert diagnostic["after_stop"]["exitcode"] == -15 and diagnostic["after_stop"]["closed"] is True
    assert diagnostic["parent_stop_initiated"] is True
    assert native._process.terminate_calls == 1
    assert native._process.join_calls == [3.]
    with pytest.raises(NativeWorkerError, match="no automatic restart"):
        native._request({"command": "rollout"})
    assert pipe.sent == []


def test_diagnostic_send_failure_retains_original_str_and_before_after_stop(diagnostic_fake_native):
    pipe = DiagnosticFakePipe(send_error=BrokenPipeError("broken"))
    native = diagnostic_fake_native(pipe)
    with pytest.raises(NativeWorkerError) as caught:
        native._request({"command": "rollout"})
    assert str(caught.value) == "Native worker connection closed"
    diagnostic = caught.value.diagnostics
    assert diagnostic["event"] == "send_ipc_error" and diagnostic["ipc_exception_type"] == "BrokenPipeError"
    assert diagnostic["before_stop"]["alive"] is True
    assert diagnostic["after_stop"]["exitcode"] == -15
    assert native._process.terminate_calls == 1


@pytest.mark.parametrize("native_failure", [False, True])
def test_diagnostic_worker_protocol_tags_errors_without_native_or_physics(monkeypatch, tmp_path, native_failure):
    import resource
    import nmpc_native_batch as module
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(module.os, "chdir", lambda path: None)
    monkeypatch.setattr(resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(module, "_validated_library", lambda path: (Path(path), {}, Path("fake_mujoco")))
    monkeypatch.setattr(module, "_no_callbacks", lambda: None)
    monkeypatch.setattr(module.pickle, "loads", lambda value: SimpleNamespace(_address=1))
    addresses = iter((2, 3))
    monkeypatch.setattr(module.mujoco, "MjData", lambda model: SimpleNamespace(_address=next(addresses)))
    calls = []
    def fake_c(*args):
        calls.append(args)
        args[-2].value = b"Original native return text"
        return 1
    monkeypatch.setattr(module, "_load_child_library", lambda *args: SimpleNamespace(tron1_batch=fake_c))
    request = {"command": "rollout", "qpos": np.zeros((1,15)), "qvel": np.zeros((1,14)),
               "history": np.zeros((1,3,9,3)), "actions": np.zeros((1,1,8) if native_failure else (1,8))}
    messages = iter((request, {"command": "close"}))
    class ChildPipe:
        def __init__(self):
            self.sent, self.closed = [], False
        def recv(self):
            return next(messages)
        def send(self, response):
            self.sent.append(response)
        def close(self):
            self.closed = True
    pipe = ChildPipe()
    module._worker(pipe, "fake_native", b"trusted_fake_model", np.zeros(33, dtype=np.int32),
                   np.zeros(8), str(tmp_path), 512*1024**2)
    assert pipe.closed and len(pipe.sent) == 3
    ready, error, closed = pipe.sent
    assert ready["ok"] and closed == {"ok": True}  # Loop remains usable after either rejection.
    if native_failure:
        assert len(calls) == 1
        assert error == {"ok": False, "error": "Original native return text", "error_kind": "native_c_return", "native_return_code": 1}
    else:
        assert calls == []
        assert error == {"ok": False, "error": "ValueError: Native worker plan shape mismatch", "error_kind": "child_exception", "child_exception_type": "ValueError"}


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    if mujoco.__version__ != "3.8.1" or sys.platform != "linux":
        pytest.skip("Optional prototype requires Linux MuJoCo 3.8.1")
    return build_native_library(tmp_path_factory.mktemp("native_batch_build"))


def contract():
    return {**ACTUATOR_SETTINGS, "actuator_model": "dc_motor", "physics_dt": .005, "decimation": 4}


def synthetic_model():
    legs = []
    for side, lateral in (("L", .14), ("R", -.14)):
        legs.append(f'<body name="abad_{side}_Link" pos="0 {lateral} 0">'
                    f'<joint name="abad_{side}_Joint" axis="1 0 0"/>'
                    '<inertial pos=".02 0 0" mass=".3" diaginertia=".003 .003 .003"/>'
                    f'<body name="hip_{side}_Link" pos="0 0 -.1">'
                    f'<joint name="hip_{side}_Joint" axis="0 1 0"/>'
                    '<inertial pos=".03 0 -.04" mass=".5" diaginertia=".005 .005 .005"/>'
                    f'<body name="knee_{side}_Link" pos="0 0 -.15">'
                    f'<joint name="knee_{side}_Joint" axis="0 1 0"/>'
                    '<inertial pos=".02 0 -.05" mass=".4" diaginertia=".004 .004 .004"/>'
                    f'<body name="wheel_{side}_Link" pos="0 0 -.15">'
                    f'<joint name="wheel_{side}_Joint" axis="0 1 0" limited="false"/>'
                    '<inertial pos=".01 0 0" mass=".2" diaginertia=".002 .002 .002"/>'
                    '<geom type="sphere" size=".08"/>'
                    '</body></body></body></body>')
    # Deliberately use MJCF-native leg-by-leg order, not logical action order.
    native_order = [f"{name}_{side}_Joint" for side in ("L", "R") for name in ("abad", "hip", "knee", "wheel")]
    actuators = ''.join(f'<motor name="{name}" joint="{name}" ctrllimited="true" '
                        f'ctrlrange="{-12 if name.startswith("wheel") else -80} '
                        f'{12 if name.startswith("wheel") else 80}"/>' for name in native_order)
    xml = ('<mujoco><compiler angle="radian"/><option timestep=".005" integrator="implicitfast" '
           'solver="Newton" iterations="100" tolerance="1e-10"/>'
           '<default><joint range="-2 2" damping="0" armature="0"/>'
           '<geom friction=".7 0 0" condim="3" solref=".01 1"/></default>'
           '<worldbody><geom type="plane" size="2 2 .1"/>'
           '<body name="base_Link" pos="0 0 .48"><freejoint/>'
           '<inertial pos=".05 0 -.02" quat=".995004 0 .0998334 0" mass="2" diaginertia=".05 .05 .05"/>'
           '<geom type="box" size=".1 .1 .04"/>' + ''.join(legs) +
           '</body></worldbody><actuator>' + actuators + '</actuator></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def inputs(model, *, batch=2, horizon=3):
    q = np.tile(model.qpos0, (batch, 1))
    q[:, 2] -= .005
    v = np.zeros((batch, model.nv))
    vids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
    v[:, :3] = [1., .3, -.1]
    v[:, 3:6] = [.1, -.2, .3]
    v[0, vids] = [20., -20., 8., -8., 12., -12., 150., -150.]
    # One moderate-velocity row remains in contact after integration rather
    # than only touching during mj_step's pre-integration collision solve.
    q[-1, 2] -= .01
    rng = np.random.default_rng(91)
    actions = rng.uniform(-1, 1, (batch, horizon, 8))
    actions[0, 0] = [-1., 1., -1., 1., -1., 1., -1., 1.]
    history = rng.normal(size=(batch, 3, 9, 3))
    return q, v, actions, history


def python_batch(model, q0, q, v, actions, history):
    """Unchanged torque helper and Jacobian/contact extractor as the oracle."""
    model = copy.copy(model)
    d, measure = mujoco.MjData(model), mujoco.MjData(model)
    extractor = MujocoObjectiveState(model)
    qids = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
    vids = np.array([int(model.joint(name).dofadr[0]) for name in JOINT_NAMES])
    aids = np.array([int(model.actuator(name).id) for name in JOINT_NAMES])
    helper = SimpleNamespace(qids=qids, vids=vids, q0=q0,
                             effort=np.r_[[80.]*6, [12.]*2], no_load=np.r_[[15.]*6, [100.]*2])
    output = {name: [] for name in ("pre_qpos", "pre_qvel", "pre_body_pos_w", "pre_body_quat_w",
              "pre_body_lin_vel_w", "pre_body_ang_vel_w", "qpos", "qvel", "torques", "body_pos_w",
              "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_net_forces_w",
              "interval_time_s", "contact_count", "contact_force_history_w")}
    for initial_q, initial_v, plan, initial_history in zip(q, v, actions, history):
        state_q, state_v, hist = initial_q.copy(), initial_v.copy(), initial_history.copy()
        rows = {name: [] for name in output}
        for action in plan:
            mujoco.mj_resetData(model, measure)
            measure.qpos[:], measure.qvel[:] = state_q, state_v
            mujoco.mj_forward(model, measure)
            pre = extractor.extract(measure)
            rows["pre_qpos"].append(state_q.copy()); rows["pre_qvel"].append(state_v.copy())
            for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                rows["pre_" + name].append(pre[name].copy())
            mujoco.mj_resetData(model, d)
            d.qpos[:], d.qvel[:] = state_q, state_v
            stages = {name: [] for name in output if not name.startswith("pre_") and name != "contact_force_history_w"}
            for _ in range(4):
                tau = Predictor.torque(helper, action, d.qpos, d.qvel)
                d.ctrl[aids] = tau
                mujoco.mj_step(model, d); mujoco.mj_forward(model, d)
                hist, force = extractor.update_contact_history(d, hist)
                measured = extractor.extract(d, body_net_forces_w=force)
                for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_net_forces_w"):
                    stages[name].append(measured[name].copy())
                stages["qpos"].append(d.qpos.copy()); stages["qvel"].append(d.qvel.copy())
                stages["torques"].append(tau.copy()); stages["interval_time_s"].append(d.time)
                stages["contact_count"].append(d.ncon)
            for name, values in stages.items(): rows[name].append(np.array(values))
            rows["contact_force_history_w"].append(hist.copy())
            state_q, state_v = d.qpos.copy(), d.qvel.copy()
        for name, values in rows.items(): output[name].append(np.array(values))
    return {name: np.array(value) for name, value in output.items()}


def assert_parity(native, expected):
    assert set(native) == set(expected)
    for name in expected:
        np.testing.assert_allclose(native[name], expected[name], rtol=1e-10, atol=1e-9, err_msg=name)


def test_synthetic_contact_saturation_and_all_substep_measurements_match(native_library):
    model = synthetic_model(); q0 = np.linspace(-.1, .1, 8)
    q, v, actions, history = inputs(model)
    # Retain a quiet, lightly loaded contact row while the first row exercises
    # hard speed/torque saturation. Random high-gain leg targets can launch the
    # tiny synthetic robot before post-step contact forces are measured.
    q[-1, 2] = model.qpos0[2] - .001
    v[-1] = 0.
    actions[-1] = -q0
    before = [value.copy() for value in (q, v, actions, history)]
    expected = python_batch(model, q0, q, v, actions, history)
    with NativeBatchTransition(model, q0, contract(), native_library) as native:
        actual = native.rollout(q, v, actions, initial_history=history)
        assert native.worker_alive
        assert native.metadata["fatal_errors_process_isolated"] is True
        assert native.metadata["production_cem_wired"] is False
        assert_parity(actual, expected)
        assert np.max(actual["contact_count"]) > 0
        assert np.max(np.abs(actual["body_net_forces_w"])) > 0
        assert np.max(np.abs(actual["torques"][..., :6])) == pytest.approx(80.)
        np.testing.assert_allclose(actual["interval_time_s"], np.broadcast_to([.005,.01,.015,.02], (2,3,4)))
        assert_parity(native.rollout(q, v, actions, initial_history=history), expected)
    for value, original in zip((q, v, actions, history), before): np.testing.assert_array_equal(value, original)


def test_model_snapshot_ownership_and_batch_order_cold_reset(native_library):
    model = synthetic_model(); q0 = np.zeros(8)
    q, v, actions, history = inputs(model)
    expected = python_batch(model, q0, q, v, actions, history)
    with NativeBatchTransition(model, q0, contract(), native_library) as native:
        # Mutating this test-owned original cannot change the child snapshot.
        model.opt.timestep = .01
        model.geom_friction[:, 0] = .01
        actual = native.rollout(q[::-1], v[::-1], actions[::-1], initial_history=history[::-1])
        assert_parity(actual, {name: value[::-1] for name, value in expected.items()})


def test_invalid_arrays_fail_before_native_call_and_worker_remains_valid(native_library):
    model = synthetic_model(); q0 = np.zeros(8)
    q, v, actions, history = inputs(model, batch=1, horizon=1)
    with NativeBatchTransition(model, q0, contract(), native_library) as native:
        wrong = actions.copy(); wrong[0,0,0] = 2.
        with pytest.raises(ValueError, match="normalized"):
            native.rollout(q, v, wrong, initial_history=history)
        wrong = q.copy(); wrong[0,3:7] = 0
        with pytest.raises(ValueError, match="quaternions"):
            native.rollout(wrong, v, actions, initial_history=history)
        wrong = v.copy(); wrong[0,0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            native.rollout(q, wrong, actions, initial_history=history)
        native.rollout(q, v, actions, initial_history=history)


def test_native_validation_error_does_not_kill_the_isolated_worker(native_library):
    model = synthetic_model(); q0 = np.zeros(8)
    q, v, actions, history = inputs(model, batch=1, horizon=1)
    with NativeBatchTransition(model, q0, contract(), native_library) as native:
        wrong = actions.copy(); wrong[0,0,0] = 2.
        with pytest.raises(NativeBatchError, match="normalized"):
            native._request({"command":"rollout", "qpos":q, "qvel":v, "history":history, "actions":wrong})
        assert native.worker_alive
        native.rollout(q, v, actions, initial_history=history)


def test_real_native_fatal_error_is_contained_and_never_silently_restarted(native_library):
    parent_pid = os.getpid()
    native = NativeBatchTransition(synthetic_model(), np.zeros(8), contract(), native_library)
    try:
        with pytest.raises(NativeWorkerError, match="Native worker exited"):
            native._request({"command":"_test_native_fatal"})
        assert os.getpid() == parent_pid and not native.worker_alive
        with pytest.raises(NativeWorkerError, match="terminal"):
            native._request({"command":"_test_native_fatal"})
    finally:
        native.close()
        native.close()


def test_wrong_physics_or_actuator_contract_and_build_overwrite_are_rejected(native_library):
    model = synthetic_model(); model.opt.timestep = .002
    with pytest.raises(ValueError, match="integration contract"):
        NativeBatchTransition(model, np.zeros(8), contract(), native_library)
    with pytest.raises(ValueError, match="actuator/clock"):
        NativeBatchTransition(synthetic_model(), np.zeros(8), {**contract(), "leg_kp":100.}, native_library)
    with pytest.raises(FileExistsError, match="Preserving"):
        build_native_library(native_library.parent)


@pytest.mark.parametrize("mu", [.16, .7])
def test_licensed_tron1_contact_and_saturation_substeps_match_existing_helpers(native_library, mu):
    raw = ROOT / "outputs/car_failure_sideslip_20261015/raw"
    model_path = raw / "mujoco_model.xml"
    motion = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz")
    contract_path = raw / "mu0p180/isaac/policy_contract.json"
    if not all(path.is_file() for path in (model_path, motion, contract_path)):
        pytest.skip("Licensed local TRON1 model/locked reference unavailable")
    api = load_pinned_evaluator()
    configuration = api.read_contract(contract_path)
    model, _ = api.load_model(model_path, no_visual_mesh=True)
    api.configure_model(model, configuration)
    model.geom_friction[:, 0] = mu
    reference = api.MotionReference(motion, configuration)
    data = mujoco.MjData(model)
    api.MujocoState(model, data, reference).initialize()
    q, v, actions, history = inputs(model)
    q[:] = data.qpos
    q[0, 2] -= .008
    q[1, 2] += .015
    expected = python_batch(model, reference.q0, q, v, actions, history)
    with NativeBatchTransition(model, reference.q0, configuration, native_library) as native:
        actual = native.rollout(q, v, actions, initial_history=history)
        assert_parity(actual, expected)
        assert np.max(actual["contact_count"]) > 0
        assert np.max(np.abs(actual["body_net_forces_w"])) > 0
