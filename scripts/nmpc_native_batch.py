"""OPTIONAL isolated, single-worker native transition prototype; not wired to CEM.

Preserves the current 20 ms cold reset and four 5 ms torque/step/forward/contact
samples. It does not use stock rollout, callbacks, hidden root forces, a new
actuator model, objective approximation, or a real-time claim. IPC overhead is
part of any future end-to-end latency measurement, not something to subtract.

Build is explicit, local and dependency-free beyond the installed C++ compiler
and MuJoCo 3.8.1 wheel's headers/library. A persistent subprocess owns the model
snapshot and both MjData workspaces; fatal MuJoCo errors or segmentation faults
cannot terminate the calling controller process. There is no implicit restart.

The underscored _address properties are not advertised as a stable Python API.
Their native-pointer meanings were inspected in official 3.8.1 structs.cc:
https://github.com/google-deepmind/mujoco/blob/3.8.1/python/mujoco/structs.cc .
This module deliberately version-locks that bridge and never accepts pointers
from a caller. Only owned, live MuJoCo wrappers inside the child are passed to C.
The model pickle is created from a live trusted MjModel by this parent process;
this is not a loader for untrusted downloaded pickle/checkpoint files.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import threading

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/nmpc_batch.cpp"
VERSION = "3.8.1"
VERSION_NUMBER = 3008001
JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
               "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")
BODY_NAMES = ("base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
              "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link")
PRE_WIDTH, STEP_WIDTH, STAGE_WIDTH = 146, 183, 959
CALLBACKS = ("control", "passive", "sensor", "contactfilter", "act_dyn", "act_gain", "act_bias")
ACTUATOR_SETTINGS = {"leg_position_scale": 1., "leg_kp": 500., "leg_kd": 10.,
                     "wheel_torque_scale_nm": 12., "leg_torque_limit_nm": 80.,
                     "wheel_torque_limit_nm": 12., "leg_saturation_effort_nm": 80.,
                     "wheel_saturation_effort_nm": 12., "leg_motor_velocity_limit_rad_s": 15.,
                     "wheel_motor_velocity_limit_rad_s": 100.}


def _bounded_error_diagnostics(value):
    """Detached primitive allowlist, bounded before reaching caller telemetry."""
    if value is None:
        return None
    fields = {"event", "native_return_code", "child_exception_type", "ipc_exception_type",
              "parent_stop_initiated", "timeout_s", "mujoco_log_tail", "worker", "before_stop", "after_stop"}
    status_fields = {"alive", "closed", "pid", "exitcode"}
    def primitive(item):
        if item is None or type(item) in (bool, int):
            return item
        if type(item) is float and np.isfinite(item):
            return item
        if type(item) is str:
            return item.encode("utf-8", errors="replace")[:2048].decode("utf-8", errors="ignore")
        raise TypeError("Native diagnostics require finite built-in primitives")
    if type(value) is not dict or any(type(k) is not str or k not in fields for k in value):
        raise TypeError("Native diagnostics contain unsupported fields")
    result = {}
    for key, item in value.items():
        if key in ("worker", "before_stop", "after_stop"):
            if type(item) is not dict or any(type(k) is not str or k not in status_fields for k in item):
                raise TypeError("Native worker diagnostics contain unsupported fields")
            result[key] = {k: primitive(v) for k, v in item.items()}
        else:
            result[key] = primitive(item)
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > 8192:
        raise ValueError("Native diagnostic byte budget exceeded")
    return result


class NativeBatchError(RuntimeError):
    def __init__(self, *args, diagnostics=None):
        # Optional, detached telemetry; preserve existing exception classes,
        # message text and RuntimeError catch behavior.
        super().__init__(*args)
        try:
            self.diagnostics = _bounded_error_diagnostics(diagnostics)
        except Exception:
            self.diagnostics = None


class NativeWorkerError(NativeBatchError):
    """Isolated worker exited, timed out, or failed initialization; no restart."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _installation():
    if sys.platform != "linux" or mujoco.__version__ != VERSION or mujoco.mj_version() != VERSION_NUMBER:
        raise NativeBatchError("Native prototype requires Linux and exact MuJoCo 3.8.1")
    package = Path(mujoco.__file__).resolve().parent
    include = package / "include"
    library = package / "libmujoco.so.3.8.1"
    header = include / "mujoco/mujoco.h"
    if not header.is_file() or not library.is_file():
        raise NativeBatchError("Pinned installed MuJoCo headers/library are unavailable")
    if "#define mjVERSION_HEADER 3008001" not in header.read_text():
        raise NativeBatchError("Installed header differs from MuJoCo 3.8.1")
    return include, library.resolve(), header


def build_native_library(output_dir, *, compiler="c++"):
    """Explicit opt-in build; refuses to overwrite either binary or manifest."""
    include, mujoco_library, header = _installation()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = output / "libtron1_nmpc_batch.so"
    manifest = library.with_suffix(".json")
    if library.exists() or manifest.exists():
        raise FileExistsError("Preserving existing native prototype build")
    command = [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", str(SOURCE),
               "-I" + str(include), str(mujoco_library), "-Wl,-rpath," + str(mujoco_library.parent),
               "-ldl", "-o", str(library)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = {"schema_version": 1, "abi_version": 1, "mujoco_version": VERSION,
                "header_version": VERSION_NUMBER, "source_sha256": _sha(SOURCE),
                "mujoco_header_sha256": _sha(header), "mujoco_library_sha256": _sha(mujoco_library),
                "native_library_sha256": _sha(library), "compiler_command": command,
                "experimental_optional_backend": True, "production_cem_wired": False,
                "workers": 1, "global_callbacks_installed": False}
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return library


def _validated_library(path):
    include, mujoco_library, header = _installation()
    path = Path(path).resolve(strict=True)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("mujoco_version") != VERSION
            or metadata.get("abi_version") != 1 or metadata.get("header_version") != VERSION_NUMBER
            or metadata.get("source_sha256") != _sha(SOURCE)
            or metadata.get("native_library_sha256") != _sha(path)
            or metadata.get("mujoco_header_sha256") != _sha(header)
            or metadata.get("mujoco_library_sha256") != _sha(mujoco_library)):
        raise NativeBatchError("Native build provenance/version/hash mismatch")
    return path, metadata, mujoco_library


def _no_callbacks():
    # Read the public C variables as well: Python getters do not report a
    # callback installed directly by native code. Do not reject mjcb_time:
    # Python's MjData constructor installs its ordinary native profiler timer.
    _, path, _ = _installation()
    library = ctypes.CDLL(str(path))
    if any(ctypes.c_void_p.in_dll(library, "mjcb_" + name).value is not None for name in CALLBACKS):
        raise NativeBatchError("Native prototype requires callback-free physics")


def _validate_model(model, contract):
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("Pass a live owned MuJoCo MjModel, not a raw pointer or filename")
    _no_callbacks()
    if (model.nq != 15 or model.nv != 14 or model.nu != 8
            or any(getattr(model, name) for name in
                   ("na", "nmocap", "nuserdata", "nplugin", "nhistory", "nflex", "neq", "ntendon"))
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            or model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON
            or not np.isclose(model.opt.timestep, .005, rtol=0, atol=1e-15)):
        raise ValueError("Unsupported model size, hidden state or integration contract")
    if (contract.get("actuator_model") != "dc_motor"
            or any(not np.isclose(contract.get(key, np.nan), value, rtol=0, atol=1e-12)
                   for key, value in ACTUATOR_SETTINGS.items())
            or contract.get("physics_dt") != .005 or contract.get("decimation") != 4):
        raise ValueError("Native prototype requires the unchanged original actuator/clock contract")
    qids = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES], dtype=np.int32)
    vids = np.array([int(model.joint(name).dofadr[0]) for name in JOINT_NAMES], dtype=np.int32)
    aids = np.array([int(model.actuator(name).id) for name in JOINT_NAMES], dtype=np.int32)
    bids = np.array([int(model.body(name).id) for name in BODY_NAMES], dtype=np.int32)
    if (model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or model.jnt_qposadr[0] != 0 or model.jnt_dofadr[0] != 0):
        raise ValueError("Native prototype requires the existing floating-base layout")
    for logical, actuator in enumerate(aids):
        joint = int(model.actuator_trnid[actuator, 0])
        effort = 80. if logical < 6 else 12.
        if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or model.jnt_qposadr[joint] != qids[logical] or model.jnt_dofadr[joint] != vids[logical]
                or model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_NONE
                or not np.array_equal(model.actuator_gear[actuator], [1., 0., 0., 0., 0., 0.])
                or model.actuator_gainprm[actuator, 0] != 1.
                or not model.actuator_ctrllimited[actuator]
                or not np.array_equal(model.actuator_ctrlrange[actuator], [-effort, effort])):
            raise ValueError("Native prototype requires the unchanged torque-motor transmission")
    return np.r_[qids, vids, aids, bids].astype(np.int32)


def _load_child_library(path, expected_mujoco):
    lib = ctypes.CDLL(str(path))
    for name, expected in (("tron1_batch_abi", 1), ("tron1_header_version", VERSION_NUMBER),
                           ("tron1_runtime_version", VERSION_NUMBER), ("tron1_real_size", 8),
                           ("tron1_stage_width", STAGE_WIDTH)):
        function = getattr(lib, name)
        function.argtypes, function.restype = [], ctypes.c_int
        if function() != expected:
            raise NativeBatchError("Native ABI/version/precision mismatch: " + name)
    lib.tron1_linked_mujoco_path.argtypes = []
    lib.tron1_linked_mujoco_path.restype = ctypes.c_char_p
    linked = lib.tron1_linked_mujoco_path()
    if not linked or Path(os.fsdecode(linked)).resolve() != expected_mujoco:
        raise NativeBatchError("Native backend linked a different MuJoCo shared library")
    real = ctypes.POINTER(ctypes.c_double)
    integer = ctypes.POINTER(ctypes.c_int)
    lib.tron1_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int, integer, real, real, real, real, real,
                               real, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
    lib.tron1_batch.restype = ctypes.c_int
    return lib


def _worker(pipe, library_path, model_blob, ids, q0, workdir, max_output_bytes):
    """Only this isolated process ever loads/calls the native prototype."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(workdir)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        path, _, expected_mujoco = _validated_library(library_path)
        _no_callbacks()
        library = _load_child_library(path, expected_mujoco)
        model = pickle.loads(model_blob)  # Blob created by this trusted parent only.
        data, measurement = mujoco.MjData(model), mujoco.MjData(model)
        # Both raw addresses refer to live child-owned wrappers, never caller ints.
        addresses = [ctypes.c_void_p(obj._address) for obj in (model, data, measurement)]
        if any(not pointer.value for pointer in addresses) or addresses[1].value == addresses[2].value:
            raise NativeBatchError("Invalid owned native workspace addresses")
        pipe.send({"ok": True, "worker_pid": os.getpid()})
        real = ctypes.POINTER(ctypes.c_double)
        while True:
            message = pipe.recv()
            if message["command"] == "close":
                pipe.send({"ok": True})
                return
            if message["command"] == "_test_native_fatal":
                library.tron1_native_error_probe()
                raise NativeBatchError("Native fatal-error probe unexpectedly returned")
            if message["command"] != "rollout":
                raise ValueError("Unknown native worker request")
            try:
                inputs = [np.require(message[name], dtype=np.float64, requirements=["C", "A"])
                          for name in ("qpos", "qvel", "history", "actions")]
                if inputs[-1].ndim != 3 or inputs[-1].shape[-1] != 8:
                    raise ValueError("Native worker plan shape mismatch")
                batch, horizon = inputs[-1].shape[:2]
                if (not 1 <= batch <= 4096 or not 1 <= horizon <= 1024
                        or inputs[0].shape != (batch, 15) or inputs[1].shape != (batch, 14)
                        or inputs[2].shape != (batch, 3, 9, 3)
                        or batch*horizon*STAGE_WIDTH*8 > max_output_bytes):
                    raise ValueError("Native worker input shape or memory budget mismatch")
                output = np.empty((batch, horizon, STAGE_WIDTH), dtype=np.float64)
                error = ctypes.create_string_buffer(2048)
                result = library.tron1_batch(
                    *addresses, batch, horizon, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    q0.ctypes.data_as(real), *(value.ctypes.data_as(real) for value in inputs),
                    output.ctypes.data_as(real), output.size, error, len(error))
                if result:
                    pipe.send({"ok": False, "error": error.value.decode(errors="replace"),
                               "error_kind": "native_c_return", "native_return_code": int(result)})
                else:
                    pipe.send({"ok": True, "output": output})
            except Exception as error:
                pipe.send({"ok": False, "error": f"{type(error).__name__}: {error}",
                           "error_kind": "child_exception", "child_exception_type": type(error).__name__})
    except EOFError:
        return
    except Exception as error:
        try:
            pipe.send({"ok": False, "error": f"{type(error).__name__}: {error}",
                       "error_kind": "child_outer_exception", "child_exception_type": type(error).__name__})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        pipe.close()


def _unpack(output):
    pre = output[..., :PRE_WIDTH]
    sub = output[..., PRE_WIDTH:PRE_WIDTH + 4*STEP_WIDTH].reshape(*output.shape[:2], 4, STEP_WIDTH)
    result = {"pre_qpos": pre[..., :15], "pre_qvel": pre[..., 15:29],
              "qpos": sub[..., :15], "qvel": sub[..., 15:29], "torques": sub[..., 29:37],
              "interval_time_s": sub[..., -2], "contact_count": sub[..., -1].astype(np.int32),
              "contact_force_history_w": output[..., -81:].reshape(*output.shape[:2], 3, 9, 3)}
    for prefix, value, offset in (("pre_", pre, 29), ("", sub, 37)):
        for name, width in (("body_pos_w", 3), ("body_quat_w", 4),
                            ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
            result[prefix + name] = value[..., offset:offset + 9*width].reshape(*value.shape[:-1], 9, width)
            offset += 9*width
        if not prefix:
            result["body_net_forces_w"] = value[..., offset:offset + 27].reshape(*value.shape[:-1], 9, 3)
    return result


class NativeBatchTransition:
    """Explicit optional backend with process-isolated, serialized single-worker calls.

    Inputs are independent initial states (B,15), (B,14), history (B,3,9,3)
    and normalized action plans (B,H,8). No objective/reference phase is advanced
    here. Output substep arrays have (B,H,4,...) and pre-stage arrays (B,H,...).
    All interval times restart at zero; returned contact history is (B,H,3,9,3).
    """

    def __init__(self, model, q0, contract, library, *, timeout_s=30., max_output_bytes=512*1024**2):
        self._closed = True
        self._lock = threading.Lock()
        self._installation = _installation()
        ids = _validate_model(model, contract)
        q0 = np.array(q0, dtype=np.float64, order="C", copy=True)
        if q0.shape != (8,) or not np.isfinite(q0).all():
            raise ValueError("Expected finite 8-joint reference offset q0")
        if not np.isfinite(timeout_s) or timeout_s <= 0 or max_output_bytes < STAGE_WIDTH*8:
            raise ValueError("Invalid native worker timeout or output-memory budget")
        path, metadata, _ = _validated_library(library)
        blob = pickle.dumps(model, protocol=5)
        self.metadata = {**copy.deepcopy(metadata), "model_snapshot_sha256": hashlib.sha256(blob).hexdigest(),
                         "model_snapshot_owned_by_child": True, "fatal_errors_process_isolated": True,
                         "body_names": list(BODY_NAMES), "joint_names": list(JOINT_NAMES),
                         "physics_dt_s": .005, "control_dt_s": .02, "ipc_included_in_latency": True,
                         "measurement_semantics": "post-mj_forward link origins/quaternions and original link-COM world velocities",
                         "reset_semantics": "cold mj_resetData at EACH action interval; 4 torque/mj_step/mj_forward/contact samples"}
        self.timeout_s, self.max_output_bytes = float(timeout_s), int(max_output_bytes)
        self._workdir = tempfile.TemporaryDirectory(prefix="tron1_native_batch_")
        context = mp.get_context("spawn")
        self._pipe, child = context.Pipe()
        self._process = context.Process(target=_worker,
            args=(child, str(path), blob, ids, q0, self._workdir.name, self.max_output_bytes), daemon=True)
        self._process.start()
        child.close()
        self._closed = False
        try:
            ready = self._receive()
            self.metadata["worker_pid"] = ready["worker_pid"]
        except BaseException:
            self.close()
            raise

    @property
    def worker_alive(self):
        return not self._closed and self._process.is_alive()

    def _worker_status(self):
        """Best-effort nonblocking snapshot, never a lifecycle/restart action."""
        status = {"alive": None, "closed": bool(getattr(self, "_closed", True)),
                  "pid": None, "exitcode": None}
        process = getattr(self, "_process", None)
        if process is not None:
            for key, read in (("alive", process.is_alive),
                              ("pid", lambda: process.pid),
                              ("exitcode", lambda: process.exitcode)):
                try:
                    status[key] = read()
                except Exception:
                    pass
        return status

    def _receive(self):
        try:
            if not self._pipe.poll(self.timeout_s):
                before = self._worker_status()
                self._stop()
                raise NativeWorkerError("Native worker timeout; isolated worker stopped, no automatic restart",
                    diagnostics={"event": "receive_timeout", "timeout_s": self.timeout_s,
                                 "before_stop": before, "after_stop": self._worker_status(),
                                 "parent_stop_initiated": True})
            response = self._pipe.recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            self._process.join(timeout=1.)
            code = self._process.exitcode
            log = Path(self._workdir.name) / "MUJOCO_LOG.TXT"
            details = log.read_text(errors="replace")[-2000:] if log.is_file() else ""
            before = self._worker_status()
            self._stop()
            raise NativeWorkerError(f"Native worker exited (code={code}); {details}",
                diagnostics={"event": "receive_ipc_error", "ipc_exception_type": type(error).__name__,
                             "before_stop": before, "after_stop": self._worker_status(),
                             "parent_stop_initiated": True, "mujoco_log_tail": details}) from error
        if not response.get("ok"):
            raise NativeBatchError(response.get("error", "Unknown native worker failure"),
                diagnostics={"event": response.get("error_kind", "child_error_reply"),
                             "native_return_code": response.get("native_return_code"),
                             "child_exception_type": response.get("child_exception_type"),
                             "worker": self._worker_status(), "parent_stop_initiated": False})
        return response

    def _request(self, message):
        if self._closed or not self._process.is_alive():
            raise NativeWorkerError("Native worker is closed or terminal; no automatic restart",
                diagnostics={"event": "request_worker_unavailable", "worker": self._worker_status(),
                             "parent_stop_initiated": False})
        try:
            self._pipe.send(message)
        except (EOFError, BrokenPipeError, OSError) as error:
            before = self._worker_status()
            self._stop()
            raise NativeWorkerError("Native worker connection closed",
                diagnostics={"event": "send_ipc_error", "ipc_exception_type": type(error).__name__,
                             "before_stop": before, "after_stop": self._worker_status(),
                             "parent_stop_initiated": True}) from error
        return self._receive()

    def rollout(self, initial_qpos, initial_qvel, actions, *, initial_history):
        """No mutation of caller arrays; includes IPC, but no timing benchmark here."""
        qpos, qvel, history, controls = [np.array(value, dtype=np.float64, order="C", copy=True)
                                      for value in (initial_qpos, initial_qvel, initial_history, actions)]
        if controls.ndim != 3 or controls.shape[-1] != 8:
            raise ValueError("Normalized plans must have shape (batch,horizon,8)")
        batch, horizon = controls.shape[:2]
        if (not 1 <= batch <= 4096 or not 1 <= horizon <= 1024
                or qpos.shape != (batch, 15) or qvel.shape != (batch, 14)
                or history.shape != (batch, 3, 9, 3)):
            raise ValueError("Native initial-state/history/plan dimensions differ")
        if any(not np.isfinite(value).all() for value in (qpos, qvel, history, controls)):
            raise ValueError("Native inputs must all be finite")
        if np.max(np.abs(controls)) > 1. + 1e-9:
            raise ValueError("Native actions must remain normalized to [-1,1]")
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1., rtol=0, atol=1e-5):
            raise ValueError("Native initial root quaternions must be normalized wxyz")
        if batch*horizon*STAGE_WIDTH*8 > self.max_output_bytes:
            raise ValueError("Native batch output exceeds the explicit memory budget")
        with self._lock:
            response = self._request({"command": "rollout", "qpos": qpos, "qvel": qvel,
                                      "history": history, "actions": controls})
        return _unpack(response["output"])

    def _stop(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3.)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=3.)
        else:
            self._process.join(timeout=1.)
        self._pipe.close()

    def close(self):
        with self._lock:
            if not getattr(self, "_closed", True):
                try:
                    if self._process.is_alive():
                        self._request({"command": "close"})
                        self._process.join(timeout=3.)
                finally:
                    self._stop()
            if hasattr(self, "_workdir"):
                self._workdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
