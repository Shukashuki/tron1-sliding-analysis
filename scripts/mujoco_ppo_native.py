"""Explicit-build, persistent MuJoCo 3.8.1 CPU plant for isolated PPO training.

This is NOT the CEM backend: no control-interval resets are performed. Each row
retains its Python-owned model, data, solver state and clock until reset(). The
private ``_address`` bridge is pinned to the inspected official 3.8.1 wrappers;
raw pointers and caller-loaded native libraries are never accepted. Fatal native
errors terminate this isolated training process. No global error handlers or
callbacks are installed. Do not call from a controller process that must survive
a MuJoCo fatal error, or mutate model/data objects concurrently with this class.
Pinned wrapper implementation: https://github.com/google-deepmind/mujoco/blob/3.8.1/python/mujoco/structs.cc
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/ppo_batch.cpp"
VERSION, VERSION_NUMBER, OUTPUT_WIDTH = "3.8.1", 3008001, 287
JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
               "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")
BODY_NAMES = ("base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
              "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link")
CALLBACKS = ("control", "passive", "sensor", "contactfilter", "act_dyn", "act_gain", "act_bias")
ERROR_CODES = {0: "ok", 1: "preexisting_warning", 2: "physics_warning",
               3: "external_force", 4: "nonfinite_state", 5: "clock_discontinuity"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _installation():
    if sys.platform != "linux" or mujoco.__version__ != VERSION or mujoco.mj_version() != VERSION_NUMBER:
        raise RuntimeError("Native PPO requires Linux and exact MuJoCo 3.8.1")
    package = Path(mujoco.__file__).resolve().parent
    include, library = package / "include", package / "libmujoco.so.3.8.1"
    header = include / "mujoco/mujoco.h"
    if not header.is_file() or not library.is_file() or "#define mjVERSION_HEADER 3008001" not in header.read_text():
        raise RuntimeError("Pinned MuJoCo 3.8.1 headers/library unavailable")
    return include, library.resolve(), header


def build_ppo_native_library(output_dir, *, compiler="c++", openmp=True):
    """Explicit local build only; never overwrites a binary or build manifest.

    OpenMP is requested by default. If the compiler rejects only the OpenMP
    feature probe, build a serial library and record that fallback explicitly.
    Actual source compilation failures are never retried or suppressed.
    """
    include, installed, header = _installation()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = output / "libtron1_ppo_batch.so"
    manifest = library.with_suffix(".json")
    if library.exists() or manifest.exists():
        raise FileExistsError("Preserving existing native PPO build")
    use_openmp = False
    if openmp:
        probe = subprocess.run([compiler, "-std=c++17", "-fopenmp", "-x", "c++", "-",
                                "-fsyntax-only"], input="#include <omp.h>\nint main(){return omp_get_max_threads();}\n",
                               text=True, capture_output=True)
        use_openmp = probe.returncode == 0
    command = [compiler, "-std=c++17", "-O3", "-fPIC", "-shared"]
    if use_openmp:
        command.append("-fopenmp")
    command += [str(SOURCE), "-I"+str(include), str(installed),
                "-Wl,-rpath,"+str(installed.parent), "-ldl", "-o", str(library)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = {"schema_version": 1, "abi_version": 1, "mujoco_version": VERSION,
                "header_version": VERSION_NUMBER, "source_sha256": _sha(SOURCE),
                "mujoco_header_sha256": _sha(header), "mujoco_library_sha256": _sha(installed),
                "native_library_sha256": _sha(library), "compiler_command": command,
                "openmp_requested": bool(openmp), "openmp_enabled": use_openmp,
                "global_callbacks_installed": False, "persistent_integration": True}
    manifest.write_text(json.dumps(metadata, indent=2)+"\n")
    return library


def _validated_library(path):
    _, installed, header = _installation()
    path = Path(path).resolve(strict=True)
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = {"schema_version": 1, "abi_version": 1, "mujoco_version": VERSION,
                "header_version": VERSION_NUMBER, "source_sha256": _sha(SOURCE),
                "mujoco_header_sha256": _sha(header), "mujoco_library_sha256": _sha(installed),
                "native_library_sha256": _sha(path)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Native PPO build provenance/version/hash mismatch")
    return path, metadata, installed


def _no_callbacks(installed):
    library = ctypes.CDLL(str(installed))
    if any(ctypes.c_void_p.in_dll(library, "mjcb_"+name).value is not None for name in CALLBACKS):
        raise RuntimeError("Native PPO requires callback-free physics")


def _model_indices(model, data):
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("Pass live Python-owned MjModel/MjData objects, never raw pointers")
    if data.model is not model or data.threadpool:
        raise ValueError("Data must belong to its paired model without a nested MuJoCo threadpool")
    if (model.nq != 15 or model.nv != 14 or model.nu != 8 or
            any(getattr(model, name) for name in
                ("na", "nmocap", "nuserdata", "nplugin", "nhistory", "nflex", "neq", "ntendon")) or
            model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST or
            model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON or
            not np.isclose(model.opt.timestep, .005, rtol=0, atol=1e-15) or
            data.qpos.shape != (15,) or data.qvel.shape != (14,)):
        raise ValueError("Unsupported PPO model size, hidden state or clock/integrator")
    if (model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0
            or model.jnt_dofadr[0] != 0):
        raise ValueError("PPO requires the original floating-base coordinate layout")
    qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    vids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
    aids = [int(model.actuator(name).id) for name in JOINT_NAMES]
    bids = [int(model.body(name).id) for name in BODY_NAMES]
    for logical, actuator in enumerate(aids):
        joint = int(model.actuator_trnid[actuator, 0])
        effort = 80. if logical < 6 else 12.
        if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT or
                model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or
                model.jnt_qposadr[joint] != qids[logical] or model.jnt_dofadr[joint] != vids[logical] or
                model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE or
                model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED or
                model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_NONE or
                not np.array_equal(model.actuator_gear[actuator], [1.,0.,0.,0.,0.,0.]) or
                model.actuator_gainprm[actuator,0] != 1. or not model.actuator_ctrllimited[actuator] or
                not np.array_equal(model.actuator_ctrlrange[actuator], [-effort,effort])):
            raise ValueError("Native PPO requires unchanged torque-motor transmission")
    return np.array(qids+vids+aids+bids, dtype=np.int32)


class NativePPOBatch:
    """Persistent per-environment MuJoCo integration in this training process.

    Caller retains ordinary Python references; this object also retains every
    live model/data to prevent deallocation. Different environments must have
    distinct models and data (DR mutations cannot affect another environment).
    Errors are returned per row and remain latched until explicit reset; there
    is no silent reset. Actions are cast to float32, clipped, then simulated in
    float64, matching the normalized actuator input contract.
    """
    def __init__(self, models, datas, q0s, library, num_threads=8):
        self.models, self.datas = tuple(models), tuple(datas)
        self.batch = len(self.models)
        if not 1 <= self.batch <= 4096 or len(self.datas) != self.batch:
            raise ValueError("PPO models/data need equal nonempty batch size <=4096")
        if type(num_threads) is not int or not 1 <= num_threads <= 64:
            raise ValueError("num_threads must be an integer in [1,64]")
        if len({id(obj) for obj in self.models}) != self.batch or len({id(obj) for obj in self.datas}) != self.batch:
            raise ValueError("Each environment must own a distinct model and data")
        path, build, installed = _validated_library(library)
        _no_callbacks(installed)
        self.indices = np.stack([_model_indices(m,d) for m,d in zip(self.models,self.datas)])
        self.q0s = np.require(q0s, dtype=np.float64, requirements=["C","A","O"])
        if self.q0s.shape == (8,):
            self.q0s = np.repeat(self.q0s[None], self.batch, axis=0)
        if self.q0s.shape != (self.batch,8) or not np.isfinite(self.q0s).all():
            raise ValueError("q0s must be finite [B,8] or [8]")
        self._library = ctypes.CDLL(str(path))
        for name, expected in (("abi",1), ("header_version",VERSION_NUMBER),
                               ("runtime_version",VERSION_NUMBER), ("real_size",8),
                               ("output_width",OUTPUT_WIDTH), ("openmp",int(build["openmp_enabled"]))):
            function = getattr(self._library,"tron1_ppo_"+name)
            function.argtypes, function.restype = [], ctypes.c_int
            if function() != expected:
                raise RuntimeError("Native PPO ABI/version mismatch: "+name)
        linked = self._library.tron1_ppo_linked_mujoco_path
        linked.argtypes, linked.restype = [], ctypes.c_char_p
        if Path(os.fsdecode(linked())).resolve() != installed:
            raise RuntimeError("Native PPO linked a different MuJoCo library")
        width = self._library.tron1_ppo_warning_width
        width.argtypes, width.restype = [], ctypes.c_int
        self.warning_width = width()
        if self.warning_width != int(mujoco.mjtWarning.mjNWARNING):
            raise RuntimeError("Native PPO warning layout mismatch")
        pointer = ctypes.POINTER(ctypes.c_void_p)
        real, integer = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        self._step = self._library.tron1_ppo_batch
        self._step.argtypes = [pointer,pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int,integer,
                               real,real,real,ctypes.c_size_t,integer,integer,integer,integer,
                               ctypes.POINTER(ctypes.c_char),ctypes.c_size_t]
        self._step.restype = ctypes.c_int
        self._models = (ctypes.c_void_p*self.batch)(*(m._address for m in self.models))
        self._datas = (ctypes.c_void_p*self.batch)(*(d._address for d in self.datas))
        self._errors = np.zeros(self.batch,dtype=np.int32)
        self.num_threads = min(num_threads,self.batch) if build["openmp_enabled"] else 1
        self._lock = threading.Lock()
        for model,data in zip(self.models,self.datas):
            mujoco.mj_forward(model,data)
        self.metadata = {**build, "library":str(path), "batch_size":self.batch,
                         "requested_threads":num_threads, "effective_threads":self.num_threads,
                         "effective_threads_is_upper_bound":True,
                         "physics_dt_s":.005, "decimation":4, "control_dt_s":.02,
                         "integration_state_semantics":"persistent until explicit per-environment reset",
                         "float32_normalized_actions":True, "action_clip":[-1.,1.],
                         "global_callbacks_installed":False, "native_fatal_scope":"isolated training process",
                         "body_position_semantics":"original unmerged link origin",
                         "body_velocity_semantics":"original unmerged body COM, world axes",
                         "contact_semantics":"post-substep forward, net force on body, world axes",
                         "error_codes":ERROR_CODES, "automatic_reset":False}

    def _call(self, actions, stepping):
        _no_callbacks(_installation()[1])
        with np.errstate(over="ignore",invalid="ignore"):
            actions = np.asarray(actions,dtype=np.float32)
        if actions.shape != (self.batch,8) or not np.isfinite(actions).all():
            raise ValueError("Actions must be finite [B,8]")
        actions = np.ascontiguousarray(np.clip(actions,-1.,1.),dtype=np.float64)
        output = np.empty((self.batch,OUTPUT_WIDTH),dtype=np.float64)
        completed = np.zeros(self.batch,dtype=np.int32)
        counts = np.empty((self.batch,self.warning_width),dtype=np.int32)
        lastinfo = np.empty_like(counts)
        error = ctypes.create_string_buffer(2048)
        real, integer = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        status = self._step(self._models,self._datas,self.batch,self.num_threads,int(stepping),
                            self.indices.ctypes.data_as(integer),self.q0s.ctypes.data_as(real),
                            actions.ctypes.data_as(real),output.ctypes.data_as(real),output.size,
                            self._errors.ctypes.data_as(integer),completed.ctypes.data_as(integer),
                            counts.ctypes.data_as(integer),lastinfo.ctypes.data_as(integer),error,len(error))
        if status:
            raise RuntimeError("Native PPO batch rejected: "+error.value.decode(errors="replace"))
        result = {"qpos":output[:,:15].copy(), "qvel":output[:,15:29].copy(),
                  "body_pos_w":output[:,29:56].reshape(self.batch,9,3).copy(),
                  "body_quat_w":output[:,56:92].reshape(self.batch,9,4).copy(),
                  "body_lin_vel_w":output[:,92:119].reshape(self.batch,9,3).copy(),
                  "body_ang_vel_w":output[:,119:146].reshape(self.batch,9,3).copy(),
                  "contact_substeps":output[:,146:254].reshape(self.batch,4,9,3).copy(),
                  "applied_joint_torque_nm":output[:,254:286].reshape(self.batch,4,8).copy(),
                  "times":output[:,286].copy(), "warning_counts":counts,
                  "warning_lastinfo":lastinfo, "error_codes":self._errors.copy(),
                  "substeps_completed":completed}
        result["failed"] = result["error_codes"] != 0
        return result

    def step(self, actions):
        with self._lock:
            return self._call(actions,True)

    def measure(self):
        """Read snapshots without stepping, resetting or recomputing dynamics."""
        with self._lock:
            return self._call(np.zeros((self.batch,8),dtype=np.float32),False)

    def reset(self, indices, qpos, qvel):
        """Explicit selected reset to clock zero, preserving all other envs."""
        indices = np.asarray(indices)
        if indices.ndim == 1 and indices.size == 0:
            indices = indices.astype(np.int64)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.unique(indices).size != indices.size:
            raise ValueError("Reset indices must be unique integer vector")
        if np.any(indices < 0) or np.any(indices >= self.batch):
            raise ValueError("Reset index outside batch")
        qpos, qvel = np.asarray(qpos,dtype=np.float64), np.asarray(qvel,dtype=np.float64)
        if qpos.shape != (len(indices),15) or qvel.shape != (len(indices),14):
            raise ValueError("Reset state must have shapes [selected,15] and [selected,14]")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Reset state must be finite")
        if not np.allclose(np.linalg.norm(qpos[:,3:7],axis=1),1.,rtol=0,atol=1e-6):
            raise ValueError("Reset root quaternions must be unit length")
        with self._lock:
            _no_callbacks(_installation()[1])
            for row,index in enumerate(indices):
                model,data = self.models[index],self.datas[index]
                mujoco.mj_resetData(model,data)
                data.qpos[:],data.qvel[:] = qpos[row],qvel[row]
                mujoco.mj_forward(model,data)
                self._errors[index] = 0
            return self._call(np.zeros((self.batch,8),dtype=np.float32),False)
