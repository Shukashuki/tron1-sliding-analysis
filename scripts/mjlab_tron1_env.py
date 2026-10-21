"""Persistent TRON1 MuJoCo-Warp physics through the pinned mjlab Simulation.

This is a custom TRON1 task adapter, not mjlab's unmodified G1 task. Physics,
DC-motor integration and contact-force extraction run on the selected CUDA
device. The initial parity implementation deliberately returns NumPy states to
the existing verified task/observation/reward code; those host transfers are a
performance limitation, not a different or approximate physical predictor.

No fallback to CPU physics is permitted. Dependency imports are delayed so that
asset-free mathematical tests do not need a CUDA installation.
"""
from __future__ import annotations

import copy
import hashlib
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np

from nmpc_tracking_objective import (BODY_NAMES, JOINT_NAMES, quaternion_inverse,
                                     quaternion_rotate)


MJLAB_COMMIT = "4360096b49c1bcc478668c11ce7d458db747e49d"
MJLAB_VERSION = "1.4.0"
# Match v1.4 requires_model_fields(body_ipos, recompute=set_const). Without
# these derived-field expansions, one world's randomized COM could overwrite
# another world's shared inverse-weight constants during GPU recomputation.
EXPANDED_MODEL_FIELDS = ("geom_friction", "body_ipos", "body_subtreemass",
                         "dof_invweight0", "body_invweight0", "tendon_length0",
                         "tendon_invweight0")


def world_com_velocity_deltas(q, body_ipos, deltas):
    """Convert world COM linear/angular velocity increments to free-joint dqvel.

    MuJoCo free-joint linear velocity is measured at the link origin in world
    coordinates; its angular velocity is in the free-joint local frame.
    """
    q, body_ipos, deltas = (np.asarray(v, dtype=float) for v in (q, body_ipos, deltas))
    if (q.ndim != 2 or q.shape[1] != 4 or body_ipos.shape != (len(q), 3)
            or deltas.shape != (len(q), 6)
            or not all(np.isfinite(v).all() for v in (q, body_ipos, deltas))):
        raise ValueError("Expected finite batched quaternions, COM offsets and six-dimensional deltas")
    result = np.empty_like(deltas)
    result[:, :3] = deltas[:, :3]-np.cross(deltas[:, 3:], quaternion_rotate(q, body_ipos))
    result[:, 3:] = quaternion_rotate(quaternion_inverse(q), deltas[:, 3:])
    return result


def cvel_to_com_velocities(xipos, subtree_com, cvel):
    """World COM velocities from MuJoCo's subtree-COM-referenced spatial cvel."""
    xipos, subtree_com, cvel = (np.asarray(v) for v in (xipos, subtree_com, cvel))
    angular = cvel[..., :3]
    linear = cvel[..., 3:]-np.cross(angular, subtree_com-xipos)
    return linear, angular


def _simulation_config(model):
    """Copy model solver/integration options, rather than apply mjlab defaults."""
    import mujoco
    from mjlab.sim import MujocoCfg, SimulationCfg

    def enum_name(enum_type, value, prefix):
        return enum_type(int(value)).name.removeprefix(prefix).lower()

    opt = model.opt
    cfg = MujocoCfg(
        timestep=float(opt.timestep),
        integrator=enum_name(mujoco.mjtIntegrator, opt.integrator, "mjINT_"),
        impratio=float(opt.impratio),
        cone=enum_name(mujoco.mjtCone, opt.cone, "mjCONE_"),
        jacobian=enum_name(mujoco.mjtJacobian, opt.jacobian, "mjJAC_"),
        solver=enum_name(mujoco.mjtSolver, opt.solver, "mjSOL_"),
        iterations=int(opt.iterations), tolerance=float(opt.tolerance),
        ls_iterations=int(opt.ls_iterations), ls_tolerance=float(opt.ls_tolerance),
        ccd_iterations=int(opt.ccd_iterations), gravity=tuple(float(x) for x in opt.gravity))
    # Existing flags remain present on the passed model. Do not reinterpret or
    # drop any source XML enable/disable setting when copying options.
    return SimulationCfg(mujoco=cfg, nconmax=64, njmax=256)


class MjlabWarpBatch:
    """Continuous GPU plant implementing the verified PPO batch interface.

    The factory receives already configured per-environment host models, with
    startup DR sampled by the task. Only geom_friction/body_ipos may vary by
    world. These exact fields are expanded on the GPU and constants recomputed
    through mjlab after COM changes. Solver state persists across actions.
    """
    implementation = "mjlab_simulation_custom_tron1_task"

    def __init__(self, models, datas, q0s, contract, api, *, device="cuda:0"):
        import mujoco
        import mujoco_warp as mjwarp
        import torch
        import warp as wp
        from mjlab.managers.event_manager import RecomputeLevel
        from mjlab.sim import Simulation
        from mujoco_ppo_native import _model_indices

        if not str(device).startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("mjlab TRON1 adapter requires an available CUDA device; no CPU fallback")
        if mujoco.__version__ != "3.8.1":
            raise RuntimeError("This adapter is pinned and validated against MuJoCo 3.8.1")
        if not models or len(models) != len(datas):
            raise ValueError("Pass nonempty paired per-world host models and data")
        if contract.get("domain_parameters") is not None or contract.get("actuator_model") != "dc_motor":
            raise ValueError("Only the frozen nominal TRON1 DC motor contract is supported")
        self.models, self.datas = tuple(models), tuple(datas)
        self.num_envs, self.device = len(models), str(device)
        self.torch, self.wp, self.mjwarp = torch, wp, mjwarp
        self.contract, self.api = copy.deepcopy(contract), api
        self._closed = False
        self._failed = None
        model = models[0]
        packed = _model_indices(model, datas[0])
        self.qids, self.vids, self.aids, self.bids = np.split(packed, [8, 16, 24])
        self.base_id = int(self.bids[0])
        self.q0s = np.asarray(q0s, dtype=np.float32)
        if self.q0s.shape != (self.num_envs, 8) or not np.isfinite(self.q0s).all():
            raise ValueError("Each environment requires eight finite nominal/default joint offsets")
        # Reject unnoticed per-world physical differences rather than silently
        # simulate all worlds using environment zero's unexpanded parameters.
        invariant_fields = ("body_mass", "body_inertia", "body_iquat", "geom_priority",
                            "geom_solref", "geom_solimp", "geom_size", "geom_pos",
                            "dof_armature", "dof_damping", "dof_frictionloss")
        for candidate, data in zip(models, datas):
            _model_indices(candidate, data)
            for field in invariant_fields:
                if not np.array_equal(getattr(candidate, field), getattr(model, field)):
                    raise ValueError(f"Unsupported per-world difference in {field}")
        self.sim = Simulation(self.num_envs, _simulation_config(model),
                              model=copy.copy(model), device=self.device)
        driver_version = wp.context.runtime.driver_version
        if driver_version is None or driver_version < (12, 4):
            # mjlab disables graph capture on this driver already. Also disable
            # conditional graph nodes inside mjwarp explicitly where exposed.
            if hasattr(self.sim.wp_model.opt, "graph_conditional"):
                self.sim.wp_model.opt.graph_conditional = False
        self.stream = torch.cuda.ExternalStream(wp.get_stream(self.sim.wp_device).cuda_stream,
                                               device=self.device)
        with torch.cuda.stream(self.stream), wp.ScopedDevice(self.sim.wp_device):
            self.sim.expand_model_fields(EXPANDED_MODEL_FIELDS)
            self.sim.model.geom_friction[:] = torch.as_tensor(
                np.stack([m.geom_friction for m in models]), device=self.device, dtype=torch.float32)
            self.sim.model.body_ipos[:] = torch.as_tensor(
                np.stack([m.body_ipos for m in models]), device=self.device, dtype=torch.float32)
            self.sim.recompute_constants(RecomputeLevel.set_const)
            self._qids = torch.as_tensor(self.qids, device=self.device, dtype=torch.long)
            self._vids = torch.as_tensor(self.vids, device=self.device, dtype=torch.long)
            self._aids = torch.as_tensor(self.aids, device=self.device, dtype=torch.long)
            self._bids = torch.as_tensor(self.bids, device=self.device, dtype=torch.long)
            self._q0s = torch.as_tensor(self.q0s, device=self.device)
            self._effort = torch.as_tensor([contract["leg_torque_limit_nm"]]*6+
                                            [contract["wheel_torque_limit_nm"]]*2, device=self.device)
            self._speed = torch.as_tensor([contract["leg_motor_velocity_limit_rad_s"]]*6+
                                           [contract["wheel_motor_velocity_limit_rad_s"]]*2, device=self.device)
            self._stall = torch.as_tensor([contract["leg_saturation_effort_nm"]]*6+
                                           [contract["wheel_saturation_effort_nm"]]*2, device=self.device)
            capacity = int(self.sim.wp_data.contact.worldid.size)
            ids = torch.arange(capacity, dtype=torch.int32, device=self.device)
            self._contact_ids_tensor = ids
            self._contact_ids = wp.from_torch(ids, dtype=wp.int32)
            self._contact_forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.sim.wp_device)
            self._contact_force_tensor = wp.to_torch(self._contact_forces)
            self._geom_bodyid = torch.as_tensor(model.geom_bodyid, device=self.device, dtype=torch.long)
            body_to_tracked = np.full(model.nbody, -1, dtype=np.int64)
            body_to_tracked[self.bids] = np.arange(len(self.bids))
            self._body_to_tracked = torch.as_tensor(body_to_tracked, device=self.device)
            self.sim.forward()
        installed = {}
        for name in ("mujoco", "mujoco-warp", "warp-lang", "torch"):
            installed[name] = package_metadata.version(name)
        self.metadata = {"implementation": "mjlab_simulation_custom_tron1_task",
                         "mjlab_version": MJLAB_VERSION, "mjlab_source_commit": MJLAB_COMMIT,
                         "physics": "MuJoCo-Warp CUDA, not CPU MuJoCo or PhysX",
                         "device": self.device, "installed_versions": installed,
                         "precision": "MuJoCo-Warp float32",
                         "num_worlds": self.num_envs, "physics_dt_s": .005,
                         "decimation": 4, "state_continuity": "persistent between control intervals",
                         "expanded_model_fields": list(EXPANDED_MODEL_FIELDS),
                         "contact_forces": "mjwarp.contact_force world-frame, signed net sum by original body",
                         "host_task_adapter": True,
                         "performance_caveat": "NumPy task reward and observations require host/device state transfers",
                         "cuda_graphs": bool(self.sim.use_cuda_graph),
                         "overflow_checks": {"global_contact_capacity": int(self.sim.wp_data.naconmax),
                                             "per_world_constraint_capacity": int(self.sim.wp_data.njmax),
                                             "sparse_jacobian_nnz_capacity": int(self.sim.wp_data.njmax_nnz),
                                             "latched_fatal_failure": True},
                         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def _check_open(self):
        if self._closed:
            raise RuntimeError("mjlab batch is closed")
        if self._failed is not None:
            raise FloatingPointError("Latched mjlab physics failure: "+self._failed)

    def _measure_tensor(self):
        data = self.sim.data
        cvel = data.cvel[:, self._bids]
        com = data.subtree_com[:, self.base_id].unsqueeze(1)
        angular = cvel[..., :3]
        linear = cvel[..., 3:]-self.torch.cross(angular, com-data.xipos[:, self._bids], dim=-1)
        return {"qpos": data.qpos[:], "qvel": data.qvel[:], "time_s": data.time[:],
                "body_pos_w": data.xpos[:, self._bids], "body_quat_w": data.xquat[:, self._bids],
                "body_lin_vel_w": linear, "body_ang_vel_w": angular}

    @staticmethod
    def _numpy(values):
        return {key: tensor.detach().cpu().numpy().copy() for key, tensor in values.items()}

    def measure(self):
        self._check_open()
        with self.torch.cuda.stream(self.stream), self.wp.ScopedDevice(self.sim.wp_device):
            return self._numpy(self._measure_tensor())

    def reset(self, indices, qpos, qvel):
        self._check_open()
        indices = np.asarray(indices, dtype=np.int64)
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if (indices.ndim != 1 or qpos.shape != (len(indices), 15) or qvel.shape != (len(indices), 14)
                or np.any((indices < 0) | (indices >= self.num_envs))
                or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()):
            raise ValueError("Invalid selected-world reset state")
        with self.torch.cuda.stream(self.stream), self.wp.ScopedDevice(self.sim.wp_device):
            selected = self.torch.as_tensor(indices, device=self.device, dtype=self.torch.long)
            self.sim.reset(selected)
            self.sim.data.qpos[selected] = self.torch.as_tensor(qpos, device=self.device, dtype=self.torch.float32)
            self.sim.data.qvel[selected] = self.torch.as_tensor(qvel, device=self.device, dtype=self.torch.float32)
            self.sim.forward()

    def _motor_torques(self, actions):
        torch, contract = self.torch, self.contract
        q, v = self.sim.data.qpos[:, self._qids], self.sim.data.qvel[:, self._vids]
        torque = torch.empty_like(actions)
        torque[:, :6] = contract["leg_kp"]*(self._q0s[:, :6]+contract["leg_position_scale"]*actions[:, :6]-q[:, :6])-contract["leg_kd"]*v[:, :6]
        torque[:, 6:] = contract["wheel_torque_scale_nm"]*actions[:, 6:]
        # This is the Isaac DCMotor braking envelope, including beyond-no-load
        # velocities where BOTH admissible bounds can have the same sign.
        v_limit = self._speed*(1.+self._effort/self._stall)
        curve_velocity = torch.maximum(torch.minimum(v, v_limit), -v_limit)
        lower = torch.maximum(self._stall*(-1.-curve_velocity/self._speed), -self._effort)
        upper = torch.minimum(self._stall*(1.-curve_velocity/self._speed), self._effort)
        return torch.maximum(torch.minimum(torque, upper), lower)

    def _body_contact_forces(self):
        torch, data = self.torch, self.sim.data
        self._contact_forces.zero_()
        self.mjwarp.contact_force(self.sim.wp_model, self.sim.wp_data, self._contact_ids,
                                 True, self._contact_forces)
        active = self._contact_ids_tensor < data.nacon[0]
        world = data.contact.worldid[:][active].to(torch.long)
        geom = data.contact.geom[:][active].to(torch.long)
        forces = self._contact_force_tensor[active, :3]
        result = torch.zeros((self.num_envs*len(self.bids), 3), device=self.device)
        for side, sign in ((0, -1.), (1, 1.)):
            tracked = self._body_to_tracked[self._geom_bodyid[geom[:, side]]]
            selected = tracked >= 0
            result.index_add_(0, world[selected]*len(self.bids)+tracked[selected], sign*forces[selected])
        return result.reshape(self.num_envs, len(self.bids), 3)

    def _validate_physics_status(self):
        """MuJoCo-Warp prints on overflow; make it a fatal latched run error."""
        data, torch = self.sim.data, self.torch
        cap = self.sim.wp_data.naconmax
        if (bool((data.nacon[:] > cap).any().item())
                or bool((data.ncollision[:] > cap).any().item())
                or bool((data.nefc[:] > self.sim.wp_data.njmax).any().item())):
            self._failed = "contact/broadphase/constraint capacity exceeded; physical solve was truncated"
        if self.sim.wp_model.is_sparse:
            nefc = data.nefc[:]
            last = torch.clamp(nefc.to(torch.long)-1, min=0, max=self.sim.wp_data.njmax-1)
            row_ends = torch.gather(data.efc.J_rowadr[:]+data.efc.J_rownnz[:],
                                    1, last[:, None]).squeeze(1)
            if bool(((nefc > 0) & (row_ends > self.sim.wp_data.njmax_nnz)).any().item()):
                self._failed = "sparse constraint Jacobian nonzero capacity exceeded"
        for field in ("qpos", "qvel", "qacc", "qacc_warmstart"):
            if not bool(torch.isfinite(getattr(data, field)[:]).all().item()):
                self._failed = "nonfinite "+field
                break
        if self._failed is not None:
            raise FloatingPointError("Latched mjlab physics failure: "+self._failed)

    def step(self, actions):
        self._check_open()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 8) or not np.isfinite(actions).all():
            raise ValueError("Expected finite (num_worlds,8) actions")
        torch, wp = self.torch, self.wp
        with torch.cuda.stream(self.stream), wp.ScopedDevice(self.sim.wp_device):
            action = torch.as_tensor(np.clip(actions, -1., 1.), device=self.device, dtype=torch.float32)
            torques, contacts, times = [], [], []
            for _ in range(4):
                torque = self._motor_torques(action)
                self.sim.data.ctrl[:, self._aids] = torque
                self.sim.step()
                self._validate_physics_status()
                # MuJoCo step integrates qpos after evaluating derived fields.
                # Recompute at the post-step state as in the CPU parity oracle.
                self.sim.forward()
                self._validate_physics_status()
                torques.append(torque)
                contacts.append(self._body_contact_forces())
                times.append(self.sim.data.time[:].clone())
            measured = self._measure_tensor()
            measured.update(applied_torque=torch.stack(torques, dim=1),
                            body_net_forces_w=torch.stack(contacts, dim=1),
                            substep_time_s=torch.stack(times, dim=1))
            result = self._numpy(measured)
        if not all(np.isfinite(value).all() for value in result.values()):
            self._failed = "nonfinite physical state"
            raise FloatingPointError("Nonfinite MuJoCo-Warp physics state; not an ordinary task reset")
        return result

    def apply_root_velocity_deltas(self, indices, world_com_deltas):
        self._check_open()
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return
        with self.torch.cuda.stream(self.stream), self.wp.ScopedDevice(self.sim.wp_device):
            selected = self.torch.as_tensor(indices, device=self.device, dtype=self.torch.long)
            q = self.sim.data.qpos[selected, 3:7].detach().cpu().numpy()
            com = np.stack([self.models[index].body_ipos[self.base_id] for index in indices])
            changes = world_com_velocity_deltas(q, com, world_com_deltas)
            self.sim.data.qvel[selected, :6] += self.torch.as_tensor(changes, device=self.device,
                                                                  dtype=self.torch.float32)
            self.sim.forward()

    def close(self):
        self._closed = True
