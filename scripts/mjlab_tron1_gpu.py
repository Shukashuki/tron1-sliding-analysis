"""Device-resident TRON1 bridge for the pinned mjlab/MuJoCo-Warp plant.

The frozen :mod:`mjlab_tron1_env` adapter remains the parity oracle.  This
subclass changes data movement, not the dynamics: each action still performs
four ``step(); forward()`` pairs, with the same DC motor envelope, expanded COM
constants, solver options and post-forward contact-force readout.

All public tensor methods return CUDA tensors.  There is one intentional host
scalar read at the end of ``step_tensors`` to make physical failures fatal; no
state, observations, contacts, reset indices or push velocities return to CPU.
Imports of Torch, Warp and mjlab remain lazy for asset-free inspection/tests.
"""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path

from mjlab_tron1_env import MjlabWarpBatch


STATUS_MESSAGES = {
    1: "global contact capacity exceeded",
    2: "global broadphase capacity exceeded",
    4: "per-world constraint capacity exceeded",
    8: "sparse Jacobian nonzero capacity exceeded",
    16: "nonfinite qpos",
    32: "nonfinite qvel",
    64: "nonfinite qacc",
    128: "nonfinite qacc_warmstart",
    256: "nonfinite action/torque",
    512: "invalid reset/push state or selection",
    1024: "invalid contact world/geometry index or nonfinite contact force",
    2048: "nonfinite derived physical state/readout",
}


def quaternion_rotate_torch(q, vector):
    """Rotate vectors by unit WXYZ quaternions without a device readback."""
    import torch

    cross = torch.cross(q[..., 1:], vector, dim=-1)
    return vector + 2.0 * (q[..., :1] * cross + torch.cross(q[..., 1:], cross, dim=-1))


def world_com_velocity_deltas_torch(q, body_ipos, deltas):
    """World COM velocity increments to MuJoCo free-joint velocity increments."""
    import torch

    conjugate = torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
    linear = deltas[..., :3] - torch.cross(
        deltas[..., 3:], quaternion_rotate_torch(q, body_ipos), dim=-1)
    return torch.cat((linear, quaternion_rotate_torch(conjugate, deltas[..., 3:])), dim=-1)


@lru_cache(maxsize=1)
def _device_kernels():
    """Compile lazily; fixed capacities avoid Torch boolean-index allocation.

    Kernel annotations deliberately evaluate inside this factory so importing
    this module does not import Warp.  The same kernels can be unit-tested on
    Warp's CPU device without starting any GPU simulation.
    """
    import warp as wp

    @wp.kernel(enable_backward=False)
    def accumulate_contacts(
        nacon: wp.array(dtype=wp.int32),
        world_ids: wp.array(dtype=wp.int32),
        contact_geoms: wp.array(dtype=wp.vec2i),
        geom_body: wp.array(dtype=wp.int32),
        body_to_tracked: wp.array(dtype=wp.int32),
        contact_forces: wp.array(dtype=wp.spatial_vector),
        nworld: int,
        substep: int,
        result: wp.array3d(dtype=wp.vec3),
        flags: wp.array(dtype=wp.int32),
    ):
        contact = wp.tid()
        # Launch dimension is the allocation capacity, not a host-read nacon.
        if contact >= nacon[0]:
            return
        world = world_ids[contact]
        geoms = contact_geoms[contact]
        if (world < 0 or world >= nworld or geoms[0] < 0
                or geoms[1] < 0 or geoms[0] >= geom_body.shape[0]
                or geoms[1] >= geom_body.shape[0]):
            wp.atomic_or(flags, 0, 1024)
            return
        spatial = contact_forces[contact]
        force = wp.vec3(spatial[0], spatial[1], spatial[2])
        if not (wp.isfinite(force[0]) and wp.isfinite(force[1]) and wp.isfinite(force[2])):
            wp.atomic_or(flags, 0, 1024)
            return
        # MuJoCo's contact force acts negatively on geom1, positively on geom2.
        # Tracked self-contacts contribute to BOTH bodies, matching the oracle.
        for side in range(2):
            tracked = body_to_tracked[geom_body[geoms[side]]]
            if tracked >= 0:
                sign = float(1.0)
                if side == 0:
                    sign = -1.0
                wp.atomic_add(result, world, substep, tracked, sign * force)

    @wp.kernel(enable_backward=False)
    def check_physics(
        nacon: wp.array(dtype=wp.int32),
        ncollision: wp.array(dtype=wp.int32),
        nefc: wp.array(dtype=wp.int32),
        qpos: wp.array2d(dtype=float),
        qvel: wp.array2d(dtype=float),
        qacc: wp.array2d(dtype=float),
        warmstart: wp.array2d(dtype=float),
        row_address: wp.array2d(dtype=wp.int32),
        row_nnz: wp.array2d(dtype=wp.int32),
        contact_capacity: int,
        constraint_capacity: int,
        nnz_capacity: int,
        sparse: bool,
        flags: wp.array(dtype=wp.int32),
    ):
        world = wp.tid()
        local = int(0)
        if world == 0:
            if nacon[0] > contact_capacity:
                local = local | 1
            if ncollision[0] > contact_capacity:
                local = local | 2
        count = nefc[world]
        if count > constraint_capacity:
            local = local | 4
        if sparse and count > 0:
            last = wp.min(count - 1, constraint_capacity - 1)
            if row_address[world, last] + row_nnz[world, last] > nnz_capacity:
                local = local | 8
        for column in range(qpos.shape[1]):
            if not wp.isfinite(qpos[world, column]):
                local = local | 16
        for column in range(qvel.shape[1]):
            if not wp.isfinite(qvel[world, column]):
                local = local | 32
            if not wp.isfinite(qacc[world, column]):
                local = local | 64
            if not wp.isfinite(warmstart[world, column]):
                local = local | 128
        if local != 0:
            wp.atomic_or(flags, 0, local)

    return accumulate_contacts, check_physics


class GpuMjlabBatch(MjlabWarpBatch):
    """Persistent CUDA plant with stream-safe, zero-copy tensor readout.

    Startup is intentionally inherited from the frozen adapter: host model
    validation and initial DR uploads are not in the training hot path.  Tensor
    results alias persistent simulation/readout buffers and must be cloned by a
    consumer that needs to retain them across the next plant call.

    ``reset_tensors`` accepts either a fixed ``(B,)`` boolean mask plus full
    ``(B,15)/(B,14)`` states, or unique GPU indices plus compact selected states.
    ``apply_link_velocity_deltas`` accepts a fixed mask and WORLD-LINK velocity
    deltas.  Its caller performs any COM-to-link conversion with the randomized
    COM before this method converts only angular velocity to the free-joint's
    local coordinates.
    """

    implementation = "mjlab_simulation_custom_tron1_gpu_task"

    def __init__(self, models, datas, q0s, contract, api, *, device="cuda:0"):
        super().__init__(models, datas, q0s, contract, api, device=device)
        torch, wp = self.torch, self.wp
        self._wp_stream = wp.get_stream(self.sim.wp_device)
        self._event_in = torch.cuda.Event()
        self._event_out = torch.cuda.Event()
        self._contact_kernel, self._health_kernel = _device_kernels()
        with self._physics_scope():
            self._error_flags = torch.zeros(1, dtype=torch.int32, device=self.device)
            self._error_flags_wp = wp.from_torch(self._error_flags, dtype=wp.int32)
            self._contacts = torch.zeros((self.num_envs, self.decimation, len(self.bids), 3),
                                         device=self.device, dtype=torch.float32)
            self._contacts_wp = wp.from_torch(self._contacts, dtype=wp.vec3)
            self._torques = torch.empty((self.num_envs, self.decimation, 8), device=self.device)
            self._times = torch.empty((self.num_envs, self.decimation), device=self.device)
            self._reset_mask_tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._reset_mask_wp = wp.from_torch(self._reset_mask_tensor, dtype=wp.bool)
            self._geom_body_i32 = self._geom_bodyid.to(torch.int32)
            self._body_to_tracked_i32 = self._body_to_tracked.to(torch.int32)
            self._geom_body_wp = wp.from_torch(self._geom_body_i32, dtype=wp.int32)
            self._body_to_tracked_wp = wp.from_torch(self._body_to_tracked_i32, dtype=wp.int32)
            if self.sim.wp_model.is_sparse:
                self._row_address_wp = self.sim.wp_data.efc.J_rowadr
                self._row_nnz_wp = self.sim.wp_data.efc.J_rownnz
            else:
                # Empty/special dense storage differs between mjwarp versions.
                # This fixed dummy is never dereferenced when sparse=False.
                self._dense_dummy = wp.zeros((self.num_envs, 1), dtype=wp.int32,
                                              device=self.sim.wp_device)
                self._row_address_wp = self._row_nnz_wp = self._dense_dummy
        self.metadata.update({
            "implementation": self.implementation,
            "host_task_adapter": False,
            "performance_caveat": "CUDA Graph availability remains determined by the pinned mjlab/driver",
            "device_resident_tensor_api": True,
            "state_cpu_readbacks_per_control_interval": 0,
            "fatal_status_host_scalars_per_control_interval": 1,
            "contact_reduction": "fixed-capacity Warp kernel, no Torch boolean indexing",
            "stream_handoff": "bidirectional CUDA events; no host stream synchronization",
            "physics_semantics": f"unchanged step+full-forward pairing, {self.decimation} pairs per control interval",
            "gpu_bridge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })

    @contextmanager
    def _physics_scope(self):
        """Order policy→physics→policy streams using GPU-side event waits."""
        torch, wp = self.torch, self.wp
        producer = torch.cuda.current_stream(self.device)
        distinct = producer.cuda_stream != self.stream.cuda_stream
        if distinct:
            self._event_in.record(producer)
            self.stream.wait_event(self._event_in)
        with torch.cuda.stream(self.stream), wp.ScopedDevice(self.sim.wp_device), wp.ScopedStream(self._wp_stream):
            try:
                yield
            finally:
                if distinct:
                    self._event_out.record(self.stream)
                    producer.wait_event(self._event_out)

    def _require_tensor(self, value, shape, name, *, floating=True):
        torch = self.torch
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must be a device tensor of shape {tuple(shape)}")
        if value.device != torch.device(self.device):
            raise ValueError(f"{name} must already be on {self.device}; implicit CPU transfers are forbidden")
        if floating and value.dtype != torch.float32:
            raise ValueError(f"{name} must use float32, matching the physical model")
        return value

    def _flag_nonfinite(self, value, bit):
        self._error_flags.bitwise_or_((~self.torch.isfinite(value).all()).to(self.torch.int32) * bit)

    def _record_physics_status(self):
        data = self.sim.wp_data
        self.wp.launch(
            self._health_kernel, dim=self.num_envs,
            inputs=[data.nacon, data.ncollision, data.nefc, data.qpos, data.qvel,
                    data.qacc, data.qacc_warmstart, self._row_address_wp, self._row_nnz_wp,
                    int(data.naconmax), int(data.njmax), int(data.njmax_nnz),
                    bool(self.sim.wp_model.is_sparse), self._error_flags_wp],
            device=self.sim.wp_device)

    def _raise_if_status_failed(self):
        # The ONLY device→host scalar read in a successful control interval.
        # Device bits are never cleared by ordinary episode resets.
        code = int(self._error_flags.item())
        if code:
            self._failed = "; ".join(message for bit, message in STATUS_MESSAGES.items() if code & bit)
            raise FloatingPointError("Latched mjlab physics failure: " + self._failed)

    def _body_contact_forces_into(self, substep):
        self._contact_forces.zero_()
        self.mjwarp.contact_force(self.sim.wp_model, self.sim.wp_data,
                                 self._contact_ids, True, self._contact_forces)
        self.wp.launch(
            self._contact_kernel, dim=int(self._contact_ids.size),
            inputs=[self.sim.wp_data.nacon, self.sim.wp_data.contact.worldid,
                    self.sim.wp_data.contact.geom, self._geom_body_wp,
                    self._body_to_tracked_wp, self._contact_forces,
                    self.num_envs, substep, self._contacts_wp, self._error_flags_wp],
            device=self.sim.wp_device)

    def measure_tensors(self):
        self._check_open()
        with self._physics_scope():
            return self._measure_tensor()

    def step_tensors(self, actions):
        self._check_open()
        self._require_tensor(actions, (self.num_envs, 8), "actions")
        torch = self.torch
        with self._physics_scope():
            self._flag_nonfinite(actions, 256)
            # Valid inputs are unchanged; invalid inputs remain fatal but are
            # sanitized to avoid launching a solver with NaN actuator torques.
            action = torch.nan_to_num(actions).clamp(-1.0, 1.0)
            self._contacts.zero_()
            for substep in range(self.decimation):
                torque = self._motor_torques(action)
                self._flag_nonfinite(torque, 256)
                self.sim.data.ctrl[:, self._aids] = torque
                self.sim.step()
                self._record_physics_status()
                self.sim.forward()
                self._record_physics_status()
                self._torques[:, substep].copy_(torque)
                self._body_contact_forces_into(substep)
                self._times[:, substep].copy_(self.sim.data.time[:])
            measured = self._measure_tensor()
            measured.update(applied_torque=self._torques,
                            body_net_forces_w=self._contacts,
                            substep_time_s=self._times)
            for value in measured.values():
                self._flag_nonfinite(value, 2048)
            self._raise_if_status_failed()
            return measured

    def reset_tensors(self, selection, qpos, qvel, *, forward=True):
        """Masked GPU reset; state finiteness is latched, not host-read here.

        ``forward=False`` permits the task to combine reset+push writes before
        one explicit ``forward_tensors``.  A mask of all False is valid and
        does not reset any world's clock or warm-start state.
        """
        self._check_open()
        torch = self.torch
        if not isinstance(selection, torch.Tensor) or selection.device != torch.device(self.device):
            raise ValueError("Reset selection must already be a GPU mask or GPU index tensor")
        if selection.ndim != 1:
            raise ValueError("Reset selection must be one-dimensional")
        is_mask = selection.dtype == torch.bool
        count = self.num_envs if is_mask else selection.numel()
        if is_mask and count != selection.numel():
            raise ValueError("Reset mask must have one entry per world")
        if not is_mask and selection.dtype not in (torch.int32, torch.int64):
            raise ValueError("Reset indices must be integer tensors")
        self._require_tensor(qpos, (count, 15), "qpos")
        self._require_tensor(qvel, (count, 14), "qvel")
        with self._physics_scope():
            self._flag_nonfinite(qpos, 512)
            self._flag_nonfinite(qvel, 512)
            if is_mask:
                self._reset_mask_tensor.copy_(selection)
                reset_q, reset_v = qpos, qvel
            else:
                indices = selection.to(torch.long)
                invalid = ((indices < 0) | (indices >= self.num_envs)).any()
                self._error_flags.bitwise_or_(invalid.to(torch.int32) * 512)
                safe_indices = indices.clamp(0, self.num_envs - 1)
                multiplicity = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                multiplicity.scatter_add_(0, safe_indices, torch.ones_like(safe_indices, dtype=torch.int32))
                self._error_flags.bitwise_or_((multiplicity > 1).any().to(torch.int32) * 512)
                self._reset_mask_tensor.copy_(multiplicity != 0)
                reset_q = self.sim.data.qpos[:].clone()
                reset_v = self.sim.data.qvel[:].clone()
                reset_q.index_copy_(0, safe_indices, qpos)
                reset_v.index_copy_(0, safe_indices, qvel)
            self.mjwarp.reset_data(self.sim.wp_model, self.sim.wp_data, reset=self._reset_mask_wp)
            # Fixed-size where avoids boolean index/nonzero's hidden CUDA sync.
            mask = self._reset_mask_tensor[:, None]
            self.sim.data.qpos[:] = torch.where(mask, reset_q, self.sim.data.qpos[:])
            self.sim.data.qvel[:] = torch.where(mask, reset_v, self.sim.data.qvel[:])
            if forward:
                self.sim.forward()
                self._record_physics_status()

    def apply_link_velocity_deltas(self, mask, deltas, *, forward=True):
        """Add WORLD-LINK velocity increments (angular free-joint part is local).

        ``deltas[:,:3]`` is the world linear increment at the link origin, not
        at its COM.  Use ``world_com_velocity_deltas_torch`` in a caller that
        starts from COM increments; do not subtract the COM cross-term twice.
        """
        self._check_open()
        self._require_tensor(mask, (self.num_envs,), "push mask", floating=False)
        if mask.dtype != self.torch.bool:
            raise ValueError("Push mask must be boolean")
        self._require_tensor(deltas, (self.num_envs, 6), "push deltas")
        with self._physics_scope():
            self._flag_nonfinite(deltas, 512)
            q = self.sim.data.qpos[:, 3:7]
            conjugate = self.torch.cat((q[:, :1], -q[:, 1:]), dim=-1)
            changes = self.torch.cat((deltas[:, :3],
                                      quaternion_rotate_torch(conjugate, deltas[:, 3:])), dim=-1)
            self.sim.data.qvel[:, :6] += self.torch.where(mask[:, None], changes, self.torch.zeros_like(changes))
            if forward:
                self.sim.forward()
                self._record_physics_status()

    def forward_tensors(self):
        """Explicit full forward after combined reset/push writes, without D2H."""
        self._check_open()
        with self._physics_scope():
            self.sim.forward()
            self._record_physics_status()
