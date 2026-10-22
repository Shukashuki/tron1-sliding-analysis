"""CPU MuJoCo implementation of the pinned TRON1 height-smooth PPO task.

The actor/critic layouts, adaptive reference-state initialization, held-command
reward clock and tilt/timeout reset boundaries follow the existing PhysX task.
Physics is continuous between actions: no MPC-style cold reset occurs here.
``common_mu`` is a deliberately restricted, representable DR recipe, NOT an
exact mapping of BeyondMimic's independent static/dynamic/restitution draws.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from classical_tracking import load_pinned_evaluator
from mujoco_ppo_native import ppo_clock_from_contract
from nmpc_batch_objective import BatchHeightSmoothTrackingObjective
from nmpc_tracking_objective import (BODY_NAMES, JOINT_NAMES, quaternion_inverse,
                                     quaternion_multiply, quaternion_rotate)


PUSH_BOUNDS = np.array([[-.5, .5], [-.5, .5], [-.2, .2],
                        [-.52, .52], [-.52, .52], [-.78, .78]])
BODY_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def rotation_matrix(quaternion):
    """Broadcast wxyz matrices, retaining row-major first-two-column order."""
    q = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)),
                    axis=-1).reshape((*q.shape[:-1], 3, 3))


def build_observations(state, reference, phases, q0s, previous_actions, frames):
    """Return the exact noise-free 51D actor and actor-first 120D critic."""
    p, q, v, w = (np.asarray(state[key]) for key in BODY_FIELDS)
    root_inv = quaternion_inverse(q[:, 0])
    phase = 2*np.pi*np.asarray(phases)/(frames-1)
    relative_q = quaternion_multiply(root_inv, reference["body_quat_w"][:, 0])
    actor = np.concatenate((reference["joint_pos"][:, :6], reference["joint_vel"][:, :6],
                            np.sin(phase)[:, None], np.cos(phase)[:, None],
                            quaternion_rotate(root_inv, reference["body_pos_w"][:, 0]-p[:, 0]),
                            rotation_matrix(relative_q)[..., :2].reshape(len(p), 6),
                            quaternion_rotate(root_inv, v[:, 0]),
                            quaternion_rotate(root_inv, w[:, 0]),
                            state["joint_pos"][:, :6]-np.asarray(q0s)[:, :6],
                            state["joint_vel"], previous_actions), axis=-1)
    relative_p = quaternion_rotate(root_inv[:, None], p-p[:, :1])
    nonwheel_q = quaternion_multiply(root_inv[:, None], q[:, :7])
    critic = np.concatenate((actor, relative_p.reshape(len(p), 27),
                             rotation_matrix(nonwheel_q)[..., :2].reshape(len(p), 42)), axis=-1)
    if actor.shape != (len(p), 51) or critic.shape != (len(p), 120):
        raise AssertionError("TRON1 observation layout drifted")
    if not np.isfinite(actor).all() or not np.isfinite(critic).all():
        raise FloatingPointError("Nonfinite PPO observation")
    return {"policy": actor.astype(np.float32), "critic": critic.astype(np.float32)}


class AdaptivePhaseSampler:
    """Pinned MotionCommand's failure EMA with TRON1's .2s bins /20% frame0."""
    def __init__(self, frames, rng, *, dt=.02, bin_seconds=.2, start_fraction=.2):
        self.frames, self.rng = int(frames), rng
        self.bin_count = max(2, math.ceil(frames*dt/bin_seconds))
        self.failed = np.zeros(self.bin_count)
        self.current_failed = np.zeros(self.bin_count)
        self.start_fraction = float(start_fraction)
        self.alpha, self.uniform_ratio = .001, .1

    def probabilities(self):
        # The pinned adaptive_kernel_size is one; no smoothing is implied.
        values = self.failed+self.uniform_ratio/self.bin_count
        return values/values.sum()

    def sample(self, count, *, old_phases=None, terminated=None, deterministic=False):
        if old_phases is not None and terminated is not None and np.any(terminated):
            bins = np.minimum(np.asarray(old_phases)*self.bin_count//self.frames, self.bin_count-1)
            self.current_failed[:] = np.bincount(bins[np.asarray(terminated)], minlength=self.bin_count)
        if deterministic:
            return np.zeros(count, dtype=np.int64)
        bins = self.rng.choice(self.bin_count, size=count, p=self.probabilities())
        phases = ((bins+self.rng.random(count))/self.bin_count*(self.frames-1)).astype(np.int64)
        phases[self.rng.random(count) < self.start_fraction] = 0
        return phases

    def update(self):
        self.failed[:] = self.alpha*self.current_failed+(1-self.alpha)*self.failed
        self.current_failed.fill(0.)

    def state_dict(self):
        return {"failed": self.failed.copy(), "current_failed": self.current_failed.copy(),
                "bin_count": self.bin_count, "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state["bin_count"] != self.bin_count:
            raise ValueError("Sampler bin count mismatch")
        for key, target in (("failed", self.failed), ("current_failed", self.current_failed)):
            values = np.asarray(state[key], dtype=float)
            if values.shape != target.shape or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid sampler failure statistics")
            target[:] = values
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class _PythonBatch:
    """Explicit slow verification backend; production training uses native."""
    def __init__(self, models, datas, q0s, contract, api):
        self.models, self.datas, self.q0s, self.contract, self.api = models, datas, q0s, contract, api
        self.physics_dt, self.decimation, self.step_dt = ppo_clock_from_contract(contract)
        self.qids = np.array([models[0].joint(name).qposadr[0] for name in JOINT_NAMES])
        self.vids = np.array([models[0].joint(name).dofadr[0] for name in JOINT_NAMES])
        self.aids = np.array([models[0].actuator(name).id for name in JOINT_NAMES])
        self.bids = np.array([models[0].body(name).id for name in BODY_NAMES])

    def reset(self, indices, qpos, qvel):
        for index, p, v in zip(indices, qpos, qvel):
            model, data = self.models[index], self.datas[index]
            mujoco.mj_resetData(model, data)
            data.qpos[:], data.qvel[:] = p, v
            mujoco.mj_forward(model, data)

    def measure(self):
        values = {key: [] for key in (*BODY_FIELDS, "qpos", "qvel", "time_s")}
        for model, data in zip(self.models, self.datas):
            values["qpos"].append(data.qpos.copy())
            values["qvel"].append(data.qvel.copy())
            values["time_s"].append(data.time)
            values["body_pos_w"].append(data.xpos[self.bids].copy())
            values["body_quat_w"].append(data.xquat[self.bids].copy())
            linear, angular = [], []
            for body in self.bids:
                jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
                mujoco.mj_jacBodyCom(model, data, jp, jr, int(body))
                linear.append(jp@data.qvel)
                angular.append(jr@data.qvel)
            values["body_lin_vel_w"].append(linear)
            values["body_ang_vel_w"].append(angular)
        return {key: np.asarray(value) for key, value in values.items()}

    def step(self, actions):
        batch = len(self.models)
        forces = np.zeros((batch, self.decimation, 9, 3))
        torques = np.zeros((batch, self.decimation, 8))
        times = np.zeros((batch, self.decimation))
        for index, (model, data) in enumerate(zip(self.models, self.datas)):
            for step in range(self.decimation):
                torque = self.api.action_torques(actions[index], data.qpos[self.qids],
                                                  data.qvel[self.vids], self.q0s[index], self.contract)
                data.ctrl[self.aids] = torque
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                torques[index, step], times[index, step] = torque, data.time
                wrench = np.zeros(6)
                for contact_index, contact in enumerate(data.contact):
                    mujoco.mj_contactForce(model, data, contact_index, wrench)
                    force = contact.frame.reshape(3, 3).T@wrench[:3]
                    bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                    for tracked, body in enumerate(self.bids):
                        if bodies[0] == body:
                            forces[index, step, tracked] -= force
                        if bodies[1] == body:
                            forces[index, step, tracked] += force
        result = self.measure()
        result.update(body_net_forces_w=forces, applied_torque=torques, substep_time_s=times)
        return result

    def close(self):
        pass


class MujocoPPOEnv:
    """NumPy vector environment, with explicit auto-reset/terminal observations."""
    num_actions = 8
    step_dt = .02
    physics_dt = .005
    decimation = 4

    def __init__(self, model_path, motion_file, contract_path, task_profile, *, num_envs=512,
                 native_library=None, num_threads=8, seed=42, training=True,
                 dr_mode="common_mu", ground_friction=1., fixed_mu=.7, mu_range=(.3, 1.6),
                 batch_factory=None):
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if dr_mode not in ("none", "common_mu", "mjlab_v1p4") or (not training and dr_mode != "none"):
            raise ValueError("Use none for fixed evaluation or an explicitly named training recipe")
        self.num_envs, self.training, self.seed = int(num_envs), bool(training), int(seed)
        self.rng = np.random.default_rng(seed)
        self.api = load_pinned_evaluator()
        self.contract = self.api.read_contract(contract_path)
        self.physics_dt, self.decimation, self.step_dt = ppo_clock_from_contract(self.contract)
        if native_library is not None and self.decimation != 4:
            raise ValueError("Native C++ PPO batch ABI only supports 5 ms x4; use Python or Warp for 2.5 ms x8")
        self.profile = (json.loads(Path(task_profile).read_text())
                        if isinstance(task_profile, (str, Path)) else copy.deepcopy(task_profile))
        if self.contract.get("domain_parameters") is not None:
            raise ValueError("This task uses the nominal frozen motor contract")
        self.reference = self.api.MotionReference(motion_file, self.contract)
        if not set(BODY_NAMES).issubset(self.reference.body_names):
            raise ValueError("Reference must name the nine unmerged TRON1 bodies")
        self.reference_body_order = [self.reference.body_names.index(name) for name in BODY_NAMES]
        self.frames = self.reference.frames
        self.max_episode_length = math.ceil((self.frames/self.reference.fps+.04)/self.step_dt-1.e-10)
        self.sampler = AdaptivePhaseSampler(self.frames, self.rng)
        self.dr_mode = dr_mode
        self.ground_friction, self.fixed_mu = float(ground_friction), float(fixed_mu)
        self.mu_range = ((.3, 1.2) if dr_mode == "mjlab_v1p4"
                         else tuple(float(value) for value in mu_range))
        if (len(self.mu_range) != 2 or not 0 < self.mu_range[0] <= self.mu_range[1]
                or not all(np.isfinite([*self.mu_range, self.ground_friction, self.fixed_mu]))
                or min(self.ground_friction, self.fixed_mu) <= 0):
            raise ValueError("Invalid explicitly declared friction coefficients")
        model, self.removed_visual_meshes = self.api.load_model(model_path, no_visual_mesh=True)
        self.api.configure_model(model, self.contract)
        if not np.isclose(model.opt.timestep, self.physics_dt, rtol=0., atol=1.e-15):
            raise ValueError("Configured MuJoCo model timestep disagrees with the PPO contract")
        if "mujoco_contact_solref" in self.profile:
            colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
            model.geom_solref[colliding] = self.profile["mujoco_contact_solref"]
        self.qids = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
        self.vids = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES])
        self.bids = np.array([model.body(name).id for name in BODY_NAMES])
        self.base_id = int(self.bids[0])
        self.soft_limits = np.tile([-1.e9, 1.e9], (8, 1))
        for index, name in enumerate(JOINT_NAMES[:6]):
            lower, upper = model.jnt_range[model.joint(name).id]
            center, half = (lower+upper)/2, .95*(upper-lower)/2
            self.soft_limits[index] = [center-half, center+half]
        self.objective = BatchHeightSmoothTrackingObjective(self.profile, soft_joint_limits=self.soft_limits)
        self.models = [copy.copy(model) for _ in range(self.num_envs)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.q0s = np.broadcast_to(self.reference.q0, (self.num_envs, 8)).copy()
        self.encoder_bias = np.zeros((self.num_envs, 8))
        self.robot_mu = np.full(self.num_envs, self.fixed_mu)
        self.com_offsets = np.zeros((self.num_envs, 3))
        if dr_mode in ("common_mu", "mjlab_v1p4"):
            self.robot_mu[:] = self.rng.uniform(*self.mu_range, size=self.num_envs)
            if dr_mode == "mjlab_v1p4":
                self.encoder_bias[:] = self.rng.uniform(-.01, .01, (self.num_envs, 8))
                # mjlab observes q+b-q0 and commands q0+a-b. Storing q0-b
                # implements both without corrupting physical state or targets.
                self.q0s[:, :6] -= self.encoder_bias[:, :6]
            else:
                self.q0s[:, :6] += self.rng.uniform(-.01, .01, (self.num_envs, 6))
            self.com_offsets[:] = self.rng.uniform([-.025, -.05, -.05], [.025, .05, .05],
                                                  (self.num_envs, 3))
        for index, m in enumerate(self.models):
            if dr_mode == "mjlab_v1p4":
                wheels = np.isin(m.geom_bodyid, self.bids[-2:])
                wheels &= (m.geom_contype != 0) | (m.geom_conaffinity != 0)
                m.geom_friction[wheels, 0] = self.robot_mu[index]
                # Match mjlab's G1 foot-priority material rule. Otherwise the
                # original floor's max-combine would mask low friction draws.
                m.geom_priority[wheels] = 1
            else:
                m.geom_friction[:, 0] = self.robot_mu[index]
            if dr_mode in ("common_mu", "mjlab_v1p4"):
                m.geom_friction[m.geom_bodyid == 0, 0] = self.ground_friction
                m.body_ipos[self.base_id] += self.com_offsets[index]
                # Refresh MuJoCo model constants after changing original-link COM.
                mujoco.mj_setConst(m, self.datas[index])
        self.native_library = None if native_library is None else str(Path(native_library).resolve())
        if batch_factory is not None:
            if native_library is not None:
                raise ValueError("Select exactly one explicit physics backend")
            self.batch = batch_factory(self.models, self.datas, self.q0s, self.contract, self.api)
        elif native_library is None:
            self.batch = _PythonBatch(self.models, self.datas, self.q0s, self.contract, self.api)
        else:
            from mujoco_ppo_native import NativePPOBatch
            self.batch = NativePPOBatch(self.models, self.datas, self.q0s,
                                        native_library, num_threads=num_threads)
        self.phases = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_length_buf = np.zeros(self.num_envs, dtype=np.int64)
        self.previous_actions = np.zeros((self.num_envs, 8))
        self.previous_vz = np.zeros(self.num_envs)
        self.first_reward = np.ones(self.num_envs, dtype=bool)
        self.push_timer = self.rng.uniform(1., 3., self.num_envs)
        self.episode_rewards = np.zeros(self.num_envs)
        self.total_transitions = self.total_physical_steps = self.total_resets = self.total_pushes = 0
        self._closed = False
        self._paths = {"model": str(Path(model_path).resolve()), "motion": str(Path(motion_file).resolve()),
                       "contract": str(Path(contract_path).resolve())}
        self.reset()

    def _reference(self, indices=None):
        phases = self.phases if indices is None else self.phases[indices]
        archive, order = self.reference.archive, self.reference_body_order
        result = {key: archive[key][phases][:, order].astype(float) for key in BODY_FIELDS}
        result.update(joint_pos=archive["joint_pos"][phases][:, self.reference.order].astype(float),
                      joint_vel=archive["joint_vel"][phases][:, self.reference.order].astype(float))
        return result

    def _state(self, raw=None):
        raw = self.batch.measure() if raw is None else raw
        state = {key: np.asarray(raw[key], dtype=float) for key in BODY_FIELDS}
        state.update(joint_pos=np.asarray(raw["qpos"])[:, self.qids],
                     joint_vel=np.asarray(raw["qvel"])[:, self.vids])
        return state

    def observations(self, state=None):
        obs = build_observations(self._state() if state is None else state, self._reference(),
                                 self.phases, self.q0s, self.previous_actions, self.frames)
        if self.dr_mode == "mjlab_v1p4":
            # mjlab's privileged critic uses unbiased joint position while the
            # actor receives the encoder-biased measurement.
            obs["critic"][:, 29:35] -= self.encoder_bias[:, :6].astype(np.float32)
        return obs

    def _reset_indices(self, indices, *, failed=None):
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return
        self.phases[indices] = self.sampler.sample(len(indices), old_phases=self.phases[indices],
                                                  terminated=failed, deterministic=not self.training)
        reference = self._reference(indices)
        qpos, qvel = np.zeros((len(indices), 15)), np.zeros((len(indices), 14))
        qpos[:, :3], qpos[:, 3:7] = reference["body_pos_w"][:, 0], reference["body_quat_w"][:, 0]
        qpos[:, self.qids] = np.clip(reference["joint_pos"], self.soft_limits[:, 0], self.soft_limits[:, 1])
        qvel[:, self.vids] = reference["joint_vel"]
        # Root state API specifies COM world linear/angular velocity; MuJoCo's
        # free-joint qvel is origin-world-linear and local-angular velocity.
        q = qpos[:, 3:7]
        com = np.array([self.models[index].body_ipos[self.base_id] for index in indices])
        omega = reference["body_ang_vel_w"][:, 0]
        qvel[:, :3] = reference["body_lin_vel_w"][:, 0]-np.cross(omega, quaternion_rotate(q, com))
        qvel[:, 3:6] = quaternion_rotate(quaternion_inverse(q), omega)
        self.batch.reset(indices, qpos, qvel)
        self.previous_actions[indices] = 0.
        self.previous_vz[indices] = 0.
        self.first_reward[indices] = True
        self.episode_length_buf[indices] = 0
        self.episode_rewards[indices] = 0.
        # Pinned EventManager.reset resamples non-global interval timers.
        if self.dr_mode in ("common_mu", "mjlab_v1p4"):
            self.push_timer[indices] = self.rng.uniform(1., 3., len(indices))
        self.total_resets += len(indices)

    def reset(self, *, seed=None):
        if seed is not None and int(seed) != self.seed:
            raise ValueError("Reconstruct the environment to change startup DR seed")
        self._reset_indices(np.arange(self.num_envs))
        return self.observations()

    def _apply_pushes(self):
        if self.dr_mode not in ("common_mu", "mjlab_v1p4"):
            return
        self.push_timer -= self.step_dt
        indices = np.flatnonzero(self.push_timer < 1.e-6)
        if not len(indices):
            return
        # Like push_by_setting_velocity: ADD sampled world root COM velocity.
        # Do not clear warmstarts/contact state, reset physical time, or move qpos.
        offsets = self.rng.uniform(PUSH_BOUNDS[:, 0], PUSH_BOUNDS[:, 1], (len(indices), 6))
        if hasattr(self.batch, "apply_root_velocity_deltas"):
            if self.dr_mode == "mjlab_v1p4":
                q = self.batch.measure()["qpos"][indices, 3:7]
                com = np.array([self.models[i].body_ipos[self.base_id] for i in indices])
                offsets[:, :3] += np.cross(offsets[:, 3:], quaternion_rotate(q, com))
            self.batch.apply_root_velocity_deltas(indices, offsets)
            self.push_timer[indices] = self.rng.uniform(1., 3., len(indices))
            self.total_pushes += len(indices)
            return
        for row, index in enumerate(indices):
            model, data = self.models[index], self.datas[index]
            q = data.qpos[3:7]
            angular = offsets[row, 3:]
            data.qvel[:3] += offsets[row, :3]
            if self.dr_mode == "common_mu":
                data.qvel[:3] -= np.cross(angular, quaternion_rotate(q, model.body_ipos[self.base_id]))
            data.qvel[3:6] += quaternion_rotate(quaternion_inverse(q), angular)
            mujoco.mj_forward(model, data)
        self.push_timer[indices] = self.rng.uniform(1., 3., len(indices))
        self.total_pushes += len(indices)

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, 8) or not np.isfinite(actions).all():
            raise ValueError("Actions must be a finite (num_envs,8) array")
        # Actor tensors are float32 on both backends. Score exactly the action
        # that reaches native motor integration, even for a float64 test caller.
        actions = np.clip(actions, -1., 1.).astype(np.float32).astype(np.float64)
        pre, reference = self._state(), self._reference()
        pre_h = {key: value[:, None] for key, value in pre.items() if key in BODY_FIELDS}
        reference_h = {key: value[:, None] for key, value in reference.items() if key in BODY_FIELDS}
        held = self.objective.hold_reference(reference_h, pre_h)
        physical = self.batch.step(actions)
        if np.any(physical.get("failed", False)):
            raise FloatingPointError("Native PPO physics failure, not an ordinary task reset: "
                                     +str(np.asarray(physical.get("error_codes")).tolist()))
        state = self._state(physical)
        forces = np.asarray(physical.get("contact_substeps", physical.get("body_net_forces_w")))
        if forces.shape != (self.num_envs, self.decimation, 9, 3):
            raise AssertionError("Native physical contact force layout changed")
        post = {key: state[key][:, None] for key in BODY_FIELDS}
        post.update(joint_pos=state["joint_pos"][:, None],
                    body_net_forces_w=forces[:, None, -1],
                    contact_force_history_w=forces[:, None, -3:],
                    soft_joint_pos_limits=self.soft_limits,
                    soft_joint_limits_source="mujoco_original_unmerged_0p95_ranges")
        scored = self.objective.evaluate(post, held, actions[:, None], self.previous_actions[:, None],
                                         self.previous_vz[:, None], reset_samples=self.first_reward[:, None])
        rewards = self.step_dt*scored["total_reward"][:, 0]
        self.episode_length_buf += 1
        self.episode_rewards += rewards
        gravity = np.broadcast_to([0., 0., -1.], (self.num_envs, 3))
        ref_z = quaternion_rotate(quaternion_inverse(reference["body_quat_w"][:, 0]), gravity)[:, 2]
        actual_z = quaternion_rotate(quaternion_inverse(state["body_quat_w"][:, 0]), gravity)[:, 2]
        terminated = np.abs(ref_z-actual_z) > .6
        truncated = ((self.phases >= self.frames-1) |
                     (self.episode_length_buf >= self.max_episode_length))
        self.previous_actions[:] = actions
        self.previous_vz[:] = state["body_lin_vel_w"][:, 0, 2]
        self.first_reward[:] = False
        done = terminated | truncated
        terminal = self.observations(state)
        extras = {"terminal_observation": terminal, "terminal_mask": done.copy(),
                  "time_outs": truncated.copy(), "reference_frames_for_action": self.phases.copy(),
                  "raw_reward_terms": {name: values[:, 0].copy() for name, values in scored["raw_terms"].items()},
                  "log": {"reward/"+name: float(np.mean(values[:, 0])*self.step_dt)
                          for name, values in scored["weighted_terms"].items()}}
        if np.any(done):
            extras["episode"] = {"reward": self.episode_rewards[done].copy(),
                                  "length": self.episode_length_buf[done].copy()}
            extras["log"]["episode/reward"] = float(self.episode_rewards[done].mean())
            extras["log"]["episode/length"] = float(self.episode_length_buf[done].mean())
        self.phases[~done] = np.minimum(self.phases[~done]+1, self.frames-1)
        self._reset_indices(np.flatnonzero(done), failed=terminated[done])
        self.sampler.update()
        self._apply_pushes()
        self.total_transitions += self.num_envs
        self.total_physical_steps += self.num_envs*self.decimation
        extras["physical_counters"] = self.physical_counters()
        return self.observations(), rewards.astype(np.float32), terminated, truncated, extras

    def physical_counters(self):
        return {"environment_transitions": self.total_transitions,
                "physical_environment_substeps": self.total_physical_steps,
                "environment_resets": self.total_resets, "interval_pushes": self.total_pushes}

    def metadata(self):
        return {"physics_engine": "MuJoCo", "mujoco_version": mujoco.__version__,
                "implementation": getattr(self.batch, "implementation", None) or
                                  ("native_persistent_cpu" if self.native_library else "python_reference_slow"),
                "native_backend": copy.deepcopy(getattr(self.batch, "metadata", None)),
                "num_envs": self.num_envs, "seed": self.seed, "physics_dt_s": self.physics_dt,
                "decimation": self.decimation, "policy_dt_s": self.step_dt,
                "state_continuity": "No cold reset between control intervals; reset only on episode boundaries",
                "actor_dim": 51, "critic_dim": 120, "actor_observation_corruption": False,
                "critic_order": "policy_layout51 (unbiased leg positions for mjlab_v1p4), robot_body_pos_b27, nonwheel_body_ori_b42",
                "reference_state_initialization": {"adaptive_bin_seconds": .2, "adaptive_alpha": .001,
                    "adaptive_uniform_ratio": .1, "adaptive_kernel_size": 1, "full_clip_start_fraction": .2,
                    "pose_noise": False, "velocity_noise": False, "joint_reset_noise": False},
                "reward_contract": self.objective.contract(), "reward_scale": "sum(weighted_terms)*.02",
                "absolute_height_termination": False, "absolute_xy_termination": False,
                "tilt_termination": "abs(reference_projected_gravity_z - actual_projected_gravity_z) > .6",
                "motion_end_timeout": "held reference phase >= last frame, checked before phase advance",
                "terminal_observation": "postphysics, held command, before autoreset; diagnostic only, official RSL timeout bootstrap is unchanged",
                "dr": {"mode": self.dr_mode, "startup_once_not_resampled_on_reset": True,
                    "robot_single_sliding_mu_range": list(self.mu_range) if self.training and self.dr_mode != "none" else [self.fixed_mu]*2,
                    "ground_mu": self.ground_friction if self.dr_mode != "none" else self.fixed_mu,
                    "robot_mu_realized": self.robot_mu.tolist(),
                    "joint_default_offset_range_rad": [-.01, .01] if self.dr_mode == "common_mu" else [0., 0.],
                    "encoder_bias_range_rad": [-.01, .01] if self.dr_mode == "mjlab_v1p4" else [0., 0.],
                    "encoder_bias_semantics": "observation +b, position target -b; wheel angles not observed" if self.dr_mode == "mjlab_v1p4" else None,
                    "friction_scope": "two colliding wheel geoms, shared per environment" if self.dr_mode == "mjlab_v1p4" else "all robot geoms",
                    "friction_wheel_priority": 1 if self.dr_mode == "mjlab_v1p4" else 0,
                    "base_com_offset_range_m": [[-.025, -.05, -.05], [.025, .05, .05]] if self.dr_mode != "none" else [[0.]*3]*2,
                    "push_interval_s": [1., 3.] if self.dr_mode != "none" else None,
                    "push_velocity_frame": "root link world, additive" if self.dr_mode == "mjlab_v1p4" else "root COM world, additive",
                    "push_timer_correction": "resample on all episode resets, including function-based interval event" if self.dr_mode == "mjlab_v1p4" else None,
                    "static_dynamic_distinction": False,
                    "restitution": "MuJoCo existing solver-default contact law; no restitution-coefficient equivalence claimed",
                    "material_combine": "MuJoCo geom/pair priority/friction rules; NOT PhysX arithmetic average",
                    "beyondmimic_full_material_recipe_equivalent": False},
                "sources": {name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                            for name, path in self._paths.items()},
                "physical_counters": self.physical_counters()}

    def close(self):
        if not self._closed:
            close = getattr(self.batch, "close", None)
            if close is not None:
                close()
            self._closed = True
