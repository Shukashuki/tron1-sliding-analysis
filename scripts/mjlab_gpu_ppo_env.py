"""GPU-resident TRON1 task; NumPy is used only during asset/startup loading.

The frozen host implementation remains the oracle. Physics clocks, motor law,
four post-step forwards, reward clock and termination rules are unchanged.
Only tiny event flags cross to the host in the task loop; bulk state, rewards,
actions, observations, resets and pushes stay on the selected CUDA device.
"""
from __future__ import annotations

import copy
import math

import torch

from tron1_torch_task import (TorchHeightSmoothObjective, build_observations,
                              quaternion_inverse, quaternion_rotate)

BODY_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


class TensorPhaseSampler:
    """The same failure-EMA/20% frame-zero distribution with a private RNG."""
    def __init__(self, frames, num_envs, generator, device, dtype=torch.float32):
        self.frames, self.num_envs = int(frames), int(num_envs)
        self.bin_count = max(2, math.ceil(frames*.02/.2))
        self.generator, self.device, self.dtype = generator, device, dtype
        self.failed = torch.zeros(self.bin_count, device=device, dtype=dtype)
        self.current_failed = torch.zeros_like(self.failed)
        self.alpha, self.uniform_ratio, self.start_fraction = .001, .1, .2

    def probabilities(self):
        values = self.failed+self.uniform_ratio/self.bin_count
        return values/values.sum()

    def record_failures(self, phases, terminated):
        bins = torch.clamp(phases*self.bin_count//self.frames, max=self.bin_count-1)
        self.current_failed.zero_()
        self.current_failed.scatter_add_(0, bins, terminated.to(self.dtype))

    def sample(self, *, deterministic=False):
        if deterministic:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        bins = torch.multinomial(self.probabilities(), self.num_envs, replacement=True,
                                 generator=self.generator)
        u = torch.rand((self.num_envs, 2), generator=self.generator,
                       device=self.device, dtype=self.dtype)
        phases = ((bins+u[:, 0])/self.bin_count*(self.frames-1)).to(torch.long)
        return torch.where(u[:, 1] < self.start_fraction, torch.zeros_like(phases), phases)

    def update(self):
        self.failed.mul_(1-self.alpha).add_(self.current_failed, alpha=self.alpha)
        self.current_failed.zero_()


class GpuTron1PPOEnv:
    num_actions = 8
    physics_dt, decimation, step_dt = .005, 4, .02

    def __init__(self, model_path, motion_file, contract_path, task_profile, *, num_envs=512,
                 seed=42, training=True, dr_mode="mjlab_v1p4", device="cuda:0",
                 fixed_mu=.7, ground_friction=1.):
        from mjlab_tron1_gpu import GpuMjlabBatch
        from mujoco_ppo_env import MujocoPPOEnv
        if dr_mode not in ("none", "mjlab_v1p4"):
            raise ValueError("GPU task supports fixed evaluation or the pinned mjlab recipe")
        if not str(device).startswith("cuda:"):
            raise ValueError("GPU task has no CPU simulation fallback")
        self.device = torch.device(device)
        self.dtype = torch.float32
        # Reuse audited startup model/material/encoder/COM preparation exactly.
        # The NumPy task is never stepped after this one-time construction.
        self._startup = MujocoPPOEnv(
            model_path, motion_file, contract_path, task_profile, num_envs=num_envs,
            seed=seed, training=training, dr_mode=dr_mode, fixed_mu=fixed_mu,
            ground_friction=ground_friction,
            batch_factory=lambda *a: GpuMjlabBatch(*a, device=str(self.device)))
        boot = self._startup
        self.batch = boot.batch
        self.num_envs, self.seed, self.training = boot.num_envs, boot.seed, boot.training
        self.frames, self.max_episode_length = boot.frames, boot.max_episode_length
        self.dr_mode, self.profile, self.contract = dr_mode, boot.profile, boot.contract
        self.reference, self.models, self.datas = boot.reference, boot.models, boot.datas
        self._metadata_startup = boot.metadata()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        tensor = lambda value, dtype=self.dtype: torch.as_tensor(value, device=self.device, dtype=dtype)
        self.qids = tensor(boot.qids, torch.long)
        self.vids = tensor(boot.vids, torch.long)
        self.q0s, self.encoder_bias = tensor(boot.q0s), tensor(boot.encoder_bias)
        self.soft_limits = tensor(boot.soft_limits)
        self.com_local = tensor([model.body_ipos[boot.base_id].tolist() for model in boot.models])
        archive, order = self.reference.archive, boot.reference_body_order
        self.reference_data = {key: tensor(archive[key][:, order]) for key in BODY_FIELDS}
        self.reference_data.update(
            joint_pos=tensor(archive["joint_pos"][:, self.reference.order]),
            joint_vel=tensor(archive["joint_vel"][:, self.reference.order]))
        self.objective = TorchHeightSmoothObjective(self.profile, self.soft_limits,
                                                    device=self.device, dtype=self.dtype)
        self.sampler = TensorPhaseSampler(self.frames, self.num_envs, self.generator, self.device)
        self.phases = tensor(boot.phases, torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.previous_actions = torch.zeros((self.num_envs, 8), device=self.device)
        self.previous_vz = torch.zeros(self.num_envs, device=self.device)
        self.first_reward = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.push_timer = tensor(boot.push_timer)
        self.push_bounds = tensor([[-.5, .5], [-.5, .5], [-.2, .2],
                                   [-.52, .52], [-.52, .52], [-.78, .78]])
        self.gravity = tensor([0., 0., -1.])
        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.total_transitions = self.total_physical_steps = 0
        self.total_resets = torch.tensor(self.num_envs, device=self.device, dtype=torch.long)
        self.total_pushes = torch.zeros((), device=self.device, dtype=torch.long)
        self._closed = False

    def _reference(self):
        return {key: value[self.phases] for key, value in self.reference_data.items()}

    def _state(self, raw=None):
        raw = self.batch.measure_tensors() if raw is None else raw
        state = {key: raw[key] for key in BODY_FIELDS}
        state.update(joint_pos=raw["qpos"][:, self.qids], joint_vel=raw["qvel"][:, self.vids])
        return state

    def observations(self, state=None):
        return build_observations(
            self._state() if state is None else state, self._reference(), self.phases,
            self.q0s, self.previous_actions, self.frames,
            encoder_bias=self.encoder_bias if self.dr_mode == "mjlab_v1p4" else None)

    def _uniform(self, shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def _reset_mask(self, mask, failed=None):
        if failed is None:
            failed = torch.zeros_like(mask)
        self.sampler.record_failures(self.phases, failed & mask)
        sampled = self.sampler.sample(deterministic=not self.training)
        self.phases.copy_(torch.where(mask, sampled, self.phases))
        ref = self._reference()
        qpos = torch.zeros((self.num_envs, 15), device=self.device)
        qvel = torch.zeros((self.num_envs, 14), device=self.device)
        qpos[:, :3], qpos[:, 3:7] = ref["body_pos_w"][:, 0], ref["body_quat_w"][:, 0]
        qpos[:, self.qids] = torch.clamp(ref["joint_pos"], self.soft_limits[:, 0], self.soft_limits[:, 1])
        qvel[:, self.vids] = ref["joint_vel"]
        omega = ref["body_ang_vel_w"][:, 0]
        qvel[:, :3] = ref["body_lin_vel_w"][:, 0]-torch.cross(
            omega, quaternion_rotate(qpos[:, 3:7], self.com_local), dim=-1)
        qvel[:, 3:6] = quaternion_rotate(quaternion_inverse(qpos[:, 3:7]), omega)
        self.batch.reset_tensors(mask, qpos, qvel)
        self.previous_actions.masked_fill_(mask[:, None], 0.)
        self.previous_vz.masked_fill_(mask, 0.)
        self.first_reward.logical_or_(mask)
        self.episode_length_buf.masked_fill_(mask, 0)
        self.episode_rewards.masked_fill_(mask, 0.)
        if self.dr_mode == "mjlab_v1p4":
            self.push_timer.copy_(torch.where(mask, 1.+2.*self._uniform((self.num_envs,)), self.push_timer))
        self.total_resets.add_(mask.sum())

    def reset(self, *, seed=None):
        if seed is not None and int(seed) != self.seed:
            raise ValueError("Reconstruct the environment to change startup seed")
        self._reset_mask(torch.ones(self.num_envs, device=self.device, dtype=torch.bool))
        return self.observations()

    def _push_mask(self, mask):
        delta = self._uniform((self.num_envs, 6))
        delta = delta*(self.push_bounds[:, 1]-self.push_bounds[:, 0])+self.push_bounds[:, 0]
        self.batch.apply_link_velocity_deltas(mask, delta)
        self.push_timer.copy_(torch.where(mask, 1.+2.*self._uniform((self.num_envs,)), self.push_timer))
        self.total_pushes.add_(mask.sum())

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not isinstance(actions, torch.Tensor) or actions.device != self.device or actions.shape != (self.num_envs, 8):
            raise ValueError("Expected a (num_envs,8) action tensor on the simulation CUDA device")
        raw_actions = actions.detach().to(self.dtype)
        actions = raw_actions.clamp(-1., 1.)
        pre, reference = self._state(), self._reference()
        held = self.objective.hold_reference(reference, pre)
        # Let the bridge flag nonfinite RAW outputs before saturation. In
        # particular +/-inf must not silently become a valid +/-1 action.
        physical = self.batch.step_tensors(raw_actions)
        state = self._state(physical)
        scored = self.objective.evaluate(state, held, actions, self.previous_actions,
                                         self.previous_vz, self.first_reward,
                                         physical["body_net_forces_w"])
        rewards = self.step_dt*scored["total_reward"]
        self.episode_length_buf.add_(1)
        self.episode_rewards.add_(rewards)
        gravity = self.gravity
        ref_z = quaternion_rotate(quaternion_inverse(reference["body_quat_w"][:, 0]), gravity)[:, 2]
        robot_z = quaternion_rotate(quaternion_inverse(state["body_quat_w"][:, 0]), gravity)[:, 2]
        terminated = (ref_z-robot_z).abs() > .6
        truncated = (self.phases >= self.frames-1) | (self.episode_length_buf >= self.max_episode_length)
        done = terminated | truncated
        self.previous_actions.copy_(actions)
        self.previous_vz.copy_(state["body_lin_vel_w"][:, 0, 2])
        self.first_reward.fill_(False)
        terminal = self.observations(state)
        # Only this tiny control payload crosses the PCIe boundary. Conditional
        # event forwards must remain identical to the frozen physics sequence.
        will_push = ((self.push_timer-self.step_dt < 1.e-6) & ~done
                     if self.dr_mode == "mjlab_v1p4" else torch.zeros_like(done))
        invalid = (~torch.isfinite(raw_actions).all() | ~torch.isfinite(rewards).all() |
                   ~torch.isfinite(terminal["policy"]).all() | ~torch.isfinite(terminal["critic"]).all())
        event_flags = torch.stack((done.any(), will_push.any(), invalid)).to(torch.uint8).cpu().tolist()
        if event_flags[2]:
            raise FloatingPointError("Nonfinite GPU task observation/reward; not an ordinary reset")
        extras = {"terminal_observation": terminal, "terminal_mask": done,
                  "time_outs": truncated, "reference_frames_for_action": self.phases.clone(),
                  "raw_reward_terms": scored["raw_terms"],
                  "log": {"reward/"+key: value.mean()*self.step_dt
                          for key, value in scored["weighted_terms"].items()}}
        self.phases.copy_(torch.where(done, self.phases, torch.clamp(self.phases+1, max=self.frames-1)))
        if event_flags[0]:
            self._reset_mask(done, failed=terminated)
        self.sampler.update()
        if self.dr_mode == "mjlab_v1p4":
            self.push_timer.sub_(self.step_dt)
            if event_flags[1]:
                self._push_mask(will_push)
        self.total_transitions += self.num_envs
        self.total_physical_steps += self.num_envs*self.decimation
        # When no event changed physics, reuse the post-step tensors directly.
        final_state = None if event_flags[0] or event_flags[1] else state
        return self.observations(final_state), rewards, terminated, truncated, extras

    def metadata(self):
        result = copy.deepcopy(self._metadata_startup)
        result["reward_contract"]["kind"] = "torch_device_exact_height_smooth_tracking_objective"
        result.update(implementation="mjlab_gpu_resident_tron1_task",
                      native_backend=copy.deepcopy(self.batch.metadata),
                      task_arrays_device=str(self.device), host_task_adapter=False,
                      random_stream="private per-device Torch generator; startup draws unchanged, subsequent distributions unchanged but not NumPy draw-for-draw",
                      task_event_host_transfer="one 3-byte reset/push/fault payload per interval; no bulk tensors",
                      physics_forward_schedule="unchanged: 4 step + 4 forward, conditional reset and push forwards",
                      physical_counters={"environment_transitions": self.total_transitions,
                                         "physical_environment_substeps": self.total_physical_steps,
                                         "environment_resets": int(self.total_resets.item()),
                                         "interval_pushes": int(self.total_pushes.item())})
        return result

    def close(self):
        self._closed = True
        self._startup.close()


class TensorRslWrapper:
    """RSL-RL3 VecEnv protocol without any bulk host/device conversions."""
    def __init__(self, env, device=None):
        self.env, self.device = env, env.device
        self.num_envs, self.num_actions = env.num_envs, env.num_actions
        self.max_episode_length = env.max_episode_length

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    def _obs(self, values):
        from tensordict import TensorDict
        return TensorDict(values, batch_size=[self.num_envs])

    def get_observations(self):
        return self._obs(self.env.observations())

    def reset(self):
        return self._obs(self.env.reset()), {}

    def step(self, actions):
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        return (self._obs(obs), reward, (terminated | truncated).to(torch.long),
                {"time_outs": truncated, "log": extras["log"]})
