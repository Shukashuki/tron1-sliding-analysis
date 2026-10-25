"""O reuses the frozen K eight-substep collector and adds command bookkeeping.

No extra sensor refresh, force write, action mutation or actor input. The K
ownership marker is deliberately retained; an O owner cannot stack with K.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import ground_contact_reward_v2o as definition
import ppo_ground_contact_v2k as parent

PARENT_COLLECTOR_SHA256 = "3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d"
JOINTS = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
          "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")
MARKER, SENSORS, WHEELS, GROUND = parent.MARKER, parent.SENSORS, parent.WHEELS, parent.GROUND
PHYSICS_DT, POLICY_DT, SUBSTEPS = parent.PHYSICS_DT, parent.POLICY_DT, parent.SUBSTEPS
configure_sensors, clock, require = parent.configure_sensors, parent.clock, parent.require


class GroundContactV2O(parent.GroundContactV2K):
    def __init__(self, env, settings=None, *, allow_cpu_fixture=False):
        self.o_settings = definition.validate_settings(settings)
        definition.verify_base()
        require(hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() == PARENT_COLLECTOR_SHA256,
                "Frozen K physical collector differs")
        require(tuple(env.scene["robot"].joint_names) == JOINTS,
                "Require declared interleaved leg/wheel joint action order")
        for name in ("_tron1_ground_contact_v2l", "_tron1_ground_contact_v2m", "_tron1_ground_contact_v2n"):
            require(not hasattr(env, name) and not hasattr(env.scene, name), "Require a fresh single collector")
        # Parent arithmetic continues to receive exactly K settings.
        super().__init__(env, parent.definition.DEFAULT_SETTINGS, allow_cpu_fixture=allow_cpu_fixture)
        self.last_command = self.torch.zeros((self.num_envs, 8), device=self.device)
        self.current_command = self.torch.zeros_like(self.last_command)
        self.previous_command_for_reward = self.torch.zeros_like(self.last_command)
        self.last_normal = self.torch.zeros((self.num_envs, 2), device=self.device)
        self.preceding_normal_for_reward = self.torch.zeros_like(self.last_normal)
        self.command_fault = self.torch.zeros(self.num_envs, device=self.device, dtype=self.torch.bool)
        self.command_reads = self.command_reset_calls = 0
        self.last_command_read = None
        self.command_penalty_sum = self.torch.zeros((), device=self.device)
        self.o_raw_sum = self.torch.zeros((), device=self.device)
        self.preceding_support_count = self.torch.zeros((), device=self.device, dtype=self.torch.long)

    def step(self, actions, *args, **kwargs):
        self._attached()
        require(not self.inside_step, "Reentrant control step")
        require(self.torch.is_tensor(actions) and tuple(actions.shape) == (self.num_envs, 8)
                and actions.dtype == self.torch.float32 and actions.device == self.device,
                "Require actual explicit-device float32 raw policy commands[N,8]")
        self.preceding_normal_for_reward.copy_(self.last_normal)
        self.previous_command_for_reward.copy_(self.last_command)
        self.current_command.copy_(actions.clamp(-1., 1.))
        # Update before the underlying step: a later autoreset must clear the
        # cache, rather than being accidentally overwritten after step returns.
        self.last_command.copy_(self.current_command)
        self.command_fault.logical_or_(~self.torch.isfinite(actions).all(-1))
        return super().step(actions, *args, **kwargs)

    def reward(self):
        first = self.last_read != self.steps
        value_k = super().reward()
        components = definition.command_torch(self.preceding_normal_for_reward,
            self.current_command, self.previous_command_for_reward, validate=False)
        value = value_k-components["stance_command_penalty"]
        self.command_fault.logical_or_(~self.torch.isfinite(value))
        self.reward_fault.logical_or_(~self.torch.isfinite(value))
        if first:
            self.command_reads += 1
            self.last_command_read = self.steps
            self.command_penalty_sum.add_(self.torch.nan_to_num(components["stance_command_penalty"]).sum())
            self.o_raw_sum.add_(self.torch.nan_to_num(value).sum())
            self.preceding_support_count.add_((self.preceding_normal_for_reward > 5.).any(-1).sum())
            # This is the exact completed final physical sample, not a new
            # sensor read and not the first sample following the next command.
            self.last_normal.copy_(self.forces[:, -1])
        return value

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.last_command[ids] = 0.
        self.last_normal[ids] = 0.
        self.command_reset_calls += 1

    def audit(self, health=False):
        result = super().audit(health=health)
        result.update(reward_definition_version="ground_contact_v2o_any_wheel",
            settings=self.o_settings, parent_K_settings=self.settings,
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definition_sha256=hashlib.sha256(Path(definition.__file__).read_bytes()).hexdigest(),
            immutable_v2k_definition_sha256=definition.BASE_K_DEFINITION_SHA256,
            immutable_v2k_collector_sha256=PARENT_COLLECTOR_SHA256,
            support_gap_coefficient_previous=1., support_gap_coefficient_current=1.,
            force_slew_cost_added=False, stance_command_cost_added=True,
            stance_command_delta_std_rad=self.o_settings["stance_command_delta_std_rad"],
            stance_command_penalty_weight=self.o_settings["stance_command_penalty_weight"],
            stance_command_cost_mode=self.o_settings["stance_command_cost_mode"],
            joint_action_order=list(JOINTS), leg_command_scale_rad=1., raw_action_clip=[-1., 1.],
            left_leg_action_indices=list(definition.LEFT), right_leg_action_indices=list(definition.RIGHT),
            command_unique_reward_windows=self.command_reads, command_cache_reset_calls=self.command_reset_calls,
            preceding_force_uses_previous_completed_native_sample=True,
            reset_clears_preceding_force_and_previous_command=True,
            command_history_updated_before_underlying_step=True,
            additional_force_refreshes_or_physical_mutations=False,
            airborne_other_leg_and_alternating_stance_allowed=True,
            wheel_torque_actions_in_new_cost=False, other14_reward_terms_changed=False)
        if health:
            count = max(1, self.num_envs*self.command_reads)
            result.update(device_finite_stance_command_health=not bool(self.command_fault.any().item()),
                command_environment_reward_samples=self.num_envs*self.command_reads,
                preceding_anywheel_support_environment_samples=int(self.preceding_support_count.item()),
                stance_command_penalty_mean=float(self.command_penalty_sum.item())/count,
                mean_raw_reward_parent_K=result["mean_raw_reward"],
                mean_raw_reward=float(self.o_raw_sum.item())/count)
        return result


def install(env, settings=None): return GroundContactV2O(env, settings)


def reward_term(weight=.4):
    require(weight == .4, "O grounded external weight stays+.4")
    from isaaclab.managers import ManagerTermBase, RewardTermCfg
    class ContactReward(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env

        def __call__(self, env):
            require(env is self.env, "Different reward environment")
            owner = getattr(env, MARKER, None)
            require(isinstance(owner, GroundContactV2O), "Require actual O physical/command collector")
            return owner.reward()

        def reset(self, env_ids=None):
            owner = getattr(self.env, MARKER, None)
            if owner is not None: owner.reset(env_ids)
    return RewardTermCfg(func=ContactReward, weight=weight)
