"""Single-episode actuator-command diagnostic, preserving raw PPO decisions.

Passthrough calls the installed action method exactly. The intervention ramps
the six processed leg position targets over eight real physical substeps;
wheel effort commands and the policy/reward/observation functions stay fixed.
This is an inference-time intervention, not a newly qualified trained policy.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import MethodType

import numpy as np

MODES = ("passthrough", "linear_position_8substeps")


class LegCommandProbe:
    def __init__(self, env, mode, *, allow_cpu_fixture=False):
        import torch
        self.torch, self.env, self.mode = torch, env, mode
        assert mode in MODES and env.num_envs == 1 and env.physics_dt == .0025 and env.step_dt == .02
        assert allow_cpu_fixture or torch.device(env.device).type == "cuda"
        self.robot = env.scene["robot"]
        self.term = env.action_manager.get_term("joint_pos")
        self.wheels = env.action_manager.get_term("wheel_effort")
        assert allow_cpu_fixture or type(self.term).__name__ == "JointPositionAction"
        assert self.term._asset is self.robot and len(self.term._joint_ids) == 6
        assert len(self.wheels._joint_ids) == 2 and not set(self.term._joint_ids) & set(self.wheels._joint_ids)
        self.ids = list(self.term._joint_ids)
        self.initial_counter = int(env._sim_step_counter)
        self.original_apply = self.term.apply_actions
        self.sources = {}
        if not allow_cpu_fixture:
            for func in (self.original_apply, self.robot.set_joint_position_target, type(env).step):
                path = Path(inspect.getsourcefile(func)).resolve()
                self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.rows = []
        self.previous = self.desired = None
        self.wrapper = MethodType(lambda owner: self.apply(), self.term)
        self.term.apply_actions = self.wrapper

    def apply(self):
        torch = self.torch
        counter = int(self.env._sim_step_counter)
        assert counter == self.initial_counter+len(self.rows)+1, "Only consecutive physical action applications are allowed"
        assert self.term.apply_actions is self.wrapper
        phase = len(self.rows) % 8
        if phase == 0:
            self.previous = self.robot.data.joint_pos_target[:, self.ids].clone()
            self.desired = self.term.processed_actions.clone()
        else:
            assert torch.equal(self.desired, self.term.processed_actions), "Desired target changed inside a held-action window"
        assert self.previous.shape == self.desired.shape == (1, 6)
        if self.mode == "passthrough":
            result = self.original_apply()
        else:
            fraction = (phase+1)/8.
            command = self.previous+(self.desired-self.previous)*fraction
            if phase == 7:
                command = self.desired  # Exact endpoint; no floating-point endpoint drift.
            result = self.robot.set_joint_position_target(command, joint_ids=self.ids)
        plain = lambda value: value.detach().cpu().numpy().copy()
        self.rows.append(dict(sim_counter=counter, substep=phase+1,
            previous_policy_target_rad=plain(self.previous[0]),
            desired_policy_target_rad=plain(self.desired[0]),
            actual_leg_position_target_rad=plain(self.robot.data.joint_pos_target[0, self.ids]),
            unchanged_wheel_effort_target_nm=plain(self.wheels.processed_actions[0])))
        return result

    def save(self, path):
        assert len(self.rows) == 2792 and self.term.apply_actions is self.wrapper
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in self.sources.items())
        values = {name: np.asarray([row[name] for row in self.rows]) for name in self.rows[0]}
        values["time_before_step_s"] = np.arange(2792)*.0025
        values["leg_joint_names"] = np.asarray([self.robot.joint_names[i] for i in self.ids])
        values["mode"] = np.asarray(self.mode)
        assert all(np.isfinite(value).all() for name,value in values.items() if value.dtype.kind in "fi")
        target = values["desired_policy_target_rad"].reshape(349, 8, 6)
        actual = values["actual_leg_position_target_rad"].reshape(349, 8, 6)
        old = values["previous_policy_target_rad"].reshape(349, 8, 6)
        np.testing.assert_array_equal(target, np.broadcast_to(target[:, :1], target.shape))
        np.testing.assert_array_equal(old, np.broadcast_to(old[:, :1], old.shape))
        np.testing.assert_array_equal(actual[:, -1], target[:, -1])
        if self.mode == "passthrough":
            np.testing.assert_array_equal(actual, target)
        else:
            expected = old+(target-old)*(np.arange(1, 9)[None, :, None]/8.)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        np.savez_compressed(path, **values)
        return dict(mode=self.mode, physical_steps=2792, policy_decisions=349,
            interpolated_leg_joints=values["leg_joint_names"].tolist(), physics_dt_s=.0025, policy_dt_s=.02,
            interpolation_window_s=.02 if self.mode != "passthrough" else 0.,
            actual_intermediate_leg_targets_changed=self.mode != "passthrough",
            wheel_effort_interpolation=False, raw_actor_observation_action_mapping_changed=False,
            actor_parameters_and_reward_functions_changed=False, physical_normal_or_friction_model_changed=False,
            command_endpoints_exact=True, SDK_source_sha256=self.sources,
            validation_only_intervention=True, trained_for_interpolated_execution=False,
            pose_recording_interpolation_or_correction=False, task_success_claimed=False)
