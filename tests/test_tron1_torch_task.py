"""CPU float64/float32 parity for the device-resident TRON1 task arithmetic."""
from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("Optional Torch dependency unavailable") from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

import tron1_torch_task as gpu
from mujoco_ppo_env import build_observations as numpy_observations
from mujoco_ppo_env import rotation_matrix as numpy_rotation_matrix
from nmpc_batch_objective import BatchHeightSmoothTrackingObjective
from nmpc_tracking_objective import (quaternion_inverse, quaternion_multiply,
                                     quaternion_rotate)


def _fixture(batch=23):
    rng = np.random.default_rng(721)

    def state():
        q = rng.normal(size=(batch, 9, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return {"body_pos_w": rng.normal(scale=.3, size=(batch, 9, 3)),
                "body_quat_w": q,
                "body_lin_vel_w": rng.normal(size=(batch, 9, 3)),
                "body_ang_vel_w": rng.normal(size=(batch, 9, 3)),
                "joint_pos": rng.normal(size=(batch, 8)),
                "joint_vel": rng.normal(size=(batch, 8))}

    pre, post, ref = state(), state(), state()
    actions, previous = rng.uniform(-1., 1., size=(2, batch, 8))
    vz = rng.normal(size=batch)
    first = rng.random(batch) < .3
    forces = rng.normal(size=(batch, 4, 9, 3))*4
    q0 = rng.normal(size=(batch, 8))
    phases = rng.integers(0, 349, size=batch)
    limits = np.array([[-.5, .5]]*6+[[-1.e9, 1.e9]]*2)
    profile = json.loads((ROOT/"config/car_drift_height_smooth_task.json").read_text())
    return pre, post, ref, actions, previous, vz, first, forces, q0, phases, limits, profile


def _tensors(values, dtype):
    if isinstance(values, dict):
        return {key: _tensors(value, dtype) for key, value in values.items()}
    value = np.asarray(values)
    return torch.as_tensor(value, dtype=(torch.bool if value.dtype.kind == "b" else
                                        torch.long if value.dtype.kind in "iu" else dtype))


def _observation_oracle(state, ref, phases, q0, previous):
    """Original observation arithmetic before its final float32 output cast."""
    p, q, v, w = (state[key] for key in gpu.BODY_FIELDS)
    inv = quaternion_inverse(q[:, 0])
    phase = 2*np.pi*phases/348
    rel_q = quaternion_multiply(inv, ref["body_quat_w"][:, 0])
    actor = np.concatenate((ref["joint_pos"][:, :6], ref["joint_vel"][:, :6],
                            np.sin(phase)[:, None], np.cos(phase)[:, None],
                            quaternion_rotate(inv, ref["body_pos_w"][:, 0]-p[:, 0]),
                            numpy_rotation_matrix(rel_q)[..., :2].reshape(len(p), 6),
                            quaternion_rotate(inv, v[:, 0]), quaternion_rotate(inv, w[:, 0]),
                            state["joint_pos"][:, :6]-q0[:, :6], state["joint_vel"], previous), axis=-1)
    rel_p = quaternion_rotate(inv[:, None], p-p[:, :1])
    rel_q = quaternion_multiply(inv[:, None], q[:, :7])
    critic = np.concatenate((actor, rel_p.reshape(len(p), 27),
                             numpy_rotation_matrix(rel_q)[..., :2].reshape(len(p), 42)), axis=-1)
    return {"policy": actor, "critic": critic}


class TorchTaskParityTests(unittest.TestCase):
    def test_quaternion_broadcast_and_sign(self):
        pre, _, ref, *_ = _fixture()
        q = _tensors(pre["body_quat_w"][:, :1], torch.float64)
        vector = _tensors(ref["body_pos_w"], torch.float64)
        rotated = gpu.quaternion_rotate(q, vector).numpy()
        np.testing.assert_allclose(rotated, quaternion_rotate(q.numpy(), vector.numpy()), atol=2.e-15)
        np.testing.assert_allclose(gpu.quaternion_angle(q, -q).numpy(), 0., atol=2.e-15)
        np.testing.assert_allclose(gpu.rotation_matrix(q).numpy(), numpy_rotation_matrix(q.numpy()), atol=2.e-15)

    def test_actor_and_critic_f64_and_f32(self):
        pre, _, ref, _, previous, _, _, _, q0, phases, *_ = _fixture()
        expected = _observation_oracle(pre, ref, phases, q0, previous)
        rounded = numpy_observations(pre, ref, phases, q0, previous, 349)
        for dtype, atol in ((torch.float64, 8.e-15), (torch.float32, 2.e-6)):
            actual = gpu.build_observations(_tensors(pre, dtype), _tensors(ref, dtype),
                                            _tensors(phases, dtype), _tensors(q0, dtype),
                                            _tensors(previous, dtype), 349)
            for key in actual:
                self.assertEqual(actual[key].dtype, dtype)
                self.assertEqual(actual[key].device.type, "cpu")
                np.testing.assert_allclose(actual[key].numpy(), expected[key], rtol=2.e-6 if dtype==torch.float32 else 0., atol=atol)
                if dtype == torch.float64:
                    np.testing.assert_allclose(actual[key].numpy().astype(np.float32), rounded[key], atol=0.)

    def test_privileged_critic_encoder_bias_not_applied_twice(self):
        pre, _, ref, _, previous, _, _, _, q0, phases, *_ = _fixture()
        bias = np.full_like(q0, .007)
        arguments = (_tensors(pre, torch.float64), _tensors(ref, torch.float64),
                     _tensors(phases, torch.float64), _tensors(q0-bias, torch.float64),
                     _tensors(previous, torch.float64), 349)
        actual = gpu.build_observations(*arguments, encoder_bias=_tensors(bias, torch.float64))
        explicit = gpu.build_observations(*arguments, critic_q0s=_tensors(q0, torch.float64))
        expected = _observation_oracle(pre, ref, phases, q0-bias, previous)
        expected["critic"][:, 29:35] -= bias[:, :6]
        for key in actual:
            np.testing.assert_allclose(actual[key].numpy(), expected[key], atol=8.e-15, rtol=0.)
            torch.testing.assert_close(actual[key], explicit[key], atol=8.e-15, rtol=0.)

    def _compare_reward(self, dtype, *, leg_rate=False):
        pre, post, ref, action, previous, vz, first, forces, _, _, limits, profile = _fixture()
        if leg_rate:
            profile["leg_target_rate_weight"] = -.0001
        numpy_objective = BatchHeightSmoothTrackingObjective(profile, soft_joint_limits=limits)
        pre_h = {key: value[:, None] for key, value in pre.items() if key in gpu.BODY_FIELDS}
        ref_h = {key: value[:, None] for key, value in ref.items() if key in gpu.BODY_FIELDS}
        post_h = {key: value[:, None] for key, value in post.items() if key in gpu.BODY_FIELDS}
        post_h.update(joint_pos=post["joint_pos"][:, None],
                      body_net_forces_w=forces[:, None, -1],
                      contact_force_history_w=forces[:, None, -3:])
        held_np = numpy_objective.hold_reference(ref_h, pre_h)
        expected = numpy_objective.evaluate(post_h, held_np, action[:, None], previous[:, None],
                                            vz[:, None], reset_samples=first[:, None])
        objective = gpu.TorchHeightSmoothObjective(profile, limits, dtype=dtype)
        held = objective.hold_reference(_tensors(ref, dtype), _tensors(pre, dtype))
        for key, value in held.items():
            np.testing.assert_allclose(value.numpy(), getattr(held_np, key)[:, 0],
                                       rtol=3.e-6 if dtype==torch.float32 else 0.,
                                       atol=5.e-7 if dtype==torch.float32 else 3.e-15)
        actual = objective.evaluate(_tensors(post, dtype), held, _tensors(action, dtype),
                                     _tensors(previous, dtype), _tensors(vz, dtype),
                                     _tensors(first, dtype), _tensors(forces, dtype))
        self.assertEqual(len(actual["raw_terms"]), 16 if leg_rate else 15)
        atol, rtol = ((3.e-5, 3.e-5) if dtype==torch.float32 else (3.e-13, 3.e-13))
        for group in ("raw_terms", "weighted_terms", "diagnostics"):
            self.assertEqual(set(actual[group]), set(expected[group]))
            for name, value in actual[group].items():
                np.testing.assert_allclose(value.numpy(), expected[group][name][:, 0],
                                           atol=atol, rtol=rtol, err_msg=group+"/"+name)
        for name in ("total_reward", "stage_cost", "base_com_vz"):
            np.testing.assert_allclose(actual[name].numpy(), expected[name][:, 0], atol=atol, rtol=rtol)
        np.testing.assert_array_equal(actual["raw_terms"]["height_vertical_acceleration"].numpy()[first], 0.)

    def test_all_fifteen_reward_terms_float64(self):
        self._compare_reward(torch.float64)

    def test_all_fifteen_reward_terms_float32(self):
        self._compare_reward(torch.float32)

    def test_optional_leg_target_rate(self):
        self._compare_reward(torch.float64, leg_rate=True)

    def test_contact_thresholds_last_three_and_nonwheel_exclusion(self):
        pre, post, ref, action, previous, vz, first, forces, _, _, limits, profile = _fixture(batch=4)
        forces[:] = 0.
        forces[:, -1, 7:, 2] = 5.001
        forces[0, 0, 0, 2] = 20.  # Oldest substep is not part of history.
        forces[1, 1, 1, 2] = 1.001
        forces[2, -1, 1, 2] = 1.  # Strict > threshold.
        forces[3, -1, 8, 2] = 5.  # Both wheels must strictly exceed 5N.
        objective = gpu.TorchHeightSmoothObjective(profile, limits, dtype=torch.float64)
        held = objective.hold_reference(_tensors(ref, torch.float64), _tensors(pre, torch.float64))
        actual = objective.evaluate(_tensors(post, torch.float64), held, _tensors(action, torch.float64),
                                     _tensors(previous, torch.float64), _tensors(vz, torch.float64),
                                     _tensors(first, torch.float64), _tensors(forces, torch.float64))
        np.testing.assert_array_equal(actual["raw_terms"]["undesired_contacts"].numpy(), [0., 1., 0., 0.])
        np.testing.assert_array_equal(actual["raw_terms"]["drift_grounded"].numpy(), [1., 1., 1., 0.])

    def test_height_invariance_and_zero_sideslip_weight(self):
        pre, post, ref, action, previous, vz, first, forces, _, _, limits, profile = _fixture()
        profile["reward_weights"]["sideslip"] = 0.
        objective = gpu.TorchHeightSmoothObjective(profile, limits, dtype=torch.float64)

        def score(state, reference):
            held = objective.hold_reference(_tensors(reference, torch.float64), _tensors(pre, torch.float64))
            return objective.evaluate(_tensors(state, torch.float64), held, _tensors(action, torch.float64),
                                       _tensors(previous, torch.float64), _tensors(vz, torch.float64),
                                       _tensors(first, torch.float64), _tensors(forces, torch.float64))

        before = score(post, ref)
        changed_post, changed_ref = copy.deepcopy(post), copy.deepcopy(ref)
        changed_post["body_pos_w"][..., 2] += 500.
        changed_ref["body_pos_w"][..., 2] -= 250.
        changed_ref["body_lin_vel_w"][..., 2] += 100.
        after = score(changed_post, changed_ref)
        torch.testing.assert_close(before["total_reward"], after["total_reward"], atol=0., rtol=0.)
        np.testing.assert_array_equal(after["weighted_terms"]["drift_sideslip"].numpy(), 0.)

    def test_held_reference_copies_inputs(self):
        pre, _, ref, *_, limits, profile = _fixture()
        objective = gpu.TorchHeightSmoothObjective(profile, limits, dtype=torch.float64)
        reference = _tensors(ref, torch.float64)
        held = objective.hold_reference(reference, _tensors(pre, torch.float64))
        frozen = {key: value.clone() for key, value in held.items()}
        for value in reference.values():
            value.add_(10.)
        for key in frozen:
            torch.testing.assert_close(held[key], frozen[key], atol=0., rtol=0.)

    def test_hot_functions_have_no_explicit_host_transfers(self):
        forbidden = {"item", "cpu", "numpy", "tolist"}
        for function in (gpu.build_observations, gpu.TorchHeightSmoothObjective.hold_reference,
                         gpu.TorchHeightSmoothObjective.evaluate):
            source = inspect.getsource(function)
            import textwrap
            parsed = ast.parse(textwrap.dedent(source))
            called = {node.func.attr for node in ast.walk(parsed)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            self.assertFalse(called & forbidden, function.__name__)


if __name__ == "__main__":
    unittest.main()
