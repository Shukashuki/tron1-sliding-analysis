"""Reference coherence, paired trials and probability-reporting boundaries."""
import sys
import copy
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from hockey_speed_study import scale_reference, evaluation_cases, perturbed_root, wilson_interval
from summarize_hockey_speed_study import summarize


class SpeedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT/"results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2"
        cls.motion = dict(np.load(directory/"motion.npz", allow_pickle=False))
        cls.diagnostics = dict(np.load(directory/"command_diagnostics.npz", allow_pickle=False))

    def test_translation_scaling_preserves_pose_and_relative_motion(self):
        m, d = scale_reference(self.motion, self.diagnostics, 2.)
        for key in ("joint_pos", "joint_vel", "body_quat_w", "body_ang_vel_w"):
            np.testing.assert_array_equal(m[key], self.motion[key])
        np.testing.assert_array_equal(m["body_pos_w"][..., 2], self.motion["body_pos_w"][..., 2])
        np.testing.assert_allclose(m["body_pos_w"]-m["body_pos_w"][:, :1],
                                   self.motion["body_pos_w"]-self.motion["body_pos_w"][:, :1], atol=1e-14)
        np.testing.assert_allclose(m["body_lin_vel_w"]-m["body_lin_vel_w"][:, :1],
                                   self.motion["body_lin_vel_w"]-self.motion["body_lin_vel_w"][:, :1], atol=1e-14)
        t = d["time_s"]
        np.testing.assert_allclose(np.gradient(d["qpos"][:, :2], t, axis=0), d["root_link_velocity_w_m_s"][:, :2], atol=1e-12)
        np.testing.assert_allclose(d["qpos"][-1, :2]-d["qpos"][0, :2],
            2*(self.diagnostics["qpos"][-1, :2]-self.diagnostics["qpos"][0, :2]), atol=1e-14)

    def test_one_x_is_identity_and_bad_speed_rejected(self):
        m, d = scale_reference(self.motion, self.diagnostics, 1.)
        for key in self.motion:
            np.testing.assert_array_equal(m[key], self.motion[key])
        for key in self.diagnostics:
            np.testing.assert_array_equal(d[key], self.diagnostics[key])
        for value in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                scale_reference(self.motion, self.diagnostics, value)

    def test_perturbations_preserve_height_geometry_and_nominal_state(self):
        cases = evaluation_cases()
        self.assertEqual(len(cases), 33)
        self.assertEqual(cases, evaluation_cases())
        r = Rotation.from_euler('xyz', [.11, -.16, .7])
        pose = np.r_[[1., 2., .8], r.as_quat()[[3, 0, 1, 2]]]
        velocity = np.array([1.2, .3, .02, .1, -.2, .3])
        p0, v0 = perturbed_root(pose, velocity, cases[0])
        np.testing.assert_array_equal(p0, pose); np.testing.assert_array_equal(v0, velocity)
        for case in cases[1:]:
            p, v = perturbed_root(pose, velocity, case)
            r2 = Rotation.from_quat(p[3:][[1, 2, 3, 0]])
            np.testing.assert_allclose(r2.as_matrix()[2], r.as_matrix()[2], atol=1e-15)
            self.assertEqual(p[2], pose[2]); self.assertEqual(v[2], velocity[2])
            self.assertLessEqual(abs(np.linalg.norm(p[3:])-1), 1e-14)
        np.testing.assert_array_equal(pose[:3], [1., 2., .8])

    def test_small_counts_do_not_imply_certain_success(self):
        lo, hi = wilson_interval(3, 3)
        self.assertLess(lo, .45); self.assertAlmostEqual(hi, 1.)
        lo0, hi0 = wilson_interval(0, 3)
        self.assertAlmostEqual(lo0, 0.); self.assertAlmostEqual(hi0, 1-lo)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_missing_models_are_not_in_success_denominator(self):
        plan = json.loads((ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1/plan.json').read_text())
        empty = summarize(plan, {})
        self.assertTrue(all(g['nominal_seed_success_rate'] is None for g in empty['groups']))
        cell = plan['cells'][0]
        cases = [dict(id=c['id'], kind=c['kind'], success=i == 0 or i <= 8) for i,c in enumerate(plan['cases'])]
        report = dict(method=cell['method'], seed=cell['seed'], speed_scale=cell['speed_scale'], cases=cases,
            nominal_success=True, perturbation_trials=32, perturbation_successes=8, perturbation_success_rate=.25,
            conditional_Wilson_95_interval=wilson_interval(8,32), failure_gate_counts_nonexclusive={})
        result = summarize(plan, {cell['id']:report})
        group = next(g for g in result['groups'] if g['completed_models'])
        self.assertEqual(group['completed_models'], 1)
        self.assertEqual(group['nominal_seed_success_rate'], 1.)
        self.assertEqual(group['conditional_rollout_rate_mean'], .25)
        self.assertEqual(len(group['missing_cells']), 2)
        self.assertEqual(result['status'], 'partial_or_not_started')
        corrupt = copy.deepcopy(report); corrupt['cases'][1],corrupt['cases'][2]=corrupt['cases'][2],corrupt['cases'][1]
        with self.assertRaises(ValueError):
            summarize(plan, {cell['id']:corrupt})
        corrupt = copy.deepcopy(report); corrupt['perturbation_success_rate'] = 1.
        with self.assertRaises(ValueError):
            summarize(plan, {cell['id']:corrupt})


if __name__ == "__main__":
    unittest.main()
