"""Counterexamples for contact shaping and policy provenance, without Isaac."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from hockey_contact_search import objective, initialize_policy
from prepare_hockey_contact_search import protocols
from test_hockey_bilateral_training import fixture


class ContactSearchTests(unittest.TestCase):
    def test_single_missed_substep_has_dense_contact_gradient_but_not_slide_credit(self):
        samples,mg,cfg=fixture()
        all_contact=objective(samples,mg,cfg)
        samples['touching'][:,3,1]=0
        missing=objective(samples,mg,cfg)
        self.assertAlmostEqual(float(missing['contact_fraction']),7/8)
        self.assertAlmostEqual(float(missing['contact_loss']),1/8)
        self.assertEqual(float(missing['sliding']),0.)
        self.assertGreater(float(all_contact['sliding']),0.)
        samples['touching'].zero_()
        self.assertEqual(float(objective(samples,mg,cfg)['contact_fraction']),0.)

    def test_contact_and_clearance_do_not_depend_on_load_split(self):
        samples,mg,cfg=fixture()
        initial=objective(samples,mg,cfg)
        samples['normal_fz'][...,0]=0
        samples['normal_fz'][...,1]=mg.item()
        changed=objective(samples,mg,cfg)
        for key in ('contact_fraction','contact_loss','clearance'):
            torch.testing.assert_close(initial[key],changed[key])
        samples['normal_fz'].zero_()
        self.assertEqual(float(objective(samples,mg,cfg)['contact_fraction']),1.)
        self.assertEqual(float(objective(samples,mg,cfg)['sliding']),0.)

    def test_liftoff_and_high_acceleration_cost_without_unbounded_penalties(self):
        samples,mg,cfg=fixture()
        samples['post_gap'][:,-1,0]=.001
        r=objective(samples,mg,cfg)
        self.assertGreater(float(r['clearance']),0.)
        self.assertEqual(float(r['contact_fraction']),7/8)
        samples['post_gap'].fill_(100)
        samples['angular_acceleration_sq'].fill_(1e9)
        r=objective(samples,mg,cfg)
        self.assertEqual(float(r['clearance']),1.)
        self.assertEqual(float(r['angular_acceleration']),1.)

    def test_budget_and_failure_return_not_disguised_by_dt_or_cold_start_label(self):
        spec=json.loads((ROOT/'config/hockey_contact_reward_search_v1.json').read_text())
        parent=json.loads((ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1/protocols/hockey_s100_seed42.json').read_text())
        bilateral=fixture()[2]
        cells=protocols(spec,parent,bilateral)
        self.assertEqual(len(cells),9)
        for name,cfg in cells:
            self.assertEqual(cfg['formal']['num_envs']*cfg['formal']['iterations']*cfg['ppo']['steps_per_env'],3072000)
            self.assertEqual(cfg['reward_weights']['fall']*.02,-8.)
            self.assertFalse(cfg['bilateral_sliding']['gate_positive_tracking_on_bilateral_window'])
            self.assertEqual(cfg['reward_weights']['xy'],parent['reward_weights']['xy'])
            self.assertEqual(cfg['speed_scale'],1.)
            # Even an unrealistically perfect 0.14 s prefix earns less than
            # its one fall cost. This is a unit-scale check, not a PPO guarantee.
            max_rate=sum(max(w,0) for w in cfg['reward_weights'].values())
            max_rate+=sum(max(w,0) for w in cfg['bilateral_sliding']['weights'].values())
            self.assertLess(.14*max_rate-8.,0.)
        self.assertEqual(parent['reward_weights']['fall'],-2.)

    def test_warm_start_preserves_actor_and_normalizer_but_resets_exploration_only(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor=torch.nn.Linear(3,2)
                self.log_std=torch.nn.Parameter(torch.zeros(2))
                self.register_buffer('normalizer_mean',torch.tensor([1.,2.,3.]))
        source=Policy();target=Policy()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'parent.pt'
            torch.save(dict(model_state_dict=source.state_dict(),iter=999),path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            runner=SimpleNamespace(alg=SimpleNamespace(policy=target),tot_timesteps=0,current_learning_iteration=0)
            result=initialize_policy(runner,path,digest,.1)
            for key,value in source.state_dict().items():
                expected=torch.full_like(value,math.log(.1)) if key=='log_std' else value
                torch.testing.assert_close(target.state_dict()[key],expected)
            self.assertFalse(result['optimizer_loaded'])
            with self.assertRaisesRegex(ValueError,'hash'):
                initialize_policy(runner,path,'wrong',.1)
            runner.tot_timesteps=24
            with self.assertRaisesRegex(ValueError,'budget'):
                initialize_policy(runner,path,digest,.1)


if __name__=='__main__':unittest.main()
