import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hockey_training_continuation import initialize_continuation, matching_recipe, state_digest


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(3, 1)
        self.log_std = torch.nn.Parameter(torch.tensor([-.7, -1.2]))
        self.register_buffer('normalizer_mean', torch.tensor([.5, -.4, .9]))


def runner():
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    return SimpleNamespace(tot_timesteps=0, current_learning_iteration=0,
        alg=SimpleNamespace(policy=policy, optimizer=optimizer, learning_rate=1e-4, rnd=None))


def step(r):
    x = torch.tensor([[1., -.5, .2]]) - r.alg.policy.normalizer_mean
    loss = r.alg.policy.actor(x).square().sum() + r.alg.policy.critic(x).square().sum() + r.alg.policy.log_std.square().sum()
    r.alg.optimizer.zero_grad()
    loss.backward()
    r.alg.optimizer.step()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def saved(tmp_path):
    r = runner()
    step(r)
    step(r)
    checkpoint = tmp_path/'model_final.pt'
    torch.save(dict(model_state_dict=r.alg.policy.state_dict(),
                    optimizer_state_dict=r.alg.optimizer.state_dict(), iter=249, infos=None), checkpoint)
    source = dict(ppo=dict(schedule='fixed', learning_rate=1e-4, steps_per_env=24, init_noise_std=.1),
                  formal=dict(iterations=250, num_envs=512), smoke=dict(iterations=2, num_envs=32),
                  reward_weights=dict(contact_loss=-4.), physics_dt_s=.0025,
                  warm_start=dict(checkpoint='original'), initialization='fresh_optimizer')
    source_path = tmp_path/'protocol.json'
    source_path.write_text(json.dumps(source))
    report_path = tmp_path/'report.json'
    report = dict(status='training_completed_not_task_success', config_sha256=sha(source_path),
                  transitions=3072000, worlds=512, bilateral_audit=dict(fault_seen=False),
                  checkpoint_sha256=sha(checkpoint), actor_sha256='actor-hash')
    report_path.write_text(json.dumps(report))
    target = copy.deepcopy(source)
    target['formal']['iterations'] = 750
    target.update(initialization='optimizer_continuation', warm_start=dict(checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), actor_sha256='actor-hash'),
        training_continuation=dict(kind='hockey_optimizer_continuation_v1', source_protocol=str(source_path),
            source_protocol_sha256=sha(source_path), source_report=str(report_path), source_report_sha256=sha(report_path)))
    return r, target


def test_preserves_next_optimizer_update_and_noise(saved):
    original, cfg = saved
    restored = runner()
    result = initialize_continuation(restored, cfg)
    assert result['optimizer_loaded'] and not result['exploration_reset']
    assert result['prior_same_recipe_transitions'] == 3072000
    assert restored.current_learning_iteration == restored.tot_timesteps == 0
    assert state_digest(original.alg.policy.state_dict()) == state_digest(restored.alg.policy.state_dict())
    assert state_digest(original.alg.optimizer.state_dict()) == state_digest(restored.alg.optimizer.state_dict())
    step(original)
    step(restored)
    assert state_digest(original.alg.policy.state_dict()) == state_digest(restored.alg.policy.state_dict())
    assert state_digest(original.alg.optimizer.state_dict()) == state_digest(restored.alg.optimizer.state_dict())


@pytest.mark.parametrize('field', ['reward', 'dt', 'num_envs', 'schedule', 'noise'])
def test_rejects_changed_learning_experiment(saved, field):
    _, cfg = saved
    if field == 'reward': cfg['reward_weights']['contact_loss'] = -5.
    elif field == 'dt': cfg['physics_dt_s'] = .005
    elif field == 'num_envs': cfg['formal']['num_envs'] = 256
    elif field == 'schedule': cfg['ppo']['schedule'] = 'adaptive'
    elif field == 'noise': cfg['ppo']['init_noise_std'] = .2
    with pytest.raises(ValueError, match='preserve'):
        initialize_continuation(runner(), cfg)


def test_rejects_corrupt_checkpoint_before_loading(saved):
    _, cfg = saved
    cfg['warm_start']['checkpoint_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='intact'):
        initialize_continuation(runner(), cfg)


@pytest.mark.parametrize('change', ['unfinished', 'wrong_budget', 'collector_fault', 'wrong_iteration', 'empty_optimizer'])
def test_rejects_invalid_source_even_with_updated_hashes(saved, change):
    _, cfg = saved
    settings = cfg['training_continuation']
    p = Path(settings['source_report']); report = json.loads(p.read_text())
    if change == 'unfinished': report['status'] = 'training'
    elif change == 'wrong_budget': report['transitions'] -= 1
    elif change == 'collector_fault': report['bilateral_audit']['fault_seen'] = True
    else:
        cp = Path(cfg['warm_start']['checkpoint']); state = torch.load(cp, weights_only=True)
        if change == 'wrong_iteration': state['iter'] = 250
        else: state['optimizer_state_dict']['state'] = {}
        torch.save(state, cp)
        cfg['warm_start']['checkpoint_sha256'] = report['checkpoint_sha256'] = sha(cp)
    p.write_text(json.dumps(report)); settings['source_report_sha256'] = sha(p)
    with pytest.raises(ValueError): initialize_continuation(runner(), cfg)


def test_rejects_nonzero_segment_counters(saved):
    _, cfg = saved
    r = runner(); r.current_learning_iteration = 249
    with pytest.raises(ValueError, match='zero'):
        initialize_continuation(r, cfg)


def test_only_iteration_budget_and_provenance_can_change(saved):
    _, cfg = saved
    source = json.loads(Path(cfg['training_continuation']['source_protocol']).read_text())
    matching_recipe(source, cfg)
    cfg['formal']['iterations'] = 0
    with pytest.raises(ValueError, match='positive integer'): matching_recipe(source, cfg)
